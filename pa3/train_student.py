"""Trilha C — o aluno: ResNet-18 ImageNet treinada no train inteiro com os pseudo-rótulos do professor
+ entropia cruzada nos rótulos verdadeiros do orçamento.

Modos:
  python -m pa3.train_student tempo                 # mede uma época (planejamento)
  python -m pa3.train_student ablacao               # variantes A-E, orçamento 5, seed 0; escolha NO VAL
  python -m pa3.train_student metodo                # variante escolhida: orçamentos 1, 5 (seeds 1, 2), 20
  python -m pa3.train_student entrega               # melhor seed NO VAL do orçamento 5 -> weights/aluno_final.pt

Como o aluno é treinado
- Pool não rotulado = as 6.505 imagens do train MENOS as do orçamento. Cada imagem do pool tem um
  alvo q (21 probabilidades) vindo do professor (cache da Etapa 1, CLIP ViT-B/16).
- As imagens do orçamento entram com o rótulo VERDADEIRO (nunca com o pseudo), num segundo fluxo:
  a cada passo, 64 imagens do pool + 16 rotuladas (o conjunto rotulado é ciclado), um único
  forward; perda = CE_suave(pool) + 1,0 · CE(rotuladas).
- Uma "época" = uma passada pelo pool. Época escolhida pelo F1 macro no val (desempate: menor perda
  de val), como no piso.

Variantes do tratamento de ruído (PLANO.md, Etapa 2):
  A  pseudo-rótulo duro (argmax) do zero-shot, sem tratamento;
  B  A + filtro por confiança: só entram imagens do pool com max p_zs >= τ; τ ∈ {0,3; 0,5; 0,7}
     escolhido NO VAL (cobertura no train: 60%, 35%, 23%);
  C  rótulos suaves do zero-shot com temperatura: q = softmax(log p_zs / T);
  D  rótulos suaves do professor MELHORADO do orçamento: q = softmax(log p_mel / T);
  E  D + amostragem balanceada do pool pela classe pseudo-rotulada (peso 1/n da classe argmax).

Temperatura (C, D, E): escolhida SEM val e SEM test_public, por calibração do professor nos rótulos
do orçamento: o T da grade que minimiza a log-loss do professor nesses rótulos. Para o zero-shot
isso é fora da amostra (ele nunca treinou neles). Para o melhorado usamos as previsões fora da dobra
da mesma CV da Etapa 1 (α e C já escolhidos lá). No orçamento 1 não há CV possível para o melhorado:
T = 1 a priori (declarado aqui antes de rodar).

Regras: o test_public só é avaliado no estado já escolhido. No modo `ablacao` as métricas de teste
só são calculadas DEPOIS de a variante (e o τ de B) ter sido escolhida no val. O professor não é
recalculado aqui (as probabilidades vêm do cache, que nunca viu o val).
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import log_loss

from pa3 import data, metrics, models, teacher
from pa3.train_sup import dispositivo, fixa_seed, prediz

RAIZ = Path(__file__).resolve().parent.parent
RES_DIR = RAIZ / "results"
PREDS = RES_DIR / "preds"
CACHE = RAIZ / "cache"
PESOS_CACHE = CACHE / "alunos"
WEIGHTS = RAIZ / "weights"
RES = 224

# Hiperparâmetros fixos a priori (iguais aos do piso onde faz sentido); só época, τ e variante são
# escolhidos no val. As épocas foram dimensionadas pelo modo `tempo` (ver RESULTADOS.md).
HP = {"lr": 3e-4, "wd": 0.05, "epocas": 20, "bs_pool": 64, "bs_rot": 16, "peso_rot": 1.0}
GRADE_TAU = (0.3, 0.5, 0.7)
GRADE_T = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)
T_PRIORI_R1_MELHORADO = 1.0

VARIANTES = {
    "A": {"fonte": "zs", "suave": False, "limiar": 0.0, "balanceado": False},
    "B": {"fonte": "zs", "suave": False, "limiar": None, "balanceado": False},  # τ na grade, escolhido no val
    "C": {"fonte": "zs", "suave": True, "limiar": 0.0, "balanceado": False},
    "D": {"fonte": "melhorado", "suave": True, "limiar": 0.0, "balanceado": False},
    "E": {"fonte": "melhorado", "suave": True, "limiar": 0.0, "balanceado": True},
}
DESCRICAO = {
    "A": "duro, zero-shot, sem tratamento",
    "B": "duro, zero-shot, filtro por confiança",
    "C": "suave, zero-shot, temperatura",
    "D": "suave, professor melhorado",
    "E": "suave, prof. melhorado + amostragem balanceada",
}


# ------------------------------------------------------------------------------------ alvos

def _logp_professor(fonte: str, orcamento: int) -> np.ndarray:
    """log-probabilidades (6505,21) do professor para o train, na ordem de data.ids_train()."""
    ids = json.load(open(CACHE / "prof_train_ids.json"))["ids"]
    assert ids == data.ids_train()
    if fonte == "zs":
        return teacher.log_softmax(np.load(CACHE / "prof_zs_train_logits.npy").astype(np.float64))
    return np.load(CACHE / f"prof_melhorado_r{orcamento}_train_logprobs.npy").astype(np.float64)


def _melhor_T(logp: np.ndarray, y: np.ndarray) -> tuple[float, list[dict]]:
    tabela = [{"T": T, "logloss": float(log_loss(y, np.exp(teacher.log_softmax(logp / T)), labels=np.arange(21)))}
              for T in GRADE_T]
    return min(tabela, key=lambda r: r["logloss"])["T"], tabela


def temperatura(fonte: str, orcamento: int) -> dict:
    """T por calibração do professor nos rótulos do orçamento (nunca val, nunca test_public)."""
    orc = data.rotulos(f"rotulos_{orcamento}")
    ids = data.ids_train()
    pos = {i: k for k, i in enumerate(ids)}
    idx = np.array([pos[i] for i in orc["image_id"]])
    y = orc["y"].to_numpy()
    if fonte == "zs":
        T, tab = _melhor_T(_logp_professor("zs", orcamento)[idx], y)
        return {"T": T, "criterio": "log-loss do zero-shot nos rótulos do orçamento", "tabela": tab}
    if orcamento == 1:
        return {"T": T_PRIORI_R1_MELHORADO, "criterio": "orçamento 1: sem CV, T=1 a priori", "tabela": []}
    # previsões fora da dobra do professor melhorado, com o (α, C) escolhido na Etapa 1
    hp = json.load(open(RES_DIR / "professor.json"))["melhorado"][str(orcamento)]["hp"]
    Ftr = np.load(CACHE / "prof_clip_b16_recorte_train_img.npy")
    Z = np.load(CACHE / "prof_zs_train_logits.npy").astype(np.float64)
    X, Zr = Ftr[idx], Z[idx]
    dobra = orc.groupby("label").cumcount().to_numpy()
    lp_oof = np.zeros((len(y), 21))
    for j in range(dobra.max() + 1):
        tr, te = dobra != j, dobra == j
        lp_oof[te] = teacher.lr_logprob(X[tr], y[tr], X[te], hp["C"])
    lp = teacher.combina(Zr, lp_oof, hp["alfa"])
    T, tab = _melhor_T(lp, y)
    return {"T": T, "criterio": f"log-loss fora da dobra do professor melhorado (α={hp['alfa']}, C={hp['C']})",
            "tabela": tab}


def alvos(cfg: dict, orcamento: int, limiar: float) -> dict:
    """Alvos do pool. Devolve q (6505,21), conf (6505,), classe pseudo (6505,), T usado."""
    logp = _logp_professor(cfg["fonte"], orcamento)
    p1 = np.exp(logp)
    conf, pseudo = p1.max(1), p1.argmax(1)
    if cfg["suave"]:
        tinfo = temperatura(cfg["fonte"], orcamento)
        q = np.exp(teacher.log_softmax(logp / tinfo["T"]))
    else:
        tinfo = {"T": None}
        q = np.eye(21)[pseudo]
    return {"q": q.astype(np.float32), "conf": conf, "pseudo": pseudo, "temperatura": tinfo, "limiar": limiar}


# ------------------------------------------------------------------------------------ treino

def treina_aluno(Xall, alv: dict, idx_rot: np.ndarray, y_rot: np.ndarray, Xva, yva, seed: int, device,
                 balanceado: bool, epocas: int | None = None, max_passos: int | None = None, verbose=True):
    """Treina o aluno; devolve (modelo no estado escolhido no val, histórico, época escolhida, info)."""
    epocas = epocas or HP["epocas"]
    fixa_seed(seed)
    rot = set(idx_rot.tolist())
    pool = np.array([i for i in range(len(Xall)) if i not in rot and alv["conf"][i] >= alv["limiar"]])
    q_t = torch.from_numpy(alv["q"])
    if balanceado:
        cont = np.bincount(alv["pseudo"][pool], minlength=21)
        w = 1.0 / cont[alv["pseudo"][pool]]
        w = w / w.sum()
    model = models.cria("resnet18").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=HP["lr"], weight_decay=HP["wd"])
    passos_ep = int(np.ceil(len(pool) / HP["bs_pool"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epocas * passos_ep)
    gen = torch.Generator().manual_seed(seed)
    rng = np.random.RandomState(seed)
    yva_t = torch.from_numpy(yva)
    hist = {k: [] for k in ("perda_pool", "perda_rot", "acc_rot", "perda_val", "f1_val", "acc_val", "segundos")}
    melhor, melhor_estado, ep_escolhida = (-1.0, np.inf), None, 0
    fila_rot = np.array([], dtype=int)
    for ep in range(1, epocas + 1):
        t0 = time.time()
        model.train()
        ordem = rng.choice(pool, size=len(pool), replace=True, p=w) if balanceado else rng.permutation(pool)
        s_pool, s_rot, ac_rot, n_rot, n_passos = 0.0, 0.0, 0, 0, 0
        for i in range(0, len(ordem), HP["bs_pool"]):
            iu = ordem[i:i + HP["bs_pool"]]
            if len(fila_rot) < HP["bs_rot"]:
                fila_rot = np.concatenate([fila_rot, rng.permutation(len(idx_rot))])
            jl, fila_rot = fila_rot[:HP["bs_rot"]], fila_rot[HP["bs_rot"]:]
            il = idx_rot[jl]
            todos = np.concatenate([iu, il])
            ordem_ler = np.argsort(todos)  # leitura ordenada do array grande (mais rápida)
            lote = np.empty((len(todos),) + Xall.shape[1:], dtype=np.uint8)
            lote[ordem_ler] = Xall[todos[ordem_ler]]
            x = data.normaliza(data.aumenta(data.para_tensor(lote, device), gen))
            logit = model(x)
            nu = len(iu)
            lu = -(q_t[iu].to(device) * F.log_softmax(logit[:nu], 1)).sum(1).mean()
            yl = torch.from_numpy(y_rot[jl]).to(device)
            ll = F.cross_entropy(logit[nu:], yl)
            perda = lu + HP["peso_rot"] * ll
            opt.zero_grad(set_to_none=True)
            perda.backward()
            opt.step()
            sched.step()
            s_pool += lu.item() * nu
            s_rot += ll.item() * len(jl)
            ac_rot += (logit[nu:].argmax(1) == yl).sum().item()
            n_rot += len(jl)
            n_passos += 1
            if max_passos and n_passos >= max_passos:
                break
        lv = torch.from_numpy(prediz(model, Xva, device))
        pv = lv.argmax(1).numpy()
        r = metrics.resumo(yva, pv)
        hist["perda_pool"].append(s_pool / max(1, min(len(ordem), n_passos * HP["bs_pool"])))
        hist["perda_rot"].append(s_rot / n_rot)
        hist["acc_rot"].append(ac_rot / n_rot)
        hist["perda_val"].append(float(F.cross_entropy(lv, yva_t)))
        hist["f1_val"].append(r["f1_macro"])
        hist["acc_val"].append(r["acuracia"])
        hist["segundos"].append(time.time() - t0)
        if (r["f1_macro"], -hist["perda_val"][-1]) > (melhor[0], -melhor[1]):
            melhor = (r["f1_macro"], hist["perda_val"][-1])
            melhor_estado = copy.deepcopy(model.state_dict())
            ep_escolhida = ep
        if verbose:
            print(f"    ep {ep:2d} perda_pool {hist['perda_pool'][-1]:.3f} perda_rot {hist['perda_rot'][-1]:.3f} "
                  f"acc_rot {hist['acc_rot'][-1]:.2f} perda_val {hist['perda_val'][-1]:.3f} f1_val {r['f1_macro']:.4f} "
                  f"({hist['segundos'][-1]:.0f}s)", flush=True)
        if max_passos:
            break
    model.load_state_dict(melhor_estado)
    info = {"n_pool": int(len(pool)), "n_rotuladas": int(len(idx_rot)),
            "cobertura_pool": float(len(pool) / (len(Xall) - len(idx_rot))),
            "pseudo_por_classe_no_pool": np.bincount(alv["pseudo"][pool], minlength=21).tolist()}
    return model, hist, ep_escolhida, info


def _dados(orcamento: int):
    ids, Xall = data.imagens("train", RES)
    orc = data.rotulos(f"rotulos_{orcamento}")
    pos = {i: k for k, i in enumerate(ids)}
    idx_rot = np.array([pos[i] for i in orc["image_id"]])
    Xva, yva, _ = data.conjunto_rotulado("val", RES)
    return Xall, idx_rot, orc["y"].to_numpy(), Xva, yva


def roda(tag: str, cfg: dict, orcamento: int, seed: int, limiar: float, device, dados=None) -> dict:
    """Treina um aluno, salva logits de val e de test_public do estado escolhido e o state_dict.
    NÃO calcula métrica de teste (isso é feito depois, quando as escolhas já estão feitas)."""
    Xall, idx_rot, y_rot, Xva, yva = dados or _dados(orcamento)
    alv = alvos(cfg, orcamento, limiar)
    print(f"[{tag}] orçamento {orcamento} seed {seed} cfg {cfg} τ={limiar} T={alv['temperatura']['T']}", flush=True)
    t0 = time.time()
    model, hist, ep, info = treina_aluno(Xall, alv, idx_rot, y_rot, Xva, yva, seed, device, cfg["balanceado"])
    PREDS.mkdir(parents=True, exist_ok=True)
    PESOS_CACHE.mkdir(parents=True, exist_ok=True)
    np.save(PREDS / f"aluno_{tag}_val_logits.npy", prediz(model, Xva, device))
    Xte, _, _ = data.conjunto_rotulado("test_public", RES)
    np.save(PREDS / f"aluno_{tag}_test_logits.npy", prediz(model, Xte, device))  # só guardado; métrica depois
    torch.save(model.state_dict(), PESOS_CACHE / f"aluno_{tag}.pt")
    run = {"tag": tag, "cfg": cfg, "orcamento": orcamento, "seed": seed, "limiar": limiar,
           "temperatura": alv["temperatura"], "epoca_escolhida": ep, "f1_val": hist["f1_val"][ep - 1],
           "acc_val": hist["acc_val"][ep - 1], "hist": hist, "segundos": time.time() - t0,
           "n_parametros": models.n_parametros(model), **info}
    print(f"[{tag}] época {ep} f1_val {run['f1_val']:.4f} ({run['segundos']:.0f}s)", flush=True)
    if device.type == "mps":
        torch.mps.empty_cache()
    return run


def _metricas_teste(tag: str) -> dict:
    yte = data.rotulos("test_public")["y"].to_numpy()
    p = np.load(PREDS / f"aluno_{tag}_test_logits.npy").argmax(1)
    r = metrics.resumo(yte, p)
    return {"test_f1_macro": r["f1_macro"], "test_acuracia": r["acuracia"]}


def _salva(obj, nome):
    with open(RES_DIR / nome, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ------------------------------------------------------------------------------------ modos

def modo_tempo():
    device = dispositivo()
    dados = _dados(5)
    alv = alvos(VARIANTES["C"], 5, 0.0)
    Xall, idx_rot, y_rot, Xva, yva = dados
    passos = 20
    # aquecimento + medida de `passos` passos; extrapola para a época inteira
    _, hist, _, info = treina_aluno(Xall, alv, idx_rot, y_rot, Xva, yva, 0, device, False, epocas=1, max_passos=5,
                                    verbose=False)
    t0 = time.time()
    _, hist, _, info = treina_aluno(Xall, alv, idx_rot, y_rot, Xva, yva, 0, device, False, epocas=1,
                                    max_passos=passos, verbose=False)
    dt = time.time() - t0
    passos_ep = int(np.ceil(info["n_pool"] / HP["bs_pool"]))
    seg_passo = (hist["segundos"][0]) / passos
    saida = {"passos_medidos": passos, "seg_por_passo_com_val": seg_passo, "passos_por_epoca": passos_ep,
             "seg_por_epoca_estimado": seg_passo * passos_ep, "total_medido": dt, "T_C": alv["temperatura"]}
    print(saida, flush=True)
    _salva(saida, "aluno_tempo.json")


def modo_ablacao(orcamento=5, seed=0):
    device = dispositivo()
    dados = _dados(orcamento)
    runs = []
    for v, cfg in VARIANTES.items():
        limiares = GRADE_TAU if cfg["limiar"] is None else (cfg["limiar"],)
        for tau in limiares:
            tag = f"abl_{v}" + (f"_tau{tau}" if cfg["limiar"] is None else "") + f"_r{orcamento}_s{seed}"
            r = roda(tag, cfg, orcamento, seed, tau, device, dados)
            r["variante"] = v
            runs.append(r)
            _salva({"hp": HP, "runs": runs, "parcial": True}, "aluno_ablacao.json")
    # ---- escolhas SÓ no val: τ de B, depois a variante
    melhor_por_var = {}
    for v in VARIANTES:
        rs = [r for r in runs if r["variante"] == v]
        melhor_por_var[v] = max(rs, key=lambda r: (r["f1_val"], -min(r["hist"]["perda_val"])))["tag"]
    escolhida = max(VARIANTES, key=lambda v: next(r["f1_val"] for r in runs if r["tag"] == melhor_por_var[v]))
    # ---- só agora: métricas no test_public, para reportar
    for r in runs:
        r.update(_metricas_teste(r["tag"]))
    rb = next(r for r in runs if r["tag"] == melhor_por_var["B"])
    saida = {"hp": HP, "orcamento": orcamento, "seed": seed, "grade_tau": GRADE_TAU, "grade_T": GRADE_T,
             "tau_escolhido_B": rb["limiar"], "tag_por_variante": melhor_por_var, "variante_escolhida": escolhida,
             "cfg_escolhida": {**VARIANTES[escolhida], "limiar": next(r["limiar"] for r in runs
                                                                      if r["tag"] == melhor_por_var[escolhida])},
             "criterio": "maior F1 macro no val na época escolhida no val (τ de B escolhido antes, também no val)",
             "runs": runs}
    _salva(saida, "aluno_ablacao.json")
    for r in runs:
        print(f"{r['tag']:28s} f1_val {r['f1_val']:.4f} test {r['test_f1_macro']:.4f}", flush=True)
    print("variante escolhida no val:", escolhida, saida["cfg_escolhida"])


def modo_metodo():
    abl = json.load(open(RES_DIR / "aluno_ablacao.json"))
    v, cfg = abl["variante_escolhida"], abl["cfg_escolhida"]
    device = dispositivo()
    runs = []
    # orçamento 5 seed 0 = o próprio run da ablação (mesma configuração e seed)
    r0 = next(r for r in abl["runs"] if r["tag"] == abl["tag_por_variante"][v])
    runs.append({**r0, "reaproveitado_da_ablacao": True})
    for orc, seed in ((5, 1), (5, 2), (1, 0), (20, 0)):
        tag = f"metodo_{v}_r{orc}_s{seed}"
        r = roda(tag, {k: cfg[k] for k in ("fonte", "suave", "balanceado")} | {"limiar": cfg["limiar"]},
                 orc, seed, cfg["limiar"], device)
        r["variante"] = v
        runs.append(r)
        _salva({"variante": v, "cfg": cfg, "runs": runs, "parcial": True}, "aluno_metodo.json")
    # seed entregue: escolhida no val, entre as 3 seeds do orçamento 5
    r5 = [r for r in runs if r["orcamento"] == 5]
    entregue = max(r5, key=lambda r: (r["f1_val"], -min(r["hist"]["perda_val"])))["tag"]
    for r in runs:
        r.update(_metricas_teste(r["tag"]))
    _salva({"variante": v, "cfg": cfg, "hp": HP, "runs": runs, "tag_entregue": entregue,
            "criterio_entrega": "maior F1 macro no val entre as 3 seeds do orçamento 5"}, "aluno_metodo.json")
    for r in runs:
        print(f"{r['tag']:28s} f1_val {r['f1_val']:.4f} test {r['test_f1_macro']:.4f}", flush=True)
    print("entregue:", entregue)


def modo_entrega():
    met = json.load(open(RES_DIR / "aluno_metodo.json"))
    tag = met["tag_entregue"]
    r = next(r for r in met["runs"] if r["tag"] == tag)
    model = models.resnet18(21, imagenet=False)
    model.load_state_dict(torch.load(PESOS_CACHE / f"aluno_{tag}.pt", map_location="cpu"))
    n = models.n_parametros(model)
    assert n <= 12_000_000
    WEIGHTS.mkdir(exist_ok=True)
    pacote = {"state_dict": model.state_dict(), "classes": data.classes(), "resolucao": RES,
              "redimensionamento": "achatada (res x res, sem recorte, bicúbica)",
              "media": data.MEDIA_IMAGENET, "desvio": data.DESVIO_IMAGENET, "arquitetura": "torchvision resnet18, fc 512->21",
              "n_parametros": n, "origem": tag, "f1_val": r["f1_val"]}
    torch.save(pacote, WEIGHTS / "aluno_final.pt")
    # conferência: recarrega em CPU e recalcula o F1 no test_public
    Xte, yte, _ = data.conjunto_rotulado("test_public", RES)
    p = prediz(model, Xte, torch.device("cpu")).argmax(1)
    res = metrics.resumo(yte, p)
    saida = {"tag": tag, "n_parametros": n, "test_cpu": res, "test_registrado": r["test_f1_macro"],
             "tamanho_MB": (WEIGHTS / "aluno_final.pt").stat().st_size / 1e6}
    print(saida)
    _salva(saida, "aluno_entregue.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("modo", choices=["tempo", "ablacao", "metodo", "entrega"])
    a = ap.parse_args()
    {"tempo": modo_tempo, "ablacao": modo_ablacao, "metodo": modo_metodo, "entrega": modo_entrega}[a.modo]()


if __name__ == "__main__":
    main()
