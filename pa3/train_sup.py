"""Parte 0 — o piso: treino supervisionado só com os rótulos do orçamento.

Modos:
  python -m pa3.train_sup resolucao            # (b) ResNet-18 em 224 e 320, seeds 0-2, escolha no VAL
  python -m pa3.train_sup piso --res 224       # (a) CNN do zero e (b) ResNet-18 na resolução fixada
  python -m pa3.train_sup piso_b --orcamento K --seeds 0 [1 2]   # só (b), para a curva de orçamento
  python -m pa3.train_sup throughput           # tempo de uma época no train inteiro (planejamento da Etapa 2)

Regras: a época é escolhida pelo F1 macro no val (desempate: menor perda de val). O test_public só é
avaliado no estado já escolhido, para reportar. Nenhuma decisão olha para ele.
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

from pa3 import data, metrics, models, plots

RAIZ = Path(__file__).resolve().parent.parent
RES_DIR = RAIZ / "results"

# Hiperparâmetros fixos (não foram ajustados; só resolução e época são escolhidas no val).
HP = {
    "resnet18": {"lr": 3e-4, "wd": 0.05, "epocas": 60, "bs": 32},
    "cnn": {"lr": 1e-3, "wd": 0.05, "epocas": 150, "bs": 32},
}


def dispositivo() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def fixa_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def prediz(model, X: np.ndarray, device, bs: int = 128) -> np.ndarray:
    """Logits (N,21) para X uint8 (N,H,W,3), sem aumentação."""
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        x = data.normaliza(data.para_tensor(X[i:i + bs], device))
        out.append(model(x).float().cpu())
    return torch.cat(out).numpy()


def treina(nome: str, Xtr, ytr, Xva, yva, seed: int, device, hp: dict | None = None, verbose=True):
    """Treina e devolve (modelo no melhor estado do val, histórico, época escolhida, época de mínimo da perda de val)."""
    hp = hp or HP[nome]
    fixa_seed(seed)
    model = models.cria(nome).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    passos_ep = int(np.ceil(len(Xtr) / hp["bs"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=hp["epocas"] * passos_ep)
    gen = torch.Generator().manual_seed(seed)
    rng = np.random.RandomState(seed)
    yva_t = torch.from_numpy(yva)
    hist = {k: [] for k in ("perda_treino", "acc_treino", "perda_val", "f1_val", "acc_val")}
    melhor = (-1.0, np.inf)
    melhor_estado, ep_escolhida = None, 0
    for ep in range(1, hp["epocas"] + 1):
        model.train()
        ordem = rng.permutation(len(Xtr))
        soma, acertos = 0.0, 0
        for i in range(0, len(ordem), hp["bs"]):
            idx = ordem[i:i + hp["bs"]]
            x = data.normaliza(data.aumenta(data.para_tensor(Xtr[idx], device), gen))
            y = torch.from_numpy(ytr[idx]).to(device)
            logit = model(x)
            perda = F.cross_entropy(logit, y)
            opt.zero_grad(set_to_none=True)
            perda.backward()
            opt.step()
            sched.step()
            soma += perda.item() * len(idx)
            acertos += (logit.argmax(1) == y).sum().item()
        lv = torch.from_numpy(prediz(model, Xva, device))
        pv = lv.argmax(1).numpy()
        r = metrics.resumo(yva, pv)
        hist["perda_treino"].append(soma / len(Xtr))
        hist["acc_treino"].append(acertos / len(Xtr))
        hist["perda_val"].append(float(F.cross_entropy(lv, yva_t)))
        hist["f1_val"].append(r["f1_macro"])
        hist["acc_val"].append(r["acuracia"])
        chave = (r["f1_macro"], -hist["perda_val"][-1])
        if chave > (melhor[0], -melhor[1]):
            melhor = (r["f1_macro"], hist["perda_val"][-1])
            melhor_estado = copy.deepcopy(model.state_dict())
            ep_escolhida = ep
        if verbose and (ep % 10 == 0 or ep == 1):
            print(f"  [{nome} s{seed}] ep {ep:3d} perda_tr {hist['perda_treino'][-1]:.3f} "
                  f"acc_tr {hist['acc_treino'][-1]:.2f} perda_val {hist['perda_val'][-1]:.3f} "
                  f"f1_val {r['f1_macro']:.3f}", flush=True)
    model.load_state_dict(melhor_estado)
    ep_min_val = int(np.argmin(hist["perda_val"]) + 1)
    return model, hist, ep_escolhida, ep_min_val


def _salva_json(obj, nome):
    RES_DIR.mkdir(exist_ok=True)
    with open(RES_DIR / nome, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def modo_resolucao(orcamento: int, seeds=(0, 1, 2)):
    device = dispositivo()
    Xva_por_res, saida = {}, {"orcamento": orcamento, "hp": HP["resnet18"], "runs": []}
    for res in (224, 320):
        Xtr, ytr, _ = data.conjunto_rotulado(f"rotulos_{orcamento}", res)
        Xva, yva, _ = data.conjunto_rotulado("val", res)
        for s in seeds:
            t0 = time.time()
            _, hist, ep, ep_min = treina("resnet18", Xtr, ytr, Xva, yva, s, device)
            saida["runs"].append({"res": res, "seed": s, "epoca_escolhida": ep,
                                  "f1_val": hist["f1_val"][ep - 1], "acc_val": hist["acc_val"][ep - 1],
                                  "perda_val_min": min(hist["perda_val"]), "epoca_min_perda_val": ep_min,
                                  "segundos": time.time() - t0})
            print(saida["runs"][-1], flush=True)
    for res in (224, 320):
        f1s = [r["f1_val"] for r in saida["runs"] if r["res"] == res]
        saida[f"f1_val_media_{res}"] = float(np.mean(f1s))
        saida[f"f1_val_desvio_{res}"] = float(np.std(f1s))
    saida["res_escolhida"] = 224 if saida["f1_val_media_224"] >= saida["f1_val_media_320"] else 320
    saida["criterio"] = "maior F1 macro médio no val (3 seeds) na época escolhida no val; empate -> 224 (mais barata)"
    _salva_json(saida, "piso_resolucao.json")
    print("resolução escolhida:", saida["res_escolhida"])


def modo_piso(res: int, orcamento: int, seeds=(0, 1, 2), modelos=("cnn", "resnet18")):
    """Com os dois modelos grava results/piso_r{K}.json e as figuras (Etapa 1). Só com resnet18
    (Etapa 2, curva de orçamento) grava results/piso_b_r{K}.json, sem figuras, e salva os logits de
    teste de todas as seeds (para a quebra por classe)."""
    so_b = tuple(modelos) == ("resnet18",)
    device = dispositivo()
    nomes = data.classes()
    Xtr, ytr, _ = data.conjunto_rotulado(f"rotulos_{orcamento}", res)
    Xva, yva, _ = data.conjunto_rotulado("val", res)
    Xte, yte, ids_te = data.conjunto_rotulado("test_public", res)
    saida = {"orcamento": orcamento, "res": res, "hp": HP, "runs": []}
    for nome in modelos:
        for s in seeds:
            t0 = time.time()
            model, hist, ep, ep_min = treina(nome, Xtr, ytr, Xva, yva, s, device)
            # Só agora, com a época já escolhida no val, o test_public é avaliado.
            lt = prediz(model, Xte, device)
            pt = lt.argmax(1)
            r = metrics.resumo(yte, pt)
            run = {"modelo": nome, "seed": s, "epoca_escolhida": ep, "epoca_min_perda_val": ep_min,
                   "f1_val": hist["f1_val"][ep - 1], "acc_val": hist["acc_val"][ep - 1],
                   "test_f1_macro": r["f1_macro"], "test_acuracia": r["acuracia"],
                   "n_parametros": models.n_parametros(model), "segundos": time.time() - t0,
                   "hist": hist}
            saida["runs"].append(run)
            print({k: v for k, v in run.items() if k != "hist"}, flush=True)
            if so_b:
                (RES_DIR / "preds").mkdir(parents=True, exist_ok=True)
                np.save(RES_DIR / "preds" / f"piso_b_r{orcamento}_s{s}_test_logits.npy", lt)
                run["f1_por_classe_test"] = dict(zip(nomes, metrics.f1_por_classe(yte, pt).round(4).tolist()))
            elif nome == "resnet18" and s == 0:
                cm = metrics.confusao(yte, pt)
                run["confusao_test"] = cm.tolist()
                run["f1_por_classe_test"] = dict(zip(nomes, metrics.f1_por_classe(yte, pt).round(4).tolist()))
                plots.confusao(cm, nomes, f"Piso (b) ResNet-18 ImageNet, {orcamento} rótulos/classe, {res}px, seed 0 — "
                               f"test_public (F1 macro {r['f1_macro']:.3f})", "piso_confusao.png")
                plots.curva_treino_val(hist, ep_min, ep,
                                       f"Piso (b) ResNet-18, {orcamento} rótulos/classe ({len(Xtr)} imagens), {res}px, seed 0",
                                       "piso_curva.png")
                (RES_DIR / "preds").mkdir(parents=True, exist_ok=True)
                np.save(RES_DIR / "preds" / f"piso_b_r{orcamento}_s0_test_logits.npy", lt)
    for nome in modelos:
        rs = [r for r in saida["runs"] if r["modelo"] == nome]
        saida[f"{nome}_test_f1_media"] = float(np.mean([r["test_f1_macro"] for r in rs]))
        saida[f"{nome}_test_f1_desvio"] = float(np.std([r["test_f1_macro"] for r in rs]))
        saida[f"{nome}_test_acc_media"] = float(np.mean([r["test_acuracia"] for r in rs]))
        saida[f"{nome}_test_acc_desvio"] = float(np.std([r["test_acuracia"] for r in rs]))
    _salva_json(saida, f"piso_b_r{orcamento}.json" if so_b else f"piso_r{orcamento}.json")


def modo_throughput(bs: int = 64, passos: int = 30):
    """Tempo de época de treino da ResNet-18 no train inteiro (6.505 imagens), medido em MPS."""
    device = dispositivo()
    saida = {"device": str(device), "bs": bs, "n_train": 6505}
    for res in (224, 320):
        _, arr = data.imagens("train", res)
        fixa_seed(0)
        model = models.cria("resnet18").to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        gen = torch.Generator().manual_seed(0)
        rng = np.random.RandomState(0)
        tempos = []
        for k in range(passos + 5):
            idx = np.sort(rng.choice(len(arr), bs, replace=False))
            if k == 5:
                torch.mps.synchronize()
                t0 = time.time()
            x = data.normaliza(data.aumenta(data.para_tensor(arr[idx], device), gen))
            y = torch.randint(0, 21, (bs,), device=device)
            perda = F.cross_entropy(model(x), y)
            opt.zero_grad()
            perda.backward()
            opt.step()
        torch.mps.synchronize()
        dt = (time.time() - t0) / passos
        # inferência
        model.eval()
        with torch.no_grad():
            torch.mps.synchronize()
            t1 = time.time()
            for i in range(0, 1024, 128):
                model(data.normaliza(data.para_tensor(arr[i:i + 128], device))).cpu()
            torch.mps.synchronize()
        inf = (time.time() - t1) / 1024
        saida[str(res)] = {"seg_por_passo": dt, "img_por_seg_treino": bs / dt,
                           "seg_por_epoca_train_inteiro": dt * 6505 / bs,
                           "seg_por_img_inferencia_mps": inf}
        print(res, saida[str(res)], flush=True)
    _salva_json(saida, "throughput_mps.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("modo", choices=["resolucao", "piso", "piso_b", "throughput"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--orcamento", type=int, default=5)
    a = ap.parse_args()
    if a.modo == "resolucao":
        modo_resolucao(a.orcamento)
    elif a.modo == "piso":
        modo_piso(a.res, a.orcamento)
    elif a.modo == "piso_b":
        modo_piso(a.res, a.orcamento, tuple(a.seeds), modelos=("resnet18",))
    else:
        modo_throughput()


if __name__ == "__main__":
    main()
