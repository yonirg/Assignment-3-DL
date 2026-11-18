"""Parte 3 — Galeria de falhas e atalhos: Grad-CAM (layer4) e oclusão no modelo entregue.

    python -m pa3.cam

Usa só o modelo entregue (`weights/aluno_final.pt`) e o test_public, que aqui serve apenas para
DIAGNÓSTICO (nenhuma escolha de modelo sai daqui). Roda em CPU.

Regra de seleção dos casos (declarada antes de olhar as figuras; nada escolhido a dedo):
- 3 ERROS: os erros de MAIOR confiança, percorridos em ordem decrescente de confiança, aceitando um
  caso só se a classe verdadeira E a classe prevista ainda não apareceram entre os já aceitos
  (para não repetir o mesmo par/atalho três vezes).
- 5 ACERTOS: sorteio com semente fixa (0) entre os acertos, aceitando um caso só se a classe
  verdadeira ainda não apareceu, até ter 3 diurnos e 2 noturnos (flag `data.flag_noite`). O sorteio
  (e não "os mais confiantes") é para os acertos serem representativos, não os mais fáceis.

Grad-CAM (Selvaraju et al., 2017), implementação própria: ativações A (512×7×7) da saída da
`layer4` e gradiente do logit da classe PREVISTA em relação a elas; peso de cada canal = média
espacial do gradiente; mapa = ReLU(Σ peso·A), interpolado bilinearmente para 224×224 e normalizado
para [0, 1]. Como a resolução do mapa é 7×7, cada célula cobre 32×32 pixels da imagem a 224.

Oclusão (complemento): um quadrado cinza (média ImageNet) de 48×48 px desliza com passo 16 px; o
mapa é a queda da probabilidade da classe prevista quando aquela região é tapada (positivo = a
região sustentava a previsão).

Teste "tapar o animal" (complemento, para responder "animal ou fundo" sem depender só do mapa):
depois de olhar cada figura, desenhamos à mão uma caixa aproximada em volta do animal (em pixels da
imagem 224×224; `CAIXAS` abaixo). Isso NÃO é rótulo nem entra em nenhuma previsão ou escolha: serve só
para medir a probabilidade da classe prevista (i) com o animal tapado de cinza e (ii) com tudo menos o
animal tapado. Se tapar o animal quase não muda a previsão, o modelo está usando o fundo.

Saídas: results/figs/galeria_erros.png, results/figs/galeria_acertos.png (+ versões compactas
`*_slide.png` para a apresentação),
results/figs/galeria/{erro,acerto}_<k>_<image_id>.png e results/galeria.json (casos e números).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from pa3 import data, metrics, models  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
PESOS = RAIZ / "weights" / "aluno_final.pt"
FIGS = RAIZ / "results" / "figs"
GAL = FIGS / "galeria"
N_ERROS, N_ACERTOS_DIA, N_ACERTOS_NOITE = 3, 3, 2
SEMENTE = 0
OCL_LADO, OCL_PASSO = 48, 16
# caixas aproximadas do animal (x0, y0, x1, y1) a 224 px, desenhadas olhando as figuras (só diagnóstico)
CAIXAS = {
    "046de929caa4702b": (70, 110, 145, 170),   # gambá (erro 1)
    "ece8e099eb242e5d": (193, 143, 224, 190),  # cão na borda direita, atrás da caixa (erro 2)
    "6353c192fb1adc35": (104, 104, 134, 128),  # gato (erro 3)
    "fbbf64cb17e8330d": (0, 105, 22, 142),     # coiote cortado na borda esquerda (acerto 1)
    "8ba773a0aeb9cb8b": (30, 102, 55, 120),    # corvo (acerto 2)
    "2539e50f0cfcec84": (56, 160, 74, 180),    # tâmia junto à mangueira (acerto 3)
    "a6f1f0fd730dbbe7": (38, 132, 108, 162),   # gambá-norte-americano borrado (acerto 4)
    "2475c79619133dfb": (82, 160, 118, 224),   # galinha em primeiro plano (acerto 5)
}


def carrega_modelo():
    pac = torch.load(PESOS, map_location="cpu", weights_only=False)
    m = models.resnet18(len(pac["classes"]), imagenet=False)
    m.load_state_dict(pac["state_dict"])
    m.eval()
    return m, pac


def _entrada(x_uint8: np.ndarray) -> torch.Tensor:
    return data.normaliza(data.para_tensor(x_uint8, torch.device("cpu")))


@torch.no_grad()
def probabilidades(m, X: np.ndarray, bs: int = 64) -> np.ndarray:
    return np.concatenate([F.softmax(m(_entrada(X[i:i + bs])), 1).numpy() for i in range(0, len(X), bs)])


def grad_cam(m, x_uint8: np.ndarray, classe: int) -> np.ndarray:
    """Mapa Grad-CAM (224×224, em [0,1]) da `layer4` para a classe dada. x_uint8 (H,W,3)."""
    guarda = {}

    def fwd(_mod, _inp, out):
        guarda["A"] = out
        out.register_hook(lambda g: guarda.__setitem__("G", g))

    h = m.layer4.register_forward_hook(fwd)
    try:
        x = _entrada(x_uint8[None])
        m.zero_grad()
        logits = m(x)
        logits[0, classe].backward()
    finally:
        h.remove()
    A, G = guarda["A"][0].detach(), guarda["G"][0]
    pesos = G.mean(dim=(1, 2))
    cam = F.relu((pesos[:, None, None] * A).sum(0))
    cam = F.interpolate(cam[None, None], size=x_uint8.shape[:2], mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    return (cam / cam.max().clamp(min=1e-8)).numpy()


@torch.no_grad()
def oclusao(m, x_uint8: np.ndarray, classe: int) -> np.ndarray:
    """Queda da probabilidade da classe ao tapar cada região (média das janelas que cobrem o pixel)."""
    H, W = x_uint8.shape[:2]
    p0 = float(F.softmax(m(_entrada(x_uint8[None])), 1)[0, classe])
    cinza = np.array([int(255 * c) for c in data.MEDIA_IMAGENET], dtype=np.uint8)
    lote, pos = [], []
    for y in range(0, H - OCL_LADO + 1, OCL_PASSO):
        for x in range(0, W - OCL_LADO + 1, OCL_PASSO):
            t = x_uint8.copy()
            t[y:y + OCL_LADO, x:x + OCL_LADO] = cinza
            lote.append(t)
            pos.append((y, x))
    probs = probabilidades(m, np.stack(lote))[:, classe]
    soma, cont = np.zeros((H, W)), np.zeros((H, W))
    for (y, x), p in zip(pos, probs):
        soma[y:y + OCL_LADO, x:x + OCL_LADO] += p0 - p
        cont[y:y + OCL_LADO, x:x + OCL_LADO] += 1
    return soma / np.maximum(cont, 1)


@torch.no_grad()
def tapa_animal(m, x_uint8: np.ndarray, classe: int, caixa) -> dict:
    """p(classe) original, com o animal tapado e com só o animal visível (resto tapado)."""
    cinza = np.array([int(255 * c) for c in data.MEDIA_IMAGENET], dtype=np.uint8)
    x0, y0, x1, y1 = caixa
    sem = x_uint8.copy()
    sem[y0:y1, x0:x1] = cinza
    so = np.empty_like(x_uint8)
    so[:] = cinza
    so[y0:y1, x0:x1] = x_uint8[y0:y1, x0:x1]
    P = probabilidades(m, np.stack([x_uint8, sem, so]))
    return {"caixa_xyxy": list(caixa), "area_px": int((x1 - x0) * (y1 - y0)),
            "p_original": float(P[0, classe]), "p_animal_tapado": float(P[1, classe]),
            "p_so_animal": float(P[2, classe]),
            "prevista_animal_tapado": int(P[1].argmax()), "prevista_so_animal": int(P[2].argmax())}


def seleciona(y: np.ndarray, pred: np.ndarray, conf: np.ndarray, noite: np.ndarray) -> tuple[list, list]:
    """Aplica a regra declarada no cabeçalho. Devolve índices (erros, acertos)."""
    erros, usadas = [], set()
    for i in np.argsort(-conf):
        if y[i] == pred[i]:
            continue
        if y[i] in usadas or pred[i] in usadas:
            continue
        erros.append(int(i))
        usadas |= {int(y[i]), int(pred[i])}
        if len(erros) == N_ERROS:
            break
    rng = np.random.default_rng(SEMENTE)
    acertos, classes_usadas, falta = [], set(), {False: N_ACERTOS_DIA, True: N_ACERTOS_NOITE}
    for i in rng.permutation(np.flatnonzero(y == pred)):
        if y[i] in classes_usadas or falta[bool(noite[i])] == 0:
            continue
        acertos.append(int(i))
        classes_usadas.add(int(y[i]))
        falta[bool(noite[i])] -= 1
        if sum(falta.values()) == 0:
            break
    return erros, acertos


def _sobrepoe(ax, img, mapa, titulo, cmap="jet", vmin=None, vmax=None):
    ax.imshow(img)
    im = ax.imshow(mapa, cmap=cmap, alpha=0.45, vmin=vmin, vmax=vmax)
    ax.set_title(titulo, fontsize=9)
    ax.set_xticks([0, 56, 112, 168, 223])
    ax.set_yticks([0, 56, 112, 168, 223])
    ax.tick_params(labelsize=6)
    return im


def _painel(axs, caso, img, cam, ocl):
    titulo = (f"verdadeira: {caso['verdadeira']}\nprevista: {caso['prevista']} ({caso['confianca']:.0%})"
              f"\n{'sem cor (noite/IR)' if caso['noite'] else 'colorida (dia)'} · {caso['image_id']}")
    axs[0].imshow(img)
    if "tapa" in caso:
        t = caso["tapa"]
        x0, y0, x1, y1 = t["caixa_xyxy"]
        axs[0].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="lime", lw=1.2, ls="--"))
        titulo += (f"\np(prevista): animal tapado {t['p_animal_tapado']:.2f} · só o animal {t['p_so_animal']:.2f}")
    axs[0].set_title(titulo, fontsize=9)
    axs[0].set_xticks([0, 56, 112, 168, 223])
    axs[0].set_yticks([0, 56, 112, 168, 223])
    axs[0].tick_params(labelsize=6)
    _sobrepoe(axs[1], img, cam, "Grad-CAM (layer4)", vmin=0, vmax=1)
    lim = max(abs(ocl).max(), 1e-3)
    _sobrepoe(axs[2], img, ocl, f"oclusão 48 px (queda máx. {ocl.max():.2f})", cmap="bwr", vmin=-lim, vmax=lim)


def figura_caso(caso, img, cam, ocl, arq):
    fig, axs = plt.subplots(1, 3, figsize=(11, 4.5))
    _painel(axs, caso, img, cam, ocl)
    fig.tight_layout()
    GAL.mkdir(parents=True, exist_ok=True)
    fig.savefig(arq, dpi=150, bbox_inches="tight")
    plt.close(fig)


def figura_galeria(casos, imgs, cams, ocls, titulo, arq):
    fig, axs = plt.subplots(len(casos), 3, figsize=(11, 4.2 * len(casos)))
    for k, caso in enumerate(casos):
        _painel(axs[k], caso, imgs[k], cams[k], ocls[k])
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.25 / len(casos)))
    fig.suptitle(titulo, fontsize=12)
    fig.savefig(arq, dpi=130, bbox_inches="tight")
    plt.close(fig)


def figura_slide(casos, imgs, cams, arq):
    """Versão compacta para a apresentação: uma coluna por caso (imagem com a caixa do animal em cima,
    Grad-CAM embaixo), legenda curta com verdadeira/prevista/confiança e o teste de tapar o animal."""
    n = len(casos)
    fig, axs = plt.subplots(2, n, figsize=(3.2 * n, 7.4))
    for k, caso in enumerate(casos):
        t = caso.get("tapa")
        cor = "sem cor/IR" if caso["noite"] else "dia"
        rot = (f"{caso['verdadeira']}\n→ {caso['prevista']} ({caso['confianca']:.0%})\n{cor}"
               if caso["verdadeira"] != caso["prevista"] else
               f"{caso['verdadeira']} ({caso['confianca']:.0%})\n{cor}")
        axs[0, k].imshow(imgs[k])
        if t:
            x0, y0, x1, y1 = t["caixa_xyxy"]
            axs[0, k].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="lime", lw=2, ls="--"))
        axs[0, k].set_title(rot, fontsize=12)
        axs[1, k].imshow(imgs[k])
        axs[1, k].imshow(cams[k], cmap="jet", alpha=0.45, vmin=0, vmax=1)
        if t:
            axs[1, k].set_title(f"animal tapado: {t['p_original']:.2f} → {t['p_animal_tapado']:.2f}", fontsize=12)
        for a in axs[:, k]:
            a.set_xticks([])
            a.set_yticks([])
    axs[0, 0].set_ylabel("imagem (224 px)", fontsize=12)
    axs[1, 0].set_ylabel("Grad-CAM layer4", fontsize=12)
    fig.tight_layout()
    fig.savefig(arq, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    torch.manual_seed(0)
    m, pac = carrega_modelo()
    nomes = pac["classes"]
    assert nomes == data.classes()
    X, y, ids = data.conjunto_rotulado("test_public", pac["resolucao"])
    P = probabilidades(m, X)
    pred, conf = P.argmax(1), P.max(1)
    res = metrics.resumo(y, pred)
    print(f"conferência: F1 macro {res['f1_macro']:.4f}, acurácia {res['acuracia']:.4f} (esperado 0,4699 / 0,5512)")
    noite = data.flag_noite(X)
    erros, acertos = seleciona(y, pred, conf, noite)
    saida = {"regra": __doc__.split("Regra de seleção")[1].split("Grad-CAM (Selvaraju")[0].strip(),
             "caixas": "desenhadas à mão olhando as figuras, só para o teste de tapar o animal",
             "test_public": res, "n_erros": int((y != pred).sum()), "n_acertos": int((y == pred).sum())}
    for grupo, idxs in (("erro", erros), ("acerto", acertos)):
        casos, imgs, cams, ocls = [], [], [], []
        for k, i in enumerate(idxs, 1):
            top3 = np.argsort(-P[i])[:3]
            caso = {"k": k, "image_id": ids[i], "verdadeira": nomes[y[i]], "prevista": nomes[pred[i]],
                    "confianca": float(conf[i]), "noite": bool(noite[i]),
                    "p_verdadeira": float(P[i, y[i]]), "top3": [(nomes[j], float(P[i, j])) for j in top3]}
            cam = grad_cam(m, X[i], int(pred[i]))
            ocl = oclusao(m, X[i], int(pred[i]))
            # onde o Grad-CAM acende: centro de massa e caixa da região com mapa >= 0,5
            yy, xx = np.nonzero(cam >= 0.5)
            caso["cam_caixa_ge_0.5_xyxy"] = [int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())]
            caso["cam_frac_area_ge_0.5"] = float((cam >= 0.5).mean())
            yo, xo = np.unravel_index(np.argmax(ocl), ocl.shape)
            caso["oclusao_pico_xy"] = [int(xo), int(yo)]
            caso["oclusao_queda_max"] = float(ocl.max())
            if ids[i] in CAIXAS:
                t = tapa_animal(m, X[i], int(pred[i]), CAIXAS[ids[i]])
                t["prevista_animal_tapado"] = nomes[t["prevista_animal_tapado"]]
                t["prevista_so_animal"] = nomes[t["prevista_so_animal"]]
                caso["tapa"] = t
            arq = GAL / f"{grupo}_{k}_{ids[i]}.png"
            figura_caso(caso, X[i], cam, ocl, arq)
            caso["figura"] = str(arq.relative_to(RAIZ))
            casos.append(caso)
            imgs.append(X[i])
            cams.append(cam)
            ocls.append(ocl)
            print(grupo, k, json.dumps(caso, ensure_ascii=False))
        titulo = ("Erros de maior confiança (classes distintas) — modelo entregue, test_public" if grupo == "erro"
                  else "Acertos sorteados (semente 0; 3 dia + 2 noite, classes distintas) — modelo entregue, test_public")
        figura_galeria(casos, imgs, cams, ocls, titulo, FIGS / f"galeria_{grupo}s.png")
        figura_slide(casos, imgs, cams, FIGS / f"galeria_{grupo}s_slide.png")
        saida[grupo + "s"] = casos
    with open(RAIZ / "results" / "galeria.json", "w") as f:
        json.dump(saida, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
