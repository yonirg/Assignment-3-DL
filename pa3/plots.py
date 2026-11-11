"""Figuras do PA3 (matplotlib, PNG em results/figs/)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FIGS = Path(__file__).resolve().parent.parent / "results" / "figs"


def _salva(fig, nome):
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / nome, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return FIGS / nome


def confusao(cm: np.ndarray, nomes: list[str], titulo: str, nome_arq: str, normaliza=True):
    """Matriz de confusão; normalizada por linha (fração da classe verdadeira) com contagens anotadas."""
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1) if normaliza else cm
    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1 if normaliza else None)
    ax.set_xticks(range(len(nomes)), nomes, rotation=90, fontsize=8)
    ax.set_yticks(range(len(nomes)), nomes, fontsize=8)
    ax.set_xlabel("prevista")
    ax.set_ylabel("verdadeira")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=6,
                        color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_title(titulo)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="fração da linha" if normaliza else "n")
    return _salva(fig, nome_arq)


def curva_treino_val(hist: dict, epoca_decora: int, epoca_escolhida: int, titulo: str, nome_arq: str):
    """hist: listas por época de perda_treino, perda_val, f1_val, acc_treino."""
    ep = np.arange(1, len(hist["perda_treino"]) + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
    a1.plot(ep, hist["perda_treino"], label="treino (105 imagens)")
    a1.plot(ep, hist["perda_val"], label="val (202 imagens)")
    a1.axvline(epoca_decora, color="red", ls="--", lw=1, label=f"mínimo da perda de val (época {epoca_decora})")
    a1.set_xlabel("época")
    a1.set_ylabel("entropia cruzada")
    a1.set_yscale("log")
    a1.legend(fontsize=8)
    a1.set_title("perda")
    a2.plot(ep, hist["acc_treino"], label="acurácia treino")
    a2.plot(ep, hist["f1_val"], label="F1 macro val")
    a2.axvline(epoca_escolhida, color="green", ls="--", lw=1, label=f"época escolhida no val ({epoca_escolhida})")
    a2.axvline(epoca_decora, color="red", ls="--", lw=1)
    a2.set_xlabel("época")
    a2.set_ylim(0, 1.02)
    a2.legend(fontsize=8)
    a2.set_title("acurácia de treino e F1 de validação")
    fig.suptitle(titulo)
    return _salva(fig, nome_arq)


def barras_dia_noite(nomes: list[str], f1_dia, f1_noite, n_dia, n_noite, titulo: str, nome_arq: str,
                     resumo_txt: str = ""):
    x = np.arange(len(nomes))
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.bar(x - 0.2, f1_dia, 0.4, label="dia (colorida)", color="#e0a030")
    ax.bar(x + 0.2, f1_noite, 0.4, label="noite / IR", color="#404a80")
    for i in range(len(nomes)):
        ax.text(x[i] - 0.2, 0.01, str(n_dia[i]), ha="center", fontsize=6, rotation=90, color="black")
        ax.text(x[i] + 0.2, 0.01, str(n_noite[i]), ha="center", fontsize=6, rotation=90, color="white")
    ax.set_xticks(x, nomes, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("F1 por classe")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title(titulo + ("\n" + resumo_txt if resumo_txt else ""), fontsize=10)
    return _salva(fig, nome_arq)
