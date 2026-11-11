"""Métricas: F1 macro (principal), acurácia, F1 por classe, matriz de confusão e pares confundidos."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def resumo(y, p, so_presentes: bool = False) -> dict:
    """F1 macro e acurácia. Com so_presentes=True o F1 macro é a média só sobre as classes que
    aparecem em `y` (usar em subconjuntos, p.ex. dia/noite: uma classe sem nenhuma imagem noturna
    não pode entrar como F1 = 0 na média)."""
    labels = np.unique(y) if so_presentes else np.arange(21)
    return {"f1_macro": float(f1_score(y, p, average="macro", labels=labels, zero_division=0)),
            "acuracia": float(accuracy_score(y, p)), "n_classes_na_media": int(len(labels))}


def f1_por_classe(y, p, n=21, nan_sem_suporte: bool = False) -> np.ndarray:
    f = f1_score(y, p, average=None, labels=np.arange(n), zero_division=0).astype(float)
    if nan_sem_suporte:
        f[np.bincount(y, minlength=n) == 0] = np.nan
    return f


def confusao(y, p, n=21) -> np.ndarray:
    return confusion_matrix(y, p, labels=np.arange(n))


def pares_confundidos(cm: np.ndarray, nomes: list[str], k: int = 10) -> list[dict]:
    """Os k pares (verdadeira -> prevista) fora da diagonal com mais imagens."""
    fora = cm.copy()
    np.fill_diagonal(fora, 0)
    ordem = np.dstack(np.unravel_index(np.argsort(-fora, axis=None), fora.shape))[0][:k]
    tot = cm.sum(1)
    return [{"verdadeira": nomes[i], "prevista": nomes[j], "n": int(fora[i, j]),
             "frac_da_classe": float(fora[i, j] / max(tot[i], 1))} for i, j in ordem if fora[i, j] > 0]
