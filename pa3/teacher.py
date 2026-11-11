"""Professor (Trilha C): BioCLIP e CLIP ViT-B/16 (OpenAI) via open_clip, zero-shot e melhorado.

    python -m pa3.teacher

Regras (PLANO.md):
- O professor vê só `train` e `test_public` (este só para medir). Este arquivo NUNCA lê o `val`.
- A escolha do professor (e do pré-processamento) é feita pela acurácia zero-shot nos 105 rótulos
  de `rotulos_5.csv` (subconjunto do train). O test_public é avaliado para todos, só para reportar.
- Professor melhorado: regressão logística sobre os embeddings com os rótulos do orçamento,
  combinada com o zero-shot: log p = α·log_softmax(zs) + (1-α)·log_softmax(lr) (renormalizado).
  α e C por validação cruzada DENTRO dos rótulos do orçamento, em dobras por grupo: a dobra j
  segura o j-ésimo rótulo de cada classe. Pelo enunciado, os K rótulos de uma classe vêm de K
  câmeras-noite diferentes, então nenhuma câmera-noite de uma classe fica dos dois lados da dobra.
  Com orçamento 1 não há como validar (segurar o único rótulo tira a classe do treino): usamos
  valores fixados a priori (α=0,5, C=1), declarados aqui antes de rodar.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss

from pa3 import data, metrics, plots

RAIZ = Path(__file__).resolve().parent.parent
CACHE = RAIZ / "cache"
RES_DIR = RAIZ / "results"

PROFESSORES = {
    "bioclip": {"modelo": "hf-hub:imageomics/bioclip", "pretrained": None},
    "clip_b16": {"modelo": "ViT-B-16", "pretrained": "openai"},
}
PREPROCS = ("recorte", "achatada")  # recorte = pré-processamento padrão do open_clip (lado menor 224 + center crop)
TEMPLATE = "a photo of a {}"
SPLITS_PROFESSOR = ("train", "test_public")  # nunca "val"

GRADE_C = [0.01, 0.1, 1.0, 10.0, 100.0]
GRADE_ALFA = [round(float(a), 2) for a in np.linspace(0, 1, 11)]
PRIORI_R1 = {"alfa": 0.5, "C": 1.0}


def _dispositivo():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def carrega_professor(nome: str, device):
    import open_clip

    cfg = PROFESSORES[nome]
    if cfg["pretrained"]:
        model, _, preprocess = open_clip.create_model_and_transforms(cfg["modelo"], pretrained=cfg["pretrained"])
        tok = open_clip.get_tokenizer(cfg["modelo"])
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(cfg["modelo"])
        tok = open_clip.get_tokenizer(cfg["modelo"])
    return model.to(device).eval(), preprocess, tok


def _ids(split):
    return data.ids_train() if split == "train" else data.rotulos(split)["image_id"].tolist()


@torch.no_grad()
def embeddings_imagem(nome: str, prep: str, split: str, device, model=None, preprocess=None, bs=64):
    assert split in SPLITS_PROFESSOR, "o professor nunca vê o val"
    arq = CACHE / f"prof_{nome}_{prep}_{split}_img.npy"
    if arq.exists():
        return np.load(arq)
    ids = _ids(split)
    pasta = data.DADOS / "images" / split
    # média/desvio vêm do próprio transform do modelo
    norm = [t for t in preprocess.transforms if t.__class__.__name__ == "Normalize"][0]
    out = []
    for i in range(0, len(ids), bs):
        lote = []
        for j in ids[i:i + bs]:
            im = Image.open(pasta / f"{j}.jpg").convert("RGB")
            if prep == "recorte":
                lote.append(preprocess(im))
            else:
                im = im.resize((224, 224), Image.BICUBIC)
                t = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255.0
                lote.append(norm(t))
        f = model.encode_image(torch.stack(lote).to(device)).float()
        out.append(torch.nn.functional.normalize(f, dim=-1).cpu())
    emb = torch.cat(out).numpy()
    np.save(arq, emb)
    return emb


@torch.no_grad()
def embeddings_texto(nome, device, model, tok):
    textos = [TEMPLATE.format(c.replace("_", " ")) for c in data.classes()]
    t = model.encode_text(tok(textos).to(device)).float()
    t = torch.nn.functional.normalize(t, dim=-1).cpu().numpy()
    escala = float(model.logit_scale.exp().item())
    np.save(CACHE / f"prof_{nome}_texto.npy", t)
    return t, escala, textos


def log_softmax(z):
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def lr_logprob(Xtr, ytr, X, C, n=21):
    """log-probabilidades (N,21) da regressão logística; classes ausentes no treino ficam com -inf."""
    clf = LogisticRegression(C=C, max_iter=5000)
    clf.fit(Xtr, ytr)
    lp = np.full((len(X), n), -np.inf)
    lp[:, clf.classes_] = clf.predict_log_proba(X)
    return lp


def combina(zs_logit, lp_lr, alfa):
    lz = log_softmax(zs_logit)
    m = alfa * lz + (1 - alfa) * lp_lr if alfa < 1 else lz
    return log_softmax(m)


def professor_melhorado(F_tr, Z_tr, orc_df, ids_train):
    """Escolhe (α, C) por CV dentro dos rótulos e devolve (hp, tabela_cv, função de predição)."""
    pos = {i: k for k, i in enumerate(ids_train)}
    idx = np.array([pos[i] for i in orc_df["image_id"]])
    y = orc_df["y"].to_numpy()
    X, Z = F_tr[idx], Z_tr[idx]
    dobra = orc_df.groupby("label").cumcount().to_numpy()  # j-ésimo rótulo de cada classe
    K = dobra.max() + 1
    tabela = []
    if K == 1:
        hp = dict(PRIORI_R1, criterio="orçamento 1: sem CV possível; valores a priori")
    else:
        for C in GRADE_C:
            lp_oof = np.zeros((len(y), 21))
            for j in range(K):
                tr, te = dobra != j, dobra == j
                lp_oof[te] = lr_logprob(X[tr], y[tr], X[te], C)
            for a in GRADE_ALFA:
                lp = combina(Z, lp_oof, a)
                p = lp.argmax(1)
                tabela.append({"C": C, "alfa": a,
                               "f1_cv": float(f1_score(y, p, average="macro", labels=np.arange(21), zero_division=0)),
                               "acc_cv": float((p == y).mean()),
                               "logloss_cv": float(log_loss(y, np.exp(lp), labels=np.arange(21)))})
        melhor = max(tabela, key=lambda r: (r["f1_cv"], -r["logloss_cv"]))
        hp = {"alfa": melhor["alfa"], "C": melhor["C"], "f1_cv": melhor["f1_cv"], "acc_cv": melhor["acc_cv"],
              "criterio": f"CV {K} dobras (j-ésimo rótulo de cada classe); maior F1 macro, desempate menor log-loss"}
    clf_X, clf_y = X, y

    def prediz(F, Zl):
        return combina(Zl, lr_logprob(clf_X, clf_y, F, hp["C"]), hp["alfa"])

    return hp, tabela, prediz


def main():
    CACHE.mkdir(exist_ok=True)
    RES_DIR.mkdir(exist_ok=True)
    device = _dispositivo()
    nomes = data.classes()
    ids_tr = data.ids_train()
    te = data.rotulos("test_public")
    yte = te["y"].to_numpy()
    r5 = data.rotulos("rotulos_5")
    pos = {i: k for k, i in enumerate(ids_tr)}
    idx5 = np.array([pos[i] for i in r5["image_id"]])

    saida = {"template": TEMPLATE, "configs": {}}
    emb = {}
    for nome in PROFESSORES:
        model, preprocess, tok = carrega_professor(nome, device)
        T, escala, textos = embeddings_texto(nome, device, model, tok)
        saida["prompts"] = textos
        for prep in PREPROCS:
            F = {s: embeddings_imagem(nome, prep, s, device, model, preprocess) for s in SPLITS_PROFESSOR}
            Z = {s: escala * F[s] @ T.T for s in F}
            emb[(nome, prep)] = (F, Z, escala)
            p5 = Z["train"][idx5].argmax(1)
            pt = Z["test_public"].argmax(1)
            c = {"escala_logit": escala,
                 "acc_zs_rotulos5": float((p5 == r5["y"].to_numpy()).mean()),
                 "f1_zs_rotulos5": metrics.resumo(r5["y"].to_numpy(), p5)["f1_macro"],
                 "test_f1_macro": metrics.resumo(yte, pt)["f1_macro"],
                 "test_acuracia": metrics.resumo(yte, pt)["acuracia"]}
            saida["configs"][f"{nome}/{prep}"] = c
            print(nome, prep, c, flush=True)
        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    # ---- escolha do professor: SÓ pela acurácia zero-shot nos 105 rótulos do train
    escolha = max(saida["configs"], key=lambda k: (saida["configs"][k]["acc_zs_rotulos5"],
                                                    saida["configs"][k]["f1_zs_rotulos5"]))
    saida["escolhido"] = escolha
    saida["criterio_escolha"] = "maior acurácia zero-shot nos 105 rótulos de rotulos_5 (desempate F1 macro neles)"
    nome, prep = escolha.split("/")
    F, Z, escala = emb[(nome, prep)]
    print("professor escolhido:", escolha, flush=True)

    # ---- análise do zero-shot no test_public (= matriz de ruído)
    pt = Z["test_public"].argmax(1)
    cm = metrics.confusao(yte, pt)
    f1c = metrics.f1_por_classe(yte, pt)
    _, arr_te = data.imagens("test_public", 224)
    noite = data.flag_noite(arr_te)
    analise = {"resumo": metrics.resumo(yte, pt), "confusao": cm.tolist(),
               "f1_por_classe": dict(zip(nomes, f1c.round(4).tolist())),
               "suporte_por_classe": dict(zip(nomes, cm.sum(1).tolist())),
               "pares_confundidos": metrics.pares_confundidos(cm, nomes, 12),
               "previsoes_por_classe": dict(zip(nomes, np.bincount(pt, minlength=21).tolist())),
               "n_noite": int(noite.sum()), "n_dia": int((~noite).sum()),
               "dia": metrics.resumo(yte[~noite], pt[~noite], so_presentes=True),
               "noite": metrics.resumo(yte[noite], pt[noite], so_presentes=True),
               "bounding_box": "o pacote não traz bounding box: quebra por tamanho do animal não é possível"}
    f1_dia = metrics.f1_por_classe(yte[~noite], pt[~noite], nan_sem_suporte=True)
    f1_noite = metrics.f1_por_classe(yte[noite], pt[noite], nan_sem_suporte=True)
    n_dia = np.bincount(yte[~noite], minlength=21)
    n_noite = np.bincount(yte[noite], minlength=21)
    analise["dia_noite_por_classe"] = {c: {"f1_dia": None if np.isnan(f1_dia[i]) else float(f1_dia[i]), "n_dia": int(n_dia[i]),
                                           "f1_noite": None if np.isnan(f1_noite[i]) else float(f1_noite[i]), "n_noite": int(n_noite[i])}
                                       for i, c in enumerate(nomes)}
    # previsões noturnas: para onde vão
    analise["previsoes_noite"] = dict(zip(nomes, np.bincount(pt[noite], minlength=21).tolist()))
    saida["analise_zs_escolhido"] = analise
    plots.confusao(cm, nomes, f"Professor zero-shot ({escolha}, '{TEMPLATE}') — test_public = matriz de ruído "
                   f"(F1 macro {analise['resumo']['f1_macro']:.3f})", "professor_confusao.png")
    plots.barras_dia_noite(nomes, f1_dia, f1_noite, n_dia, n_noite,
                           f"Professor zero-shot ({escolha}) — F1 por classe no test_public, dia × noite "
                           "(números = imagens)",
                           "professor_dia_noite.png",
                           f"dia: F1 macro {analise['dia']['f1_macro']:.3f}, acc {analise['dia']['acuracia']:.3f} (n={analise['n_dia']})"
                           f" | noite/IR: F1 macro {analise['noite']['f1_macro']:.3f}, acc {analise['noite']['acuracia']:.3f} (n={analise['n_noite']})")

    # ---- probabilidades zero-shot para o train inteiro
    np.save(CACHE / "prof_zs_train_logits.npy", Z["train"])
    np.save(CACHE / "prof_zs_train_probs.npy", np.exp(log_softmax(Z["train"])))
    with open(CACHE / "prof_train_ids.json", "w") as f:
        json.dump({"professor": escolha, "ids": ids_tr}, f)

    # ---- professor melhorado com os rótulos, orçamentos 1, 5, 20
    saida["melhorado"] = {}
    for K in (1, 5, 20):
        orc = data.rotulos(f"rotulos_{K}")
        hp, tabela, prediz = professor_melhorado(F["train"], Z["train"], orc, ids_tr)
        lp_te = prediz(F["test_public"], Z["test_public"])
        p = lp_te.argmax(1)
        r = metrics.resumo(yte, p)
        lp_tr = prediz(F["train"], Z["train"])
        np.save(CACHE / f"prof_melhorado_r{K}_train_probs.npy", np.exp(lp_tr))
        np.save(CACHE / f"prof_melhorado_r{K}_train_logprobs.npy", lp_tr)
        saida["melhorado"][str(K)] = {"hp": hp, "test": r,
                                      "dia": metrics.resumo(yte[~noite], p[~noite], so_presentes=True),
                                      "noite": metrics.resumo(yte[noite], p[noite], so_presentes=True),
                                      "f1_por_classe": dict(zip(nomes, metrics.f1_por_classe(yte, p).round(4).tolist())),
                                      "cv": tabela}
        print("melhorado", K, hp, r, flush=True)
    # também: zero-shot do outro professor, para a tabela (só reporte)
    with open(RES_DIR / "professor.json", "w") as f:
        json.dump(saida, f, indent=2, ensure_ascii=False)
    pd.DataFrame([{"config": k, **v} for k, v in saida["configs"].items()]).to_csv(RES_DIR / "professor_configs.csv", index=False)


if __name__ == "__main__":
    main()
