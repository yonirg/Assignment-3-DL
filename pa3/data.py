"""Dados do PA3: leitura dos CSVs, pré-carga das imagens em memória e aumentação em lote na GPU.

Estrutura do pacote (inspecionada na Etapa 1, ver README/RESULTADOS):
    data/pa3_dados/classes.csv              colunas idx,label (21 classes)
    data/pa3_dados/labels/{rotulos_1,rotulos_5,rotulos_20,val,test_public}.csv   colunas image_id,label
    data/pa3_dados/images/{train,val,test_public}/<image_id>.jpg
`image_id` é um hash hexadecimal de 16 caracteres (nome do arquivo sem `.jpg`).

Decisões:
- As imagens têm proporções variadas (512 x 270..370). Para não perder as bordas (o animal pode
  estar no canto), cada imagem é redimensionada para um quadrado `res x res` SEM corte (achatada).
  O mesmo vale para treino e avaliação, então a distorção é consistente.
- As imagens ficam pré-carregadas como uint8 (N, res, res, 3) e salvas em `cache/` para reuso.
- A aumentação (RandomResizedCrop scale>=0,5, flip, color jitter leve) é feita em lote na GPU com
  `roi_align`, para o treino não ficar preso em decodificação/transformação na CPU.
- Flag "noturna/IR": média de (max-min) entre canais RGB < 3 (em 0..255). No test_public o
  histograma é bimodal: 391 imagens < 1 e o resto > 8, sem nada entre 1 e 8.
"""
from __future__ import annotations

import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

RAIZ = Path(__file__).resolve().parent.parent
DADOS = RAIZ / "data" / "pa3_dados"
CACHE = RAIZ / "cache"

MEDIA_IMAGENET = (0.485, 0.456, 0.406)
DESVIO_IMAGENET = (0.229, 0.224, 0.225)
LIMIAR_NOITE = 3.0


def classes() -> list[str]:
    df = pd.read_csv(DADOS / "classes.csv").sort_values("idx")
    return df["label"].tolist()


def rotulos(nome: str) -> pd.DataFrame:
    """nome em {rotulos_1, rotulos_5, rotulos_20, val, test_public}. Devolve image_id, label, y."""
    df = pd.read_csv(DADOS / "labels" / f"{nome}.csv", dtype={"image_id": str})
    mapa = {c: i for i, c in enumerate(classes())}
    df["y"] = df["label"].map(mapa).astype(int)
    return df


def ids_train() -> list[str]:
    """Todos os 6.505 ids do train, em ordem alfabética (ordem canônica do cache)."""
    return sorted(f[:-4] for f in os.listdir(DADOS / "images" / "train") if f.endswith(".jpg"))


def _pasta(split: str) -> Path:
    return DADOS / "images" / ("train" if split.startswith("rotulos") else split)


def _carrega(caminho: str, res: int) -> np.ndarray:
    im = Image.open(caminho).convert("RGB").resize((res, res), Image.BICUBIC)
    return np.asarray(im, dtype=np.uint8)


def imagens(split: str, res: int) -> tuple[list[str], np.ndarray]:
    """Imagens de um split (train/val/test_public) como uint8 (N,res,res,3), com cache em disco."""
    assert split in ("train", "val", "test_public")
    CACHE.mkdir(exist_ok=True)
    arq = CACHE / f"imgs_{split}_{res}.npy"
    if split == "train":
        ids = ids_train()
    else:
        ids = rotulos(split)["image_id"].tolist()
    if arq.exists():
        arr = np.load(arq, mmap_mode=None)
        assert len(arr) == len(ids)
        return ids, arr
    caminhos = [str(_pasta(split) / f"{i}.jpg") for i in ids]
    with Pool(8) as p:
        lista = p.map(partial(_carrega, res=res), caminhos, chunksize=32)
    arr = np.stack(lista)
    np.save(arq, arr)
    return ids, arr


def flag_noite(arr: np.ndarray) -> np.ndarray:
    """True para imagens noturnas/IR (sem cor). arr uint8 (N,H,W,3)."""
    out = np.empty(len(arr), dtype=bool)
    for i in range(0, len(arr), 256):
        a = arr[i:i + 256, ::4, ::4].astype(np.int16)
        out[i:i + 256] = (a.max(-1) - a.min(-1)).mean(axis=(1, 2)) < LIMIAR_NOITE
    return out


def conjunto_rotulado(nome: str, res: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """X uint8 (N,res,res,3), y (N,), ids para rotulos_K / val / test_public."""
    df = rotulos(nome)
    if nome.startswith("rotulos"):
        ids, arr = imagens("train", res)
        pos = {i: k for k, i in enumerate(ids)}
        idx = np.array([pos[i] for i in df["image_id"]])
        return arr[idx], df["y"].to_numpy(), df["image_id"].tolist()
    ids, arr = imagens(nome, res)
    assert ids == df["image_id"].tolist()
    return arr, df["y"].to_numpy(), ids


# ---------------------------------------------------------------- tensores e aumentação na GPU

def para_tensor(x_uint8: np.ndarray, device) -> torch.Tensor:
    """(N,H,W,3) uint8 -> (N,3,H,W) float em [0,1] no device."""
    return torch.from_numpy(x_uint8).to(device).permute(0, 3, 1, 2).float().div_(255.0)


def normaliza(x: torch.Tensor, media=MEDIA_IMAGENET, desvio=DESVIO_IMAGENET) -> torch.Tensor:
    m = torch.tensor(media, device=x.device).view(1, 3, 1, 1)
    s = torch.tensor(desvio, device=x.device).view(1, 3, 1, 1)
    return (x - m) / s


def aumenta(x: torch.Tensor, gen: torch.Generator, escala=(0.5, 1.0), razao=(3 / 4, 4 / 3),
            jitter=0.2) -> torch.Tensor:
    """RandomResizedCrop + flip horizontal + color jitter leve, em lote. x float (N,3,H,W) em [0,1].

    `gen` é um gerador de CPU (semente fixa) para a aumentação ser reprodutível.
    """
    from torchvision.ops import roi_align

    n, _, h, w = x.shape
    area = torch.empty(n).uniform_(*escala, generator=gen) * h * w
    logr = torch.empty(n).uniform_(np.log(razao[0]), np.log(razao[1]), generator=gen)
    r = torch.exp(logr)
    cw = torch.sqrt(area * r).clamp(max=w)
    ch = torch.sqrt(area / r).clamp(max=h)
    x0 = torch.rand(n, generator=gen) * (w - cw)
    y0 = torch.rand(n, generator=gen) * (h - ch)
    caixas = torch.stack([torch.arange(n, dtype=torch.float32), x0, y0, x0 + cw, y0 + ch], 1)
    out = roi_align(x, caixas.to(x.device), (h, w), spatial_scale=1.0, sampling_ratio=2, aligned=True)
    flip = torch.rand(n, generator=gen) < 0.5
    out[flip.to(x.device)] = out[flip.to(x.device)].flip(-1)
    if jitter > 0:
        b = (1 + (torch.rand(n, generator=gen) * 2 - 1) * jitter).to(x.device).view(n, 1, 1, 1)
        c = (1 + (torch.rand(n, generator=gen) * 2 - 1) * jitter).to(x.device).view(n, 1, 1, 1)
        s = (1 + (torch.rand(n, generator=gen) * 2 - 1) * jitter).to(x.device).view(n, 1, 1, 1)
        out = out * b
        media = out.mean(dim=(1, 2, 3), keepdim=True)
        out = (out - media) * c + media
        cinza = out.mean(dim=1, keepdim=True)
        out = (out - cinza) * s + cinza
        out = out.clamp_(0, 1)
    return out
