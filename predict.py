"""Classifica as imagens de uma pasta com o modelo entregue do PA3 (aluno ResNet-18, Trilha C).

    python predict.py --imagens PASTA --saida preds.csv [--pesos weights/aluno_final.pt]

- Roda em CPU e sem internet: a arquitetura é criada com `weights=None` e os pesos vêm do
  state_dict local (`weights/aluno_final.pt`). Nenhuma chamada de rede.
- Não depende de `data/` nem de `cache/`: a lista de classes (na ordem de `classes.csv`), a
  resolução e a normalização vêm do próprio checkpoint.
- Pré-processamento idêntico ao do treino: RGB, redimensionamento bicúbico para 224×224 SEM recorte
  (imagem achatada), normalização ImageNet.
- Imprime o número de parâmetros (`sum(p.numel() for p in model.parameters())`).
- Escreve um CSV `image_id,label`, com `image_id` = nome do arquivo sem extensão (como nos CSVs do
  pacote) e `label` = nome da classe.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image

EXTENSOES = {".jpg", ".jpeg", ".png"}
RAIZ = Path(__file__).resolve().parent


def carrega_modelo(caminho: Path):
    pac = torch.load(caminho, map_location="cpu", weights_only=False)
    model = torchvision.models.resnet18(weights=None)  # sem download: pesos vêm do arquivo local
    model.fc = nn.Linear(model.fc.in_features, len(pac["classes"]))
    model.load_state_dict(pac["state_dict"])
    model.eval()
    return model, pac


def carrega_imagem(caminho: Path, res: int) -> np.ndarray:
    im = Image.open(caminho).convert("RGB").resize((res, res), Image.BICUBIC)
    return np.asarray(im, dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--imagens", required=True, help="pasta com as imagens (.jpg/.jpeg/.png)")
    ap.add_argument("--saida", required=True, help="CSV de saída (image_id,label)")
    ap.add_argument("--pesos", default=str(RAIZ / "weights" / "aluno_final.pt"))
    ap.add_argument("--lote", type=int, default=32)
    a = ap.parse_args()

    t0 = time.time()
    torch.set_grad_enabled(False)
    device = torch.device("cpu")
    model, pac = carrega_modelo(Path(a.pesos))
    n_par = sum(p.numel() for p in model.parameters())
    print(f"modelo: {pac.get('arquitetura', 'resnet18')} | parâmetros: {n_par} "
          f"({'≤' if n_par <= 12_000_000 else '>'} 12 M)")
    classes, res = pac["classes"], pac["resolucao"]
    media = torch.tensor(pac["media"]).view(1, 3, 1, 1)
    desvio = torch.tensor(pac["desvio"]).view(1, 3, 1, 1)

    arquivos = sorted(p for p in Path(a.imagens).iterdir() if p.suffix.lower() in EXTENSOES)
    if not arquivos:
        raise SystemExit(f"nenhuma imagem em {a.imagens}")
    linhas = []
    for i in range(0, len(arquivos), a.lote):
        lote = arquivos[i:i + a.lote]
        x = torch.from_numpy(np.stack([carrega_imagem(p, res) for p in lote]))
        x = x.permute(0, 3, 1, 2).float().div_(255.0)
        x = (x - media) / desvio
        pred = model(x.to(device)).argmax(1).tolist()
        linhas += [(p.stem, classes[k]) for p, k in zip(lote, pred)]

    with open(a.saida, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "label"])
        w.writerows(linhas)
    print(f"{len(linhas)} imagens classificadas em {time.time() - t0:.1f} s (CPU) -> {a.saida}")


if __name__ == "__main__":
    main()
