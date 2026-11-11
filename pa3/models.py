"""Modelos: CNN pequena do zero (piso a) e ResNet-18 ImageNet (piso b e aluno)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision


class CNNPequena(nn.Module):
    """4 blocos conv-BN-ReLU-pool + GAP + linear (piso a, treinada do zero)."""

    def __init__(self, n_classes: int = 21, larguras=(32, 64, 128, 256)):
        super().__init__()
        camadas, c_in = [], 3
        for c in larguras:
            camadas += [nn.Conv2d(c_in, c, 3, padding=1, bias=False), nn.BatchNorm2d(c),
                        nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            c_in = c
        self.features = nn.Sequential(*camadas)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c_in, n_classes)

    def forward(self, x):
        return self.fc(torch.flatten(self.pool(self.features(x)), 1))


def resnet18(n_classes: int = 21, imagenet: bool = True) -> nn.Module:
    """ResNet-18 do torchvision; com imagenet=True parte dos pesos IMAGENET1K_V1."""
    pesos = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if imagenet else None
    m = torchvision.models.resnet18(weights=pesos)
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    return m


def cria(nome: str, n_classes: int = 21) -> nn.Module:
    if nome == "cnn":
        return CNNPequena(n_classes)
    if nome == "resnet18":
        return resnet18(n_classes, imagenet=True)
    raise ValueError(nome)


def n_parametros(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
