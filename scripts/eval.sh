#!/usr/bin/env bash
# Avalia o modelo entregue (weights/aluno_final.pt) no test_public: roda predict.py em CPU, sem
# internet, e recalcula F1 macro e acurácia a partir do CSV gerado.
# Uso (da raiz do repositório; dados em data/pa3_dados):  bash scripts/eval.sh
# Esperado: F1 macro 0,4699 e acurácia 0,5512 (RESULTADOS.md, "Modelo entregue").
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
DADOS=data/pa3_dados
SAIDA=results/preds_test_public.csv

HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="" /usr/bin/time -p \
  $PY predict.py --imagens "$DADOS/images/test_public" --saida "$SAIDA"

$PY - "$SAIDA" "$DADOS/labels/test_public.csv" <<'EOF'
import sys
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

pred = pd.read_csv(sys.argv[1], dtype={"image_id": str})
ver = pd.read_csv(sys.argv[2], dtype={"image_id": str})
df = ver.merge(pred, on="image_id", suffixes=("_verdadeiro", "_previsto"), how="left")
assert df["label_previsto"].notna().all(), "há imagens do gabarito sem previsão"
assert len(pred) == len(ver), f"{len(pred)} previsões para {len(ver)} imagens"
f1 = f1_score(df["label_verdadeiro"], df["label_previsto"], average="macro")
acc = accuracy_score(df["label_verdadeiro"], df["label_previsto"])
print(f"test_public: {len(df)} imagens | F1 macro {f1:.4f} | acurácia {acc:.4f}")
EOF
