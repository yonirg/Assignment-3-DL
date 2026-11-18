#!/usr/bin/env bash
# Reproduz todos os experimentos do PA3 (Etapas 1, 2 e 3), um treino por vez na GPU (MPS).
# Uso: bash scripts/run_all.sh            (da raiz do repositório; dados em data/pa3_dados)
# Tempo total aproximado no M4 Pro: ~3 h (Etapa 1 ~40 min, Etapa 2 ~2 h).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p logs

# ---- Etapa 1: piso (Parte 0) e professor
$PY -u -m pa3.train_sup resolucao      2>&1 | tee logs/piso_resolucao.log   # 224 vs 320 no val
$PY -u -m pa3.train_sup piso --res 224 2>&1 | tee logs/piso_r5.log          # piso (a) e (b), 3 seeds
$PY -u -m pa3.train_sup throughput     2>&1 | tee logs/throughput.log
$PY -u -m pa3.teacher                  2>&1 | tee logs/professor.log        # zero-shot + melhorado (cache/)

# ---- Etapa 2: aluno, ablação do ruído e curva de orçamento (Parte 1 Trilha C e Parte 2)
$PY -u -m pa3.train_student tempo      2>&1 | tee logs/aluno_tempo.log      # mede uma época
$PY -u -m pa3.train_student ablacao    2>&1 | tee logs/aluno_ablacao.log    # A-E, orçamento 5, seed 0; escolha no val
$PY -u -m pa3.train_student metodo     2>&1 | tee logs/aluno_metodo.log     # variante escolhida: 1, 5 (seeds 1-2), 20
$PY -u -m pa3.train_student entrega    2>&1 | tee logs/aluno_entrega.log    # melhor seed no val -> weights/aluno_final.pt
for K in 1 20; do
  $PY -u -m pa3.train_sup piso_b --orcamento $K --seeds 0 2>&1 | tee logs/piso_b_r$K.log
done
$PY -u -m pa3.train_sup piso_b --orcamento 5 --seeds 0 1 2 2>&1 | tee logs/piso_b_r5.log  # refaz (b) guardando logits por seed
$PY -u -m pa3.analise_aluno            2>&1 | tee logs/analise_aluno.log    # curva_orcamento.csv + figuras

# ---- Etapa 3: galeria da Parte 3 (CPU, ~1 min) e avaliação do modelo entregue
$PY -u -m pa3.cam                      2>&1 | tee logs/galeria.log          # Grad-CAM, oclusão, tapar o animal
bash scripts/eval.sh                   2>&1 | tee logs/eval.log             # predict.py no test_public + F1
