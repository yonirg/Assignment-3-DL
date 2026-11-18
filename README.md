# PA3 — Poucos rótulos (Trilha C: professor e aluno)

Classificação de 21 espécies em imagens de câmera trap (ENA24-detection) com 5 rótulos por classe e
6.505 imagens sem rótulo. Um professor aberto (CLIP ViT-B/16, zero-shot) rotula o train; um aluno
pequeno (ResNet-18 ImageNet, 11.187.285 parâmetros) é treinado nesses pseudo-rótulos filtrados por
confiança (τ = 0,5, escolhido no val) junto com os 105 rótulos verdadeiros.

- Números (casa única): [`RESULTADOS.md`](RESULTADOS.md)
- Plano e regras: [`PLANO.md`](PLANO.md) · Uso de IA: [`AI_LOG.md`](AI_LOG.md) · Enunciado: `docs/enunciado.txt`
- Apresentação (com a fala nas notas): `apresentacao/PA3_apresentacao.pptx`

## Ambiente

Python 3.12 (o 3.14 do sistema não tem torch), testado num Mac M4 Pro (treino em MPS; avaliação em CPU).

```bash
uv venv --python 3.12 .venv
uv pip install -r requirements.txt     # versões fixadas
# dados (pa3_dados.zip, 415 MB) em data/ — só necessários para treinar e para o eval.sh
.venv/bin/gdown 1g7FGqtXOk_ziNKBwQAk8BxjPOaPCnbui -O data/pa3_dados.zip && (cd data && unzip -q pa3_dados.zip)
```

## Os dois comandos

**Treinar tudo** (Etapas 1 e 2: piso, professor, ablação, curva de orçamento, modelo entregue e
figuras; ~3 h no M4 Pro, um treino por vez):

```bash
bash scripts/run_all.sh
```

**Avaliar o modelo entregue** (roda `predict.py` em CPU, sem internet, no test_public e imprime F1
macro e acurácia recalculados do CSV; ~14 s para 811 imagens):

```bash
bash scripts/eval.sh
# modelo: torchvision resnet18, fc 512->21 | parâmetros: 11187285 (≤ 12 M)
# test_public: 811 imagens | F1 macro 0.4699 | acurácia 0.5512
```

`predict.py` sozinho (o que o monitor roda no test_private):

```bash
.venv/bin/python predict.py --imagens PASTA --saida preds.csv
```

CPU, sem nenhuma chamada de rede (`weights=None` + state_dict local), imprime o número de
parâmetros e escreve `image_id,label` (`image_id` = nome do arquivo sem extensão; `label` = nome da
classe de `classes.csv`). Depende só de `weights/aluno_final.pt` (44,8 MB), que já traz a lista de
classes, a resolução e a normalização; não lê `data/` nem `cache/`.

A galeria da Parte 3 (Grad-CAM, oclusão e teste de tapar o animal) é gerada por
`.venv/bin/python -m pa3.cam` (~1 min em CPU; precisa dos dados).

## Resultados (test_public, F1 macro)

| método | 1 rótulo/classe | 5 rótulos/classe | 20 rótulos/classe |
|---|---|---|---|
| Piso (b): ResNet-18 ImageNet só com os rótulos | 0,235 | 0,439 ± 0,011 | 0,625 |
| Professor zero-shot (CLIP ViT-B/16; não usa rótulos) | 0,375 | 0,375 | 0,375 |
| Professor melhorado (zero-shot + regressão logística nos rótulos) | 0,380 | 0,538 | 0,665 |
| **Aluno** (pseudo-rótulo duro do zero-shot, filtro τ = 0,5) | **0,417** | **0,480 ± 0,017** | **0,572** |
| Teto do monitor (todos os rótulos do train) | — | 0,686 | — |

Orçamento 5: média ± desvio de 3 seeds. Piso (a), CNN do zero com 105 imagens: 0,228 ± 0,024.
Modelo entregue (seed escolhida no val): F1 macro **0,4699**, acurácia 0,5512.

O que os números dizem, sem arredondar para cima: o aluno ganha do piso com 1 e 5 rótulos, mas perde
com 20; fica abaixo do professor melhorado com 5 e 20; a variante escolhida no val (B) foi só a quarta
no test_public (a D, rótulos suaves do professor melhorado, deu 0,543). Na galeria, a maioria dos
acertos sorteados se sustenta mesmo com o animal tapado: o modelo usa muito o fundo. Detalhes,
tabelas por classe e a discussão em [`RESULTADOS.md`](RESULTADOS.md); dados brutos da curva em
`curva_orcamento.csv`.

## Estrutura

```
pa3/            data.py (dados, cache, aumentação na GPU, flag noite/IR), models.py, metrics.py, plots.py,
                train_sup.py (piso), teacher.py (professor zero-shot e melhorado),
                train_student.py (aluno, ablação, curva, entrega), analise_aluno.py (curva e figuras),
                cam.py (Grad-CAM, oclusão e galeria da Parte 3)
scripts/        run_all.sh (treina tudo), eval.sh (avalia o modelo entregue)
predict.py      entregável (CPU, offline)
weights/        aluno_final.pt (modelo entregue)
results/        JSON/CSV de cada experimento, preds/ (logits), figs/ (PNG; galeria/ com um PNG por caso)
logs/           saída de cada comando
apresentacao/   PA3_apresentacao.pptx (fala de cada slide nas notas)
docs/           enunciado.txt
cache/, data/   fora do git (embeddings do professor, imagens pré-carregadas; pacote de dados)
curva_orcamento.csv, RESULTADOS.md, PLANO.md, AI_LOG.md
```

Comandos individuais de cada etapa (o que o `run_all.sh` encadeia):

```bash
.venv/bin/python -m pa3.train_sup resolucao      # 224 vs 320 no val
.venv/bin/python -m pa3.train_sup piso --res 224 # Parte 0: piso (a) e (b)
.venv/bin/python -m pa3.train_sup throughput     # tempo de época no train inteiro
.venv/bin/python -m pa3.teacher                  # professor zero-shot e melhorado
.venv/bin/python -m pa3.train_student tempo      # mede uma época do aluno
.venv/bin/python -m pa3.train_student ablacao    # ablação do ruído A-E (orçamento 5, seed 0), escolha no val
.venv/bin/python -m pa3.train_student metodo     # variante escolhida nos orçamentos 1, 5 (seeds 1, 2) e 20
.venv/bin/python -m pa3.train_student entrega    # melhor seed no val -> weights/aluno_final.pt
.venv/bin/python -m pa3.train_sup piso_b --orcamento 1 --seeds 0   # piso (b) para a curva (também 20; e 5 com seeds 0 1 2)
.venv/bin/python -m pa3.analise_aluno            # curva_orcamento.csv + figuras da Etapa 2
.venv/bin/python -m pa3.cam                      # Parte 3: galeria
```

## Pacote de dados (inspecionado)

```
data/pa3_dados/
  LEIAME.md, classes.csv                (colunas idx,label; 21 classes)
  labels/rotulos_1.csv  (21)  rotulos_5.csv (105)  rotulos_20.csv (420)   aninhados, todos ids do train
  labels/val.csv (202, até 10/classe; horse 2)   labels/test_public.csv (811; horse 4, bear 82)
  images/train/ (6.505)  images/val/ (202)  images/test_public/ (811)
```

- Todos os CSVs: colunas `image_id,label`. `image_id` = hash hexadecimal de 16 caracteres = nome do
  arquivo sem `.jpg` (ex.: `64e6f65bcf6e4dcf`).
- Imagens RGB JPEG, 512 px de largura, altura 270–370 (faixa de texto da câmera cortada). Para não
  cortar animais na borda, todas são achatadas para 224×224 sem recorte (treino e avaliação).
- Sem bounding box, sem câmera e sem horário como metadado.
- Imagens sem cor (noite/IR; flag em `pa3/data.py`): train 3.643/6.505, val 105/202, test_public 391/811.
