# PA3 — Poucos rótulos (Trilha C: professor e aluno)

Classificação de 21 espécies em imagens de câmera trap (ENA24-detection) com 5 rótulos por classe.
Plano: `PLANO.md`. Números: `RESULTADOS.md`. Enunciado em texto: `docs/enunciado.txt`.

## Ambiente

```bash
uv venv --python 3.12 .venv
uv pip install -r requirements.txt     # versões fixadas
# dados (pa3_dados.zip, 415 MB) em data/
.venv/bin/gdown 1g7FGqtXOk_ziNKBwQAk8BxjPOaPCnbui -O data/pa3_dados.zip && (cd data && unzip -q pa3_dados.zip)
```

## Etapa 1 (comandos)

```bash
.venv/bin/python -m pa3.train_sup resolucao      # 224 vs 320 no val
.venv/bin/python -m pa3.train_sup piso --res 224 # Parte 0: piso (a) e (b)
.venv/bin/python -m pa3.train_sup throughput     # tempo de época no train inteiro
.venv/bin/python -m pa3.teacher                  # professor zero-shot e melhorado
```

## Estrutura do pacote de dados (inspecionada)

```
data/pa3_dados/
  LEIAME.md, classes.csv                (colunas idx,label; 21 classes)
  labels/rotulos_1.csv  (21)  rotulos_5.csv (105)  rotulos_20.csv (420)   aninhados, todos ids do train
  labels/val.csv (202, até 10/classe; horse 2)   labels/test_public.csv (811; horse 4, bear 82)
  images/train/ (6.505)  images/val/ (202)  images/test_public/ (811)
```

- Todos os CSVs: colunas `image_id,label`. `image_id` = hash hexadecimal de 16 caracteres = nome do
  arquivo sem `.jpg` (ex.: `64e6f65bcf6e4dcf`).
- Imagens RGB JPEG, 512 px de largura, altura 270–370 (faixa de texto da câmera cortada).
- Sem bounding box, sem câmera e sem horário como metadado.
- Classes: american_black_bear, american_crow, bird, bobcat, chicken, coyote, dog, domestic_cat,
  eastern_chipmunk, eastern_cottontail, eastern_fox_squirrel, eastern_gray_squirrel, grey_fox, horse,
  northern_raccoon, red_fox, striped_skunk, virginia_opossum, white_tailed_deer, wild_turkey, woodchuck.
- Imagens noturnas/IR (sem cor; flag em `pa3/data.py`): train 3.643/6.505, val 105/202, test_public 391/811.
