# RESULTADOS

Casa única dos números do PA3. Cada número abaixo foi copiado do arquivo de origem indicado,
gerado pelos scripts do repositório. Métrica principal: F1 macro no `test_public` (811 imagens).
Nenhuma escolha olhou para o `test_public`; o professor nunca viu o `val`.

## Etapa 1 — Piso e professor

### Resolução (escolhida no val)

Origem: `results/piso_resolucao.json` (log `logs/piso_resolucao.log`; comando `python -m pa3.train_sup resolucao`).
Piso (b) ResNet-18 ImageNet, `rotulos_5` (105 imagens), 3 seeds, F1 macro **no val** na época escolhida no val.

| resolução | F1 val (média ± desvio, 3 seeds) |
|---|---|
| 224 | 0,4023 ± 0,0208 |
| 320 | 0,4010 ± 0,0099 |

**Fixada: 224** (maior média no val; a diferença de 0,0013 é empate na prática, e 224 é ~2× mais barata).

### Parte 0 — Piso (só `rotulos_5.csv`, 224 px)

Origem: `results/piso_r5.json` (log `logs/piso_r5.log`; comando `python -m pa3.train_sup piso --res 224`).
Época escolhida pelo maior F1 macro no val (desempate: menor perda de val).

| modelo | parâmetros | seed | época (val) | F1 val | **F1 macro test_public** | acurácia test_public |
|---|---|---|---|---|---|---|
| (a) CNN do zero (4 blocos conv-BN-ReLU-pool + GAP + linear) | 394.293 | 0 | 86 | 0,2595 | **0,2612** | 0,3033 |
| (a) | | 1 | 94 | 0,2501 | 0,2171 | 0,2774 |
| (a) | | 2 | 110 | 0,2381 | 0,2055 | 0,2552 |
| (a) média ± desvio | | | | | 0,2279 ± 0,0240 | 0,2787 ± 0,0197 |
| (b) ResNet-18 ImageNet, fine-tuning | 11.187.285 | 0 | 42 | 0,4147 | **0,4520** | 0,4784 |
| (b) | | 1 | 44 | 0,3731 | 0,4252 | 0,4353 |
| (b) | | 2 | 7 | 0,4192 | 0,4408 | 0,4649 |
| (b) média ± desvio | | | | | 0,4393 ± 0,0110 | 0,4595 ± 0,0180 |

Hiperparâmetros fixos (não ajustados): (b) AdamW lr 3e-4, wd 0,05, 60 épocas, batch 32, cosine;
(a) AdamW lr 1e-3, wd 0,05, 150 épocas, batch 32, cosine. Aumentação: RandomResizedCrop
(escala 0,5–1, razão 3/4–4/3), flip horizontal, color jitter ±0,2 (brilho/contraste/saturação).
Imagens achatadas para 224×224 sem recorte.

Figuras: `results/figs/piso_confusao.png` (matriz de confusão de (b), seed 0, test_public) e
`results/figs/piso_curva.png` (curva treino × val de (b), seed 0).

**Quando começa a decorar (b, seed 0).** A acurácia de treino passa de 95% na época 5 (seeds 1 e 2:
épocas 5 e 6). A perda de val cai de 2,69 (época 1) para 2,45 (época 2) e depois fica praticamente
parada (mínimo 2,19 na época 20), enquanto a perda de treino continua caindo duas ordens de grandeza
até ~0,02. Ou seja, a partir da época ~5 o modelo já decorou as 105 imagens e o que muda depois é
ruído no val. Época de mínimo da perda de val por seed: 20, 5, 6. O F1 de val oscila em 0,33–0,42
depois disso; a época escolhida pelo F1 de val (42, 44, 7) é em boa parte sorte do val pequeno.
Fonte: `hist` em `results/piso_r5.json`.

### Professor zero-shot (prompt `a photo of a {nome}`, `_` → espaço)

Origem: `results/professor.json` e `results/professor_configs.csv` (log `logs/professor.log`; comando `python -m pa3.teacher`).
Escolha do professor **só** pela acurácia zero-shot nos 105 rótulos de `rotulos_5` (imagens do train).
Comparei também o pré-processamento: `recorte` (padrão do open_clip: lado menor 224 + center crop)
e `achatada` (224×224 sem corte), pelo mesmo critério.

| professor / pré-proc. | acurácia zs nos 105 rótulos (critério) | F1 zs nos 105 | F1 macro test_public | acurácia test_public |
|---|---|---|---|---|
| BioCLIP / recorte | 0,3333 | 0,3318 | 0,3099 | 0,3490 |
| BioCLIP / achatada | 0,3238 | 0,2966 | 0,3736 | 0,4106 |
| **CLIP ViT-B/16 OpenAI / recorte (escolhido)** | **0,4286** | 0,3670 | **0,3746** | **0,4118** |
| CLIP ViT-B/16 OpenAI / achatada | 0,4000 | 0,3584 | 0,3307 | 0,3810 |

**Linha de referência do professor zero-shot: F1 macro 0,3746, acurácia 0,4118** (CLIP ViT-B/16, recorte).

Matriz de ruído: `results/figs/professor_confusao.png`. Dia × noite: `results/figs/professor_dia_noite.png`.

**Dia × noite (flag "noturna/IR" = média de max−min entre canais < 3; 391 noturnas, 420 diurnas).**
F1 macro em cada subconjunto calculado só sobre as classes presentes nele.

| subconjunto | n | classes presentes | F1 macro | acurácia |
|---|---|---|---|---|
| dia | 420 | 16 | 0,4295 | 0,4286 |
| noite / IR | 391 | 15 | 0,2915 | 0,3939 |

Atenção: espécie e dia/noite estão fortemente confundidos no test_public (coyote, cottontail,
skunk, opossum, deer só aparecem à noite; crow, chicken, squirrels, turkey, woodchuck só de dia),
então a quebra mistura efeito de iluminação com efeito de classe.

**F1 por classe do professor zero-shot no test_public** (suporte; nº de previsões):

| classe | F1 | suporte | previsões |
|---|---|---|---|
| american_black_bear | 0,7832 | 82 | 61 |
| eastern_gray_squirrel | 0,7200 | 34 | 41 |
| white_tailed_deer | 0,6458 | 33 | 63 |
| chicken | 0,6275 | 60 | 42 |
| wild_turkey | 0,5970 | 27 | 40 |
| virginia_opossum | 0,4892 | 55 | 84 |
| coyote | 0,4884 | 31 | 55 |
| american_crow | 0,4808 | 63 | 41 |
| horse | 0,4615 | 4 | 9 |
| eastern_fox_squirrel | 0,3836 | 32 | 41 |
| bird | 0,3333 | 22 | 8 |
| eastern_cottontail | 0,3030 | 26 | 7 |
| grey_fox | 0,2529 | 39 | 48 |
| woodchuck | 0,2474 | 22 | 75 |
| eastern_chipmunk | 0,2222 | 31 | 32 |
| northern_raccoon | 0,2128 | 27 | 67 |
| red_fox | 0,2041 | 43 | 6 |
| bobcat | 0,1782 | 34 | 67 |
| striped_skunk | 0,1304 | 31 | 15 |
| dog | 0,1053 | 67 | 9 |
| domestic_cat | 0,0000 | 48 | 0 |

**Pares mais confundidos (verdadeira → prevista, n, fração da classe verdadeira):**
eastern_chipmunk → virginia_opossum 21 (68%); domestic_cat → bobcat 17 (35%); chicken → woodchuck 17 (28%);
eastern_cottontail → bobcat 15 (58%); red_fox → grey_fox 15 (35%); virginia_opossum → northern_raccoon 15 (27%);
eastern_fox_squirrel → eastern_chipmunk 13 (41%); american_crow → wild_turkey 13 (21%); dog → american_crow 12 (18%);
dog → grey_fox 11 (16%); dog → coyote 10 (15%); dog → woodchuck 10 (15%).
O par citado no enunciado aparece fraco aqui: bird → american_crow 4 (18% de bird).
Erros sistemáticos: `domestic_cat` nunca é previsto (vira bobcat/raccoon), `dog` quase nunca (9 previsões
para 67 cães, espalhados em crow/fox/coyote/woodchuck), `red_fox` noturno vira grey_fox (F1 noturno 0 em 38),
e há classes "ímã" que recebem muito mais previsões que o suporte (woodchuck 75 vs 22, bobcat 67 vs 34,
northern_raccoon 67 vs 27, virginia_opossum 84 vs 55).

**Bounding box:** o pacote não traz bounding box, então a quebra por tamanho do animal não é possível.

### Professor melhorado com os rótulos

`log p = α·log_softmax(zs) + (1−α)·log_softmax(LR)`, LR = regressão logística sobre os embeddings
(normalizados) do CLIP ViT-B/16. α ∈ {0; 0,1; …; 1}, C ∈ {0,01; 0,1; 1; 10; 100}, por CV dentro dos
rótulos com dobras "j-ésimo rótulo de cada classe" (cada rótulo de uma classe vem de uma câmera-noite
diferente). Critério: F1 macro das previsões fora da dobra; desempate menor log-loss.
Origem: `results/professor.json` (`melhorado`, inclui a tabela de CV inteira).

| orçamento | α | C | F1 CV (dentro dos rótulos) | **F1 macro test_public** | acurácia test_public | F1 dia | F1 noite |
|---|---|---|---|---|---|---|---|
| 1 | 0,5 (a priori) | 1 (a priori) | — (CV impossível) | **0,3801** | 0,4180 | 0,4310 | 0,3556 |
| 5 | 0,3 | 100 | 0,4662 | **0,5379** | 0,5771 | 0,5518 | 0,4090 |
| 20 | 0,1 | 100 | 0,6516 | **0,6646** | 0,6942 | 0,5857 | 0,6042 |

Notas: no orçamento 1 não há como validar dentro dos rótulos (segurar o único rótulo tira a classe
do treino), então α e C foram fixados a priori no código antes de rodar. Nos orçamentos 5 e 20 o C
escolhido (100) está na borda superior da grade.

### Resumo da Etapa 1 (test_public)

| método | F1 macro | acurácia |
|---|---|---|
| Piso (a) CNN do zero, seed 0 | 0,2612 | 0,3033 |
| Piso (b) ResNet-18 ImageNet, seed 0 | 0,4520 | 0,4784 |
| Piso (b), média 3 seeds | 0,4393 ± 0,0110 | 0,4595 ± 0,0180 |
| Professor zero-shot BioCLIP (recorte) | 0,3099 | 0,3490 |
| Professor zero-shot CLIP ViT-B/16 (recorte, escolhido) | 0,3746 | 0,4118 |
| Professor melhorado, orçamento 1 | 0,3801 | 0,4180 |
| Professor melhorado, orçamento 5 | 0,5379 | 0,5771 |
| Professor melhorado, orçamento 20 | 0,6646 | 0,6942 |
| Teto do monitor (todos os rótulos do train) | 0,686 | — |

### Throughput (planejamento da Etapa 2)

Origem: `results/throughput_mps.json` (log `logs/throughput.log`; `python -m pa3.train_sup throughput`).
ResNet-18, MPS (M4 Pro), batch 64, com aumentação na GPU, 30 passos medidos após 5 de aquecimento.

| resolução | img/s treino | s por época no train inteiro (6.505) | s/img inferência (MPS) |
|---|---|---|---|
| 224 | 214 | 30,3 | 0,0010 |
| 320 | 105 | 61,9 | 0,0029 |

### Cache para a Etapa 2 (fora do git)

- `cache/prof_train_ids.json`: ordem dos 6.505 ids do train (alfabética) e professor escolhido.
- `cache/prof_zs_train_logits.npy`, `cache/prof_zs_train_probs.npy`: zero-shot (CLIP ViT-B/16, recorte), (6505, 21).
- `cache/prof_melhorado_r{1,5,20}_train_probs.npy` e `..._logprobs.npy`: professor melhorado por orçamento.
- `cache/prof_{bioclip,clip_b16}_{recorte,achatada}_{train,test_public}_img.npy`: embeddings de imagem.
- `cache/imgs_{train,val,test_public}_{224,320}.npy`: imagens pré-carregadas (uint8).
