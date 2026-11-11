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

## Etapa 2 — Aluno, tratamento de ruído e curva de orçamento

Aluno = ResNet-18 ImageNet (11.187.285 parâmetros), 224 px achatada, treinada no train com os
pseudo-rótulos do professor (CLIP ViT-B/16, cache da Etapa 1) + CE nos rótulos verdadeiros do
orçamento. Código: `pa3/train_student.py`; análise e figuras: `pa3/analise_aluno.py`.

**Hiperparâmetros fixos a priori** (não ajustados): AdamW lr 3e-4, wd 0,05, cosine, 20 épocas
(época = uma passada pelo pool não rotulado), por passo 64 imagens do pool + 16 rotuladas (ciclando
o conjunto rotulado) num único forward, perda = CE_suave(pool) + 1,0·CE(rotuladas); mesma aumentação
do piso. As imagens do orçamento entram só com o rótulo verdadeiro. Época escolhida pelo maior F1
macro no val (desempate: menor perda de val). Temperatura das variantes suaves escolhida sem val:
o T da grade {0,5; 0,75; 1; 1,5; 2; 3; 4} que minimiza a log-loss do professor nos rótulos do
orçamento (zero-shot: T = 1,5; melhorado com 5 rótulos, previsões fora da dobra: T = 0,75).

Tempo: `results/aluno_tempo.json` (log `logs/aluno_tempo.log`): ~0,36 s por passo, ~36 s por época
com o pool inteiro (100 passos). Total da Etapa 2 ≈ 1 h 35 min de GPU (soma de `segundos` nos JSON).

### Ablação do tratamento de ruído (orçamento 5, seed 0)

Origem: `results/aluno_ablacao.json` (log `logs/aluno_ablacao.log`; `python -m pa3.train_student ablacao`).
Figura: `results/figs/ablacao_ruido.png`. As métricas de teste foram calculadas pelo script só
depois de τ e variante estarem escolhidos no val.

| variante | descrição | T | pool usado (cobertura) | época (val) | **F1 val** | F1 test_public | acc test_public |
|---|---|---|---|---|---|---|---|
| A | duro, zero-shot, sem tratamento | — | 6.400 (100%) | 1 | 0,4208 | 0,4470 | 0,4883 |
| B τ=0,3 | duro, zero-shot, max p ≥ 0,3 | — | 3.814 (59,6%) | 7 | 0,4120 | 0,4306 | 0,4920 |
| **B τ=0,5 (escolhida)** | duro, zero-shot, max p ≥ 0,5 | — | 2.225 (34,8%) | 18 | **0,4744** | 0,4699 | 0,5512 |
| B τ=0,7 | duro, zero-shot, max p ≥ 0,7 | — | 1.458 (22,8%) | 11 | 0,4445 | 0,5141 | 0,6091 |
| C | suave, zero-shot | 1,5 | 6.400 (100%) | 11 | 0,4332 | 0,3810 | 0,4353 |
| D | suave, professor melhorado (5) | 0,75 | 6.400 (100%) | 9 | 0,4432 | 0,5428 | 0,5857 |
| E | D + amostragem balanceada pela classe pseudo | 0,75 | 6.400 (100%) | 4 | 0,4734 | 0,5059 | 0,5573 |

Referências no test_public: piso (b) seed 0 0,4520; professor zero-shot 0,3746; professor melhorado (5) 0,5379.

**Escolha (só no val): B com τ = 0,5** (τ escolhido entre 0,3/0,5/0,7 pelo F1 de val; depois B
venceu E por 0,0010 de F1 de val, 0,4744 vs 0,4734, ou seja, 0,2 imagem de val em termos de
acerto). **Atenção, registrado como está:** no test_public a ordem é outra: D (0,543) > B τ=0,7
(0,514) > E (0,506) > B τ=0,5 (0,470). Com 202 imagens de val as diferenças entre B, D e E estão
dentro do ruído do val; a regra é escolher no val e foi seguida. As variantes D e E tinham a menor
perda de val (1,80 e 1,77 contra 1,89 de B), mas o critério declarado era F1 macro. C (suave com o
zero-shot) ficou **abaixo do piso** no teste (0,381), e A também (0,447 < 0,452). A (duro, sem
tratamento) atinge o melhor F1 de val na época 1 e depois cai para ~0,30: o aluno aprende o ruído
do professor (o F1 de val por época está em `hist` no JSON).

Nota: o filtro muda o tamanho do pool e portanto o número de passos (B τ=0,5 faz 35 passos por
época contra 100 em A/C/D/E). Isso não foi controlado.

### Parte 2 — Curva de orçamento

Origem: `curva_orcamento.csv` (raiz), gerado por `python -m pa3.analise_aluno` a partir de
`results/piso_b_r{1,5,20}.json` (logs `logs/piso_b_r*.log`; `python -m pa3.train_sup piso_b --orcamento K --seeds ...`),
`results/aluno_metodo.json` (log `logs/aluno_metodo.log`; `python -m pa3.train_student metodo`) e
`results/professor.json`. Figura: `results/figs/curva_orcamento.png`.

| método | 1 rótulo/classe | 5 rótulos/classe (3 seeds, média ± desvio) | 20 rótulos/classe |
|---|---|---|---|
| Piso (b) ResNet-18 | 0,2354 | 0,4393 ± 0,0110 (0,4520 / 0,4252 / 0,4408) | 0,6247 |
| **Aluno (variante B, τ=0,5)** | **0,4167** | **0,4801 ± 0,0173** (0,4699 / 0,4660 / 0,5045) | **0,5719** |
| Professor zero-shot (não usa rótulos) | 0,3746 | 0,3746 | 0,3746 |
| Professor melhorado | 0,3801 | 0,5379 | 0,6646 |
| Teto do monitor | — | 0,686 | — |

Acurácia (mesma ordem): piso 0,2503 / 0,4595 ± 0,0180 / 0,6683; aluno 0,5228 / 0,5602 ± 0,0128 / 0,6079.

Desvio populacional (ddof = 0) sobre 3 seeds. O piso do orçamento 5 foi refeito em modo `piso_b`
(para guardar os logits de cada seed) e reproduziu exatamente os F1 de val e de teste da Etapa 1
(`reprodutibilidade_piso_r5` em `results/aluno_analise.json`).

Leitura: o aluno ganha muito do piso com 1 rótulo (+0,18), ganha +0,04 com 5 e **perde com 20
(0,572 vs 0,625)**. Com o filtro fixo τ = 0,5 do zero-shot, o pool é o mesmo em todos os orçamentos
(~35% do train) e os pseudo-rótulos não melhoram com mais rótulos; com 20 rótulos/classe o piso já
supera o zero-shot e o ruído do professor passa a atrapalhar. O aluno também fica abaixo do
professor melhorado com 5 e 20 rótulos. Escolhas de época e seed por orçamento: `aluno_metodo.json`
(épocas 18, 8, 3 para seeds 0, 1, 2 do orçamento 5; 13 no orçamento 1; 9 no 20).

### Quebra por classe no orçamento 5 (média das 3 seeds)

Origem: `results/por_classe_r5.csv` e `results/aluno_analise.json` (`por_classe_r5`, `evidencia`).
Figuras: `results/figs/ganho_por_classe.png`, `results/figs/aluno_vs_professor.png`,
`results/figs/aluno_dia_noite.png`. F1 macro: piso 0,4393 → aluno 0,4801 (média das F1 por classe).

| classe (suporte) | F1 piso | F1 aluno | Δ aluno − piso | F1 prof. zs | F1 prof. melhorado (5) |
|---|---|---|---|---|---|
| american_crow (63) | 0,2858 | 0,7966 | +0,5108 | 0,4808 | 0,6604 |
| american_black_bear (82) | 0,4836 | 0,8199 | +0,3362 | 0,7832 | 0,7368 |
| wild_turkey (27) | 0,2473 | 0,5698 | +0,3226 | 0,5970 | 0,6471 |
| woodchuck (22) | 0,2300 | 0,4739 | +0,2439 | 0,2474 | 0,5532 |
| chicken (60) | 0,5923 | 0,7862 | +0,1940 | 0,6275 | 0,7385 |
| coyote (31) | 0,4383 | 0,5916 | +0,1533 | 0,4884 | 0,5517 |
| eastern_cottontail (26) | 0,0589 | 0,1609 | +0,1020 | 0,3030 | 0,6944 |
| eastern_gray_squirrel (34) | 0,7970 | 0,8954 | +0,0984 | 0,7200 | 0,5926 |
| virginia_opossum (55) | 0,8160 | 0,8559 | +0,0399 | 0,4892 | 0,6800 |
| white_tailed_deer (33) | 0,6208 | 0,6468 | +0,0259 | 0,6458 | 0,7606 |
| northern_raccoon (27) | 0,2680 | 0,2890 | +0,0210 | 0,2128 | 0,5205 |
| bobcat (34) | 0,1537 | 0,1532 | −0,0006 | 0,1782 | 0,1500 |
| grey_fox (39) | 0,3252 | 0,3236 | −0,0016 | 0,2529 | 0,1579 |
| bird (22) | 0,1187 | 0,0918 | −0,0269 | 0,3333 | 0,3030 |
| dog (67) | 0,6355 | 0,5844 | −0,0511 | 0,1053 | 0,6500 |
| red_fox (43) | 0,3686 | 0,2712 | −0,0974 | 0,2041 | 0,3030 |
| eastern_fox_squirrel (32) | 0,4008 | 0,3003 | −0,1005 | 0,3836 | 0,5882 |
| eastern_chipmunk (31) | 0,9513 | 0,7641 | −0,1872 | 0,2222 | 0,9667 |
| horse (4) | 0,8222 | 0,5886 | −0,2336 | 0,4615 | 0,4286 |
| domestic_cat (48) | 0,2380 | 0,0000 | −0,2380 | 0,0000 | 0,1639 |
| striped_skunk (31) | 0,3737 | 0,1199 | −0,2538 | 0,1304 | 0,4494 |

**Ganharam mais:** american_crow, american_black_bear, wild_turkey, woodchuck, chicken, coyote.
**Não ganharam (Δ ≤ 0):** bobcat e grey_fox (≈ 0), e perderam bird, dog, red_fox,
eastern_fox_squirrel, eastern_chipmunk, horse, domestic_cat (F1 0 nas 3 seeds) e striped_skunk.

**Onde o aluno supera o professor.** Acima do zero-shot em 14 de 21 classes (white_tailed_deer por +0,001; domestic_cat empata em 0); acima dos dois
professores em 8: eastern_gray_squirrel, virginia_opossum, american_black_bear, american_crow,
chicken, coyote, horse, grey_fox. Maiores vantagens sobre o zero-shot: eastern_chipmunk (+0,54),
dog (+0,48), virginia_opossum (+0,37), american_crow (+0,32), woodchuck (+0,23).

**Por quê (evidência nos dados, `evidencia` em `results/aluno_analise.json`):**

1. *O filtro deixa passar pseudo-rótulos certos, e eles se concentram em poucas classes.* Rodando o
   zero-shot no test_public como se não tivesse rótulo (só diagnóstico; τ foi escolhido no val),
   31,7% das imagens passam em max p ≥ 0,5, com acurácia 0,747 nas que passam contra 0,256 nas que
   não passam (dia 0,769, noite 0,729). A precisão das previsões que passam é alta justamente nas
   classes em que o aluno mais ganha: american_black_bear 0,98 (49 previsões que passam),
   virginia_opossum 0,96 (28), american_crow 0,95 (20), wild_turkey 0,91 (22), white_tailed_deer
   0,87 (30). Nenhuma previsão de bird, dog, domestic_cat (nunca previsto), eastern_cottontail ou
   eastern_gray_squirrel passa no filtro: nessas classes o aluno só tem os 5 rótulos verdadeiros e
   ainda vê o pool cheio de outras classes, por isso não melhora (ou piora) em relação ao piso.
2. *O aluno corrige os erros do professor de baixa confiança.* O erro sistemático do zero-shot
   chipmunk → opossum (21 de 31) é de baixa confiança: só 3 previsões de chipmunk passam (todas
   erradas), mas 28 de opossum passam com precisão 0,96. O aluno aprende "opossum" com pseudo-rótulos
   limpos e chipmunk com os rótulos verdadeiros: F1 opossum 0,49 → 0,86 e chipmunk 0,22 → 0,76 contra
   o zero-shot. Quando aluno e zero-shot discordam (484 de 811 imagens, média das seeds), só o aluno
   acerta em 194 e só o professor em 74 (dia 117 vs 37; noite 77 vs 37).
3. *Imagens noturnas: ganho em acurácia, não em F1 macro.* Acurácia noturna: aluno 0,496, piso
   0,411, zero-shot 0,394, melhorado 0,486. F1 macro noturno: aluno 0,328 (0,287 / 0,318 / 0,378 por
   seed), piso 0,365, zero-shot 0,292, melhorado 0,409. O aluno aprende bem a aparência IR das classes
   noturnas frequentes e com pseudo-rótulo confiável (recall noturno: urso 0,81 vs piso 0,19 e
   zero-shot 0,65; opossum 0,97 vs 0,78 / 0,62; coyote 0,98 vs 0,68 / 0,68; veado 0,99 vs 0,77 / 0,94),
   mas perde as noturnas raras ou sem pseudo-rótulo (skunk 0,11 vs piso 0,59; domestic_cat 0 vs 0,24;
   red_fox 0,12 vs 0,33). A hipótese "o aluno aprende IR e por isso supera o professor à noite" só
   vale para as classes que o filtro alimenta.
4. *Classes ímã herdadas.* O aluno prevê coyote 77 vezes (suporte 31), white_tailed_deer 69 (33),
   wild_turkey 63 (27), virginia_opossum 71 (55) — média das seeds. O professor tinha os ímãs
   woodchuck (75), bobcat (67), raccoon (67), que o filtro removeu (o aluno prevê woodchuck 24,
   raccoon 23), mas os ímãs das classes com pseudo-rótulo abundante aparecem no aluno.

### Modelo entregue

Origem: `results/aluno_entregue.json` (log `logs/aluno_entrega.log`; `python -m pa3.train_student entrega`).
Seed escolhida pelo maior F1 de val entre as 3 seeds do orçamento 5: seed 0 (F1 val 0,4744 contra
0,4742 da seed 1 e 0,4565 da seed 2 — praticamente empate entre 0 e 1).

| arquivo | parâmetros (`sum(p.numel())`) | origem | F1 val | F1 macro test_public (recalculado em CPU) | acurácia |
|---|---|---|---|---|---|
| `weights/aluno_final.pt` (44,8 MB) | **11.187.285** (≤ 12 M) | `abl_B_tau0.5_r5_s0` | 0,4744 | 0,4699 | 0,5512 |

O pacote guarda `state_dict`, `classes`, `resolucao` (224), `redimensionamento` (achatada, bicúbica),
`media`/`desvio` (ImageNet) e `arquitetura`. O F1 recalculado em CPU bate com o registrado no treino.
Observação: a seed entregue (0,470) é a pior das três no test_public (0,470 / 0,466 / 0,505 →
média 0,480); como a escolha é no val, isso fica assim.

### Conferências feitas pelo script de análise

- Previsões do professor no test_public recalculadas do cache batem com `professor.json`
  (zs 0,3746; melhorado 0,3801 / 0,5379 / 0,6646).
- F1 recalculado dos logits salvos (`results/preds/`) bate com o registrado para as 3 seeds do aluno e do piso.
