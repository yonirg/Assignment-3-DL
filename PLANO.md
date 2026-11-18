# PLANO — PA3 "Poucos rótulos" (Trilha C: professor e aluno)

Trabalho conduzido por mim (escolhas e critérios em `AI_LOG.md`); plano detalhado pelo Claude Fable 5.2 e implementação por agentes Claude Opus 5.5.
Enunciado: `~/Downloads/PA3.pdf` (texto extraído em `docs/enunciado.txt` na Etapa 1).

## Decisões fechadas

| Tema | Decisão | Por quê |
|---|---|---|
| Trilha | **C — professor e aluno** | Mais simples de explicar (um modelo grande ensina um pequeno) e mais visual (matriz de ruído, professor × aluno por classe). Mais barata no M4: o professor só faz inferência uma vez. |
| Backbone do aluno | **ResNet-18 ImageNet** (torchvision, ~11,2 M parâmetros < 12 M) | Mesmo backbone do teto do monitor (0,686), comparação direta; Grad-CAM trivial na `layer4`. |
| Professor | **BioCLIP** (`hf-hub:imageomics/bioclip`, via `open_clip`), com CLIP ViT-B/16 OpenAI como comparação | Pesos abertos, rodam local. Escolha entre professores feita **só** com os 105 rótulos do train (acurácia zero-shot neles), nunca com val nem test_public. |
| Hardware | M4 Pro, 24 GB, `device="mps"` no treino; `predict.py` em CPU | |
| Ambiente | `uv venv --python 3.12 .venv` + `requirements.txt` | Python 3.14 do sistema não tem torch. |
| Métrica | F1 macro (principal) + acurácia, sempre no `test_public`; seleção de época/hiperparâmetro **só no val** | Regra do enunciado. |

## Regras que não podem ser quebradas (conferir a cada script)

1. `test_public` só para reportar. Nenhuma escolha (época, limiar, temperatura, arquitetura, professor) olha para ele.
2. O professor nunca vê o `val`. Pode ver `train` e `test_public` (este só para medir).
3. Hiperparâmetros do professor melhorado: validação cruzada **dentro** dos rótulos do orçamento.
4. Nenhum dado externo além de `pa3_dados.zip`; nenhum rótulo manual; nenhuma API de LLM para rotular.
5. Modelo entregue ≤ 12 M parâmetros, sem professor dentro, roda offline em CPU em < 10 min para ~800 imagens.
6. Números reportados são os que saíram dos scripts. Se algo falhar ou der ruim, registrar como está.

## Estrutura do repositório

```
pa3/            pacote: data.py, models.py, train_sup.py, teacher.py, train_student.py, metrics.py, cam.py, plots.py
scripts/        run_all.sh (treina tudo), eval.sh (avalia o modelo entregue)
predict.py      entregável
weights/        aluno_final.pt (modelo entregue)
results/        csv + json de cada experimento; figs/ com os PNG (galeria/ = um PNG por caso da Parte 3)
cache/          embeddings e logits do professor (.npy) — fora do git
data/           pa3_dados descompactado — fora do git
docs/           enunciado.txt
apresentacao/   PA3_apresentacao.pptx (com as falas nas notas) + gera_apresentacao.py (gera o deck)
curva_orcamento.csv, README.md, AI_LOG.md, RESULTADOS.md
```

`RESULTADOS.md` é a casa única dos números: cada etapa acrescenta a sua seção (tabela + caminho do arquivo de origem). A apresentação e o README citam de lá.

## Etapa 1 — Ambiente, dados, piso e professor

1. `git init`, `.gitignore` (`data/`, `cache/`, `.venv/`, `__pycache__/`), venv, `requirements.txt` (torch, torchvision, timm, open_clip_torch, scikit-learn, pandas, matplotlib, pillow, gdown, python-pptx).
2. Baixar `pa3_dados.zip` com `gdown` (id `1g7FGqtXOk_ziNKBwQAk8BxjPOaPCnbui`) para `data/`, descompactar, **inspecionar** (pastas, colunas dos CSV, `classes.csv`, formato do `image_id`) e documentar no README.
3. `pa3/data.py`: datasets para train (sem rótulo), orçamentos 1/5/20, val, test_public. Pré-carregar as imagens redimensionadas em memória (uint8) para o treino não ficar preso em decodificação de JPEG. Flag barata "noturna/IR" (saturação média ≈ 0) calculada por imagem — usada na análise do professor.
4. **Parte 0 (piso)** com `rotulos_5.csv`:
   - (a) CNN pequena do zero (4 blocos conv-BN-ReLU-pool + GAP + linear);
   - (b) ResNet-18 ImageNet com fine-tuning (aumentação: RandomResizedCrop com scale ≥ 0,5, flip, color jitter leve; AdamW; cosine).
   - Resolução: comparar 224 e 320 no val em (b) e fixar para o resto do PA.
   - Saídas: F1 macro + acurácia no test_public para (a) e (b); matriz de confusão de (b); curva treino × val por época de (b), com a época em que a perda de treino segue caindo e a de val para/sobe marcada (resposta a "quando começa a decorar").
5. **Professor zero-shot**:
   - embeddings de imagem de train, test_public (e dos rótulos do orçamento, que são subconjunto do train) em cache;
   - prompt obrigatório: nome da classe (`a photo of a {nome}`, com `_` → espaço). Reportar F1 zero-shot no test_public = linha de referência;
   - matriz de confusão do professor no test_public (= matriz de ruído), F1 por classe, quebra dia × noite, top pares confundidos (ex.: bird × american_crow). Se o pacote trouxer bounding box, quebrar também por tamanho do animal; se não trouxer, dizer que não dá e não inventar.
6. **Professor melhorado com os rótulos** (uma das formas de tratar ruído, permitida pelo enunciado): regressão logística sobre os embeddings do professor com os rótulos do orçamento, combinada com os logits zero-shot (`p = softmax(α·zs + (1-α)·lr)`); `α` e `C` por validação cruzada leave-one-group-out dentro dos rótulos. Reportar F1 no test_public para orçamentos 1, 5, 20.
7. Salvar em `cache/` as probabilidades do professor (zero-shot e melhorado, por orçamento) para todo o train.

**Pronto quando:** `RESULTADOS.md` tem as tabelas do piso e do professor; figuras em `results/figs/` (`piso_curva.png`, `piso_confusao.png`, `professor_confusao.png`, `professor_dia_noite.png`); commit local feito.

## Etapa 2 — Aluno, tratamento de ruído e curva de orçamento

1. `pa3/train_student.py`: ResNet-18 ImageNet treinada em `train` com pseudo-rótulos do professor + CE nos rótulos verdadeiros do orçamento (as imagens rotuladas usam o rótulo verdadeiro, não o pseudo). Seleção de época no val.
2. **Ablação do tratamento de ruído** (orçamento 5, seed 0), tabela + gráfico de barras:
   - A. pseudo-rótulo duro do zero-shot, sem tratamento;
   - B. A + filtro por confiança (limiar escolhido no val entre 3–4 valores);
   - C. rótulos suaves com temperatura (KL para a distribuição do professor);
   - D. rótulos suaves do professor **melhorado**;
   - E. D + amostragem balanceada pela classe pseudo-rotulada (corrige urso × cavalo).
   O melhor **no val** vira "o método".
3. **Parte 2**: piso (b) e método nos orçamentos 1, 5 (seeds 0, 1, 2), 20. Escrever `curva_orcamento.csv` (colunas: `metodo,orcamento,seed,f1_macro,acuracia`), incluindo linhas do professor zero-shot e melhorado.
4. Gráfico obrigatório `curva_orcamento.png`: F1 macro × nº de rótulos por classe (x em log), curvas piso / aluno / professor zero-shot (constante) / professor melhorado, linha tracejada do teto 0,686, barras de erro no orçamento 5.
5. Quebra por classe no orçamento 5 (média das 3 seeds): barras de ΔF1 (aluno − piso) ordenadas; e aluno × professor por classe, destacando onde o aluno supera o professor. Explicar o porquê olhando os dados (ex.: aluno treina em 6.505 imagens do domínio e aprende a aparência IR/noturna que o professor não tem; fine-tuning corrige prompt ruim de classe genérica).
6. Modelo entregue = melhor seed no **val** do orçamento 5 → `weights/aluno_final.pt` (state_dict + lista de classes + resolução).

Orçamento de tempo: medir uma época antes de lançar tudo e ajustar épocas/resolução para o total de treino ficar em ~2–3 h. Treinos longos em background com log em `logs/`.

**Pronto quando:** `curva_orcamento.csv` completo, figuras `ablacao_ruido.png`, `curva_orcamento.png`, `ganho_por_classe.png`, `aluno_vs_professor.png`, seção nova em `RESULTADOS.md`, commit local.

## Etapa 3 — Galeria, entregáveis e apresentação

1. **Parte 3**: Grad-CAM (`layer4`) implementado em `pa3/cam.py`. 3 erros + 5 acertos do modelo final no test_public: imagem, classe verdadeira, prevista, confiança, mapa sobreposto. Diagnóstico concreto de cada um, escrito **depois de olhar a figura** (tamanho aproximado do animal em pixels a 224/320, dia/noite, onde o mapa acende). Responder: acertou olhando o animal ou o fundo?
2. `predict.py --imagens PASTA --saida preds.csv`: CPU, sem internet (`weights=None` + state_dict local), imprime nº de parâmetros, CSV `image_id,label` com nomes de `classes.csv`. Testar no test_public: conferir que o F1 bate com o reportado e cronometrar.
3. `README.md` (ambiente, um comando que treina = `scripts/run_all.sh`, um que avalia = `scripts/eval.sh`), `AI_LOG.md` (como a IA foi usada: planejamento Fable 5.2, execução Opus 5.5, episódios reais anotados durante as etapas — bugs, decisões revertidas, números que surpreenderam — e o que preciso saber explicar).
4. **Apresentação** `apresentacao/PA3_apresentacao.pptx` (entrega extra, com a fala de cada slide nas notas do apresentador, em português, ~12–15 slides): problema → dados e armadilhas → piso e quando decora → ideia professor/aluno em um diagrama → professor zero-shot e matriz de ruído → onde o professor erra → tratamento de ruído (ablação) → curva de orçamento → ganho por classe / onde o aluno supera o professor → galeria de erros → galeria de acertos → pergunta da capivara (sinais de fora-da-distribuição: confiança/entropia baixa do aluno, discordância aluno × professor, distância no espaço de embeddings, professor com vocabulário aberto) → limitações → resumo. Todos os números vêm de `RESULTADOS.md`. Referência de estilo: `~/repos/PA2 - DL/PA2_apresentacao.pptx`.

**Pronto quando:** `predict.py` roda limpo num shell novo, deck abre e foi conferido visualmente, commit local.

## Verificação final (sessão principal)

Rodar `predict.py` no test_public e recalcular F1; contar parâmetros; conferir que nenhum script lê test_public para seleção (grep) e que o professor não lê val; abrir as figuras e o deck.
