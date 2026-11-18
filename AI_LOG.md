# AI_LOG — como usei IA no PA3

O enunciado permite e espera o uso de IA, desde que eu entenda o que entrego. Este arquivo
descreve o que eu decidi, como a IA foi usada, os episódios reais que aconteceram (anotados durante a execução, não
reconstruídos depois) e, no fim, o que preciso saber explicar sem ajuda.

## Meu papel e o da IA

Conduzi o trabalho e usei a IA como ferramenta de planejamento detalhado e de implementação. O que
foi decisão minha:

- **Forma de trabalhar.** Separar planejamento e execução: um modelo escreve o plano a partir do
  enunciado, outro implementa, com conferência entre as etapas.
- **Trilha.** Defini o critério: a trilha mais simples de explicar e com mais material visual para a
  apresentação. Por esse critério ficou a Trilha C (professor e aluno), que a IA tinha recomendado
  entre as três.
- **Ambiente.** Treinar na GPU integrada do meu Mac, onde fica o repositório e a autorização para
  baixar o pacote de dados e os pesos abertos do professor.
- **Entregas.** Pedi a apresentação em `.pptx` com a fala de cada slide, além dos entregáveis
  obrigatórios, e decidi publicar o repositório no mesmo formato dos PAs anteriores.
- **Resultados como saíram.** A IA apontou que uma segunda rodada (pseudo-rótulos do professor
  melhorado, limiar por orçamento) subiria o número, mas seria uma mudança motivada pelo test_public
  já visto. Publiquei sem essa rodada.

O que a IA fez, sob essa direção:

- **Planejamento detalhado: Claude Fable 5.2.** A partir do PDF do enunciado e das minhas escolhas, o
  plano (`PLANO.md`) detalhou as decisões técnicas antes de qualquer código: ResNet-18 ImageNet como
  aluno, CLIP/BioCLIP como candidatos a professor, as regras duras (nada escolhido no test_public,
  professor nunca vê o val, ≤ 12 M parâmetros), a ordem das etapas e o critério de "pronto" de cada
  uma.
- **Execução: agentes Claude Opus 5.5 no Claude Code**, em três etapas, cada uma num agente com o
  plano como briefing: (1) ambiente, dados, piso e professor; (2) aluno, ablação do ruído e curva de
  orçamento; (3) galeria da Parte 3, `predict.py`, README, este log e a apresentação. Os agentes
  escreveram o código em `pa3/`, rodaram os treinos no Mac M4 Pro (MPS) e registraram os números em
  `RESULTADOS.md` copiando dos arquivos que os scripts geram.
- **Verificação entre etapas.** Entre uma etapa e outra, a sessão principal conferiu o que foi
  entregue (números contra os JSON, regras do plano, commits) antes de liberar a próxima. Cada etapa
  terminou com commit local.
- **O que a IA não fez.** Nenhuma API de LLM classificou ou rotulou imagens. Na Etapa 3 o agente
  *olhou* as figuras da galeria para escrever o diagnóstico (dia/noite, tamanho do animal, onde o mapa
  acende) e desenhou caixas aproximadas em volta dos animais para o teste de "tapar o animal"; isso é
  diagnóstico de 8 imagens do test_public, não vira rótulo, previsão nem escolha de modelo.

## Episódios

### Etapa 1 — ambiente, dados, piso e professor

- **Inspeção antes de codar.** O `LEIAME.md` do pacote diz que todos os CSVs têm `image_id,label` e
  que `image_id` é o nome do arquivo sem `.jpg` (hash de 16 caracteres). Confirmado por script:
  21/105/420 ids nos orçamentos, aninhados (`rotulos_1 ⊂ rotulos_5 ⊂ rotulos_20`), todos em
  `images/train`. O pacote **não traz bounding box** nem câmera/horário, então a quebra do professor
  por tamanho do animal não foi feita e não foi inventada.
- **Proporções variadas → achatar em vez de recortar.** As imagens têm 512 de largura e 270–370 de
  altura; um center crop quadrado jogaria fora até ~45% da largura (e o animal pode estar na borda).
  Decisão: redimensionar para `res × res` sem corte, igual em treino e avaliação.
- **Flag de noite saiu limpa.** A média de (max − min) entre canais RGB no test_public é bimodal: 391
  imagens com valor < 1 e o resto > 8. Limiar 3 sem ambiguidade. (Na Etapa 3 vimos que a flag mede
  "sem cor", não "noite": uma das imagens da galeria é IR diurno. As figuras agora dizem "sem cor".)
- **Aumentação na GPU.** Imagens pré-carregadas em uint8 e RandomResizedCrop em lote na GPU com
  `roi_align` (testado em MPS antes de usar), para o treino não ficar preso em decodificação de JPEG.
- **Orçamento 1 não admite CV dentro dos rótulos** (segurar o único rótulo tira a classe do treino).
  α = 0,5 e C = 1 fixados a priori no código antes de rodar; para 5 e 20, CV por dobras "j-ésimo
  rótulo de cada classe" (cada rótulo vem de uma câmera-noite diferente).
- **Bug encontrado: F1 dia × noite subestimado.** Na primeira rodada, o F1 macro de cada subconjunto
  usava as 21 classes, e classes ausentes no subconjunto entravam como F1 = 0. Saída antiga: dia 0,327
  / noite 0,208. Corrigido em `metrics.resumo(..., so_presentes=True)`: dia 0,430 (16 classes) /
  noite 0,292 (15). Log antigo guardado em `logs/professor_v1_bug_dia_noite.log`.
- **Número que surpreendeu: o piso supera o professor zero-shot** (0,452 na seed 0 contra 0,375). E o
  BioCLIP, treinado em espécies, perdeu para o CLIP genérico nos 105 rótulos (acerto 0,33 vs 0,43).
- **Critério e teste discordaram no pré-processamento.** Nos 105 rótulos o recorte ganhou do achatado
  para os dois professores; no test_public o BioCLIP achatado (0,374) ficaria bem acima do recortado
  (0,310). Segui a regra. Exemplo concreto de por que não se escolhe olhando o teste.
- **Professor melhorado com 20 rótulos chega a 0,665**, perto do teto (0,686). O C escolhido (100)
  ficou na borda da grade; anotado.
- **Erros sistemáticos do professor:** `domestic_cat` nunca previsto, `dog` 9 previsões para 67,
  `red_fox` noturno → `grey_fox`, chipmunk → opossum (21/31). O par do enunciado bird × crow é fraco
  aqui (4 de 22).
- **Resolução 224 vs 320 empatou** no val (0,402 vs 0,401); fiquei com 224, ~2× mais rápida.
- **Reprodutibilidade:** os treinos do modo `piso` reproduziram exatamente o F1 de val das mesmas
  seeds do modo `resolucao`.
- **"Quando decora" não tem resposta nítida pela perda de val**: a acurácia de treino passa de 95% na
  época 5, mas a perda de val fica quase plana; a época escolhida no val variou muito (42, 44, 7).
- **A CNN do zero nem decora em 150 épocas** (acurácia de treino 0,87–0,91 com aumentação).

### Etapa 2 — aluno, ruído e curva de orçamento

- **Temperatura sem val e sem teste.** O T dos rótulos suaves foi escolhido pela calibração do
  professor nos próprios rótulos do orçamento (menor log-loss): T = 1,5 para o zero-shot; T = 0,75
  para o melhorado (previsões fora da dobra; ele estava subconfiante).
- **Teste de mesa antes de lançar.** As previsões do professor no test_public foram recalculadas do
  cache e bateram com `professor.json` até a 4ª casa.
- **Medida de tempo antes de lançar:** ~0,36 s/passo → 20 épocas por treino, 11 treinos, ~1 h 35 min
  de GPU no total.
- **O val escolheu uma variante que não é a melhor no teste.** B (τ = 0,5) ganhou de E por 0,0010 no
  val (0,4744 vs 0,4734); no test_public a melhor seria D (0,543) e B ficou em 0,470. Mantive B. O
  script só calcula o F1 de teste da ablação depois de gravar a escolha.
- **A variante sem tratamento aprende o ruído:** A tem o melhor F1 de val na época 1 (0,42) e cai para
  ~0,30 depois. C (suave do zero-shot) ficou abaixo do piso no teste (0,381).
- **Resultado ruim registrado: com 20 rótulos o aluno perde do piso** (0,572 vs 0,625): τ fixo, pool
  filtrado praticamente igual em todos os orçamentos.
- **"O aluno aprende IR" só se sustenta em parte.** Acurácia noturna sobe (0,496 vs 0,411), F1 macro
  noturno cai (0,328 vs 0,365): ganham as noturnas frequentes com pseudo-rótulo bom, perdem as raras.
- **domestic_cat herdado do professor:** o aluno também nunca prevê gato (0 nas 3 seeds).
- **Reprodutibilidade:** o piso refeito no modo `piso_b` reproduziu exatamente a Etapa 1; o modelo
  entregue recarregado em CPU dá o mesmo F1 (0,4699) que no treino em MPS.
- **Detalhe de engenharia:** a primeira versão do `modo_piso` sobrescreveria as figuras do piso ao
  rodar os orçamentos 1 e 20; separado em `piso_b` antes de rodar.
- **Variante F (aluno re-rotulando) não foi feita:** os números de teste da ablação já tinham sido
  vistos, e acrescentar variante depois abriria espaço para escolha contaminada pelo teste.

### Etapa 3 — galeria, entregáveis e apresentação

- **Regra de seleção antes de olhar.** A regra dos casos da galeria foi escrita no código antes de
  gerar as figuras: erros de maior confiança com classes distintas; acertos sorteados (semente 0),
  3 coloridos + 2 sem cor, classes distintas — sorteio para não escolher só os acertos fáceis.
- **O Grad-CAM enganou, e isso mudou o diagnóstico.** Olhando as figuras, o mapa acendia sobre o gato
  (erro gato → lince) e sobre a galinha (acerto), o que sugeria "o modelo olhou o animal". Para não
  depender só do mapa (7×7, cada célula cobre 32×32 px), acrescentei oclusão e o teste de tapar o
  animal com uma caixa cinza. Resultado: tapar o gato mantém 0,92 de "lince", tapar a galinha mantém
  0,91, a tâmia 0,93, o opossum 0,83. A resposta à pergunta "animal ou fundo" virou "majoritariamente
  fundo", o contrário do que o Grad-CAM sozinho diria.
- **A flag "noite" é "sem cor".** O erro cão → galinha foi marcado "noite/IR", mas a foto é diurna em
  IR (céu claro). As legendas passaram a dizer "sem cor (noite/IR)".
- **`predict.py` testado offline de verdade.** Além de rodar num shell novo com `HF_HUB_OFFLINE=1`,
  rodei com `socket` bloqueado (qualquer conexão levanta erro): o CSV saiu idêntico byte a byte.
  F1 recalculado do CSV: 0,4699; 811 imagens em ~14 s de CPU.
- **Bug do python-pptx que impedia o deck de abrir no Quick Look.** Sem LibreOffice na máquina, a
  renderização para conferência foi pelo Quick Look do macOS — que se recusava a abrir o deck. Por
  bisseção (deck mínimo com e sem notas), achei a causa: o python-pptx cria o `notesMaster` mas não
  escreve `<p:notesMasterIdLst>` em `presentation.xml`. Acrescentar o elemento resolveu
  (`apresentacao/gera_apresentacao.py`). O renderizador também misturava imagens entre slides no deck
  inteiro, então cada slide foi renderizado isoladamente e conferido.
- **Restrições respeitadas na Etapa 3:** nada foi retreinado; o modelo entregue é o mesmo; a galeria
  usa o test_public só como diagnóstico.

## O que preciso saber explicar

1. **Por que câmera trap é difícil** e o que isso faz com os métodos do laboratório: animal pequeno,
   noite/IR sem cor, fundo fixo por câmera (atalho), desbalanceamento. Por que achatei a imagem em
   vez de recortar.
2. **Regras de seleção.** Por que o val (202 imagens) é o único lugar de escolha, por que o test_public
   não pode ser usado nem para escolher pré-processamento do professor, e por que o professor não pode
   ver o val. O episódio B × D: a escolha no val não foi a melhor no teste, e mantive.
3. **F1 macro vs acurácia.** Cavalo (4 imagens no teste) pesa igual a urso (82). Por que o aluno pode
   subir a acurácia noturna e baixar o F1 macro noturno ao mesmo tempo.
4. **O piso e "quando decora":** acurácia de treino > 95% na época 5 com 105 imagens; perda de val
   quase plana; por que a época escolhida no val varia tanto entre seeds.
5. **Zero-shot do CLIP:** embeddings de imagem e de texto ("a photo of a {classe}"), similaridade de
   cosseno, softmax. Por que um modelo treinado em ~400 milhões de pares imagem-texto perde de 105 rótulos aqui
   (domínio IR, animais pequenos, prompts ruins para "dog"/"domestic_cat").
6. **Matriz de confusão do professor = matriz de ruído** dos pseudo-rótulos: cada célula fora da
   diagonal é um rótulo errado que o aluno recebe como verdade. Classes ímã.
7. **Professor melhorado:** regressão logística sobre os embeddings com os rótulos do orçamento,
   combinada com o zero-shot (`α·log p_zs + (1−α)·log p_LR`), α e C por CV dentro dos rótulos; por que
   no orçamento 1 isso não é possível.
8. **Tratamento de ruído:** filtro por confiança (τ), rótulos suaves com temperatura (o que T faz com a
   distribuição; por que escolhi T pela log-loss nos rótulos), amostragem balanceada. Por que A
   aprende o ruído (melhor val na época 1). O custo do filtro: cobertura de 35% e classes que nunca
   passam (gato, cão, pássaro).
9. **Onde e por que o aluno supera o professor:** precisão alta dos pseudo-rótulos que passam no filtro
   (0,747 vs 0,256), concentrada em poucas classes; correção de erros de baixa confiança (tâmia →
   opossum); 194 vs 74 discordâncias resolvidas a favor do aluno.
10. **Curva de orçamento:** por que o ganho encolhe e vira perda com 20 rótulos (τ fixo, pseudo-rótulos
    que não melhoram); por que o professor melhorado fica acima do aluno.
11. **Grad-CAM:** gradiente do logit da classe em relação às ativações da `layer4`, média espacial como
    peso de cada canal, ReLU da soma ponderada; resolução 7×7. Por que ele pode "acender no animal" e
    o modelo ainda assim decidir pelo fundo — e como oclusão e "tapar o animal" testam isso.
12. **A pergunta da capivara:** os quatro sinais (confiança/entropia, discordância aluno × professor,
    distância no embedding, professor com vocabulário aberto) e o limite de cada um com os meus dados.
13. **O que o `predict.py` faz** (CPU, offline, `weights=None` + state_dict, mesmo pré-processamento do
    treino) e por que o modelo entregue tem 11.187.285 parâmetros.
