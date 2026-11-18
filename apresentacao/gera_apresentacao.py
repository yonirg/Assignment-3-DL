"""Gera apresentacao/PA3_apresentacao.pptx (16:9, português, fala de cada slide nas notas).

    .venv/bin/python apresentacao/gera_apresentacao.py

Todos os números vêm de RESULTADOS.md / curva_orcamento.csv / results/*.json; as figuras vêm de
results/figs/ (geradas pelos scripts do repositório). Fonte Arial em tudo (renderiza igual em
qualquer máquina).
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

RAIZ = Path(__file__).resolve().parent.parent
FIGS = RAIZ / "results" / "figs"
IMGS = RAIZ / "data" / "pa3_dados" / "images" / "test_public"
SAIDA = RAIZ / "apresentacao" / "PA3_apresentacao.pptx"

ESCURO = RGBColor(0x14, 0x28, 0x1D)    # verde-floresta noturno (capa e fechamento)
TEXTO = RGBColor(0x1E, 0x24, 0x2B)
SUAVE = RGBColor(0x5B, 0x64, 0x6E)
AMBAR = RGBColor(0xE0, 0xA0, 0x30)     # "dia" nas figuras
NOITE = RGBColor(0x40, 0x4A, 0x80)     # "noite/IR" nas figuras
VERDE = RGBColor(0x2C, 0x8C, 0x4A)
VERMELHO = RGBColor(0xC0, 0x39, 0x2B)
CARTAO = RGBColor(0xF1, 0xF4, 0xF2)
BRANCO = RGBColor(0xFF, 0xFF, 0xFF)
FONTE = "Arial"

W, H = Inches(13.333), Inches(7.5)
prs = Presentation()
prs.slide_width, prs.slide_height = W, H
LAYOUT = prs.slide_layouts[6]  # em branco
N_SLIDE = [0]


def _fundo(slide, cor):
    f = slide.background.fill
    f.solid()
    f.fore_color.rgb = cor


def texto(slide, x, y, w, h, paragrafos, tam=16, cor=TEXTO, negrito=False, alinh=PP_ALIGN.LEFT,
          ancora=MSO_ANCHOR.TOP, espaco=6, nome=None):
    """paragrafos: str ou lista de str / (str, dict) com chaves tam, cor, negrito, italico."""
    tb = slide.shapes.add_textbox(x, y, w, h)
    if nome:
        tb.name = nome
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = ancora
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    if isinstance(paragrafos, str):
        paragrafos = [paragrafos]
    for i, p in enumerate(paragrafos):
        s, o = (p, {}) if isinstance(p, str) else p
        par = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        par.alignment = o.get("alinh", alinh)
        par.space_after = Pt(o.get("espaco", espaco))
        for j, pedaco in enumerate(s.split("\n")):  # "\n" vira quebra de linha (<a:br/>)
            if j:
                par.add_line_break()
            r = par.add_run()
            r.text = pedaco
            r.font.name = FONTE
            r.font.size = Pt(o.get("tam", tam))
            r.font.bold = o.get("negrito", negrito)
            r.font.italic = o.get("italico", False)
            r.font.color.rgb = o.get("cor", cor)
    return tb


def caixa(slide, x, y, w, h, cor=CARTAO, borda=None, forma=MSO_SHAPE.ROUNDED_RECTANGLE):
    s = slide.shapes.add_shape(forma, x, y, w, h)
    s.fill.solid()
    s.fill.fore_color.rgb = cor
    if borda is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = borda
        s.line.width = Pt(1.5)
    s.shadow.inherit = False
    if forma == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = 0.08
    return s


def etiqueta(slide, rotulo, x=Inches(0.6), y=Inches(0.35), cor=NOITE):
    w = Inches(0.13 * len(rotulo) + 0.35)
    s = caixa(slide, x, y, w, Inches(0.32), cor=cor)
    s.adjustments[0] = 0.5
    tf = s.text_frame
    tf.margin_top = tf.margin_bottom = 0
    r = tf.paragraphs[0].add_run()
    r.text = rotulo
    r.font.name, r.font.size, r.font.bold, r.font.color.rgb = FONTE, Pt(11), True, BRANCO
    tf.paragraphs[0].alignment = PP_ALIGN.CENTER
    return s


def figura(slide, caminho, x, y, w_max, h_max, alinh="centro"):
    """Insere a figura sem distorcer, cabendo em (w_max, h_max)."""
    iw, ih = Image.open(caminho).size
    esc = min(w_max / iw, h_max / ih)
    w, h = int(iw * esc), int(ih * esc)
    dx = (w_max - w) // 2 if alinh == "centro" else 0
    return slide.shapes.add_picture(str(caminho), x + dx, y + (h_max - h) // 2, w, h)


def novo(titulo=None, tag=None, escuro=False, notas=""):
    s = prs.slides.add_slide(LAYOUT)
    N_SLIDE[0] += 1
    _fundo(s, ESCURO if escuro else BRANCO)
    if tag:
        etiqueta(s, tag, cor=AMBAR if escuro else NOITE)
    if titulo:
        texto(s, Inches(0.6), Inches(0.72), Inches(12.1), Inches(0.75), titulo, tam=30, negrito=True,
              cor=BRANCO if escuro else TEXTO, nome="titulo")
    if not escuro:
        texto(s, Inches(12.3), Inches(7.05), Inches(0.8), Inches(0.35), str(N_SLIDE[0]), tam=10,
              cor=SUAVE, alinh=PP_ALIGN.RIGHT)
    s.notes_slide.notes_text_frame.text = notas.strip()
    return s


def numero(slide, x, y, w, valor, legenda, cor=TEXTO, tam=36):
    texto(slide, x, y, w, Inches(0.7), valor, tam=tam, negrito=True, cor=cor)
    texto(slide, x, y + Inches(0.68), w, Inches(0.6), legenda, tam=12, cor=SUAVE)


def seta(slide, x1, y1, x2, y2, cor=SUAVE):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, x1, y1, x2, y2)
    c.line.color.rgb = cor
    c.line.width = Pt(2.25)
    ln = c.line._get_or_add_ln()
    from lxml import etree
    etree.SubElement(ln, "{http://schemas.openxmlformats.org/drawingml/2006/main}tailEnd",
                     type="triangle", w="med", len="med")
    return c


I = Inches

# 1 ---------------------------------------------------------------- capa
s = novo(escuro=True, tag="PA3 · Trilha C", notas="""
Bom dia. O meu PA3 é sobre classificar 21 espécies em fotos de câmera trap tendo só 5 rótulos por
classe, ou seja, 105 imagens rotuladas, e mais de seis mil imagens sem rótulo. Escolhi a Trilha C,
professor e aluno: um modelo grande e aberto, o CLIP, rotula as imagens sem rótulo, e um modelo pequeno,
uma ResNet-18 que roda em CPU, aprende com esses rótulos. Adianto o tom: os ganhos existem, mas são
modestos, e uma parte do que vou mostrar é justamente onde o método não funcionou e por quê.
""")
texto(s, I(0.6), I(2.3), I(12), I(1.2), "Poucos rótulos: fauna em câmeras trap", tam=44, negrito=True, cor=BRANCO)
texto(s, I(0.6), I(3.45), I(12), I(0.6), "21 espécies, 5 rótulos por classe, 6.505 imagens sem rótulo",
      tam=22, cor=RGBColor(0xC8, 0xD6, 0xCC))
texto(s, I(0.6), I(4.05), I(12), I(0.6), "Professor CLIP ViT-B/16 → aluno ResNet-18 (11,2 M parâmetros, CPU)",
      tam=22, cor=RGBColor(0xC8, 0xD6, 0xCC))
texto(s, I(0.6), I(6.3), I(12), I(0.5), "Aprendizado Profundo · FGV · novembro de 2026", tam=14,
      cor=RGBColor(0x9F, 0xB3, 0xA6))

# 2 ---------------------------------------------------------------- problema
s = novo("Câmera trap não é MNIST", tag="O problema", notas="""
Por que isso é difícil? No laboratório de dados não rotulados, os métodos funcionavam bem no MNIST porque
o dígito ocupa a imagem inteira, o fundo é preto e as classes são balanceadas. Aqui nada disso vale. As
duas fotos à direita são do meu test_public: à esquerda, um gato à noite, em infravermelho, que ocupa
uns 30 por 24 pixels depois de redimensionar para 224; à direita, uma tâmia de uns 18 por 20 pixels ao lado
de um tambor de chuva. Muitas fotos são noturnas, sem cor. A mesma câmera fotografa sempre o mesmo fundo,
e o fundo vira um atalho tentador. E as classes são desbalanceadas: no pool de treino há 690 imagens de
urso e 45 de cavalo. Guardem o atalho do fundo: ele volta na galeria.
""")
cartoes = [("Animal pequeno", "em 1/4 das imagens ocupa < 2% do quadro", AMBAR),
           ("Noite / IR", "391 de 811 imagens do test_public sem cor", NOITE),
           ("Fundo fixo", "a mesma câmera repete o cenário: atalho", VERDE),
           ("Desbalanceado", "690 ursos e 45 cavalos no train", VERMELHO)]
for k, (t, d, c) in enumerate(cartoes):
    y = I(1.75 + 1.25 * k)
    caixa(s, I(0.6), y, I(5.2), I(1.05))
    caixa(s, I(0.8), y + I(0.3), I(0.45), I(0.45), cor=c, forma=MSO_SHAPE.OVAL)
    texto(s, I(1.45), y + I(0.1), I(4.2), I(0.45), t, tam=18, negrito=True)
    texto(s, I(1.45), y + I(0.52), I(4.3), I(0.45), d, tam=14, cor=SUAVE)
for k, (iid, leg) in enumerate([("6353c192fb1adc35", "gato, noite/IR: ~30×24 px a 224"),
                                ("2539e50f0cfcec84", "tâmia junto à mangueira: ~18×20 px")]):
    x = I(6.2 + 3.45 * k)
    im = Image.open(IMGS / f"{iid}.jpg").resize((224, 224), Image.BICUBIC)
    p = RAIZ / "apresentacao" / f"_tmp_{iid}.png"
    im.save(p)
    s.shapes.add_picture(str(p), x, I(1.9), I(3.25), I(3.25))
    p.unlink()
    texto(s, x, I(5.2), I(3.25), I(0.5), leg, tam=12, cor=SUAVE)
texto(s, I(6.2), I(5.85), I(6.6), I(0.9), "Imagens achatadas para 224×224 (como o modelo vê). "
      "Fonte: test_public.", tam=12, cor=SUAVE)

# 3 ---------------------------------------------------------------- dados e regras
s = novo("Dados, splits e regras do jogo", tag="Dados", notas="""
Os dados são do ENA24, câmeras do leste da América do Norte, uma espécie por imagem. O train tem 6.505
imagens sem rótulo, exceto as dos arquivos de orçamento: 1, 5 ou 20 por classe, aninhados. O padrão é 5,
que dá 105 imagens. O val tem 202 imagens, pequeno de propósito, e é só nele que escolhi época, limiar
e variante. O test_public, 811 imagens, serve só para reportar. O split agrupa por câmera e noite, então
uma rajada nunca aparece em dois conjuntos, mas a mesma câmera pode aparecer em noites diferentes. As
regras que segui à risca: o professor pode olhar o train e o test_public para ser medido, nunca o val;
o modelo entregue tem no máximo 12 milhões de parâmetros e roda em CPU sem internet; nenhum dado externo e
nenhum rótulo manual. A métrica principal é o F1 macro, que pesa cavalo igual a urso.
""")
linhas = [("conjunto", "imagens", "rótulo", "uso"),
          ("train", "6.505", "não*", "pseudo-rótulos do professor"),
          ("rotulos_1 / 5 / 20", "21 / 105 / 420", "sim", "orçamentos (subconjuntos do train)"),
          ("val", "202", "sim", "toda escolha de modelo"),
          ("test_public", "811", "sim", "só reportar"),
          ("test_private", "~800", "—", "fica com o monitor")]
tab = s.shapes.add_table(len(linhas), 4, I(0.6), I(1.75), I(7.4), I(3.3)).table
for j, wj in enumerate([2.0, 1.5, 0.9, 3.0]):
    tab.columns[j].width = I(wj)
for i, lin in enumerate(linhas):
    for j, v in enumerate(lin):
        cel = tab.cell(i, j)
        cel.text = v
        r = cel.text_frame.paragraphs[0].runs[0]
        r.font.name, r.font.size, r.font.bold = FONTE, Pt(14), i == 0
        r.font.color.rgb = BRANCO if i == 0 else TEXTO
        cel.fill.solid()
        cel.fill.fore_color.rgb = NOITE if i == 0 else (CARTAO if i % 2 else BRANCO)
texto(s, I(0.6), I(5.2), I(7.4), I(1.2),
      ["* exceto as imagens dos orçamentos. Split por câmera-noite: a mesma câmera pode voltar em outra noite.",
       "Métrica principal: F1 macro (cavalo pesa igual a urso)."], tam=13, cor=SUAVE)
caixa(s, I(8.4), I(1.75), I(4.35), I(4.6))
texto(s, I(8.65), I(1.9), I(3.9), I(4.4),
      [("Regras que segui", {"tam": 18, "negrito": True, "espaco": 12}),
       "Escolhas só no val (202 imagens)",
       "Professor nunca vê o val",
       "Professor fica fora do modelo entregue",
       "≤ 12 M parâmetros, CPU, sem internet",
       "Sem dado externo, sem rótulo manual, sem LLM rotulando"], tam=15, espaco=10)

# 4 ---------------------------------------------------------------- piso
s = novo("Piso: só os 105 rótulos — e quando decora", tag="Parte 0", notas="""
O piso é o que se consegue só com os 105 rótulos. A versão (a), uma CNN pequena treinada do zero, fica em
F1 macro 0,228 mais ou menos 0,024, média de três seeds. A versão (b), a ResNet-18 do ImageNet com
fine-tuning, que é o mesmo backbone do resto do trabalho, chega a 0,439 mais ou menos 0,011. Na curva,
a acurácia de treino passa de 95% já na época 5: a partir daí o modelo decorou as 105 imagens. A perda de
treino continua caindo duas ordens de grandeza, mas a perda de validação fica praticamente parada, com
mínimo de 2,19 na época 20. A época escolhida pelo F1 de val variou muito entre seeds, 42, 44 e 7, o que já
mostra que o val de 202 imagens é ruidoso. Resolução: 224 e 320 empataram no val, fiquei com 224, que é
duas vezes mais barata.
""")
figura(s, FIGS / "piso_curva.png", I(0.5), I(1.65), I(9.0), I(4.4))
numero(s, I(9.9), I(1.8), I(3.0), "0,439 ± 0,011", "piso (b) ResNet-18 ImageNet\nF1 macro, 3 seeds", tam=30)
numero(s, I(9.9), I(3.25), I(3.0), "0,228 ± 0,024", "piso (a) CNN do zero", cor=SUAVE, tam=30)
numero(s, I(9.9), I(4.5), I(3.0), "época ~5", "acurácia de treino > 95%:\ndecorou as 105 imagens", cor=VERMELHO, tam=30)
texto(s, I(0.6), I(6.2), I(9), I(0.6), "Perda de val quase plana depois da época 2 (mínimo 2,19 na 20); "
      "época escolhida no val: 42 / 44 / 7.", tam=13, cor=SUAVE)

# 5 ---------------------------------------------------------------- ideia professor/aluno
s = novo("A ideia: o professor rotula, o aluno aprende", tag="Trilha C", notas="""
A ideia da Trilha C em um diagrama. O professor é o CLIP ViT-B/16, um modelo aberto treinado em pares de
imagem e texto. Ele classifica as 6.505 imagens do train sem nenhum treino, comparando a imagem com o
texto "a photo of a" mais o nome da classe. Isso gera um pseudo-rótulo e uma confiança para cada imagem.
Como esses rótulos são ruidosos, filtro: só entram as imagens em que a probabilidade máxima do
professor é pelo menos 0,5, o que dá 2.225 imagens, cerca de 35% do pool. O aluno, uma ResNet-18 partindo
do ImageNet, treina nessas imagens junto com os 105 rótulos verdadeiros, e nas imagens rotuladas usa
sempre o rótulo verdadeiro. O professor fica de fora do modelo entregue: o que vai para o campo é só o
aluno, com 11,2 milhões de parâmetros, rodando em CPU.
""")
blocos = [(I(0.6), I(2.0), "6.505 imagens\nsem rótulo (train)", CARTAO, TEXTO),
          (I(3.55), I(2.0), "Professor\nCLIP ViT-B/16 zero-shot\n\"a photo of a {classe}\"", NOITE, BRANCO),
          (I(6.5), I(2.0), "pseudo-rótulo +\nconfiança max p", CARTAO, TEXTO),
          (I(9.45), I(2.0), "filtro max p ≥ 0,5\n2.225 imagens (34,8%)", AMBAR, TEXTO)]
for x, y, t, c, ct in blocos:
    caixa(s, x, y, I(2.6), I(1.35), cor=c)
    texto(s, x, y, I(2.6), I(1.35), t, tam=15, negrito=True, cor=ct, alinh=PP_ALIGN.CENTER, ancora=MSO_ANCHOR.MIDDLE)
for k in range(3):
    seta(s, I(3.2 + 2.95 * k), I(2.675), I(3.55 + 2.95 * k), I(2.675))
caixa(s, I(6.5), I(4.4), I(3.4), I(1.45), cor=VERDE)
texto(s, I(6.5), I(4.4), I(3.4), I(1.45), "Aluno ResNet-18 ImageNet\n11,2 M parâmetros · CPU\n(é o modelo entregue)",
      tam=15, negrito=True, cor=BRANCO, alinh=PP_ALIGN.CENTER, ancora=MSO_ANCHOR.MIDDLE)
seta(s, I(10.75), I(3.35), I(9.9), I(4.6))
caixa(s, I(0.6), I(4.4), I(3.6), I(1.45), cor=CARTAO)
texto(s, I(0.6), I(4.4), I(3.6), I(1.45), "105 rótulos verdadeiros\n(5 por classe)", tam=15, negrito=True,
      alinh=PP_ALIGN.CENTER, ancora=MSO_ANCHOR.MIDDLE)
seta(s, I(4.2), I(5.125), I(6.5), I(5.125))
texto(s, I(0.6), I(6.2), I(12), I(0.8), "O professor nunca vê o val e não entra no modelo entregue. "
      "Nas imagens rotuladas o aluno usa o rótulo verdadeiro, não o pseudo.", tam=14, cor=SUAVE)

# 6 ---------------------------------------------------------------- professor zero-shot
s = novo("Professor zero-shot: abaixo do piso", tag="Parte 1", notas="""
Primeiro, qual professor. Testei o BioCLIP, treinado em espécies, e o CLIP ViT-B/16 genérico. A escolha
foi feita só com os 105 rótulos do train, nunca com o val nem com o teste: o CLIP acertou 43% deles
contra 33% do BioCLIP, e fiquei com o CLIP. No test_public, com o prompt obrigatório, o CLIP zero-shot
dá F1 macro 0,375, abaixo do piso de 0,439. Ou seja, o professor sozinho é pior que 105 rótulos. À
direita está a matriz de confusão do professor no test_public, que é a matriz de ruído dos
pseudo-rótulos: cada linha é a classe verdadeira, e o que estiver fora da diagonal é ruído que o aluno
vai receber como se fosse verdade.
""")
figura(s, FIGS / "professor_confusao.png", I(5.6), I(1.45), I(7.3), I(5.9))
linhas = [("professor", "acerto nos 105\n(critério)", "F1 test_public"),
          ("CLIP ViT-B/16", "0,429", "0,375"), ("BioCLIP", "0,333", "0,310")]
tab = s.shapes.add_table(3, 3, I(0.6), I(1.8), I(4.8), I(1.6)).table
for j, wj in enumerate([1.8, 1.6, 1.4]):
    tab.columns[j].width = I(wj)
for i, lin in enumerate(linhas):
    for j, v in enumerate(lin):
        cel = tab.cell(i, j)
        cel.text = v
        for par in cel.text_frame.paragraphs:
            for r in par.runs:
                r.font.name, r.font.size, r.font.bold = FONTE, Pt(14), i == 0 or (i == 1 and j == 0)
                r.font.color.rgb = BRANCO if i == 0 else TEXTO
        cel.fill.solid()
        cel.fill.fore_color.rgb = NOITE if i == 0 else (CARTAO if i % 2 else BRANCO)
numero(s, I(0.6), I(3.8), I(4.8), "0,375 < 0,439", "F1 macro: professor zero-shot < piso (b)", cor=VERMELHO, tam=32)
texto(s, I(0.6), I(5.2), I(4.8), I(1.6), ["Pré-processamento (recorte do open_clip) também escolhido "
      "só nos 105 rótulos.", "Matriz à direita = matriz de ruído dos pseudo-rótulos."], tam=13, cor=SUAVE)

# 7 ---------------------------------------------------------------- onde o professor erra
s = novo("Onde o professor erra de forma sistemática", tag="Parte 1", notas="""
Olhando a matriz de ruído, os erros não são aleatórios. domestic_cat nunca é previsto: zero previsões
para 48 gatos, que viram lince ou guaxinim. dog quase nunca: 9 previsões para 67 cães, espalhados em
corvo, raposa, coiote e marmota. red_fox noturno vira grey_fox: 15 casos, e o F1 noturno de red_fox é zero.
E 21 das 31 tâmias viram opossum. Há também classes ímã, que recebem muito mais previsões do que existem,
como woodchuck com 75 previsões para 22 imagens. O gráfico mostra o F1 por classe de dia e de noite: o F1
macro cai de 0,430 de dia para 0,292 à noite. Um cuidado: espécie e iluminação estão misturadas no
test_public, coiote só aparece à noite e corvo só de dia, então parte dessa diferença é de classe e não de
luz. E o par que o enunciado sugeria, bird com american_crow, aparece fraco aqui, 4 de 22.
""")
figura(s, FIGS / "professor_dia_noite.png", I(0.5), I(1.55), I(8.4), I(4.1))
erros = [("domestic_cat: 0 previsões", "48 gatos → bobcat / raccoon"),
         ("dog: 9 previsões para 67", "→ crow, fox, coyote, woodchuck"),
         ("red_fox → grey_fox: 15", "F1 noturno de red_fox = 0"),
         ("chipmunk → opossum: 21 de 31", "erro de baixa confiança")]
for k, (t, d) in enumerate(erros):
    y = I(1.6 + 1.08 * k)
    caixa(s, I(9.15), y, I(3.75), I(0.95))
    texto(s, I(9.3), y + I(0.05), I(3.5), I(0.45), t, tam=15, negrito=True)
    texto(s, I(9.3), y + I(0.48), I(3.5), I(0.4), d, tam=13, cor=SUAVE)
numero(s, I(0.6), I(5.75), I(4), "dia 0,430 · noite 0,292", "F1 macro do professor (classes presentes)", tam=24)
texto(s, I(5.0), I(5.8), I(7.9), I(1.2), "Espécie e luz se confundem no test_public: coiote, opossum, "
      "veado só à noite; corvo, galinha, peru só de dia.", tam=13, cor=SUAVE)

# 8 ---------------------------------------------------------------- ablação
s = novo("Tratamento do ruído: cinco variantes, escolha no val", tag="Parte 1", notas="""
Testei cinco formas de usar os pseudo-rótulos, todas no orçamento 5, seed 0. A, rótulo duro do
zero-shot sem tratamento. B, o mesmo com filtro por confiança, com limiar 0,3, 0,5 ou 0,7. C, rótulos
suaves do zero-shot com temperatura. D, rótulos suaves do professor melhorado, que é o zero-shot
combinado com uma regressão logística nos 105 rótulos. E, D com amostragem balanceada. As barras claras são
o F1 de val, que é o critério; as escuras são o teste, só reportado. No val ganhou B com tau 0,5, por
0,0010 sobre E: 0,4744 contra 0,4734, o que é um quinto de imagem de val. No teste, depois, a ordem foi
outra: D deu 0,543, B com 0,7 deu 0,514, E 0,506, e a minha escolha, 0,470, ficou em quarto. Mantive B,
porque a regra é escolher no val. É um exemplo concreto do que um val de 202 imagens não consegue separar.
""")
figura(s, FIGS / "ablacao_ruido.png", I(0.4), I(1.5), I(8.9), I(5.0))
numero(s, I(9.6), I(1.7), I(3.4), "B, τ = 0,5", "escolhida no val: 0,4744\n(E: 0,4734, diferença 0,0010)", cor=NOITE, tam=30)
numero(s, I(9.6), I(3.25), I(3.4), "4ª no teste", "teste: D 0,543 > B τ=0,7 0,514 >\nE 0,506 > B τ=0,5 0,470", cor=VERMELHO, tam=30)
texto(s, I(9.6), I(4.85), I(3.4), I(1.8), ["A (sem tratamento): melhor val na época 1, depois cai — aprende o ruído.",
      "C (suave zero-shot): 0,381 no teste, abaixo do piso."], tam=13, cor=SUAVE)

# 9 ---------------------------------------------------------------- curva de orçamento
s = novo("Curva de orçamento: ganha com 1 e 5 rótulos, perde com 20", tag="Parte 2", notas="""
A curva de orçamento. No eixo x, rótulos por classe em escala log; no y, F1 macro no test_public. O
aluno, em azul, ganha muito do piso com 1 rótulo, 0,417 contra 0,235, ganha pouco com 5, 0,480 contra
0,439 em média de três seeds, e perde com 20 rótulos: 0,572 contra 0,625 do piso. O motivo mais provável:
o filtro é o mesmo, tau 0,5 sobre o zero-shot, em todos os orçamentos, então os pseudo-rótulos não
melhoram quando chegam mais rótulos, e com 20 por classe o piso já é melhor do que eles. O professor
melhorado, em laranja tracejado, fica acima do aluno com 5 e com 20, e com 20 chega a 0,665, perto do
teto do monitor, 0,686. Ou seja: com 5 rótulos o meu aluno recupera só uma parte da distância entre o
piso e o teto.
""")
figura(s, FIGS / "curva_orcamento.png", I(0.4), I(1.5), I(7.6), I(5.4))
linhas = [("F1 macro", "1", "5 (3 seeds)", "20"),
          ("piso (b)", "0,235", "0,439 ± 0,011", "0,625"),
          ("aluno", "0,417", "0,480 ± 0,017", "0,572"),
          ("prof. zero-shot", "0,375", "0,375", "0,375"),
          ("prof. melhorado", "0,380", "0,538", "0,665"),
          ("teto do monitor", "—", "0,686", "—")]
tab = s.shapes.add_table(len(linhas), 4, I(8.2), I(1.75), I(4.8), I(3.3)).table
for j, wj in enumerate([1.55, 0.75, 1.6, 0.9]):
    tab.columns[j].width = I(wj)
for i, lin in enumerate(linhas):
    for j, v in enumerate(lin):
        cel = tab.cell(i, j)
        cel.text = v
        r = cel.text_frame.paragraphs[0].runs[0]
        r.font.name, r.font.size, r.font.bold = FONTE, Pt(13), i in (0, 2)
        r.font.color.rgb = BRANCO if i == 0 else (VERMELHO if (i == 2 and j == 3) else TEXTO)
        cel.fill.solid()
        cel.fill.fore_color.rgb = NOITE if i == 0 else (CARTAO if i % 2 else BRANCO)
texto(s, I(8.2), I(5.3), I(4.8), I(1.6), ["Mesmo τ = 0,5 em todos os orçamentos: o pool filtrado (~35%) "
      "não melhora com mais rótulos.", "Aluno abaixo do professor melhorado com 5 e 20."], tam=13, cor=SUAVE)

# 10 --------------------------------------------------------------- ganho por classe
s = novo("Ganho por classe: concentrado em poucas espécies", tag="Parte 2", notas="""
Quebrando o ganho por classe no orçamento 5, média das três seeds. Os maiores ganhos sobre o piso são
american_crow, de 0,29 para 0,80, urso, de 0,48 para 0,82, e peru, de 0,25 para 0,57; também marmota,
galinha e coiote. Não ganharam nada bobcat e grey_fox. E perderam várias: striped_skunk caiu de 0,37 para
0,12, domestic_cat de 0,24 para zero, cavalo de 0,82 para 0,59 e tâmia de 0,95 para 0,76. O padrão é
claro: ganham as classes em que o professor é confiante e certo, porque são essas que passam no filtro e
chegam ao aluno em quantidade. Perdem as classes que o professor não reconhece, como gato, ou que quase
não passam no filtro, como skunk: para elas o aluno só tem os 5 rótulos e ainda vê o pool cheio de outras
classes.
""")
figura(s, FIGS / "ganho_por_classe.png", I(0.4), I(1.45), I(7.0), I(5.6))
numero(s, I(7.9), I(1.7), I(5), "+0,51 american_crow", "0,29 → 0,80 · também urso +0,34, peru +0,32", cor=VERDE, tam=28)
numero(s, I(7.9), I(3.1), I(5), "−0,25 striped_skunk", "0,37 → 0,12 · gato 0,24 → 0,00 · cavalo −0,23", cor=VERMELHO, tam=28)
texto(s, I(7.9), I(4.6), I(5), I(2.2), ["Ganham as classes em que o professor é confiante e certo "
      "(passam no filtro).", "Perdem as que o professor não reconhece ou quase não passam no filtro: "
      "o aluno só tem os 5 rótulos delas."], tam=15)

# 11 --------------------------------------------------------------- onde o aluno supera o professor
s = novo("Onde o aluno supera o professor, e por quê", tag="Parte 1", notas="""
Em quais classes o aluno supera o professor? Ele fica acima do zero-shot em 14 das 21 classes, e acima dos
dois professores, zero-shot e melhorado, em 8, marcadas com estrela: esquilo-cinzento, opossum, urso,
corvo, galinha, coiote, cavalo e grey_fox. Por quê? Primeiro, o filtro deixa passar pseudo-rótulos
certos. Rodando o zero-shot no test_public só como diagnóstico, as previsões com confiança acima de 0,5
acertam 75%, contra 26% das que ficam abaixo, e a precisão é altíssima justamente onde o aluno ganha:
urso 0,98, opossum 0,96, corvo 0,95. Segundo, o aluno corrige erros de baixa confiança do professor. O
caso tâmia virando opossum é de baixa confiança, não passa no filtro; o aluno aprende opossum com
pseudo-rótulos limpos e tâmia com os rótulos verdadeiros. Quando os dois discordam, só o aluno acerta em
194 imagens e só o professor em 74.
""")
figura(s, FIGS / "aluno_vs_professor.png", I(0.3), I(1.4), I(9.4), I(4.5))
numero(s, I(9.85), I(1.6), I(3.3), "0,747 vs 0,256", "acerto do zero-shot com max p ≥ 0,5\nvs abaixo (test_public, diagnóstico)", cor=VERDE, tam=28)
numero(s, I(9.85), I(3.15), I(3.3), "194 vs 74", "discordâncias em que só o aluno /\nsó o professor acerta", cor=NOITE, tam=28)
texto(s, I(0.6), I(5.95), I(12.3), I(1.2),
      [("Acima do zero-shot em 14 de 21 classes; acima dos dois professores em 8 (★).", {"negrito": True}),
       "Noite: acurácia sobe (aluno 0,496 vs piso 0,411), mas o F1 macro noturno cai (0,328 vs 0,365): "
       "ganham as noturnas frequentes com pseudo-rótulo bom; perdem skunk, gato, red_fox."], tam=14)

# 12 --------------------------------------------------------------- galeria de erros
s = novo("Galeria de erros: um erro de leitura, dois de cenário", tag="Parte 3", notas="""
Agora a galeria. Regra de seleção, fixada no código antes de olhar: os três erros de maior confiança, sem
repetir classe. Em cima a imagem como o modelo vê, 224 por 224, com uma caixa verde que desenhei em volta
do animal; embaixo o Grad-CAM na layer4. Além do mapa, fiz um teste simples: tapar o animal de cinza e
ver quanto cai a confiança. Primeiro caso, gambá visto como esquilo-raposa com 99%: noite, animal de uns
75 por 60 pixels, mapa em cima dele, e tapar o gambá derruba a confiança para 0,17. O modelo olhou o
animal e leu errado. Segundo, cão visto como galinha com 96%: o cão está cortado na borda direita, atrás
de uma caixa de compostagem, e o mapa está na caixa. Tapar o cão não muda nada, 0,96 para 0,95: atalho de
cenário. Terceiro, gato visto como lince com 95%: o mapa está no gato, mas tapar o gato quase não muda,
0,95 para 0,92. Aqui o Grad-CAM engana: o mapa é de 7 por 7, e a célula sobre o gato cobre também o chão
em volta. É o erro que o aluno herdou do professor.
""")
figura(s, FIGS / "galeria_erros_slide.png", I(0.4), I(1.45), I(8.3), I(5.75))
for k, (t, d, c) in enumerate([("Gambá → esquilo (99%)", "olhou o animal e leu errado:\ntapar derruba 0,99 → 0,17", NOITE),
                               ("Cão → galinha (96%)", "mapa na caixa de compostagem;\ntapar o cão: 0,96 → 0,95", VERMELHO),
                               ("Gato → lince (95%)", "mapa no gato, mas tapar\nquase não muda: 0,95 → 0,92", VERMELHO)]):
    y = I(1.6 + 1.6 * k)
    caixa(s, I(8.95), y, I(4.0), I(1.4))
    caixa(s, I(9.1), y + I(0.2), I(0.3), I(0.3), cor=c, forma=MSO_SHAPE.OVAL)
    texto(s, I(9.5), y + I(0.08), I(3.4), I(0.45), t, tam=15, negrito=True)
    texto(s, I(9.5), y + I(0.55), I(3.4), I(0.8), d, tam=13, cor=SUAVE)
texto(s, I(8.95), I(6.45), I(4.0), I(0.6), "Regra: erros de maior confiança, classes distintas.", tam=12, cor=SUAVE)

# 13 --------------------------------------------------------------- galeria de acertos
s = novo("Galeria de acertos: o modelo acertou olhando o fundo", tag="Parte 3", notas="""
A pergunta inversa: cinco acertos, sorteados com semente fixa, três de dia e dois de noite, sem repetir
classe. Sorteio, e não os mais confiantes, para não escolher os mais fáceis. A resposta honesta é que, na
maioria, o modelo acertou pelo fundo. Na tâmia, o animal tem uns 18 por 20 pixels, o mapa acende no tambor
de chuva e tapar a tâmia mantém 0,93. No opossum, à noite, o mapa está nas árvores e postes iluminados, e
tapar o animal mantém 0,83. Na galinha, o mapa até cobre a galinha, mas tapá-la mantém 0,91. O corvo é
o caso mais saudável: tapar o corvo baixa de 0,97 para 0,61, mas ainda é corvo. E no coiote, com só 46% de
confiança, o mapa fica no capim ao lado; tapar o animal tira a classe, mas o Grad-CAM ali é
inconclusivo. Conclusão: o Grad-CAM sozinho seria otimista; o teste de tapar mostra que o cenário basta
na maior parte desses acertos. São oito casos, servem para ilustrar o mecanismo, não para medir a
frequência.
""")
figura(s, FIGS / "galeria_acertos_slide.png", I(0.3), I(1.4), I(12.7), I(5.0))
texto(s, I(0.6), I(6.45), I(12.2), I(0.9),
      [("Tapar o animal: quase não muda em 3 de 5 (tâmia 0,93, galinha 0,91, opossum 0,83); "
        "corvo cai para 0,61 e continua corvo; coiote (46%) muda de classe.", {"negrito": True, "espaco": 2}),
       "Regra: sorteio com semente 0 entre os acertos, 3 dia + 2 noite, classes distintas. "
       "Grad-CAM inconclusivo no coiote."], tam=12)

# 14 --------------------------------------------------------------- capivara
s = novo("E a primeira capivara no Uruguai?", tag="Pergunta", notas="""
A pergunta da capivara, que respondo sem implementar. O modelo sempre vai dizer uma das 21 classes.
O que no meu método daria um sinal de que a imagem não pertence a nenhuma? Quatro candidatos, cada um
com o seu limite. Um: a confiança ou a entropia da saída do aluno. É barato e já está no modelo entregue,
mas a galeria mostrou erros com 99, 96 e 95% de confiança, então confiança alta não garante nada.
Dois: a discordância entre aluno e professor. Só que eles já discordam em 484 das 811 imagens do teste,
que são todas de classes conhecidas, então o sinal seria muito ruidoso, e o professor também é forçado
a escolher entre 21 classes. Três: a distância da imagem às imagens de treino no espaço de embeddings
do aluno. O problema é que o embedding carrega muito fundo, como vimos: uma câmera nova no Uruguai
ficaria longe mesmo com um gato comum. Quatro: o professor com vocabulário aberto, acrescentando
"capybara" ou "outro animal" aos prompts. O CLIP pode conhecer a capivara, mas ele não está no modelo
entregue e é fraco à noite.
""")
sinais = [("Confiança / entropia do aluno", "barato, já está no modelo entregue",
           "a galeria tem erros com 99%, 96% e 95% de confiança"),
          ("Discordância aluno × professor", "dois modelos diferentes, erros diferentes",
           "já discordam em 484 de 811 imagens conhecidas; professor também força 21 classes"),
          ("Distância no espaço de embeddings", "longe de todo o train = suspeito",
           "o embedding carrega fundo: câmera nova fica longe mesmo com espécie conhecida"),
          ("Professor com vocabulário aberto", "prompts \"capybara\", \"outro animal\"",
           "professor fora do modelo entregue; zero-shot fraco à noite (F1 0,292)")]
for k, (t, pro, lim) in enumerate(sinais):
    x, y = I(0.6 + 6.25 * (k % 2)), I(1.7 + 2.55 * (k // 2))
    caixa(s, x, y, I(5.95), I(2.3))
    texto(s, x + I(0.25), y + I(0.15), I(5.5), I(0.5), t, tam=18, negrito=True)
    texto(s, x + I(0.25), y + I(0.72), I(5.5), I(0.7), [("+ " + pro, {"cor": VERDE})], tam=15)
    texto(s, x + I(0.25), y + I(1.3), I(5.5), I(0.9), [("limite: " + lim, {"cor": VERMELHO})], tam=15)

# 15 --------------------------------------------------------------- limitações
s = novo("Limitações e o que eu faria diferente", tag="Fechamento", notas="""
O que limitou o resultado e o que eu faria diferente. Primeiro, o val de 202 imagens é ruidoso: a
variante escolhida venceu por 0,0010, e no teste ficou em quarto. Eu faria a escolha com mais de uma
seed, ou com validação cruzada dentro dos rótulos, como fiz no professor melhorado. Segundo, usei o
mesmo tau 0,5 em todos os orçamentos, e por isso o aluno ficou abaixo do piso com 20 rótulos; o limiar
deveria acompanhar a qualidade do professor em cada orçamento. Terceiro, a fonte dos pseudo-rótulos: o
professor melhorado é claramente melhor que o zero-shot, 0,538 contra 0,375 com 5 rótulos, e seria a
fonte natural; mas como já vi o teste, não troquei de método agora. Quarto, o filtro muda o número de
passos por época, 35 contra 100, e isso não foi controlado. E por fim, a galeria mostra que o modelo usa
muito o fundo; recortar o animal com um detector, ou aumentações que quebrem o cenário, seriam o próximo
passo.
""")
lims = [("Val de 202 imagens é ruidoso", "B venceu E por 0,0010 no val e foi 4ª no teste",
         "escolher com várias seeds ou CV dentro dos rótulos"),
        ("Mesmo τ = 0,5 em todos os orçamentos", "aluno 0,572 < piso 0,625 com 20 rótulos",
         "limiar que acompanhe a qualidade do professor"),
        ("Fonte dos pseudo-rótulos", "professor melhorado 0,538 vs zero-shot 0,375 (5 rótulos)",
         "usar o melhorado (não troquei: o teste já foi visto)"),
        ("O modelo usa o fundo", "3 de 5 acertos resistem a tapar o animal",
         "recortar o animal / aumentação que quebre o cenário")]
for k, (t, ev, faria) in enumerate(lims):
    y = I(1.65 + 1.3 * k)
    caixa(s, I(0.6), y, I(12.15), I(1.12))
    texto(s, I(0.85), y + I(0.1), I(4.2), I(0.95), t, tam=17, negrito=True, ancora=MSO_ANCHOR.MIDDLE)
    texto(s, I(5.1), y + I(0.1), I(3.9), I(0.95), ev, tam=14, cor=SUAVE, ancora=MSO_ANCHOR.MIDDLE)
    texto(s, I(9.1), y + I(0.1), I(3.5), I(0.95), "→ " + faria, tam=14, cor=VERDE, ancora=MSO_ANCHOR.MIDDLE)
texto(s, I(0.6), I(6.85), I(12), I(0.45), "Também: o filtro muda os passos por época (35 vs 100) e isso não foi controlado.",
      tam=12, cor=SUAVE)

# 16 --------------------------------------------------------------- resumo
s = novo("Resumo", tag="Resumo", escuro=True, notas="""
Para fechar. Com 5 rótulos por classe, o aluno professor-e-aluno chega a F1 macro 0,480, contra 0,439
do piso e 0,375 do professor zero-shot; o modelo entregue dá 0,4699 e roda as 811 imagens em uns 14
segundos de CPU, com 11,2 milhões de parâmetros. O ganho vem das classes em que o professor é confiante e
certo, e some nas que ele não reconhece. O método não bate o professor melhorado, e perde do piso com 20
rótulos; a escolha feita no val não foi a melhor no teste, e mantive assim. E a galeria mostra que boa
parte dos acertos se apoia no cenário. Tudo o que mostrei está no repositório, reproduzível com um
comando para treinar e outro para avaliar. Obrigado.
""")
claro = RGBColor(0xC8, 0xD6, 0xCC)
for k, (v, l) in enumerate([("0,480 ± 0,017", "aluno, 5 rótulos (piso 0,439; professor zs 0,375)"),
                            ("0,4699", "modelo entregue · 11,2 M parâmetros · ~14 s em CPU (811 imagens)"),
                            ("0,538 / 0,625", "professor melhorado (5) e piso (20) ficam acima do aluno")]):
    texto(s, I(0.6), I(1.75 + 1.25 * k), I(4.2), I(0.8), v, tam=34, negrito=True, cor=AMBAR)
    texto(s, I(4.9), I(1.9 + 1.25 * k), I(8), I(0.8), l, tam=18, cor=claro)
texto(s, I(0.6), I(5.6), I(12.1), I(1.4),
      ["Ganho onde o professor é confiante e certo; perda onde ele não reconhece a classe (gato, skunk).",
       "Boa parte dos acertos se apoia no cenário. Tudo reproduzível: scripts/run_all.sh e scripts/eval.sh."],
      tam=17, cor=BRANCO)

# python-pptx cria o notesMaster mas não escreve <p:notesMasterIdLst> em presentation.xml; sem isso o
# Quick Look do macOS (OfficeImport) não abre o arquivo. Acrescentei o elemento logo após sldIdLst.
from lxml import etree  # noqa: E402
from pptx.opc.constants import RELATIONSHIP_TYPE as RT  # noqa: E402

_P = "http://schemas.openxmlformats.org/presentationml/2006/main"
_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_pres = prs.part._element
if _pres.find(f"{{{_P}}}notesMasterIdLst") is None:
    _rid = next(r.rId for r in prs.part.rels.values() if r.reltype == RT.NOTES_MASTER)
    _lst = etree.Element(f"{{{_P}}}notesMasterIdLst")
    etree.SubElement(_lst, f"{{{_P}}}notesMasterId").set(f"{{{_R}}}id", _rid)
    _pres.find(f"{{{_P}}}sldIdLst").addnext(_lst)

prs.save(SAIDA)
print(f"{N_SLIDE[0]} slides -> {SAIDA}")
