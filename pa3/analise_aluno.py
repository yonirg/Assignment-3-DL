"""Etapa 2 — análise: curva de orçamento (Parte 2), figura da ablação, ganho por classe, aluno × professor.

    python -m pa3.analise_aluno

Só LÊ resultados já produzidos (results/*.json, results/preds/*.npy, cache do professor). Nenhuma
escolha é feita aqui: tudo o que foi escolhido (época, τ, T, variante, seed entregue) já foi escolhido
no val pelos scripts de treino. O test_public é usado aqui só para medir.

Saídas: curva_orcamento.csv (raiz), results/aluno_analise.json, results/figs/{curva_orcamento,
ablacao_ruido, ganho_por_classe, aluno_vs_professor, aluno_dia_noite}.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pa3 import data, metrics, teacher  # noqa: E402
from pa3.plots import _salva  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
RES_DIR = RAIZ / "results"
PREDS = RES_DIR / "preds"
CACHE = RAIZ / "cache"
TETO = 0.686
ORCS = (1, 5, 20)


def _j(nome):
    return json.load(open(RES_DIR / nome))


def professor_test_preds() -> dict:
    """Previsões do professor no test_public, recalculadas do cache (sem val): zero-shot e melhorado."""
    prof = _j("professor.json")
    esc = prof["configs"]["clip_b16/recorte"]["escala_logit"]
    Fte = np.load(CACHE / "prof_clip_b16_recorte_test_public_img.npy")
    Ftr = np.load(CACHE / "prof_clip_b16_recorte_train_img.npy")
    T = np.load(CACHE / "prof_clip_b16_texto.npy")
    Zte = esc * Fte @ T.T
    out = {"zs": Zte.argmax(1), "zs_conf": np.exp(teacher.log_softmax(Zte)).max(1)}
    pos = {i: k for k, i in enumerate(data.ids_train())}
    for K in ORCS:
        orc = data.rotulos(f"rotulos_{K}")
        idx = np.array([pos[i] for i in orc["image_id"]])
        hp = prof["melhorado"][str(K)]["hp"]
        lp = teacher.combina(Zte, teacher.lr_logprob(Ftr[idx], orc["y"].to_numpy(), Fte, hp["C"]), hp["alfa"])
        out[f"mel{K}"] = lp.argmax(1)
    return out


def main():
    nomes = data.classes()
    yte = data.rotulos("test_public")["y"].to_numpy()
    _, arr_te = data.imagens("test_public", 224)
    noite = data.flag_noite(arr_te)
    prof = _j("professor.json")
    abl = _j("aluno_ablacao.json")
    met = _j("aluno_metodo.json")
    piso = {K: _j(f"piso_b_r{K}.json") for K in ORCS}
    piso_etapa1 = _j("piso_r5.json")
    out = {}

    # ---------------------------------------------------------------- conferências
    pp = professor_test_preds()
    out["conferencia_professor"] = {
        "zs_recalculado": metrics.resumo(yte, pp["zs"])["f1_macro"],
        "zs_registrado": prof["analise_zs_escolhido"]["resumo"]["f1_macro"],
        **{f"mel{K}_recalculado": metrics.resumo(yte, pp[f"mel{K}"])["f1_macro"] for K in ORCS},
        **{f"mel{K}_registrado": prof["melhorado"][str(K)]["test"]["f1_macro"] for K in ORCS}}
    b1 = {r["seed"]: r for r in piso_etapa1["runs"] if r["modelo"] == "resnet18"}
    out["reprodutibilidade_piso_r5"] = [{"seed": r["seed"], "f1_val_etapa1": b1[r["seed"]]["f1_val"],
                                         "f1_val_rerun": r["f1_val"], "test_etapa1": b1[r["seed"]]["test_f1_macro"],
                                         "test_rerun": r["test_f1_macro"]} for r in piso[5]["runs"]]

    # ---------------------------------------------------------------- curva_orcamento.csv
    linhas = []
    for K in ORCS:
        for r in piso[K]["runs"]:
            linhas.append({"metodo": "piso_b_resnet18", "orcamento": K, "seed": r["seed"],
                           "f1_macro": r["test_f1_macro"], "acuracia": r["test_acuracia"]})
    for r in sorted(met["runs"], key=lambda r: (r["orcamento"], r["seed"])):
        linhas.append({"metodo": f"aluno_{met['variante']}", "orcamento": r["orcamento"], "seed": r["seed"],
                       "f1_macro": r["test_f1_macro"], "acuracia": r["test_acuracia"]})
    zs = prof["analise_zs_escolhido"]["resumo"]
    for K in ORCS:  # o zero-shot não usa rótulo: mesmo valor em todos os orçamentos
        linhas.append({"metodo": "professor_zero_shot", "orcamento": K, "seed": "",
                       "f1_macro": zs["f1_macro"], "acuracia": zs["acuracia"]})
    for K in ORCS:
        t = prof["melhorado"][str(K)]["test"]
        linhas.append({"metodo": "professor_melhorado", "orcamento": K, "seed": "",
                       "f1_macro": t["f1_macro"], "acuracia": t["acuracia"]})
    df = pd.DataFrame(linhas)
    df.to_csv(RAIZ / "curva_orcamento.csv", index=False, float_format="%.4f")

    def resumo_metodo(m):
        d = df[df.metodo == m]
        res = {}
        for K in ORCS:
            dk = d[d.orcamento == K]
            res[K] = {"f1_media": float(dk.f1_macro.mean()), "f1_desvio": float(dk.f1_macro.std(ddof=0)),
                      "acc_media": float(dk.acuracia.mean()), "acc_desvio": float(dk.acuracia.std(ddof=0)), "n": len(dk)}
        return res

    met_aluno = f"aluno_{met['variante']}"
    curvas = {m: resumo_metodo(m) for m in ("piso_b_resnet18", met_aluno, "professor_zero_shot", "professor_melhorado")}
    out["curva"] = curvas

    # ---------------------------------------------------------------- curva_orcamento.png
    fig, ax = plt.subplots(figsize=(7.5, 5))
    estilos = {"piso_b_resnet18": ("Piso (b) ResNet-18, só rótulos", "#7f7f7f", "o", "-"),
               met_aluno: (f"Aluno ResNet-18 (variante {met['variante']})", "#1f77b4", "s", "-"),
               "professor_melhorado": ("Professor melhorado (CLIP + LR)", "#ff7f0e", "^", "--"),
               "professor_zero_shot": ("Professor zero-shot (CLIP ViT-B/16)", "#d62728", "", ":")}
    for m, (rot, cor, mk, ls) in estilos.items():
        y = [curvas[m][K]["f1_media"] for K in ORCS]
        e = [curvas[m][K]["f1_desvio"] if K == 5 else 0 for K in ORCS]
        if mk:
            ax.errorbar(ORCS, y, yerr=e, label=rot, color=cor, marker=mk, ls=ls, capsize=4, lw=2)
        else:
            ax.plot(ORCS, y, label=rot, color=cor, ls=ls, lw=1.5)
        for K, v in zip(ORCS, y):
            if mk:
                desl = ((-40, 2) if K == 20 else (6, -14)) if m == "piso_b_resnet18" else (6, 6)
                if m == met_aluno and K == 20:
                    desl = (6, -12)
                ax.annotate(f"{v:.3f}", (K, v), textcoords="offset points", xytext=desl, fontsize=8, color=cor)
    ax.axhline(TETO, color="black", ls="--", lw=1, label=f"Teto do monitor (todos os rótulos): {TETO}")
    ax.set_xscale("log")
    ax.set_xticks(ORCS, [str(k) for k in ORCS])
    ax.set_xlabel("rótulos por classe (escala log)")
    ax.set_ylabel("F1 macro no test_public")
    ax.set_ylim(0, 0.8)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Curva de orçamento — barras de erro: ±1 desvio em 3 seeds (orçamento 5)", fontsize=10)
    _salva(fig, "curva_orcamento.png")

    # ---------------------------------------------------------------- ablacao_ruido.png
    runs = abl["runs"]
    rot_x = [f"{r['variante']}" + (f"\nτ={r['limiar']}" if r["variante"] == "B" else "") for r in runs]
    x = np.arange(len(runs))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(x - 0.2, [r["f1_val"] for r in runs], 0.4, label="F1 macro val (critério de escolha)", color="#9ecae1")
    ax.bar(x + 0.2, [r["test_f1_macro"] for r in runs], 0.4, label="F1 macro test_public (só reporte)", color="#3182bd")
    for i, r in enumerate(runs):
        ax.text(x[i] - 0.2, r["f1_val"] + 0.01, f"{r['f1_val']:.3f}", ha="center", fontsize=7)
        ax.text(x[i] + 0.2, r["test_f1_macro"] + 0.01, f"{r['test_f1_macro']:.3f}", ha="center", fontsize=7)
    escolhida_tag = abl["tag_por_variante"][abl["variante_escolhida"]]
    i_esc = next(i for i, r in enumerate(runs) if r["tag"] == escolhida_tag)
    ax.annotate("escolhida no val", (x[i_esc] - 0.2, runs[i_esc]["f1_val"] + 0.05), ha="center", fontsize=8,
                color="darkgreen", fontweight="bold")
    p5 = next(r for r in piso[5]["runs"] if r["seed"] == 0)
    ax.axhline(p5["test_f1_macro"], color="#7f7f7f", ls="-", lw=1, label=f"piso (b) seed 0, test: {p5['test_f1_macro']:.3f}")
    ax.axhline(zs["f1_macro"], color="#d62728", ls=":", lw=1.2, label=f"professor zero-shot, test: {zs['f1_macro']:.3f}")
    t5 = prof["melhorado"]["5"]["test"]["f1_macro"]
    ax.axhline(t5, color="#ff7f0e", ls="--", lw=1.2, label=f"professor melhorado (5), test: {t5:.3f}")
    ax.set_xticks(x, rot_x)
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("F1 macro")
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.set_title("Ablação do tratamento de ruído — orçamento 5, seed 0\nA duro zs | B duro zs + filtro τ | "
                 "C suave zs (T) | D suave prof. melhorado | E = D + amostragem balanceada", fontsize=9)
    _salva(fig, "ablacao_ruido.png")

    # ---------------------------------------------------------------- por classe, orçamento 5 (3 seeds)
    al5 = sorted([r for r in met["runs"] if r["orcamento"] == 5], key=lambda r: r["seed"])
    pred_al = {r["seed"]: np.load(PREDS / f"aluno_{r['tag']}_test_logits.npy").argmax(1) for r in al5}
    pred_pi = {s: np.load(PREDS / f"piso_b_r5_s{s}_test_logits.npy").argmax(1) for s in (0, 1, 2)}
    # conferência: F1 recalculado dos logits = registrado
    out["conferencia_logits"] = {
        "aluno": [(metrics.resumo(yte, pred_al[r["seed"]])["f1_macro"], r["test_f1_macro"]) for r in al5],
        "piso": [(metrics.resumo(yte, pred_pi[r["seed"]])["f1_macro"], r["test_f1_macro"]) for r in piso[5]["runs"]]}
    f1_al = np.mean([metrics.f1_por_classe(yte, p) for p in pred_al.values()], 0)
    f1_pi = np.mean([metrics.f1_por_classe(yte, p) for p in pred_pi.values()], 0)
    f1_zs = metrics.f1_por_classe(yte, pp["zs"])
    f1_m5 = metrics.f1_por_classe(yte, pp["mel5"])
    sup = np.bincount(yte, minlength=21)
    tab = pd.DataFrame({"classe": nomes, "suporte": sup, "f1_piso": f1_pi, "f1_aluno": f1_al, "f1_prof_zs": f1_zs,
                        "f1_prof_melhorado5": f1_m5})
    tab["ganho_aluno_menos_piso"] = tab.f1_aluno - tab.f1_piso
    tab["aluno_menos_zs"] = tab.f1_aluno - tab.f1_prof_zs
    tab["aluno_menos_melhorado"] = tab.f1_aluno - tab.f1_prof_melhorado5
    tab.round(4).to_csv(RES_DIR / "por_classe_r5.csv", index=False)
    out["por_classe_r5"] = tab.round(4).to_dict(orient="records")

    o = tab.sort_values("ganho_aluno_menos_piso")
    fig, ax = plt.subplots(figsize=(8, 7))
    cores = ["#2ca02c" if v > 0 else "#d62728" for v in o.ganho_aluno_menos_piso]
    ax.barh(o.classe, o.ganho_aluno_menos_piso, color=cores)
    for i, (v, a, p) in enumerate(zip(o.ganho_aluno_menos_piso, o.f1_aluno, o.f1_piso)):
        ax.text(v + (0.01 if v >= 0 else -0.01), i, f"{p:.2f} → {a:.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=7)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("ΔF1 no test_public (aluno − piso), média de 3 seeds")
    lim = max(abs(o.ganho_aluno_menos_piso).max() + 0.15, 0.3)
    ax.set_xlim(-lim, lim)
    ax.set_title(f"Ganho por classe no orçamento 5 — F1 macro: piso {f1_pi.mean():.3f} → aluno {f1_al.mean():.3f}\n"
                 "(texto: F1 piso → F1 aluno)", fontsize=10)
    _salva(fig, "ganho_por_classe.png")

    o = tab.sort_values("f1_aluno", ascending=False)
    xx = np.arange(21)
    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.bar(xx - 0.27, o.f1_aluno, 0.27, label=f"aluno (média 3 seeds) — macro {f1_al.mean():.3f}", color="#1f77b4")
    ax.bar(xx, o.f1_prof_zs, 0.27, label=f"professor zero-shot — macro {f1_zs.mean():.3f}", color="#d62728")
    ax.bar(xx + 0.27, o.f1_prof_melhorado5, 0.27, label=f"professor melhorado (5) — macro {f1_m5.mean():.3f}",
           color="#ff7f0e")
    for i, (a, z, m) in enumerate(zip(o.f1_aluno, o.f1_prof_zs, o.f1_prof_melhorado5)):
        if a > max(z, m):
            ax.text(i - 0.27, a + 0.02, "★", ha="center", fontsize=9, color="#1f77b4")
    ax.set_xticks(xx, [f"{c} ({s})" for c, s in zip(o.classe, o.suporte)], rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("F1 no test_public")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.set_title("Aluno × professor por classe, orçamento 5 (★ = aluno acima dos dois professores; "
                 "parênteses = suporte no test_public)", fontsize=10)
    _salva(fig, "aluno_vs_professor.png")

    # ---------------------------------------------------------------- evidência: dia × noite, discordâncias
    def dn(p):
        return {"dia": metrics.resumo(yte[~noite], p[~noite], so_presentes=True),
                "noite": metrics.resumo(yte[noite], p[noite], so_presentes=True)}

    evid = {"n_dia": int((~noite).sum()), "n_noite": int(noite.sum()),
            "aluno_por_seed": {s: dn(p) for s, p in pred_al.items()},
            "piso_por_seed": {s: dn(p) for s, p in pred_pi.items()},
            "prof_zs": dn(pp["zs"]), "prof_melhorado5": dn(pp["mel5"])}
    for k in ("dia", "noite"):
        for nome_m, dic in (("aluno", evid["aluno_por_seed"]), ("piso", evid["piso_por_seed"])):
            evid[f"{nome_m}_{k}_f1_media"] = float(np.mean([v[k]["f1_macro"] for v in dic.values()]))
            evid[f"{nome_m}_{k}_acc_media"] = float(np.mean([v[k]["acuracia"] for v in dic.values()]))
    # acerto (recall) por classe e por dia/noite: aluno (média seeds) × professores
    rec = []
    for c in range(21):
        for k, m in (("dia", ~noite), ("noite", noite)):
            sel = m & (yte == c)
            if sel.sum() == 0:
                continue
            rec.append({"classe": nomes[c], "turno": k, "n": int(sel.sum()),
                        "recall_aluno": float(np.mean([(p[sel] == c).mean() for p in pred_al.values()])),
                        "recall_piso": float(np.mean([(p[sel] == c).mean() for p in pred_pi.values()])),
                        "recall_zs": float((pp["zs"][sel] == c).mean()),
                        "recall_mel5": float((pp["mel5"][sel] == c).mean())})
    evid["recall_por_classe_turno"] = rec
    # discordância aluno × professor zero-shot (por seed, depois média): quem acerta quando discordam
    disc = []
    for s, p in pred_al.items():
        for k, m in (("dia", ~noite), ("noite", noite), ("tudo", np.ones_like(noite))):
            d = m & (p != pp["zs"])
            disc.append({"seed": s, "turno": k, "n": int(m.sum()), "discordam": int(d.sum()),
                         "so_aluno_acerta": int((d & (p == yte)).sum()),
                         "so_prof_zs_acerta": int((d & (pp["zs"] == yte)).sum())})
    evid["discordancia_aluno_zs"] = disc
    # previsões por classe (classes "ímã" do professor)
    evid["previsoes_por_classe"] = {"suporte": dict(zip(nomes, sup.tolist())),
                                    "prof_zs": dict(zip(nomes, np.bincount(pp["zs"], minlength=21).tolist())),
                                    "prof_melhorado5": dict(zip(nomes, np.bincount(pp["mel5"], minlength=21).tolist())),
                                    "aluno_media": dict(zip(nomes, np.mean([np.bincount(p, minlength=21)
                                                                            for p in pred_al.values()], 0).round(1).tolist()))}
    # concordância do aluno com o professor do qual aprendeu (fonte dos pseudo-rótulos)
    fonte = "mel5" if met["cfg"]["fonte"] == "melhorado" else "zs"
    evid["concordancia_aluno_com_seu_professor_test"] = float(np.mean([(p == pp[fonte]).mean() for p in pred_al.values()]))
    # diagnóstico do filtro (variante B): o zero-shot rodado no test_public como se não tivesse rótulo.
    # Só diagnóstico: o τ foi escolhido no val. Precisão = acurácia das previsões que passam no filtro.
    tau = abl["tau_escolhido_B"]
    passa = pp["zs_conf"] >= tau
    filtro = {"tau": tau, "cobertura": float(passa.mean()),
              "precisao_passa": float((pp["zs"][passa] == yte[passa]).mean()),
              "precisao_nao_passa": float((pp["zs"][~passa] == yte[~passa]).mean())}
    for k, m in (("dia", ~noite), ("noite", noite)):
        filtro[f"cobertura_{k}"] = float(passa[m].mean())
        filtro[f"precisao_passa_{k}"] = float((pp["zs"][passa & m] == yte[passa & m]).mean())
    filtro["por_classe_prevista"] = {nomes[c]: {"n_prev": int((pp["zs"] == c).sum()),
                                                "n_prev_passa": int(((pp["zs"] == c) & passa).sum()),
                                                "precisao_todas": float((yte[pp["zs"] == c] == c).mean()) if (pp["zs"] == c).any() else None,
                                                "precisao_passa": float((yte[(pp["zs"] == c) & passa] == c).mean())
                                                if ((pp["zs"] == c) & passa).any() else None}
                                     for c in range(21)}
    evid["filtro_zs_test"] = filtro
    out["evidencia"] = evid

    # figura: F1 macro dia × noite, piso / aluno / professores
    fig, ax = plt.subplots(figsize=(7, 4))
    grupos = [("piso (b)", evid["piso_dia_f1_media"], evid["piso_noite_f1_media"], "#7f7f7f"),
              ("aluno", evid["aluno_dia_f1_media"], evid["aluno_noite_f1_media"], "#1f77b4"),
              ("prof. zero-shot", evid["prof_zs"]["dia"]["f1_macro"], evid["prof_zs"]["noite"]["f1_macro"], "#d62728"),
              ("prof. melhorado (5)", evid["prof_melhorado5"]["dia"]["f1_macro"],
               evid["prof_melhorado5"]["noite"]["f1_macro"], "#ff7f0e")]
    for i, (rot, d_, n_, cor) in enumerate(grupos):
        ax.bar(i - 0.2, d_, 0.4, color=cor, alpha=0.55)
        ax.bar(i + 0.2, n_, 0.4, color=cor, hatch="//", edgecolor="black")
        ax.text(i - 0.2, d_ + 0.01, f"{d_:.3f}", ha="center", fontsize=8)
        ax.text(i + 0.2, n_ + 0.01, f"{n_:.3f}", ha="center", fontsize=8)
    ax.set_xticks(range(4), [g[0] for g in grupos])
    ax.set_ylabel("F1 macro (só classes presentes)")
    ax.set_ylim(0, 0.85)
    ax.bar([], [], color="gray", alpha=0.55, label=f"dia (n={evid['n_dia']})")
    ax.bar([], [], color="gray", hatch="//", edgecolor="black", label=f"noite/IR (n={evid['n_noite']})")
    ax.legend(fontsize=8)
    ax.set_title("Dia × noite no test_public, orçamento 5 (aluno e piso: média de 3 seeds)", fontsize=10)
    _salva(fig, "aluno_dia_noite.png")

    with open(RES_DIR / "aluno_analise.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=float)
    print(json.dumps({k: out[k] for k in ("conferencia_professor", "reprodutibilidade_piso_r5", "curva")},
                     indent=1, default=float))


if __name__ == "__main__":
    main()
