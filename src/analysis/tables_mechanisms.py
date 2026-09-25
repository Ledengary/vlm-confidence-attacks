from src.analysis.latexlib import (
    load_artifact,
    write_output,
    fnum,
)


def _s0(s):
    return s[1:] if s.startswith("0") else s


def _cic(pt, ci):
    return fnum(pt, 3) + r"~{\tiny($" + _s0(fnum(ci[0], 3)) + r"$,$" + _s0(fnum(ci[1], 3)) + r"$)}"


def _range(lo, hi, nd):
    a = fnum(lo, nd)
    b = fnum(hi, nd)
    if a == b:
        return a
    return a + "--" + b


def build_within_answer():
    data = load_artifact("within_answer.json")
    est = data["estimators"]
    lines = [
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{adjustbox}{max width=\textwidth}",
        r"\begin{tabular}{@{}lrrlrl@{}}",
        r"\toprule",
        r"estimator & clean mean & clean median & clean per-cell range & adv mean & adv per-cell range \\",
        r"\midrule",
    ]
    for i, e in enumerate(est):
        adv_mean = fnum(e["adv_mean"], 3)
        if e["dagger"]:
            adv_mean = adv_mean + r"$^{\dagger}$"
        row = (
            f"{e['label']} & {fnum(e['clean_mean'], 3)} & {fnum(e['clean_median'], 3)} & "
            f"{_range(e['clean_min'], e['clean_max'], 3)} & {adv_mean} & "
            f"{_range(e['adv_min'], e['adv_max'], 3)} " + r"\\"
        )
        lines.append(row)
        if e["label"] == "CCPS-D":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", ""]
    lines.append(r"\vspace{4pt}")
    lines.append(r"{\footnotesize\itshape Per-model attacked within-answer AUROC (mean over 3 cells $\times$ 5 deployed channels):}\\[2pt]")
    lines += [
        r"\footnotesize",
        r"\begin{tabular}{@{}lc@{}}",
        r"\toprule",
        r"model & attacked within-answer AUROC \\",
        r"\midrule",
    ]
    for pm in data["per_model"]:
        cell = fnum(pm["auroc"], 3)
        if pm["note"]:
            cell = cell + " " + pm["note"]
        lines.append(f"{pm['label']} & {cell} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    d = data["coupled_dagger"]
    lines.append(r"\vspace{2pt}")
    lines.append(
        r"{\footnotesize $^{\dagger}$COUPLED's adversarial mean (" + fnum(d["twelve_cell"], 3)
        + r") is the twelve-cell mean and includes the survivor cell, Molmo/POPE (adversarial within-answer "
        + fnum(d["molmo_pope"], 3) + r"); the eleven-cell mean excluding it is " + fnum(d["eleven_cell"], 3) + r".}"
    )
    return write_output("tables/appendices/within_answer.tex", lines)


def build_ceiling_floor():
    cells = load_artifact("ceiling_floor.json")["cells"]
    from src.analysis.latexlib import MODEL_ORDER, SET_SHORT, MODEL_LONG
    lines = [
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{@{}llrrrrrrrc@{}}",
        r"\toprule",
        r"model & set & cv5 & held-out & operative & $\rho$ & $w$ & $c^{\star}$ & ceiling & holds \\",
        r" &  & (cross-fit) & (GQA) & floor &  &  &  & $c^{\star}\!+\!(1\!-\!\rho)w$ &  \\",
        r"\midrule",
    ]
    by_model = {m: [c for c in cells if c["model"] == m] for m in MODEL_ORDER}
    for mi, m in enumerate(MODEL_ORDER):
        group = by_model[m]
        for si, c in enumerate(group):
            head = r"\multirow{3}{*}{" + MODEL_LONG[m] + "}" if si == 0 else ""
            heldout = "n/a" if c["heldout"] is None else fnum(c["heldout"], 3)
            holds = "yes" if c["holds"] else "no"
            row = (
                f"{head} & {SET_SHORT[c['set']]} & {fnum(c['cv5'], 3)} & {heldout} & "
                f"{fnum(c['operative'], 3)} & {fnum(c['rho'], 3)} & {fnum(c['w'], 3)} & "
                f"{fnum(c['cstar'], 3)} & {fnum(c['ceiling'], 3)} & {holds} " + r"\\"
            )
            lines.append(row)
        if mi != len(MODEL_ORDER) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return write_output("tables/appendices/ceiling_floor.tex", lines)


def build_long_answer_full():
    data = load_artifact("long_answer_full.json")
    models = data["models"]
    lines = [
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{@{}llrrc@{}}",
        r"\toprule",
        r"model & readout & clean AUROC & attacked AUROC & regime \\",
        r"\midrule",
    ]
    for mi, mo in enumerate(models):
        for ri, ro in enumerate(mo["readouts"]):
            if ri == 0:
                head = r"\multirow{3}{*}{" + mo["label"] + "}"
                tail = r"\multirow{3}{*}{" + mo["regime"] + r"} \\"
            else:
                head = ""
                tail = r" \\"
            lines.append(
                f"{head} & {ro['label']} & {_cic(ro['clean'], ro['clean_ci'])} & {_cic(ro['attacked'], ro['attacked_ci'])} & {tail}"
            )
        if mi != len(models) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    lines.append(r"\vspace{5pt}")
    lines.append(r"{\footnotesize\itshape Per-model diagnostics (signal location, answer length, byte-identity feasibility):}\\[2pt]")
    lines += [
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4.5pt}",
        r"\begin{adjustbox}{max width=\textwidth}",
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r"model & final-locus clean & reasoning-prompt median (tok) & native median (tok) & feasible-step rate & rejects/item-dir \\",
        r"\midrule",
    ]
    for mo in models:
        native = fnum(mo["native_median"], 1)
        if mo["native_long"]:
            native = native + r"$^{\ast}$"
        lines.append(
            f"{mo['label']} & {fnum(mo['final_locus'], 3)} & {fnum(mo['cot_median'], 1)} & "
            f"{native} & {fnum(mo['feasible_rate'], 4)} & {fnum(mo['rejects'], 2)} " + r"\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", ""]
    fn = data["footnote"]
    lines.append(r"\vspace{2pt}")
    lines.append(
        r"{\footnotesize $^{\ast}$Molmo is the only natively long model (neutral-prompt median "
        + fnum(fn["molmo_native_median"], 1) + r" tokens, frac$\ge$20 $=$ " + fnum(fn["molmo_frac_ge20"], 2)
        + r"); the others emit 2 to 7 tokens without a chain-of-thought prompt. The resisters' attacked floor, the minimum over P(IK) and SAPLMA on InternVL and LLaVA, is "
        + fnum(fn["resister_floor"], 3)
        + r" (LLaVA SAPLMA). Feasible-step rate is the median over TokProb item-directions of feasible iterates divided by $62$ (two restarts times $31$ checks), equal to $2/62$; the per-model means range from $0.04$ to $0.21$. rejects/item-dir is the mean number of fresh-decode rejections per item-direction (8 to 16).}"
    )
    return write_output("tables/appendices/long_answer_full.tex", lines)


def build_answer_coupled_subspace():
    models = load_artifact("answer_coupled_subspace.json")["models"]
    lines = [
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}llrrrrr@{}}",
        r"\toprule",
        r"model & hidden dim & $k$ & coupled energy & free energy & random ctrl & var.\ frac \\",
        r"\midrule",
    ]
    for mi, mo in enumerate(models):
        for ri, r in enumerate(mo["rows"]):
            if ri == 0:
                head = r"\multirow{3}{*}{" + mo["label"] + r"} & \multirow{3}{*}{" + str(mo["dim"]) + "}"
            else:
                head = r" &"
            row = (
                f"{head} & {r['k']} & {fnum(r['coupled'], 2)}\\% & {fnum(r['free'], 2)}\\% & "
                f"{fnum(r['random'], 2)}\\% & {fnum(r['varfrac'], 2)}\\% " + r"\\"
            )
            lines.append(row)
        if mi != len(models) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return write_output("tables/appendices/answer_coupled_subspace.tex", lines)


BUILDERS = {
    "within_answer": build_within_answer,
    "ceiling_floor": build_ceiling_floor,
    "long_answer_full": build_long_answer_full,
    "answer_coupled_subspace": build_answer_coupled_subspace,
}
