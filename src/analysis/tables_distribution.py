from src.analysis.latexlib import (
    MODEL_ORDER,
    SET_ORDER,
    MODEL_LONG,
    SET_SHORT,
    load_artifact,
    write_output,
    fnum,
)


def ratio_str(v):
    if v >= 10:
        return str(int(round(v)))
    return fnum(v, 1)


def build_kl_constraint():
    data = load_artifact("kl_constraint.json")["cells"]
    lines = [
        r"\small",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{@{}llrrr@{}}",
        r"\toprule",
        r"model & dataset & clean AUROC & attacked, KL$\le$0.2 & attacked, KL$\le$0.05 \\",
        r"\midrule",
    ]
    for mi, m in enumerate(MODEL_ORDER):
        for si, s in enumerate(SET_ORDER):
            cell = data[m][s]
            if si == 0:
                head = r"\multirow{3}{*}{" + MODEL_LONG[m] + "}"
            else:
                head = ""
            row = (
                f"{head} & {SET_SHORT[s]} & {fnum(cell['clean'], 3)} & "
                f"{fnum(cell['att_02'], 3)} & {fnum(cell['att_005'], 3)} "
            )
            lines.append(row + r"\\")
        if mi != len(MODEL_ORDER) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return write_output("tables/appendices/kl_constraint.tex", lines)


def build_kl_surviving():
    art = load_artifact("kl_surviving.json")
    data = art["cells"]
    mean = art["mean"]
    lines = [
        r"\footnotesize",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{@{}llrrrrrr@{}}",
        r"\toprule",
        r"model & dataset & clean & unb. & att@0.2 & att@0.05 & surv@0.2 & surv@0.05 \\",
        r"\midrule",
    ]
    for mi, m in enumerate(MODEL_ORDER):
        for si, s in enumerate(SET_ORDER):
            cell = data[m][s]
            if si == 0:
                head = r"\multirow{3}{*}{" + MODEL_LONG[m] + "}"
            else:
                head = ""
            row = (
                f"{head} & {SET_SHORT[s]} & {fnum(cell['clean'], 3)} & "
                f"{fnum(cell['unb'], 3)} & {fnum(cell['att_02'], 3)} & "
                f"{fnum(cell['att_005'], 3)} & {fnum(cell['surv_02'], 3)} & "
                f"{fnum(cell['surv_005'], 3)} "
            )
            lines.append(row + r"\\")
        lines.append(r"\midrule")
    meanrow = (
        r"\textbf{mean} & & "
        f"{fnum(mean['clean'], 3)} & {fnum(mean['unb'], 3)} & "
        f"{fnum(mean['att_02'], 3)} & {fnum(mean['att_005'], 3)} & "
        r"\textbf{" + fnum(mean["surv_02"], 3) + r"} & \textbf{"
        + fnum(mean["surv_005"], 3) + r"} \\"
    )
    lines.append(meanrow)
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        "",
        r"\vspace{3pt}",
        r"\noindent{\footnotesize surv = fraction of the full unbounded inversion the KL-bounded attack retains; unb.\ is the unbounded attacked AUROC and att@$\delta$ the attacked AUROC under the bound $\delta$.}",
    ]
    return write_output("tables/appendices/kl_surviving.tex", lines)


def build_kl_behavioral():
    art = load_artifact("kl_behavioral.json")
    models = art["models"]
    deltas = art["deltas"]
    lines = [
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4.5pt}",
        r"\begin{tabular}{@{}llrrrrrrrr@{}}",
        r"\toprule",
        r"model & $\delta$ & top-1 & top-5 & top-10 & TV(pos0) & samp.\ TV & real.\ KL & benign KL & ratio \\",
        r"\midrule",
    ]
    for mi, m in enumerate(MODEL_ORDER):
        entry = models[m]
        for di, d in enumerate(deltas):
            met = entry[d]
            head = r"\multirow{2}{*}{" + MODEL_LONG[m] + "}" if di == 0 else ""
            benign = fnum(entry["benign_kl_mean"], 4)
            if di == 0:
                ratio = ratio_str(entry["ratio"]) + r"$\times$"
            else:
                ratio = ratio_str(float(d) / entry["benign_kl_mean"]) + r"$\times$"
            cells = [
                head,
                d,
                fnum(met["t1"], 3),
                fnum(met["t5"], 3),
                fnum(met["t10"], 3),
                fnum(met["tv"], 3),
                fnum(met["stv"], 3),
                fnum(met["klm"], 3),
                benign,
                ratio,
            ]
            lines.append(" & ".join(cells) + r" \\")
        if mi != len(MODEL_ORDER) - 1:
            lines.append(r"\midrule")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        "",
        r"\vspace{3pt}",
        r"\noindent{\footnotesize top-$k$ = fraction of the top-$k$ next-token set preserved relative to clean; TV(pos0) and samp.\ TV are total-variation distances at the first answer position (argmax and sampled); real.\ KL is the mean realized $\mathrm{KL}(p_{\mathrm{clean}}\|p_{\mathrm{adv}})$; ratio is the KL bound $\delta$ over the benign single-greylevel-dither KL. The clean-versus-clean sampled-TV noise floor is $\approx$0.07.}",
    ]
    return write_output("tables/appendices/kl_behavioral.tex", lines)


def build_gemma_sensitivity():
    models = load_artifact("gemma_sensitivity.json")["models"]
    lines = [
        r"\small",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r"model & dither KL & Gaussian KL & normalized dither & clean entropy & logit gap \\",
        r"\midrule",
    ]
    for m in MODEL_ORDER:
        e = models[m]
        cols = [
            fnum(e["dither_kl"], 4),
            fnum(e["gauss_kl"], 4),
            fnum(e["nd_rms"], 5),
            fnum(e["clean_entropy"], 3),
            fnum(e["logit_gap"], 3),
        ]
        if m == "gemma3_12b":
            name = r"\textbf{" + MODEL_LONG[m] + "}"
            cols = [r"\textbf{" + c + "}" for c in cols]
        else:
            name = MODEL_LONG[m]
        lines.append(name + " & " + " & ".join(cols) + r" \\")
    nd = models
    footnote = (
        r"\noindent{\footnotesize Normalized dither is the RMS of the one-greylevel perturbation in the model's own input units: Gemma "
        + fnum(nd["gemma3_12b"]["nd_rms"], 5)
        + r", second only to Molmo's "
        + fnum(nd["molmo2_4b"]["nd_rms"], 5)
        + r" and well below InternVL's "
        + fnum(nd["internvl3_5_2b"]["nd_rms"], 5)
        + r".}"
    )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        "",
        r"\vspace{3pt}",
        footnote,
    ]
    return write_output("tables/appendices/gemma_sensitivity.tex", lines)


BUILDERS = {
    "kl_constraint": build_kl_constraint,
    "kl_surviving": build_kl_surviving,
    "kl_behavioral": build_kl_behavioral,
    "gemma_sensitivity": build_gemma_sensitivity,
}
