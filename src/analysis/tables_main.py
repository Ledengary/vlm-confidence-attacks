from src.analysis.latexlib import MODEL_ORDER, SET_ORDER, MODEL_LONG, SET_SHORT, load_artifact, write_output, fnum
from src.analysis.templates import ladder_full_footnote, ladder_parity_footnote

PARITY_ORDER = ["gemma", "llava", "internvl", "molmo"]
PARITY_LONG = {
    "gemma": "Gemma-3-12B",
    "llava": "LLaVA-OV-7B",
    "internvl": "InternVL3.5-2B",
    "molmo": "Molmo2-4B",
}
PARITY_SHORT = {"gemma": "Gemma", "llava": "LLaVA", "internvl": "InternVL", "molmo": "Molmo"}

LADDER_BV = {"internvl": "InternVL", "molmo": "Molmo", "llava": "LLaVA", "gemma": "Gemma"}
LADDER_SET = {"gqa": "GQA", "vqa": "VQAv2", "pope": "POPE"}
LADDER_COLS = ["clean", "transfer", "hsj", "spsa", "c2"]


def _ladder_note(r):
    parts = []
    if r["verifier"] == "molmo" and r["dataset"] == "pope":
        parts.append("near-chance clean")
    if r["verifier"] == "molmo":
        parts.append("parity")
    if r["base"] == "gemma" and r["verifier"] == "molmo":
        parts.append("WB under-powered")
    if r["verifier"] == "internvl":
        parts.append(r"score $n{=}100$")
    return "; ".join(parts)


def _ladder_means(rows):
    return {c: sum(r[c] for r in rows) / len(rows) for c in LADDER_COLS}


def _ladder_aggregates(rows):
    excl = [r for r in rows if not (r["base"] == "gemma" and r["verifier"] == "molmo")]
    bitexact = [r for r in rows if r["verifier"] != "molmo"]
    return _ladder_means(rows), _ladder_means(excl), _ladder_means(bitexact), bitexact

CORE_HEADER = [
    r"\toprule",
    r" & & Clean & \multicolumn{3}{c}{Attacked AUROC} & \multicolumn{5}{c}{Decomposition} \\",
    r"\cmidrule(lr){4-6}\cmidrule(lr){7-11}",
    r"Model & Dataset & (median) & WC-oracle & IQR & label-free & $w$ & $A_{\mathrm{same}}$ & $A_{\mathrm{cross}}$ & $A_{\mathrm{adv}}$ & $\rho$ \\",
    r"\midrule",
]


def build_core_decomp():
    cells = load_artifact("core_decomp.json")["cells"]
    lf = load_artifact("label_free.json")["cells"]
    survivor = max(
        ((m, s) for m in MODEL_ORDER for s in SET_ORDER),
        key=lambda k: cells[k[0]][k[1]]["a_adv"],
    )
    out = [r"\small", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{llcccc@{\hspace{8pt}}ccccc}"]
    out += CORE_HEADER
    for mi, m in enumerate(MODEL_ORDER):
        for si, s in enumerate(SET_ORDER):
            c = cells[m][s]
            lfm = (lf[m][s]["pik"] + lf[m][s]["saplma"]) / 2.0
            model_cell = r"\multirow{3}{*}{" + MODEL_LONG[m] + "}" if si == 0 else ""
            ds = SET_SHORT[s]
            if (m, s) == survivor:
                ds = ds + r"$^{\dagger}$\,"
            cellvals = [
                model_cell, ds,
                fnum(c["clean_median"], 3), fnum(c["oracle_median"], 3), fnum(c["iqr"], 3), fnum(lfm, 3),
                fnum(c["w"], 3), fnum(c["a_same"], 3), fnum(c["a_cross"], 3), fnum(c["a_adv"], 3), fnum(c["rho"], 3),
            ]
            out.append(" & ".join(cellvals) + r" \\")
        if mi != len(MODEL_ORDER) - 1:
            out.append(r"\midrule")
    out += [r"\bottomrule", r"\end{tabular}"]
    return write_output("tables/core_decomp.tex", out)


def _ci(auroc, ci):
    return fnum(auroc, 3) + r"~{\tiny(" + fnum(ci[0], 2) + "," + fnum(ci[1], 2) + ")}"


def build_ladder_full():
    rows = load_artifact("ladder_full.json")["rows"]
    m12, mexcl, mbit, bitexact = _ladder_aggregates(rows)
    out = [r"\footnotesize", r"\setlength{\tabcolsep}{5pt}", r"\resizebox{\textwidth}{!}{%",
           r"\begin{tabular}{@{}llll rrr l@{}}", r"\toprule",
           r"base$\to$verifier & set & clean [95\% CI] & transfer [95\% CI] & decision & score & white-box & notes \\",
           r" & & & & HSJ & SPSA (600 queries) &  & \\", r"\midrule"]
    for r in rows:
        bv = LADDER_BV[r["base"]] + r"$\to$" + LADDER_BV[r["verifier"]]
        cells = [
            bv, LADDER_SET[r["dataset"]], _ci(r["clean"], r["clean_ci"]), _ci(r["transfer"], r["transfer_ci"]),
            fnum(r["hsj"], 3), fnum(r["spsa"], 3), fnum(r["c2"], 3), _ladder_note(r),
        ]
        out.append(" & ".join(cells) + r" \\")
    out.append(r"\midrule")

    def mrow(label, mm):
        vals = " & ".join(r"\textbf{" + fnum(mm[c], 3) + "}" for c in LADDER_COLS)
        return r"\textbf{" + label + r"} & & " + vals + r" & \\"

    out.append(mrow("mean (12 pairs)", m12))
    out.append(mrow(r"mean, excl.\ under-powered white-box", mexcl))
    out.append(mrow("mean, bit-exact verifier (9)", mbit))
    out += [r"\bottomrule", r"\end{tabular}}"]
    wb_bit = [r["c2"] for r in bitexact]
    foot = ladder_full_footnote(fnum(mexcl["c2"], 3), fnum(mbit["c2"], 3), fnum(min(wb_bit), 2), fnum(max(wb_bit), 2))
    out.append(foot)
    return write_output("tables/appendices/ladder_full.tex", out)


def build_ladder_main():
    rows = load_artifact("ladder_full.json")["rows"]
    m12, mexcl, mbit, _ = _ladder_aggregates(rows)
    out = [r"\small", r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrr@{}}", r"\toprule",
           r" & clean & no access & decision-only & score-only & white-box \\", r"\midrule"]

    def row(label, mm):
        return label + " & " + " & ".join(fnum(mm[c], 3) for c in LADDER_COLS) + r" \\"

    out.append(row("All 12 pairs", m12))
    out.append(row("Excluding under-powered cell", mexcl))
    out.append(row("Bit-exact verifiers (9)", mbit))
    out += [r"\bottomrule", r"\end{tabular*}"]
    return write_output("tables/ladder_main.tex", out)


def build_ladder_parity():
    d = load_artifact("ladder_parity.json")
    fam = d["families"]
    out = [r"\small", r"\setlength{\tabcolsep}{6pt}", r"\begin{tabular}{@{}lrcl@{}}", r"\toprule",
           r"verifier family & max parity gap & bit-exact & cells affected \\", r"\midrule"]
    for v in PARITY_ORDER:
        mg = fam[v]["max_gap"]
        bit = mg == 0.0
        if bit:
            affected = "none"
        else:
            affected = f"{fam[v]['n']} {PARITY_SHORT[v]}-verifier pairs (max {fnum(mg, 3)})"
        out.append(" & ".join([PARITY_LONG[v], fnum(mg, 3), "yes" if bit else "no", affected]) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    out.append(ladder_parity_footnote(fnum(d["molmo_mean_gap"], 3)))
    return write_output("tables/appendices/ladder_parity.tex", out)


BUILDERS = {
    "core_decomp": build_core_decomp,
    "ladder_full": build_ladder_full,
    "ladder_main": build_ladder_main,
    "ladder_parity": build_ladder_parity,
}
