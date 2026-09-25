from src.analysis.latexlib import (
    MODEL_ORDER,
    SET_ORDER,
    MODEL_LONG,
    SET_SHORT,
    load_artifact,
    write_output,
    fnum,
)

READOUT_LABEL = {"pik": "P(IK)", "coupled": "COUPLED", "tokprob": "TokProb"}


def _strip0(s):
    return s[1:] if s.startswith("0") else s


LABELFREE_ORDER = ["tokprob", "pik", "saplma", "ccps", "ccpsd", "full", "coupled"]
LABELFREE_NAME = {"tokprob": "TokProb", "pik": "P(IK)", "saplma": "SAPLMA", "ccps": "CCPS",
                  "ccpsd": "CCPS-D", "full": "FULL", "coupled": "COUPLED"}
LABELFREE_CONSTRUCTION = {"tokprob": "union", "pik": "diag", "saplma": "diag", "ccps": "surrogate",
                          "ccpsd": "diag", "full": "diag", "coupled": "union"}


def build_label_free():
    data = load_artifact("label_free.json")["cells"]
    present = [e for e in LABELFREE_ORDER if e in data[MODEL_ORDER[0]][SET_ORDER[0]]]
    header = "model & dataset & " + " & ".join(LABELFREE_NAME[e] for e in present) + r" \\"
    lines = [
        r"\small",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{@{}ll" + "r" * len(present) + r"@{}}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    n_below = 0
    n_total = 0
    for mi, m in enumerate(MODEL_ORDER):
        for si, s in enumerate(SET_ORDER):
            cell = data[m][s]
            vals = [cell[e] for e in present]
            n_total += len(vals)
            n_below += sum(1 for v in vals if v < 0.5)
            head = r"\multirow{3}{*}{" + MODEL_LONG[m] + "}" if si == 0 else ""
            lines.append(f"{head} & {SET_SHORT[s]} & " + " & ".join(fnum(v, 3) for v in vals) + r" \\")
        if mi != len(MODEL_ORDER) - 1:
            lines.append(r"\midrule")
    count = f"All {n_below} of {n_total}" if n_below == n_total else f"{n_below} of {n_total}"
    diag = [LABELFREE_NAME[e] for e in present if LABELFREE_CONSTRUCTION[e] == "diag"]
    union = [LABELFREE_NAME[e] for e in present if LABELFREE_CONSTRUCTION[e] == "union"]
    surr = [LABELFREE_NAME[e] for e in present if LABELFREE_CONSTRUCTION[e] == "surrogate"]
    segs = []
    if diag:
        segs.append(", ".join(diag) + " use each readout's own aimed endpoints")
    if union:
        segs.append(", ".join(union) + " use the witnessed union over the aimed objectives")
    if surr:
        segs.append(", ".join(surr) + " uses the CCPS-D-aimed surrogate")
    run = (r"Endpoints: " + "; ".join(segs)
           + r". The own-aimed and surrogate endpoints are the 30-step, 2-restart search "
             r"(Table~\ref{tab:app-hyper}); the TokProb and COUPLED union spans all five aimed "
             r"objectives and includes COUPLED's own 60-step, 3-restart endpoints.")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        "",
        r"\vspace{3pt}",
        r"\noindent{\footnotesize " + count + r" label-free profiles fall below chance ($0.5$). " + run + r"}",
    ]
    return write_output("tables/appendices/label_free.tex", lines)


def _ci_cell(e):
    pt = fnum(e["a"], 3)
    lo = _strip0(fnum(e["ci"][0], 3))
    hi = _strip0(fnum(e["ci"][1], 3))
    return pt + r"~{\tiny($" + lo + r"$,$" + hi + r"$)}"


def build_multiseed_attacked():
    data = load_artifact("multiseed_attacked.json")["cells"]
    lines = [
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}lll r lll rr@{}}",
        r"\toprule",
        r"model & set & readout & clean & seed 23 (n=1000) & seed 24 (n=120) & seed 25 (n=120) & mean & sd \\",
        r"\midrule",
    ]
    for mi, m in enumerate(MODEL_ORDER):
        for si, s in enumerate(SET_ORDER):
            cell = data[m][s]
            head = r"\multirow{3}{*}{" + MODEL_LONG[m] + "}" if si == 0 else ""
            setlbl = SET_SHORT[s] + (r"$^{\dagger}$" if cell["survivor"] else "")
            readout = READOUT_LABEL[cell["readout"]]
            clean = fnum(cell["clean"], 3)
            s23 = _ci_cell(cell["seeds"]["23"])
            s24 = _ci_cell(cell["seeds"]["24"])
            s25 = _ci_cell(cell["seeds"]["25"])
            mean = fnum(cell["mean"], 3)
            sd = fnum(cell["sd"], 3)
            lines.append(
                f"{head} & {setlbl} & {readout} & {clean} & {s23} & {s24} & {s25} & {mean} & {sd} " + r"\\"
            )
        if mi != len(MODEL_ORDER) - 1:
            lines.append(r"\midrule")
    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        "",
        r"\vspace{3pt}",
        r"\noindent{\footnotesize $^{\dagger}$Sole above-chance survivor. Seeds 24 and 25 reseed both the item pool and the attack randomness, so their wider 95\% CIs reflect the smaller $n{=}120$ draw. The across-seed mean and sd are over the three seed point estimates.}",
    ]
    return write_output("tables/appendices/multiseed_attacked.tex", lines)


def build_multiseed_survivor():
    d = load_artifact("multiseed_survivor.json")
    rows = [
        (r"$w$ (same-answer weight)", "w"),
        (r"$A_{\mathrm{same}}$", "a_same"),
        (r"$A_{\mathrm{cross}}$", "a_cross"),
    ]
    lines = [
        r"\small",
        r"\setlength{\tabcolsep}{8pt}",
        r"\begin{tabular}{@{}lrrr@{}}",
        r"\toprule",
        r"quantity & seed 23 & seed 24 & seed 25 \\",
        r"\midrule",
    ]
    for label, key in rows:
        lines.append(
            f"{label} & {fnum(d['seed23'][key], 3)} & {fnum(d['seed24'][key], 3)} & {fnum(d['seed25'][key], 3)} " + r"\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"",
        r"\vspace{3pt}",
        r"\noindent{\footnotesize Same-answer pairs are about half the cell on every seed ($w\approx0.52$), which is why a same-answer term above 0.5 is enough to hold the binding attacked AUROC marginally above chance.}",
    ]
    return write_output("tables/appendices/multiseed_survivor.tex", lines)


BUILDERS = {
    "label_free": build_label_free,
    "multiseed_attacked": build_multiseed_attacked,
    "multiseed_survivor": build_multiseed_survivor,
}
