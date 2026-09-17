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


def build_label_free():
    data = load_artifact("label_free.json")["cells"]
    lines = [
        r"\small",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{@{}llrr@{}}",
        r"\toprule",
        r"model & dataset & P(IK) label-free & SAPLMA label-free \\",
        r"\midrule",
    ]
    for mi, m in enumerate(MODEL_ORDER):
        for si, s in enumerate(SET_ORDER):
            cell = data[m][s]
            pik = fnum(cell["pik"], 3)
            sap = fnum(cell["saplma"], 3)
            head = r"\multirow{3}{*}{" + MODEL_LONG[m] + "}" if si == 0 else ""
            lines.append(f"{head} & {SET_SHORT[s]} & {pik} & {sap} " + r"\\")
        if mi != len(MODEL_ORDER) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
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
