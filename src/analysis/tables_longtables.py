from src.analysis.latexlib import MODEL_SHORT, MODEL_LONG, SET_SHORT, load_artifact, write_output, fnum
from src.analysis.templates import (
    certificate_caption,
    certificate_footer,
    decomp_full_caption,
    phenomenon_caption,
    phenomenon_footer,
)

CH_DISPLAY = {
    "tokprob": "TokProb",
    "pik": "P(IK)",
    "saplma": "SAPLMA",
    "ccps": "CCPS",
    "ccpsd": "CCPS-D",
    "FULL": "FULL",
    "COUPLED": "COUPLED",
}

CERT_COLSPEC = [
    r"\begin{longtable}{",
    r">{\footnotesize}l",
    r">{\footnotesize}l",
    r">{\footnotesize}l",
    r">{\footnotesize}l",
    r">{\footnotesize}r",
    r">{\footnotesize}r",
    r">{\footnotesize}r",
    r">{\footnotesize}r",
    r">{\footnotesize}r",
    r">{\footnotesize}r",
    r">{\footnotesize}c",
    r"}",
]

CERT_HEADER = r"estimator & model & set & $\gamma^{\ast}$ [95\% CI] & clean sep. & $G(0)$ & disp. & max & $n$ feas. & frac$>\gamma^{\ast}$ & refuted \\"


def _is_na(row):
    return row["gamma_grid"] == 0.0


def build_certificate():
    rows = load_artifact("certificate.json")["rows"]
    n_total = len(rows)
    na_rows = [r for r in rows if _is_na(r)]
    n_ref = n_total - len(na_rows)
    na = na_rows[0]
    cap = certificate_caption(
        n_ref, n_total,
        CH_DISPLAY[na["channel"]], MODEL_SHORT[na["model"]], SET_SHORT[na["set"]],
        fnum(na["g0"], 3), fnum(na["floor"], 3),
    )
    foot = certificate_footer(n_ref, n_total, CH_DISPLAY[na["channel"]], MODEL_SHORT[na["model"]], SET_SHORT[na["set"]])
    out = [r"\setlength{\LTcapwidth}{\textwidth}", r"\setlength{\tabcolsep}{3pt}"]
    out += CERT_COLSPEC
    out += [cap, r"\label{tab:app-certificate}\\", r"\toprule", CERT_HEADER, r"\midrule", r"\endfirsthead"]
    out += [
        r"\multicolumn{11}{@{}l}{\emph{Table~\ref{tab:app-certificate} continued from previous page}}\\",
        r"\toprule", CERT_HEADER, r"\midrule", r"\endhead",
        r"\midrule", r"\multicolumn{11}{r@{}}{\emph{continued on next page}}\\", r"\endfoot",
        r"\bottomrule",
    ]
    out += foot
    out += [r"\endlastfoot"]
    prev_ch = None
    prev_model = None
    for r in rows:
        if prev_ch is not None and r["channel"] != prev_ch:
            out.append(r"\midrule")
            prev_model = None
        elif prev_model is not None and r["model"] != prev_model:
            out.append(r"\addlinespace")
        est = CH_DISPLAY[r["channel"]] if r["channel"] != prev_ch else ""
        mod = MODEL_SHORT[r["model"]] if r["model"] != prev_model else ""
        if _is_na(r):
            gamma = "n/a"
            refuted = "n/a"
            frac = "n/a"
        else:
            lo, hi = r["ci"]
            gamma = fnum(r["gamma_exact"], 3) + r"~{\scriptsize[" + fnum(lo, 3) + "," + fnum(hi, 3) + "]}"
            refuted = "yes"
            frac = fnum(r["frac"], 3)
        cells = [
            est, mod, SET_SHORT[r["set"]], gamma,
            fnum(r["sep"], 3), fnum(r["g0"], 3), fnum(r["disp"], 3), fnum(r["wmax"], 3),
            str(r["nfeas"]), frac, refuted,
        ]
        out.append(" & ".join(cells) + r" \\")
        prev_ch = r["channel"]
        prev_model = r["model"]
    out.append(r"\end{longtable}")
    return write_output("tables/appendices/certificate.tex", out)


DECOMP_COLSPEC = [
    r"\begin{longtable}{",
    r">{\small}l",
    r">{\small}l",
    r">{\small}l",
    r">{\small}r",
    r">{\small}r",
    r">{\small}r",
    r">{\small}r",
    r">{\small}r",
    r">{\small}r",
    r"}",
]

DECOMP_HEADER = r"Model & Dataset & Estimator & $w$ & $A_{\mathrm{same}}$ & $A_{\mathrm{cross}}$ & $A_{\mathrm{adv}}$ & $\rho$ & recon.\ err. \\"


def build_decomp_full():
    rows = load_artifact("decomp_full.json")["rows"]
    for r in rows:
        r["recon_err"] = abs(r["w"] * r["a_same"] + (1.0 - r["w"]) * r["a_cross"] - r["a_adv"])
    max_re = max(r["recon_err"] for r in rows)
    survivor = max(rows, key=lambda r: r["a_adv"])
    out = [r"\setlength{\LTcapwidth}{\textwidth}", r"\setlength{\tabcolsep}{5pt}"]
    out += DECOMP_COLSPEC
    out += [decomp_full_caption(max_re) + r"\label{tab:app-decomp-full}\\", r"\toprule", DECOMP_HEADER, r"\midrule", r"\endfirsthead"]
    out += [
        r"\multicolumn{9}{@{}l}{\footnotesize\itshape Table~\ref{tab:app-decomp-full} continued from previous page}\\",
        r"\toprule", DECOMP_HEADER, r"\midrule", r"\endhead",
        r"\midrule", r"\multicolumn{9}{r@{}}{\footnotesize\itshape continued on next page}\\", r"\endfoot",
        r"\bottomrule",
        r"\multicolumn{9}{@{}l}{\footnotesize $^{\dagger}$The sole profile above chance.}\\",
        r"\endlastfoot",
    ]
    prev_model = None
    prev_set = None
    for r in rows:
        if prev_model is not None and r["model"] != prev_model:
            out.append(r"\midrule")
            prev_set = None
        elif prev_set is not None and r["set"] != prev_set:
            out.append(r"\addlinespace")
        mod = MODEL_LONG[r["model"]] if r["model"] != prev_model else ""
        ds = SET_SHORT[r["set"]] if r["set"] != prev_set else ""
        adv = fnum(r["a_adv"], 3)
        if r is survivor:
            adv = adv + r"$^{\dagger}$\,"
        cells = [
            mod, ds, CH_DISPLAY[r["estimator"]],
            fnum(r["w"], 3), fnum(r["a_same"], 3), fnum(r["a_cross"], 3),
            adv, fnum(r["rho"], 3), f"{r['recon_err']:.1e}",
        ]
        out.append(" & ".join(cells) + r" \\")
        prev_model = r["model"]
        prev_set = r["set"]
    out.append(r"\end{longtable}")
    return write_output("tables/appendices/decomp_full.tex", out)


PHEN_HEADER = r"estimator & model & dataset & clean & attacked & floor & $<$floor & $<$chance \\"


def build_phenomenon_full():
    rows = load_artifact("phenomenon_full.json")["rows"]
    n_total = len(rows)
    for r in rows:
        r["below_floor"] = r["attacked"] <= r["floor"]
        r["below_chance"] = r["attacked"] < 0.5
    n_chance = sum(1 for r in rows if r["below_chance"])
    n_floor = sum(1 for r in rows if r["below_floor"])
    exc = [r for r in rows if not r["below_chance"]][0]
    cap = phenomenon_caption(n_total)
    foot = phenomenon_footer(
        n_chance, n_total,
        CH_DISPLAY[exc["estimator"]], MODEL_SHORT[exc["model"]], SET_SHORT[exc["set"]],
        fnum(exc["attacked"], 3), n_floor,
    )
    out = [r"\footnotesize", r"\setlength{\tabcolsep}{6pt}", r"\setlength{\LTcapwidth}{\linewidth}"]
    out += [r"\begin{longtable}{@{\extracolsep{\fill}}lll rr r cc@{}}"]
    out += [cap + r"\label{tab:app-phenomenon}\\", r"\toprule", PHEN_HEADER, r"\midrule", r"\endfirsthead"]
    out += [
        r"\multicolumn{8}{@{}l}{\emph{Table~\ref{tab:app-phenomenon} continued from previous page}}\\",
        r"\toprule", PHEN_HEADER, r"\midrule", r"\endhead",
        r"\midrule", r"\multicolumn{8}{r@{}}{\emph{continued on next page}}\\", r"\endfoot",
        r"\bottomrule", foot, r"\endlastfoot",
    ]
    prev_est = None
    prev_model = None
    for r in rows:
        if prev_est is not None and r["estimator"] != prev_est:
            out.append(r"\midrule")
            prev_model = None
        elif prev_model is not None and r["model"] != prev_model:
            out.append(r"\addlinespace")
        est = CH_DISPLAY[r["estimator"]] if r["estimator"] != prev_est else ""
        mod = MODEL_SHORT[r["model"]] if r["model"] != prev_model else ""
        cells = [
            est, mod, SET_SHORT[r["set"]],
            fnum(r["clean"], 3), fnum(r["attacked"], 3), fnum(r["floor"], 3),
            "yes" if r["below_floor"] else "no",
            "yes" if r["below_chance"] else "no",
        ]
        out.append(" & ".join(cells) + r" \\")
        prev_est = r["estimator"]
        prev_model = r["model"]
    out.append(r"\end{longtable}")
    out.append(r"\normalsize")
    return write_output("tables/appendices/phenomenon_full.tex", out)


_NUMWORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}


def build_certificate_headline():
    rows = load_artifact("certificate_headline.json")["rows"]
    n_total = len(rows)
    na_rows = [r for r in rows if _is_na(r)]
    n_na = len(na_rows)
    n_ref = n_total - n_na
    na_caption = " and ".join(
        f"{CH_DISPLAY[r['channel']]} on {MODEL_SHORT[r['model']]}/{SET_SHORT[r['set']]}" for r in na_rows
    )
    na_foot = ", ".join(
        f"{CH_DISPLAY[r['channel']]}/{MODEL_SHORT[r['model']]}/{SET_SHORT[r['set']]}" for r in na_rows
    )
    remaining = "the remaining one" if n_na == 1 else f"the remaining {_NUMWORD.get(n_na, str(n_na))}"
    row_word = "that row shows" if n_na == 1 else "those rows show"
    cap = (
        r"\caption{\textbf{The uniform amplitude certificate, refuted in "
        + f"{n_ref} of {n_total}"
        + r" estimator-by-cell profiles and unattainable in " + remaining + r".} "
        + r"$\gamma^{\ast}$ (with bootstrap 95\% CI) is the amplitude below which a certified readout provably keeps discriminating above the cell's operative floor, reported as the continuous crossing of the clean separation curve; clean sep.\ is the mean clean pairwise separation and $G(0)$ the clean AUROC, both over all correct-incorrect pairs; disp.\ is the mean per-item displacement the aimed attack witnesses and max the largest, both on the feasibility-restricted set of $n$ feas.\ items; frac$>\gamma^{\ast}$ is the fraction of feasible items whose witness exceeds $\gamma^{\ast}$, each falsifying a uniform certificate at that level; refuted marks whether a feasible witness exceeds $\gamma^{\ast}$. For "
        + na_caption
        + r" the clean AUROC $G(0)$ already lies below the operative floor, so no positive amplitude is certifiable; "
        + row_word
        + r" n/a for both $\gamma^{\ast}$ and refuted, since the certificate is unattainable rather than refuted.}"
    )
    foot = [
        r"\multicolumn{11}{@{}l}{\footnotesize "
        + f"{n_ref} of {n_total}"
        + r" profiles refuted; " + _NUMWORD.get(n_na, str(n_na)) + r" unattainable (" + na_foot + r":}\\",
        r"\multicolumn{11}{@{}l}{\footnotesize $G(0)$ below the operative floor, so no positive amplitude is certifiable).}\\",
    ]
    out = [r"\setlength{\LTcapwidth}{\textwidth}", r"\setlength{\tabcolsep}{3pt}"]
    out += CERT_COLSPEC
    out += [cap, r"\label{tab:app-certificate}\\", r"\toprule", CERT_HEADER, r"\midrule", r"\endfirsthead"]
    out += [
        r"\multicolumn{11}{@{}l}{\emph{Table~\ref{tab:app-certificate} continued from previous page}}\\",
        r"\toprule", CERT_HEADER, r"\midrule", r"\endhead",
        r"\midrule", r"\multicolumn{11}{r@{}}{\emph{continued on next page}}\\", r"\endfoot",
        r"\bottomrule",
    ]
    out += foot
    out += [r"\endlastfoot"]
    prev_ch = None
    prev_model = None
    for r in rows:
        if prev_ch is not None and r["channel"] != prev_ch:
            out.append(r"\midrule")
            prev_model = None
        elif prev_model is not None and r["model"] != prev_model:
            out.append(r"\addlinespace")
        est = CH_DISPLAY[r["channel"]] if r["channel"] != prev_ch else ""
        mod = MODEL_SHORT[r["model"]] if r["model"] != prev_model else ""
        if _is_na(r):
            gamma = "n/a"; refuted = "n/a"; frac = "n/a"
        else:
            lo, hi = r["ci"]
            gamma = fnum(r["gamma_exact"], 3) + r"~{\scriptsize[" + fnum(lo, 3) + "," + fnum(hi, 3) + "]}"
            refuted = "yes"; frac = fnum(r["frac"], 3)
        cells = [
            est, mod, SET_SHORT[r["set"]], gamma,
            fnum(r["sep"], 3), fnum(r["g0"], 3), fnum(r["disp"], 3), fnum(r["wmax"], 3),
            str(r["nfeas"]), frac, refuted,
        ]
        out.append(" & ".join(cells) + r" \\")
        prev_ch = r["channel"]
        prev_model = r["model"]
    out.append(r"\end{longtable}")
    return write_output("tables/appendices/certificate_headline.tex", out)


def build_single_objective():
    D = load_artifact("single_objective.json")
    rows = D["rows"]
    idx = {(r["estimator"], r["model"], r["set"]): r for r in rows}
    n = len(rows)
    n_bc = sum(1 for r in rows if r["below_chance"])
    n_lf = sum(1 for r in rows if r["le_floor"])
    lf_txt = f"all {n}" if n_lf == n else f"{n_lf} of {n}"
    from src.analysis.latexlib import MODEL_ORDER, SET_ORDER
    EO = ["TokProb", "P(IK)", "SAPLMA", "CCPS", "CCPS-D", "FULL", "COUPLED"]
    cap = (
        r"\caption{\textbf{Single-objective attacked AUROC.} Each readout is scored only on the endpoint aimed at it, in the harmful direction, with no per-item selection across objectives. "
        r"single-obj is the attacked AUROC on that one endpoint (CCPS uses the CCPS-D-aimed endpoint; TokProb, which is never aimed, uses the one objective that damages it most in each cell); union is the worst-case oracle of Table~\ref{tab:app-phenomenon}, which selects per item across all aimed endpoints; diff is union minus single-obj. `$\le$floor' marks single-obj at or below the operative floor; `$<$chance' marks single-obj below $0.5$. Even without the union, "
        + f"{n_bc} of {n}" + r" profiles fall below chance and " + lf_txt + r" at or below the floor.}"
    )
    hdr = r"estimator & model & set & clean & single-obj [95\% CI] & union & diff & $\le$floor & $<$chance \\"
    out = [r"\setlength{\LTcapwidth}{\textwidth}", r"\setlength{\tabcolsep}{4pt}",
           r"\begin{longtable}{@{}lll rrrr cc@{}}", cap + r"\label{tab:app-single-objective}\\",
           r"\toprule", hdr, r"\midrule", r"\endfirsthead",
           r"\multicolumn{9}{@{}l}{\emph{Table~\ref{tab:app-single-objective} continued}}\\", r"\toprule",
           hdr, r"\midrule", r"\endhead", r"\midrule",
           r"\multicolumn{9}{r@{}}{\emph{continued on next page}}\\", r"\endfoot",
           r"\bottomrule", r"\endlastfoot"]
    prev = None
    for e in EO:
        for m in MODEL_ORDER:
            for s in SET_ORDER:
                r = idx[(e, m, s)]
                if prev is not None and e != prev:
                    out.append(r"\midrule")
                est = e if e != prev else ""
                lo, hi = r["single_ci"]
                ci = fnum(r["single"], 3) + r"~{\scriptsize[" + fnum(lo, 3) + "," + fnum(hi, 3) + "]}"
                out.append(
                    f"{est} & {MODEL_SHORT[m]} & {SET_SHORT[s]} & {fnum(r['clean'], 3)} & {ci} & "
                    f"{fnum(r['wc_union'], 3)} & {fnum(r['diff'], 3)} & {'yes' if r['le_floor'] else 'no'} & "
                    f"{'yes' if r['below_chance'] else 'no'} " + r"\\"
                )
                prev = e
    out.append(r"\end{longtable}")
    return write_output("tables/appendices/single_objective.tex", out)


BUILDERS = {
    "certificate": build_certificate,
    "certificate_headline": build_certificate_headline,
    "single_objective": build_single_objective,
    "decomp_full": build_decomp_full,
    "phenomenon_full": build_phenomenon_full,
}
