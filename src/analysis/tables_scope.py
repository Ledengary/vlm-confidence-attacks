from src.analysis.latexlib import (
    MODEL_ORDER,
    SET_ORDER,
    MODEL_LONG,
    MODEL_SHORT,
    SET_SHORT,
    load_artifact,
    write_output,
    fnum,
)
from src.analysis.templates import universal_footnote


def _yn(flag):
    return "yes" if flag else "no"


def build_gate_harm():
    data = load_artifact("gate_harm.json")["cells"]
    lines = [
        r"\footnotesize",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{@{}llrrrrr@{}}",
        r"\toprule",
        r"model & set & acc-clean & w2ac (tr) & acc-adv (tr) & w2ac (aim) & acc-adv (aim) \\",
        r"\midrule",
    ]
    for mi, m in enumerate(MODEL_ORDER):
        for si, s in enumerate(SET_ORDER):
            c = data[m][s]
            head = r"\multirow{3}{*}{" + MODEL_LONG[m] + "}" if si == 0 else ""
            lines.append(
                f"{head} & {SET_SHORT[s]} & {fnum(c['acc_clean'], 3)} & {fnum(c['w2ac'], 3)} & "
                f"{fnum(c['acc_adv'], 3)} & {fnum(c['w2ac_aimed'], 3)} & {fnum(c['acc_adv_aimed'], 3)} " + r"\\"
            )
        if mi != len(MODEL_ORDER) - 1:
            lines.append(r"\midrule")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        "",
        r"\vspace{8pt}",
        r"\noindent{\footnotesize\itshape Accepted accuracy at natural prevalence, no-gate baseline vs gated:}\\[2pt]",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{@{}llrrrrcc@{}}",
        r"\toprule",
        r"model & set & no-gate & gated cln & gated (tr) & gated (aim) & $<$base (tr) & $<$base (aim) \\",
        r"\midrule",
    ]
    for mi, m in enumerate(MODEL_ORDER):
        for si, s in enumerate(SET_ORDER):
            c = data[m][s]
            head = r"\multirow{3}{*}{" + MODEL_LONG[m] + "}" if si == 0 else ""
            lines.append(
                f"{head} & {SET_SHORT[s]} & {fnum(c['p_natural'], 3)} & {fnum(c['rew_clean'], 3)} & "
                f"{fnum(c['rew_adv_transfer'], 3)} & {fnum(c['rew_adv_aimed'], 3)} & "
                f"{_yn(c['below_transfer'])} & {_yn(c['below_aimed'])} " + r"\\"
            )
        if mi != len(MODEL_ORDER) - 1:
            lines.append(r"\midrule")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        "",
        r"\vspace{3pt}",
        r"\noindent{\footnotesize The gate is the FULL hidden-state gate at a $10\%$ clean false-accept rate, fixed under attack. \emph{tr} is the transfer token-probability attack, \emph{aim} the gate-aimed attack; \emph{w2ac} is the fraction of wrong answers flipped from abstain to accept. Upper block is the balanced pool, so an acc-adv below $0.5$ is worse than random acceptance. Lower block reweights accepted accuracy to the benchmark's natural correctness prevalence; the no-gate baseline accepts every answer, and $<$base marks gated accepted accuracy below it.}",
    ]
    return write_output("tables/appendices/gate_harm.tex", lines)


OP_LABEL = {
    "clean10fpr": r"clean-10\%-FPR",
    "youdenJ": "Youden J",
    "tau0.5": r"$\tau{=}0.5$",
}


def build_gate_robustness():
    d = load_artifact("gate_robustness.json")
    lines = [
        r"\footnotesize",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{@{}lrrrrc@{}}",
        r"\toprule",
        r"operating point & clean FPR & acc-clean & acc-adv & drop & cells acc-adv$<$0.5 (bal.) \\",
        r"\midrule",
    ]
    for op in d["op_order"]:
        b = d["operating_points"][op]
        lines.append(
            f"{OP_LABEL[op]} & {fnum(b['clean_fpr'], 3)} & {fnum(b['acc_clean'], 3)} & "
            f"{fnum(b['acc_adv'], 3)} & {fnum(b['drop'], 3)} & {b['n_below']}/{b['n_total']} " + r"\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        "",
        r"\vspace{8pt}",
        r"\noindent{\footnotesize\itshape Risk-coverage AURC (FULL locus, higher is worse), per cell, clean vs attacked:}\\[2pt]",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{@{}llrr@{}}",
        r"\toprule",
        r"model & set & AURC clean & AURC attacked \\",
        r"\midrule",
    ]
    cells = d["aurc"]["cells"]
    for m in MODEL_ORDER:
        for si, s in enumerate(SET_ORDER):
            c = cells[m][s]
            head = r"\multirow{3}{*}{" + MODEL_LONG[m] + "}" if si == 0 else ""
            lines.append(f"{head} & {SET_SHORT[s]} & {fnum(c['clean'], 3)} & {fnum(c['attacked'], 3)} " + r"\\")
        lines.append(r"\midrule")
    lines.append(
        r"\textbf{FULL mean} & & \textbf{" + fnum(d["full_mean_clean"], 3) + r"} & \textbf{" + fnum(d["full_mean_attacked"], 3) + r"} \\"
    )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    foot = (
        r"\noindent{\footnotesize AURC is the area under the risk-coverage curve integrated over all "
        r"coverage (higher is worse). Over the 60 diagonal profiles (12 cells $\times$ 5 estimators) the "
        f"mean rises from {fnum(d['diag60_clean'], 3)} clean to {fnum(d['diag60_attacked'], 3)} attacked; "
        f"the FULL-locus 12-cell mean shown here rises from {fnum(d['full_mean_clean'], 3)} to "
        f"{fnum(d['full_mean_attacked'], 3)}." + r"}"
    )
    lines += [r"\vspace{3pt}", foot]
    return write_output("tables/appendices/gate_robustness.tex", lines)


def _cellpair(v, bold):
    txt = f"{fnum(v['clean'], 3)}/{fnum(v['attacked'], 3)}"
    return r"\textbf{" + txt + "}" if bold else txt


def build_frontier_sweep():
    d = load_artifact("frontier_sweep.json")
    cells = d["cells"]
    labs = d["beta_labels"]
    lines = [
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}ll cccc r c@{}}",
        r"\toprule",
        r"model & dataset & $\beta{=}0.3$ & $\beta{=}1$ & $\beta{=}3$ & $\beta{=}10$ & peak att & escape \\",
        r" &  & \multicolumn{4}{c}{clean\,/\,attacked (seed 23; peak $\beta$ bold)} &  &  \\",
        r"\midrule",
    ]
    for mi, m in enumerate(MODEL_ORDER):
        for si, s in enumerate(SET_ORDER):
            c = cells[m][s]
            head = r"\multirow{3}{*}{" + MODEL_LONG[m] + "}" if si == 0 else ""
            ds = SET_SHORT[s]
            if c["is_anchor"]:
                ds += r"$^{\S}$"
            if c["is_nearest_miss"]:
                ds += r"$^{\ddagger}$"
            cols = " & ".join(_cellpair(c["grid"][lab], c["peak_beta"] == lab) for lab in labs)
            esc = "yes" if c["escape"] else "no"
            lines.append(f"{head} & {ds} & {cols} & {fnum(c['peak_att'], 3)} & {esc} " + r"\\")
        if mi != len(MODEL_ORDER) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    nm = d["nearest"]
    sd = nm["peak_att_seeds"]
    nm_s23 = None
    for m in MODEL_ORDER:
        for s in SET_ORDER:
            c = cells[m][s]
            if c["is_nearest_miss"]:
                nm_s23 = c["grid"][c["peak_beta"]]
    foot = (
        r"\noindent{\footnotesize $^{\S}$Anchor cell (LLaVA-OV/VQAv2), trained at full budget $K{=}4$, "
        r"inner steps 10; the other eleven cells use the reduced training-time inner attack $K{=}8$, "
        r"inner steps 5, with the evaluation attack unchanged at full strength. A gradient-penalty "
        r"variant, run for the anchor only, reaches clean " + fnum(d["gradpen_clean"], 3) + " and attacked "
        + fnum(d["gradpen_attacked"], 3) + r", also no escape. $^{\ddagger}$Nearest miss: as three-seed means, peak clean "
        + fnum(nm["peak_clean"], 3) + " clears the clean leg (threshold " + fnum(nm["clean_threshold"], 3)
        + ") but peak att " + fnum(nm["peak_att"], 3) + " (sd " + fnum(nm["peak_att_sd"], 3)
        + "; seeds " + fnum(sd[0], 3) + ", " + fnum(sd[1], 3) + ", " + fnum(sd[2], 3)
        + r") stays below the 0.55 bar; the row shows the seed-23 values " + fnum(nm_s23["clean"], 3)
        + " and " + fnum(nm_s23["attacked"], 3)
        + r"; it is the highest attacked AUROC anywhere. Escape ("
        + f"{d['n_escape']} of {d['n_cells']}"
        + r") requires peak clean above the clean threshold and peak att above 0.55.}"
    )
    lines += [r"\vspace{3pt}", foot]
    return write_output("tables/appendices/frontier_sweep.tex", lines)


NUMWORD = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]


def build_attack_invariance():
    d = load_artifact("attack_invariance.json")
    ba = d["block_a"]["cells"]
    first = ba[MODEL_ORDER[0]][SET_ORDER[0]]
    n_coord, n_lf = first["coord_total"], first["lf_total"]
    lines = [
        r"\small",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{@{}llcc@{}}",
        r"\toprule",
        f"model & dataset & coordinated (of {n_coord}) & label-agnostic (of {n_lf}) " + r"\\",
        r"\midrule",
    ]
    for m in MODEL_ORDER:
        for si, s in enumerate(SET_ORDER):
            c = ba[m][s]
            head = r"\multirow{3}{*}{" + MODEL_LONG[m] + "}" if si == 0 else ""
            la = f"{c['lf_pass']}/{c['lf_total']}"
            if c["lf_pass"] < c["lf_total"]:
                la += r"$^{\dagger}$"
            lines.append(f"{head} & {SET_SHORT[s]} & {c['coord_pass']}/{c['coord_total']} & {la} " + r"\\")
        lines.append(r"\midrule")
    lines.append(
        r"\textbf{total} & & \textbf{" + f"{d['total_coord']}/{d['total_coord_max']}" + r"} & \textbf{"
        + f"{d['total_lf']}/{d['total_lf_max']}" + r"} \\"
    )
    lines += [r"\bottomrule", r"\end{tabular}", ""]

    faillist = ", ".join(f"{x['config']}/{MODEL_SHORT[x['model']]}/{SET_SHORT[x['set']]}" for x in d["lf_failures"])
    lines += [
        r"\vspace{3pt}",
        r"\noindent{\small $^{\dagger}$The " + NUMWORD[len(d["lf_failures"])] + r" label-agnostic misses are all Gemma-3-12B and directionless",
        "(alignment below the null bar), not reversed: " + faillist + ", a token-probability",
        r"saturation effect; they pass the unsigned-drift criterion. Coordinated configs: A1--A6 (budget",
        r"$\{2,4,8,16\}/255$, PGD steps $\{5,15,30,60\}$), B1--B2 (two $L_2$ norms), D1--D2 (aim P(IK)/SAPLMA).}",
        "",
        r"\vspace{8pt}",
        r"\noindent{\small\itshape Margin-band control (SAPLMA; clean and attacked AUROC on the attack-engaged subset):}\\[2pt]",
        r"\small",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{@{}llrrrr@{}}",
        r"\toprule",
        r"model & dataset & clean AUROC & attacked (unbounded) & attacked (margin) & floor \\",
        r"\midrule",
    ]
    bb = d["block_b"]["cells"]
    for mi, m in enumerate(MODEL_ORDER):
        for si, s in enumerate(SET_ORDER):
            c = bb[m][s]
            head = r"\multirow{3}{*}{" + MODEL_LONG[m] + "}" if si == 0 else ""
            lines.append(
                f"{head} & {SET_SHORT[s]} & {fnum(c['clean'], 3)} & {fnum(c['adv_unbounded'], 3)} & "
                f"{fnum(c['adv_margin'], 3)} & {fnum(c['floor'], 3)} " + r"\\"
            )
        if mi != len(MODEL_ORDER) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", ""]

    above = " and ".join(f"{MODEL_SHORT[x['model']]}/{SET_SHORT[x['set']]} {fnum(x['value'], 3)}" for x in d["above_half"])
    n = d["n_cells"]
    lines += [
        r"\vspace{3pt}",
        r"\noindent{\small Attacked (margin) holds the answer margin in the 0.25 band; it is at or below the",
        f"operative floor in {d['n_below_floor']} of {n} cells (below chance in {d['n_below_half']} of {n}; {above} stay",
        r"above 0.5 but below their floors). The collapse persists when the margin is not allowed to run free.}",
    ]
    return write_output("tables/appendices/attack_invariance.tex", lines)


def _sg(x, nd=3):
    return ("$+" if x >= 0 else "$-") + f"{abs(x):.{nd}f}" + "$"


def build_cross_model_transfer():
    d = load_artifact("cross_model_transfer.json")
    cells = d["cells"]
    lines = [
        r"\footnotesize",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{@{}llrrrl@{}}",
        r"\toprule",
        r"source & set & white-box tok drop & transfer mean & transfer max & worst target \\",
        r"\midrule",
    ]
    sources = d["sources"]
    all_wb = []
    for mi, m in enumerate(sources):
        for si, s in enumerate(SET_ORDER):
            c = cells[m][s]
            all_wb.append(c["white_box"])
            head = r"\multirow{3}{*}{" + MODEL_LONG[m] + "}" if si == 0 else ""
            lines.append(
                f"{head} & {SET_SHORT[s]} & {_sg(c['white_box'])} & {_sg(c['transfer_mean'])} & "
                f"{_sg(c['transfer_max'])} & {MODEL_SHORT[c['worst_target']]} " + r"\\"
            )
        if mi != len(sources) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    mp = d["max_pair"]
    worst_pair = f"{MODEL_SHORT[mp['source']]} to {MODEL_SHORT[mp['target']]} on {SET_SHORT[mp['set']]}"
    foot = (
        r"\noindent{\footnotesize Metric: token-probability AUROC drop. The same-model (white-box) attack "
        "removes $" + fnum(min(all_wb), 3) + "$ to $" + fnum(max(all_wb), 3) + "$; the same image transferred "
        "to a different target removes at most $+" + fnum(mp["value"], 3) + "$ (" + worst_pair
        + f", the largest of the {d['n_offdiag']} off-diagonal transfers), and most are below $+0.03$." + r"}"
    )
    lines += [r"\vspace{3pt}", foot]
    return write_output("tables/appendices/cross_model_transfer.tex", lines)


def build_universal_perturbation():
    d = load_artifact("universal_perturbation.json")
    ba = d["block_a"]
    models = d["models"]
    sets = d["sets"]
    lines = [
        r"\footnotesize",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{@{}llrrr@{}}",
        r"\toprule",
        r"model & set & max$|$uni$-$rand$|$ & per-image SAPLMA drop & ratio \\",
        r"\midrule",
    ]
    ratios = []
    for mi, m in enumerate(models):
        for si, s in enumerate(sets):
            c = ba[m][s]
            ratio = f"{round(c['pim'], 3) / round(c['maxdiff'], 3):.0f}"
            ratios.append(int(ratio))
            head = r"\multirow{3}{*}{" + MODEL_LONG[m] + "}" if si == 0 else ""
            lines.append(
                f"{head} & {SET_SHORT[s]} & {fnum(c['maxdiff'], 3)} & {fnum(c['pim'], 3)} & {ratio}" + r"$\times$ \\"
            )
        if mi != len(models) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", ""]

    lines += [
        r"\vspace{8pt}",
        r"\noindent{\footnotesize\itshape Universal vs equal-norm random delta, per channel (Gemma/GQA):}\\[2pt]",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{@{}lrrrl@{}}",
        r"\toprule",
        r"channel & clean & uni drop & rand drop & uni$-$rand [95\% CI] \\",
        r"\midrule",
    ]
    channels = d["block_b"]["channels"]
    n_contain = 0
    for ch in channels:
        lo, hi = ch["ci"]
        if lo <= 0 <= hi:
            n_contain += 1
        ci = r" {\scriptsize[" + _sg(lo) + "," + _sg(hi) + "]}"
        lines.append(
            f"{ch['name']} & {fnum(ch['clean'], 3)} & {_sg(ch['uni_drop'])} & {_sg(ch['rand_drop'])} & "
            f"{_sg(ch['diff'])}{ci} " + r"\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", ""]

    pims = [ba[m][s]["pim"] for m in models for s in sets]
    maxdiffs = [ba[m][s]["maxdiff"] for m in models for s in sets]
    foot = universal_footnote(
        fnum(max(maxdiffs), 3), NUMWORD[len(pims)], NUMWORD[n_contain], NUMWORD[len(channels)],
        fnum(min(pims), 3), fnum(max(pims), 3), str(min(ratios)), str(max(ratios)),
    )
    lines += [r"\vspace{3pt}", foot]
    return write_output("tables/appendices/universal_perturbation.tex", lines)


def build_defenses_partial():
    d = load_artifact("defenses_partial.json")
    lines = [
        r"\small",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"defense & P(IK) survival range & median & corruption range & median \\",
        r"\midrule",
    ]
    for x in d["block_a"]["defenses"]:
        sr = f"{x['surv_min']:.2f}--{x['surv_max']:.2f}"
        cr = f"{x['corr_min']:.2f}--{x['corr_max']:.2f}"
        lines.append(f"{x['label']} & {sr} & {fnum(x['surv_med'], 3)} & {cr} & {fnum(x['corr_med'], 3)} " + r"\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        "",
        r"\vspace{3pt}",
        r"\noindent{\small survival = fraction of the P(IK) manipulable range the purifier leaves ($1.0$ = removes nothing; values above $1.0$ mean the purifier enlarges the range); corruption = own-answer corruption the purifier costs.}",
        "",
        r"\vspace{8pt}",
        r"\noindent{\small\itshape Randomized smoothing under the adaptive EOT attack (SAPLMA):}\\[2pt]",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lll r rr r rr r r c@{}}",
        r"\toprule",
        r"model & set & $\sigma$ & clean & tr.\ disp & aim disp & ratio & tr.\ adv & aim adv & floor & $\gamma^{\ast}$ & disp$>\gamma^{\ast}$ \\",
        r"\midrule",
    ]
    b = d["block_b"]["base"]
    lines.append(
        f"{MODEL_SHORT[b['model']]} & {SET_SHORT[b['set']]} & 0 (base) & {fnum(b['clean'], 3)} & -- & "
        f"{fnum(b['aim_disp'], 3)} & -- & -- & {fnum(b['aim_adv'], 3)} & {fnum(b['floor'], 3)} & "
        f"{fnum(b['gamma_star'], 3)} & {_yn(b['holds'])} " + r"\\"
    )
    for r in d["block_b"]["rows"]:
        ratio = f"{r['aim_disp'] / r['tr_disp']:.1f}"
        lines.append(
            f"{MODEL_SHORT[r['model']]} & {SET_SHORT[r['set']]} & {r['sigma']} & {fnum(r['clean'], 3)} & "
            f"{fnum(r['tr_disp'], 3)} & {fnum(r['aim_disp'], 3)} & {ratio}$\\times$ & {fnum(r['tr_adv'], 3)} & "
            f"{fnum(r['aim_adv'], 3)} & {fnum(r['floor'], 3)} & {fnum(r['gamma_star'], 3)} & {_yn(r['holds'])} " + r"\\"
        )
    for m in d["block_b"]["not_run"]:
        lines.append(f"{MODEL_SHORT[m]} & -- & -- & not run & -- & -- & -- & -- & -- & -- & -- & -- " + r"\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
    ]
    if d["block_b"]["not_run"]:
        names = " and ".join(MODEL_SHORT[m] for m in d["block_b"]["not_run"])
        lines += [
            "",
            r"\vspace{3pt}",
            r"\noindent{\small " + names + r" were not run under the aimed EOT attack.}",
        ]
    return write_output("tables/appendices/defenses_partial.tex", lines)


def build_gate_harm_clipped():
    data = load_artifact("gate_harm_clipped.json")["cells"]
    lines = [
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}ll r rr rr rr@{}}",
        r"\toprule",
        r" &  &  & \multicolumn{2}{c}{B1 oracle} & \multicolumn{2}{c}{B2 label-free} & \multicolumn{2}{c}{B3 cross-probe} \\",
        r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
        r"model & set & acc-cln & w2ac & acc-adv & w2ac & acc-adv & w2ac & acc-adv \\",
        r"\midrule",
    ]
    for mi, m in enumerate(MODEL_ORDER):
        for si, s in enumerate(SET_ORDER):
            c = data[m][s]
            head = r"\multirow{3}{*}{" + MODEL_LONG[m] + "}" if si == 0 else ""
            lines.append(
                f"{head} & {SET_SHORT[s]} & {fnum(c['acc_clean'], 3)} & {fnum(c['w2ac_b1'], 3)} & {fnum(c['acc_adv_b1'], 3)} & "
                f"{fnum(c['w2ac_b2'], 3)} & {fnum(c['acc_adv_b2'], 3)} & {fnum(c['w2ac_b3'], 3)} & {fnum(c['acc_adv_b3'], 3)} " + r"\\"
            )
        if mi != len(MODEL_ORDER) - 1:
            lines.append(r"\midrule")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        "",
        r"\vspace{6pt}",
        r"\begin{tabular}{@{}ll rrrr ccc@{}}",
        r"\toprule",
        r"model & set & no-gate & B1 & B2 & B3 & $<$base B1 & $<$base B2 & $<$base B3 \\",
        r"\midrule",
    ]
    for mi, m in enumerate(MODEL_ORDER):
        for si, s in enumerate(SET_ORDER):
            c = data[m][s]
            head = r"\multirow{3}{*}{" + MODEL_LONG[m] + "}" if si == 0 else ""
            lines.append(
                f"{head} & {SET_SHORT[s]} & {fnum(c['p_natural'], 3)} & {fnum(c['rew_adv_b1'], 3)} & {fnum(c['rew_adv_b2'], 3)} & "
                f"{fnum(c['rew_adv_b3'], 3)} & {_yn(c['below_b1'])} & {_yn(c['below_b2'])} & {_yn(c['below_b3'])} " + r"\\"
            )
        if mi != len(MODEL_ORDER) - 1:
            lines.append(r"\midrule")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        "",
        r"\vspace{6pt}",
        r"\begin{tabular}{@{}ll rrr@{}}",
        r"\toprule",
        r"model & set & AURC clean & AURC B1 & AURC B2 \\",
        r"\midrule",
    ]
    for mi, m in enumerate(MODEL_ORDER):
        for si, s in enumerate(SET_ORDER):
            c = data[m][s]
            head = r"\multirow{3}{*}{" + MODEL_LONG[m] + "}" if si == 0 else ""
            lines.append(
                f"{head} & {SET_SHORT[s]} & {fnum(c['aurc_clean'], 3)} & {fnum(c['aurc_b1'], 3)} & {fnum(c['aurc_b2'], 3)} " + r"\\"
            )
        if mi != len(MODEL_ORDER) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return write_output("tables/appendices/gate_harm_clipped.tex", lines)


def build_gate_robustness_clipped():
    d = load_artifact("gate_robustness_clipped.json")
    lines = [
        r"\footnotesize",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{@{}lrr rrc rrc@{}}",
        r"\toprule",
        r" &  &  & \multicolumn{3}{c}{B1 oracle} & \multicolumn{3}{c}{B2 label-free} \\",
        r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}",
        r"operating point & clean FPR & acc-clean & acc-adv & drop & $<$0.5 & acc-adv & drop & $<$0.5 \\",
        r"\midrule",
    ]
    for op in d["op_order"]:
        b = d["upper"][op]
        b1 = b["B1"]; b2 = b["B2"]
        lines.append(
            f"{OP_LABEL[op]} & {fnum(b['clean_fpr'], 3)} & {fnum(b['acc_clean'], 3)} & "
            f"{fnum(b1['acc_adv'], 3)} & {fnum(b1['drop'], 3)} & {b1['n_below']}/{b1['n_total']} & "
            f"{fnum(b2['acc_adv'], 3)} & {fnum(b2['drop'], 3)} & {b2['n_below']}/{b2['n_total']} " + r"\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return write_output("tables/appendices/gate_robustness_clipped.tex", lines)


def build_answer_preservation_headline():
    data = load_artifact("answer_preservation_headline.json")["cells"]
    lines = [
        r"\small",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{@{}ll rrrrc@{}}",
        r"\toprule",
        r"model & set & items & FULL fallback & all-obj fallback & items w/ any fb & excluded \\",
        r"\midrule",
    ]
    for mi, m in enumerate(MODEL_ORDER):
        for si, s in enumerate(SET_ORDER):
            c = data[m][s]
            head = r"\multirow{3}{*}{" + MODEL_LONG[m] + "}" if si == 0 else ""
            lines.append(
                f"{head} & {SET_SHORT[s]} & {c['n_items']} & {c['full_fb']}/{c['full_endpoints']} "
                f"({fnum(c['full_fb_pct'], 2)}\\%) & {fnum(c['all_fb_pct'], 2)}\\% & {c['items_any_fb']} & {c['excluded']} " + r"\\"
            )
        if mi != len(MODEL_ORDER) - 1:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return write_output("tables/appendices/answer_preservation_headline.tex", lines)


def _rng3(lo, hi):
    return fnum(lo, 3) + "--" + fnum(hi, 3)


def build_within_answer_headline():
    d = load_artifact("within_answer_headline.json")
    lines = [
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lrrlrl@{}}",
        r"\toprule",
        r"estimator & clean mean & clean median & clean per-cell range & adv mean & adv per-cell range \\",
        r"\midrule",
    ]
    for e in d["estimators"]:
        adv_mean = fnum(e["adv_mean"], 3) + (r"$^{\dagger}$" if e["dagger"] else "")
        lines.append(
            f"{e['label']} & {fnum(e['clean_mean'], 3)} & {fnum(e['clean_median'], 3)} & "
            f"{_rng3(e['clean_min'], e['clean_max'])} & {adv_mean} & {_rng3(e['adv_min'], e['adv_max'])} " + r"\\"
        )
        if e["label"] == "CCPS-D":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", "", r"\vspace{4pt}", r"\footnotesize",
              r"\begin{tabular}{@{}lc@{}}", r"\toprule",
              r"model & attacked within-answer AUROC \\", r"\midrule"]
    for pm in d["per_model"]:
        lines.append(f"{pm['label']} & {fnum(pm['auroc'], 3)} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return write_output("tables/appendices/within_answer_headline.tex", lines)


def build_budget_sweep():
    d = load_artifact("budget_sweep.json")
    eps = d["budgets"]
    up = d["upper"]; lo = d["lower"]
    lines = [r"\footnotesize", r"\setlength{\tabcolsep}{5pt}",
             r"\begin{tabular}{@{}ll r" + ("rr" * len(eps)) + r"@{}}", r"\toprule"]
    lines.append(r" &  &  & " + " & ".join([r"\multicolumn{2}{c}{" + e + r"/255}" for e in eps]) + r" \\")
    cm = ""; col = 4
    for e in eps:
        cm += r"\cmidrule(lr){" + str(col) + "-" + str(col + 1) + "}"; col += 2
    lines.append(cm)
    lines.append(r"model & set & clean & " + " & ".join(["WC-oracle & label-free" for e in eps]) + r" \\")
    lines.append(r"\midrule")
    for mi, m in enumerate(MODEL_ORDER):
        for si, s in enumerate(SET_ORDER):
            u = up[f"{m}/{s}"]
            head = r"\multirow{3}{*}{" + MODEL_LONG[m] + "}" if si == 0 else ""
            cells = []
            for e in eps:
                cells.append(fnum(u["by_eps"][e]["oracle"], 3)); cells.append(fnum(u["by_eps"][e]["lf"], 3))
            lines.append(f"{head} & {SET_SHORT[s]} & {fnum(u['clean'], 3)} & " + " & ".join(cells) + r" \\")
        if mi != len(MODEL_ORDER) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", "", r"\vspace{6pt}",
              r"\noindent{\footnotesize\itshape Reachable-range width and fraction exceeding $\gamma^{\ast}$, per model pooled over the three benchmarks:}\\[2pt]",
              r"\footnotesize", r"\setlength{\tabcolsep}{6pt}",
              r"\begin{tabular}{@{}ll " + ("rr" * len(eps)) + r"@{}}", r"\toprule"]
    lines.append(r" &  & " + " & ".join([r"\multicolumn{2}{c}{" + e + r"/255}" for e in eps]) + r" \\")
    cm2 = ""; col = 3
    for e in eps:
        cm2 += r"\cmidrule(lr){" + str(col) + "-" + str(col + 1) + "}"; col += 2
    lines.append(cm2)
    lines.append(r"model & readout & " + " & ".join([r"mean width & frac$>\gamma^{\ast}$" for e in eps]) + r" \\")
    lines.append(r"\midrule")
    RD = [("pik", "P(IK)"), ("saplma", "SAPLMA"), ("FULL", "FULL")]
    for mi, m in enumerate(MODEL_ORDER):
        for gi, (gk, disp) in enumerate(RD):
            l = lo[f"{m}/{gk}"]
            head = r"\multirow{3}{*}{" + MODEL_LONG[m] + "}" if gi == 0 else ""
            cells = []
            for e in eps:
                cells.append(fnum(l["by_eps"][e]["meanW"], 3)); cells.append(fnum(l["by_eps"][e]["frac"], 3))
            lines.append(f"{head} & {disp} & " + " & ".join(cells) + r" \\")
        if mi != len(MODEL_ORDER) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return write_output("tables/appendices/budget_sweep.tex", lines)


_RE_RD = [("pik", "P(IK)"), ("saplma", "SAPLMA"), ("tokprob", "TokProb")]


def _ratio2(x):
    # round the stored reach/separation ratio to two decimals, half-up (standard convention), so an
    # x.xx5 value like 1.315 renders 1.32 rather than the IEEE/float-repr 1.31 that "%.2f" would give.
    from decimal import Decimal, ROUND_HALF_UP
    return str(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def build_robust_encoder_main():
    m336 = load_artifact("metrics_eps4.json")
    m224 = load_artifact("metrics224_eps4.json")
    blocks = [("GQA", "336", m336, "llava15_7b_clip", "llava15_7b_fare", "gqa"),
              ("GQA", "224", m224, "llava15_7b_clip224", "llava15_7b_fare224", "gqa"),
              ("POPE", "336", m336, "llava15_7b_clip", "llava15_7b_fare", "pope")]
    lines = [
        r"\footnotesize",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{@{}lll rrr rrr@{}}",
        r"\toprule",
        r" &  &  & \multicolumn{3}{c}{Standard CLIP} & \multicolumn{3}{c}{FARE-4} \\",
        r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}",
        r"Set & Input & Readout & clean & oracle & label-free & clean & oracle & label-free \\",
        r"\midrule",
    ]
    for bi, (setname, inp, mm, cv, fv, sk) in enumerate(blocks):
        cr = mm[f"{cv}_{sk}_eps4"]["readouts"]
        fr = mm[f"{fv}_{sk}_eps4"]["readouts"]
        for ri, (rk, disp) in enumerate(_RE_RD):
            sh = r"\multirow{3}{*}{" + setname + "}" if ri == 0 else ""
            ih = r"\multirow{3}{*}{" + inp + "}" if ri == 0 else ""
            c = cr[rk]; f = fr[rk]
            lines.append(
                f"{sh} & {ih} & {disp} & {fnum(c['clean_auroc'], 3)} & {fnum(c['oracle_attacked_auroc'], 3)} & "
                f"{fnum(c['labelfree_attacked_auroc'], 3)} & {fnum(f['clean_auroc'], 3)} & "
                f"{fnum(f['oracle_attacked_auroc'], 3)} & {fnum(f['labelfree_attacked_auroc'], 3)} " + r"\\"
            )
        if bi != len(blocks) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return write_output("tables/appendices/robust_encoder_main.tex", lines)


def build_robust_encoder_ladder():
    M = {("336", 2): load_artifact("metrics_eps2.json"),
         ("336", 4): load_artifact("metrics_eps4.json"),
         ("336", 8): load_artifact("metrics_eps8.json"),
         ("224", 2): load_artifact("metrics224_eps2.json"),
         ("224", 4): load_artifact("metrics224_eps4.json")}
    rows = [("336", 2), ("336", 4), ("336", 8), ("224", 2), ("224", 4)]
    towers = [("Standard", "llava15_7b_clip", "llava15_7b_clip224"),
              ("FARE-4", "llava15_7b_fare", "llava15_7b_fare224")]
    lines = [
        r"\footnotesize",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{@{}lll rrr rrr@{}}",
        r"\toprule",
        r" &  &  & \multicolumn{3}{c}{oracle attacked AUROC} & \multicolumn{3}{c}{reach / separation} \\",
        r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}",
        r"Tower & Input & $\epsilon$ & P(IK) & SAPLMA & TokProb & P(IK) & SAPLMA & TokProb \\",
        r"\midrule",
    ]
    for ti, (tname, v336, v224) in enumerate(towers):
        for ri, (inp, eps) in enumerate(rows):
            variant = v336 if inp == "336" else v224
            rd = M[(inp, eps)][f"{variant}_gqa_eps{eps}"]["readouts"]
            th = r"\multirow{5}{*}{" + tname + "}" if ri == 0 else ""
            ih = r"\multirow{3}{*}{336}" if ri == 0 else (r"\multirow{2}{*}{224}" if ri == 3 else "")
            orc = {k: fnum(rd[k]["oracle_attacked_auroc"], 3) for k in ("pik", "saplma", "tokprob")}
            if tname == "FARE-4" and inp == "224" and eps == 2:
                orc["saplma"] = orc["saplma"] + r"$^{\dagger}$"
            rat = {k: _ratio2(rd[k]["width_over_sep"]) for k in ("pik", "saplma", "tokprob")}
            lines.append(
                f"{th} & {ih} & {eps}/255 & {orc['pik']} & {orc['saplma']} & {orc['tokprob']} & "
                f"{rat['pik']} & {rat['saplma']} & {rat['tokprob']} " + r"\\"
            )
        if ti != len(towers) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return write_output("tables/appendices/robust_encoder_ladder.tex", lines)


def build_combined_attack():
    d = load_artifact("combined_attack.json")
    order = ["llava_onevision_7b", "molmo2_4b"]
    lines = [
        r"\footnotesize",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"model & flip rate & accepted after stage 1 & accepted after stage 2 & end-to-end \\",
        r"\midrule",
    ]
    for m in order:
        c = d[m]
        lines.append(
            f"{MODEL_LONG[m]} & {fnum(c['flip_rate'], 3)} & {fnum(c['accept_after_s1'], 3)} & "
            f"{fnum(c['accept_after_s2'], 3)} & {fnum(c['end_to_end'], 3)} " + r"\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return write_output("tables/appendices/combined_attack.tex", lines)


BUILDERS = {
    "gate_harm": build_gate_harm,
    "combined_attack": build_combined_attack,
    "robust_encoder_main": build_robust_encoder_main,
    "robust_encoder_ladder": build_robust_encoder_ladder,
    "gate_robustness": build_gate_robustness,
    "gate_harm_clipped": build_gate_harm_clipped,
    "gate_robustness_clipped": build_gate_robustness_clipped,
    "answer_preservation_headline": build_answer_preservation_headline,
    "within_answer_headline": build_within_answer_headline,
    "budget_sweep": build_budget_sweep,
    "frontier_sweep": build_frontier_sweep,
    "attack_invariance": build_attack_invariance,
    "cross_model_transfer": build_cross_model_transfer,
    "universal_perturbation": build_universal_perturbation,
    "defenses_partial": build_defenses_partial,
}
