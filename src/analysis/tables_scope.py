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
    foot = (
        r"\noindent{\footnotesize $^{\S}$Anchor cell (LLaVA-OV/VQAv2), trained at full budget $K{=}4$, "
        r"inner steps 10; the other eleven cells use the reduced training-time inner attack $K{=}8$, "
        r"inner steps 5, with the evaluation attack unchanged at full strength. A gradient-penalty "
        r"variant, run for the anchor only, reaches clean " + fnum(d["gradpen_clean"], 3) + " and attacked "
        + fnum(d["gradpen_attacked"], 3) + r", also no escape. $^{\ddagger}$Nearest miss: peak clean "
        + fnum(nm["peak_clean"], 3) + " clears the clean leg (threshold " + fnum(nm["clean_threshold"], 3)
        + ") but peak att " + fnum(nm["peak_att"], 3) + " (three-seed sd " + fnum(nm["peak_att_sd"], 3)
        + "; seeds " + fnum(sd[0], 3) + ", " + fnum(sd[1], 3) + ", " + fnum(sd[2], 3)
        + r") stays below the 0.55 bar; it is the highest attacked AUROC anywhere. Escape ("
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
        "",
        r"\vspace{3pt}",
        r"\noindent{\small Molmo and Gemma were not run under the aimed EOT attack.}",
    ]
    return write_output("tables/appendices/defenses_partial.tex", lines)


BUILDERS = {
    "gate_harm": build_gate_harm,
    "gate_robustness": build_gate_robustness,
    "frontier_sweep": build_frontier_sweep,
    "attack_invariance": build_attack_invariance,
    "cross_model_transfer": build_cross_model_transfer,
    "universal_perturbation": build_universal_perturbation,
    "defenses_partial": build_defenses_partial,
}
