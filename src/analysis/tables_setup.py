import os

import yaml

from src.analysis.latexlib import (
    MODEL_ORDER,
    SET_ORDER,
    MODEL_LONG,
    SET_SHORT,
    load_artifact,
    write_output,
    fnum,
    REPO_ROOT,
)


def build_perturbation_norms():
    data = load_artifact("perturbation_norms.json")["cells"]
    lines = [
        r"\small",
        r"\setlength{\tabcolsep}{7pt}",
        r"\begin{tabular}{@{}llrr@{}}",
        r"\toprule",
        r"model & set & L-inf mean & L-inf max \\",
        r"\midrule",
    ]
    for mi, m in enumerate(MODEL_ORDER):
        for si, s in enumerate(SET_ORDER):
            cell = data[m][s]
            mean = fnum(cell["linf_mean"], 4)
            mx = fnum(cell["linf_max"], 4)
            if si == 0:
                head = r"\multirow{3}{*}{" + MODEL_LONG[m] + "}"
            else:
                head = ""
            lines.append(f"{head} & {SET_SHORT[s]} & {mean} & {mx} " + r"\\")
        if mi != len(MODEL_ORDER) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return write_output("tables/appendices/perturbation_norms.tex", lines)


def build_answer_preservation():
    data = load_artifact("answer_preservation.json")["cells"]
    lines = [
        r"\begin{tabular}{@{}ll rrrr@{}}",
        r"\toprule",
        r"model & set & rigor $n$ & rigor rate & main-attack byte-id & main-attack rate \\",
        r"\midrule",
    ]
    for mi, m in enumerate(MODEL_ORDER):
        for si, s in enumerate(SET_ORDER):
            cell = data[m][s]
            head = r"\multirow{3}{*}{" + MODEL_LONG[m] + "}" if si == 0 else ""
            rate = fnum(cell["rigor_rate"], 4)
            bid = cell["byte_id"]
            btot = cell["byte_total"]
            mrate = fnum(bid / btot, 4)
            lines.append(f"{head} & {SET_SHORT[s]} & {cell['rigor_n']} & {rate} & {bid}/{btot} & {mrate} " + r"\\")
        if mi != len(MODEL_ORDER) - 1:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return write_output("tables/appendices/answer_preservation.tex", lines)


def build_perceptibility():
    art = load_artifact("perceptibility.json")
    data = art["cells"]
    benign = art["benign"]
    order = ["internvl3_5_2b", "gemma3_12b", "molmo2_4b", "llava_onevision_7b"]
    lines = [
        r"\small",
        r"\begin{tabular}{@{}llrrrr@{}}",
        r"\toprule",
        r"Model & Set & LPIPS mean & LPIPS max & SSIM mean & SSIM min \\",
        r"\midrule",
    ]
    for m in order:
        present = [s for s in SET_ORDER if s in data[m]]
        for s in present:
            cell = data[m][s]
            head = r"\multirow{" + str(len(present)) + r"}{*}{" + MODEL_LONG[m] + "}" if s == present[0] else ""
            lm = fnum(cell["lpips_mean"], 3)
            lx = fnum(cell["lpips_max"], 3)
            sm = fnum(cell["ssim_mean"], 3)
            si = fnum(cell["ssim_min"], 3)
            lines.append(f"{head} & {SET_SHORT[s]} & {lm} & {lx} & {sm} & {si} " + r"\\")
        lines.append(r"\midrule")
    lines.append(r"\multicolumn{2}{@{}l}{\emph{Benign ref: JPEG q95}} & "
                 + f"{fnum(benign['jpeg_lpips'], 3)} & -- & {fnum(benign['jpeg_ssim'], 3)} & -- " + r"\\")
    lines.append(r"\multicolumn{2}{@{}l}{\emph{Benign ref: 2px crop-resize}} & "
                 + f"{fnum(benign['crop_lpips'], 3)} & -- & {fnum(benign['crop_ssim'], 3)} & -- " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return write_output("tables/appendices/perceptibility.tex", lines)


def build_judge_validation():
    art = load_artifact("judge_validation.json")
    ov = art["overall"]
    lo, hi = ov["kappa_ci"]
    ci = r"{\scriptsize[" + f"{fnum(lo, 3)},{fnum(hi, 3)}" + r"]}"
    lines = [
        r"\begin{tabular}{@{}lrrl@{}}",
        r"\toprule",
        r"stratum & $n$ & agreement & Cohen $\kappa$ [95\% CI] / detail \\",
        r"\midrule",
        f"overall & {ov['n']} & {fnum(ov['agreement'], 3)} & {fnum(ov['kappa'], 2)} {ci} " + r"\\",
        r"\midrule",
    ]
    for ds in ["GQA", "VQAv2", "POPE"]:
        c = art["per_dataset"][ds]
        lines.append(f"{ds} & {c['n']} & {fnum(c['agreement'], 3)} & n/a " + r"\\")
    lines.append(r"\midrule")
    for m in MODEL_ORDER:
        c = art["per_model"][m]
        lines.append(f"{MODEL_LONG[m]} & {c['n']} & {fnum(c['agreement'], 3)} & n/a " + r"\\")
    w = art["wrong"]
    lines.append(f"human-wrong & {w['n']} & {w['n_agree']}/{w['n']} agree & false-accept {w['false_accept']} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return write_output("tables/appendices/judge_validation.tex", lines)


def build_canonical_draw():
    data = load_artifact("canonical_draw.json")["cells"]
    lines = [
        r"\small",
        r"\begin{tabular}{@{}llc r@{}}",
        r"\toprule",
        r"model & set & regenerated draw matches pool & pool size \\",
        r"\midrule",
    ]
    for mi, m in enumerate(MODEL_ORDER):
        for si, s in enumerate(SET_ORDER):
            cell = data[m][s]
            head = r"\multirow{3}{*}{" + MODEL_LONG[m] + "}" if si == 0 else ""
            lines.append(f"{head} & {SET_SHORT[s]} & {cell['match']} & {cell['pool_size']}/{cell['target']} " + r"\\")
        if mi != len(MODEL_ORDER) - 1:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return write_output("tables/appendices/canonical_draw.tex", lines)


def build_hyperparameters():
    with open(os.path.join(REPO_ROOT, "configs", "hyperparameters.yaml")) as f:
        cfg = yaml.safe_load(f)
    lines = []
    if cfg.get("size"):
        lines.append("\\" + cfg["size"])
    lines.append(r"\begin{tabular}{" + cfg["colspec"] + "}")
    lines.append(r"\toprule")
    lines.append(" & ".join(cfg["header"]) + r" \\")
    lines.append(r"\midrule")
    groups = cfg["groups"]
    for gi, g in enumerate(groups):
        lines.append(r"\multicolumn{4}{@{}l}{\textbf{" + g["name"] + r"}} \\")
        for row in g["rows"]:
            lines.append(" & ".join(row) + r" \\")
        if gi != len(groups) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return write_output("tables/appendices/hyperparameters.tex", lines)


BUILDERS = {
    "perturbation_norms": build_perturbation_norms,
    "answer_preservation": build_answer_preservation,
    "perceptibility": build_perceptibility,
    "judge_validation": build_judge_validation,
    "canonical_draw": build_canonical_draw,
    "hyperparameters": build_hyperparameters,
}
