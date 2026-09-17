import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.analysis.latexlib import MODEL_ORDER, SET_ORDER, ARTIFACTS, OUTPUTS, load_artifact

SWEEP_BETAS = ["0.3", "1", "3", "10"]


def _points():
    ut = load_artifact("frontier_untrained.json")
    untrained = ut["points"]
    sweep = load_artifact("frontier_sweep.json")["cells"]
    s1 = [(round(p["clean"], 3), round(p["attacked"], 3)) for p in untrained]
    s2 = []
    for m in MODEL_ORDER:
        for s in SET_ORDER:
            c = sweep[m][s]
            s2.append((round(c["peak_clean"], 3), round(c["peak_att"], 3)))
    g = sweep["gemma3_12b"]["gqa_val_eval"]["grid"]
    s2b = [(round(g[b]["clean"], 3), round(g[b]["attacked"], 3)) for b in SWEEP_BETAS]
    bqa = round(ut["b_qa"], 3)
    return s1, s2, s2b, bqa


def build_frontier():
    s1, s2, s2b, bqa = _points()
    coords = {"untrained": s1, "adv_trained": s2, "gemma_gqa_sweep": s2b, "b_qa": [bqa, bqa]}
    with open(os.path.join(ARTIFACTS, "frontier_points.json"), "w") as f:
        json.dump(coords, f, indent=2)
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.axhspan(0.55, 0.68, xmin=(0.52 - 0.38) / (0.90 - 0.38), color=(0.231, 0.435, 0.714), alpha=0.08)
    ax.axhline(0.50, color="gray", lw=1.2)
    ax.axhline(0.55, color="gray", lw=0.8, ls=":")
    ax.plot([p[0] for p in s1], [p[1] for p in s1], marker="o", ls="none", ms=4, color=(0.153, 0.294, 0.454), label="untrained")
    ax.plot([p[0] for p in s2b], [p[1] for p in s2b], marker="^", ls="--", ms=4, lw=0.8, color=(0.710, 0.396, 0.114), label="Gemma/GQA sweep (1 cell)")
    ax.plot([p[0] for p in s2], [p[1] for p in s2], marker="s", ls="none", ms=4.5, color=(0.776, 0.447, 0.031), label="adv-trained (strongest)")
    ax.plot([bqa], [bqa], marker="*", ls="none", ms=11, color=(0.58, 0.0, 0.83), label="b_qa (image-invariant)")
    ax.set_xlim(0.38, 0.90)
    ax.set_ylim(-0.03, 0.68)
    ax.set_xlabel("Clean AUROC")
    ax.set_ylabel("Attacked AUROC")
    ax.grid(True, color="0.9")
    ax.legend(loc="upper center", ncol=2, fontsize=8, framealpha=1.0)
    fig.tight_layout()
    out = os.path.join(OUTPUTS, "figures", "frontier.pdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


BUILDERS = {"frontier": build_frontier}
