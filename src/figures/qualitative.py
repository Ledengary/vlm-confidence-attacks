import json
import os

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.analysis.latexlib import ARTIFACTS, OUTPUTS

QDIR = os.path.join(ARTIFACTS, "qualitative")


def _load_rgb(rel):
    return np.asarray(Image.open(os.path.join(QDIR, rel)).convert("RGB"), dtype=np.float64) / 255.0


def _wrap(s, n):
    out, line = [], ""
    for w in s.split():
        if len(line) + len(w) + 1 > n:
            out.append(line); line = w
        else:
            line = (line + " " + w).strip()
    if line:
        out.append(line)
    return out


def build_qualitative():
    spec = json.load(open(os.path.join(QDIR, "sample.json")))
    eps = spec["eps_over_255"] / 255.0
    rows = spec["rows"]
    nrows = len(rows)
    fig = plt.figure(figsize=(8.4, 2.35 * nrows))
    gs = fig.add_gridspec(nrows, 4, width_ratios=[1, 1, 1, 1.18], wspace=0.045, hspace=0.12,
                          left=0.006, right=0.994, top=0.95, bottom=0.02)
    titles = ["clean image", "adversarial image", r"perturbation ($\times %d$, centered)" % round(1.0 / (2 * eps)), ""]
    col = lambda v: (0.102, 0.490, 0.235) if v == "ACCEPT" else (0.702, 0.149, 0.118)

    for ri, r in enumerate(rows):
        clean = _load_rgb(r["clean_image"]); adv = _load_rgb(r["adv_image"])
        delta = adv - clean
        viz = np.clip(0.5 + delta / (2.0 * eps), 0.0, 1.0)
        linf = float(np.max(np.abs(delta)) * 255.0)
        for ci, img in enumerate((clean, adv, viz)):
            ax = fig.add_subplot(gs[ri, ci])
            ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor((0.533, 0.533, 0.533)); sp.set_linewidth(0.6)
            if ri == 0:
                ax.set_title(titles[ci], fontsize=9.5, pad=4)

        axt = fig.add_subplot(gs[ri, 3]); axt.axis("off")
        cf = r["clean_full"]; af = r["attacked_full"]; thr = r["tau"]
        gc = "ACCEPT" if cf >= thr else "REJECT"
        ga = "ACCEPT" if af >= thr else "REJECT"
        correct = bool(r["correct"])
        tag = "correct answer" if correct else "wrong answer"
        tagcol = (0.102, 0.490, 0.235) if correct else (0.702, 0.149, 0.118)
        harm = "correct answer now gated out" if correct else "wrong answer now admitted"

        ops = []
        ops.append(("t", f"{r['model_long']}  ·  {r['set_long']}", 0.086, dict(fs=10, fontweight="bold")))
        ops.append(("t", tag, 0.084, dict(fs=9.2, color=tagcol, fontweight="bold")))
        for ln in _wrap("Q: " + r["question"], 40):
            ops.append(("t", ln, 0.074, dict(fs=9.0)))
        ops.append(("t", f"A (clean = adv): “{r['answer']}”", 0.082, dict(fs=9.0, fontweight="bold")))
        ops.append(("t", "readout: FULL   gate: 10% clean-FA", 0.082, dict(fs=8.8, color=(0.200, 0.200, 0.200))))
        ops.append(("t", f"score  {cf:.3f}  →  {af:.3f}   (τ={thr:.3f})", 0.082, dict(fs=9.2)))
        ops.append(("gate", (gc, ga), 0.082, {}))
        ops.append(("t", harm, 0.080, dict(fs=8.6, style="italic", color=(0.333, 0.333, 0.333))))
        ops.append(("t", f"$\\ell_\\infty$={linf:.1f}/255  ·  byte-identical answer", 0.0, dict(fs=8.0, color=(0.467, 0.467, 0.467))))

        rend = fig.canvas.get_renderer()

        def seg(x, y, txt, **kw):
            t = axt.text(x, y, txt, transform=axt.transAxes, va="top", ha="left", fontsize=9.2, **kw)
            return x + t.get_window_extent(renderer=rend).width / axt.get_window_extent(renderer=rend).width

        total = sum(dy for _, _, dy, _ in ops)
        y = 0.5 + total / 2.0
        for kind, payload, dy, kw in ops:
            if kind == "t":
                axt.text(0.02, y, payload, transform=axt.transAxes, va="top", ha="left",
                         fontsize=kw.pop("fs", 9.2), **kw)
            else:
                g_c, g_a = payload
                x = seg(0.02, y, "gate  ")
                x = seg(x, y, g_c, fontweight="bold", color=col(g_c))
                x = seg(x, y, "  →  ")
                x = seg(x, y, g_a, fontweight="bold", color=col(g_a))
            y -= dy

    out = os.path.join(OUTPUTS, "figures", "qualitative_examples.pdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)
    return out


BUILDERS = {"qualitative_examples": build_qualitative}
