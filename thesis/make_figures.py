"""Build every figure the thesis embeds.

The numbers here are copied from REVIEW_14CLASS.md, which is the running
record of the experiments; nothing is invented for the sake of a nicer
curve. Labels inside the figures are Latin so the plots do not depend on
a Persian font being installed on whatever machine rebuilds them -- the
captions in the thesis itself are Persian.

    python thesis/make_figures.py

Writes PNG at 300 dpi into thesis/figures/.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "figures"
DPI = 300

INK = "#1f2933"
ACCENT = "#2f6690"
ACCENT2 = "#c05746"
MUTED = "#8c9aa5"


def _style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.name}")


def _box(ax, x, y, w, h, text, face="#eef3f7", edge=ACCENT, size=8):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.1, facecolor=face, edgecolor=edge,
    ))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=size, color=INK, linespacing=1.45)


def _arrow(ax, start, end, style="-|>"):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=11,
        linewidth=1.1, color=MUTED, shrinkA=2, shrinkB=2,
    ))


def pipeline():
    """Frozen encoder -> cached features -> temporal MIL head."""
    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _box(ax, 0.02, 0.60, 0.20, 0.24,
         "Surveillance video\n(untrimmed, one\nvideo-level label)")
    _box(ax, 0.27, 0.60, 0.20, 0.24,
         "16 clips\n16 frames each\n224x224")
    _box(ax, 0.52, 0.60, 0.21, 0.24,
         "VideoMAE ViT-L\n(frozen, Kinetics)\ntoken mean-pool",
         face="#f5efe6", edge=ACCENT2)
    _box(ax, 0.78, 0.60, 0.20, 0.24,
         "cached features\n[16, 1024]\none .npy / video")

    _arrow(ax, (0.22, 0.72), (0.27, 0.72))
    _arrow(ax, (0.47, 0.72), (0.52, 0.72))
    _arrow(ax, (0.73, 0.72), (0.78, 0.72))

    ax.text(0.50, 0.92, "Stage 1  -  run once, 150 min for 1895 videos",
            ha="center", fontsize=8.5, color=MUTED, style="italic")

    _box(ax, 0.02, 0.14, 0.20, 0.24,
         "LayerNorm + Linear\n+ positional\nembedding")
    _box(ax, 0.27, 0.14, 0.20, 0.24,
         "Transformer\nencoder over\nthe 16 clips")
    _box(ax, 0.52, 0.21, 0.21, 0.17, "clip logits\n[16, 14]")
    _box(ax, 0.78, 0.21, 0.20, 0.17,
         "top-k MIL pooling\n-> video logits",
         face="#f5efe6", edge=ACCENT2)

    _arrow(ax, (0.22, 0.26), (0.27, 0.26))
    _arrow(ax, (0.47, 0.26), (0.52, 0.29))
    _arrow(ax, (0.73, 0.29), (0.78, 0.29))

    # cached features feed the head: down, across, then into stage 2
    ax.plot([0.88, 0.88], [0.60, 0.50], color=MUTED, linewidth=1.1)
    ax.plot([0.88, 0.12], [0.50, 0.50], color=MUTED, linewidth=1.1)
    _arrow(ax, (0.12, 0.50), (0.12, 0.38))
    ax.text(0.50, 0.515, "features reloaded from disk each epoch",
            fontsize=7.5, color=MUTED, ha="center")

    ax.text(0.50, 0.05,
            "Stage 2  -  trains in ~80 s, so hyper-parameters can be swept",
            ha="center", fontsize=8.5, color=MUTED, style="italic")

    _save(fig, "fig_pipeline.png")


def two_stage():
    """The evaluation-time decision that replaced mean pooling."""
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _box(ax, 0.03, 0.62, 0.26, 0.26,
         "per-clip softmax\nP(c | clip), c = 0..13")
    _box(ax, 0.37, 0.62, 0.26, 0.26,
         "anomaly evidence\ns = max over clips\nof 1 - P(Normal)",
         face="#f5efe6", edge=ACCENT2)
    _box(ax, 0.71, 0.62, 0.26, 0.26,
         "s < threshold ?", face="#eef3f7")

    _arrow(ax, (0.29, 0.75), (0.37, 0.75))
    _arrow(ax, (0.63, 0.75), (0.71, 0.75))

    _box(ax, 0.71, 0.16, 0.26, 0.20, "predict Normal",
         face="#eaf1e8", edge="#5b7d55")
    _arrow(ax, (0.84, 0.62), (0.84, 0.36))
    ax.text(0.865, 0.49, "yes", fontsize=8, color=MUTED)

    _box(ax, 0.08, 0.16, 0.52, 0.20,
         "average P over the top-k most anomalous clips,\n"
         "argmax over classes 1..13",
         face="#f5efe6", edge=ACCENT2)
    _arrow(ax, (0.73, 0.62), (0.61, 0.31))
    ax.text(0.655, 0.44, "no", fontsize=8, color=MUTED)

    ax.text(0.50, 0.03,
            "Stage 1 answers \"is it anomalous\", stage 2 answers \"which anomaly\"",
            ha="center", fontsize=8.5, color=MUTED, style="italic")

    _save(fig, "fig_two_stage.png")


def macro_f1_progress():
    labels = [
        "MLP head\nViT-B",
        "temporal head\nViT-B",
        "+ lr 3e-4",
        "+ ViT-L\nfeatures",
        "+ mixup 0.2\n+ logit adjust",
    ]
    values = [0.2975, 0.3444, 0.3645, 0.4122, 0.4298]

    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    colors = [MUTED] * 4 + [ACCENT2]
    colors[3] = ACCENT
    bars = ax.bar(labels, values, color=colors, width=0.58)

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.006,
                f"{v:.4f}", ha="center", fontsize=8.5)

    ax.axhline(0.3881, color=ACCENT2, linestyle="--", linewidth=1)
    ax.text(4.42, 0.3905, "X-CLIP fine-tuned\n(0.3881)", fontsize=7.5,
            color=ACCENT2, va="bottom", ha="right")

    ax.set_ylabel("macro F1")
    ax.set_ylim(0, 0.50)
    ax.tick_params(axis="x", labelsize=8)
    _save(fig, "fig_macro_f1_progress.png")


def backbone_compare():
    metrics = ["binary AUC", "accuracy", "type accuracy", "macro F1"]
    xclip = [0.9629, 0.6419, 0.4257, 0.3881]
    mae_b = [0.9649, 0.6552, 0.4214, 0.3645]
    mae_l = [0.9610, 0.6897, 0.5000, 0.4298]

    x = range(len(metrics))
    w = 0.26

    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    b1 = ax.bar([i - w for i in x], xclip, w, label="X-CLIP fine-tuned",
                color=MUTED)
    b2 = ax.bar(list(x), mae_b, w, label="VideoMAE ViT-B (frozen)",
                color=ACCENT)
    b3 = ax.bar([i + w for i in x], mae_l, w,
                label="VideoMAE ViT-L (frozen, final)", color=ACCENT2)

    for group in (b1, b2, b3):
        for bar in group:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.012,
                    f"{bar.get_height():.3f}", ha="center", fontsize=7)

    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("score")
    ax.legend(frameon=False, fontsize=8, ncol=1, loc="upper right")
    _save(fig, "fig_backbone_compare.png")


def bootstrap_ci():
    names = ["binary AUC", "accuracy", "type accuracy", "macro F1"]
    point = [0.9610, 0.6897, 0.5000, 0.4298]
    lo = [0.938, 0.635, 0.420, 0.339]
    hi = [0.980, 0.745, 0.585, 0.503]

    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    y = list(range(len(names)))[::-1]

    for yi, p, a, b in zip(y, point, lo, hi):
        ax.plot([a, b], [yi, yi], color=MUTED, linewidth=2.4,
                solid_capstyle="butt")
        ax.plot([a, a], [yi - 0.14, yi + 0.14], color=MUTED, linewidth=1.4)
        ax.plot([b, b], [yi - 0.14, yi + 0.14], color=MUTED, linewidth=1.4)
        ax.plot([p], [yi], "o", color=ACCENT2, markersize=6, zorder=3)
        ax.text(b + 0.012, yi, f"{p:.4f}  [{a:.3f}, {b:.3f}]",
                va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(0.25, 1.30)
    ax.set_xlabel("value (4000 bootstrap resamples of the test videos)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    _save(fig, "fig_bootstrap_ci.png")


def seed_ranges():
    fig, ax = plt.subplots(figsize=(6.0, 2.6))

    groups = [
        ("VideoMAE ViT-B\n(5 seeds)", 0.310, 0.354, 0.3405, MUTED),
        ("VideoMAE ViT-L\n(5 seeds)", 0.370, 0.423, 0.3955, ACCENT2),
    ]

    for i, (name, lo, hi, mean, colour) in enumerate(groups):
        y = 1 - i
        ax.plot([lo, hi], [y, y], color=colour, linewidth=8, alpha=0.30,
                solid_capstyle="round")
        ax.plot([mean], [y], "D", color=colour, markersize=7, zorder=3)
        ax.text(hi + 0.006, y, f"mean {mean:.4f}", va="center", fontsize=8.5)
        ax.text(lo - 0.006, y, f"{lo:.3f}", va="center", ha="right",
                fontsize=7.5, color=MUTED)

    ax.set_yticks([1, 0])
    ax.set_yticklabels([g[0] for g in groups], fontsize=8.5)
    ax.set_xlim(0.28, 0.48)
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel("macro F1 per seed (range)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.text(0.38, -0.45, "the two ranges do not overlap  (Mann-Whitney p = 1/252)",
            fontsize=8, color=ACCENT2, ha="center", style="italic")
    _save(fig, "fig_seed_ranges.png")


def mixup_logit():
    labels = ["baseline", "mixup 0.2", "mixup 0.2\n+ adjust",
              "mixup 0.4\n+ adjust", "mixup 0.4"]
    macro = [0.4122, 0.4161, 0.4298, 0.4248, 0.4025]
    typea = [0.4786, 0.4714, 0.5000, 0.4929, 0.4643]

    x = range(len(labels))
    w = 0.36

    fig, ax = plt.subplots(figsize=(6.6, 3.1))
    ax.bar([i - w / 2 for i in x], macro, w, label="macro F1", color=ACCENT)
    ax.bar([i + w / 2 for i in x], typea, w, label="anomaly-type accuracy",
           color=MUTED)

    for i, (m, t) in enumerate(zip(macro, typea)):
        ax.text(i - w / 2, m + 0.008, f"{m:.4f}", ha="center", fontsize=7.5)
        ax.text(i + w / 2, t + 0.008, f"{t:.4f}", ha="center", fontsize=7.5)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 0.62)
    ax.set_ylabel("score")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _save(fig, "fig_mixup_logit.png")


def noise_floor():
    """Why a small gain on a 290-video test set proves nothing."""
    fig, ax = plt.subplots(figsize=(6.2, 2.9))

    n = list(range(2, 260))
    sigma = 0.034
    import math
    inflation = [sigma * math.sqrt(2 * math.log(k)) for k in n]

    ax.plot(n, inflation, color=ACCENT, linewidth=1.6)
    for k, note in ((8, "8 runs"), (200, "200 runs")):
        v = sigma * math.sqrt(2 * math.log(k))
        ax.plot([k], [v], "o", color=ACCENT2, markersize=5, zorder=3)
        ax.annotate(f"{note}\n+{v:.3f}", (k, v), textcoords="offset points",
                    xytext=(6, -14), fontsize=8, color=ACCENT2)

    ax.set_xlabel("number of evaluations against the same test split")
    ax.set_ylabel("expected inflation\nof the best score")
    ax.set_xlim(0, 260)
    ax.set_ylim(0, 0.14)
    _save(fig, "fig_noise_floor.png")


def main():
    _style()
    print("writing figures:")
    pipeline()
    two_stage()
    macro_f1_progress()
    backbone_compare()
    bootstrap_ci()
    seed_ranges()
    mixup_logit()
    noise_floor()


if __name__ == "__main__":
    main()
