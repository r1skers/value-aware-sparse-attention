"""Final figures for Artifact 6.3/6.4.

This script only visualizes existing experiment outputs. It does not rerun
models or allocations.

Outputs:
  results/final_figures/fig_63_exact_budget_ecdf.png
  results/final_figures/fig_64_metric_boundary_triptych.png
  results/final_figures/fig_63_bert_hnorm_heatmap.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT_DIR = RESULTS / "final_figures"

BERT_EXACT = RESULTS / "bert" / "stage2c_bert_exact_budget.csv"
GPT2_EXACT = RESULTS / "gpt2" / "stage3_gpt2_exact_budget.csv"
BERT_HNORM = RESULTS / "bert" / "hnorm_census.npy"
STAGE4A_GPT2 = RESULTS / "metric_boundary" / "stage4a_gpt2_wo_projected.csv"
STAGE4B = RESULTS / "metric_boundary" / "stage4b_gpt2_logit_kl.csv"
STAGE4C = RESULTS / "metric_boundary" / "stage4c_gpt2_whole_layer_kl.csv"

METHODS = [
    ("mass", "Mass"),
    ("UTC_abs", "UTC-abs"),
    ("UTC_rel", "UTC-rel"),
    ("UTC_rel_hat", "UTC-rel-hat"),
]

COLORS = {
    "Mass": "#7A7A7A",
    "UTC-abs": "#4C78A8",
    "UTC-rel": "#B279A2",
    "UTC-rel-hat": "#2F7D46",
    "Projected oracle": "#D55E00",
    "Fixed": "#222222",
}


def set_style():
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 14,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#DDDDDD",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.75,
        }
    )


def ecdf(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    x = np.sort(values)
    y = np.arange(1, x.size + 1) / x.size
    return x, y


def plot_exact_budget_ecdf():
    bert = pd.read_csv(BERT_EXACT)
    gpt2 = pd.read_csv(GPT2_EXACT)
    panels = [("BERT exact-budget", bert), ("GPT-2 exact-budget", gpt2)]

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2), dpi=180, sharey=True)
    for ax, (title, df) in zip(axes, panels):
        for key, label in METHODS:
            x, y = ecdf(df[f"{key}_gap"])
            ax.plot(x, y, label=label, color=COLORS[label], linewidth=2.0)

        ax.axvline(0.0, color="#222222", linewidth=1.0, linestyle="--", alpha=0.75)
        ax.axvline(1.0, color="#222222", linewidth=1.0, linestyle=":", alpha=0.75)
        ax.set_title(title)
        ax.set_xlabel("Gap closed vs fixed-k")
        ax.set_xlim(-1.25, 1.08)
        ax.set_ylim(0.0, 1.0)
        ax.text(0.02, 0.94, f"n = {len(df)}", transform=ax.transAxes, color="#444444")

        for key, label in METHODS:
            vals = df[f"{key}_gap"].to_numpy(dtype=np.float64)
            p10 = np.nanquantile(vals, 0.10)
            median = np.nanmedian(vals)
            if label == "UTC-rel-hat":
                ax.scatter([p10, median], [0.10, 0.50], color=COLORS[label], s=34, zorder=4)
                ax.annotate(
                    "rel-hat p10 / median",
                    xy=(median, 0.50),
                    xytext=(0.30, 0.62),
                    textcoords="axes fraction",
                    arrowprops={"arrowstyle": "->", "color": COLORS[label], "lw": 1.0},
                    color=COLORS[label],
                    fontsize=9,
                )

    axes[0].set_ylabel("ECDF")
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("Real-attention exact-budget distributions")
    fig.text(
        0.5,
        0.01,
        "Higher is better. Values below 0 are worse than fixed-k; 1 is the restricted-oracle reference.",
        ha="center",
        color="#444444",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    out = OUT_DIR / "fig_63_exact_budget_ecdf.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def aggregate_reduction(df, method, metric="mean_kl"):
    fixed = df[f"fixed_{metric}"].sum()
    value = df[f"{method}_{metric}"].sum()
    return 1.0 - value / fixed


def stage4_values():
    gpt2_exact = pd.read_csv(GPT2_EXACT)
    gpt2_wo = pd.read_csv(STAGE4A_GPT2)
    s4b = pd.read_csv(STAGE4B)
    s4c = pd.read_csv(STAGE4C)

    rel_hat = {
        "Head output": float(gpt2_exact["UTC_rel_hat_gap"].mean()),
        "W_O projected": float(gpt2_wo["UTC_rel_hat_gap"].mean()),
        "Single-head KL": aggregate_reduction(s4b, "UTC_rel_hat"),
        "Whole-layer KL": aggregate_reduction(s4c, "UTC_rel_hat"),
    }
    oracle = {
        "Head output": 1.0,
        "W_O projected": 1.0,
        "Single-head KL": aggregate_reduction(s4b, "projected_oracle"),
        "Whole-layer KL": aggregate_reduction(s4c, "projected_oracle"),
    }
    utc_abs = {
        "Head output": float(gpt2_exact["UTC_abs_gap"].mean()),
        "W_O projected": float(gpt2_wo["UTC_abs_gap"].mean()),
        "Single-head KL": aggregate_reduction(s4b, "UTC_abs"),
        "Whole-layer KL": aggregate_reduction(s4c, "UTC_abs"),
    }
    return rel_hat, utc_abs, oracle, s4b, s4c


def plot_metric_boundary_triptych():
    rel_hat, utc_abs, oracle, s4b, s4c = stage4_values()
    stages = list(rel_hat.keys())
    x = np.arange(len(stages))

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15.0, 5.3),
        dpi=180,
        gridspec_kw={"width_ratios": [1.15, 0.95, 1.15]},
    )

    ax = axes[0]
    # the metric FAMILY changes between rung 2 and rung 3: local rungs report
    # mean gap closed (local oracle = 1 by definition), behavioral rungs report
    # aggregate KL reduction. Draw solid segments within each family and a
    # dashed segment across the boundary so the switch is visually honest.
    for series, label, lw in [
        (oracle, "Local oracle", 2.4),
        (rel_hat, "UTC-rel-hat", 2.4),
        (utc_abs, "UTC-abs", 2.0),
    ]:
        color = COLORS["Projected oracle"] if label == "Local oracle" else COLORS[label]
        vals = [series[s] for s in stages]
        ax.plot(x[:2], vals[:2], marker="o", linewidth=lw, color=color, label=label)
        ax.plot(x[1:3], vals[1:3], linewidth=lw, color=color, linestyle=(0, (2, 3)), alpha=0.8)
        ax.plot(x[2:], vals[2:], marker="o", linewidth=lw, color=color)
    ax.axvspan(1.5, 1.62, color="#BBBBBB", alpha=0.6, zorder=0)
    ax.text(1.56, 1.06, "metric family changes", rotation=90, fontsize=8,
            color="#555555", ha="center", va="top")
    ax.axhline(0.0, color="#222222", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(stages, rotation=25, ha="right")
    ax.set_ylabel("Mean gap closed  |  aggregate KL reduction")
    ax.set_title("Metric ladder")
    ax.set_ylim(-0.65, 1.12)
    ax.legend(frameon=False, loc="lower left")
    ax.annotate(
        "$\\equiv$ 1 by definition",
        xy=(0.5, 1.0),
        xytext=(0.15, 0.83),
        arrowprops={"arrowstyle": "->", "color": COLORS["Projected oracle"], "lw": 1.0},
        color=COLORS["Projected oracle"],
        fontsize=9,
    )
    ax.annotate(
        "local oracle stops\nbeing behavioral oracle",
        xy=(3, oracle["Whole-layer KL"]),
        xytext=(1.75, -0.47),
        arrowprops={"arrowstyle": "->", "color": COLORS["Projected oracle"], "lw": 1.1},
        color=COLORS["Projected oracle"],
        fontsize=9,
    )

    ax = axes[1]
    labels = ["single-head\nKL", "whole-layer\nKL"]
    oracle_vals = [oracle["Single-head KL"], oracle["Whole-layer KL"]]
    rel_vals = [rel_hat["Single-head KL"], rel_hat["Whole-layer KL"]]
    width = 0.34
    xpos = np.arange(2)
    ax.bar(xpos - width / 2, oracle_vals, width, color=COLORS["Projected oracle"], label="local oracle", alpha=0.9)
    ax.bar(xpos + width / 2, rel_vals, width, color=COLORS["UTC-rel-hat"], label="rel-hat", alpha=0.9)
    ax.axhline(0.0, color="#222222", linewidth=1.0)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.set_title("Oracle inversion")
    ax.set_ylabel("Aggregate KL reduction")
    ax.set_ylim(-0.62, 0.26)
    ax.legend(frameon=False, loc="lower left")
    for i, val in enumerate(oracle_vals):
        ax.text(i - width / 2, val - 0.035, f"{val:.2f}", ha="center", va="top", fontsize=9)
    for i, val in enumerate(rel_vals):
        ax.text(i + width / 2, val + 0.025, f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    ax = axes[2]
    by_layer = []
    for layer, g in s4c.groupby("layer"):
        by_layer.append(
            {
                "layer": int(layer),
                "oracle": aggregate_reduction(g, "projected_oracle"),
                "UTC_abs": aggregate_reduction(g, "UTC_abs"),
                "UTC_rel_hat": aggregate_reduction(g, "UTC_rel_hat"),
            }
        )
    by_layer = sorted(by_layer, key=lambda d: d["layer"])
    lx = np.arange(len(by_layer))
    # clip the oracle's L11 outlier (-3.19) so rel-hat's own depth gradient —
    # a sign flip from +0.11 at L0 to -0.21 at L11 — stays visible
    clip_floor = -0.62
    oracle_vals = [d["oracle"] for d in by_layer]
    oracle_clipped = [max(v, clip_floor) for v in oracle_vals]
    ax.plot(lx, oracle_clipped, marker="o", linewidth=2.4, color=COLORS["Projected oracle"], label="local oracle")
    ax.plot(lx, [d["UTC_abs"] for d in by_layer], marker="o", linewidth=2.0, color=COLORS["UTC-abs"], label="UTC-abs")
    ax.plot(lx, [d["UTC_rel_hat"] for d in by_layer], marker="o", linewidth=2.0, color=COLORS["UTC-rel-hat"], label="UTC-rel-hat")
    for xi, (v, c) in enumerate(zip(oracle_vals, oracle_clipped)):
        if v < clip_floor:
            ax.annotate(
                f"late-layer local optimum is anti-behavioral\n(true value {v:.2f}, clipped)",
                xy=(xi, c),
                xytext=(xi - 1.75, c + 0.28),
                arrowprops={"arrowstyle": "->", "color": COLORS["Projected oracle"], "lw": 1.1},
                color=COLORS["Projected oracle"],
                fontsize=9,
            )
    ax.axhline(0.0, color="#222222", linewidth=1.0)
    ax.set_xticks(lx)
    ax.set_xticklabels([f"L{d['layer']}" for d in by_layer])
    ax.set_title("Depth gradient under whole-layer KL")
    ax.set_ylabel("Aggregate KL reduction")
    ax.set_ylim(-0.72, 0.28)
    ax.legend(frameon=False, loc="lower left")

    fig.suptitle("Metric boundary: local error control stops at behavioral KL")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = OUT_DIR / "fig_64_metric_boundary_triptych.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_bert_hnorm_heatmap():
    hnorm = np.load(BERT_HNORM)
    if hnorm.ndim != 3:
        raise ValueError(f"expected hnorm shape (layer, head, row), got {hnorm.shape}")
    heat = np.nanmean(hnorm, axis=2)

    fig, ax = plt.subplots(figsize=(8.6, 6.6), dpi=180)
    im = ax.imshow(heat, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_title("BERT normalized-attention entropy census")
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_xticks(np.arange(12))
    ax.set_yticks(np.arange(12))

    for layer in range(heat.shape[0]):
        for head in range(heat.shape[1]):
            val = heat[layer, head]
            color = "white" if val < 0.45 else "black"
            ax.text(head, layer, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean H(p) / log n")
    ax.text(
        0.0,
        -0.12,
        "Darker cells are sharper heads; brighter cells are diffuse/high-entropy heads.",
        transform=ax.transAxes,
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout()
    out = OUT_DIR / "fig_63_bert_hnorm_heatmap.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_style()
    outputs = [
        plot_exact_budget_ecdf(),
        plot_metric_boundary_triptych(),
        plot_bert_hnorm_heatmap(),
    ]
    for path in outputs:
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
