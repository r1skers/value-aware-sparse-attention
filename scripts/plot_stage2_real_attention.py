"""Plot Stage 2/3 real-attention summary figures.

Outputs:
  - results/stage2_real_attention_method_ladder.png
  - results/stage2_rel_hat_starvation_failures.png
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
BERT_CSV = RESULTS / "bert" / "stage2c_bert_exact_budget.csv"
GPT2_CSV = RESULTS / "gpt2" / "stage3_gpt2_exact_budget.csv"
FAIL_CSV = RESULTS / "bert" / "stage2c_rel_hat_failure_diagnosis.csv"


METHODS = [
    ("mass", "Mass"),
    ("UTC_abs", "UTC-abs"),
    ("UTC_rel", "UTC-rel"),
    ("UTC_rel_hat", "UTC-rel-hat"),
]


COLORS = {
    "BERT": "#3B6EA8",
    "GPT-2": "#D17A22",
    "Mass": "#7A7A7A",
    "UTC-abs": "#5B8E7D",
    "UTC-rel": "#B65F82",
    "UTC-rel-hat": "#2F6F3E",
}


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def gap_values(rows, method_key):
    return np.array([float(row[f"{method_key}_gap"]) for row in rows], dtype=np.float64)


def summarize_model(rows):
    summary = {}
    for key, label in METHODS:
        values = gap_values(rows, key)
        summary[label] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p10": float(np.quantile(values, 0.10)),
            "p90": float(np.quantile(values, 0.90)),
        }
    return summary


def plot_method_ladder():
    bert = summarize_model(read_rows(BERT_CSV))
    gpt2 = summarize_model(read_rows(GPT2_CSV))
    models = [("BERT", bert), ("GPT-2", gpt2)]
    labels = [label for _, label in METHODS]
    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(10.6, 5.8), dpi=180)
    for offset, (model_name, summary) in zip([-width / 2, width / 2], models):
        means = [summary[label]["mean"] for label in labels]
        p10 = [summary[label]["p10"] for label in labels]
        bars = ax.bar(
            x + offset,
            means,
            width,
            label=f"{model_name} mean",
            color=COLORS[model_name],
            alpha=0.84,
            edgecolor="white",
            linewidth=1.0,
        )
        ax.scatter(
            x + offset,
            p10,
            color="#171717",
            s=26,
            zorder=4,
            label=f"{model_name} p10" if model_name == "BERT" else None,
        )
        for bar, mean in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                mean + (0.035 if mean >= 0 else -0.055),
                f"{mean:.2f}",
                ha="center",
                va="bottom" if mean >= 0 else "top",
                fontsize=9,
            )

    ax.axhline(0.0, color="#2A2A2A", linewidth=1.0)
    ax.axhline(1.0, color="#2A2A2A", linewidth=0.9, linestyle=":", alpha=0.7)
    ax.text(
        len(labels) - 0.35,
        1.015,
        "restricted oracle",
        ha="right",
        va="bottom",
        fontsize=9,
        color="#333333",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Gap closed vs fixed-k")
    ax.set_title("Exact-budget method ladder transfers from BERT to GPT-2")
    ax.set_ylim(-0.36, 1.08)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.8, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left", ncol=2)
    ax.text(
        0.01,
        -0.18,
        "Bars show mean gap closed; black dots show the 10th percentile. Higher is better.",
        transform=ax.transAxes,
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout()
    out = RESULTS / "stage2_real_attention_method_ladder.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_failure_starvation():
    rows = read_rows(FAIL_CSV)
    class_order = [
        "denom_overestimate_starvation",
        "numerator_underestimate_starvation",
        "mild_score_miscalibration_starvation",
    ]
    class_labels = {
        "denom_overestimate_starvation": "denom\noverestimate",
        "numerator_underestimate_starvation": "numerator\nunderestimate",
        "mild_score_miscalibration_starvation": "mild score\nmiscalibration",
    }
    class_colors = {
        "denom_overestimate_starvation": "#C44E52",
        "numerator_underestimate_starvation": "#4C72B0",
        "mild_score_miscalibration_starvation": "#55A868",
    }

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.2, 5.4),
        dpi=180,
        gridspec_kw={"width_ratios": [1.2, 0.9]},
    )
    ax = axes[0]
    for cls in class_order:
        subset = [row for row in rows if row["failure_class"] == cls]
        x = np.array([int(row["k_oracle"]) for row in subset])
        y = np.array([int(row["k_rel_hat"]) for row in subset])
        budgets = np.array([int(row["budget"]) for row in subset])
        sizes = 48 + budgets * 3
        ax.scatter(
            x,
            y,
            s=sizes,
            color=class_colors[cls],
            alpha=0.86,
            edgecolor="white",
            linewidth=0.8,
            label=class_labels[cls].replace("\n", " "),
        )

    max_k = max(int(row["k_oracle"]) for row in rows) + 2
    ax.plot([0, max_k], [0, max_k], color="#222222", linestyle="--", linewidth=1.0)
    ax.text(
        max_k - 1,
        max_k - 0.5,
        "y = x",
        ha="right",
        va="bottom",
        fontsize=9,
        color="#333333",
    )
    ax.set_xlim(0, max_k)
    ax.set_ylim(0, max_k)
    ax.set_xlabel("Restricted-oracle k on worst row")
    ax.set_ylabel("UTC-rel-hat k on worst row")
    ax.set_title("All below-fixed failures are row-starvation cases")
    ax.grid(color="#E0E0E0", linewidth=0.8, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right", fontsize=8)

    ax2 = axes[1]
    counts = [sum(row["failure_class"] == cls for row in rows) for cls in class_order]
    bars = ax2.bar(
        np.arange(len(class_order)),
        counts,
        color=[class_colors[cls] for cls in class_order],
        alpha=0.9,
        edgecolor="white",
        linewidth=1.0,
    )
    ax2.set_xticks(np.arange(len(class_order)))
    ax2.set_xticklabels([class_labels[cls] for cls in class_order], fontsize=9)
    ax2.set_ylabel("Failure count")
    ax2.set_title("Failure subtypes split evenly")
    ax2.set_ylim(0, max(counts) + 2.5)
    ax2.grid(axis="y", color="#E0E0E0", linewidth=0.8, alpha=0.8)
    ax2.spines[["top", "right"]].set_visible(False)
    for bar, count in zip(bars, counts):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            count + 0.25,
            str(count),
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax2.text(
        0.02,
        -0.22,
        "The failed max-error row always has k_rel-hat < k_oracle (21/21).",
        transform=ax2.transAxes,
        fontsize=9,
        color="#444444",
    )

    fig.tight_layout()
    out = RESULTS / "stage2_rel_hat_starvation_failures.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    ladder = plot_method_ladder()
    failures = plot_failure_starvation()
    print(f"saved: {ladder}")
    print(f"saved: {failures}")


if __name__ == "__main__":
    main()
