"""Plot Stage 1 matched-budget gap closed by UTC, dropped mass, and hybrid."""

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STAGE1_PATH = ROOT / "scripts" / "stage1_evaluate_value_proxies.py"
RESULTS_DIR = ROOT / "results"


def load_stage1():
    spec = importlib.util.spec_from_file_location("stage1_evaluate_value_proxies", STAGE1_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    stage1 = load_stage1()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    q_scales = [0.25, 0.5, 1.0, 2.0, 4.0]
    rows = stage1.run_utc_allocation_sweep(q_scales, target_mean_k=40)
    rows = sorted(rows, key=lambda row: row["H_norm_mean"])

    h_norm = np.array([row["H_norm_mean"] for row in rows])
    mass_gap = np.array([row["mass_gap_closed"] for row in rows])
    utc_gap = np.array([row["utc_gap_closed"] for row in rows])
    hybrid_gap = np.array([row["hybrid_gap_closed"] for row in rows])
    labels = [f"q={row['q_scale']:g}" for row in rows]

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot(h_norm, mass_gap, "o-", label="dropped-mass adaptive", linewidth=2)
    ax.plot(h_norm, utc_gap, "o-", label="UTC value proxy", linewidth=2)
    ax.plot(h_norm, hybrid_gap, "o-", label="entropy-routed hybrid", linewidth=2)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.9)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.9)

    label_offsets = {
        "q=0.25": (4, 10),
        "q=0.5": (4, 8),
        "q=1": (4, 8),
        "q=2": (4, 8),
        "q=4": (4, -16),
    }
    for x, y, label in zip(h_norm, hybrid_gap, labels):
        ax.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=label_offsets.get(label, (4, 6)),
            fontsize=8,
        )

    ax.set_xlabel("mean normalized entropy H(p) / log(n)")
    ax.set_ylabel("fraction of fixed-to-oracle gap closed")
    ax.set_title(
        "Entropy-routed hybrid combines UTC and dropped-mass regimes\n"
        "synthetic single seed, target mean k=40, router H_norm=0.94"
    )
    ax.set_ylim(-0.15, 1.1)
    ax.invert_xaxis()
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax.text(
        0.02,
        -0.18,
        "diffuse / high entropy",
        ha="left",
        va="top",
        fontsize=9,
        transform=ax.transAxes,
    )
    ax.text(
        0.98,
        -0.18,
        "sharp / low entropy",
        ha="right",
        va="top",
        fontsize=9,
        transform=ax.transAxes,
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    out_path = RESULTS_DIR / "stage1_utc_vs_mass_gap_closed.png"
    fig.savefig(out_path, dpi=160)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
