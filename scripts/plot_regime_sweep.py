'''Closing figure for v1: how the dominant error signal shifts with entropy regime.'''

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STAGE0_PATH = ROOT / "scripts" / "stage0_sanity_check.py"
RESULTS_DIR = ROOT / "results"


def load_stage0():
    spec = importlib.util.spec_from_file_location("stage0_sanity_check", STAGE0_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    stage0 = load_stage0()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    q_scales = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    rows = stage0.run_q_scale_sweep(q_scales, N=128, d=64, k=30)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.plot(q_scales, [r["corr_entropy_error"] for r in rows], "o-", label="corr(entropy, error)")
    ax.plot(q_scales, [r["corr_delta_error"] for r in rows], "o-", label="corr(delta, error)")
    ax.plot(q_scales, [r["corr_centroid_error"] for r in rows], "o-", label="corr(centroid_dist, error)")
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xscale("log")
    ax.set_xticks(q_scales)
    ax.set_xticklabels([str(x) for x in q_scales])
    ax.set_xlabel("q_scale (log scale, larger = sharper attention)")
    ax.set_ylabel("Pearson correlation with true error")
    ax.set_title("Which signal explains row-to-row error, by regime", fontsize=10)
    ax.legend(fontsize=8)

    Q, K, V = stage0.gen_qkv(128, 64, seed=0, q_scale=1.0)
    P, _ = stage0.full_attention(Q, K, V)
    stats = stage0.check_all_rows_decomposition(P, V, k=30)
    H_norm = stage0.attention_entropy(P) / np.log(P.shape[-1])

    ax = axes[1]
    sc = ax.scatter(H_norm, stats["true_errors"], c=stats["centroid_dists"], cmap="viridis", s=28)
    ax.set_xlabel("normalized entropy H(p) / log(n)")
    ax.set_ylabel("true output error ||o - o~||")
    ax.set_title("q_scale=1.0 baseline: entropy alone doesn't cleanly separate error", fontsize=10)
    fig.colorbar(sc, ax=ax, label="||mu_R - mu_S||")

    fig.tight_layout()
    out_path = RESULTS_DIR / "regime_sweep_summary.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
