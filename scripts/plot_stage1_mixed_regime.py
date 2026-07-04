"""Stage 1 closing figure: budget-delegated hybrid on the mixed-regime population.

Left panel: matched budget (mean k=40) — gap closed per method, 3 seeds.
Right panel: budget sweep (20/40/60) at threshold 0.94 — mass vs UTC vs hybrid-b.

Re-runs the underlying experiments (a few minutes); numbers match the
"Mixed-regime router test" and "Sensitivity sweeps" sections of the log.
"""

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def load(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gather():
    stage0 = load("stage0_sanity_check")
    stage1 = load("stage1_evaluate_value_proxies")
    mixed = load("stage1_mixed_regime_router")
    sens = load("stage1_router_sensitivity")

    seeds = [0, 1, 2]
    budgets = [20, 40, 60]
    N, d, iters, thr = 128, 64, 12, 0.94

    sweep = {name: {b: [] for b in budgets} for name in ["mass", "UTC", "hybrid-b"]}
    v0_at_40 = []

    for seed in seeds:
        Q, K, V, _ = mixed.gen_mixed_qkv(N, d, seed)
        P, _ = stage0.full_attention(Q, K, V)
        H_norm = stage0.attention_entropy(P) / np.log(N)
        mask = H_norm >= thr

        for budget in budgets:
            _, mass_ks = stage1.calibrate_tau_for_mean_k(stage0, P, budget, iters=iters)
            _, utc_ks = stage1.calibrate_utc_for_mean_k(P, V, budget, iters=iters)
            _, oracle_ks = stage1.calibrate_epsilon_for_mean_k(stage0, P, V, budget, iters=iters)
            hb_ks = sens.hybrid_b_ks(stage1, P, V, mass_ks, mask, iters=iters)

            fixed_m = sens.max_rel(stage0, P, V, np.full(N, budget))
            oracle_m = sens.max_rel(stage0, P, V, oracle_ks)
            gap = fixed_m - oracle_m

            sweep["mass"][budget].append((fixed_m - sens.max_rel(stage0, P, V, mass_ks)) / gap)
            sweep["UTC"][budget].append((fixed_m - sens.max_rel(stage0, P, V, utc_ks)) / gap)
            sweep["hybrid-b"][budget].append((fixed_m - sens.max_rel(stage0, P, V, hb_ks)) / gap)

            if budget == 40:
                v0_ks, _ = stage1.entropy_routed_hybrid_k(stage0, P, V, budget, entropy_threshold=thr)
                v0_at_40.append((fixed_m - sens.max_rel(stage0, P, V, v0_ks)) / gap)

    return sweep, v0_at_40, budgets


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sweep, v0_at_40, budgets = gather()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    # left: matched budget k=40, method comparison
    ax = axes[0]
    names = ["mass", "UTC", "routing-only\n(hybrid-v0)", "budget-delegated\n(hybrid-b)"]
    data = [sweep["mass"][40], sweep["UTC"][40], v0_at_40, sweep["hybrid-b"][40]]
    colors = ["C0", "C1", "0.6", "C2"]
    x = np.arange(len(names))
    means = [np.mean(v) for v in data]
    ax.bar(x, means, width=0.6, color=colors, alpha=0.85)
    for xi, vals in zip(x, data):
        ax.plot(np.full(len(vals), xi), vals, "ko", ms=4, zorder=3)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.9)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("fraction of fixed-to-oracle gap closed")
    ax.set_title("Mixed-regime population, matched budget (mean k=40)", fontsize=10)
    ax.text(0.02, 0.03, "dots = individual seeds; dotted line = restricted oracle",
            transform=ax.transAxes, fontsize=8, color="0.35")

    # right: budget sweep
    ax = axes[1]
    for name, color in [("mass", "C0"), ("UTC", "C1"), ("hybrid-b", "C2")]:
        means = [np.mean(sweep[name][b]) for b in budgets]
        ax.plot(budgets, means, "o-", color=color, label=name, linewidth=2)
        for b in budgets:
            ax.plot(np.full(len(sweep[name][b]), b), sweep[name][b], "o",
                    color=color, ms=3, alpha=0.45)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.9)
    ax.set_xticks(budgets)
    ax.set_xlabel("mean retained budget k")
    ax.set_ylabel("fraction of fixed-to-oracle gap closed")
    ax.set_title("Budget sweep (router threshold 0.94)", fontsize=10)
    ax.legend(fontsize=9)

    fig.tight_layout()
    out_path = RESULTS_DIR / "stage1_mixed_regime_summary.png"
    fig.savefig(out_path, dpi=160)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
