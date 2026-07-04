"""Sensitivity sweeps for the budget-delegated hybrid (hybrid-b).

Two of the fixed choices behind the mixed-regime result could plausibly flip
the conclusion; this script unfixes both on the same mixed population:

1. router threshold sweep (was fixed at 0.94, inherited from per-dataset
   analysis) at fixed budget mean k = 40;
2. budget-level sweep (was fixed at mean k = 40) at fixed threshold 0.94.

mass / UTC / oracle do not depend on the threshold, so within a seed they are
calibrated once per budget and reused across thresholds.
"""

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hybrid_b_ks(stage1, P, V, mass_ks, mask, iters=12):
    ks = mass_ks.copy()
    if mask.any():
        hi_budget = mass_ks[mask].mean()
        _, hi = stage1.calibrate_utc_for_mean_k(P[mask], V, hi_budget, iters=iters)
        ks[mask] = hi
    return ks


def max_rel(stage0, P, V, ks):
    stats = stage0.check_all_rows_decomposition(P, V, k=ks)
    return stage0.relative_errors(P, V, stats).max()


def main():
    stage0 = load("stage0_sanity_check")
    stage1 = load("stage1_evaluate_value_proxies")
    mixed = load("stage1_mixed_regime_router")

    seeds = [0, 1, 2]
    N, d = 128, 64
    iters = 12

    print("[threshold sweep: hybrid-b gap closed, budget mean k=40]")
    thresholds = [0.85, 0.90, 0.94, 0.97]
    header = "seed | " + " | ".join(f"thr={t}" for t in thresholds) + " | mass | UTC"
    print(header)
    print("-" * len(header))
    for seed in seeds:
        Q, K, V, _ = mixed.gen_mixed_qkv(N, d, seed)
        P, _ = stage0.full_attention(Q, K, V)
        H_norm = stage0.attention_entropy(P) / np.log(N)

        _, mass_ks = stage1.calibrate_tau_for_mean_k(stage0, P, 40, iters=iters)
        _, utc_ks = stage1.calibrate_utc_for_mean_k(P, V, 40, iters=iters)
        _, oracle_ks = stage1.calibrate_epsilon_for_mean_k(stage0, P, V, 40, iters=iters)

        fixed_m = max_rel(stage0, P, V, np.full(N, 40))
        oracle_m = max_rel(stage0, P, V, oracle_ks)
        gap = fixed_m - oracle_m

        def gc(ks):
            return (fixed_m - max_rel(stage0, P, V, ks)) / gap

        vals = []
        for thr in thresholds:
            mask = H_norm >= thr
            hb = hybrid_b_ks(stage1, P, V, mass_ks, mask, iters=iters)
            vals.append(gc(hb))

        print(
            f"{seed:>4} | "
            + " | ".join(f"{v:>8.3f}" for v in vals)
            + f" | {gc(mass_ks):>5.3f} | {gc(utc_ks):>5.3f}"
        )

    print("\n[budget sweep: gap closed at threshold 0.94]")
    budgets = [20, 40, 60]
    print("seed | budget | mass | UTC | hybrid-b | oracle mean k ok?")
    print("-" * 60)
    for seed in seeds:
        Q, K, V, _ = mixed.gen_mixed_qkv(N, d, seed)
        P, _ = stage0.full_attention(Q, K, V)
        H_norm = stage0.attention_entropy(P) / np.log(N)
        mask = H_norm >= 0.94

        for budget in budgets:
            _, mass_ks = stage1.calibrate_tau_for_mean_k(stage0, P, budget, iters=iters)
            _, utc_ks = stage1.calibrate_utc_for_mean_k(P, V, budget, iters=iters)
            _, oracle_ks = stage1.calibrate_epsilon_for_mean_k(stage0, P, V, budget, iters=iters)
            hb = hybrid_b_ks(stage1, P, V, mass_ks, mask, iters=iters)

            fixed_m = max_rel(stage0, P, V, np.full(N, budget))
            oracle_m = max_rel(stage0, P, V, oracle_ks)
            gap = fixed_m - oracle_m

            budget_ok = abs(oracle_ks.mean() - budget) <= 1.0

            def gc(ks):
                return (fixed_m - max_rel(stage0, P, V, ks)) / gap if gap > 1e-15 else np.nan

            print(
                f"{seed:>4} | {budget:>6} | "
                f"{gc(mass_ks):>5.3f} | {gc(utc_ks):>5.3f} | "
                f"{gc(hb):>8.3f} | {str(budget_ok):>5}"
            )


if __name__ == "__main__":
    main()
