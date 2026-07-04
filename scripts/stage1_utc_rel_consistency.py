"""Stage 1 consistency check for the UTC relative-risk normalization.

Real BERT exposed a target mismatch: UTC-abs scores absolute output error,
while the allocation benchmarks use relative error. This script reruns the
synthetic mixed-regime benchmark with both scores:

  UTC-abs = delta * ||mu_R_hat - mu_S||
  UTC-rel = UTC-abs / (||mu_S|| + eta)

The goal is not to tune a new synthetic method. It asks whether the
real-data-motivated relative normalization invalidates Stage 1 synthetic
conclusions or leaves them essentially intact.
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


def max_rel(stage0, P, V, ks):
    stats = stage0.check_all_rows_decomposition(P, V, k=ks)
    return stage0.relative_errors(P, V, stats).max()


def gap_closed(stage0, P, V, fixed_ks, oracle_ks, ks):
    fixed_m = max_rel(stage0, P, V, fixed_ks)
    oracle_m = max_rel(stage0, P, V, oracle_ks)
    gap = fixed_m - oracle_m
    return (fixed_m - max_rel(stage0, P, V, ks)) / gap if gap > 1e-15 else np.nan


def hybrid_b_rel_ks(stage1, P, V, mass_ks, mask, iters=12):
    ks = mass_ks.copy()
    if mask.any():
        hi_budget = mass_ks[mask].mean()
        _, hi = stage1.calibrate_utc_rel_for_mean_k(P[mask], V, hi_budget, iters=iters)
        ks[mask] = hi
    return ks


def hybrid_b_rel_hat_ks(stage1, P, V, mass_ks, mask, iters=12):
    ks = mass_ks.copy()
    if mask.any():
        hi_budget = mass_ks[mask].mean()
        _, hi = stage1.calibrate_utc_rel_hat_for_mean_k(P[mask], V, hi_budget, iters=iters)
        ks[mask] = hi
    return ks


def output_norm_cv(P, V):
    norms = np.linalg.norm(P @ V, axis=1)
    return norms.mean(), norms.std() / (norms.mean() + 1e-12)


def run(seed, budget, threshold=0.94, N=128, d=64, iters=12):
    stage0 = load("stage0_sanity_check")
    stage1 = load("stage1_evaluate_value_proxies")
    mixed = load("stage1_mixed_regime_router")
    sens = load("stage1_router_sensitivity")

    Q, K, V, _ = mixed.gen_mixed_qkv(N, d, seed)
    P, _ = stage0.full_attention(Q, K, V)
    H_norm = stage0.attention_entropy(P) / np.log(N)
    mask = H_norm >= threshold

    _, mass_ks = stage1.calibrate_tau_for_mean_k(stage0, P, budget, iters=iters)
    _, utc_abs_ks = stage1.calibrate_utc_for_mean_k(P, V, budget, iters=iters)
    _, utc_rel_ks = stage1.calibrate_utc_rel_for_mean_k(P, V, budget, iters=iters)
    _, utc_rel_hat_ks = stage1.calibrate_utc_rel_hat_for_mean_k(P, V, budget, iters=iters)
    _, oracle_ks = stage1.calibrate_epsilon_for_mean_k(stage0, P, V, budget, iters=iters)
    fixed_ks = np.full(N, budget)

    hb_abs_ks = sens.hybrid_b_ks(stage1, P, V, mass_ks, mask, iters=iters)
    hb_rel_ks = hybrid_b_rel_ks(stage1, P, V, mass_ks, mask, iters=iters)
    hb_rel_hat_ks = hybrid_b_rel_hat_ks(stage1, P, V, mass_ks, mask, iters=iters)
    norm_mean, norm_cv = output_norm_cv(P, V)

    return {
        "seed": seed,
        "budget": budget,
        "H_hi_frac": mask.mean(),
        "norm_mean": norm_mean,
        "norm_cv": norm_cv,
        "mass": gap_closed(stage0, P, V, fixed_ks, oracle_ks, mass_ks),
        "UTC_abs": gap_closed(stage0, P, V, fixed_ks, oracle_ks, utc_abs_ks),
        "UTC_rel": gap_closed(stage0, P, V, fixed_ks, oracle_ks, utc_rel_ks),
        "UTC_rel_hat": gap_closed(stage0, P, V, fixed_ks, oracle_ks, utc_rel_hat_ks),
        "hybrid_b_abs": gap_closed(stage0, P, V, fixed_ks, oracle_ks, hb_abs_ks),
        "hybrid_b_rel": gap_closed(stage0, P, V, fixed_ks, oracle_ks, hb_rel_ks),
        "hybrid_b_rel_hat": gap_closed(stage0, P, V, fixed_ks, oracle_ks, hb_rel_hat_ks),
        "oracle_mean_k": oracle_ks.mean(),
    }


def main():
    seeds = [0, 1, 2]
    budgets = [20, 40, 60]
    rows = [run(seed, budget) for seed in seeds for budget in budgets]

    print("[UTC-rel synthetic consistency: mixed-regime benchmark]")
    print(
        "seed | budget | norm_cv | mass | UTC_abs | UTC_rel | UTC_rel_hat | "
        "hybrid_b_abs | hybrid_b_rel | hybrid_b_rel_hat | oracle_k"
    )
    print("-" * 132)
    for row in rows:
        print(
            f"{row['seed']:>4} | "
            f"{row['budget']:>6} | "
            f"{row['norm_cv']:>7.3f} | "
            f"{row['mass']:>5.3f} | "
            f"{row['UTC_abs']:>7.3f} | "
            f"{row['UTC_rel']:>7.3f} | "
            f"{row['UTC_rel_hat']:>11.3f} | "
            f"{row['hybrid_b_abs']:>12.3f} | "
            f"{row['hybrid_b_rel']:>12.3f} | "
            f"{row['hybrid_b_rel_hat']:>16.3f} | "
            f"{row['oracle_mean_k']:>8.2f}"
        )

    print("\n[mean gap closed across 9 seed-budget settings]")
    for key in [
        "mass",
        "UTC_abs",
        "UTC_rel",
        "UTC_rel_hat",
        "hybrid_b_abs",
        "hybrid_b_rel",
        "hybrid_b_rel_hat",
    ]:
        vals = np.array([row[key] for row in rows])
        print(f"{key:<13} mean={vals.mean():.3f} min={vals.min():.3f} max={vals.max():.3f}")


if __name__ == "__main__":
    main()
