"""Mixed-regime router test: does per-row entropy routing actually earn its keep?

The per-dataset router experiment was circular: every synthetic dataset had a
single regime, so the router acted as a dataset-level switch with an in-sample
threshold, and the hybrid could only ever match the better pure strategy.

This test builds ONE population whose rows span regimes (each row draws its own
q_scale), so:
  1. the router faces genuine per-row decisions,
  2. a hybrid can in principle beat BOTH pure strategies,
  3. dropped-mass gets to show its cross-regime budget shifting (a single tau
     naturally gives sharp rows tiny k and diffuse rows large k),
  4. the v0 hybrid's design flaw becomes visible: it pins each branch to the
     same mean budget, forbidding exactly that cross-regime budget transfer.

The router threshold (H_norm >= 0.94) is carried over from the per-dataset
analysis and NOT tuned on this data.

hybrid-b: dropped-mass decides the budget split across router groups (Q,K-only,
so allowed), UTC re-allocates within the high-entropy group.
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


def gen_mixed_qkv(N, d, seed, scales=(0.25, 0.5, 1.0, 2.0, 4.0)):
    rng = np.random.default_rng(seed)
    row_scales = rng.choice(scales, size=N)
    Q = rng.standard_normal((N, d)) * row_scales[:, None]
    K = rng.standard_normal((N, d))
    V = rng.standard_normal((N, d))
    return Q, K, V, row_scales


def evaluate(stage0, P, V, ks, mask):
    stats = stage0.check_all_rows_decomposition(P, V, k=ks)
    rel = stage0.relative_errors(P, V, stats)
    ks_arr = stats["ks"]
    return {
        "mean_k": ks_arr.mean(),
        "max_rel": rel.max(),
        "mean_rel": rel.mean(),
        "hi_max_rel": rel[mask].max() if mask.any() else np.nan,
        "lo_max_rel": rel[~mask].max() if (~mask).any() else np.nan,
        "hi_mean_k": ks_arr[mask].mean() if mask.any() else np.nan,
        "lo_mean_k": ks_arr[~mask].mean() if (~mask).any() else np.nan,
    }


def run_seed(stage0, stage1, seed, N=128, d=64, target_mean_k=40, threshold=0.94):
    Q, K, V, row_scales = gen_mixed_qkv(N, d, seed)
    P, _ = stage0.full_attention(Q, K, V)
    H_norm = stage0.attention_entropy(P) / np.log(N)
    mask = H_norm >= threshold

    _, mass_ks = stage1.calibrate_tau_for_mean_k(stage0, P, target_mean_k)
    _, utc_ks = stage1.calibrate_utc_for_mean_k(P, V, target_mean_k)
    _, oracle_ks = stage1.calibrate_epsilon_for_mean_k(stage0, P, V, target_mean_k)
    hybrid_ks, _ = stage1.entropy_routed_hybrid_k(
        stage0, P, V, target_mean_k, entropy_threshold=threshold
    )

    # hybrid-b: budget split across groups from the global mass calibration,
    # UTC re-allocation inside the high-entropy group only
    hybrid_b_ks = mass_ks.copy()
    if mask.any():
        hi_budget = mass_ks[mask].mean()
        _, hi_utc_ks = stage1.calibrate_utc_for_mean_k(P[mask], V, hi_budget)
        hybrid_b_ks[mask] = hi_utc_ks

    fixed_ks = np.full(N, target_mean_k)

    methods = {
        "fixed": fixed_ks,
        "mass": mass_ks,
        "UTC": utc_ks,
        "hybrid-v0": hybrid_ks,
        "hybrid-b": hybrid_b_ks,
        "oracle": oracle_ks,
    }

    results = {name: evaluate(stage0, P, V, ks, mask) for name, ks in methods.items()}
    results["_meta"] = {
        "hi_fraction": mask.mean(),
        "H_norm_range": (H_norm.min(), H_norm.max()),
    }
    return results


def print_seed(seed, results):
    meta = results["_meta"]
    print(
        f"\n[seed {seed}] high-entropy rows (H_norm>=thr): {meta['hi_fraction']:.2f}, "
        f"H_norm range [{meta['H_norm_range'][0]:.3f}, {meta['H_norm_range'][1]:.3f}]"
    )
    fixed_max = results["fixed"]["max_rel"]
    oracle_max = results["oracle"]["max_rel"]
    gap = fixed_max - oracle_max

    print(
        "method    | mean k | max_rel | mean_rel | gap closed | "
        "hi-grp max_rel | lo-grp max_rel | hi mean k | lo mean k"
    )
    print("-" * 118)
    for name in ["fixed", "mass", "UTC", "hybrid-v0", "hybrid-b", "oracle"]:
        r = results[name]
        gap_closed = (fixed_max - r["max_rel"]) / gap if gap > 1e-15 else np.nan
        print(
            f"{name:<9} | "
            f"{r['mean_k']:>6.2f} | "
            f"{r['max_rel']:>7.4f} | "
            f"{r['mean_rel']:>8.4f} | "
            f"{gap_closed:>10.4f} | "
            f"{r['hi_max_rel']:>14.4f} | "
            f"{r['lo_max_rel']:>14.4f} | "
            f"{r['hi_mean_k']:>9.2f} | "
            f"{r['lo_mean_k']:>9.2f}"
        )


def main():
    stage0 = load("stage0_sanity_check")
    stage1 = load("stage1_evaluate_value_proxies")

    seeds = [0, 1, 2]
    all_results = {}
    for seed in seeds:
        all_results[seed] = run_seed(stage0, stage1, seed)
        print_seed(seed, all_results[seed])

    print("\n[summary across seeds: gap closed on overall max_rel]")
    print("method    | " + " | ".join(f"seed {s}" for s in seeds))
    print("-" * 45)
    for name in ["mass", "UTC", "hybrid-v0", "hybrid-b"]:
        vals = []
        for seed in seeds:
            fixed_max = all_results[seed]["fixed"]["max_rel"]
            oracle_max = all_results[seed]["oracle"]["max_rel"]
            gap = fixed_max - oracle_max
            r = all_results[seed][name]
            vals.append((fixed_max - r["max_rel"]) / gap if gap > 1e-15 else np.nan)
        print(f"{name:<9} | " + " | ".join(f"{v:>6.3f}" for v in vals))


if __name__ == "__main__":
    main()
