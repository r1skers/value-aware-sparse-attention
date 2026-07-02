"""Answer the open question from the log: at the SAME average retained budget,
how much does the oracle (uses V) beat the dropped-mass baseline (Q,K-only)?

The two existing sweeps (oracle error_budgeted_k vs epsilon, and
dropped_mass_budgeted_k vs tau) are not on a shared x-axis -- eps and tau are
different thresholds, so their tables can't be read side by side directly.
This calibrates eps and tau by bisection so both methods land on the same
mean retained budget, then compares max relative error directly.
"""

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STAGE0_PATH = ROOT / "scripts" / "stage0_sanity_check.py"


def load_stage0():
    spec = importlib.util.spec_from_file_location("stage0_sanity_check", STAGE0_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def calibrate_epsilon_for_mean_k(stage0, P, V, target_mean_k, lo=1e-4, hi=5.0, iters=14):
    for _ in range(iters):
        mid = (lo + hi) / 2
        ks = stage0.error_budgeted_k(P, V, mid)
        if ks.mean() > target_mean_k:
            lo = mid
        else:
            hi = mid
    return mid, stage0.error_budgeted_k(P, V, mid)


def calibrate_tau_for_mean_k(stage0, P, target_mean_k, lo=1e-4, hi=0.999, iters=14):
    for _ in range(iters):
        mid = (lo + hi) / 2
        ks = stage0.dropped_mass_budgeted_k(P, mid)
        if ks.mean() > target_mean_k:
            lo = mid
        else:
            hi = mid
    return mid, stage0.dropped_mass_budgeted_k(P, mid)


def main():
    stage0 = load_stage0()
    N, d = 128, 64
    Q, K, V = stage0.gen_qkv(N, d, seed=0, q_scale=1.0)
    P, _ = stage0.full_attention(Q, K, V)

    targets = [80, 60, 40]

    print("target_k | oracle mean k | oracle max_rel | mass mean k | mass max_rel | fixed max_rel")
    print("-" * 95)
    for target in targets:
        eps, oracle_ks = calibrate_epsilon_for_mean_k(stage0, P, V, target)
        tau, mass_ks = calibrate_tau_for_mean_k(stage0, P, target)

        oracle_stats = stage0.check_all_rows_decomposition(P, V, k=oracle_ks)
        mass_stats = stage0.check_all_rows_decomposition(P, V, k=mass_ks)
        fixed_stats = stage0.check_all_rows_decomposition(P, V, k=target)

        oracle_rel = stage0.relative_errors(P, V, oracle_stats)
        mass_rel = stage0.relative_errors(P, V, mass_stats)
        fixed_rel = stage0.relative_errors(P, V, fixed_stats)

        print(
            f"{target:>8} | "
            f"{oracle_ks.mean():>13.2f} | "
            f"{oracle_rel.max():>14.4f} | "
            f"{mass_ks.mean():>11.2f} | "
            f"{mass_rel.max():>12.4f} | "
            f"{fixed_rel.max():>13.4f}"
        )
        print(f"           (calibrated eps={eps:.4f}, tau={tau:.4f})")


if __name__ == "__main__":
    main()
