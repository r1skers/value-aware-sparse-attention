"""Sweep the matched-budget three-way comparison (oracle / dropped-mass / fixed)
across q_scale regimes.

Prediction from the v1 regime sweep: in high-entropy regimes (small q_scale)
value geometry dominates row-to-row error, so the oracle's advantage over the
Q,K-only dropped-mass baseline should be LARGEST there; in sharp low-entropy
regimes (large q_scale) dropped mass explains almost everything, so the
dropped-mass baseline should close most of the gap to the oracle.

All three methods are calibrated to the same mean retained k per q_scale, so
the comparison is at equal average budget throughout.
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


def calibrate_epsilon_for_mean_k(stage0, P, V, target_mean_k, lo=0.0, hi=10.0, iters=16):
    ks = None
    for _ in range(iters):
        mid = (lo + hi) / 2
        ks = stage0.error_budgeted_k(P, V, mid)
        if ks.mean() > target_mean_k:
            lo = mid
        else:
            hi = mid
    return mid, ks


def calibrate_tau_for_mean_k(stage0, P, target_mean_k, lo=0.0, hi=1.0, iters=16):
    ks = None
    for _ in range(iters):
        mid = (lo + hi) / 2
        ks = stage0.dropped_mass_budgeted_k(P, mid)
        if ks.mean() > target_mean_k:
            lo = mid
        else:
            hi = mid
    return mid, ks


def main():
    stage0 = load_stage0()
    N, d = 128, 64
    target_mean_k = 40
    q_scales = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

    print(f"matched mean budget: target mean k = {target_mean_k}, N={N}, d={d}, seed=0")
    print()
    print(
        "q_scale | H_norm mean | oracle mean k | mass mean k | oracle max_rel | "
        "mass max_rel | fixed max_rel | gap closed by mass"
    )
    print("-" * 128)

    for q_scale in q_scales:
        Q, K, V = stage0.gen_qkv(N, d, seed=0, q_scale=q_scale)
        P, _ = stage0.full_attention(Q, K, V)
        H_norm_mean = (stage0.attention_entropy(P) / np.log(N)).mean()

        _, oracle_ks = calibrate_epsilon_for_mean_k(stage0, P, V, target_mean_k)
        _, mass_ks = calibrate_tau_for_mean_k(stage0, P, target_mean_k)

        # in very sharp regimes the error hits float zero at small k, so no
        # threshold can force the methods to spend the full budget -- the
        # budget constraint stops binding and the comparison degenerates
        saturated = (
            abs(oracle_ks.mean() - target_mean_k) > 1.0
            or abs(mass_ks.mean() - target_mean_k) > 1.0
        )

        oracle_stats = stage0.check_all_rows_decomposition(P, V, k=oracle_ks)
        mass_stats = stage0.check_all_rows_decomposition(P, V, k=mass_ks)
        fixed_stats = stage0.check_all_rows_decomposition(P, V, k=target_mean_k)

        oracle_max = stage0.relative_errors(P, V, oracle_stats).max()
        mass_max = stage0.relative_errors(P, V, mass_stats).max()
        fixed_max = stage0.relative_errors(P, V, fixed_stats).max()

        # fraction of the fixed->oracle gap that the Q,K-only baseline recovers
        gap_total = fixed_max - oracle_max
        gap_closed = (fixed_max - mass_max) / gap_total if gap_total > 1e-15 else float("nan")

        flag = "  [SATURATED - budget not binding, row not comparable]" if saturated else ""
        print(
            f"{q_scale:>7.2f} | "
            f"{H_norm_mean:>11.4f} | "
            f"{oracle_ks.mean():>13.2f} | "
            f"{mass_ks.mean():>11.2f} | "
            f"{oracle_max:>14.6f} | "
            f"{mass_max:>12.6f} | "
            f"{fixed_max:>13.6f} | "
            f"{gap_closed:>18.4f}"
            f"{flag}"
        )


if __name__ == "__main__":
    main()
