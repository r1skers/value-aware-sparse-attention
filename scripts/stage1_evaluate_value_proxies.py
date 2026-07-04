"""Evaluate cheap value-side proxies for value centroid displacement.

First proxy:
  UTC (Uniform-Tail Centroid) proxy.

In high-entropy regimes, dropped attention weights over the tail are close to
uniform. Approximate the dropped weighted centroid mu_R by the unweighted mean
of dropped values:

  mu_R_hat = (sum_i V_i - sum_{i in S} V_i) / |R|

This uses one sequence-level precomputed sum and retained V, but does not scan
dropped V per query row.
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


def uniform_tail_centroid_proxy(P, V, k):
    n_rows, n = P.shape
    if k >= n:
        return np.zeros(n_rows)

    total_v = V.sum(axis=0)
    proxy_dists = np.empty(n_rows, dtype=np.float64)

    for row in range(n_rows):
        p = P[row]
        topk_idx = np.argsort(-p)[:k]

        retained_mass = p[topk_idx].sum()
        mu_s = (p[topk_idx, None] * V[topk_idx]).sum(axis=0) / retained_mass

        retained_unweighted_sum = V[topk_idx].sum(axis=0)
        mu_r_hat = (total_v - retained_unweighted_sum) / (n - k)

        proxy_dists[row] = np.linalg.norm(mu_r_hat - mu_s)

    return proxy_dists


def utc_error_proxy_for_row(p, V, k, total_v):
    n = p.shape[0]
    if k >= n:
        return 0.0

    topk_idx = np.argsort(-p)[:k]
    retained_mass = p[topk_idx].sum()
    dropped_mass = 1.0 - retained_mass

    mu_s = (p[topk_idx, None] * V[topk_idx]).sum(axis=0) / retained_mass
    retained_unweighted_sum = V[topk_idx].sum(axis=0)
    mu_r_hat = (total_v - retained_unweighted_sum) / (n - k)

    return dropped_mass * np.linalg.norm(mu_r_hat - mu_s)


def utc_rel_error_proxy_for_row(p, V, k, total_v, eta=1e-12):
    n = p.shape[0]
    if k >= n:
        return 0.0

    topk_idx = np.argsort(-p)[:k]
    retained_mass = p[topk_idx].sum()
    dropped_mass = 1.0 - retained_mass

    mu_s = (p[topk_idx, None] * V[topk_idx]).sum(axis=0) / retained_mass
    retained_unweighted_sum = V[topk_idx].sum(axis=0)
    mu_r_hat = (total_v - retained_unweighted_sum) / (n - k)

    abs_proxy = dropped_mass * np.linalg.norm(mu_r_hat - mu_s)
    return abs_proxy / (np.linalg.norm(mu_s) + eta)


def utc_rel_hat_error_proxy_for_row(p, V, k, total_v, eta=1e-12):
    n = p.shape[0]
    if k >= n:
        return 0.0

    topk_idx = np.argsort(-p)[:k]
    retained_mass = p[topk_idx].sum()
    dropped_mass = 1.0 - retained_mass

    mu_s = (p[topk_idx, None] * V[topk_idx]).sum(axis=0) / retained_mass
    retained_unweighted_sum = V[topk_idx].sum(axis=0)
    mu_r_hat = (total_v - retained_unweighted_sum) / (n - k)
    o_hat = retained_mass * mu_s + dropped_mass * mu_r_hat

    abs_proxy = dropped_mass * np.linalg.norm(mu_r_hat - mu_s)
    return abs_proxy / (np.linalg.norm(o_hat) + eta)


def proxy_budgeted_k(P, V, threshold, scorer):
    n_rows, n = P.shape
    total_v = V.sum(axis=0)
    ks = np.empty(n_rows, dtype=int)

    for row in range(n_rows):
        p = P[row]
        for k in range(1, n + 1):
            score = scorer(p, V, k, total_v)
            if score <= threshold:
                ks[row] = k
                break

    return ks


def utc_budgeted_k(P, V, threshold):
    return proxy_budgeted_k(P, V, threshold, utc_error_proxy_for_row)


def utc_rel_budgeted_k(P, V, threshold):
    return proxy_budgeted_k(P, V, threshold, utc_rel_error_proxy_for_row)


def utc_rel_hat_budgeted_k(P, V, threshold):
    return proxy_budgeted_k(P, V, threshold, utc_rel_hat_error_proxy_for_row)


def calibrate_proxy_for_mean_k(P, V, target_mean_k, scorer, budgeted_fn, iters=16):
    n = P.shape[1]
    total_v = V.sum(axis=0)
    hi = max(scorer(P[row], V, 1, total_v) for row in range(P.shape[0]))
    lo = 0.0
    ks = np.full(P.shape[0], n, dtype=int)

    for _ in range(iters):
        mid = (lo + hi) / 2
        ks = budgeted_fn(P, V, mid)
        if ks.mean() > target_mean_k:
            lo = mid
        else:
            hi = mid

    return mid, ks


def calibrate_utc_for_mean_k(P, V, target_mean_k, iters=16):
    return calibrate_proxy_for_mean_k(
        P, V, target_mean_k, utc_error_proxy_for_row, utc_budgeted_k, iters=iters
    )


def calibrate_utc_rel_for_mean_k(P, V, target_mean_k, iters=16):
    return calibrate_proxy_for_mean_k(
        P, V, target_mean_k, utc_rel_error_proxy_for_row, utc_rel_budgeted_k, iters=iters
    )


def calibrate_utc_rel_hat_for_mean_k(P, V, target_mean_k, iters=16):
    return calibrate_proxy_for_mean_k(
        P,
        V,
        target_mean_k,
        utc_rel_hat_error_proxy_for_row,
        utc_rel_hat_budgeted_k,
        iters=iters,
    )


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


def entropy_routed_hybrid_k(stage0, P, V, target_mean_k, entropy_threshold=0.94):
    """Route high-entropy rows to UTC and lower-entropy rows to dropped mass."""
    H_norm = stage0.attention_entropy(P) / np.log(P.shape[1])
    high_entropy = H_norm >= entropy_threshold
    ks = np.empty(P.shape[0], dtype=int)

    if high_entropy.any():
        _, utc_ks = calibrate_utc_for_mean_k(P[high_entropy], V, target_mean_k)
        ks[high_entropy] = utc_ks

    if (~high_entropy).any():
        _, mass_ks = calibrate_tau_for_mean_k(stage0, P[~high_entropy], target_mean_k)
        ks[~high_entropy] = mass_ks

    return ks, high_entropy


def run_utc_proxy_sweep(q_scales, N=128, d=64, seed=0, k=40):
    stage0 = load_stage0()
    rows = []

    for q_scale in q_scales:
        Q, K, V = stage0.gen_qkv(N, d, seed=seed, q_scale=q_scale)
        P, _ = stage0.full_attention(Q, K, V)
        stats = stage0.check_all_rows_decomposition(P, V, k=k)

        true_centroid = stats["centroid_dists"]
        errors = stats["true_errors"]
        deltas = stats["deltas"]
        utc = uniform_tail_centroid_proxy(P, V, k)
        utc_error_proxy = deltas * utc

        H_norm = stage0.attention_entropy(P) / np.log(N)

        rows.append({
            "q_scale": q_scale,
            "H_norm_mean": H_norm.mean(),
            "mean_error": errors.mean(),
            "mean_delta": deltas.mean(),
            "corr_delta_error": stage0.pearson_corr(deltas, errors),
            "corr_true_centroid_error": stage0.pearson_corr(true_centroid, errors),
            "corr_utc_centroid": stage0.pearson_corr(utc, true_centroid),
            "corr_delta_utc_error": stage0.pearson_corr(utc_error_proxy, errors),
            "utc_rel_mae": np.mean(np.abs(utc - true_centroid) / (true_centroid + 1e-12)),
        })

    return rows


def print_utc_proxy_sweep(rows):
    print("\n[UTC proxy sweep: fixed top-k, k=40]")
    print(
        "q_scale | H_norm mean | mean err | mean delta | "
        "corr(delta,err) | corr(trueC,err) | corr(UTC,trueC) | "
        "corr(delta*UTC,err) | UTC rel MAE"
    )
    print("-" * 150)
    for row in rows:
        print(
            f"{row['q_scale']:>7.2f} | "
            f"{row['H_norm_mean']:>11.4f} | "
            f"{row['mean_error']:>8.4f} | "
            f"{row['mean_delta']:>10.4f} | "
            f"{row['corr_delta_error']:>15.4f} | "
            f"{row['corr_true_centroid_error']:>15.4f} | "
            f"{row['corr_utc_centroid']:>16.4f} | "
            f"{row['corr_delta_utc_error']:>19.4f} | "
            f"{row['utc_rel_mae']:>11.4f}"
        )


def run_utc_allocation_sweep(
    q_scales,
    N=128,
    d=64,
    seed=0,
    target_mean_k=40,
    entropy_threshold=0.94,
):
    stage0 = load_stage0()
    rows = []

    for q_scale in q_scales:
        Q, K, V = stage0.gen_qkv(N, d, seed=seed, q_scale=q_scale)
        P, _ = stage0.full_attention(Q, K, V)
        H_norm_mean = (stage0.attention_entropy(P) / np.log(N)).mean()

        _, oracle_ks = calibrate_epsilon_for_mean_k(stage0, P, V, target_mean_k)
        _, mass_ks = calibrate_tau_for_mean_k(stage0, P, target_mean_k)
        _, utc_ks = calibrate_utc_for_mean_k(P, V, target_mean_k)
        hybrid_ks, hybrid_utc_mask = entropy_routed_hybrid_k(
            stage0,
            P,
            V,
            target_mean_k,
            entropy_threshold=entropy_threshold,
        )

        fixed_stats = stage0.check_all_rows_decomposition(P, V, k=target_mean_k)
        mass_stats = stage0.check_all_rows_decomposition(P, V, k=mass_ks)
        utc_stats = stage0.check_all_rows_decomposition(P, V, k=utc_ks)
        hybrid_stats = stage0.check_all_rows_decomposition(P, V, k=hybrid_ks)
        oracle_stats = stage0.check_all_rows_decomposition(P, V, k=oracle_ks)

        fixed_max = stage0.relative_errors(P, V, fixed_stats).max()
        mass_max = stage0.relative_errors(P, V, mass_stats).max()
        utc_max = stage0.relative_errors(P, V, utc_stats).max()
        hybrid_max = stage0.relative_errors(P, V, hybrid_stats).max()
        oracle_max = stage0.relative_errors(P, V, oracle_stats).max()

        total_gap = fixed_max - oracle_max
        mass_gap_closed = (fixed_max - mass_max) / total_gap if total_gap > 1e-15 else np.nan
        utc_gap_closed = (fixed_max - utc_max) / total_gap if total_gap > 1e-15 else np.nan
        hybrid_gap_closed = (fixed_max - hybrid_max) / total_gap if total_gap > 1e-15 else np.nan

        saturated = (
            abs(oracle_ks.mean() - target_mean_k) > 1.0
            or abs(mass_ks.mean() - target_mean_k) > 1.0
            or abs(utc_ks.mean() - target_mean_k) > 1.0
            or abs(hybrid_ks.mean() - target_mean_k) > 1.0
        )

        rows.append({
            "q_scale": q_scale,
            "H_norm_mean": H_norm_mean,
            "fixed_max": fixed_max,
            "mass_max": mass_max,
            "utc_max": utc_max,
            "hybrid_max": hybrid_max,
            "oracle_max": oracle_max,
            "mass_mean_k": mass_ks.mean(),
            "utc_mean_k": utc_ks.mean(),
            "hybrid_mean_k": hybrid_ks.mean(),
            "oracle_mean_k": oracle_ks.mean(),
            "mass_gap_closed": mass_gap_closed,
            "utc_gap_closed": utc_gap_closed,
            "hybrid_gap_closed": hybrid_gap_closed,
            "hybrid_utc_fraction": hybrid_utc_mask.mean(),
            "entropy_threshold": entropy_threshold,
            "saturated": saturated,
        })

    return rows


def print_utc_allocation_sweep(rows):
    print("\n[matched-budget allocation: fixed vs mass vs UTC vs hybrid vs restricted oracle]")
    print(
        "q_scale | H_norm mean | oracle max | hybrid max | UTC max | mass max | fixed max | "
        "hybrid gap | UTC gap | mass gap | UTC-routed"
    )
    print("-" * 140)
    for row in rows:
        flag = "  [SATURATED]" if row["saturated"] else ""
        print(
            f"{row['q_scale']:>7.2f} | "
            f"{row['H_norm_mean']:>11.4f} | "
            f"{row['oracle_max']:>10.6f} | "
            f"{row['hybrid_max']:>10.6f} | "
            f"{row['utc_max']:>7.6f} | "
            f"{row['mass_max']:>8.6f} | "
            f"{row['fixed_max']:>9.6f} | "
            f"{row['hybrid_gap_closed']:>10.4f} | "
            f"{row['utc_gap_closed']:>7.4f} | "
            f"{row['mass_gap_closed']:>8.4f} | "
            f"{row['hybrid_utc_fraction']:>10.2f}"
            f"{flag}"
        )


def main():
    q_scales = [0.25, 0.5, 1.0, 2.0, 4.0]
    rows = run_utc_proxy_sweep(q_scales, k=40)
    print_utc_proxy_sweep(rows)

    allocation_rows = run_utc_allocation_sweep(q_scales, target_mean_k=40)
    print_utc_allocation_sweep(allocation_rows)


if __name__ == "__main__":
    main()
