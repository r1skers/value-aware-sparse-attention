"""Guard variants for the real-BERT adaptive failure (L11 h9).

Two candidate fixes, different philosophies:
1. floor guard (codex): clip every adaptive k at k_min — robustness bandaid,
   costs budget everywhere, one new hyperparameter.
2. relative-UTC score (mechanism fix): the diagnosis showed the failing rows
   have small output norm, and our threshold target is RELATIVE error while
   mass/UTC scores are ABSOLUTE quantities. Normalize the UTC score by
   ||mu_S(k)|| (= the sparse output itself, retained-side, free):

     score_rel(k) = delta(k) * ||mu_R_hat(k) - mu_S(k)|| / (||mu_S(k)|| + eta)

Both are evaluated at matched mean budget on the failure head (L11 h9) AND the
success head (L0 h4) — a fix that breaks the L0 UTC gain is no fix.
"""

import argparse
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BERT_DIR = ROOT / "results" / "bert"


def load(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rule_ks(P, V, threshold, scorer):
    n_rows, n = P.shape
    total_v = V.sum(axis=0)
    ks = np.empty(n_rows, dtype=int)
    for row in range(n_rows):
        p = P[row]
        for k in range(1, n + 1):
            if scorer(p, V, k, total_v) <= threshold:
                ks[row] = k
                break
    return ks


def calibrate_rule(P, V, target_mean_k, scorer, floor=1, iters=10):
    n_rows = P.shape[0]
    total_v = V.sum(axis=0)
    hi = max(scorer(P[row], V, 1, total_v) for row in range(n_rows))
    lo = 0.0
    ks = None
    for _ in range(iters):
        mid = (lo + hi) / 2
        ks = np.maximum(rule_ks(P, V, mid, scorer), floor)
        if ks.mean() > target_mean_k:
            lo = mid
        else:
            hi = mid
    return ks


def calibrate_mass_floored(stage0, P, target_mean_k, floor=1, iters=10):
    lo, hi = 0.0, 1.0
    ks = None
    for _ in range(iters):
        mid = (lo + hi) / 2
        ks = np.maximum(stage0.dropped_mass_budgeted_k(P, mid), floor)
        if ks.mean() > target_mean_k:
            lo = mid
        else:
            hi = mid
    return ks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--head", type=int, required=True)
    parser.add_argument("--target", type=int, default=64)
    parser.add_argument("--floor", type=int, default=32)
    args = parser.parse_args()

    stage0 = load("stage0_sanity_check")
    stage1 = load("stage1_evaluate_value_proxies")
    sens = load("stage1_router_sensitivity")

    data = np.load(BERT_DIR / f"qkv_layer{args.layer}.npz")
    P, V = data["P"][args.head], data["V"][args.head]
    N = P.shape[0]
    target = args.target
    floor = args.floor

    def utc_abs_scorer(p, V_, k, total_v):
        return stage1.utc_error_proxy_for_row(p, V_, k, total_v)

    _, oracle_ks = stage1.calibrate_epsilon_for_mean_k(stage0, P, V, target, iters=10)
    _, mass_ks = stage1.calibrate_tau_for_mean_k(stage0, P, target, iters=10)

    variants = {
        "fixed": np.full(N, target),
        "mass": mass_ks,
        f"mass+floor{floor}": calibrate_mass_floored(stage0, P, target, floor=floor),
        "UTC (abs)": calibrate_rule(P, V, target, utc_abs_scorer),
        "UTC-rel": calibrate_rule(P, V, target, stage1.utc_rel_error_proxy_for_row),
        f"UTC-rel+floor{floor}": calibrate_rule(
            P, V, target, stage1.utc_rel_error_proxy_for_row, floor=floor
        ),
        "oracle": oracle_ks,
    }

    fixed_m = sens.max_rel(stage0, P, V, variants["fixed"])
    oracle_m = sens.max_rel(stage0, P, V, variants["oracle"])
    gap = fixed_m - oracle_m

    print(f"[guard variants] L{args.layer} h{args.head}, target mean k={target}, N={N}")
    print("variant           | mean k | max_rel | gap closed")
    print("-" * 56)
    for name, ks in variants.items():
        m = sens.max_rel(stage0, P, V, ks)
        gc = (fixed_m - m) / gap if gap > 1e-15 else np.nan
        print(f"{name:<17} | {ks.mean():>6.2f} | {m:>7.4f} | {gc:>10.4f}")


if __name__ == "__main__":
    main()
