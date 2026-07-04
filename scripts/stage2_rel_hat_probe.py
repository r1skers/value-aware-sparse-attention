"""Probe a third scorer: UTC-rel-hat, normalizing by the UTC-estimated FULL
output norm instead of the retained-centroid norm.

The held-out sweep showed the abs-vs-rel selector is unsolved and named the
missing variable: denominator stability. UTC-rel divides by ||mu_S(k)||, which
is a poor estimate of ||o|| exactly where it matters (small k, sink rows whose
top-k values are near-zero punctuation -> denominator collapses -> score
explodes -> budget misallocation). But UTC already estimates mu_R, so a
better denominator is free:

    o_hat(k)  = (1 - delta(k)) * mu_S(k) + delta(k) * mu_R_hat(k)
    score(k)  = delta(k) * ||mu_R_hat(k) - mu_S(k)|| / (||o_hat(k)|| + eta)

Test cases (all previously measured, so abs/rel numbers are anchors):
  synthetic q0.25 b40   : abs won big  (0.65 vs 0.11)  -- rel's failure case
  synthetic q1.0  b40   : rel won big  (-0.07 vs 0.72) -- rel's repair case
  BERT L11 h9 k=64      : rel repair   (-0.71 vs 0.87)
  BERT L11 h9 k=16      : tight budget, late layer     -- rel's danger zone
  BERT L0  h4 k=64      : abs ahead    (0.53 vs 0.36)
"""

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


def utc_rel_hat_scorer(p, V, k, total_v, eta=1e-12):
    n = p.shape[0]
    if k >= n:
        return 0.0
    topk_idx = np.argsort(-p)[:k]
    retained_mass = p[topk_idx].sum()
    delta = 1.0 - retained_mass
    mu_s = (p[topk_idx, None] * V[topk_idx]).sum(axis=0) / retained_mass
    mu_r_hat = (total_v - V[topk_idx].sum(axis=0)) / (n - k)
    o_hat = (1.0 - delta) * mu_s + delta * mu_r_hat
    return delta * np.linalg.norm(mu_r_hat - mu_s) / (np.linalg.norm(o_hat) + eta)


def run_case(label, stage0, stage1, sens, guard, P, V, target):
    N = P.shape[0]

    def abs_scorer(p, V_, k, total_v):
        return stage1.utc_error_proxy_for_row(p, V_, k, total_v)

    _, oracle_ks = stage1.calibrate_epsilon_for_mean_k(stage0, P, V, target, iters=10)
    fixed_m = sens.max_rel(stage0, P, V, np.full(N, target))
    oracle_m = sens.max_rel(stage0, P, V, oracle_ks)
    gap = fixed_m - oracle_m

    row = {"label": label}
    for name, scorer in [
        ("abs", abs_scorer),
        ("rel", stage1.utc_rel_error_proxy_for_row),
        ("rel-hat", utc_rel_hat_scorer),
    ]:
        ks = guard.calibrate_rule(P, V, target, scorer)
        m = sens.max_rel(stage0, P, V, ks)
        row[name] = (fixed_m - m) / gap if gap > 1e-15 else np.nan
        row[f"{name}_mean_k"] = ks.mean()
    return row


def main():
    stage0 = load("stage0_sanity_check")
    stage1 = load("stage1_evaluate_value_proxies")
    sens = load("stage1_router_sensitivity")
    guard = load("stage2_guard_variants")

    cases = []

    for q_scale, tag in [(0.25, "rel-failure"), (1.0, "rel-repair")]:
        Q, K, V = stage0.gen_qkv(128, 64, seed=0, q_scale=q_scale)
        P, _ = stage0.full_attention(Q, K, V)
        cases.append((f"synth q={q_scale} b40 ({tag})", P, V, 40))

    data11 = np.load(BERT_DIR / "qkv_layer11.npz")
    data0 = np.load(BERT_DIR / "qkv_layer0.npz")
    cases.append(("BERT L11 h9 k=64 (rel-repair)", data11["P"][9], data11["V"][9], 64))
    cases.append(("BERT L11 h9 k=16 (tight)", data11["P"][9], data11["V"][9], 16))
    cases.append(("BERT L0 h4 k=64 (abs-ahead)", data0["P"][4], data0["V"][4], 64))

    print("case                            | abs     | rel     | rel-hat | (gap closed; higher=better)")
    print("-" * 100)
    for label, P, V, target in cases:
        row = run_case(label, stage0, stage1, sens, guard, P, V, target)
        print(
            f"{row['label']:<31} | {row['abs']:>7.3f} | {row['rel']:>7.3f} | {row['rel-hat']:>7.3f}"
        )


if __name__ == "__main__":
    main()
