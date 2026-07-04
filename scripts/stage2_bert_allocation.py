"""Stage 2 step 1: run the stage-1 matched-budget comparison on real BERT heads.

Loads the (P, V) arrays saved by stage2_bert_qkv.py and runs
fixed / dropped-mass / UTC / hybrid-b / restricted-oracle at matched mean
budget on selected heads. Head selection: the layer-0 head with the largest
high-entropy population (UTC country) and a layer-5 head (sharp country).

Router threshold 0.90 — inside the plateau band [0.85, 0.97] validated on
synthetic data, not tuned on this data. Budget: mean k = 64 (25% of N=256,
matching the ~31% used in synthetic runs).
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


def analyze_head(stage0, stage1, sens, P, V, label, target_mean_k=64, thr=0.90, iters=10):
    N = P.shape[0]
    H_norm = stage0.attention_entropy(P) / np.log(N)
    mask = H_norm >= thr

    _, mass_ks = stage1.calibrate_tau_for_mean_k(stage0, P, target_mean_k, iters=iters)
    _, utc_ks = stage1.calibrate_utc_for_mean_k(P, V, target_mean_k, iters=iters)
    _, oracle_ks = stage1.calibrate_epsilon_for_mean_k(stage0, P, V, target_mean_k, iters=iters)
    hb_ks = sens.hybrid_b_ks(stage1, P, V, mass_ks, mask, iters=iters)

    fixed_m = sens.max_rel(stage0, P, V, np.full(N, target_mean_k))
    oracle_m = sens.max_rel(stage0, P, V, oracle_ks)
    gap = fixed_m - oracle_m

    def gc(ks):
        return (fixed_m - sens.max_rel(stage0, P, V, ks)) / gap if gap > 1e-15 else np.nan

    budget_ok = (
        abs(oracle_ks.mean() - target_mean_k) <= 2.0
        and abs(mass_ks.mean() - target_mean_k) <= 2.0
        and abs(utc_ks.mean() - target_mean_k) <= 2.0
    )

    print(
        f"{label:<22} | {H_norm.mean():>6.3f} | {mask.mean():>8.2f} | "
        f"{gc(mass_ks):>6.3f} | {gc(utc_ks):>6.3f} | {gc(hb_ks):>8.3f} | "
        f"{fixed_m:>9.4f} | {oracle_m:>10.4f} | {str(budget_ok):>6}"
    )


def main():
    stage0 = load("stage0_sanity_check")
    stage1 = load("stage1_evaluate_value_proxies")
    sens = load("stage1_router_sensitivity")

    print(
        "head                   | H_mean | mask frac | mass   | UTC    | hybrid-b | fixed max | oracle max | budget ok"
    )
    print("-" * 128)

    for layer in [0, 5, 11]:
        data = np.load(BERT_DIR / f"qkv_layer{layer}.npz")
        P_all, V_all = data["P"], data["V"]
        n_heads, N, _ = P_all.shape

        # per-head high-entropy population
        H = -(P_all * np.log(P_all + 1e-12)).sum(axis=-1) / np.log(N)  # (H, N)
        frac_high = (H >= 0.90).mean(axis=1)

        # the most diffuse head and the most typical (median) head of the layer
        head_hi = int(np.argmax(frac_high))
        head_med = int(np.argsort(H.mean(axis=1))[n_heads // 2])

        for head, tag in [(head_hi, "most-diffuse"), (head_med, "median")]:
            analyze_head(
                stage0, stage1, sens,
                P_all[head], V_all[head],
                label=f"L{layer} h{head} ({tag})",
            )


if __name__ == "__main__":
    main()
