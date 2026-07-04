"""Diagnose real BERT heads where adaptive allocation loses to fixed-k.

The first BERT pass found L11 h9 where mass, UTC, and hybrid-b are all worse
than fixed-k. This script prints row-level evidence for one head: where the
worst errors occur, how each method allocates k, and whether special-token or
attention-sink structure is involved.
"""

import importlib.util
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
BERT_DIR = ROOT / "results" / "bert"


def load(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row_metrics(stage0, P, V, ks):
    stats = stage0.check_all_rows_decomposition(P, V, k=ks)
    rel = stage0.relative_errors(P, V, stats)
    out_norm = np.linalg.norm(P @ V, axis=1)
    return {
        "ks": stats["ks"],
        "abs_err": stats["true_errors"],
        "rel_err": rel,
        "delta": stats["deltas"],
        "centroid": stats["centroid_dists"],
        "out_norm": out_norm,
    }


def top_attention_tokens(p, tokens, n=5):
    order = np.argsort(-p)[:n]
    return " ".join(f"{idx}:{tokens[idx]}:{p[idx]:.3f}" for idx in order)


def print_method_summary(name, metrics):
    worst = int(np.argmax(metrics["rel_err"]))
    print(
        f"{name:<9} mean_k={metrics['ks'].mean():6.2f} "
        f"max_rel={metrics['rel_err'][worst]:8.4f} "
        f"worst_row={worst:3d} k={metrics['ks'][worst]:3d} "
        f"delta={metrics['delta'][worst]:.4f} "
        f"C={metrics['centroid'][worst]:.4f} "
        f"abs={metrics['abs_err'][worst]:.4f} "
        f"||o||={metrics['out_norm'][worst]:.4f}"
    )


def main():
    stage0 = load("stage0_sanity_check")
    stage1 = load("stage1_evaluate_value_proxies")
    sens = load("stage1_router_sensitivity")
    bert_qkv = load("stage2_bert_qkv")

    layer = 11
    head = 9
    target_mean_k = 64
    threshold = 0.90
    iters = 12

    data = np.load(BERT_DIR / f"qkv_layer{layer}.npz")
    P_all, V_all = data["P"], data["V"]
    P, V = P_all[head], V_all[head]
    N = P.shape[0]

    doc_idx, category, text = bert_qkv.pick_document()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=N)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())

    H_norm = stage0.attention_entropy(P) / np.log(N)
    mask = H_norm >= threshold

    _, mass_ks = stage1.calibrate_tau_for_mean_k(stage0, P, target_mean_k, iters=iters)
    _, utc_ks = stage1.calibrate_utc_for_mean_k(P, V, target_mean_k, iters=iters)
    _, oracle_ks = stage1.calibrate_epsilon_for_mean_k(stage0, P, V, target_mean_k, iters=iters)
    hybrid_b_ks = sens.hybrid_b_ks(stage1, P, V, mass_ks, mask, iters=iters)
    fixed_ks = np.full(N, target_mean_k)

    methods = {
        "fixed": row_metrics(stage0, P, V, fixed_ks),
        "mass": row_metrics(stage0, P, V, mass_ks),
        "UTC": row_metrics(stage0, P, V, utc_ks),
        "hybrid-b": row_metrics(stage0, P, V, hybrid_b_ks),
        "oracle": row_metrics(stage0, P, V, oracle_ks),
    }

    print(f"[BERT head diagnosis] layer={layer}, head={head}, category={category}, tokens={N}")
    print(
        f"H_norm mean={H_norm.mean():.4f}, min={H_norm.min():.4f}, "
        f"max={H_norm.max():.4f}, frac>= {threshold:.2f} = {mask.mean():.3f}"
    )
    print("\n[method summary]")
    for name in ["fixed", "mass", "UTC", "hybrid-b", "oracle"]:
        print_method_summary(name, methods[name])

    fixed = methods["fixed"]
    print("\n[rows where adaptive loses most vs fixed: mass_rel - fixed_rel]")
    loss_order = np.argsort(-(methods["mass"]["rel_err"] - fixed["rel_err"]))[:12]
    header = (
        "row tok        H     fixed_rel mass_rel UTC_rel oracle_rel "
        "k_fix k_mass k_UTC k_orcl p_CLS p_SEP p_max top_attention"
    )
    print(header)
    print("-" * len(header))
    for row in loss_order:
        p = P[row]
        print(
            f"{row:3d} {tokens[row][:10]:<10} "
            f"{H_norm[row]:.3f} "
            f"{fixed['rel_err'][row]:9.4f} "
            f"{methods['mass']['rel_err'][row]:8.4f} "
            f"{methods['UTC']['rel_err'][row]:7.4f} "
            f"{methods['oracle']['rel_err'][row]:10.4f} "
            f"{fixed['ks'][row]:5d} "
            f"{methods['mass']['ks'][row]:6d} "
            f"{methods['UTC']['ks'][row]:5d} "
            f"{methods['oracle']['ks'][row]:6d} "
            f"{p[0]:5.3f} "
            f"{p[-1]:5.3f} "
            f"{p.max():5.3f} "
            f"{top_attention_tokens(p, tokens)}"
        )

    print("\n[budget allocation by entropy group]")
    for group_name, group_mask in [("high", mask), ("low", ~mask)]:
        if not group_mask.any():
            continue
        print(
            f"{group_name:<5} rows={group_mask.sum():3d} "
            f"H_mean={H_norm[group_mask].mean():.3f} "
            f"fixed_k={fixed['ks'][group_mask].mean():.2f} "
            f"mass_k={methods['mass']['ks'][group_mask].mean():.2f} "
            f"UTC_k={methods['UTC']['ks'][group_mask].mean():.2f} "
            f"oracle_k={methods['oracle']['ks'][group_mask].mean():.2f} "
            f"mass_max_rel={methods['mass']['rel_err'][group_mask].max():.4f} "
            f"oracle_max_rel={methods['oracle']['rel_err'][group_mask].max():.4f}"
        )

    print("\n[column attention received: possible sink keys]")
    col_mass = P.sum(axis=0)
    for idx in np.argsort(-col_mass)[:12]:
        print(
            f"col={idx:3d} tok={tokens[idx]:<10} received={col_mass[idx]:8.3f} "
            f"mean_p={col_mass[idx] / N:.4f}"
        )


if __name__ == "__main__":
    main()
