"""Stage 4A: metric-boundary test with W_O-projected head errors.

Previous stages evaluated sparse attention in each head's value-output space:

    ||o_head - sparse(o_head)|| / ||o_head||

This script keeps the same exact-budget allocation protocol and deployable
scorers, but evaluates errors after each head's contribution is projected by
the attention output projection W_O. The projected restricted oracle is
recomputed under this metric.
"""

import argparse
import csv
import importlib.util
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "metric_boundary"
METHODS = ["mass", "UTC_abs", "UTC_rel", "UTC_rel_hat"]


def load(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def projected_oracle_curve_full(P, V, projector, eta=1e-12):
    """True relative error curve after per-head W_O projection."""
    n_rows, n = P.shape
    curve = np.zeros((n_rows, n), dtype=np.float64)
    for row, p in enumerate(P):
        order = np.argsort(-p)
        p_sorted = p[order]
        v_sorted = V[order]
        retained_mass = np.cumsum(p_sorted)
        retained_weighted = np.cumsum(p_sorted[:, None] * v_sorted, axis=0)
        full_o = p @ V
        full_proj = full_o @ projector
        denom = np.linalg.norm(full_proj) + eta
        for k_idx in range(n - 1):
            mu_s = retained_weighted[k_idx] / retained_mass[k_idx]
            err = (full_o - mu_s) @ projector
            curve[row, k_idx] = np.linalg.norm(err) / denom
    return curve


def projected_max_rel_full(P, V, ks, projector, eta=1e-12):
    vals = []
    for row, p in enumerate(P):
        k = int(ks[row])
        if k >= P.shape[1]:
            vals.append(0.0)
            continue
        idx = np.argsort(-p)[:k]
        mu_s = (p[idx, None] * V[idx]).sum(axis=0) / p[idx].sum()
        full_o = p @ V
        vals.append(
            np.linalg.norm((full_o - mu_s) @ projector)
            / (np.linalg.norm(full_o @ projector) + eta)
        )
    return float(np.max(vals))


def projected_mean_rel_full(P, V, ks, projector, eta=1e-12):
    vals = []
    for row, p in enumerate(P):
        k = int(ks[row])
        if k >= P.shape[1]:
            vals.append(0.0)
            continue
        idx = np.argsort(-p)[:k]
        mu_s = (p[idx, None] * V[idx]).sum(axis=0) / p[idx].sum()
        full_o = p @ V
        vals.append(
            np.linalg.norm((full_o - mu_s) @ projector)
            / (np.linalg.norm(full_o @ projector) + eta)
        )
    return float(np.mean(vals))


def gap_closed(fixed_max, oracle_max, method_max):
    gap = fixed_max - oracle_max
    if gap <= 1e-15:
        return np.nan
    return (fixed_max - method_max) / gap


def bert_projector(model, layer, head):
    hidden = model.config.hidden_size
    n_heads = model.config.num_attention_heads
    head_dim = hidden // n_heads
    start, end = head * head_dim, (head + 1) * head_dim
    weight = model.encoder.layer[layer].attention.output.dense.weight.detach().double().numpy()
    # PyTorch Linear: y = x @ weight.T + bias. Slice input columns for this head.
    return weight[:, start:end].T


def analyze_bert_head(stage2c, P, V, projector, budget):
    n = P.shape[0]
    fixed_ks = np.full(n, budget, dtype=int)
    scorer_curves = stage2c.score_curves(P, V)
    projected_oracle = projected_oracle_curve_full(P, V, projector)

    allocations = {
        "oracle": stage2c.threshold_exact_budget_allocation(projected_oracle, budget)[1],
    }
    for name in METHODS:
        allocations[name] = stage2c.threshold_exact_budget_allocation(scorer_curves[name], budget)[1]

    fixed_max = projected_max_rel_full(P, V, fixed_ks, projector)
    fixed_mean = projected_mean_rel_full(P, V, fixed_ks, projector)
    oracle_max = projected_max_rel_full(P, V, allocations["oracle"], projector)
    oracle_mean = projected_mean_rel_full(P, V, allocations["oracle"], projector)

    row = {
        "fixed_max_rel": fixed_max,
        "fixed_mean_rel": fixed_mean,
        "oracle_max_rel": oracle_max,
        "oracle_mean_rel": oracle_mean,
        "oracle_mean_k": float(allocations["oracle"].mean()),
    }
    for name in METHODS:
        method_max = projected_max_rel_full(P, V, allocations[name], projector)
        method_mean = projected_mean_rel_full(P, V, allocations[name], projector)
        row[f"{name}_max_rel"] = method_max
        row[f"{name}_mean_rel"] = method_mean
        row[f"{name}_mean_k"] = float(allocations[name].mean())
        row[f"{name}_gap"] = gap_closed(fixed_max, oracle_max, method_max)
    row["rel_hat_below_fixed"] = row["UTC_rel_hat_max_rel"] > fixed_max
    row["rel_hat_beats_abs_rel"] = row["UTC_rel_hat_max_rel"] <= min(
        row["UTC_abs_max_rel"], row["UTC_rel_max_rel"]
    )
    row["rel_hat_beats_mass"] = row["UTC_rel_hat_max_rel"] <= row["mass_max_rel"]
    return row


def gpt2_projector(model, layer, head):
    hidden = model.config.n_embd
    n_heads = model.config.n_head
    head_dim = hidden // n_heads
    start, end = head * head_dim, (head + 1) * head_dim
    weight = model.h[layer].attn.c_proj.weight.detach().double().numpy()
    # GPT-2 Conv1D: y = x @ weight + bias. Slice input rows for this head.
    return weight[start:end, :]


def projected_oracle_curve_causal(P_head, V_head, projector, query_rows, supports, eta=1e-12):
    width = int(supports.max())
    curve = np.zeros((len(query_rows), width), dtype=np.float64)
    for r, i in enumerate(query_rows):
        s = i + 1
        p = P_head[i, :s]
        V = V_head[:s]
        order = np.argsort(-p)
        p_sorted = p[order]
        v_sorted = V[order]
        retained_mass = np.cumsum(p_sorted)
        retained_weighted = np.cumsum(p_sorted[:, None] * v_sorted, axis=0)
        full_o = p @ V
        denom = np.linalg.norm(full_o @ projector) + eta
        for k_idx in range(s - 1):
            mu_s = retained_weighted[k_idx] / retained_mass[k_idx]
            curve[r, k_idx] = np.linalg.norm((full_o - mu_s) @ projector) / denom
    return curve


def projected_max_rel_causal(P_head, V_head, ks, projector, query_rows, eta=1e-12):
    vals = []
    for i, k in zip(query_rows, ks):
        s = i + 1
        if k >= s:
            vals.append(0.0)
            continue
        p = P_head[i, :s]
        V = V_head[:s]
        idx = np.argsort(-p)[: int(k)]
        mu_s = (p[idx, None] * V[idx]).sum(axis=0) / p[idx].sum()
        full_o = p @ V
        vals.append(
            np.linalg.norm((full_o - mu_s) @ projector)
            / (np.linalg.norm(full_o @ projector) + eta)
        )
    return float(np.max(vals))


def projected_mean_rel_causal(P_head, V_head, ks, projector, query_rows, eta=1e-12):
    vals = []
    for i, k in zip(query_rows, ks):
        s = i + 1
        if k >= s:
            vals.append(0.0)
            continue
        p = P_head[i, :s]
        V = V_head[:s]
        idx = np.argsort(-p)[: int(k)]
        mu_s = (p[idx, None] * V[idx]).sum(axis=0) / p[idx].sum()
        full_o = p @ V
        vals.append(
            np.linalg.norm((full_o - mu_s) @ projector)
            / (np.linalg.norm(full_o @ projector) + eta)
        )
    return float(np.mean(vals))


def analyze_gpt2_head(stage3, P_head, V_head, projector, budget):
    n_tokens = P_head.shape[0]
    query_rows = [i for i in range(n_tokens) if i + 1 >= stage3.MIN_SUPPORT]
    supports = np.array([i + 1 for i in query_rows])
    width = int(supports.max())

    scorer_curves = {name: np.zeros((len(query_rows), width)) for name in ["oracle", *METHODS]}
    prefix_sums = np.cumsum(V_head, axis=0)
    for r, i in enumerate(query_rows):
        s = i + 1
        row_curves = stage3.causal_score_curves(P_head[i, :s], V_head[:s], prefix_sums[i])
        for name in scorer_curves:
            scorer_curves[name][r, :s] = row_curves[name]

    projected_oracle = projected_oracle_curve_causal(P_head, V_head, projector, query_rows, supports)
    fixed_ks = np.minimum(np.full(len(query_rows), budget), supports)
    allocations = {
        "oracle": stage3.exact_budget_allocation(projected_oracle, supports, budget),
    }
    for name in METHODS:
        allocations[name] = stage3.exact_budget_allocation(scorer_curves[name], supports, budget)

    fixed_max = projected_max_rel_causal(P_head, V_head, fixed_ks, projector, query_rows)
    fixed_mean = projected_mean_rel_causal(P_head, V_head, fixed_ks, projector, query_rows)
    oracle_max = projected_max_rel_causal(P_head, V_head, allocations["oracle"], projector, query_rows)
    oracle_mean = projected_mean_rel_causal(P_head, V_head, allocations["oracle"], projector, query_rows)

    row = {
        "fixed_max_rel": fixed_max,
        "fixed_mean_rel": fixed_mean,
        "oracle_max_rel": oracle_max,
        "oracle_mean_rel": oracle_mean,
        "oracle_mean_k": float(allocations["oracle"].mean()),
    }
    for name in METHODS:
        method_max = projected_max_rel_causal(P_head, V_head, allocations[name], projector, query_rows)
        method_mean = projected_mean_rel_causal(P_head, V_head, allocations[name], projector, query_rows)
        row[f"{name}_max_rel"] = method_max
        row[f"{name}_mean_rel"] = method_mean
        row[f"{name}_mean_k"] = float(allocations[name].mean())
        row[f"{name}_gap"] = gap_closed(fixed_max, oracle_max, method_max)
    row["rel_hat_below_fixed"] = row["UTC_rel_hat_max_rel"] > fixed_max
    row["rel_hat_beats_abs_rel"] = row["UTC_rel_hat_max_rel"] <= min(
        row["UTC_abs_max_rel"], row["UTC_rel_max_rel"]
    )
    row["rel_hat_beats_mass"] = row["UTC_rel_hat_max_rel"] <= row["mass_max_rel"]
    return row


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(name, rows):
    print(f"\n[{name} W_O-projected summary]")
    print(f"rows={len(rows)}")
    for method in METHODS:
        vals = np.array([r[f"{method}_gap"] for r in rows], dtype=np.float64)
        vals = vals[~np.isnan(vals)]
        print(
            f"{method:<12} mean={vals.mean():.3f} "
            f"median={np.median(vals):.3f} p10={np.quantile(vals, 0.1):.3f}"
        )
    hat = np.array([r["UTC_rel_hat_gap"] for r in rows], dtype=np.float64)
    abs_g = np.array([r["UTC_abs_gap"] for r in rows], dtype=np.float64)
    rel_g = np.array([r["UTC_rel_gap"] for r in rows], dtype=np.float64)
    ok = ~np.isnan(hat)
    print(
        f"rel-hat >= max(abs, rel): "
        f"{(hat[ok] >= np.maximum(abs_g[ok], rel_g[ok]) - 1e-12).sum()}/{ok.sum()} "
        f"= {np.mean(hat[ok] >= np.maximum(abs_g[ok], rel_g[ok]) - 1e-12):.3f}"
    )
    print(f"rel-hat below fixed (<0): {(hat[ok] < 0).sum()}/{ok.sum()}")


def run_bert(args):
    sweep = load("stage2_bert_abs_rel_sweep")
    stage2c = load("stage2c_bert_exact_budget")
    docs = sweep.select_documents(args.bert_docs, skip=1)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased", attn_implementation="eager")
    model.eval()

    rows = []
    for doc_num, (doc_idx, category, text) in enumerate(docs):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_tokens)
        extracted = sweep.extract_layers(model, inputs, args.bert_layers)
        print(f"[BERT doc {doc_num + 1}/{len(docs)}] idx={doc_idx}, category={category}")
        for layer in args.bert_layers:
            P_all, V_all = extracted[layer]
            for head in range(P_all.shape[0]):
                projector = bert_projector(model, layer, head)
                for budget in args.budgets:
                    metrics = analyze_bert_head(stage2c, P_all[head], V_all[head], projector, budget)
                    rows.append({
                        "model": "bert-base-uncased",
                        "doc_idx": doc_idx,
                        "category": category,
                        "layer": layer,
                        "head": head,
                        "tokens": int(inputs["input_ids"].shape[1]),
                        "budget": budget,
                        **metrics,
                    })
                print(f"  BERT layer={layer:02d} head={head:02d} rows={len(rows)}")
    return rows


def run_gpt2(args):
    sweep = load("stage2_bert_abs_rel_sweep")
    stage3 = load("stage3_gpt2_cross_model")
    docs = sweep.select_documents(args.gpt2_docs, skip=11)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModel.from_pretrained("gpt2", attn_implementation="eager")
    model.eval()

    rows = []
    for doc_num, (doc_idx, category, text) in enumerate(docs):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_tokens)
        with torch.no_grad():
            out = model(**inputs, output_attentions=True, output_hidden_states=True)
        attentions = [a[0].double().numpy() for a in out.attentions]
        print(f"[GPT-2 doc {doc_num + 1}/{len(docs)}] idx={doc_idx}, category={category}")
        for layer in args.gpt2_layers:
            P, V, anchor = stage3.extract_gpt2_layer(model, out.hidden_states, attentions, layer)
            print(f"  GPT-2 layer={layer:02d} anchor={anchor:.2e}")
            for head in range(P.shape[0]):
                projector = gpt2_projector(model, layer, head)
                for budget in args.budgets:
                    metrics = analyze_gpt2_head(stage3, P[head], V[head], projector, budget)
                    rows.append({
                        "model": "gpt2",
                        "doc_idx": doc_idx,
                        "category": category,
                        "layer": layer,
                        "head": head,
                        "tokens": int(inputs["input_ids"].shape[1]),
                        "budget": budget,
                        **metrics,
                    })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert-docs", type=int, default=10)
    parser.add_argument("--gpt2-docs", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--bert-layers", type=int, nargs="+", default=list(range(12)))
    parser.add_argument("--gpt2-layers", type=int, nargs="+", default=[0, 5, 11])
    parser.add_argument("--budgets", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--bert-out", type=Path, default=OUT_DIR / "stage4a_bert_wo_projected.csv")
    parser.add_argument("--gpt2-out", type=Path, default=OUT_DIR / "stage4a_gpt2_wo_projected.csv")
    parser.add_argument("--model", choices=["bert", "gpt2", "both"], default="both")
    args = parser.parse_args()

    if args.model in {"bert", "both"}:
        bert_rows = run_bert(args)
        write_csv(args.bert_out, bert_rows)
        summarize("BERT", bert_rows)
        print(f"saved: {args.bert_out}")
    if args.model in {"gpt2", "both"}:
        gpt2_rows = run_gpt2(args)
        write_csv(args.gpt2_out, gpt2_rows)
        summarize("GPT-2", gpt2_rows)
        print(f"saved: {args.gpt2_out}")


if __name__ == "__main__":
    main()
