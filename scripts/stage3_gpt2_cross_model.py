"""Stage 3: cross-model validation of the scorer ladder on GPT-2 (causal).

The entire method stack (mass / UTC-abs / UTC-rel / UTC-rel-hat / oracle,
exact-budget protocol) is moved to GPT-2 small UNCHANGED. Causal attention
requires two adaptations that change no method definition:
  - each query row's support is its causal prefix; UTC's tail estimate uses
    per-row prefix value sums (exactly what a causal kernel maintains anyway);
  - per-row k is capped at the row's support, and only rows with support >= 65
    are analyzed so pruning is non-trivial.

Pre-registered expectations are in experiment_log.md (written before running).
"""

import csv
import importlib.util
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "gpt2"
OUT_CSV = OUT_DIR / "stage3_gpt2_exact_budget.csv"

LAYERS = [0, 5, 11]
BUDGETS = [8, 16, 32]
MAX_TOKENS = 128
MIN_SUPPORT = 65
METHODS = ["mass", "UTC_abs", "UTC_rel", "UTC_rel_hat"]


def load(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract_gpt2_layer(model, hidden_states, attentions, layer):
    """Recompute per-head P, V from the block's ln_1 output and c_attn weights;
    anchor against the model's reported attention probabilities."""
    block = model.h[layer]
    h_in = block.ln_1(hidden_states[layer])[0]  # (N, hidden), pre-LN input
    n_tokens = h_in.shape[0]
    n_heads = model.config.n_head
    head_dim = model.config.n_embd // n_heads

    with torch.no_grad():
        qkv = block.attn.c_attn(h_in)  # (N, 3*hidden)
    q, k, v = qkv.split(model.config.n_embd, dim=1)

    def split_heads(x):
        return x.reshape(n_tokens, n_heads, head_dim).permute(1, 0, 2).double()

    Q, K, V = split_heads(q), split_heads(k), split_heads(v)
    logits = (Q @ K.transpose(1, 2)) / np.sqrt(head_dim)
    mask = torch.tril(torch.ones(n_tokens, n_tokens, dtype=torch.bool))
    logits = logits.masked_fill(~mask, float("-inf"))
    P = torch.softmax(logits, dim=-1).numpy()

    anchor = np.abs(P - attentions[layer]).max()
    return P, V.numpy(), anchor


def causal_score_curves(p_support, V_support, prefix_v_sum, eta=1e-12):
    """Score curves over k=1..s for ONE query row with causal support s."""
    s = p_support.shape[0]
    order = np.argsort(-p_support)
    p_sorted = p_support[order]
    v_sorted = V_support[order]

    retained_mass = np.cumsum(p_sorted)
    retained_weighted = np.cumsum(p_sorted[:, None] * v_sorted, axis=0)
    retained_unweighted = np.cumsum(v_sorted, axis=0)
    full_o = p_support @ V_support
    full_norm = np.linalg.norm(full_o)

    curves = {name: np.zeros(s) for name in ["oracle", "mass", "UTC_abs", "UTC_rel", "UTC_rel_hat"]}
    for k in range(1, s):
        i = k - 1
        mass = retained_mass[i]
        dropped = max(0.0, 1.0 - mass)
        curves["mass"][i] = dropped
        if dropped <= 1e-15:
            continue
        mu_s = retained_weighted[i] / mass
        mu_r = (full_o - retained_weighted[i]) / dropped
        curves["oracle"][i] = dropped * np.linalg.norm(mu_r - mu_s) / (full_norm + eta)

        mu_r_hat = (prefix_v_sum - retained_unweighted[i]) / (s - k)
        proxy_abs = dropped * np.linalg.norm(mu_r_hat - mu_s)
        o_hat = mass * mu_s + dropped * mu_r_hat
        curves["UTC_abs"][i] = proxy_abs
        curves["UTC_rel"][i] = proxy_abs / (np.linalg.norm(mu_s) + eta)
        curves["UTC_rel_hat"][i] = proxy_abs / (np.linalg.norm(o_hat) + eta)
    return curves


def exact_budget_allocation(curve, supports, target_mean_k):
    """Exact-budget allocation with per-row k capped at the row's support."""
    n_rows, width = curve.shape
    total_budget = int(round(n_rows * target_mean_k))

    values = np.unique(curve)
    lo, hi = 0, len(values) - 1
    best_ks = supports.copy()

    def ks_for(threshold):
        ok = curve <= threshold
        for r in range(n_rows):
            ok[r, supports[r] - 1] = True  # k = support always admissible
            ok[r, supports[r]:] = False
        return ok.argmax(axis=1) + 1

    while lo <= hi:
        mid = (lo + hi) // 2
        ks = ks_for(values[mid])
        if ks.sum() <= total_budget:
            best_ks = ks
            hi = mid - 1
        else:
            lo = mid + 1

    ks = best_ks.copy()
    leftover = total_budget - int(ks.sum())
    while leftover > 0:
        cand = np.flatnonzero(ks < supports)
        if cand.size == 0:
            break
        gains = curve[cand, ks[cand] - 1] - curve[cand, ks[cand]]
        chosen = cand[np.argmax(gains)]
        ks[chosen] += 1
        leftover -= 1
    return ks


def row_rel_error(p_support, V_support, k, eta=1e-12):
    s = p_support.shape[0]
    if k >= s:
        return 0.0
    idx = np.argsort(-p_support)[:k]
    m = p_support[idx].sum()
    delta = 1.0 - m
    if delta <= 1e-15:
        return 0.0
    mu_s = (p_support[idx, None] * V_support[idx]).sum(axis=0) / m
    o = p_support @ V_support
    return np.linalg.norm(o - mu_s) / (np.linalg.norm(o) + eta)


def analyze_head(P_head, V_head, budget):
    n_tokens = P_head.shape[0]
    query_rows = [i for i in range(n_tokens) if i + 1 >= MIN_SUPPORT]
    n_rows = len(query_rows)
    supports = np.array([i + 1 for i in query_rows])
    width = int(supports.max())

    curves = {name: np.zeros((n_rows, width)) for name in ["oracle", "mass", "UTC_abs", "UTC_rel", "UTC_rel_hat"]}
    prefix_sums = np.cumsum(V_head, axis=0)
    for r, i in enumerate(query_rows):
        s = i + 1
        row_curves = causal_score_curves(P_head[i, :s], V_head[:s], prefix_sums[i])
        for name in curves:
            curves[name][r, :s] = row_curves[name]

    def max_rel_for(ks):
        return max(
            row_rel_error(P_head[i, : i + 1], V_head[: i + 1], int(k))
            for i, k in zip(query_rows, ks)
        )

    fixed_ks = np.minimum(np.full(n_rows, budget), supports)
    fixed_max = max_rel_for(fixed_ks)

    result = {"fixed_max_rel": fixed_max}
    oracle_ks = exact_budget_allocation(curves["oracle"], supports, budget)
    oracle_max = max_rel_for(oracle_ks)
    result["oracle_max_rel"] = oracle_max
    gap = fixed_max - oracle_max

    for name in METHODS:
        ks = exact_budget_allocation(curves[name], supports, budget)
        assert abs(ks.mean() - budget) < 0.01, f"budget mismatch for {name}"
        m = max_rel_for(ks)
        result[f"{name}_max_rel"] = m
        result[f"{name}_gap"] = (fixed_max - m) / gap if gap > 1e-15 else np.nan
    return result


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sweep = load("stage2_bert_abs_rel_sweep")

    docs = sweep.select_documents(3, skip=11)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModel.from_pretrained("gpt2", attn_implementation="eager")
    model.eval()

    fieldnames = None
    all_rows = []
    for doc_num, (doc_idx, category, text) in enumerate(docs):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_TOKENS)
        n_tokens = int(inputs["input_ids"].shape[1])
        with torch.no_grad():
            out = model(**inputs, output_attentions=True, output_hidden_states=True)
        attentions = [a[0].double().numpy() for a in out.attentions]

        print(f"[doc {doc_num + 1}/3] idx={doc_idx}, category={category}, tokens={n_tokens}")
        for layer in LAYERS:
            P, V, anchor = extract_gpt2_layer(model, out.hidden_states, attentions, layer)
            print(f"  layer {layer}: extraction anchor max|dP| = {anchor:.2e}")
            for head in range(P.shape[0]):
                for budget in BUDGETS:
                    metrics = analyze_head(P[head], V[head], budget)
                    row = {
                        "doc_idx": doc_idx, "category": category, "layer": layer,
                        "head": head, "tokens": n_tokens, "budget": budget, **metrics,
                    }
                    all_rows.append(row)
                    if fieldnames is None:
                        fieldnames = list(row.keys())

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nsaved: {OUT_CSV} ({len(all_rows)} rows)")
    print("\n[Stage 3 GPT-2 summary]")
    for name in METHODS:
        gaps = np.array([r[f"{name}_gap"] for r in all_rows])
        valid = gaps[~np.isnan(gaps)]
        print(f"{name:<12} mean={valid.mean():.3f} median={np.median(valid):.3f} p10={np.quantile(valid, 0.1):.3f}")

    hat = np.array([r["UTC_rel_hat_gap"] for r in all_rows])
    abs_g = np.array([r["UTC_abs_gap"] for r in all_rows])
    rel_g = np.array([r["UTC_rel_gap"] for r in all_rows])
    ok = ~np.isnan(hat)
    best = (hat[ok] >= np.maximum(abs_g[ok], rel_g[ok]) - 1e-12)
    print(f"\nrel-hat >= max(abs, rel): {best.sum()}/{ok.sum()} = {best.mean():.3f}")
    print(f"rel-hat catastrophic (< -1): {(hat[ok] < -1).sum()}")
    print(f"rel-hat below fixed (< 0): {(hat[ok] < 0).sum()}")


if __name__ == "__main__":
    main()
