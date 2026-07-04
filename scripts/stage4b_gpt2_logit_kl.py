"""Stage 4B: GPT-2 next-token logit/KL metric-boundary probe.

Stage 4A showed that the scorer ladder survives projection through each
head's W_O slice. This script pushes one rung further: replace one GPT-2
head's attention output with the sparse output selected by each allocator,
continue the real model forward pass, and measure next-token distribution
drift against the dense model.

This is intentionally a metric-boundary test, not a new allocation method.
The deployable scorers are unchanged. The offline reference named
``projected_oracle`` is the Stage-4A W_O-projected restricted oracle; it is
not claimed to be a true logit-space oracle.
"""

import argparse
import csv
import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "metric_boundary"
DEFAULT_OUT = OUT_DIR / "stage4b_gpt2_logit_kl.csv"

METHODS = ["fixed", "projected_oracle", "mass", "UTC_abs", "UTC_rel", "UTC_rel_hat"]
SCORER_METHODS = ["mass", "UTC_abs", "UTC_rel", "UTC_rel_hat"]


def load(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def split_heads(x, n_heads):
    batch, n_tokens, hidden = x.shape
    head_dim = hidden // n_heads
    return x.view(batch, n_tokens, n_heads, head_dim).permute(0, 2, 1, 3)


def merge_heads(x):
    batch, n_heads, n_tokens, head_dim = x.shape
    return x.permute(0, 2, 1, 3).contiguous().view(batch, n_tokens, n_heads * head_dim)


def manual_attention_context(block, hidden):
    n_heads = block.attn.num_heads
    head_dim = block.attn.head_dim
    qkv = block.attn.c_attn(hidden)
    q, k, v = qkv.split(block.attn.embed_dim, dim=2)
    qh = split_heads(q, n_heads)
    kh = split_heads(k, n_heads)
    vh = split_heads(v, n_heads)

    scores = (qh @ kh.transpose(-1, -2)) / np.sqrt(head_dim)
    n_tokens = hidden.shape[1]
    mask = torch.tril(torch.ones(n_tokens, n_tokens, dtype=torch.bool, device=hidden.device))
    scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores, dim=-1)
    return probs @ vh


def manual_block_forward(block, hidden):
    residual = hidden
    normed = block.ln_1(hidden)
    context = manual_attention_context(block, normed)
    attn_output = block.attn.c_proj(merge_heads(context))
    attn_output = block.attn.resid_dropout(attn_output)
    hidden = residual + attn_output

    residual = hidden
    hidden = block.ln_2(hidden)
    feed_forward = block.mlp(hidden)
    return residual + feed_forward


def extract_p_v(transformer, hidden_states, attentions, layer):
    """Recompute P,V for all heads and anchor against model attentions."""
    block = transformer.h[layer]
    h_in = block.ln_1(hidden_states[layer])  # (1, N, H)
    n_heads = transformer.config.n_head
    head_dim = transformer.config.n_embd // n_heads

    with torch.no_grad():
        qkv = block.attn.c_attn(h_in)
    q, k, v = qkv.split(transformer.config.n_embd, dim=2)
    qh = split_heads(q, n_heads)[0].double()
    kh = split_heads(k, n_heads)[0].double()
    vh = split_heads(v, n_heads)[0].double()

    logits = (qh @ kh.transpose(1, 2)) / np.sqrt(head_dim)
    n_tokens = logits.shape[-1]
    mask = torch.tril(torch.ones(n_tokens, n_tokens, dtype=torch.bool, device=logits.device))
    logits = logits.masked_fill(~mask, float("-inf"))
    P = torch.softmax(logits, dim=-1).cpu().numpy()
    V = vh.cpu().numpy()
    anchor = float(np.abs(P - attentions[layer]).max())
    return P, V, anchor


def gpt2_projector(transformer, layer, head):
    hidden = transformer.config.n_embd
    n_heads = transformer.config.n_head
    head_dim = hidden // n_heads
    start, end = head * head_dim, (head + 1) * head_dim
    weight = transformer.h[layer].attn.c_proj.weight.detach().double().cpu().numpy()
    # GPT-2 Conv1D: y = x @ weight + bias.
    return weight[start:end, :]


def allocation_curves(stage3, stage4a, P_head, V_head, projector, min_support):
    n_tokens = P_head.shape[0]
    query_rows = [i for i in range(n_tokens) if i + 1 >= min_support]
    supports = np.array([i + 1 for i in query_rows], dtype=int)
    width = int(supports.max())

    curves = {name: np.zeros((len(query_rows), width), dtype=np.float64) for name in ["oracle", *SCORER_METHODS]}
    prefix_sums = np.cumsum(V_head, axis=0)
    for r, i in enumerate(query_rows):
        s = i + 1
        row_curves = stage3.causal_score_curves(P_head[i, :s], V_head[:s], prefix_sums[i])
        for name in curves:
            curves[name][r, :s] = row_curves[name]

    projected_oracle = stage4a.projected_oracle_curve_causal(
        P_head, V_head, projector, query_rows, supports
    )
    return query_rows, supports, curves, projected_oracle


def allocations_for_budget(stage3, query_rows, supports, curves, projected_oracle, budget):
    fixed = np.minimum(np.full(len(query_rows), budget, dtype=int), supports)
    allocations = {
        "fixed": fixed,
        "projected_oracle": stage3.exact_budget_allocation(projected_oracle, supports, budget),
    }
    for name in SCORER_METHODS:
        allocations[name] = stage3.exact_budget_allocation(curves[name], supports, budget)
    return allocations


def sparse_head_context(P_head, V_head, query_rows, ks):
    """Build one head's sparse renormalized context for all token rows."""
    full = P_head @ V_head
    context = full.copy()
    for row, k in zip(query_rows, ks):
        s = row + 1
        if k >= s:
            continue
        p = P_head[row, :s]
        idx = np.argsort(-p)[: int(k)]
        mass = p[idx].sum()
        if mass <= 1e-15:
            continue
        context[row] = (p[idx, None] * V_head[idx]).sum(axis=0) / mass
    return context


def run_layer_from_patched_head(transformer, lm_head, hidden_in, layer, head, full_contexts, method_contexts):
    """Patch one head at ``layer`` and continue the model for all methods as a batch."""
    block = transformer.h[layer]
    n_heads = transformer.config.n_head
    head_dim = transformer.config.n_embd // n_heads
    device = hidden_in.device
    dtype = hidden_in.dtype

    with torch.no_grad():
        residual = hidden_in.repeat(len(METHODS), 1, 1)
        context_heads = torch.as_tensor(full_contexts, dtype=dtype, device=device)
        context_heads = context_heads.unsqueeze(0).repeat(len(METHODS), 1, 1, 1)

        patched = []
        for name in METHODS:
            patched.append(torch.as_tensor(method_contexts[name], dtype=dtype, device=device))
        patched = torch.stack(patched, dim=0)
        context_heads[:, head, :, :] = patched

        attn_input = merge_heads(context_heads)
        attn_output = block.attn.c_proj(attn_input)
        attn_output = block.attn.resid_dropout(attn_output)
        hidden = residual + attn_output

        residual = hidden
        hidden = block.ln_2(hidden)
        feed_forward = block.mlp(hidden)
        hidden = residual + feed_forward

        for next_layer in range(layer + 1, len(transformer.h)):
            hidden = manual_block_forward(transformer.h[next_layer], hidden)
        hidden = transformer.ln_f(hidden)
        logits = lm_head(hidden)
    return logits


def kl_and_logit_drift(full_logits, sparse_logits, min_pos):
    """Mean/max next-token KL and logit L2 over positions min_pos..N-2."""
    if sparse_logits.ndim == 2:
        sparse_logits = sparse_logits.unsqueeze(0)
    start = min_pos
    stop = full_logits.shape[0] - 1
    full_slice = full_logits[start:stop]
    sparse_slice = sparse_logits[:, start:stop, :]
    if full_slice.shape[0] == 0:
        return {
            "mean_kl": np.nan,
            "max_kl": np.nan,
            "mean_logit_l2": np.nan,
            "max_logit_l2": np.nan,
        }

    full_logp = F.log_softmax(full_slice, dim=-1)
    full_p = full_logp.exp()
    sparse_logp = F.log_softmax(sparse_slice, dim=-1)
    kl = (full_p.unsqueeze(0) * (full_logp.unsqueeze(0) - sparse_logp)).sum(dim=-1)
    l2 = torch.linalg.vector_norm(sparse_slice - full_slice.unsqueeze(0), dim=-1)
    return {
        "mean_kl": kl.mean(dim=1).detach().cpu().numpy(),
        "max_kl": kl.max(dim=1).values.detach().cpu().numpy(),
        "mean_logit_l2": l2.mean(dim=1).detach().cpu().numpy(),
        "max_logit_l2": l2.max(dim=1).values.detach().cpu().numpy(),
    }


def improvement_vs_fixed(fixed, value):
    if not np.isfinite(fixed) or fixed <= 1e-15:
        return np.nan
    return 1.0 - value / fixed


def summarize(rows):
    print(f"\n[Stage 4B GPT-2 logit/KL summary] rows={len(rows)}")
    for metric in ["mean_kl", "max_kl", "mean_logit_l2"]:
        print(f"\nmetric={metric} improvement vs fixed")
        for name in ["projected_oracle", *SCORER_METHODS]:
            vals = np.array([r[f"{name}_{metric}_improvement"] for r in rows], dtype=np.float64)
            vals = vals[~np.isnan(vals)]
            print(
                f"{name:<17} mean={vals.mean(): .3f} "
                f"median={np.median(vals): .3f} p10={np.quantile(vals, 0.1): .3f} "
                f"below0={(vals < 0).sum()}"
            )

    rel = np.array([r["UTC_rel_hat_mean_kl_improvement"] for r in rows], dtype=np.float64)
    abs_v = np.array([r["UTC_abs_mean_kl_improvement"] for r in rows], dtype=np.float64)
    rel_v = np.array([r["UTC_rel_mean_kl_improvement"] for r in rows], dtype=np.float64)
    ok = ~np.isnan(rel)
    best = rel[ok] >= np.maximum(abs_v[ok], rel_v[ok]) - 1e-12
    print(f"\nrel-hat >= max(abs, rel) on mean KL: {best.sum()}/{ok.sum()} = {best.mean():.3f}")
    print(f"rel-hat below fixed on mean KL: {(rel[ok] < 0).sum()}/{ok.sum()}")


def append_pre_registration():
    log_path = ROOT / "docs" / "experiment_log.md"
    marker = "### Stage 4B pre-registration: GPT-2 next-token KL/logit drift"
    text = log_path.read_text(encoding="utf-8")
    if marker in text:
        return
    with log_path.open("a", encoding="utf-8") as f:
        f.write(
            "\n\n"
            f"{marker}\n\n"
            "Goal: test whether the Stage-4 scorer ladder survives a real causal-LM\n"
            "behavior metric, not just head-space or W_O-projected vector error.\n"
            "For each GPT-2 document/layer/head/budget, replace one head's attention\n"
            "context with the sparse context selected by each allocator, continue the\n"
            "actual model forward pass, and measure next-token KL/logit drift against\n"
            "the dense model.\n\n"
            "Binding expectations before running:\n\n"
            "1. Rel-hat should remain the leading deployable scorer by mean KL\n"
            "   improvement versus fixed-k, though further attenuation is expected.\n"
            "2. The ladder should not collapse into rel-hat being broadly worse than\n"
            "   fixed-k. A small number of failures is acceptable; widespread negative\n"
            "   improvement is a metric-boundary failure.\n"
            "3. The Stage-4A projected oracle is an offline reference, not a true\n"
            "   logit-space oracle. If rel-hat beats it under KL, treat that as metric\n"
            "   mismatch/discreteness, not a miracle.\n"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", type=int, default=3)
    parser.add_argument("--skip", type=int, default=11)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 5, 11])
    parser.add_argument("--heads", type=int, nargs="+", default=list(range(12)))
    parser.add_argument("--budgets", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--min-support", type=int, default=65)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    append_pre_registration()

    if args.smoke:
        args.docs = 1
        args.layers = args.layers[:1]
        args.heads = args.heads[:2]
        args.budgets = args.budgets[:2]
        args.out = OUT_DIR / "stage4b_gpt2_logit_kl_smoke.csv"

    sweep = load("stage2_bert_abs_rel_sweep")
    stage3 = load("stage3_gpt2_cross_model")
    stage4a = load("stage4_metric_boundary_wo")

    docs = sweep.select_documents(args.docs, skip=args.skip)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2", attn_implementation="eager")
    model.eval()
    transformer = model.transformer

    rows = []
    with torch.no_grad():
        for doc_num, (doc_idx, category, text) in enumerate(docs):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_tokens)
            out = model(**inputs, output_attentions=True, output_hidden_states=True)
            full_logits = out.logits[0].detach()
            attentions = [a[0].double().cpu().numpy() for a in out.attentions]
            n_tokens = int(inputs["input_ids"].shape[1])
            print(f"[doc {doc_num + 1}/{len(docs)}] idx={doc_idx}, category={category}, tokens={n_tokens}")

            for layer in args.layers:
                P, V, anchor = extract_p_v(transformer, out.hidden_states, attentions, layer)
                print(f"  layer={layer:02d} extraction anchor max|dP|={anchor:.2e}")
                hidden_in = out.hidden_states[layer]
                full_contexts = np.einsum("hij,hjd->hid", P, V)
                for head in args.heads:
                    projector = gpt2_projector(transformer, layer, head)
                    query_rows, supports, curves, projected_oracle = allocation_curves(
                        stage3, stage4a, P[head], V[head], projector, args.min_support
                    )
                    min_pos = min(query_rows)
                    for budget in args.budgets:
                        allocations = allocations_for_budget(
                            stage3, query_rows, supports, curves, projected_oracle, budget
                        )
                        contexts = {
                            name: sparse_head_context(P[head], V[head], query_rows, ks)
                            for name, ks in allocations.items()
                        }
                        sparse_logits = run_layer_from_patched_head(
                            transformer, model.lm_head, hidden_in, layer, head, full_contexts, contexts
                        )
                        drift = kl_and_logit_drift(full_logits, sparse_logits, min_pos)

                        fixed_i = METHODS.index("fixed")
                        row = {
                            "model": "gpt2",
                            "doc_idx": doc_idx,
                            "category": category,
                            "layer": layer,
                            "head": head,
                            "tokens": n_tokens,
                            "budget": budget,
                            "eval_positions": n_tokens - 1 - min_pos,
                            "anchor_max_dP": anchor,
                        }
                        for i, name in enumerate(METHODS):
                            row[f"{name}_mean_k"] = float(allocations[name].mean())
                            for metric in ["mean_kl", "max_kl", "mean_logit_l2", "max_logit_l2"]:
                                val = float(drift[metric][i])
                                row[f"{name}_{metric}"] = val
                                row[f"{name}_{metric}_improvement"] = improvement_vs_fixed(
                                    float(drift[metric][fixed_i]), val
                                )
                        rows.append(row)
                    print(f"    head={head:02d} rows={len(rows)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nsaved: {args.out}")
    summarize(rows)


if __name__ == "__main__":
    main()
