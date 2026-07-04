"""Stage 4C: GPT-2 whole-layer sparse intervention under next-token KL.

Stage 4B patched one head at a time. That is a harsh and deployment-mismatched
test: most single-head interventions have tiny downstream effects and are easy
for the model's redundancy to absorb. This script patches all heads in a GPT-2
layer simultaneously, using the same per-head exact-budget allocators as before,
then continues the real model forward and measures next-token KL/logit drift.

This is the final metric-boundary check before deciding whether the KL mismatch
is mostly a single-head artifact or a genuine downstream objective mismatch.
"""

import argparse
import csv
import importlib.util
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "metric_boundary"
DEFAULT_OUT = OUT_DIR / "stage4c_gpt2_whole_layer_kl.csv"

METHODS = ["fixed", "projected_oracle", "mass", "UTC_abs", "UTC_rel", "UTC_rel_hat"]
SCORER_METHODS = ["mass", "UTC_abs", "UTC_rel", "UTC_rel_hat"]


def load(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_layer_from_contexts(stage4b, transformer, lm_head, hidden_in, layer, method_contexts):
    """Patch the full layer attention context for every method as a batch."""
    block = transformer.h[layer]
    device = hidden_in.device
    dtype = hidden_in.dtype

    with torch.no_grad():
        residual = hidden_in.repeat(len(METHODS), 1, 1)
        contexts = []
        for name in METHODS:
            contexts.append(torch.as_tensor(method_contexts[name], dtype=dtype, device=device))
        context_heads = torch.stack(contexts, dim=0)

        attn_input = stage4b.merge_heads(context_heads)
        attn_output = block.attn.c_proj(attn_input)
        attn_output = block.attn.resid_dropout(attn_output)
        hidden = residual + attn_output

        residual = hidden
        hidden = block.ln_2(hidden)
        feed_forward = block.mlp(hidden)
        hidden = residual + feed_forward

        for next_layer in range(layer + 1, len(transformer.h)):
            hidden = stage4b.manual_block_forward(transformer.h[next_layer], hidden)
        hidden = transformer.ln_f(hidden)
        logits = lm_head(hidden)
    return logits


def dense_reconstruction_error(stage4b, transformer, lm_head, full_logits, hidden_in, layer, full_contexts):
    contexts = {name: full_contexts for name in METHODS}
    logits = run_layer_from_contexts(stage4b, transformer, lm_head, hidden_in, layer, contexts)
    diff = (logits[0] - full_logits).abs()
    return float(diff.max()), float(diff.mean())


def build_layer_contexts(stage3, stage4a, stage4b, transformer, P, V, layer, budget, min_support):
    full_contexts = np.einsum("hij,hjd->hid", P, V)
    layer_contexts = {name: full_contexts.copy() for name in METHODS}
    mean_ks = {name: [] for name in METHODS}

    for head in range(P.shape[0]):
        projector = stage4b.gpt2_projector(transformer, layer, head)
        query_rows, supports, curves, projected_oracle = stage4b.allocation_curves(
            stage3, stage4a, P[head], V[head], projector, min_support
        )
        allocations = stage4b.allocations_for_budget(
            stage3, query_rows, supports, curves, projected_oracle, budget
        )
        for name in METHODS:
            layer_contexts[name][head] = stage4b.sparse_head_context(
                P[head], V[head], query_rows, allocations[name]
            )
            mean_ks[name].append(float(np.mean(allocations[name])))
    return layer_contexts, {name: float(np.mean(vals)) for name, vals in mean_ks.items()}


def improvement_vs_fixed(fixed, value):
    if not np.isfinite(fixed) or fixed <= 1e-15:
        return np.nan
    return 1.0 - value / fixed


def summarize(rows):
    print(f"\n[Stage 4C GPT-2 whole-layer KL summary] rows={len(rows)}")
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

    fixed_total = float(np.sum([r["fixed_mean_kl"] for r in rows]))
    print("\naggregate mean-KL reduction vs fixed")
    for name in ["projected_oracle", *SCORER_METHODS]:
        total = float(np.sum([r[f"{name}_mean_kl"] for r in rows]))
        print(f"{name:<17} reduction={1.0 - total / fixed_total: .3f} total={total:.6g}")

    rel = np.array([r["UTC_rel_hat_mean_kl_improvement"] for r in rows], dtype=np.float64)
    abs_v = np.array([r["UTC_abs_mean_kl_improvement"] for r in rows], dtype=np.float64)
    rel_v = np.array([r["UTC_rel_mean_kl_improvement"] for r in rows], dtype=np.float64)
    ok = ~np.isnan(rel)
    best = rel[ok] >= np.maximum(abs_v[ok], rel_v[ok]) - 1e-12
    print(f"\nrel-hat >= max(abs, rel) on mean KL: {best.sum()}/{ok.sum()} = {best.mean():.3f}")
    print(f"rel-hat below fixed on mean KL: {(rel[ok] < 0).sum()}/{ok.sum()}")


def append_pre_registration():
    log_path = ROOT / "docs" / "experiment_log.md"
    marker = "### Stage 4C pre-registration: GPT-2 whole-layer KL intervention"
    text = log_path.read_text(encoding="utf-8")
    if marker in text:
        return
    with log_path.open("a", encoding="utf-8") as f:
        f.write(
            "\n\n"
            f"{marker}\n\n"
            "Goal: decide whether the Stage-4B KL mismatch is mostly an artifact\n"
            "of single-head marginal interventions or a genuine downstream-objective\n"
            "mismatch. Patch all heads in one GPT-2 layer simultaneously using the\n"
            "same per-head exact-budget allocators, continue the real causal model,\n"
            "and measure next-token KL/logit drift against dense GPT-2.\n\n"
            "Binding fork before running:\n\n"
            "1. If the scorer ladder re-emerges under whole-layer intervention,\n"
            "   then Stage 4B's failure is largely a single-head redundancy/noise\n"
            "   artifact and local sparse-attention control remains behaviorally\n"
            "   useful at deployment-shaped scale.\n"
            "2. If the ladder still fails, then the boundary is deeper: local\n"
            "   attention/W_O error is insufficient as a behavioral metric, and any\n"
            "   future scorer should be KL-aware/layer-aware rather than just a\n"
            "   sharper local-error estimator.\n"
            "3. The projected oracle remains a local W_O oracle, not a KL oracle.\n"
            "   Its performance under KL is diagnostic only.\n"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", type=int, default=3)
    parser.add_argument("--skip", type=int, default=11)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 5, 11])
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
        args.budgets = args.budgets[:2]
        args.out = OUT_DIR / "stage4c_gpt2_whole_layer_kl_smoke.csv"

    sweep = load("stage2_bert_abs_rel_sweep")
    stage3 = load("stage3_gpt2_cross_model")
    stage4a = load("stage4_metric_boundary_wo")
    stage4b = load("stage4b_gpt2_logit_kl")

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
                P, V, anchor = stage4b.extract_p_v(transformer, out.hidden_states, attentions, layer)
                hidden_in = out.hidden_states[layer]
                full_contexts = np.einsum("hij,hjd->hid", P, V)
                max_diff, mean_diff = dense_reconstruction_error(
                    stage4b, transformer, model.lm_head, full_logits, hidden_in, layer, full_contexts
                )
                print(
                    f"  layer={layer:02d} anchor={anchor:.2e} "
                    f"dense_logit_diff(max={max_diff:.2e}, mean={mean_diff:.2e})"
                )
                for budget in args.budgets:
                    layer_contexts, mean_ks = build_layer_contexts(
                        stage3, stage4a, stage4b, transformer, P, V, layer, budget, args.min_support
                    )
                    logits = run_layer_from_contexts(
                        stage4b, transformer, model.lm_head, hidden_in, layer, layer_contexts
                    )
                    drift = stage4b.kl_and_logit_drift(full_logits, logits, args.min_support - 1)
                    fixed_i = METHODS.index("fixed")
                    row = {
                        "model": "gpt2",
                        "doc_idx": doc_idx,
                        "category": category,
                        "layer": layer,
                        "tokens": n_tokens,
                        "budget": budget,
                        "eval_positions": n_tokens - args.min_support,
                        "anchor_max_dP": anchor,
                        "dense_max_logit_diff": max_diff,
                        "dense_mean_logit_diff": mean_diff,
                    }
                    for i, name in enumerate(METHODS):
                        row[f"{name}_mean_k"] = mean_ks[name]
                        for metric in ["mean_kl", "max_kl", "mean_logit_l2", "max_logit_l2"]:
                            val = float(drift[metric][i])
                            row[f"{name}_{metric}"] = val
                            row[f"{name}_{metric}_improvement"] = improvement_vs_fixed(
                                float(drift[metric][fixed_i]), val
                            )
                    rows.append(row)
                print(f"    rows={len(rows)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nsaved: {args.out}")
    summarize(rows)


if __name__ == "__main__":
    main()
