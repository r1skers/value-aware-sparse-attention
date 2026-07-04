"""Held-out BERT sweep for UTC-abs, UTC-rel, and UTC-rel-hat.

This originally tested the selector hypothesis from the Stage 1 consistency
patch:

  norm-CV high AND budget not tight -> UTC-rel should beat UTC-abs
  otherwise                         -> UTC-abs should be safer

That selector failed. The next hypothesis is that UTC-rel-hat fixes the
denominator instead of selecting between abs and rel:

  o_hat = (1 - delta) * mu_S + delta * mu_R_hat
  score = delta * ||mu_R_hat - mu_S|| / (||o_hat|| + eta)

No thresholds are tuned inside this script.
"""

import argparse
import csv
import importlib.util
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import fetch_20newsgroups
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "bert"


def load(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_documents(count, min_chars=2000, skip=1):
    bunch = fetch_20newsgroups(
        subset="train", remove=("headers", "footers", "quotes")
    )
    docs = []
    for idx, text in enumerate(bunch.data):
        if len(text) < min_chars:
            continue
        if len(docs) < skip:
            docs.append(None)
            continue
        docs.append((idx, bunch.target_names[bunch.target[idx]], text))
        if len([d for d in docs if d is not None]) >= count:
            break
    selected = [d for d in docs if d is not None]
    if len(selected) < count:
        raise RuntimeError("not enough long 20 Newsgroups documents")
    return selected


def extract_layers(model, inputs, layers):
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)

    n_tokens = inputs["input_ids"].shape[1]
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads
    extracted = {}

    for layer in layers:
        h_in = out.hidden_states[layer][0].double()
        sa = model.encoder.layer[layer].attention.self

        def split_heads(x):
            return x.reshape(n_tokens, n_heads, head_dim).permute(1, 0, 2)

        with torch.no_grad():
            Q = split_heads(sa.query(h_in.float()).double())
            K = split_heads(sa.key(h_in.float()).double())
            V = split_heads(sa.value(h_in.float()).double())

        logits = Q @ K.transpose(1, 2) / np.sqrt(head_dim)
        P = torch.softmax(logits, dim=-1).numpy()
        extracted[layer] = (P, V.numpy())

    return extracted


def retained_norms(P, V, k):
    norms = np.empty(P.shape[0], dtype=np.float64)
    for row, p in enumerate(P):
        idx = np.argsort(-p)[:k]
        mass = p[idx].sum()
        mu_s = (p[idx, None] * V[idx]).sum(axis=0) / mass
        norms[row] = np.linalg.norm(mu_s)
    return norms


def max_rel(stage0, P, V, ks):
    stats = stage0.check_all_rows_decomposition(P, V, k=ks)
    return stage0.relative_errors(P, V, stats).max()


def gap_closed(stage0, P, V, fixed_ks, oracle_ks, ks):
    fixed_m = max_rel(stage0, P, V, fixed_ks)
    oracle_m = max_rel(stage0, P, V, oracle_ks)
    gap = fixed_m - oracle_m
    return (fixed_m - max_rel(stage0, P, V, ks)) / gap if gap > 1e-15 else np.nan


def predicted_winner(mu_s_cv, budget_frac, cv_threshold=0.25, tight_frac=0.20):
    if mu_s_cv >= cv_threshold and budget_frac >= tight_frac:
        return "rel"
    return "abs"


def analyze_head(stage0, stage1, P, V, budget, iters, cv_threshold, tight_frac):
    n = P.shape[0]
    fixed_ks = np.full(n, budget)
    _, oracle_ks = stage1.calibrate_epsilon_for_mean_k(stage0, P, V, budget, iters=iters)
    _, mass_ks = stage1.calibrate_tau_for_mean_k(stage0, P, budget, iters=iters)
    _, abs_ks = stage1.calibrate_utc_for_mean_k(P, V, budget, iters=iters)
    _, rel_ks = stage1.calibrate_utc_rel_for_mean_k(P, V, budget, iters=iters)
    _, rel_hat_ks = stage1.calibrate_utc_rel_hat_for_mean_k(P, V, budget, iters=iters)

    mass_gap = gap_closed(stage0, P, V, fixed_ks, oracle_ks, mass_ks)
    abs_gap = gap_closed(stage0, P, V, fixed_ks, oracle_ks, abs_ks)
    rel_gap = gap_closed(stage0, P, V, fixed_ks, oracle_ks, rel_ks)
    rel_hat_gap = gap_closed(stage0, P, V, fixed_ks, oracle_ks, rel_hat_ks)

    mu_s = retained_norms(P, V, budget)
    full_norms = np.linalg.norm(P @ V, axis=1)
    H_norm = stage0.attention_entropy(P) / np.log(n)

    mu_s_cv = mu_s.std() / (mu_s.mean() + 1e-12)
    full_cv = full_norms.std() / (full_norms.mean() + 1e-12)
    pred = predicted_winner(mu_s_cv, budget / n, cv_threshold, tight_frac)
    actual = "rel" if rel_gap > abs_gap else "abs"

    return {
        "H_mean": H_norm.mean(),
        "frac_H_ge_090": (H_norm >= 0.90).mean(),
        "mu_s_norm_cv": mu_s_cv,
        "full_norm_cv": full_cv,
        "mass_gap": mass_gap,
        "UTC_abs_gap": abs_gap,
        "UTC_rel_gap": rel_gap,
        "UTC_rel_hat_gap": rel_hat_gap,
        "predicted": pred,
        "actual": actual,
        "correct": pred == actual,
        "delta_rel_abs": rel_gap - abs_gap,
        "delta_rel_hat_best": rel_hat_gap - max(abs_gap, rel_gap),
        "oracle_mean_k": oracle_ks.mean(),
        "mass_mean_k": mass_ks.mean(),
        "abs_mean_k": abs_ks.mean(),
        "rel_mean_k": rel_ks.mean(),
        "rel_hat_mean_k": rel_hat_ks.mean(),
        "budget_ok": (
            abs(oracle_ks.mean() - budget) <= 2.0
            and abs(mass_ks.mean() - budget) <= 2.0
            and abs(abs_ks.mean() - budget) <= 2.0
            and abs(rel_ks.mean() - budget) <= 2.0
            and abs(rel_hat_ks.mean() - budget) <= 2.0
        ),
    }


def summarize(rows):
    comparable = [r for r in rows if r["budget_ok"]]
    decisive = [r for r in comparable if abs(r["delta_rel_abs"]) >= 0.02]
    if decisive:
        acc = np.mean([r["correct"] for r in decisive])
    else:
        acc = np.nan

    rel_wins = [r for r in comparable if r["actual"] == "rel"]
    abs_wins = [r for r in comparable if r["actual"] == "abs"]
    rel_hat_beats_both = [
        r for r in comparable if r["UTC_rel_hat_gap"] >= max(r["UTC_abs_gap"], r["UTC_rel_gap"])
    ]
    rel_hat_disasters = [r for r in comparable if r["UTC_rel_hat_gap"] < -1.0]

    print("\n[summary]")
    print(f"rows total={len(rows)}, comparable={len(comparable)}, decisive={len(decisive)}")
    print(f"selector accuracy on decisive comparable rows={acc:.3f}")
    print(f"rel wins={len(rel_wins)}, abs wins={len(abs_wins)}")
    print(
        "rel-hat >= max(abs, rel): "
        f"{len(rel_hat_beats_both)}/{len(comparable)} "
        f"({len(rel_hat_beats_both) / len(comparable):.3f})"
        if comparable
        else "rel-hat >= max(abs, rel): n/a"
    )
    print(f"rel-hat catastrophic failures (< -1): {len(rel_hat_disasters)}")
    for budget in sorted({r["budget"] for r in comparable}):
        subset = [r for r in comparable if r["budget"] == budget]
        rel_count = sum(r["actual"] == "rel" for r in subset)
        rel_hat_count = sum(
            r["UTC_rel_hat_gap"] >= max(r["UTC_abs_gap"], r["UTC_rel_gap"]) for r in subset
        )
        print(
            f"budget={budget}: n={len(subset)}, rel_wins={rel_count}, "
            f"mean(abs_gap)={np.mean([r['UTC_abs_gap'] for r in subset]):.3f}, "
            f"mean(rel_gap)={np.mean([r['UTC_rel_gap'] for r in subset]):.3f}, "
            f"mean(rel_hat_gap)={np.mean([r['UTC_rel_hat_gap'] for r in subset]):.3f}, "
            f"rel_hat_best={rel_hat_count}/{len(subset)}"
        )

    if rel_wins:
        print(
            f"rel-win mu_S CV mean={np.mean([r['mu_s_norm_cv'] for r in rel_wins]):.3f}, "
            f"budget frac mean={np.mean([r['budget_frac'] for r in rel_wins]):.3f}"
        )
    if abs_wins:
        print(
            f"abs-win mu_S CV mean={np.mean([r['mu_s_norm_cv'] for r in abs_wins]):.3f}, "
            f"budget frac mean={np.mean([r['budget_frac'] for r in abs_wins]):.3f}"
        )

    print("\n[largest rel-over-abs wins]")
    for r in sorted(comparable, key=lambda x: -x["delta_rel_abs"])[:8]:
        print(
            f"doc={r['doc_idx']} L{r['layer']}h{r['head']} b={r['budget']} "
            f"delta={r['delta_rel_abs']:.3f} CV={r['mu_s_norm_cv']:.3f} "
            f"H={r['H_mean']:.3f} abs={r['UTC_abs_gap']:.3f} rel={r['UTC_rel_gap']:.3f}"
        )

    print("\n[largest abs-over-rel wins]")
    for r in sorted(comparable, key=lambda x: x["delta_rel_abs"])[:8]:
        print(
            f"doc={r['doc_idx']} L{r['layer']}h{r['head']} b={r['budget']} "
            f"delta={r['delta_rel_abs']:.3f} CV={r['mu_s_norm_cv']:.3f} "
            f"H={r['H_mean']:.3f} abs={r['UTC_abs_gap']:.3f} rel={r['UTC_rel_gap']:.3f}"
        )

    print("\n[largest rel-hat improvements over best(abs, rel)]")
    for r in sorted(comparable, key=lambda x: -x["delta_rel_hat_best"])[:8]:
        print(
            f"doc={r['doc_idx']} L{r['layer']}h{r['head']} b={r['budget']} "
            f"delta={r['delta_rel_hat_best']:.3f} CV={r['mu_s_norm_cv']:.3f} "
            f"H={r['H_mean']:.3f} abs={r['UTC_abs_gap']:.3f} "
            f"rel={r['UTC_rel_gap']:.3f} rel_hat={r['UTC_rel_hat_gap']:.3f}"
        )

    print("\n[largest rel-hat losses vs best(abs, rel)]")
    for r in sorted(comparable, key=lambda x: x["delta_rel_hat_best"])[:8]:
        print(
            f"doc={r['doc_idx']} L{r['layer']}h{r['head']} b={r['budget']} "
            f"delta={r['delta_rel_hat_best']:.3f} CV={r['mu_s_norm_cv']:.3f} "
            f"H={r['H_mean']:.3f} abs={r['UTC_abs_gap']:.3f} "
            f"rel={r['UTC_rel_gap']:.3f} rel_hat={r['UTC_rel_hat_gap']:.3f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 5, 11])
    parser.add_argument("--budgets", type=int, nargs="+", default=[16, 32, 48])
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--cv-threshold", type=float, default=0.25)
    parser.add_argument("--tight-frac", type=float, default=0.20)
    args = parser.parse_args()

    stage0 = load("stage0_sanity_check")
    stage1 = load("stage1_evaluate_value_proxies")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    docs = select_documents(args.docs, skip=1)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased", attn_implementation="eager")
    model.eval()

    rows = []
    for doc_num, (doc_idx, category, text) in enumerate(docs):
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=args.max_tokens
        )
        n_tokens = int(inputs["input_ids"].shape[1])
        extracted = extract_layers(model, inputs, args.layers)
        print(f"\n[doc {doc_num}] idx={doc_idx}, category={category}, tokens={n_tokens}")

        for layer in args.layers:
            P_all, V_all = extracted[layer]
            for head in range(P_all.shape[0]):
                for budget in args.budgets:
                    if budget >= n_tokens:
                        continue
                    metrics = analyze_head(
                        stage0,
                        stage1,
                        P_all[head],
                        V_all[head],
                        budget,
                        args.iters,
                        args.cv_threshold,
                        args.tight_frac,
                    )
                    row = {
                        "doc_num": doc_num,
                        "doc_idx": doc_idx,
                        "category": category,
                        "layer": layer,
                        "head": head,
                        "tokens": n_tokens,
                        "budget": budget,
                        "budget_frac": budget / n_tokens,
                        **metrics,
                    }
                    rows.append(row)

    out_path = OUT_DIR / "stage2_abs_rel_sweep.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summarize(rows)
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
