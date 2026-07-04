"""Stage 2C: exact-budget BERT allocation protocol.

Stage 2B used threshold calibration: choose the smallest k per row whose score
falls below a threshold, then compare only rows whose mean k happens to match
the requested budget. Real BERT heads often saturate under that protocol, so
most rows are not matched-budget comparable.

This script changes the protocol. For each method, build a score curve
score(row, k), choose a min-max threshold under the requested total budget, and
then force any leftover budget to be spent using that method's own marginal
score improvements. No method may spend fewer tokens than fixed-k.
"""

import argparse
import csv
import importlib.util
from pathlib import Path

import numpy as np
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "bert"
DEFAULT_OUT = OUT_DIR / "stage2c_bert_exact_budget.csv"


def load(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row_key(row):
    return (
        int(row["doc_idx"]),
        int(row["layer"]),
        int(row["head"]),
        int(row["budget"]),
    )


def read_completed(path):
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as f:
        return {row_key(row) for row in csv.DictReader(f)}


def append_row(path, fieldnames, row):
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def score_curves(P, V, eta=1e-12):
    """Return method score curves with shape (query_rows, k=1..n)."""
    n_rows, n = P.shape
    total_v = V.sum(axis=0)
    curves = {
        "oracle": np.zeros((n_rows, n), dtype=np.float64),
        "mass": np.zeros((n_rows, n), dtype=np.float64),
        "UTC_abs": np.zeros((n_rows, n), dtype=np.float64),
        "UTC_rel": np.zeros((n_rows, n), dtype=np.float64),
        "UTC_rel_hat": np.zeros((n_rows, n), dtype=np.float64),
    }

    for row, p in enumerate(P):
        order = np.argsort(-p)
        p_sorted = p[order]
        v_sorted = V[order]

        retained_mass = np.cumsum(p_sorted)
        retained_weighted = np.cumsum(p_sorted[:, None] * v_sorted, axis=0)
        retained_unweighted = np.cumsum(v_sorted, axis=0)
        full_o = p @ V
        full_norm = np.linalg.norm(full_o)

        for k_idx in range(n):
            k = k_idx + 1
            mass = retained_mass[k_idx]
            dropped = max(0.0, 1.0 - mass)
            mu_s = retained_weighted[k_idx] / mass

            curves["mass"][row, k_idx] = dropped
            if k == n or dropped <= 1e-15:
                continue

            mu_r = (full_o - retained_weighted[k_idx]) / dropped
            true_abs = dropped * np.linalg.norm(mu_r - mu_s)
            curves["oracle"][row, k_idx] = true_abs / (full_norm + eta)

            mu_r_hat = (total_v - retained_unweighted[k_idx]) / (n - k)
            proxy_abs = dropped * np.linalg.norm(mu_r_hat - mu_s)
            o_hat = mass * mu_s + dropped * mu_r_hat

            curves["UTC_abs"][row, k_idx] = proxy_abs
            curves["UTC_rel"][row, k_idx] = proxy_abs / (np.linalg.norm(mu_s) + eta)
            curves["UTC_rel_hat"][row, k_idx] = proxy_abs / (np.linalg.norm(o_hat) + eta)

    return curves


def first_k_under_threshold(curve, threshold):
    ok = curve <= threshold
    # Every row has score 0 at k=n, so argmax is safe after forcing the last col.
    ok[:, -1] = True
    return ok.argmax(axis=1) + 1


def threshold_exact_budget_allocation(curve, target_mean_k):
    """Allocate exactly n_rows * target_mean_k tokens using one method's score.

    The first phase finds the smallest risk threshold whose minimal k choices
    fit within the budget. The second phase spends any leftover budget using the
    same score curve's one-step marginal improvements.
    """
    n_rows, n = curve.shape
    total_budget = int(round(n_rows * target_mean_k))
    if not n_rows <= total_budget <= n_rows * n:
        raise ValueError("target budget outside feasible [1, n] range")

    values = np.unique(curve)
    lo, hi = 0, len(values) - 1
    best_ks = np.full(n_rows, n, dtype=int)
    best_threshold = values[-1]

    while lo <= hi:
        mid = (lo + hi) // 2
        threshold = values[mid]
        ks = first_k_under_threshold(curve, threshold)
        if ks.sum() <= total_budget:
            best_ks = ks
            best_threshold = threshold
            hi = mid - 1
        else:
            lo = mid + 1

    ks = best_ks.copy()
    leftover = total_budget - int(ks.sum())

    while leftover > 0:
        candidates = np.flatnonzero(ks < n)
        if candidates.size == 0:
            break
        current = curve[candidates, ks[candidates] - 1]
        nxt = curve[candidates, ks[candidates]]
        gains = current - nxt
        chosen = candidates[np.argmax(gains)]
        ks[chosen] += 1
        leftover -= 1

    return best_threshold, ks


def max_rel(stage0, P, V, ks):
    stats = stage0.check_all_rows_decomposition(P, V, k=ks)
    return stage0.relative_errors(P, V, stats).max()


def mean_rel(stage0, P, V, ks):
    stats = stage0.check_all_rows_decomposition(P, V, k=ks)
    return stage0.relative_errors(P, V, stats).mean()


def gap_closed(fixed_max, oracle_max, method_max):
    gap = fixed_max - oracle_max
    if gap <= 1e-15:
        return np.nan
    return (fixed_max - method_max) / gap


def analyze_head_exact(stage0, P, V, budget):
    n_rows = P.shape[0]
    fixed_ks = np.full(n_rows, budget, dtype=int)
    curves = score_curves(P, V)

    fixed_max = max_rel(stage0, P, V, fixed_ks)
    fixed_mean = mean_rel(stage0, P, V, fixed_ks)

    row = {
        "fixed_mean_k": float(fixed_ks.mean()),
        "fixed_max_rel": fixed_max,
        "fixed_mean_rel": fixed_mean,
    }

    allocations = {}
    for name, curve in curves.items():
        threshold, ks = threshold_exact_budget_allocation(curve, budget)
        allocations[name] = ks
        method_max = max_rel(stage0, P, V, ks)
        method_mean = mean_rel(stage0, P, V, ks)
        row[f"{name}_threshold"] = threshold
        row[f"{name}_mean_k"] = float(ks.mean())
        row[f"{name}_min_k"] = int(ks.min())
        row[f"{name}_max_k"] = int(ks.max())
        row[f"{name}_max_rel"] = method_max
        row[f"{name}_mean_rel"] = method_mean

    oracle_max = row["oracle_max_rel"]
    for name in ["mass", "UTC_abs", "UTC_rel", "UTC_rel_hat"]:
        row[f"{name}_gap"] = gap_closed(fixed_max, oracle_max, row[f"{name}_max_rel"])

    row["rel_hat_beats_abs_rel"] = (
        row["UTC_rel_hat_max_rel"] <= min(row["UTC_abs_max_rel"], row["UTC_rel_max_rel"])
    )
    row["rel_hat_below_fixed"] = row["UTC_rel_hat_max_rel"] > fixed_max
    row["rel_hat_beats_mass"] = row["UTC_rel_hat_max_rel"] <= row["mass_max_rel"]
    row["oracle_below_fixed"] = oracle_max <= fixed_max
    return row


def numeric_rows(path):
    rows = []
    if not path.exists():
        return rows
    int_fields = {"doc_num", "doc_idx", "layer", "head", "tokens", "budget"}
    bool_fields = {
        "rel_hat_beats_abs_rel",
        "rel_hat_below_fixed",
        "rel_hat_beats_mass",
        "oracle_below_fixed",
    }
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            parsed = {}
            for key, value in row.items():
                if key in int_fields:
                    parsed[key] = int(value)
                elif key in bool_fields:
                    parsed[key] = value == "True"
                else:
                    try:
                        parsed[key] = float(value)
                    except ValueError:
                        parsed[key] = value
            rows.append(parsed)
    return rows


def summarize(path):
    rows = numeric_rows(path)
    if not rows:
        print("[summary] no rows yet")
        return

    methods = ["mass", "UTC_abs", "UTC_rel", "UTC_rel_hat"]
    print("\n[Stage 2C exact-budget summary]")
    print(f"rows total={len(rows)}")
    print(
        "rel-hat <= min(abs, rel): "
        f"{sum(r['rel_hat_beats_abs_rel'] for r in rows)}/{len(rows)} "
        f"({np.mean([r['rel_hat_beats_abs_rel'] for r in rows]):.3f})"
    )
    print(
        "rel-hat <= mass: "
        f"{sum(r['rel_hat_beats_mass'] for r in rows)}/{len(rows)} "
        f"({np.mean([r['rel_hat_beats_mass'] for r in rows]):.3f})"
    )
    print(
        "rel-hat worse than fixed: "
        f"{sum(r['rel_hat_below_fixed'] for r in rows)}/{len(rows)}"
    )

    print("\nmethod       mean gap  median gap  p10 gap   mean max_rel")
    for name in methods:
        gaps = np.array([r[f"{name}_gap"] for r in rows], dtype=np.float64)
        max_rels = np.array([r[f"{name}_max_rel"] for r in rows], dtype=np.float64)
        valid = gaps[~np.isnan(gaps)]
        print(
            f"{name:<12} "
            f"{np.mean(valid):>8.3f}  {np.median(valid):>10.3f}  "
            f"{np.quantile(valid, 0.1):>7.3f}  {np.mean(max_rels):>12.4f}"
        )

    print("\nby budget")
    for budget in sorted({r["budget"] for r in rows}):
        subset = [r for r in rows if r["budget"] == budget]
        print(
            f"k={budget}: n={len(subset)}, "
            f"mass_gap={np.nanmean([r['mass_gap'] for r in subset]):.3f}, "
            f"abs_gap={np.nanmean([r['UTC_abs_gap'] for r in subset]):.3f}, "
            f"rel_gap={np.nanmean([r['UTC_rel_gap'] for r in subset]):.3f}, "
            f"rel_hat_gap={np.nanmean([r['UTC_rel_hat_gap'] for r in subset]):.3f}, "
            f"rel_hat_below_fixed={sum(r['rel_hat_below_fixed'] for r in subset)}"
        )

    print("\nby layer")
    for layer in sorted({r["layer"] for r in rows}):
        subset = [r for r in rows if r["layer"] == layer]
        print(
            f"L{layer:02d}: n={len(subset)}, "
            f"rel_hat_gap={np.nanmean([r['UTC_rel_hat_gap'] for r in subset]):.3f}, "
            f"rel_hat<=abs/rel={np.mean([r['rel_hat_beats_abs_rel'] for r in subset]):.3f}, "
            f"rel_hat<fixed={sum(r['rel_hat_below_fixed'] for r in subset)}"
        )

    print("\nworst rel-hat rows")
    for r in sorted(rows, key=lambda x: x["UTC_rel_hat_gap"])[:8]:
        print(
            f"doc={r['doc_idx']} L{r['layer']}h{r['head']} k={r['budget']} "
            f"rel_hat_gap={r['UTC_rel_hat_gap']:.3f} "
            f"fixed={r['fixed_max_rel']:.4f} oracle={r['oracle_max_rel']:.4f} "
            f"rel_hat={r['UTC_rel_hat_max_rel']:.4f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--layers", type=int, nargs="+", default=list(range(12)))
    parser.add_argument("--budgets", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--fresh", action="store_true", help="delete existing output before running")
    args = parser.parse_args()

    stage0 = load("stage0_sanity_check")
    sweep = load("stage2_bert_abs_rel_sweep")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.fresh and args.out.exists():
        args.out.unlink()

    completed = read_completed(args.out)
    docs = sweep.select_documents(args.docs, skip=1)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased", attn_implementation="eager")
    model.eval()

    fieldnames = None
    if args.out.exists():
        with args.out.open(newline="", encoding="utf-8") as f:
            fieldnames = csv.DictReader(f).fieldnames

    total_expected = len(docs) * len(args.layers) * model.config.num_attention_heads * len(args.budgets)
    print(
        f"[Stage 2C] docs={len(docs)}, layers={args.layers}, budgets={args.budgets}, "
        f"expected rows={total_expected}, already done={len(completed)}"
    )

    for doc_num, (doc_idx, category, text) in enumerate(docs):
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=args.max_tokens
        )
        n_tokens = int(inputs["input_ids"].shape[1])
        extracted = sweep.extract_layers(model, inputs, args.layers)
        print(f"\n[doc {doc_num + 1}/{len(docs)}] idx={doc_idx}, category={category}, tokens={n_tokens}")

        for layer in args.layers:
            P_all, V_all = extracted[layer]
            for head in range(P_all.shape[0]):
                for budget in args.budgets:
                    if budget >= n_tokens:
                        continue
                    key = (doc_idx, layer, head, budget)
                    if key in completed:
                        continue
                    metrics = analyze_head_exact(stage0, P_all[head], V_all[head], budget)
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
                    if fieldnames is None:
                        fieldnames = list(row.keys())
                    append_row(args.out, fieldnames, row)
                    completed.add(key)
                print(
                    f"  layer={layer:02d} head={head:02d} "
                    f"done={len(completed)}/{total_expected}"
                )

    summarize(args.out)
    print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
