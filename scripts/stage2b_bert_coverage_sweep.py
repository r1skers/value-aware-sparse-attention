"""Stage 2B: resumable larger BERT coverage sweep.

This expands the held-out BERT test from a few layers to broader coverage:
multiple documents, all BERT-base layers/heads, and several budget levels.
Rows are written incrementally so the sweep can be resumed after interruption.
"""

import argparse
import csv
import importlib.util
from pathlib import Path

import numpy as np
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "bert"
DEFAULT_OUT = OUT_DIR / "stage2b_bert_coverage.csv"


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


def numeric_rows(path):
    rows = []
    if not path.exists():
        return rows
    int_fields = {"doc_num", "doc_idx", "layer", "head", "tokens", "budget"}
    bool_fields = {"correct", "budget_ok"}
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
    comparable = [r for r in rows if r["budget_ok"]]
    if not rows:
        print("[summary] no rows yet")
        return

    print("\n[Stage 2B running summary]")
    print(f"rows total={len(rows)}, comparable={len(comparable)}")
    if not comparable:
        return

    rel_hat_best = [
        r
        for r in comparable
        if r["UTC_rel_hat_gap"] >= max(r["UTC_abs_gap"], r["UTC_rel_gap"])
    ]
    disasters = [r for r in comparable if r["UTC_rel_hat_gap"] < -1.0]
    print(
        f"rel-hat >= max(abs, rel): {len(rel_hat_best)}/{len(comparable)} "
        f"({len(rel_hat_best) / len(comparable):.3f})"
    )
    print(f"rel-hat catastrophic failures (< -1): {len(disasters)}")

    for budget in sorted({r["budget"] for r in comparable}):
        subset = [r for r in comparable if r["budget"] == budget]
        print(
            f"budget={budget}: n={len(subset)}, "
            f"mean mass={np.mean([r['mass_gap'] for r in subset]):.3f}, "
            f"abs={np.mean([r['UTC_abs_gap'] for r in subset]):.3f}, "
            f"rel={np.mean([r['UTC_rel_gap'] for r in subset]):.3f}, "
            f"rel-hat={np.mean([r['UTC_rel_hat_gap'] for r in subset]):.3f}"
        )

    for layer in sorted({r["layer"] for r in comparable}):
        subset = [r for r in comparable if r["layer"] == layer]
        if len(subset) < 3:
            continue
        print(
            f"layer={layer:02d}: n={len(subset)}, "
            f"rel-hat={np.mean([r['UTC_rel_hat_gap'] for r in subset]):.3f}, "
            f"best-rate={sum(r['UTC_rel_hat_gap'] >= max(r['UTC_abs_gap'], r['UTC_rel_gap']) for r in subset) / len(subset):.3f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--layers", type=int, nargs="+", default=list(range(12)))
    parser.add_argument("--budgets", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--iters", type=int, default=6)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--fresh", action="store_true", help="delete existing output before running")
    args = parser.parse_args()

    stage0 = load("stage0_sanity_check")
    stage1 = load("stage1_evaluate_value_proxies")
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
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames

    total_expected = len(docs) * len(args.layers) * model.config.num_attention_heads * len(args.budgets)
    print(
        f"[Stage 2B] docs={len(docs)}, layers={args.layers}, budgets={args.budgets}, "
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

                    metrics = sweep.analyze_head(
                        stage0,
                        stage1,
                        P_all[head],
                        V_all[head],
                        budget,
                        args.iters,
                        cv_threshold=0.25,
                        tight_frac=0.20,
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
