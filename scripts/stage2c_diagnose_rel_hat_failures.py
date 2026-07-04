"""Diagnose the Stage 2C UTC-rel-hat below-fixed cases.

Stage 2C removed the budget-comparability issue by forcing every method to
spend the exact same head-local token budget. The remaining below-fixed rows
are therefore real method failures. This script drills into those failures at
the query-row level.
"""

import argparse
import csv
import importlib.util
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "bert"
DEFAULT_IN = OUT_DIR / "stage2c_bert_exact_budget.csv"
DEFAULT_OUT = OUT_DIR / "stage2c_rel_hat_failure_diagnosis.csv"


def load(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_failure_rows(path):
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["rel_hat_below_fixed"] != "True":
                continue
            parsed = {}
            for key, value in row.items():
                try:
                    if key in {"doc_num", "doc_idx", "layer", "head", "tokens", "budget"}:
                        parsed[key] = int(value)
                    elif value in {"True", "False"}:
                        parsed[key] = value == "True"
                    else:
                        parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value
            rows.append(parsed)
    return rows


def rel_errors(stage0, P, V, ks):
    stats = stage0.check_all_rows_decomposition(P, V, k=ks)
    return stage0.relative_errors(P, V, stats), stats


def row_proxy_details(p, V, k, eta=1e-12):
    n = p.shape[0]
    order = np.argsort(-p)
    top = order[:k]
    tail = order[k:]
    full_o = p @ V
    full_norm = np.linalg.norm(full_o)
    retained_mass = p[top].sum()
    dropped_mass = 1.0 - retained_mass
    mu_s = (p[top, None] * V[top]).sum(axis=0) / retained_mass

    if k >= n or dropped_mass <= 1e-15:
        return {
            "full_norm": full_norm,
            "mu_s_norm": np.linalg.norm(mu_s),
            "o_hat_norm": full_norm,
            "true_abs": 0.0,
            "proxy_abs": 0.0,
            "true_rel": 0.0,
            "proxy_rel_hat": 0.0,
            "numerator_ratio": np.nan,
            "denom_ratio": 1.0,
            "score_ratio": np.nan,
            "dropped_mass": 0.0,
            "true_centroid": 0.0,
            "utc_centroid": 0.0,
        }

    mu_r = (p[tail, None] * V[tail]).sum(axis=0) / dropped_mass
    true_centroid = np.linalg.norm(mu_r - mu_s)
    true_abs = dropped_mass * true_centroid

    total_v = V.sum(axis=0)
    retained_unweighted = V[top].sum(axis=0)
    mu_r_hat = (total_v - retained_unweighted) / (n - k)
    utc_centroid = np.linalg.norm(mu_r_hat - mu_s)
    proxy_abs = dropped_mass * utc_centroid
    o_hat = retained_mass * mu_s + dropped_mass * mu_r_hat
    o_hat_norm = np.linalg.norm(o_hat)

    true_rel = true_abs / (full_norm + eta)
    proxy_rel_hat = proxy_abs / (o_hat_norm + eta)

    return {
        "full_norm": full_norm,
        "mu_s_norm": np.linalg.norm(mu_s),
        "o_hat_norm": o_hat_norm,
        "true_abs": true_abs,
        "proxy_abs": proxy_abs,
        "true_rel": true_rel,
        "proxy_rel_hat": proxy_rel_hat,
        "numerator_ratio": proxy_abs / (true_abs + eta),
        "denom_ratio": o_hat_norm / (full_norm + eta),
        "score_ratio": proxy_rel_hat / (true_rel + eta),
        "dropped_mass": dropped_mass,
        "true_centroid": true_centroid,
        "utc_centroid": utc_centroid,
    }


def classify(details, k_rel_hat, k_oracle):
    if k_rel_hat < k_oracle:
        denom_high = details["denom_ratio"] > 1.25
        numerator_low = details["numerator_ratio"] < 0.60
        score_low = details["score_ratio"] < 0.60
        if denom_high and numerator_low:
            return "mixed_under_num_over_denom_starvation"
        if denom_high:
            return "denom_overestimate_starvation"
        if numerator_low:
            return "numerator_underestimate_starvation"
        if score_low:
            return "score_underestimate_starvation"
        return "mild_score_miscalibration_starvation"
    if k_rel_hat > k_oracle:
        if details["denom_ratio"] < 0.5:
            return "overallocated_denominator_collapse"
        return "overallocated_budget_diversion"
    if details["score_ratio"] < 0.5:
        return "same_k_underestimated_risk"
    return "same_k_other"


def write_rows(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    print("\n[Stage 2C rel-hat failure diagnosis]")
    print(f"failure head-budget rows: {len(rows)}")
    print("by budget:", dict(sorted(Counter(r["budget"] for r in rows).items())))
    print("by layer:", dict(sorted(Counter(r["layer"] for r in rows).items())))
    print("classification:", dict(Counter(r["failure_class"] for r in rows)))

    print(
        "worst-row k relation: "
        f"rel_hat<oracle={sum(r['k_rel_hat'] < r['k_oracle'] for r in rows)}, "
        f"rel_hat=oracle={sum(r['k_rel_hat'] == r['k_oracle'] for r in rows)}, "
        f"rel_hat>oracle={sum(r['k_rel_hat'] > r['k_oracle'] for r in rows)}"
    )
    print(
        "median ratios on worst rows: "
        f"score_proxy/true={np.median([r['score_ratio'] for r in rows]):.3f}, "
        f"proxy_abs/true_abs={np.median([r['numerator_ratio'] for r in rows]):.3f}, "
        f"||o_hat||/||o||={np.median([r['denom_ratio'] for r in rows]):.3f}"
    )
    print(
        "abs positive among failures: "
        f"{sum(r['UTC_abs_gap'] > 0 for r in rows)}/{len(rows)}"
    )

    print("\nworst failures")
    for r in sorted(rows, key=lambda x: x["UTC_rel_hat_gap"])[:10]:
        print(
            f"doc={r['doc_idx']} L{r['layer']}h{r['head']} k={r['budget']} "
            f"gap={r['UTC_rel_hat_gap']:.3f} q={r['worst_query']}({r['worst_token']}) "
            f"k_relhat/oracle/fixed={r['k_rel_hat']}/{r['k_oracle']}/{r['budget']} "
            f"true_rel={r['rel_hat_row_rel']:.3f} "
            f"proxy/true={r['score_ratio']:.3f} "
            f"num={r['numerator_ratio']:.3f} denom={r['denom_ratio']:.3f} "
            f"{r['failure_class']}"
        )

    grouped = defaultdict(list)
    for r in rows:
        grouped[r["failure_class"]].append(r)
    print("\nclass examples")
    for name, subset in sorted(grouped.items(), key=lambda item: -len(item[1])):
        ex = sorted(subset, key=lambda x: x["UTC_rel_hat_gap"])[0]
        print(
            f"{name}: n={len(subset)}, example doc={ex['doc_idx']} "
            f"L{ex['layer']}h{ex['head']} k={ex['budget']} "
            f"gap={ex['UTC_rel_hat_gap']:.3f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2c-csv", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    stage0 = load("stage0_sanity_check")
    sweep = load("stage2_bert_abs_rel_sweep")
    exact = load("stage2c_bert_exact_budget")

    failures = read_failure_rows(args.stage2c_csv)
    needed_by_doc = defaultdict(set)
    for row in failures:
        needed_by_doc[row["doc_idx"]].add(row["layer"])

    docs_by_idx = {doc_idx: (category, text) for doc_idx, category, text in sweep.select_documents(10, skip=1)}
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased", attn_implementation="eager")
    model.eval()

    diagnosis = []
    for doc_idx, layers in sorted(needed_by_doc.items()):
        category, text = docs_by_idx[doc_idx]
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        token_strings = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())
        extracted = sweep.extract_layers(model, inputs, sorted(layers))
        tokens = int(inputs["input_ids"].shape[1])
        print(f"[doc] idx={doc_idx}, category={category}, layers={sorted(layers)}, tokens={tokens}")

        for base in [r for r in failures if r["doc_idx"] == doc_idx]:
            P_all, V_all = extracted[base["layer"]]
            P = P_all[base["head"]]
            V = V_all[base["head"]]
            curves = exact.score_curves(P, V)
            allocations = {
                name: exact.threshold_exact_budget_allocation(curve, base["budget"])[1]
                for name, curve in curves.items()
            }

            fixed_ks = np.full(P.shape[0], base["budget"], dtype=int)
            rel_hat_rel, _ = rel_errors(stage0, P, V, allocations["UTC_rel_hat"])
            fixed_rel, _ = rel_errors(stage0, P, V, fixed_ks)
            oracle_rel, _ = rel_errors(stage0, P, V, allocations["oracle"])
            abs_rel, _ = rel_errors(stage0, P, V, allocations["UTC_abs"])
            rel_rel, _ = rel_errors(stage0, P, V, allocations["UTC_rel"])
            mass_rel, _ = rel_errors(stage0, P, V, allocations["mass"])

            worst = int(np.argmax(rel_hat_rel))
            rel_hat_vs_oracle = allocations["UTC_rel_hat"] - allocations["oracle"]
            k_rel_hat = int(allocations["UTC_rel_hat"][worst])
            k_oracle = int(allocations["oracle"][worst])
            details = row_proxy_details(P[worst], V, k_rel_hat)
            oracle_details_at_relhat_k = row_proxy_details(P[worst], V, k_oracle)

            out = {
                **base,
                "tokens": tokens,
                "worst_query": worst,
                "worst_token": token_strings[worst],
                "k_fixed": int(fixed_ks[worst]),
                "k_oracle": k_oracle,
                "k_mass": int(allocations["mass"][worst]),
                "k_abs": int(allocations["UTC_abs"][worst]),
                "k_rel": int(allocations["UTC_rel"][worst]),
                "k_rel_hat": k_rel_hat,
                "worst_k_deficit_vs_oracle": int(k_oracle - k_rel_hat),
                "rel_hat_rows_below_oracle": int((rel_hat_vs_oracle < 0).sum()),
                "rel_hat_rows_above_oracle": int((rel_hat_vs_oracle > 0).sum()),
                "rel_hat_missing_tokens_vs_oracle": int(np.maximum(-rel_hat_vs_oracle, 0).sum()),
                "rel_hat_extra_tokens_vs_oracle": int(np.maximum(rel_hat_vs_oracle, 0).sum()),
                "rel_hat_row_rel": float(rel_hat_rel[worst]),
                "fixed_row_rel": float(fixed_rel[worst]),
                "oracle_row_rel": float(oracle_rel[worst]),
                "mass_row_rel": float(mass_rel[worst]),
                "abs_row_rel": float(abs_rel[worst]),
                "rel_row_rel": float(rel_rel[worst]),
                "full_norm": float(details["full_norm"]),
                "mu_s_norm": float(details["mu_s_norm"]),
                "o_hat_norm": float(details["o_hat_norm"]),
                "true_abs": float(details["true_abs"]),
                "proxy_abs": float(details["proxy_abs"]),
                "true_rel_at_rel_hat_k": float(details["true_rel"]),
                "proxy_rel_hat_at_rel_hat_k": float(details["proxy_rel_hat"]),
                "score_ratio": float(details["score_ratio"]),
                "numerator_ratio": float(details["numerator_ratio"]),
                "denom_ratio": float(details["denom_ratio"]),
                "dropped_mass": float(details["dropped_mass"]),
                "true_centroid": float(details["true_centroid"]),
                "utc_centroid": float(details["utc_centroid"]),
                "oracle_true_rel_at_oracle_k": float(oracle_details_at_relhat_k["true_rel"]),
                "failure_class": classify(details, k_rel_hat, k_oracle),
            }
            diagnosis.append(out)

    write_rows(args.out, diagnosis)
    summarize(diagnosis)
    print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
