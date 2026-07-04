"""Stage 2 step 0: extract per-head attention rows and values from real BERT.

Three jobs, in order of importance:
1. Extraction sanity anchor: recompute per-head P = softmax(QK^T/sqrt(d)) from
   layer inputs and the layer's own weight matrices, and verify it matches the
   model's reported attention probabilities. If this doesn't hold, nothing
   downstream is trustworthy.
2. Regime census: where do real BERT attention rows live on the H_norm axis?
   The synthetic work showed the value of value-awareness concentrates in the
   high-entropy regime; this asks whether that regime is actually populated.
3. Save per-head (P, V) arrays for selected layers so the stage-1 machinery
   (mass / UTC / hybrid-b / oracle at matched budget) can run on real data.

Text: one 20 Newsgroups document (same corpus as the BERT probing project),
tokenized to <= 256 tokens, no padding (so no attention-mask complications).
[CLS]/[SEP] rows are kept — they are part of real attention behavior.
"""

import importlib.util
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import fetch_20newsgroups
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "bert"
SAVE_LAYERS = [0, 5, 11]
MAX_TOKENS = 256


def load_stage0():
    path = ROOT / "scripts" / "stage0_sanity_check.py"
    spec = importlib.util.spec_from_file_location("stage0_sanity_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pick_document(min_chars=2000):
    bunch = fetch_20newsgroups(
        subset="train", remove=("headers", "footers", "quotes")
    )
    for idx, text in enumerate(bunch.data):
        if len(text) >= min_chars:
            return idx, bunch.target_names[bunch.target[idx]], text
    raise RuntimeError("no document long enough")


def main():
    stage0 = load_stage0()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    doc_idx, category, text = pick_document()
    print(f"document: 20newsgroups train idx={doc_idx}, category={category}")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained(
        "bert-base-uncased", attn_implementation="eager"
    )
    model.eval()

    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=MAX_TOKENS
    )
    n_tokens = inputs["input_ids"].shape[1]
    print(f"tokens: {n_tokens}")

    with torch.no_grad():
        out = model(
            **inputs, output_attentions=True, output_hidden_states=True
        )

    attentions = [a[0].double().numpy() for a in out.attentions]  # (H, N, N)
    n_layers = len(attentions)
    n_heads, N, _ = attentions[0].shape
    head_dim = model.config.hidden_size // n_heads

    # --- job 2: regime census over ALL layers/heads ---
    print("\n[regime census: H_norm of real BERT attention rows]")
    print("layer | min    | mean   | max    | frac>=0.90 | frac>=0.94 | frac<=0.50")
    print("-" * 78)
    census = np.zeros((n_layers, n_heads, N))
    for layer in range(n_layers):
        P = attentions[layer]
        H = -(P * np.log(P + 1e-12)).sum(axis=-1)  # (H, N)
        H_norm = H / np.log(N)
        census[layer] = H_norm
        flat = H_norm.ravel()
        print(
            f"{layer:>5} | {flat.min():.4f} | {flat.mean():.4f} | {flat.max():.4f} | "
            f"{(flat >= 0.90).mean():>10.3f} | {(flat >= 0.94).mean():>10.3f} | "
            f"{(flat <= 0.50).mean():>10.3f}"
        )
    flat_all = census.ravel()
    print(
        f"\nall layers pooled: mean={flat_all.mean():.4f}, "
        f"frac>=0.90={(flat_all >= 0.90).mean():.3f}, "
        f"frac>=0.94={(flat_all >= 0.94).mean():.3f}, "
        f"frac<=0.50={(flat_all <= 0.50).mean():.3f}"
    )
    np.save(OUT_DIR / "hnorm_census.npy", census)

    # --- jobs 1 + 3: recompute Q,K,V per head for selected layers ---
    print("\n[extraction anchor + save: recomputed P vs model attentions]")
    print("layer | max |P_recomputed - P_model| | identity max abs diff (head 0, k=40)")
    print("-" * 84)
    for layer in SAVE_LAYERS:
        h_in = out.hidden_states[layer][0].double()  # (N, hidden) input to layer
        sa = model.encoder.layer[layer].attention.self

        def split_heads(x):
            return x.reshape(N, n_heads, head_dim).permute(1, 0, 2)  # (H, N, d)

        with torch.no_grad():
            Q = split_heads(sa.query(h_in.float()).double())
            K = split_heads(sa.key(h_in.float()).double())
            V = split_heads(sa.value(h_in.float()).double())

        logits = Q @ K.transpose(1, 2) / np.sqrt(head_dim)
        P_re = torch.softmax(logits, dim=-1).numpy()
        P_model = attentions[layer]
        anchor_diff = np.abs(P_re - P_model).max()

        V_np = V.numpy()
        stats = stage0.check_all_rows_decomposition(P_re[0], V_np[0], k=40)
        identity_diff = stats["abs_diffs"].max()

        np.savez_compressed(
            OUT_DIR / f"qkv_layer{layer}.npz",
            P=P_re,
            V=V_np,
            layer=layer,
            doc_idx=doc_idx,
            category=category,
        )
        print(f"{layer:>5} | {anchor_diff:>26.3e} | {identity_diff:>28.3e}")

    print(f"\nsaved: {OUT_DIR}\\qkv_layer{{{','.join(map(str, SAVE_LAYERS))}}}.npz")


if __name__ == "__main__":
    main()
