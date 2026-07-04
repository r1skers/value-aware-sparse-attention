# Stage 3 Summary: GPT-2 Cross-Model Validation

Status: **complete 2026-07-03**. Details are in
[`docs/experiment_log.md`](../docs/experiment_log.md) under the GPT-2
cross-model sections.

## Question

Stage 2 established rel-hat as the leading local scorer candidate on BERT.
Stage 3 asks:

```text
Is rel-hat a BERT-specific fix, or does the scorer ladder transfer to a
different architecture and attention pattern?
```

Answer: **it transfers to GPT-2 small causal attention without changing the
method definitions.**

## Protocol

Model: `gpt2` / GPT-2 small.

Dataset: 3 held-out 20 Newsgroups documents, skipped past the BERT tuning docs.

Sweep:

```text
docs: 3
layers: 0, 5, 11
heads: 12
budgets: 8, 16, 32
rows: 324
```

Causal adaptations:

- each query row only attends to its causal prefix;
- per-row support is capped by prefix length;
- analysis keeps rows with support >= 65;
- UTC uses prefix value sums instead of full sequence sums.

This changes the support geometry, not the scoring definitions.

## Extraction Sanity Check

For each selected layer, Q,K,V are recomputed from GPT-2's `c_attn` projection
and checked against model-returned attention probabilities:

```text
max |P_recomputed - P_model| ~= 1e-6
```

## Exact-Budget Result

Gap closed vs fixed-k:

```text
method       mean    median   p10     below fixed
mass         0.017   0.265   -1.186    130
UTC-abs      0.309   0.602   -0.842     77
UTC-rel      0.579   0.837    0.185     27
UTC-rel-hat  0.828   0.881    0.642      3
```

Pairwise checks:

```text
rel-hat >= max(abs, rel): 250 / 324 = 0.772
rel-hat below fixed:        3 / 324
catastrophic (< -1):        0 / 324
```

Interpretation: the method ladder is not a BERT-only phenomenon. The same
cheap value-aware relative scorer remains strong in causal GPT-2 attention.

## Why This Matters

Stage 2 could have been explained away as a BERT-specific sink/punctuation
artifact. Stage 3 makes that explanation much weaker:

- model family changes from encoder BERT to causal decoder GPT-2;
- attention support changes from full sequence to causal prefix;
- UTC remains deployable-cheap because prefix value sums are natural in causal
  kernels;
- rel-hat remains the leading local scorer candidate.

## Caveats

- The sweep is still small: 3 documents and layers 0/5/11.
- This is local attention-output error, not next-token KL or perplexity.
- The oracle is still restricted to top-k-by-probability retained sets.

## Transition to Stage 4

After Stage 3, the local scorer story is strong enough to ask a harder metric
question:

```text
Does local sparse-attention error remain meaningful after W_O and downstream
model computation?
```

Stage 4 answers this with W_O projection, single-head KL, and whole-layer KL.
