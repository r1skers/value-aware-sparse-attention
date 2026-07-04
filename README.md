# Value-Aware Sparse Attention

Research-style implementation project for measuring sparse-attention pruning
error, designing cheap value-aware proxies, and mapping where local attention
error stops being a model-behavior oracle.

This is **not** a deployable sparse-attention kernel or a wall-clock speedup
claim. It is an error-analysis artifact: a sequence of reproducible PyTorch
experiments, real BERT/GPT-2 attention probes, and metric-boundary tests.

## Core Identity

For one attention row, let `S` be the retained set, `R` the dropped set,
`delta = sum_{i in R} p_i`, and `mu_S`, `mu_R` the retained/dropped value
centroids after attention-probability weighting. Renormalized top-k pruning
has exact output error:

```math
\|o - \tilde{o}\| = \delta \|\mu_R - \mu_S\|.
```

This identity separates two factors:

- dropped probability mass `delta`, available from Q,K/P only;
- value-centroid displacement `||mu_R - mu_S||`, which depends on V.

The project asks:

```text
When is Q,K-only pruning enough?
When does value geometry matter?
Can a cheap value-aware proxy recover oracle-like allocation gains?
Where does local sparse-attention error stop predicting model behavior?
```

## Current Status

Stage 4 is closed. The project now has four layers of evidence:

1. **Formula and regime map**: the decomposition is verified to float precision.
   Dropped mass dominates sharp attention; value geometry dominates diffuse /
   high-entropy attention.
2. **Cheap value proxy**: UTC / UTC-rel-hat use sequence-level value summaries
   and retained values, avoiding per-query scans over dropped V. On local
   attention-output error, rel-hat is the leading deployable scorer candidate.
3. **Real-model validation**: BERT and GPT-2 attention maps preserve the local
   value-aware advantage under exact-budget evaluation; failures are diagnosable
   as tight-budget max-risk row starvation.
4. **Metric boundary**: the advantage survives W_O projection with attenuation,
   but next-token KL breaks the local-oracle reference. Local restricted
   oracles are not behavioral oracles.

The short version:

```text
Value-aware local sparse-attention error can be controlled cheaply and transfers
through W_O with attenuation, but next-token KL requires a different behavioral
reference axis.
```

## Key Results

### Stage 0 / v1: formula and signals

- Verified `||o - sparse(o)|| = delta * ||mu_R - mu_S||` to float precision.
- Showed entropy is a regime indicator, not an error predictor.
- Found a regime shift across synthetic `q_scale`: value geometry dominates
  diffuse rows; dropped mass dominates sharp rows.
- Established the language discipline: the oracle is a **restricted oracle**
  within the top-k-by-probability family, not a global subset-selection oracle.

Report: [`reports/v1_summary.md`](reports/v1_summary.md)

### Stage 1: cheap value proxies

UTC approximates the dropped weighted centroid by the unweighted tail centroid:

```text
mu_R_hat = (sum_i V_i - sum_{i in S} V_i) / |R|.
```

It has zero extra per-row dropped-V IO: `sum_i V_i` is a sequence-level
precompute and retained values are already read by sparse attention.

Main findings:

- UTC works best in high-entropy regimes, where value geometry is most needed.
- Predictor correlation is not allocation quality.
- The first entropy-router experiment was circular and was demoted to a
  consistency check.
- On mixed-regime synthetic data, budget-delegated hybrid-b beats mass and UTC
  on 3/3 seeds:

```text
gap closed | seed 0 | seed 1 | seed 2
mass       |  0.833 |  0.751 |  0.791
UTC        |  0.804 |  0.761 |  0.772
hybrid-b   |  0.855 |  0.902 |  0.878
```

Report: [`reports/stage1_summary.md`](reports/stage1_summary.md)

### Stage 2: BERT real attention

`bert-base-uncased` real attention exposes structures synthetic data did not:
mixed entropy regimes, sink/punctuation rows, small-output-norm rows, and
protocol saturation.

Main findings:

- BERT attention is a real mixed-regime population, not a hand-made q_scale
  sweep.
- UTC-abs failed on relative-error targets; UTC-rel-hat fixed the denominator
  instability by estimating the full sparse output norm.
- Threshold-based evaluation had a comparable-row bias; exact-budget evaluation
  made all 4320 head-budget rows comparable.
- Under exact budget, UTC-rel-hat closes most of the local restricted-oracle
  gap:

```text
BERT exact-budget, 4320 rows
method       mean gap   median   p10
mass          0.071     0.245   -0.980
UTC-abs       0.017     0.208   -1.090
UTC-rel       0.380     0.813   -0.179
UTC-rel-hat   0.790     0.838    0.541
```

Report: [`reports/stage2_real_attention_summary.md`](reports/stage2_real_attention_summary.md)

### Stage 3: GPT-2 cross-model validation

The same exact-budget scorer ladder transfers to GPT-2 small causal attention
without changing the method definitions. Causal UTC uses prefix value sums,
which are natural for causal kernels.

```text
GPT-2 exact-budget, 324 rows
method       mean gap   median   p10
mass          0.017     0.265   -1.186
UTC-abs       0.309     0.602   -0.842
UTC-rel       0.579     0.837    0.185
UTC-rel-hat   0.828     0.881    0.642
```

Report: [`reports/stage3_cross_model_summary.md`](reports/stage3_cross_model_summary.md)

### Stage 4: metric boundary

The project then asks whether local sparse-attention error is also a model
behavior metric.

- W_O-projected error: rel-hat remains leading, but attenuates.
- GPT-2 single-head next-token KL: local projected-oracle inverts under KL.
- GPT-2 whole-layer KL: the effect size becomes healthier, but the local scorer
  ladder still does not reappear.

Whole-layer KL aggregate reduction:

```text
projected_oracle  -0.527
mass              -0.193
UTC-abs            0.069
UTC-rel           -0.049
UTC-rel-hat        0.074
```

Conclusion: local error control is real, but local restricted oracles are not
behavioral oracles.

Report: [`reports/stage4_metric_boundary_summary.md`](reports/stage4_metric_boundary_summary.md)

## Repository Layout

```text
docs/
  experiment_log.md                         # full run-by-run research log

reports/
  v1_summary.md
  stage1_summary.md
  stage2_real_attention_summary.md
  stage3_cross_model_summary.md
  stage4_metric_boundary_summary.md

scripts/
  stage0_sanity_check.py                    # decomposition and v1 experiments
  stage1_evaluate_value_proxies.py          # UTC and allocation tests
  stage1_mixed_regime_router.py             # mixed-regime hybrid tests
  stage2_bert_qkv.py                        # BERT QKV/P,V extraction
  stage2c_bert_exact_budget.py              # BERT exact-budget protocol
  stage3_gpt2_cross_model.py                # GPT-2 causal exact-budget validation
  stage4_metric_boundary_wo.py              # W_O-projected metric boundary
  stage4b_gpt2_logit_kl.py                  # single-head next-token KL
  stage4c_gpt2_whole_layer_kl.py            # whole-layer next-token KL
  plot_final_figures.py                     # final blog/report figures

results/
  bert/
  gpt2/
  metric_boundary/
  final_figures/
```

## Reproduce

Install dependencies:

```bash
pip install -r requirements.txt
```

Core synthetic checks:

```bash
python scripts/stage0_sanity_check.py
python scripts/stage1_evaluate_value_proxies.py
python scripts/stage1_mixed_regime_router.py
```

Real-model and boundary experiments:

```bash
python scripts/stage2_bert_qkv.py
python scripts/stage2c_bert_exact_budget.py
python scripts/stage3_gpt2_cross_model.py
python scripts/stage4_metric_boundary_wo.py --model gpt2
python scripts/stage4b_gpt2_logit_kl.py
python scripts/stage4c_gpt2_whole_layer_kl.py
```

Final figures:

```bash
python scripts/plot_final_figures.py
```

Notes:

- Some scripts download or load Hugging Face models (`bert-base-uncased`,
  `gpt2`).
- The checked-in `results/` directory contains the current experiment outputs,
  so the summaries can be inspected without rerunning all model probes.
- The code is organized as research scripts, not a reusable library or CI-backed
  package.

## Scope and Non-Claims

- No GPU kernel or wall-clock speedup is claimed.
- No end-to-end perplexity improvement is claimed.
- The oracle is a restricted top-k-by-probability oracle, not global subset
  selection.
- Stage 4 shows a boundary: local attention-output error and next-token KL are
  different targets.

Future work would need a KL-aware / readout-aware behavioral oracle, broader
model coverage, block-sparse/kernel experiments, or subset-selection beyond
top-k-by-probability. Those are deliberately outside the current artifact.
