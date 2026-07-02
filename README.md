# Value-Aware Sparse Attention

Research-style implementation project for studying output error in sparse
attention approximation.

The project starts from an exact decomposition for top-k attention pruning:

```math
\|o - \tilde{o}\| = \delta \|\mu_R - \mu_S\|
```

where `delta` is the dropped attention probability mass and `mu_R`, `mu_S` are
the value centroids of the dropped and retained regions. The central question
is whether cheap value-aware signals can improve sparse attention pruning
beyond Q,K-only heuristics such as entropy and dropped mass.

## Current Status

Stage 0 / v1 is complete:

- Verified the decomposition to float64 precision.
- Benchmarked entropy, dropped mass, and value centroid displacement.
- Built equal-budget comparisons for fixed top-k, entropy allocation,
  dropped-mass allocation, and a restricted value-aware oracle.
- Found a regime shift: dropped mass works in sharp attention, while value
  geometry matters most in high-entropy attention.

See the distilled report:

```text
reports/v1_summary.md
```

and the run-by-run log:

```text
docs/experiment_log.md
```

## Repository Layout

```text
scripts/
  stage0_sanity_check.py          # core decomposition and baseline experiments
  compare_matched_budget.py       # fixed vs mass vs restricted oracle at matched budget
  sweep_matched_budget_qscale.py  # matched-budget comparison across q_scale regimes
  plot_regime_sweep.py            # summary figure generation

reports/
  v1_summary.md                   # current distilled findings

docs/
  experiment_log.md               # detailed experiment log

results/
  regime_sweep_summary.png        # generated v1 figure
```

## Reproduce

Install the minimal dependencies:

```bash
pip install -r requirements.txt
```

Run the main experiment script:

```bash
python scripts/stage0_sanity_check.py
```

Run the matched-budget q_scale sweep:

```bash
python scripts/sweep_matched_budget_qscale.py
```

Generate the summary figure:

```bash
python scripts/plot_regime_sweep.py
```

## Naming Note

The current "oracle" is a **restricted oracle**: it chooses the best row-wise
`k` while the retained set is still constrained to be top-k by attention
probability. It is not a global subset-selection oracle.

## Next Step

Evaluate cheap value-aware proxies for the value centroid displacement term,
especially in high-entropy regimes where dropped-mass allocation fails.
