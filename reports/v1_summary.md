# v1 Summary: Value-Aware Error Analysis for Sparse Attention

Status: **closed 2026-07-02**. Full run-by-run detail lives in
[`docs/experiment_log.md`](../docs/experiment_log.md); this report is the
distilled research summary.

## Core Identity

For a single attention row, let `S` be the retained set, `R` the dropped set,
and:

```math
\delta = \sum_{i \in R} p_i
```

Let `mu_S` and `mu_R` be the value centroids of the retained and dropped
regions under the attention weights. Then top-k pruning with renormalization
satisfies:

```math
\|o - \tilde{o}\| = \delta \|\mu_R - \mu_S\|
```

This identity was verified to float64 precision across all rows in the
synthetic experiments (`max abs diff ~= 1e-15`). It is exact, not an empirical
fit.

## What We Tested

We compared four increasingly informative pruning signals:

```text
fixed top-k:
  same k for every row

entropy-adaptive:
  Q,K-only; uses H(p) as a row difficulty signal

dropped-mass adaptive:
  Q,K-only; chooses k to control dropped probability mass delta(k)

restricted value-aware oracle:
  Q,K,V-aware; chooses the best k within the top-k-by-probability family
  using the true output error
```

Important language discipline: the oracle here is **restricted**. It optimizes
`k` while the retained set is still forced to be the top-k entries by attention
probability. It is not a global subset-selection oracle.

## Main Findings

### 1. Entropy Is A Weak Baseline, Not The Main Signal

At `q_scale=1.0`, `k=30`:

```text
corr(entropy, error)       = 0.1404
corr(delta, error)         = 0.3377
corr(centroid_dist, error) = 0.3773
```

Equal-budget entropy allocation performed almost identically to fixed top-k,
with no stable improvement in max error. Entropy is useful as a rough regime
indicator, but not as a strong pruning-error signal.

### 2. Error Regime Shifts With Attention Sharpness

Sweeping `q_scale` changes the softmax regime:

```text
small q_scale -> diffuse / high-entropy attention
large q_scale -> sharp / low-entropy attention
```

The dominant error factor changes accordingly:

```text
High entropy:
  dropped mass is large and weakly informative across rows;
  value centroid displacement explains row-to-row error.

Sharp attention:
  top-k captures most probability mass;
  dropped mass explains nearly all remaining error.
```

See [`results/regime_sweep_summary.png`](../results/regime_sweep_summary.png).

### 3. Dropped Mass Is Strong, But Regime-Dependent

Dropped-mass adaptive top-k is the strongest Q,K-only baseline. It directly
controls the `delta` factor in the decomposition. But its power depends on the
attention regime:

```text
sharp regimes:
  near-oracle performance

high-entropy regimes:
  useless or even worse than fixed top-k
```

This is the central reason the next stage should focus on value-aware proxies
in high-entropy regimes.

### 4. Value Awareness Has Quantifiable Incremental Value

The cleanest comparison calibrates methods to the same average retained
budget and compares worst-row relative error:

```text
target_k | oracle max_rel | mass max_rel | fixed max_rel
      80 |         0.1468 |       0.1807 |        0.2360
      60 |         0.2726 |       0.3562 |        0.4127
      40 |         0.4996 |       0.6382 |        0.7176
```

At matched budget:

```text
restricted oracle < dropped-mass adaptive < fixed top-k
```

Dropped mass captures part of the fixed-to-oracle gap, but remains about
`23-31%` worse than the restricted value-aware oracle in worst-row relative
error. This is the first quantitative evidence that value geometry adds
measurable information beyond Q,K-only mass control.

### 5. The Value Gain Concentrates In High-Entropy Regimes

Sweeping the matched-budget comparison over `q_scale` confirmed the regime
prediction:

```text
q_scale | H_norm mean | oracle max_rel | mass max_rel | fixed max_rel | gap closed by mass
   0.25 |      0.9937 |       1.176514 |     1.548102 |      1.526547 |           -0.0616
   0.50 |      0.9747 |       0.942694 |     1.150635 |      1.150635 |            0.0000
   1.00 |      0.8979 |       0.499588 |     0.638230 |      0.717614 |            0.3641
   2.00 |      0.6475 |       0.091315 |     0.130784 |      0.224526 |            0.7037
   4.00 |      0.3051 |       0.001678 |     0.002408 |      0.026056 |            0.9701
```

In sharp regimes, dropped mass nearly recovers the restricted oracle. In
high-entropy regimes, dropped mass provides no useful allocation signal, and
value geometry is the remaining source of improvement.

## Measure-Theoretic Intuition

An attention row defines a discrete probability measure over token positions.
The attention output is the integral of the value function under that measure:

```math
o = \int V \, d\mu
```

Top-k pruning replaces the original measure with the conditional measure on
the retained set `S`. The error is the mass of the removed region times the
displacement between the retained and removed conditional expectations:

```math
o - \tilde{o} = \delta(\mu_R - \mu_S)
```

When attention is sharp, the removed set has tiny measure, so value geometry is
suppressed. When attention is diffuse, the removed set has nontrivial measure,
so where its value centroid sits becomes decisive.

## Current Thesis

We move from entropy-based pruning toward error-aware pruning. The
decomposition shows output error is governed by dropped mass and value
geometry. Dropped mass is the strongest Q,K-only baseline, but its power is
regime-dependent: near-oracle in sharp regimes, useless in high-entropy
regimes, which is exactly where value geometry explains the remaining gap. The
next question is whether cheap value-aware proxies can recover part of the
restricted oracle advantage there.

## Next Stage

The next work item is **cheap value-aware proxy evaluation**:

```text
Can a low-cost approximation to ||mu_R - mu_S|| recover part of the
restricted oracle advantage in high-entropy regimes?
```

Candidate proxies:

```text
weighted value variance
top-k boundary / next-block value distance
block-level retained-vs-dropped centroids
```

Evaluation target:

```text
fixed < dropped-mass < cheap value proxy < restricted oracle
```

especially in high-entropy regimes.
