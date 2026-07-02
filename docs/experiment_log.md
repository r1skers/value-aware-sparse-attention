# Experiment Log

## 2026-07-02 - Stage 0: decomposition sanity check and entropy baselines

Goal: verify the exact top-k pruning error decomposition and begin testing whether entropy is a useful row-level budget signal.

Core identity for one attention row:

```math
\|o - \tilde{o}\| = \delta \|\mu_R - \mu_S\|
```

where `S` is the retained top-k set, `R` is the dropped set, `delta = sum_{i in R} p_i`, `mu_S` is the retained value centroid, and `mu_R` is the dropped value centroid.

Current verified result:

```text
max abs diff  ~= 1e-15
mean abs diff ~= 1e-16
```

This confirms the implementation matches the exact decomposition up to float64 precision.

Observation from the first entropy baseline:

```text
fixed top-k, k=30:
  mean error ~= 0.8033

linear entropy-adaptive, k in [4, 32]:
  mean k     ~= 29.61
  mean error ~= 0.8153
```

Interpretation: the naive linear entropy-to-k rule does not outperform fixed top-k at a similar average budget in this random synthetic setting. This does not prove entropy is useless; it only says this particular mapping is not a strong budget allocator.

Next baseline design: fixed total budget.

Instead of comparing adaptive `k` against fixed `k` with potentially different total compute, impose:

```text
sum_j k_j = num_rows * target_mean_k
```

For example, with `num_rows = 128` and `target_mean_k = 30`, every method gets total budget `3840` retained positions. Fixed top-k uses `k_j = 30` for every row. Entropy-budgeted top-k redistributes the same total budget across rows according to entropy.

Implemented rule:

```text
1. Give every row k_min retained positions.
2. Compute row entropy H(p_j).
3. Distribute the remaining budget in proportion to positive entropy scores.
4. Cap each row by k_max.
5. Round to integers while preserving the exact total budget.
```

This tests the cleaner question:

```text
Given the same total top-k budget, does entropy know which rows deserve more retained tokens?
```

If entropy-budgeted does not beat fixed top-k under equal budget, the next analysis should inspect whether dropped mass and value centroid distance explain the remaining error better than entropy alone.

Implementation note: an earlier `H - H_min` weighting was too aggressive because the lowest-entropy row received no extra budget and stayed at `k_min` for all target budgets. The current baseline uses positive entropy scores directly, which is a fairer smooth entropy allocator.

Fixed-budget result at target mean `k=30`:

```text
fixed top-k, k=30:
  mean k        = 30.0
  mean error    = 0.8033
  max error     = 1.0984
  mean delta    = 0.3992

entropy-budgeted top-k, mean k=30:
  mean k        = 30.0
  k range       = [21, 31]
  total k       = 3840
  mean error    = 0.8052
  max error     = 1.1611
  mean delta    = 0.3984
```

Budget sweep:

```text
budget | fixed mean err | entropy mean err | fixed max err | entropy max err | entropy k range
     4 |         3.5239 |           3.5239 |        4.3401 |          4.3401 | [ 4,  4]
     8 |         2.2947 |           2.2945 |        2.9629 |          2.9629 | [ 7,  9]
    16 |         1.3997 |           1.4004 |        1.9173 |          1.9904 | [12, 17]
    24 |         1.0020 |           1.0002 |        1.4030 |          1.4428 | [17, 25]
    32 |         0.7505 |           0.7513 |        1.0128 |          1.0984 | [22, 33]
    48 |         0.4537 |           0.4540 |        0.6191 |          0.6976 | [32, 50]
    64 |         0.2847 |           0.2864 |        0.3833 |          0.5075 | [43, 67]
```

Interpretation: under this random synthetic setting, smooth entropy-budgeted top-k is nearly identical to fixed top-k across the budget curve. It sometimes improves mean error by a tiny amount and sometimes is slightly worse; max error is generally not better. This supports the cautious conclusion that entropy alone is not a strong budget allocation signal here, and motivates inspecting dropped mass and value centroid distance next.

Predictor analysis for fixed top-k, `k=30`:

```text
corr(entropy, error)              = 0.1404
corr(delta, error)                = 0.3377
corr(centroid_dist, error)        = 0.3773
corr(delta * centroid_dist, error)= 1.0000
```

Top-error rows:

```text
row | error  | entropy | delta  | centroid_dist | delta*centroid
121 | 1.0984 | 4.0948  | 0.3719 | 2.9537        | 1.0984
 90 | 1.0304 | 4.5472  | 0.4806 | 2.1442        | 1.0304
125 | 0.9962 | 4.4320  | 0.4548 | 2.1904        | 0.9962
114 | 0.9877 | 4.4286  | 0.4466 | 2.2115        | 0.9877
106 | 0.9865 | 4.1276  | 0.3597 | 2.7429        | 0.9865
```

Interpretation: entropy has weak correlation with output error in this setting. Dropped mass and value centroid distance each explain part of the variation, while their product exactly matches the error by construction. The top-error rows show two different failure modes: some rows have larger dropped mass, while others have moderate dropped mass but a large value centroid distance. This is the first concrete evidence for the project's value-aware framing.

## q_scale sweep

Motivation: the initial `q_scale=1.0` setup only covered a relatively high-entropy regime (`H_norm` range about `0.58-0.94`). To test whether the weak entropy result was an artifact of a narrow entropy range, we scanned `q_scale`. Scaling `Q` scales the logits, so larger `q_scale` corresponds to lower softmax temperature and sharper attention.

Fixed top-k setting: `k=30`.

```text
q_scale | H_norm mean | mean err | mean delta | corr(H,err) | corr(delta,err) | corr(centroid,err) | corr(delta,centroid)
   0.25 |      0.9937 |   1.1508 |     0.6842 |      0.0489 |          0.0540 |             0.9852 |              -0.1176
   0.50 |      0.9747 |   1.0331 |     0.5925 |      0.1025 |          0.1476 |             0.8965 |              -0.3040
   1.00 |      0.8979 |   0.8033 |     0.3992 |      0.1404 |          0.3377 |             0.3773 |              -0.6809
   2.00 |      0.6475 |   0.3574 |     0.1222 |      0.5312 |          0.7612 |            -0.2654 |              -0.7264
   4.00 |      0.3051 |   0.0275 |     0.0065 |      0.6819 |          0.9260 |            -0.5007 |              -0.6354
   8.00 |      0.1305 |   0.0002 |     0.0000 |      0.4618 |          0.9942 |            -0.2985 |              -0.3220
```

Interpretation:

```text
1. In very high-entropy regimes (`q_scale=0.25, 0.5`), dropped mass is large and relatively uniform, so value centroid distance becomes the strongest driver of row-to-row error variation.
2. In sharper low-entropy regimes (`q_scale>=2`), dropped mass becomes the dominant error signal because top-k captures almost all probability mass.
3. Entropy becomes more correlated with error in sharper regimes, but mean error also collapses toward zero.
```

This makes the Stage 0 conclusion more nuanced: entropy is not always useless, but it is regime-dependent and incomplete. The decomposition still gives the clean explanation: different regimes shift importance between `delta` and value geometry, while output error remains their product.

Note on negative `corr(centroid_dist, error)` at high `q_scale`: this should not be interpreted as "larger value geometry reduces error." In those sharper regimes, `delta` dominates error variation and is negatively correlated with `centroid_dist` (`corr(delta, centroid_dist)` is about `-0.73` at `q_scale=2` and `-0.64` at `q_scale=4`). Since `error = delta * centroid_dist`, the marginal correlation between centroid distance and error can become negative when the multiplicative partner `delta` explains most of the variation. This is a statistical artifact of the product structure, not a causal mechanism.

## Oracle error-budgeted top-k

Goal: estimate the upper-bound value of a perfect value-aware error signal. For each attention row and relative error threshold `epsilon`, choose the smallest `k` such that:

```text
||o - o_tilde_k|| / (||o|| + eta) <= epsilon
```

This is an oracle method because it directly evaluates the true row-level output error. It is not yet a deployable pruning rule.

Baseline comparison:

```text
fixed@mean k:
  fixed top-k using k = ceil(oracle mean k)

fixed needed k:
  smallest fixed k that satisfies the same worst-row relative error threshold
```

Result on `q_scale=1.0`, `N=128`, `d=64`, `seed=0`:

```text
eps  | oracle mean k | oracle max k | oracle mean rel | oracle max rel | fixed@mean k | fixed@mean max rel | fixed needed k
0.10 |         90.66 |          108 |          0.0977 |         0.1000 |           91 |             0.1777 |            108
0.20 |         70.30 |           86 |          0.1952 |         0.1998 |           71 |             0.3185 |             86
0.50 |         39.99 |           54 |          0.4895 |         0.4998 |           40 |             0.7176 |             54
1.00 |         20.59 |           31 |          0.9740 |         1.0000 |           21 |             1.5168 |             31
2.00 |          8.39 |           14 |          1.9084 |         1.9986 |            9 |             2.5914 |             14
```

Interpretation: oracle error-budgeted top-k gives a clear upper-bound curve. At equal average budget, fixed top-k can have much worse worst-row relative error. To satisfy the same per-row threshold using fixed top-k, the required fixed `k` matches the oracle max `k`, which is substantially larger than the oracle mean `k`. This shows why row-wise adaptive allocation can save budget when row difficulty is heterogeneous.

## Dropped-mass-budgeted top-k

Goal: add a stronger Q,K-only adaptive baseline between entropy and value-aware oracle. For each row, choose the smallest `k` such that:

```text
delta(k) = sum dropped p_i <= tau
```

This baseline uses only attention probabilities, not values.

Result on `q_scale=1.0`, `N=128`, `d=64`, `seed=0`:

```text
tau  | mass mean k | mass max k | max delta | mass mean rel | mass max rel | fixed@mean k | fixed@mean max rel
0.10 |       79.40 |         91 |    0.1000 |        0.1483 |       0.1921 |           80 |             0.2360
0.20 |       57.21 |         71 |    0.2000 |        0.2925 |       0.3726 |           58 |             0.4162
0.40 |       30.36 |         41 |    0.4000 |        0.6788 |       0.8201 |           31 |             0.9808
0.60 |       14.59 |         21 |    0.5997 |        1.2984 |       1.6798 |           15 |             1.8448
0.80 |        5.16 |          9 |    0.7999 |        2.6313 |       3.4211 |            6 |             3.4182
```

Interpretation: dropped-mass budgeting is a meaningful Q,K-only baseline. It generally improves worst-row relative error over fixed top-k at a similar average budget because it directly controls probability mass discarded by pruning. But it still does not control value geometry, so it is weaker than the oracle error-budgeted rule when the goal is an output-error bound.

## Matched-budget comparison: fixed vs dropped-mass vs oracle

Motivation: the oracle sweep uses `epsilon`, while the dropped-mass sweep uses `tau`; those thresholds do not live on the same axis. To compare methods directly, calibrate `epsilon` and `tau` so oracle and dropped-mass methods land on the same target mean retained budget, then compare worst-row relative error.

Script:

```text
scripts/compare_matched_budget.py
```

Result on `q_scale=1.0`, `N=128`, `d=64`, `seed=0`:

```text
target_k | oracle mean k | oracle max_rel | mass mean k | mass max_rel | fixed max_rel
      80 |         80.01 |         0.1468 |       80.01 |       0.1807 |        0.2360
           (calibrated eps=0.1469, tau=0.0982)
      60 |         59.99 |         0.2726 |       60.01 |       0.3562 |        0.4127
           (calibrated eps=0.2726, tau=0.1850)
      40 |         40.00 |         0.4996 |       40.02 |       0.6382 |        0.7176
           (calibrated eps=0.4997, tau=0.3146)
```

Interpretation:

```text
At the same average retained budget:
  oracle error-budgeted < dropped-mass-budgeted < fixed top-k
```

Dropped-mass adaptive top-k captures a substantial part of the gain from fixed top-k to the value-aware oracle:

```text
target_k=80: (fixed - mass) / (fixed - oracle) = 62.0%
target_k=60: (fixed - mass) / (fixed - oracle) = 40.3%
target_k=40: (fixed - mass) / (fixed - oracle) = 35.8%
```

But it does not close the gap to the value-aware oracle. In worst-row relative error, dropped-mass remains about `23-31%` above oracle:

```text
target_k=80: (mass - oracle) / oracle = 23.1%
target_k=60: (mass - oracle) / oracle = 30.7%
target_k=40: (mass - oracle) / oracle = 27.7%
```

This is the first clean quantitative evidence for value-aware incremental value: Q,K-only mass control is already strong, but value geometry still buys additional worst-row error reduction at matched compute budget.

Implementation note on the calibration: the first version of the bisection in
`scripts/compare_matched_budget.py` had the update direction reversed for the
oracle (`error_budgeted_k` mean k is *decreasing* in epsilon), which silently
converged to `eps ~ 5` and produced an oracle mean k of 2.2 regardless of
target. Symptom in the output: oracle max_rel stuck at ~5.0 while nominally
"matched" to budget 80. Fixed by flipping the branch; the table above is from
the corrected run.

## v1 closed

Distilled conclusion, closing figure, and open next steps moved to
[`reports/v1_summary.md`](../reports/v1_summary.md). This log stays the raw
run-by-run record; the report is the citable summary.

## q_scale sweep of the matched-budget comparison

Prediction carried over from the v1 regime sweep: value-awareness should
matter most in high-entropy regimes (where `corr(centroid_dist, error)` was
~0.99) and least in sharp regimes (where `delta` alone explained ~99% of
error variance). This experiment tests that prediction with an independent
methodology (budget allocation instead of correlation analysis).

Script: `scripts/sweep_matched_budget_qscale.py`. All methods calibrated to
mean k = 40 per q_scale; `gap closed by mass` = fraction of the
fixed-to-oracle worst-row error gap recovered by the Q,K-only baseline.

```text
q_scale | H_norm mean | oracle max_rel | mass max_rel | fixed max_rel | gap closed by mass
   0.25 |      0.9937 |       1.176514 |     1.548102 |      1.526547 |            -0.0616
   0.50 |      0.9747 |       0.942694 |     1.150635 |      1.150635 |             0.0000
   1.00 |      0.8979 |       0.499588 |     0.638230 |      0.717614 |             0.3641
   2.00 |      0.6475 |       0.091315 |     0.130784 |      0.224526 |             0.7037
   4.00 |      0.3051 |       0.001678 |     0.002408 |      0.026056 |             0.9701
   8.00 |      (excluded: budget saturation, see below)
```

The gap closed by the Q,K-only baseline rises monotonically with sharpness:
-6% -> 0% -> 36% -> 70% -> 97%. This matches the v1 prediction exactly:

1. Sharp regimes: `delta` is essentially the error, Q,K-only adaptivity
   recovers ~all of the oracle's advantage, V adds almost nothing.
2. High-entropy regimes: `delta` carries no row-level information
   (`corr(delta, error) = 0.05` at `q_scale=0.25`), so reallocating budget by
   dropped mass is noise -- at `q_scale=0.25` it is actually *worse* than
   fixed top-k. The remaining ~23% worst-row error gap to the oracle is
   reachable only with value information.

Intuition for why V matters less as attention sharpens (measure-theoretic
reading of the decomposition): `error = delta * ||mu_R - mu_S||` weights the
value-geometry displacement by the probability mass of the dropped region.
Sharper attention means the dropped tail carries vanishing mass, and a
displacement supported on a small-mass region moves the output little --
`delta -> 0` suppresses the geometry factor regardless of how large the
displacement is. Conversely, near-uniform attention puts large mass on the
dropped region, so where its value centroid sits becomes the dominant term.
The empirical monotone trend is this identity read as a weighting statement.

Saturation artifact at `q_scale=8`: the first run showed the Q,K-only
baseline "beating" the oracle by 7.9x there. Diagnosis: attention is so
sharp that most rows hit exact float-zero error at k ~= 14, so no threshold
can force either method to spend the full budget (oracle achieved mean k =
13.5, mass 21.1, against a target of 40). The budget constraint stops
binding and the comparison degenerates; the row is excluded and the script
now flags saturation explicitly. Worth keeping as an observation: in
ultra-sharp regimes the pruning problem is trivial -- extra budget cannot
even be spent.

Combined takeaway: the value of value-awareness is not a constant; it
concentrates precisely in the high-entropy regime, and the closer attention
is to uniform, the more V information becomes the *only* useful signal. Two
independent experiments (correlation analysis and matched-budget allocation)
now support this from different directions.
