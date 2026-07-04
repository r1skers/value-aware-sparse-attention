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

## Cheap value proxy v0: Uniform-Tail Centroid Proxy

Motivation: exact `||mu_R - mu_S||` requires reading dropped `V`, so it is an
offline oracle, not a deployable sparse-attention rule. The first cheap proxy
tests a high-entropy assumption: when attention over the dropped tail is close
to uniform, approximate the dropped weighted centroid by the unweighted tail
centroid.

Name:

```text
UTC proxy = Uniform-Tail Centroid proxy
```

Definition for retained set `S` and dropped set `R`:

```text
mu_R_hat = (sum_i V_i - sum_{i in S} V_i) / |R|
proxy    = ||mu_R_hat - mu_S||
score    = delta * proxy
```

Cost model: `sum_i V_i` is a sequence-level precompute; `sum_{i in S} V_i` uses
retained values already read by sparse attention. The proxy does not scan
dropped `V` per query row.

Script:

```text
scripts/stage1_evaluate_value_proxies.py
```

First result with fixed top-k, `k=40`, `N=128`, `d=64`, `seed=0`:

```text
q_scale | H_norm mean | corr(delta,err) | corr(trueC,err) | corr(UTC,trueC) | corr(delta*UTC,err) | UTC rel MAE
   0.25 |      0.9937 |          0.1512 |          0.9834 |           0.9934 |              0.9935 |      0.0086
   0.50 |      0.9747 |          0.2777 |          0.8907 |           0.9790 |              0.9792 |      0.0173
   1.00 |      0.8979 |          0.4523 |          0.3293 |           0.9857 |              0.9646 |      0.0348
   2.00 |      0.6475 |          0.7731 |         -0.2757 |           0.9951 |              0.9902 |      0.0483
   4.00 |      0.3051 |          0.9276 |         -0.5037 |           0.9934 |              0.9976 |      0.0465
```

Interpretation: on iid synthetic `V`, UTC is an unexpectedly strong proxy for
the true centroid displacement, especially in the high-entropy regime where it
was designed to work. This is a promising first cheap value-side signal, but it
may be optimistic because `V` is iid Gaussian and independent of attention
weights. Next validation should test multiple seeds and less iid value
structure before treating UTC as robust.

## UTC allocation test

Predictor correlation is not enough; the proxy must also improve budget
allocation. This test replaces the restricted oracle's true error score:

```text
E(k) = delta(k) * ||mu_R(k) - mu_S(k)||
```

with the cheap UTC score:

```text
E_hat_UTC(k) = delta(k) * ||mu_R_hat_uniform_tail(k) - mu_S(k)||
```

Then the UTC threshold is calibrated so the method lands on the same target
mean retained budget as fixed, dropped-mass, and restricted-oracle baselines.

Script:

```text
scripts/stage1_evaluate_value_proxies.py
```

Result at matched mean `k=40`:

```text
q_scale | H_norm mean | oracle max | UTC max | mass max | fixed max | UTC gap closed | mass gap closed
   0.25 |      0.9937 |   1.176514 | 1.296138 | 1.548102 |  1.526547 |         0.6582 |         -0.0616
   0.50 |      0.9747 |   0.942694 | 1.057535 | 1.150635 |  1.150635 |         0.4477 |          0.0000
   1.00 |      0.8979 |   0.499588 | 0.732066 | 0.638230 |  0.717614 |        -0.0663 |          0.3641
   2.00 |      0.6475 |   0.091315 | 0.205162 | 0.130784 |  0.224526 |         0.1454 |          0.7037
   4.00 |      0.3051 |   0.001678 | 0.004277 | 0.002408 |  0.026056 |         0.8934 |          0.9701
```

Interpretation: UTC passes the first allocation test in the high-entropy
regime where it was designed to help. At `q_scale=0.25`, dropped-mass allocation
is worse than fixed, while UTC closes about 66% of the fixed-to-oracle gap. At
`q_scale=0.5`, dropped mass is tied with fixed, while UTC closes about 45% of
the gap. In intermediate and sharp regimes, dropped mass is better or already
near oracle; UTC should not be treated as a universal replacement for
dropped-mass. Its current value is regime-specific: it is a cheap value-side
signal for diffuse attention.

Visualization:

```text
scripts/plot_stage1_gap_closed.py
results/stage1_utc_vs_mass_gap_closed.png
```

The figure plots fixed-to-oracle gap closed against mean normalized entropy. It
shows the complementarity directly: UTC is useful in high-entropy regimes,
while dropped-mass dominates as attention sharpens.

## Entropy-routed hybrid

Motivation: the Stage 1 allocation test showed complementary failure modes:
UTC helps in high-entropy regimes where dropped mass is uninformative, while
dropped-mass allocation is stronger in intermediate and sharp regimes. This
test turns entropy from a pruning score into a router:

```text
if H(p) / log(n) >= 0.94:
    choose k using UTC score
else:
    choose k using dropped-mass score
```

Each branch is calibrated to the same mean retained budget (`mean k = 40`).
That keeps this first hybrid test simple: entropy chooses which signal to
trust, but does not directly allocate more budget.

Script:

```text
scripts/stage1_evaluate_value_proxies.py
scripts/plot_stage1_gap_closed.py
```

Result:

```text
q_scale | H_norm mean | oracle max | hybrid max | UTC max | mass max | fixed max | hybrid gap | UTC gap | mass gap | UTC-routed
   0.25 |      0.9937 |   1.176514 |   1.296138 | 1.296138 | 1.548102 |  1.526547 |     0.6582 |  0.6582 |  -0.0616 |       1.00
   0.50 |      0.9747 |   0.942694 |   1.057535 | 1.057535 | 1.150635 |  1.150635 |     0.4477 |  0.4477 |   0.0000 |       0.99
   1.00 |      0.8979 |   0.499588 |   0.638230 | 0.732066 | 0.638230 |  0.717614 |     0.3641 | -0.0663 |   0.3641 |       0.01
   2.00 |      0.6475 |   0.091315 |   0.130784 | 0.205162 | 0.130784 |  0.224526 |     0.7037 |  0.1454 |   0.7037 |       0.00
   4.00 |      0.3051 |   0.001678 |   0.002408 | 0.004277 | 0.002408 |  0.026056 |     0.9701 |  0.8934 |   0.9701 |       0.00
```

Interpretation: with this threshold, the hybrid behaves like UTC in the two
high-entropy regimes and like dropped mass elsewhere. It removes the UTC
failure at `q_scale=1.0` while preserving UTC's gains at `q_scale=0.25` and
`0.5`. This is the first method-level evidence for the updated role of
entropy: not an error predictor by itself, but a cheap regime selector for
choosing between Q,K-only mass control and a cheap value-aware proxy.

Visualization:

```text
results/stage1_utc_vs_mass_gap_closed.png
```

Important caveat: this is still synthetic, single-seed evidence, and the
router threshold was selected from the observed regime split. The next
stability test should add multiple seeds and report error bands before treating
the threshold as robust.

Stronger caveat (added after review): in this per-dataset setup the experiment
is close to circular. Every dataset has a single regime, so the row-level
router degenerates into a dataset-level switch (`UTC-routed` column is ~1.00 or
~0.00 everywhere), and with an in-sample threshold the hybrid curve equals the
pointwise max of the two pure curves *by construction*. This table is a
consistency check of the routing code, not evidence that entropy routing works.
The evidence-bearing experiment is the mixed-regime population test below.

## Mixed-regime router test (the real hybrid experiment)

Design: ONE population whose rows span regimes — each row draws its own
`q_scale` from {0.25, 0.5, 1.0, 2.0, 4.0} (Q rows scaled independently; K, V
shared). Router threshold `H_norm >= 0.94` carried over from the per-dataset
analysis, NOT tuned on this data. All methods calibrated to the same overall
mean k = 40. Three seeds. Script: `scripts/stage1_mixed_regime_router.py`.

Pre-registered prediction (before running): hybrid-v0 pins each router branch
to the same mean budget, which forbids cross-regime budget transfer; on a mixed
population, dropped-mass with a single tau naturally shifts budget from sharp
rows to diffuse rows, so hybrid-v0 may lose to pure mass. A budget-respecting
variant (`hybrid-b`: the global mass calibration decides the budget split
across router groups, UTC re-allocates within the high-entropy group; both
ingredients Q,K-only-or-cheap, so deployable) should do better.

Result (gap closed on overall worst-row relative error, per seed):

```text
method    | seed 0 | seed 1 | seed 2
mass      |  0.833 |  0.751 |  0.791
UTC       |  0.804 |  0.761 |  0.772
hybrid-v0 |  0.267 |  0.164 |  0.168
hybrid-b  |  0.855 |  0.902 |  0.878
```

Representative per-group detail (seed 0): oracle sends mean k ~76 to
high-entropy rows and ~18 to sharp rows; mass tracks this split almost exactly
(74/19); hybrid-v0 is pinned at 40/40 and collapses (gap closed 0.27 vs 0.83
for pure mass); hybrid-b keeps the mass split and improves the high-entropy
group's worst-row error over mass (0.549 vs 0.629) while leaving the sharp
group untouched.

Findings:

1. The prediction was confirmed: hybrid-v0 collapses on mixed data (0.16-0.27
   gap closed, far below either pure strategy). Its earlier per-dataset
   "success" was an artifact of the circular setup.
2. **Budget transfer across regimes matters far more than signal choice.**
   Dropped-mass with one tau is a strong cross-regime budget allocator — this
   capability was structurally invisible in the per-dataset tests.
3. **hybrid-b beats both pure strategies on all three seeds** (0.855/0.902/
   0.878 vs mass 0.751-0.833 and UTC 0.761-0.804). This is the first
   non-circular, multi-seed evidence for the two-signal division of labor:
   mass decides how much budget each regime group gets, UTC decides how it is
   spent inside the high-entropy group.
4. Entropy's final role is narrower than "router": with no budget authority,
   routing alone is nearly worthless; entropy only earns its keep as the
   grouping variable for a budget-respecting delegation.

Caveats: synthetic values, one threshold (0.94, inherited not tuned — a
sensitivity sweep is still pending), N=128 single-layer setting.

## Sensitivity sweeps for hybrid-b (unfixing the two dangerous knobs)

The mixed-regime result relied on two fixed choices that could plausibly have
flipped it: the router threshold (0.94) and the budget level (mean k = 40).
Script: `scripts/stage1_router_sensitivity.py` (same mixed population, 3 seeds,
bisection iters reduced to 12 — resolution is ample for these checks).

Threshold sweep at budget 40 (hybrid-b gap closed):

```text
seed | thr=0.85 | thr=0.90 | thr=0.94 | thr=0.97 | mass | UTC
   0 |    0.890 |    0.904 |    0.856 |    0.856 | 0.834 | 0.805
   1 |    0.850 |    0.871 |    0.899 |    0.899 | 0.752 | 0.762
   2 |    0.844 |    0.856 |    0.876 |    0.876 | 0.789 | 0.750
```

hybrid-b beats both pure strategies at every threshold in [0.85, 0.97] on
every seed. The threshold sits on a plateau, not a knife-edge (0.94 and 0.97
give identical allocations — no rows fall between them). No tuning needed;
any value in the band works.

Budget sweep at threshold 0.94 (gap closed):

```text
seed | budget | mass | UTC | hybrid-b
   0 |     20 | 0.732 | 0.729 |   0.732
   0 |     40 | 0.834 | 0.805 |   0.856
   0 |     60 | 0.788 | 0.854 |   0.903
   1 |     20 | 0.855 | 0.711 |   0.895
   1 |     40 | 0.752 | 0.762 |   0.899
   1 |     60 | 0.772 | 0.841 |   0.880
   2 |     20 | 0.831 | 0.701 |   0.831
   2 |     40 | 0.789 | 0.750 |   0.876
   2 |     60 | 0.791 | 0.828 |   0.867
```

No saturation at any configuration (oracle achieved budgets all within 1.0 of
target). Findings:

1. hybrid-b >= max(mass, UTC) in all 9 configurations; strictly better in 7.
2. At the tightest budget (k=20), hybrid-b's edge vanishes on 2 of 3 seeds
   (it exactly equals mass). Mechanism: at tight budgets the overall worst row
   sits in the low-entropy group, which hybrid-b leaves untouched by design —
   the UTC refinement inside the high-entropy group cannot move the overall
   max. Honest reading: the delegation scheme adds value at moderate-to-loose
   budgets and is never harmful.
3. Side observation: pure UTC strengthens with budget and overtakes pure mass
   at k=60 on all seeds (0.83-0.85 vs 0.77-0.79). Plausible mechanism: at
   larger k the dropped tail is thin for every row, so delta loses row-to-row
   discrimination and value geometry takes over — the regime story recurring
   along the budget axis instead of the entropy axis. Worth a dedicated check
   before treating as established.

Status: the two dangerous knobs are unfixed and the mixed-regime conclusion
survives both. Remaining fixed structure is environment-level (synthetic iid
values, N=128, single layer) — the natural next chapter is real attention
maps, where the regime mixture, entropy distribution, and V structure are all
given by the data instead of chosen by us.

Visualization for the mixed-regime + sensitivity results:
`results/stage1_mixed_regime_summary.png` (generated by
`scripts/plot_stage1_mixed_regime.py`). Stage 1 distilled summary:
`reports/stage1_summary.md`.

## 2026-07-03 - Stage 2: real BERT attention (extraction + first allocation)

Pipeline: `scripts/stage2_bert_qkv.py` extracts per-head (P, V) from
bert-base-uncased on one 20 Newsgroups document (256 tokens, no padding,
[CLS]/[SEP] kept). Extraction anchor: per-head P recomputed from layer inputs
and layer weights matches the model's reported attentions to ~1e-6 (float32
forward vs float64 recompute — expected precision). Decomposition identity
holds on real data to 1e-15. Saved: `results/bert/qkv_layer{0,5,11}.npz`,
H_norm census for all 12 layers in `results/bert/hnorm_census.npy`.

Regime census (the first real-data finding): real BERT attention is a natural
mixed-regime population, far more mixed than any synthetic setting —

```text
pooled over 12 layers x 12 heads: mean H_norm = 0.50,
  ~49% of rows sharp (H_norm <= 0.50), ~5% diffuse (>= 0.90)
high-entropy population concentrates in layer 0 (42% of rows >= 0.90,
  22.6% >= 0.94); middle layers (5-8) are 70-90% sharp; late layers mixed.
```

So the battleground regime exists in real BERT but is strongly
layer-structured: layer 0 is UTC country, middle layers are mass country.

First matched-budget allocation on real heads
(`scripts/stage2_bert_allocation.py`, mean k = 64 of N = 256, router
threshold 0.90 from the validated plateau band, NOT tuned on this data):

```text
head                  | H_mean | mask | mass   | UTC    | hybrid-b | budget ok
L0  h4 (most diffuse) |  0.952 | 0.96 | -0.344 |  0.527 |  -0.325  | yes
L0  h1 (median)       |  0.888 | 0.41 |  0.604 |  0.610 |   0.604  | yes
L5  h0 (sharp)        |  0.333 | 0.00 |  0.894 |  0.927 |   0.894  | NO (saturated)
L11 h9 (most diffuse) |  0.749 | 0.05 | -1.215 | -0.714 |  -1.215  | yes
L11 h5 (median)       |  0.582 | 0.00 |  0.743 |  0.743 |   0.743  | NO (saturated)
```

Three observations:

1. **The synthetic headline survives contact with real data where it matters
   most.** On the genuinely diffuse layer-0 head, dropped-mass allocation is
   harmful (-0.34) and UTC closes 53% of the fixed-to-oracle gap — the same
   signature as synthetic q_scale=0.25, now on real attention and real values.
2. **hybrid-b inherits mass's failures on real heads.** On L0 h4 the router
   group is 96% of rows, so "delegation" degenerates: the tiny low-entropy
   remainder keeps mass's allocation and contains the binding worst row.
   Delegation assumed mass is reliable inside the low-entropy group; on real
   heads that assumption can fail. Design gap, not implementation bug.
3. **New phenomenon: heads where ALL adaptive methods are worse than fixed**
   (L11 h9: mass -1.22, UTC -0.71). Not seen anywhere in synthetic data.
   Suspected cause: real attention structure (e.g., attention-sink rows on
   [SEP]/[CLS]) breaks the link between per-row signals and per-row error in
   a way iid values never did. Needs dedicated diagnosis before any real-data
   claim about adaptive allocation.

Caveats: single document, single model; sharp heads saturate at mean k=64
(budget not binding — those rows are not matched comparisons and are flagged);
L5 head-selection printed the same head twice (cosmetic); threshold 0.90 not
re-validated on real data.

Blog numbering decision: synthetic proxy + hybrid work (stage 1) = artifact
6.2; this real-BERT chapter = artifact 6.3.

## Stage 2 diagnosis: L11 h9 adaptive failure

Question: why does BERT layer 11 head 9 make all adaptive methods worse than
fixed-k?

Script:

```text
scripts/stage2_diagnose_bert_head.py
```

Result summary (`mean k = 64`, `N = 256`, threshold `H_norm >= 0.90`):

```text
method    mean_k   max_rel  worst row  k_at_worst
fixed      64.00    0.5436       75        64
mass       64.05    1.0024       86        16
UTC        63.82    0.8134      185        16
hybrid-b   64.02    1.0024       86        16
oracle     63.39    0.1685       26        65
```

The original attention-sink suspicion was partly right but the sink is not
primarily `[CLS]` or `[SEP]`. The highest received-attention columns are
punctuation tokens:

```text
col  75 "." received 8.967
col 209 "." received 8.480
col  62 "." received 8.424
col 229 "." received 8.350
col 211 "'" received 8.279
```

Worst rows for mass all attend to nearly the same punctuation-sink key set:

```text
row token     H_norm  fixed_rel  mass_rel  UTC_rel  oracle_rel  k_mass  k_oracle  top key
 86 believed   0.584     0.1109    1.0024   0.7266      0.1676      16        52  75 "."
108 believed   0.580     0.1030    0.9696   0.7385      0.1617      16        49  75 "."
254 una        0.589     0.1818    0.8271   0.5452      0.1682      18        68  75 "."
145 morality   0.583     0.2543    0.8930   0.7111      0.1665      15        87  75 "."
```

Group-level budget diagnosis:

```text
group  rows  H_mean  mass mean k  UTC mean k  oracle mean k  mass max_rel  oracle max_rel
high     12   0.920       138.17      117.58          71.00        0.0798          0.1683
low     244   0.740        60.40       61.18          63.02        1.0024          0.1685
```

Interpretation:

1. This is not a generic failure of the decomposition or extraction; the
   restricted oracle still controls the worst-row error at a matched mean
   budget.
2. The failure is a real-data allocation pathology: mass over-spends on a
   small high-entropy group that is already safe, while some lower-entropy
   punctuation-sink rows receive `k ~= 15-20` even though the oracle needs
   `k ~= 50-90`.
3. hybrid-b inherits the failure because its low-entropy group is left as
   mass-only. The design assumption "mass is reliable inside the low-entropy
   group" is false for this head.
4. The fixed-k baseline wins here because it provides a crude but effective
   per-row floor (`k=64`) for those punctuation-sink rows.

Updated real-data lesson: the BERT stage needs a floor or risk guard, not just
entropy grouping. Candidate next baselines: clipped adaptive allocation
(`k_min` floor), low-entropy worst-row guard, or a sink-aware diagnostic feature
such as attention concentration on high-received-mass punctuation keys.

## Guard variants: floor bandaid vs relative-score mechanism fix

The L11 h9 diagnosis (punctuation-sink rows) has a layer beneath the sink
structure: the failing rows combine large centroid displacement (C ~ 3.4)
with SMALL output norm (||o|| ~ 0.3-0.4 vs ~2.0 for typical rows), and our
acceptance target is RELATIVE error while both mass (delta) and the original
UTC score (delta * ||mu_R_hat - mu_S||) are ABSOLUTE quantities. Neither
signal knows ||o||. Synthetic data never exposed this because iid values make
||o|| homogeneous across rows; real sink rows are exactly the
small-output-norm rows, so the mismatch binds there. Note also from the
diagnosis: mass simultaneously OVERSPENDS on the 12 high-entropy rows (mean
k=138 vs oracle 71) while starving the sink rows — both sides wrong.

Mechanism fix, still deployable (mu_S is the sparse output itself,
retained-side, free):

```text
score_rel(k) = delta(k) * ||mu_R_hat(k) - mu_S(k)|| / (||mu_S(k)|| + eta)
```

Head-to-head at matched mean k=64 (`scripts/stage2_guard_variants.py`):

```text
                    L11 h9 (failure head)   L0 h4 (success head)
variant             gap closed              gap closed
mass                -1.215                  -0.344
mass+floor32        -0.421                  -0.421
UTC (abs)           -0.714                   0.527
UTC-rel              0.869                   0.360
UTC-rel+floor32      0.869                   0.360
oracle               1.000                   1.000
```

Findings:

1. **UTC-rel fixes the failure head almost completely** (-0.71 -> +0.87,
   near-oracle) and is the ONLY variant positive on both heads.
2. The floor guard fails its stated goal: mass+floor32 is still worse than
   fixed on both heads, and the floor adds nothing on top of UTC-rel. The
   mechanism fix dominates the robustness bandaid (at this budget/floor).
3. Cost of the fix: on the diffuse success head, UTC-rel keeps a solid gain
   but gives up part of UTC-abs's margin (0.53 -> 0.36). Plausible cause:
   ||mu_S(k)|| at small k is a noisy normalizer on diffuse rows. Trade
   accepted — a scorer that catastrophically fails on sink-structured heads
   is not deployable regardless of its diffuse-head margin.

Decision (SUPERSEDED — see "Stage 1 consistency patch" below): UTC-rel was
provisionally promoted to default here, with the prediction that the synthetic
suite would show "little change — ||o|| is homogeneous there." The consistency
check falsified both the promotion and the prediction: UTC-rel badly weakens
the high-entropy synthetic win (0.653 -> 0.109 at q_scale=0.25). Retained as a
record of a wrong call caught by the check that was ordered alongside it.

Thesis-level note: L11 h9 is not an anomaly that breaks the framework — it is
the framework's central claim materializing in real data. delta-only signals
fail precisely where value geometry (and output-norm structure) matters, and
real attention puts that failure in the MID-entropy band via sink rows. The
synthetic heuristic "value-awareness only matters at high entropy" is
therefore incomplete on real data: entropy is a useful grouping variable but
not a sufficient risk descriptor.

## Stage 1 consistency patch: UTC-abs vs UTC-rel on synthetic mixed regimes

Motivation: the BERT L11 h9 failure revealed a mismatch between the original
UTC score (absolute-error proxy) and the benchmark target (relative error).
The relative score is theoretically motivated by the target:

```text
absolute proxy: UTC-abs(k) = delta(k) * ||mu_R_hat(k) - mu_S(k)||
relative proxy: UTC-rel(k) = UTC-abs(k) / (||mu_S(k)|| + eta)
```

But because the correction was discovered post-hoc from a real-data failure
case, it must be checked back on the synthetic Stage 1 benchmark rather than
silently promoted.

Script:

```text
scripts/stage1_utc_rel_consistency.py
```

Mixed-regime benchmark, 3 seeds x budgets {20, 40, 60}:

```text
seed | budget | norm_cv | mass | UTC_abs | UTC_rel | hybrid_b_abs | hybrid_b_rel
   0 |     20 |   0.837 | 0.732 |   0.729 |   0.320 |        0.732 |        0.732
   0 |     40 |   0.837 | 0.834 |   0.805 |   0.733 |        0.856 |        0.856
   0 |     60 |   0.837 | 0.788 |   0.854 |   0.945 |        0.903 |        0.903
   1 |     20 |   0.881 | 0.855 |   0.711 |   0.578 |        0.895 |        0.666
   1 |     40 |   0.881 | 0.752 |   0.762 |   0.813 |        0.899 |        0.827
   1 |     60 |   0.881 | 0.772 |   0.841 |   0.948 |        0.880 |        0.888
   2 |     20 |   0.901 | 0.831 |   0.701 |   0.589 |        0.831 |        0.829
   2 |     40 |   0.901 | 0.789 |   0.750 |   0.757 |        0.876 |        0.833
   2 |     60 |   0.901 | 0.791 |   0.828 |   0.906 |        0.867 |        0.877
```

Mean gap closed across all 9 settings:

```text
mass          mean=0.794 min=0.732 max=0.855
UTC_abs       mean=0.776 min=0.701 max=0.854
UTC_rel       mean=0.732 min=0.320 max=0.948
hybrid_b_abs  mean=0.860 min=0.732 max=0.903
hybrid_b_rel  mean=0.824 min=0.666 max=0.903
```

Per-regime sanity at budget 40, seed 0:

```text
q_scale | H_mean | output_norm_CV | UTC_abs gap | UTC_rel gap
   0.25 | 0.9937 |          0.030 |       0.653 |       0.109
   0.50 | 0.9747 |          0.064 |       0.452 |       0.002
   1.00 | 0.8979 |          0.254 |      -0.066 |       0.720
   2.00 | 0.6475 |          0.418 |       0.144 |       0.895
   4.00 | 0.3051 |          0.334 |       0.880 |       1.024
```

Interpretation:

1. The relative normalization is NOT a drop-in replacement for Stage 1
   synthetic results. It improves mid/sharp regimes and loose budgets, but it
   badly weakens the original high-entropy synthetic win.
2. The earlier expectation that synthetic `||o||` would be row-homogeneous is
   only true inside the most diffuse per-regime datasets; mixed-regime
   synthetic data has large output-norm variation (`CV ~= 0.84-0.90`).
3. The original Stage 1 conclusion remains stronger with UTC-abs:
   `hybrid_b_abs` is still the best synthetic mixed-regime policy on average
   and is more stable at tight budgets.
4. The real-data lesson still stands: UTC-rel is a mechanism-matched repair
   for BERT heads with small-output-norm sink rows. But the consistency check
   prevents overclaiming it as a universal default.

Updated decision: keep **UTC-abs** as the Stage 1 synthetic baseline and carry
**UTC-rel** as a real-data robustness candidate / relative-risk scorer. The
next BERT sweep should report both rather than silently replacing one with the
other.

Addendum to the consistency patch (review note): codex's own `norm_CV` column
appears to BE the selector variable for abs-vs-rel. Per-regime at budget 40:

```text
output-norm CV <= 0.06 (diffuse synthetic)  -> UTC-abs wins big (0.65 vs 0.11)
output-norm CV >= 0.25 (mid/sharp/real)     -> UTC-rel wins big (e.g. q=1.0:
                                               -0.066 -> +0.720)
```

Two consequences:

1. **The old mid-regime UTC failure is partially rehabilitated.** The Stage 1
   finding "UTC fails at q_scale=1.0 despite predictor corr 0.96" now has a
   larger identified cause: the abs-score/relative-target mismatch (norm CV =
   0.254 there). With UTC-rel the same regime scores +0.72. The
   "predictor corr != allocation quality" lesson stands, but the specific
   mid-regime failure was more mismatch than mu_R-estimation bias — a
   revision of the earlier diagnosis.
2. **Candidate cheap selector, to be tested on the multi-doc BERT sweep:**
   population CV of ||mu_S(k)|| (retained-side, free, ~ CV of ||o||) picks the
   scorer: low CV -> abs, high CV -> rel. Caveat from the mixed-population
   rows: at tight budgets (mean k=20) rel underperforms even at high CV —
   the normalizer ||mu_S(k)|| is itself noisy at small k, so budget level is
   a second selector variable. Pre-registered expectation for the BERT sweep:
   rel wins where norm-CV is high AND budget is not tight; abs wins otherwise.

Project-level pattern worth naming (third instance): every design choice so
far has turned out regime/structure-dependent WITH a cheap observable
selector — signal choice (entropy grouping), budget split (mass tau),
and now scorer choice (output-norm CV). None of the choices is universal;
all of the selectors are Q,K-or-retained-side cheap.

## Held-out BERT abs-vs-rel sweep: selector hypothesis stress test

Goal: test the pre-registered expectation from the Stage 1 consistency patch:

```text
norm-CV high AND budget not tight -> UTC-rel should beat UTC-abs
otherwise                         -> UTC-abs should be safer
```

This is a held-out check relative to the L11 h9 discovery case: three new
20 Newsgroups documents (`sci.space`, `rec.autos`, `sci.crypt`), `max_tokens =
128`, layers {0, 5, 11}, all 12 heads per layer. Script:

```text
scripts/stage2_bert_abs_rel_sweep.py
```

Saved outputs:

```text
results/bert/stage2_abs_rel_sweep_b8_16_24.csv
results/bert/stage2_abs_rel_sweep_b16_32_48.csv
```

Strict comparability is hard in real BERT because many sharp heads saturate
(the oracle and/or proxy cannot spend the requested budget). Therefore the
summary below uses only rows where fixed, oracle, mass, UTC-abs, and UTC-rel
all land within 2 tokens of the target mean budget.

Tight-budget sweep (`k = 8, 16, 24`; N=128):

```text
total head-budget rows: 324
comparable rows:        78
decisive rows:          77
pre-registered selector accuracy: 0.610

budget | comparable n | rel wins | mean UTC-abs gap | mean UTC-rel gap | mean mu_S norm CV
     8 |           57 |       18 |          -0.076 |          -3.114 |            0.662
    16 |           17 |        9 |           0.155 |          -0.287 |            0.331
    24 |            4 |        3 |           0.349 |           0.579 |            0.268

layer | comparable n | rel wins | mean UTC-abs gap | mean UTC-rel gap | mean mu_S norm CV
    0 |           42 |       15 |           0.430 |          -0.159 |            0.267
    5 |           11 |        9 |          -0.466 |          -0.276 |            0.635
   11 |           25 |        6 |          -0.530 |          -6.814 |            1.050
```

Non-tight probe (`k = 16, 32, 48`; N=128):

```text
total head-budget rows: 324
comparable rows:        20
decisive rows:          19
pre-registered selector accuracy: 0.474

budget | comparable n | rel wins | mean UTC-abs gap | mean UTC-rel gap | mean mu_S norm CV
    16 |           17 |        9 |           0.155 |          -0.287 |            0.331
    32 |            3 |        3 |           0.150 |           0.758 |            0.309

layer | comparable n | rel wins | mean UTC-abs gap | mean UTC-rel gap | mean mu_S norm CV
    0 |           16 |        8 |           0.517 |          -0.159 |            0.276
    5 |            3 |        3 |          -1.374 |           0.285 |            0.488
   11 |            1 |        1 |          -1.059 |          -0.906 |            0.667
```

Largest rel-over-abs wins look like the L11 h9 repair mechanism, usually in
middle-layer heads where UTC-abs is badly mis-scaled:

```text
doc=13 L5 h2 k=16: UTC-abs=-1.729, UTC-rel= 0.266, CV=0.597
doc=13 L5 h4 k=32: UTC-abs=-0.815, UTC-rel= 0.951, CV=0.347
doc=37 L5 h2 k=16: UTC-abs=-1.578, UTC-rel=-0.362, CV=0.520
```

Largest abs-over-rel wins are mostly late-layer tight-budget cases where the
`||mu_S(k)||` normalizer becomes a noisy small-k denominator:

```text
doc=37 L11 h4  k=8: UTC-abs=-0.371, UTC-rel=-18.690, CV=1.252
doc=37 L11 h10 k=8: UTC-abs=-0.645, UTC-rel=-18.260, CV=1.259
doc=13 L11 h4  k=8: UTC-abs=-0.319, UTC-rel=-17.110, CV=1.115
```

Interpretation:

1. The pre-registered simple selector **does not pass**. Its accuracy is weak
   (0.61 on tight budgets, 0.47 on the wider probe), and `mu_S` norm-CV alone
   is not monotone: high CV can mean "relative normalization needed" or "the
   denominator is too noisy to trust."
2. The budget part of the hypothesis survives directionally. At `k=8`, UTC-rel
   is often catastrophic; at `k=32`, the few comparable cases all favor rel.
   But real BERT saturation leaves too few non-tight comparable rows for a
   strong claim.
3. Layer/structure matters as much as CV. Layer 5 heads often benefit from
   rel; layer 11 tight-budget heads are where rel can explode. This suggests
   the selector needs a denominator-stability condition, not just denominator
   variation.
4. The earlier project-level pattern needs a stricter wording: cheap selectors
   exist for some choices (entropy for regime grouping, mass for budget
   transfer), but the abs-vs-rel selector is **not solved yet**. The held-out
   sweep prevented a tempting overfit from becoming project doctrine.

Updated next question: characterize when `||mu_S(k)||` is a reliable
normalizer. Candidate diagnostics: minimum/quantile of `||mu_S(k)||`, CV
stability as k changes, and whether the head is a late-layer punctuation-sink
head. Until this is understood, BERT sweeps should report UTC-abs and UTC-rel
side by side rather than choosing one automatically.

## UTC-rel-hat: fixing the denominator instead of selecting between scorers

The held-out sweep concluded the missing variable is denominator stability.
That points at a repair we already have the parts for: UTC-rel divides by
||mu_S(k)||, which is a poor estimate of ||o|| exactly where rel explodes
(small k, sink rows whose retained values are near-zero punctuation). But UTC
already estimates mu_R, so a better denominator is free:

```text
o_hat(k) = (1 - delta(k)) * mu_S(k) + delta(k) * mu_R_hat(k)
UTC-rel-hat(k) = delta(k) * ||mu_R_hat(k) - mu_S(k)|| / (||o_hat(k)|| + eta)
```

Conceptual reading: the oracle's acceptance criterion IS relative error.
UTC-abs was a proxy that works only when output norms are homogeneous;
UTC-rel was a broken estimator of the right quantity (collapsing
denominator); UTC-rel-hat is simply a better estimator of the quantity the
oracle actually thresholds.

Probe on all five previously-measured anchor cases
(`scripts/stage2_rel_hat_probe.py`, gap closed):

```text
case                            |    abs |    rel | rel-hat
synth q=0.25 b40 (rel failure)  |  0.668 |  0.111 |  0.900
synth q=1.0  b40 (rel repair)   | -0.066 |  0.711 |  0.711
BERT L11 h9 k=64 (rel repair)   | -0.714 |  0.869 |  0.930
BERT L11 h9 k=16 (tight budget) | -0.273 | -3.053 |  0.461
BERT L0  h4 k=64 (abs ahead)    |  0.527 |  0.360 |  0.756
```

rel-hat >= max(abs, rel) on all five anchors, strictly better on four,
including beating abs on abs's home ground. If this holds up, the abs-vs-rel
selector problem dissolves: there is no selection to make, just one better
estimator.

Discipline note: these five cases are anchors, i.e. the cases we stared at
while designing rel-hat. NOT evidence of generality. Pre-registered
expectations for the decisive tests:

```text
1. held-out BERT sweep (3 docs x 3 layers x 12 heads, both budget grids):
   rel-hat >= max(abs, rel) on the majority of comparable rows, and NO
   catastrophic failures (gap closed < -1) anywhere.
2. synthetic mixed-regime consistency suite: rel-hat should not lose to
   hybrid_b_abs's mean by more than a few points.
```

Known candidate failure mode to watch: o_hat can suffer cancellation
((1-delta)*mu_S and delta*mu_R_hat near-opposite), inflating the score where
the true output is not actually small. If the sweep shows rare large misses,
check for cancellation rows first.

## UTC-rel-hat validation on held-out BERT and synthetic mixed regimes

The decisive tests from the previous section have now been run. `UTC-rel-hat`
was added to the shared Stage 1 scorer interface:

```text
scripts/stage1_evaluate_value_proxies.py
  utc_rel_hat_error_proxy_for_row
  calibrate_utc_rel_hat_for_mean_k
```

and integrated into the held-out BERT sweep:

```text
scripts/stage2_bert_abs_rel_sweep.py
```

### Held-out BERT: tight-budget stress test

Same held-out setup as the abs-vs-rel selector test: 3 new 20 Newsgroups
documents, `max_tokens=128`, layers {0, 5, 11}, all 12 heads per layer. The
tight-budget grid is the hardest case because this is where `UTC-rel` exploded
via an unstable `||mu_S(k)||` denominator.

Saved output:

```text
results/bert/stage2_abs_rel_hat_sweep_b8_16_24.csv
```

Summary:

```text
total head-budget rows: 324
comparable rows:        74
decisive rows:          73

rel-hat >= max(abs, rel): 66/74 = 0.892
rel-hat catastrophic failures (< -1): 0

budget | comparable n | mean UTC-abs | mean UTC-rel | mean UTC-rel-hat | rel-hat best
     8 |           57 |       -0.076 |       -3.114 |           0.610 | 50/57
    16 |           16 |        0.264 |       -0.282 |           0.786 | 15/16
    24 |            1 |        0.594 |        0.594 |           0.914 |  1/1
```

The catastrophic `UTC-rel` failures from the previous sweep are fixed:

```text
doc=37 L11 h4  k=8: UTC-abs=-0.371, UTC-rel=-18.690, UTC-rel-hat= 1.056
doc=37 L11 h10 k=8: UTC-abs=-0.645, UTC-rel=-18.260, UTC-rel-hat= 0.648
doc=13 L11 h4  k=8: UTC-abs=-0.319, UTC-rel=-17.110, UTC-rel-hat= 1.056
```

Largest rel-hat losses against the better of abs/rel are small:

```text
doc=37 L5 h8  k=8: best(abs,rel)=0.948, UTC-rel-hat=0.749
doc=17 L5 h11 k=8: best(abs,rel)=0.858, UTC-rel-hat=0.715
doc=17 L0 h1  k=8: best(abs,rel)=0.528, UTC-rel-hat=0.409
```

### Held-out BERT: wider-budget probe

Saved output:

```text
results/bert/stage2_abs_rel_hat_sweep_b16_32_48.csv
```

Fewer rows are strictly comparable because many real BERT heads saturate at
larger budgets, but the direction is consistent:

```text
total head-budget rows: 324
comparable rows:        17
decisive rows:          17

rel-hat >= max(abs, rel): 16/17 = 0.941
rel-hat catastrophic failures (< -1): 0

budget | comparable n | mean UTC-abs | mean UTC-rel | mean UTC-rel-hat | rel-hat best
    16 |           16 |        0.264 |       -0.282 |           0.786 | 15/16
    32 |            1 |        0.649 |        0.688 |           0.962 |  1/1
```

Interpretation for held-out BERT: the abs-vs-rel selector hypothesis failed,
but the stronger estimator hypothesis passed the first held-out test. Replacing
the unstable retained-only denominator `||mu_S||` with the UTC-estimated full
output denominator `||o_hat||` preserves the relative-error correction while
removing the late-layer tight-budget explosions.

### Synthetic mixed-regime consistency

The same rel-hat scorer was added to the synthetic mixed-regime consistency
script:

```text
scripts/stage1_utc_rel_consistency.py
```

Result on the original 3 seeds x budgets {20, 40, 60}:

```text
seed | budget | mass | UTC_abs | UTC_rel | UTC_rel_hat | hybrid_b_abs | hybrid_b_rel | hybrid_b_rel_hat
   0 |     20 | 0.732 |   0.729 |   0.320 |       0.909 |        0.732 |        0.732 |            0.732
   0 |     40 | 0.834 |   0.805 |   0.733 |       0.933 |        0.856 |        0.856 |            0.856
   0 |     60 | 0.788 |   0.854 |   0.945 |       0.974 |        0.903 |        0.903 |            0.903
   1 |     20 | 0.855 |   0.711 |   0.578 |       0.895 |        0.895 |        0.666 |            0.895
   1 |     40 | 0.752 |   0.762 |   0.813 |       0.920 |        0.899 |        0.827 |            0.933
   1 |     60 | 0.772 |   0.841 |   0.948 |       0.948 |        0.880 |        0.888 |            0.902
   2 |     20 | 0.831 |   0.701 |   0.589 |       0.935 |        0.831 |        0.829 |            0.831
   2 |     40 | 0.789 |   0.750 |   0.757 |       0.932 |        0.876 |        0.833 |            0.906
   2 |     60 | 0.791 |   0.828 |   0.906 |       0.957 |        0.867 |        0.877 |            0.889
```

Mean gap closed:

```text
mass               mean=0.794 min=0.732 max=0.855
UTC_abs            mean=0.776 min=0.701 max=0.854
UTC_rel            mean=0.732 min=0.320 max=0.948
UTC_rel_hat        mean=0.934 min=0.895 max=0.974
hybrid_b_abs       mean=0.860 min=0.732 max=0.903
hybrid_b_rel       mean=0.824 min=0.666 max=0.903
hybrid_b_rel_hat   mean=0.872 min=0.732 max=0.933
```

This is stronger than expected: rel-hat does not merely preserve the Stage 1
synthetic conclusion; as a pure UTC scorer it beats the previous mixed-regime
hybrid on average. The budget-delegated version improves over `hybrid_b_abs`
on two medium/loose settings and ties it on tight settings where the low-entropy
group remains the bottleneck.

Caveat on "gap closed > 1": the current restricted oracle is a threshold-
calibrated row-wise error-budgeted baseline within top-k-by-p, not a global
knapsack optimizer for exact total budget. A proxy allocation can occasionally
beat that calibrated curve on worst-row error. This should be read as "better
than the current restricted-oracle calibration," not as surpassing a true
global oracle.

Updated decision: `UTC-rel-hat` becomes the leading scorer candidate. Unlike
`UTC-rel`, it is not a single-head patch: it passes the held-out BERT stress
test, removes the denominator explosion failure mode, and improves the
synthetic mixed-regime benchmark. Next evidence needed: a larger BERT sweep
with more documents/layers and a report that separates comparable from
saturated heads.

## Stage 2B: larger BERT-base coverage sweep

Goal: test whether the `UTC-rel-hat` result survives broader real-attention
coverage, not just selected pilot heads.

Script:

```text
scripts/stage2b_bert_coverage_sweep.py
```

Design:

```text
model: bert-base-uncased
data: 10 held-out 20 Newsgroups documents
max tokens: 128
layers: all 12
heads: all 12 per layer
budgets: k = 8, 16, 32
methods: fixed, mass, UTC-abs, UTC-rel, UTC-rel-hat, restricted oracle
output: results/bert/stage2b_bert_coverage.csv
```

The script writes rows incrementally and can resume after interruption. Total
grid size:

```text
10 docs x 12 layers x 12 heads x 3 budgets = 4320 head-budget rows
```

### Main comparable-row result

As expected from earlier BERT runs, many sharp heads saturate: at a requested
budget, the oracle and/or proxy cannot spend the full mean budget. These rows
are not discarded from the CSV, but the main comparison below is only on
strictly comparable rows, where fixed, oracle, mass, UTC-abs, UTC-rel, and
UTC-rel-hat all land within 2 tokens of the target mean budget.

```text
total rows:       4320
comparable rows:   718  (16.6%)
non-comparable:   3602

UTC-rel-hat >= max(UTC-abs, UTC-rel): 591/718 = 0.823
UTC-rel-hat catastrophic failures (< -1): 0
UTC-rel-hat below fixed (gap closed < 0): 9/718
```

Gap-closed distribution on comparable rows:

```text
method       mean     median    p10      min       max
mass        -0.143   -0.031   -0.980   -5.917    1.005
UTC-abs     -0.054    0.044   -0.836   -5.741    0.992
UTC-rel     -0.973    0.296   -4.916  -48.846    1.201
UTC-rel-hat  0.676    0.699    0.328   -0.482    3.222
```

The key result is not just that rel-hat often wins; it removes the disaster
tail. The retained-denominator `UTC-rel` still has extreme failures, while
`UTC-rel-hat` has no `< -1` comparable failures in this larger sweep.

### By budget

```text
budget | comparable n | mean mass | mean abs | mean rel | mean rel-hat | rel-hat best-rate | rel-hat < 0
     8 |          607 |    -0.122 |   -0.065 |   -1.143 |        0.669 |             0.812 |          9
    16 |          105 |    -0.282 |   -0.022 |   -0.071 |        0.710 |             0.895 |          0
    32 |            6 |     0.189 |    0.473 |    0.520 |        0.771 |             0.667 |          0
```

The previous held-out stress-test pattern holds: tight budgets are where
`UTC-rel` is dangerous, and `UTC-rel-hat` repairs that regime most clearly.
There are few comparable rows at k=32 because many heads have already
saturated, so the k=32 row is encouraging but not high-powered evidence.

### By layer

```text
layer | comparable n | mean rel-hat | rel-hat best-rate | rel-hat < 0
    0 |          133 |        0.714 |             0.865 |          0
    1 |           88 |        0.635 |             0.841 |          0
    2 |           78 |        0.707 |             0.846 |          0
    3 |           62 |        0.668 |             0.839 |          0
    4 |           43 |        0.692 |             0.674 |          0
    5 |           28 |        0.752 |             0.821 |          1
    6 |           29 |        0.797 |             0.759 |          0
    7 |           16 |        0.831 |             0.750 |          0
    8 |           29 |        0.739 |             0.724 |          0
    9 |           34 |        0.772 |             0.735 |          0
   10 |           89 |        0.669 |             0.798 |          5
   11 |           89 |        0.490 |             0.910 |          3
```

Layer 11 remains structurally hardest (lowest mean rel-hat), but it also has
the highest best-rate because abs/rel fail even harder there. This matches the
punctuation-sink diagnosis: late layers create the denominator pathologies
that rel-hat was designed to fix, but those same heads remain difficult.

### Failure and win cases

Largest rel-hat losses vs the better of abs/rel are moderate, not catastrophic:

```text
doc=68 L11 h5 k=8: best=0.846, rel-hat=0.129, delta=-0.717
doc=13 L4  h6 k=8: best=1.131, rel-hat=0.445, delta=-0.686
doc=54 L8  h7 k=8: best=0.937, rel-hat=0.279, delta=-0.658
```

Largest rel-hat wins are exactly the intended repair cases: abs and rel both
mis-scale badly, while rel-hat recovers a usable score.

```text
doc=17 L10 h2 k=8: abs=-5.351, rel=-48.846, rel-hat=3.222
doc=17 L11 h11 k=8: abs=-5.741, rel=-19.786, rel-hat=2.409
doc=17 L10 h3 k=8: abs=-5.294, rel=-1.057, rel-hat=2.591
```

Note: gap closed can exceed 1 because the current "restricted oracle" is a
uniform-threshold calibration within the top-k-by-p family, not a true global
matched-budget min-max oracle. Small exceedances are expected; very large
exceedances should motivate a stricter oracle calibration in future work.

### Saturation accounting

Budget mismatch counts over all 4320 rows:

```text
oracle not at target budget:   3197
mass not at target budget:     1324
UTC-abs not at target budget:  2156
UTC-rel not at target budget:  2140
UTC-rel-hat not at target:     1886
```

Interpretation: saturation is not a nuisance detail; it is a dominant property
of real BERT attention at these sequence lengths and budgets. Stage 2B
therefore supports two separate claims:

1. On rows where matched-budget comparison is valid, `UTC-rel-hat` is the
   leading scorer candidate by a clear margin.
2. A complete real-attention story must also model saturation / budget
   non-binding regimes, because most head-budget rows fall there.

Current status: `UTC-rel-hat` graduates from pilot candidate to strong Stage
2B candidate, but not final method. Next improvements should either enlarge
the comparable set (different budget grid or sequence lengths) or upgrade the
oracle/budget protocol so saturated heads are handled explicitly rather than
only flagged.

### Review addendum: the hybrid apparatus is now (provisionally) dead weight

Two things in the validation tables worth naming explicitly:

1. **Pure UTC-rel-hat beats every hybrid variant in every synthetic config**:
   mean 0.934 (min 0.895) vs hybrid_b_abs 0.860 and hybrid_b_rel_hat 0.872 —
   and wrapping rel-hat INSIDE hybrid-b makes it worse (0.934 -> 0.872),
   because the delegation pins the budget split to mass's split, which is a
   worse allocator than rel-hat left alone. Reading: Stage 1's "budget
   delegation beats routing" conclusion was conditional on weak scorers. The
   entire selector/hybrid apparatus (entropy grouping, mass budget split,
   abs-vs-rel selection) was compensation for a poor estimator of the target
   quantity; with the estimator fixed, the compensations dissolve.
   Provisional — pending the larger BERT sweep — but if it holds, hybrid-b
   retires from "method" to "diagnostic scaffolding that located the real
   problem."
2. **A few rel-hat rows slightly beat the restricted oracle** (gap closed
   1.056 at doc=37 L11 h4 k=8). Not alarming: the uniform-epsilon oracle
   equalizes per-row relative error, which under integer k and bisection
   slack is not exactly the max_rel-optimal allocation at matched mean
   budget, so small (<~6%) overshoots are possible. Watch that these stay
   small; large overshoots would instead indicate a calibration bug.

Status language agreed: UTC-rel-hat is the **leading scorer candidate** — it
estimates the same relative-risk quantity the oracle thresholds while
avoiding the unstable retained-only denominator. Not yet the final method;
the next gate is a larger BERT sweep (more documents, all 12 layers, and
ideally a second model).

## Stage 2C: exact-budget BERT protocol

Motivation: Stage 2B exposed a protocol problem. Threshold-calibrated methods
often fail to spend the requested mean budget on real BERT heads, so only
718/4320 rows were strictly comparable. That is a useful saturation finding,
but it does not answer the engineering question "under the same token budget,
which scorer allocates better?"

New script:

```text
scripts/stage2c_bert_exact_budget.py
```

Protocol:

```text
model: bert-base-uncased
data: same 10 held-out 20 Newsgroups documents as Stage 2B
max tokens: 128
layers: all 12
heads: all 12 per layer
budgets: k = 8, 16, 32
rows: 4320 head-budget rows
output: results/bert/stage2c_bert_exact_budget.csv
```

For each method, build a per-row score curve `score(row, k)` over the
top-k-by-p family. Then:

1. Find the smallest method-specific risk threshold whose minimal per-row k
   choices fit under the requested total budget.
2. Force the leftover budget to be spent using the same method's one-step
   marginal score improvements.

This changes the comparison from threshold-calibrated to exact-budget: every
method spends exactly `num_rows * target_mean_k` tokens. The unrestricted set
selection problem is still out of scope; this remains restricted to top-k-by-p.

Integrity check:

```text
rows: 4320
docs: 10
layers: 0..11
budgets: 8, 16, 32

budget mismatches:
fixed       0
oracle      0
mass        0
UTC-abs     0
UTC-rel     0
UTC-rel-hat 0
```

### Main exact-budget result

```text
rel-hat <= min(abs, rel): 3539/4320 = 0.819
rel-hat <= mass:          4273/4320 = 0.989
rel-hat worse than fixed:   21/4320
```

Gap-closed distribution over all 4320 rows:

```text
method       mean gap  median gap  p10 gap   mean max_rel
mass            0.071       0.245   -0.980        0.6036
UTC-abs         0.017       0.208   -1.090        0.6140
UTC-rel         0.380       0.813   -0.179        0.4997
UTC-rel-hat     0.790       0.838    0.541        0.3589
```

Interpretation: once the budget is forced to match, `UTC-rel-hat` remains the
leading scorer candidate by a clear margin. The Stage 2B result was not just
an artifact of filtering to comparable threshold rows. The old `UTC-rel`
improves under exact-budget compared with Stage 2B because denominator
explosions no longer automatically imply underspending, but it is still much
worse than `UTC-rel-hat`.

### By budget

```text
k=8:  n=1440, mass=-0.074, abs=-0.114, rel=-0.261, rel-hat=0.677, rel-hat<fixed=18
k=16: n=1440, mass= 0.004, abs=-0.063, rel= 0.553, rel-hat=0.801, rel-hat<fixed= 2
k=32: n=1440, mass= 0.282, abs= 0.227, rel= 0.848, rel-hat=0.892, rel-hat<fixed= 1
```

Tight budgets remain the hardest setting. Most of the remaining rel-hat
failures occur at k=8, which is now a real method failure signal rather than a
budget-comparability artifact.

### By layer

```text
layer | mean rel-hat gap | rel-hat <= abs/rel | rel-hat < fixed
    0 |            0.779 |              0.903 |               1
    1 |            0.776 |              0.897 |               0
    2 |            0.793 |              0.883 |               0
    3 |            0.778 |              0.850 |               0
    4 |            0.773 |              0.756 |               1
    5 |            0.841 |              0.769 |               1
    6 |            0.852 |              0.767 |               1
    7 |            0.893 |              0.767 |               0
    8 |            0.803 |              0.742 |               0
    9 |            0.824 |              0.811 |               2
   10 |            0.719 |              0.828 |               8
   11 |            0.651 |              0.858 |               7
```

Late layers remain structurally hardest. This agrees with the earlier sink-row
diagnosis: `UTC-rel-hat` repairs the denominator pathology, but it does not
make late-layer value geometry easy.

### Worst exact-budget rel-hat rows

```text
doc=39 L11 h2 k=8:  gap=-2.473, fixed=1.1280, oracle=0.9291, rel-hat=1.6199
doc=37 L10 h3 k=8:  gap=-1.999, fixed=1.0788, oracle=0.7904, rel-hat=1.6551
doc=17 L6  h1 k=8:  gap=-1.360, fixed=0.7766, oracle=0.6021, rel-hat=1.0139
doc=17 L10 h9 k=16: gap=-1.074, fixed=0.1828, oracle=0.1215, rel-hat=0.2488
doc=17 L10 h9 k=8:  gap=-1.070, fixed=0.3136, oracle=0.1926, rel-hat=0.4431
```

These are the next diagnostic targets. They are no longer explainable as
"method did not spend the budget"; all methods spent exactly the same budget.

Current status: Stage 2C repairs the main evaluation-protocol flaw from Stage
2B. The project can now say: under an exact matched-budget protocol on 4320
BERT head-budget rows, `UTC-rel-hat` is the leading cheap value-aware scorer,
but its remaining 21 below-fixed failures should be diagnosed before calling
it a final method.

### Review addendum on Stage 2B + proposed Stage 2C protocol

Independent recomputation from the CSV confirms all Stage 2B summary numbers
(718 comparable; 0.823 best-rate; 0 catastrophic; 9 below fixed; means
mass=-0.143, abs=-0.054, rel=-0.973, rel-hat=+0.676). Two structural notes:

1. On comparable rows, mass and UTC-abs are on average WORSE than fixed —
   on real BERT, rel-hat is the only method with positive mean gap closed.
   Even rel-hat's single worst row (-0.48) beats both alternatives on that
   same row (abs -2.02, rel -3.30).
2. Comparable rows concentrate in L0/L1/L10/L11; middle layers are almost
   entirely saturated — the comparable subset is exactly the diffuse-layer
   population from the regime census.

Proposed Stage 2C (global-epsilon protocol): saturation is not noise to be
modeled — it is the per-head matched-budget protocol misclassifying good news
("this head compresses for free") as "not comparable." Replace per-head budget
calibration with ONE model-wide epsilon on the rel-hat score: every row in
every head/layer takes the smallest k with score <= epsilon; sweep epsilon to
get a cost-error curve (total retained tokens vs max/mean relative error);
baselines are fixed-k and mass curves at the same total cost. Saturated heads
then automatically spend less and diffuse heads more — cross-HEAD budget
transfer, the project's recurring motif one level up (rows -> regime groups ->
heads/layers). This is also the deployment-shaped interface (one quality knob)
and the natural interface for any future kernel work.

Pre-registered expectations for Stage 2C: (a) the rel-hat cost-error curve
Pareto-dominates fixed-k across the budget axis; (b) middle-layer savings are
automatically transferred to layer 0/1/10/11; (c) all 4320 head-rows enter
the accounting — no "not comparable" category remains.

### Review addendum on Stage 2C

Independent recomputation from the CSV confirms all summary numbers (3539/4320
= 0.819 vs min(abs,rel); 4273/4320 vs mass; 21 below fixed; means mass 0.071 /
abs 0.017 / rel 0.380 / rel-hat 0.790; per-row budget mismatch 0). Protocol
assessment: exact-budget is the correct instrument and retro-validates Stage
2B — the "comparable-row selection bias" threat is now excluded by
construction. Exact-budget and the earlier global-epsilon proposal are duals
(fix cost / measure error vs fix quality / measure cost); the per-HEAD exact
protocol still does not pool budget ACROSS heads, so the cross-head transfer
curve remains open as the deployment-facing follow-up.

Two additional observations from the CSV:

1. Distributional robustness: rel-hat's 10th-percentile gap is +0.541 (vs abs
   -1.09, mass -0.98) — the advantage is not mean-driven.
2. UTC-rel is half-rescued by the protocol (-0.973 -> +0.380): its earlier
   catastrophes were largely allocation runaway (exploding scores hoarding
   budget under threshold calibration), which forced budgets cap. Consistent
   with the original mechanism diagnosis.

Failure profile of the 21 rows (rel-hat worse than fixed): 18/21 at k=8,
15/21 in L10/L11. Priority pattern for diagnosis: rows like doc17 L0 h11 k=8
where abs is fine (+0.50) but BOTH relative variants fail (rel -11.9, rel-hat
-0.78) point at systematic denominator misestimation of ||o|| at tiny k —
exactly where the pre-registered o_hat-cancellation failure mode should be
checked first (compare ||o_hat(k)|| vs true ||o|| on these rows).

### Stage 2C failure diagnosis: the 21 rel-hat below-fixed rows

New script:

```text
scripts/stage2c_diagnose_rel_hat_failures.py
output: results/bert/stage2c_rel_hat_failure_diagnosis.csv
```

Goal: diagnose the 21 exact-budget cases where `UTC-rel-hat` is worse than
fixed-k. Since Stage 2C forces every method to spend the same budget, these
are real method failures rather than comparability artifacts.

Method: for each failing head-budget case, reconstruct the BERT head, recompute
exact-budget allocations, find the query row that attains `UTC-rel-hat`'s max
relative error, and compare:

```text
k_rel_hat vs k_oracle vs k_fixed
true relative error at that row
proxy score / true oracle score
proxy_abs / true_abs
||o_hat|| / ||o||
```

Main finding:

```text
failure head-budget rows: 21
by budget: {8: 18, 16: 2, 32: 1}
by layer:  {0: 1, 4: 1, 5: 1, 6: 1, 9: 2, 10: 8, 11: 7}

worst-row k relation:
rel_hat < oracle: 21
rel_hat = oracle:  0
rel_hat > oracle:  0

median ratios on worst rows:
score_proxy / true_score = 0.721
proxy_abs / true_abs     = 0.807
||o_hat|| / ||o||        = 1.063
```

So the 21 failures are all **starvation failures**: the query row that
determines the head's max error receives fewer tokens under rel-hat than under
the restricted oracle. The earlier suspected `o_hat` cancellation / budget
hoarding mode is not the dominant mechanism here. In fact the opposite happens:
rel-hat tends to think the risky row is safer than it is and transfers budget
elsewhere.

The submechanisms split evenly:

```text
mild_score_miscalibration_starvation: 7
denom_overestimate_starvation:        7
numerator_underestimate_starvation:   7
```

Interpretation:

- **denom_overestimate_starvation**: `||o_hat||` is too large relative to
  `||o||`, so the relative proxy is too small even when the absolute numerator
  is roughly correct.
- **numerator_underestimate_starvation**: UTC underestimates the value
  displacement itself; the denominator is not the primary issue.
- **mild_score_miscalibration_starvation**: neither factor is catastrophically
  wrong, but tight budgets make a moderate score error enough to starve the
  max-risk row.

Worst examples:

```text
doc=39 L11 h2 k=8:  q=38 ("t"),        k_relhat/oracle/fixed=2/4/8,  score/true=0.577, num=1.053, denom=1.823
doc=37 L10 h3 k=8:  q=49 ("drug"),     k_relhat/oracle/fixed=2/4/8,  score/true=0.511, num=1.007, denom=1.970
doc=17 L6  h1 k=8:  q=0  ("[CLS]"),    k_relhat/oracle/fixed=3/6/8,  score/true=0.528, num=0.804, denom=1.524
doc=17 L10 h9 k=8:  q=62 ("-"),        k_relhat/oracle/fixed=4/11/8, score/true=0.772, num=0.543, denom=0.704
doc=54 L4  h0 k=8:  q=47 ("kurdish"),  k_relhat/oracle/fixed=1/4/8,  score/true=0.504, num=0.349, denom=0.693
```

The "abs works but rel variants fail" pattern that motivated this diagnostic
exists, but it is rare:

```text
UTC_abs positive among the 21 failures: 1/21
```

That one case (doc17 L0 h11 k=8) is a clean denominator-overestimate example:
`proxy_abs / true_abs = 1.18` but `||o_hat|| / ||o|| = 1.63`, so rel-hat
scales the risk down too far. The broader failure set is not a single
denominator pathology; it is a tight-budget starvation family with three
subtypes.

Budget-flow note: compared with the restricted oracle, rel-hat redistributes
substantial budget inside failing heads (median missing/extra token mass =
105 tokens across the 128 query rows). The failed worst row has median deficit
3 tokens vs oracle, with extreme deficits up to 28. The next diagnostic should
therefore inspect where those extra tokens go: are they protecting rows with
genuinely high proxy-but-low-true risk, or are they wasted on a small set of
proxy outliers?

Current status: rel-hat remains the leading scorer candidate, but its known
boundary is now sharper. It can starve max-risk rows under tight exact budgets,
especially in late BERT layers. The most plausible fix is not another global
denominator choice, but a guard against row starvation / proxy underestimation
near the lower tail of allocated k.

### Review addendum on the failure diagnosis

Verified from the CSV: 7/7/7 class split; 21/21 are worst-row starvation
(k_rel_hat < k_oracle, never overspending); denom_ratio ||o_hat||/||o|| mean
1.15 (>1 in 14/21), numerator_ratio mean 0.79 (<1 in 14/21).

Three notes on top of the diagnosis:

1. The pre-registered failure mode inverted in sign but not in location: the
   weak point IS the denominator estimate, but instead of o_hat spuriously
   cancelling (norm too small -> score inflated -> overspend), the TRUE output
   has structure/cancellation that the uniform-tail estimate cannot see
   (||o_hat|| too large -> risk underestimated -> starvation).
2. The three failure classes share one root and one direction. Uniform-tail
   smoothing pulls mu_R_hat toward the global mean, which simultaneously
   (a) fails to reproduce the true output's cancellation (denominator biased
   up) and (b) smooths away the true weighted tail displacement (numerator
   biased down). Both biases push the score DOWN — which is why all 21
   failures are starvation and none are overspending. The failure is
   one-sided, so the fix can be a one-sided guard rather than a symmetric
   recalibration.
3. Guard proposal (pre-registered): trust-gated protection. The trust signal
   was reserved in the original line-A design and is Q,K-side free — the TV
   distance of tail weights from uniform. Rows with high tail-TV (structured
   tail, estimate untrustworthy) receive a protective k floor; low-TV rows run
   plain rel-hat. Expectations: fixes the majority of the 21 rows while
   leaving the other 4299 essentially unchanged (mean gap ~0.790 must not
   materially drop). Baseline to beat: a plain small floor (k_min ~ 4) at
   tight budgets — note the earlier "floors fail" result (floor=32 at k=64,
   attached to mass/rel) does NOT generalize to this setting and should not
   be pattern-matched against.

## Stage 3 (pre-registration, written BEFORE running): cross-model validation on GPT-2

Motivation: the user raised the right worry at the right time — the whole
abs -> rel -> rel-hat iteration has run inside one environment
(bert-base-uncased + 20 Newsgroups). Document-level holdout cannot detect
"overfitting to BERT-ness" (its sink structure, entropy profile, value
statistics). The strongest available test is to move the ENTIRE method stack,
unchanged, to a different model family: GPT-2 small (causal attention —
different masking, different sink structure, different regime mixture).

Setup (decided before seeing any GPT-2 number):
- gpt2 (12 layers, 12 heads, d_head 64), 3 HELD-OUT 20NG documents
  (select_documents skip=11 — never used in any prior stage), max_tokens=128.
- Causal adaptation: analyze query rows with support >= 65 (row index >= 64)
  so pruning is non-trivial; each row's UTC tail estimate uses ITS OWN causal
  support (prefix value sums — note these are exactly the prefix sums a causal
  kernel would maintain anyway).
- Layers {0, 5, 11}, all heads, budgets {8, 16, 32}, exact-budget protocol
  (per-row k capped at support; every method spends the identical total).
- Methods unchanged: fixed / mass / UTC-abs / UTC-rel / UTC-rel-hat / oracle.
  No formula, threshold, or parameter is modified for GPT-2.

Pre-registered expectations:
1. rel-hat is the only scorer with positive mean gap closed (as on BERT).
2. rel-hat >= max(abs, rel) on the majority of head-budget rows.
3. No catastrophic rel-hat failures (gap < -1).
4. Failure rows, if any, remain one-sided starvation (consistent with the
   uniform-tail-bias mechanism, which is model-agnostic).

If these fail, the conclusion is that the scorer ladder is BERT-specific and
the "formula quality" worry is substantiated — that outcome would be recorded
with the same prominence as a pass.

### Stage 3 results (GPT-2 cross-model validation)

Script: `scripts/stage3_gpt2_cross_model.py`; data:
`results/gpt2/stage3_gpt2_exact_budget.csv` (324 head-budget rows).
Extraction anchored at ~1e-6 on all layers (causal recompute vs model
attentions). Method stack unchanged from BERT.

```text
method       mean gap  median  p10
mass            0.017   0.265  -1.186
UTC_abs         0.309   0.602  -0.842
UTC_rel         0.579   0.837   0.185
UTC_rel_hat     0.828   0.881   0.642

rel-hat >= max(abs, rel): 250/324 = 0.772   [expectation 2: PASS]
rel-hat catastrophic (< -1): 0              [expectation 3: PASS]
rel-hat below fixed: 3/324 = 0.9%           (BERT: 0.5%, same magnitude)
```

Pre-registration accounting:
1. "rel-hat is the ONLY positive-mean scorer" — **FAILED as worded**: all
   four methods have positive means on GPT-2. The finding "baselines are on
   average worse than fixed on real data" was BERT-specific (its punctuation
   sink structure punishes delta-family signals particularly hard) and does
   not transfer. This does not weaken rel-hat: the full ordering
   rel-hat > rel > abs > mass is preserved, and rel-hat's distribution is
   even more robust than on BERT (p10 0.642 vs 0.541).
2-3. PASS as registered.
4. One-sidedness of the 3 below-fixed rows: not yet checked (per-row k not
   stored in this CSV) — open follow-up.

Verdict on the "are we overfitting the formula ladder?" question: the
abs -> rel -> rel-hat ladder transfers to a different model family with zero
modification — causal masking, different sink structure, different regime
mixture — with every rung in the same order and the leader's margin larger.
The strongest available test does not support the overfitting hypothesis.
What WAS BERT-specific is the baselines' failure, not rel-hat's advantage.

## Stage 4A (pre-registration, written BEFORE running): metric boundary via W_O projection

Motivation: all previous wins are measured in attention-head output space:

```text
relative error = ||o_head - sparse(o_head)|| / ||o_head||
```

This is the right internal metric for the decomposition, but it is not yet the
model-facing metric. Transformer blocks see the concatenated head outputs after
the output projection `W_O`, then residual stream, MLP, and eventually logits.
If rel-hat's advantage disappears immediately after the head contribution is
projected by `W_O`, the project has been optimizing a local metric that may not
matter downstream.

Stage 4A is the cheapest falsification test: keep the exact same allocations
and exact-budget protocol, but evaluate row error after each head's own
`W_O` slice.

Setup:

```text
BERT: same 10 docs x all 12 layers x all 12 heads x budgets {8,16,32}
GPT-2: same 3 held-out docs x layers {0,5,11} x all heads x budgets {8,16,32}
methods: fixed / mass / UTC-abs / UTC-rel / UTC-rel-hat / restricted oracle
protocol: exact-budget, head-local, top-k-by-p family
metric: max relative error after applying the per-head output-projection slice
```

Important detail: the restricted oracle must be recomputed under the projected
metric, not reused from attention-output space. Otherwise the "gap closed"
denominator would refer to the wrong optimization target.

Pre-registered expectations:

1. The method ladder should broadly preserve order under W_O-projected error:
   `UTC-rel-hat` remains the leading scorer on both BERT and GPT-2.
2. rel-hat's mean gap may move, but should stay clearly positive and above
   `UTC-rel` / `UTC-abs` / `mass`.
3. If the ranking collapses or rel-hat becomes worse than fixed on many rows,
   that is a real metric-boundary failure: attention-output approximation is
   not enough, and later logit/perplexity work should be deprioritized until
   the metric is corrected.

This is not yet logit KL or perplexity. It is the minimal bridge test from
the decomposition's local metric to the model's hidden-space contribution.

### Stage 4A results: W_O-projected metric boundary

Script:

```text
scripts/stage4_metric_boundary_wo.py
outputs:
  results/metric_boundary/stage4a_bert_wo_projected.csv
  results/metric_boundary/stage4a_gpt2_wo_projected.csv
```

Extraction / protocol:

```text
BERT rows: 4320 (same 10 docs x 12 layers x 12 heads x budgets 8/16/32)
GPT-2 rows: 324 (same 3 docs x layers 0/5/11 x 12 heads x budgets 8/16/32)
protocol: exact-budget, head-local
oracle: recomputed under W_O-projected relative error
```

Implementation detail:

```text
BERT Linear output: delta_y = delta_head @ W[:, head_slice].T
GPT-2 Conv1D output: delta_y = delta_head @ W[head_slice, :]
```

#### BERT: output-space vs W_O-projected

```text
BERT attention-output metric:
method       mean    median   p10     below fixed
mass         0.071   0.245   -0.980   1709
UTC-abs      0.017   0.208   -1.090   1759
UTC-rel      0.380   0.813   -0.179    505
UTC-rel-hat  0.790   0.838    0.541     21

BERT W_O-projected metric:
method       mean    median   p10     below fixed
mass         0.060   0.226   -0.999   1728
UTC-abs      0.006   0.176   -1.062   1803
UTC-rel      0.397   0.796   -0.081    492
UTC-rel-hat  0.755   0.809    0.465     34
```

Rel-hat best-rate vs `max(abs, rel)`:

```text
attention-output: 3539/4320 = 0.819
W_O-projected:    3494/4320 = 0.809
```

By budget under W_O projection:

```text
k=8:  rel-hat mean=0.641, p10=0.320, below-fixed=25
k=16: rel-hat mean=0.765, p10=0.507, below-fixed=7
k=32: rel-hat mean=0.860, p10=0.657, below-fixed=2
```

#### GPT-2: output-space vs W_O-projected

```text
GPT-2 attention-output metric:
method       mean    median   p10     below fixed
mass         0.017   0.265   -1.186    130
UTC-abs      0.309   0.602   -0.842     77
UTC-rel      0.579   0.837    0.185     27
UTC-rel-hat  0.828   0.881    0.642      3

GPT-2 W_O-projected metric:
method       mean    median   p10     below fixed
mass         0.019   0.272   -1.100    134
UTC-abs      0.251   0.531   -0.827     90
UTC-rel      0.580   0.774    0.101     28
UTC-rel-hat  0.702   0.797    0.313     11
```

Rel-hat best-rate vs `max(abs, rel)`:

```text
attention-output: 250/324 = 0.772
W_O-projected:    227/324 = 0.701
```

By budget under W_O projection:

```text
k=8:  rel-hat mean=0.614, p10=0.184, below-fixed=5
k=16: rel-hat mean=0.709, p10=0.337, below-fixed=4
k=32: rel-hat mean=0.783, p10=0.472, below-fixed=2
```

#### Pre-registration accounting

1. **PASS with attenuation**: the ladder broadly preserves order after W_O.
   Rel-hat remains the leading scorer on both BERT and GPT-2.
2. **PASS but weaker on GPT-2**: rel-hat stays clearly positive and above the
   alternatives, but the margin attenuates:

```text
BERT rel-hat mean: 0.790 -> 0.755
GPT-2 rel-hat mean: 0.828 -> 0.702
```

3. **NO metric-boundary collapse**: rel-hat does not become broadly worse than
   fixed. However, below-fixed counts increase:

```text
BERT: 21 -> 34
GPT-2: 3 -> 11
```

Interpretation: the project is not climbing a completely wrong local metric.
The attention-output advantage survives the first model-facing projection, but
W_O narrows the margin, especially in GPT-2. This validates continuing toward
logit/KL or perplexity metrics, while warning that downstream metrics may
further attenuate the effect. The next metric-boundary step should be logit
drift / KL on GPT-2, because causal LM gives a direct next-token distribution.

### Review addendum on Stage 4A

Independent recomputation from both CSVs confirms all logged numbers. Two
additions the results section did not surface:

1. **The zero-catastrophic streak breaks on BERT under the projected metric**:
   5 rows with rel-hat gap < -1 (GPT-2: still 0). Crucially, all 5 are the
   SAME rows as the previously diagnosed 21-row starvation set (doc39 L11h2,
   doc37 L10h3, doc17 L6h1, doc17 L10h9 at k=8/16) — W_O reweighting
   amplifies the known failure mode rather than creating a new one. The
   failure set is metric-stable, which (a) raises the priority of the
   starvation guard / boundary-refinement fix, and (b) is itself evidence
   that the diagnosis found a real mechanism. On several of these rows the
   alternatives are far worse (abs -6.2, mass -10.8), so rel-hat's worst
   rows remain the least bad.
2. **The attenuation is asymmetric** (BERT mean 0.790 -> 0.755, GPT-2
   0.828 -> 0.702) and it quantifies the headroom of a W_O-aware scorer
   (project V once per head by its W_O slice — sequence-level precompute,
   deployability preserved): roughly 0.04-0.13 mean gap. Now a motivated
   variant rather than premature engineering; mechanism of the BERT/GPT-2
   asymmetry (per-head anisotropy of c_proj vs dense) is an open question.

Agreed next metric rung: logit drift / next-token KL on GPT-2 (causal LM
gives the distribution directly). Housekeeping: a duplicate Stage 4A
pre-registration block (added in parallel by the review side) was removed;
the original at its first position is the binding one.

### Stage 4B results: GPT-2 next-token KL/logit drift

Script:

```text
scripts/stage4b_gpt2_logit_kl.py
```

Outputs:

```text
results/metric_boundary/stage4b_gpt2_logit_kl_smoke.csv
results/metric_boundary/stage4b_gpt2_logit_kl.csv
```

Protocol:

```text
model: GPT-2 small
docs: same 3 held-out docs as Stage 3
layers: 0, 5, 11
heads: all 12
budgets: 8, 16, 32
rows: 324
intervention: replace ONE head's attention context with the sparse context
evaluation: continue the real causal model forward and compare next-token
            distribution against dense GPT-2
metrics: mean/max next-token KL and mean/max logit L2
```

Implementation validation:

```text
dense manual reconstruction, doc 82, head 0:
layer 0  max logit diff 0.00387, mean diff 8.7e-05
layer 5  max logit diff 0.00066, mean diff 2.3e-05
layer 11 max logit diff 0.00012, mean diff 7.2e-06
```

This was an important implementation guard. An earlier smoke-test bug replaced
non-target heads with raw V instead of dense attention contexts PV. The dense
reconstruction check caught it before the full run. The final script uses dense
PV for all non-target heads and only patches the selected head.

#### Per-configuration improvement versus fixed-k

Improvement is `1 - method_KL / fixed_KL` for each
doc/layer/head/budget row.

```text
mean next-token KL improvement:
method             mean    median   p10     below fixed
projected_oracle  -0.407   0.067   -1.490    143
mass              -0.083   0.018   -0.850    151
UTC-abs           -0.042   0.194   -0.277     76
UTC-rel           -0.205   0.069   -0.782    140
UTC-rel-hat       -0.091   0.113   -0.602    120
```

On this per-configuration ratio metric, **the pre-registered rel-hat-leading
expectation fails**. UTC-abs is the most stable deployable scorer: it has the
best median, best p10, and fewest below-fixed rows. Rel-hat is still often
useful, but it is not the leading scorer under equal-weighted next-token KL
rows.

By layer:

```text
mean KL improvement:
layer 0:  mass -0.280, UTC-abs  0.114, UTC-rel  0.264, UTC-rel-hat  0.300
layer 5:  mass  0.015, UTC-abs -0.344, UTC-rel -0.349, UTC-rel-hat -0.293
layer 11: mass  0.010, UTC-abs  0.106, UTC-rel -0.517, UTC-rel-hat -0.271
```

So the logit/KL metric is highly layer-dependent: rel-hat wins early, but loses
badly on the sampled middle/late layers.

#### Aggregate KL risk

The ratio metric above gives every configuration equal weight and is sensitive
to tiny fixed-KL denominators. Aggregating raw KL tells a different, also
important, story:

```text
aggregate mean-KL reduction vs fixed:
projected_oracle  -0.192
mass              -0.229
UTC-abs            0.146
UTC-rel            0.057
UTC-rel-hat        0.183
```

On total KL damage, rel-hat is still the best deployable scorer: it reduces
aggregate KL by 18.3% vs fixed, compared with 14.6% for UTC-abs. This creates
a real metric split:

```text
configuration-level stability: UTC-abs > UTC-rel-hat
aggregate KL risk:             UTC-rel-hat > UTC-abs
```

By aggregate layer:

```text
layer 0:  mass -0.255, UTC-abs  0.151, UTC-rel  0.163, UTC-rel-hat  0.230
layer 5:  mass -0.274, UTC-abs -0.100, UTC-rel -0.215, UTC-rel-hat -0.130
layer 11: mass  0.059, UTC-abs  0.156, UTC-rel -0.985, UTC-rel-hat -0.230
```

By aggregate budget:

```text
k=8:  mass -0.223, UTC-abs 0.142, UTC-rel 0.059, UTC-rel-hat 0.197
k=16: mass -0.246, UTC-abs 0.138, UTC-rel 0.036, UTC-rel-hat 0.129
k=32: mass -0.249, UTC-abs 0.248, UTC-rel 0.123, UTC-rel-hat 0.153
```

#### Stage 4B verdict

This is **not** a clean PASS like Stage 4A. It is a useful boundary result:

1. The attention-output / W_O ladder does not transfer unchanged to next-token
   KL. Relative rel-hat is no longer universally best.
2. UTC-abs becomes the most stable per-configuration scorer under KL.
3. Rel-hat still minimizes aggregate KL damage, so its signal is not irrelevant.
4. The Stage-4A projected oracle is not a logit oracle; it can be worse than
   deployable scorers under KL. This confirms the warning in the pre-registration.

Interpretation: Stage 4B exposes a metric mismatch, not a project collapse.
Head-space relative error is a strong local proxy, W_O projection preserves it
with attenuation, but true next-token KL introduces layer-dependent downstream
mixing. The next design question is no longer "is rel-hat always best?" but
"which downstream objective are we optimizing: per-configuration robustness or
aggregate KL risk?"


### Stage 4B pre-registration: GPT-2 next-token KL/logit drift

Goal: test whether the Stage-4 scorer ladder survives a real causal-LM
behavior metric, not just head-space or W_O-projected vector error.
For each GPT-2 document/layer/head/budget, replace one head's attention
context with the sparse context selected by each allocator, continue the
actual model forward pass, and measure next-token KL/logit drift against
the dense model.

Binding expectations before running:

1. Rel-hat should remain the leading deployable scorer by mean KL
   improvement versus fixed-k, though further attenuation is expected.
2. The ladder should not collapse into rel-hat being broadly worse than
   fixed-k. A small number of failures is acceptable; widespread negative
   improvement is a metric-boundary failure.
3. The Stage-4A projected oracle is an offline reference, not a true
   logit-space oracle. If rel-hat beats it under KL, treat that as metric
   mismatch/discreteness, not a miracle.

### Review addendum on Stage 4B

Independent CSV analysis confirms the logged numbers and surfaces a structural
fact that changes what the result means:

1. **Per-config equal-vote improvement is a dead metric at the KL level.**
   fixed_mean_kl spans 8 orders of magnitude (1e-7 to 3e-1, median ~1e-4):
   most single-head interventions have no measurable effect on the next-token
   distribution, so ratio improvements are dominated by tiny-denominator
   noise (plus NaNs and float-negative KLs). The "abs is more stable at
   config level" claim rests largely on this noisy metric (weak support in
   the large-KL half: abs +0.113 vs rel-hat +0.036).
2. **The headline is the oracle inversion.** Aggregate (KL-magnitude
   weighted) reduction vs fixed: rel-hat +18.3%, abs +14.6%, rel +5.7%,
   projected_oracle **-19.2%**, mass -22.9%. The local-error oracle makes
   total KL WORSE than fixed — the students beat the teacher. This means the
   entire gap-closed evaluation axis does not exist at the KL level: row
   importance for KL (position weighting, downstream mixing, head redundancy)
   is not the same quantity as local output error, no matter how well the
   latter is estimated.

Revised next steps (order matters — both come BEFORE designing "KL-aware
scorers", which would otherwise be built without a valid reference):

1. Construct a KL-oracle reference axis (offline, allocate k by true KL
   impact) to answer the prior question: under single-head intervention, how
   much can ANY allocation beat fixed on KL? If even the KL-oracle's margin
   is small, the bottleneck is head redundancy, not scorers.
2. Move the intervention from single-head to full-layer / all-heads
   sparsification — the deployment-shaped intervention with material KL
   effects and better conditioning. Pre-registered fork: if the scorer
   ladder re-emerges against KL under full-layer intervention, the "target
   mismatch" story reduces to "single-head marginal effects are
   redundancy-dominated noise"; if it does not, genuine KL-aware scoring is
   warranted.

Credit where due: the dense-reconstruction validation catching the raw-V
patching bug before the full run was a load-bearing quality gate.
Pre-registration accounting: the rel-hat-leading expectation FAILED at
per-config KL level (recorded as such); the aggregate view still puts
rel-hat first (+18.3%).

### Stage 4C results: GPT-2 whole-layer KL intervention

Script:

```text
scripts/stage4c_gpt2_whole_layer_kl.py
```

Outputs:

```text
results/metric_boundary/stage4c_gpt2_whole_layer_kl_smoke.csv
results/metric_boundary/stage4c_gpt2_whole_layer_kl.csv
```

Protocol:

```text
model: GPT-2 small
docs: same 3 held-out docs as Stage 3/4B
layers: 0, 5, 11
budgets: 8, 16, 32
rows: 27 (doc/layer/budget)
intervention: replace ALL heads in one layer simultaneously
allocation: same per-head exact-budget allocators as before
evaluation: continue the real causal model forward and compare next-token
            KL/logit drift against dense GPT-2
```

Implementation validation:

```text
dense whole-layer reconstruction:
doc82 layer0  max logit diff 3.87e-03, mean 8.71e-05
doc82 layer5  max logit diff 6.60e-04, mean 2.35e-05
doc82 layer11 max logit diff 1.22e-04, mean 7.20e-06
doc93/doc96 same order or smaller
```

The whole-layer intervention does increase the downstream signal compared with
Stage 4B single-head intervention:

```text
Stage 4B single-head fixed mean KL: 0.00303
Stage 4C whole-layer fixed mean KL: 0.02578
```

So the "single-head effects are too tiny" concern was real. But the main
question is whether the scorer ladder re-emerges under this better-conditioned
intervention.

#### Per-configuration improvement versus fixed-k

```text
mean next-token KL improvement:
method             mean    median   p10     below fixed
projected_oracle  -1.554  -0.289   -2.733     20
mass              -0.146  -0.140   -0.486     17
UTC-abs            0.096   0.056   -0.206     10
UTC-rel           -0.262  -0.089   -0.712     19
UTC-rel-hat       -0.026  -0.016   -0.418     15
```

Result: the rel-hat ladder does **not** re-emerge. UTC-abs is again the most
stable deployable scorer at the per-configuration KL level. Rel-hat is close to
neutral on average but has more below-fixed rows.

#### Aggregate KL risk

```text
aggregate mean-KL reduction vs fixed:
projected_oracle  -0.527
mass              -0.193
UTC-abs            0.069
UTC-rel           -0.049
UTC-rel-hat        0.074
```

Rel-hat still barely wins the aggregate KL-risk view (+7.4% vs +6.9% for
UTC-abs), but the margin is tiny compared with Stage 4B (+18.3% vs +14.6%).
The downstream KL advantage is therefore weak and objective-dependent.

#### Layer/budget structure

By layer, aggregate mean-KL reduction:

```text
layer 0:  mass -0.214, UTC-abs  0.068, UTC-rel  0.084, UTC-rel-hat  0.114
layer 5:  mass -0.115, UTC-abs  0.007, UTC-rel -0.130, UTC-rel-hat -0.051
layer 11: mass -0.045, UTC-abs  0.087, UTC-rel -1.064, UTC-rel-hat -0.205
```

Layer 0 still partially preserves the value-aware story. Layer 11 breaks it:
UTC-abs remains modestly helpful, while relative scores and the projected
oracle become actively harmful.

By budget, aggregate mean-KL reduction:

```text
k=8:  mass -0.161, UTC-abs 0.051, UTC-rel -0.028, UTC-rel-hat 0.090
k=16: mass -0.287, UTC-abs 0.089, UTC-rel -0.169, UTC-rel-hat -0.007
k=32: mass -0.295, UTC-abs 0.255, UTC-rel  0.126, UTC-rel-hat 0.172
```

#### Stage 4C verdict

The pre-registered fork resolves toward the second branch:

1. Whole-layer intervention fixes part of the Stage 4B measurement problem by
   making KL effects larger and better conditioned.
2. However, it does **not** restore the local scorer ladder. The projected
   local oracle remains bad under KL, especially in late layers.
3. Therefore the Stage 4B mismatch is not merely a single-head redundancy
   artifact. It is a genuine downstream-objective boundary: local attention /
   W_O error and next-token KL are different targets.

Practical implication for this project: do not open a new KL-aware scorer
optimization loop here. The current research-style artifact should close with
the boundary statement:

```text
Value-aware local sparse-attention error can be controlled cheaply and transfers
through W_O with attenuation, but next-token KL requires a new behavioral
reference axis. Local restricted oracles are not behavioral oracles.
```

This is a stronger and cleaner stopping point than trying to retrofit rel-hat
into a KL scorer.


### Stage 4C pre-registration: GPT-2 whole-layer KL intervention

Goal: decide whether the Stage-4B KL mismatch is mostly an artifact
of single-head marginal interventions or a genuine downstream-objective
mismatch. Patch all heads in one GPT-2 layer simultaneously using the
same per-head exact-budget allocators, continue the real causal model,
and measure next-token KL/logit drift against dense GPT-2.

Binding fork before running:

1. If the scorer ladder re-emerges under whole-layer intervention,
   then Stage 4B's failure is largely a single-head redundancy/noise
   artifact and local sparse-attention control remains behaviorally
   useful at deployment-shaped scale.
2. If the ladder still fails, then the boundary is deeper: local
   attention/W_O error is insufficient as a behavioral metric, and any
   future scorer should be KL-aware/layer-aware rather than just a
   sharper local-error estimator.
3. The projected oracle remains a local W_O oracle, not a KL oracle.
   Its performance under KL is diagnostic only.

### Review addendum on Stage 4C — and the project's boundary claim

CSV verified (27 rows; conditioning fixed as intended: median fixed KL 6.6e-3
vs ~1e-4 at 4B). The pre-registered fork resolves cleanly: the ladder does NOT
re-emerge under whole-layer intervention, so the Stage 4B result was not a
single-head artifact. The boundary is real. Two structural additions:

1. **Dose-response**: the local oracle's KL damage grows with intervention
   size (single-head -19.2% -> whole-layer -52.7%) — the more an allocation
   trusts local-error optimality, the more behavioral damage. This upgrades
   the negative result from "didn't transfer" to "actively anti-correlated at
   scale".
2. **The boundary is a gradient in depth, not a wall**:

```text
L0 : oracle -19.4%, rel-hat +11.4%   (local fidelity still helps early)
L5 : everything ~ 0                   (redundancy absorbs the layer)
L11: oracle -319%,  rel-hat -20.5%   (local optimality is harmful at readout)
```

   Mechanism reading: early-layer errors are reprocessed by many downstream
   layers, so uniform local fidelity is approximately the right objective;
   the last layer feeds the LM head directly, whose narrow readout cone makes
   head-space L2 maximally misaligned with behavior. This is the
   readout-reweights-directions lesson, sharpened: the closer to the readout,
   the harsher the reweighting.

**Boundary claim (project terminus for this arc)**: cheap local error control
for top-k sparse attention is achievable (UTC-rel-hat: zero extra per-row
V-IO, ~80% of the restricted-oracle gap, transferable BERT->GPT-2, survives
W_O projection) — but local output error is not behavioral importance. Under
next-token KL, local-error oracles invert (with dose-response and a depth
gradient), so the local framework's writ ends at the model's readout
structure. Extending it would require behavioral importance signals
(position/direction/depth-aware), which is a different research object, out
of scope for this project. Recorded as an earned negative result with
three-scale evidence, not a failure of the method within its domain.

Status: Stage 4 closed. Remaining in-domain open items (starvation guard,
cross-head pooling, set selection, kernel work) are未来 chapters; the next
deliverable is the 6.4 boundary-chapter write-up.
