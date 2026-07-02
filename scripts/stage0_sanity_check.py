''' 
'''

import numpy as np


def gen_qkv(N, d, seed=0, q_scale=1.0, dtype=np.float64):
    rng = np.random.default_rng(seed)
    Q = (rng.standard_normal((N, d)) * q_scale).astype(dtype)
    K = rng.standard_normal((N, d)).astype(dtype)
    V = rng.standard_normal((N, d)).astype(dtype)
    return Q, K, V

def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def full_attention(Q, K, V):
    d = Q.shape[-1]
    logits = Q @ K.T / np.sqrt(d)
    P = softmax(logits, axis=-1)
    O = P @ V
    return P, O

def attention_entropy(P, eps=1e-12):
    return -np.sum(P * np.log(P + eps), axis=-1)

def entropy_adaptive_k(P, k_min=4, k_max=32):
    n = P.shape[-1]
    H = attention_entropy(P)
    H_norm = H / np.log(n)
    ks = np.ceil(k_min + H_norm * (k_max - k_min)).astype(int)
    return np.clip(ks, k_min, k_max)

def allocate_integer_budget(scores, total_budget, k_min, k_max):
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.shape[0]
    total_budget = int(total_budget)
    min_budget = n * k_min
    max_budget = n * k_max
    if not min_budget <= total_budget <= max_budget:
        raise ValueError("total_budget must be between n * k_min and n * k_max")

    capacity = np.full(n, k_max - k_min, dtype=np.float64)
    remaining = total_budget - min_budget
    if remaining == 0:
        return np.full(n, k_min, dtype=int)

    weights = scores.copy()
    if weights.min() < 0:
        weights = weights - weights.min()
    if np.allclose(weights.sum(), 0.0):
        weights = np.ones_like(weights)
    weights = weights / weights.sum()

    extra_float = np.zeros(n, dtype=np.float64)
    active = np.ones(n, dtype=bool)
    remaining_float = float(remaining)

    while remaining_float > 1e-12 and active.any():
        active_idx = np.flatnonzero(active)
        active_weights = weights[active_idx]
        active_weights = active_weights / active_weights.sum()
        proposal = active_weights * remaining_float
        active_capacity = capacity[active_idx]
        over_cap = proposal > active_capacity

        if not np.any(over_cap):
            extra_float[active_idx] = proposal
            break

        capped_idx = active_idx[over_cap]
        extra_float[capped_idx] = capacity[capped_idx]
        remaining_float -= capacity[capped_idx].sum()
        active[capped_idx] = False

    extra_int = np.floor(extra_float).astype(int)
    ks = k_min + extra_int

    leftover = total_budget - ks.sum()
    if leftover > 0:
        fractional = extra_float - extra_int
        order = np.argsort(-fractional)
        for idx in order:
            if leftover == 0:
                break
            if ks[idx] < k_max:
                ks[idx] += 1
                leftover -= 1

    return ks

def entropy_budgeted_k(P, target_mean_k=30, k_min=4, k_max=64):
    n_rows = P.shape[0]
    total_budget = int(round(n_rows * target_mean_k))
    H = attention_entropy(P)
    return allocate_integer_budget(H, total_budget, k_min, k_max)

def row_decomposition_metrics(p, V, k):
    n = p.shape[0]
    if k < 1 or k > n:
        raise ValueError("k must be between 1 and the row length")

    o_full = p @ V
    if k == n:
        return {
            "true_error": 0.0,
            "decomp_error": 0.0,
            "abs_diff": 0.0,
            "delta": 0.0,
            "centroid_dist": 0.0,
        }

    topk_idx = np.argsort(-p)[:k] # 保留权重最大的 k 个索引

    mask = np.zeros_like(p, dtype=bool)
    mask[topk_idx] = True

    S = mask
    R = ~mask

    m = p[S].sum()
    delta = p[R].sum()

    mu_S = (p[S, None] * V[S]).sum(axis=0) / m
    mu_R = (p[R, None] * V[R]).sum(axis=0) / delta

    o_sparse = mu_S

    centroid_dist = np.linalg.norm(mu_R - mu_S)
    err_true = np.linalg.norm(o_full - o_sparse)
    err_decomp = delta * centroid_dist

    return {
        "true_error": err_true,
        "decomp_error": err_decomp,
        "abs_diff": abs(err_true - err_decomp),
        "delta": delta,
        "centroid_dist": centroid_dist,
    }

def check_all_rows_decomposition(P, V, k=8):
    n_rows = P.shape[0]
    if np.isscalar(k):
        ks = np.full(n_rows, int(k))
    else:
        ks = np.asarray(k, dtype=int)
        assert ks.shape == (n_rows,)

    true_errors = []
    decomp_errors = []
    abs_diffs = []
    deltas = []
    centroid_dists = []

    for row in range(n_rows):
        p = P[row]
        row_k = ks[row]
        metrics = row_decomposition_metrics(p, V, row_k)

        true_errors.append(metrics["true_error"])
        decomp_errors.append(metrics["decomp_error"])
        abs_diffs.append(metrics["abs_diff"])
        deltas.append(metrics["delta"])
        centroid_dists.append(metrics["centroid_dist"])

    return {
        "ks": ks,
        "true_errors": np.array(true_errors),
        "decomp_errors": np.array(decomp_errors),
        "abs_diffs": np.array(abs_diffs),
        "deltas": np.array(deltas),
        "centroid_dists": np.array(centroid_dists),
    }

def print_summary(name, stats):
    print(f"\n[{name}]")
    print("mean k:", stats["ks"].mean())
    print("max abs diff:", stats["abs_diffs"].max())
    print("mean abs diff:", stats["abs_diffs"].mean())
    print("max true error:", stats["true_errors"].max())
    print("mean true error:", stats["true_errors"].mean())
    print("mean delta:", stats["deltas"].mean())
    print("mean centroid dist:", stats["centroid_dists"].mean())

def run_budget_sweep(P, V, budgets, k_min=4, k_max=96):
    rows = []
    for budget in budgets:
        fixed_stats = check_all_rows_decomposition(P, V, k=budget)
        budgeted_ks = entropy_budgeted_k(P, target_mean_k=budget, k_min=k_min, k_max=k_max)
        budgeted_stats = check_all_rows_decomposition(P, V, k=budgeted_ks)

        rows.append({
            "budget": budget,
            "fixed_mean_error": fixed_stats["true_errors"].mean(),
            "entropy_mean_error": budgeted_stats["true_errors"].mean(),
            "fixed_max_error": fixed_stats["true_errors"].max(),
            "entropy_max_error": budgeted_stats["true_errors"].max(),
            "fixed_mean_delta": fixed_stats["deltas"].mean(),
            "entropy_mean_delta": budgeted_stats["deltas"].mean(),
            "entropy_k_min": budgeted_ks.min(),
            "entropy_k_max": budgeted_ks.max(),
        })
    return rows

def print_budget_sweep(rows):
    print("\n[budget sweep: fixed top-k vs entropy-budgeted top-k]")
    print(
        "budget | fixed mean err | entropy mean err | "
        "fixed max err | entropy max err | entropy k range"
    )
    print("-" * 91)
    for row in rows:
        print(
            f"{row['budget']:>6} | "
            f"{row['fixed_mean_error']:>14.4f} | "
            f"{row['entropy_mean_error']:>16.4f} | "
            f"{row['fixed_max_error']:>13.4f} | "
            f"{row['entropy_max_error']:>15.4f} | "
            f"[{row['entropy_k_min']:>2}, {row['entropy_k_max']:>2}]"
        )

def run_q_scale_sweep(q_scales, N=128, d=64, seed=0, k=30):
    rows = []
    for q_scale in q_scales:
        Q, K, V = gen_qkv(N, d, seed=seed, q_scale=q_scale)
        P, _ = full_attention(Q, K, V)
        stats = check_all_rows_decomposition(P, V, k=k)
        analysis = predictor_analysis(P, stats)
        H_norm = analysis["entropy"] / np.log(N)

        rows.append({
            "q_scale": q_scale,
            "entropy_min": H_norm.min(),
            "entropy_mean": H_norm.mean(),
            "entropy_max": H_norm.max(),
            "mean_error": stats["true_errors"].mean(),
            "mean_delta": stats["deltas"].mean(),
            "corr_entropy_error": analysis["corr_entropy_error"],
            "corr_delta_error": analysis["corr_delta_error"],
            "corr_centroid_error": analysis["corr_centroid_error"],
            "corr_delta_centroid": pearson_corr(stats["deltas"], stats["centroid_dists"]),
        })
    return rows

def print_q_scale_sweep(rows):
    print("\n[q_scale sweep: fixed top-k, k=30]")
    print(
        "q_scale | H_norm min | H_norm mean | H_norm max | "
        "mean err | mean delta | corr(H,err) | corr(delta,err) | "
        "corr(centroid,err) | corr(delta,centroid)"
    )
    print("-" * 147)
    for row in rows:
        print(
            f"{row['q_scale']:>7.2f} | "
            f"{row['entropy_min']:>10.4f} | "
            f"{row['entropy_mean']:>11.4f} | "
            f"{row['entropy_max']:>10.4f} | "
            f"{row['mean_error']:>8.4f} | "
            f"{row['mean_delta']:>10.4f} | "
            f"{row['corr_entropy_error']:>11.4f} | "
            f"{row['corr_delta_error']:>15.4f} | "
            f"{row['corr_centroid_error']:>18.4f} | "
            f"{row['corr_delta_centroid']:>20.4f}"
        )

def relative_errors(P, V, stats, eta=1e-12):
    O = P @ V
    output_norms = np.linalg.norm(O, axis=1)
    return stats["true_errors"] / (output_norms + eta)

def error_budgeted_k(P, V, epsilon_rel, eta=1e-12):
    n_rows, n = P.shape
    ks = np.empty(n_rows, dtype=int)

    for row in range(n_rows):
        p = P[row]
        o_norm = np.linalg.norm(p @ V)

        for k in range(1, n + 1):
            metrics = row_decomposition_metrics(p, V, k)
            rel_error = metrics["true_error"] / (o_norm + eta)
            if rel_error <= epsilon_rel:
                ks[row] = k
                break

    return ks

def dropped_mass_budgeted_k(P, tau):
    n_rows, n = P.shape
    ks = np.empty(n_rows, dtype=int)

    for row in range(n_rows):
        p_sorted = np.sort(P[row])[::-1]
        retained_mass = np.cumsum(p_sorted)
        dropped_mass = 1.0 - retained_mass
        valid = np.flatnonzero(dropped_mass <= tau)
        if valid.size == 0:
            ks[row] = n
        else:
            ks[row] = valid[0] + 1

    return ks

def fixed_k_needed_for_epsilon(P, V, epsilon_rel, eta=1e-12):
    n = P.shape[1]
    for k in range(1, n + 1):
        stats = check_all_rows_decomposition(P, V, k=k)
        rel = relative_errors(P, V, stats, eta=eta)
        if rel.max() <= epsilon_rel:
            return k
    return n

def run_error_budget_sweep(P, V, epsilons, eta=1e-12):
    rows = []
    for epsilon_rel in epsilons:
        oracle_ks = error_budgeted_k(P, V, epsilon_rel, eta=eta)
        oracle_stats = check_all_rows_decomposition(P, V, k=oracle_ks)
        oracle_rel = relative_errors(P, V, oracle_stats, eta=eta)

        fixed_at_mean_k = int(np.ceil(oracle_ks.mean()))
        fixed_at_mean_stats = check_all_rows_decomposition(P, V, k=fixed_at_mean_k)
        fixed_at_mean_rel = relative_errors(P, V, fixed_at_mean_stats, eta=eta)

        fixed_needed_k = fixed_k_needed_for_epsilon(P, V, epsilon_rel, eta=eta)
        fixed_needed_stats = check_all_rows_decomposition(P, V, k=fixed_needed_k)
        fixed_needed_rel = relative_errors(P, V, fixed_needed_stats, eta=eta)

        rows.append({
            "epsilon_rel": epsilon_rel,
            "oracle_mean_k": oracle_ks.mean(),
            "oracle_max_k": oracle_ks.max(),
            "oracle_mean_rel_error": oracle_rel.mean(),
            "oracle_max_rel_error": oracle_rel.max(),
            "fixed_at_mean_k": fixed_at_mean_k,
            "fixed_at_mean_mean_rel_error": fixed_at_mean_rel.mean(),
            "fixed_at_mean_max_rel_error": fixed_at_mean_rel.max(),
            "fixed_needed_k": fixed_needed_k,
            "fixed_needed_mean_rel_error": fixed_needed_rel.mean(),
            "fixed_needed_max_rel_error": fixed_needed_rel.max(),
        })
    return rows

def run_dropped_mass_sweep(P, V, taus, eta=1e-12):
    rows = []
    for tau in taus:
        mass_ks = dropped_mass_budgeted_k(P, tau)
        mass_stats = check_all_rows_decomposition(P, V, k=mass_ks)
        mass_rel = relative_errors(P, V, mass_stats, eta=eta)

        fixed_at_mean_k = int(np.ceil(mass_ks.mean()))
        fixed_stats = check_all_rows_decomposition(P, V, k=fixed_at_mean_k)
        fixed_rel = relative_errors(P, V, fixed_stats, eta=eta)

        rows.append({
            "tau": tau,
            "mass_mean_k": mass_ks.mean(),
            "mass_max_k": mass_ks.max(),
            "mass_mean_delta": mass_stats["deltas"].mean(),
            "mass_max_delta": mass_stats["deltas"].max(),
            "mass_mean_rel_error": mass_rel.mean(),
            "mass_max_rel_error": mass_rel.max(),
            "fixed_at_mean_k": fixed_at_mean_k,
            "fixed_mean_rel_error": fixed_rel.mean(),
            "fixed_max_rel_error": fixed_rel.max(),
        })
    return rows

def print_dropped_mass_sweep(rows):
    print("\n[dropped-mass-budgeted top-k sweep]")
    print(
        "tau | mass mean k | mass max k | max delta | mass mean rel | "
        "mass max rel | fixed@mean k | fixed@mean max rel"
    )
    print("-" * 111)
    for row in rows:
        print(
            f"{row['tau']:>4.2f} | "
            f"{row['mass_mean_k']:>11.2f} | "
            f"{row['mass_max_k']:>10} | "
            f"{row['mass_max_delta']:>9.4f} | "
            f"{row['mass_mean_rel_error']:>13.4f} | "
            f"{row['mass_max_rel_error']:>12.4f} | "
            f"{row['fixed_at_mean_k']:>12} | "
            f"{row['fixed_max_rel_error']:>18.4f}"
        )

def print_error_budget_sweep(rows):
    print("\n[oracle error-budgeted top-k sweep]")
    print(
        "eps | oracle mean k | oracle max k | oracle mean rel | oracle max rel | "
        "fixed@mean k | fixed@mean max rel | fixed needed k"
    )
    print("-" * 123)
    for row in rows:
        print(
            f"{row['epsilon_rel']:>4.2f} | "
            f"{row['oracle_mean_k']:>13.2f} | "
            f"{row['oracle_max_k']:>12} | "
            f"{row['oracle_mean_rel_error']:>15.4f} | "
            f"{row['oracle_max_rel_error']:>14.4f} | "
            f"{row['fixed_at_mean_k']:>12} | "
            f"{row['fixed_at_mean_max_rel_error']:>18.4f} | "
            f"{row['fixed_needed_k']:>14}"
        )

def pearson_corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if np.allclose(x.std(), 0.0) or np.allclose(y.std(), 0.0):
        return np.nan
    return np.corrcoef(x, y)[0, 1]

def predictor_analysis(P, stats):
    H = attention_entropy(P)
    errors = stats["true_errors"]
    deltas = stats["deltas"]
    centroid_dists = stats["centroid_dists"]
    decomp_proxy = deltas * centroid_dists

    return {
        "entropy": H,
        "errors": errors,
        "deltas": deltas,
        "centroid_dists": centroid_dists,
        "decomp_proxy": decomp_proxy,
        "corr_entropy_error": pearson_corr(H, errors),
        "corr_delta_error": pearson_corr(deltas, errors),
        "corr_centroid_error": pearson_corr(centroid_dists, errors),
        "corr_decomp_error": pearson_corr(decomp_proxy, errors),
    }

def print_predictor_analysis(name, analysis, top_n=5):
    print(f"\n[predictor analysis: {name}]")
    print("corr(entropy, error):", analysis["corr_entropy_error"])
    print("corr(delta, error):", analysis["corr_delta_error"])
    print("corr(centroid_dist, error):", analysis["corr_centroid_error"])
    print("corr(delta * centroid_dist, error):", analysis["corr_decomp_error"])

    order = np.argsort(-analysis["errors"])[:top_n]
    print("\ntop-error rows")
    print("row | error | entropy | delta | centroid_dist | delta*centroid")
    print("-" * 67)
    for row in order:
        print(
            f"{row:>3} | "
            f"{analysis['errors'][row]:>5.4f} | "
            f"{analysis['entropy'][row]:>7.4f} | "
            f"{analysis['deltas'][row]:>5.4f} | "
            f"{analysis['centroid_dists'][row]:>13.4f} | "
            f"{analysis['decomp_proxy'][row]:>14.4f}"
        )

if __name__ == "__main__":
    N = 128
    d = 64
    Q, K, V = gen_qkv(N, d)
    logits = Q @ K.T / np.sqrt(Q.shape[-1])
    P = softmax(logits, axis=-1)

    fixed_stats = check_all_rows_decomposition(P, V, k=30)
    print_summary("fixed top-k, k=30", fixed_stats)

    adaptive_ks = entropy_adaptive_k(P, k_min=4, k_max=32)
    adaptive_stats = check_all_rows_decomposition(P, V, k=adaptive_ks)
    print_summary("entropy-adaptive top-k, k in [4, 32]", adaptive_stats)

    budgeted_ks = entropy_budgeted_k(P, target_mean_k=30, k_min=4, k_max=64)
    budgeted_stats = check_all_rows_decomposition(P, V, k=budgeted_ks)
    print_summary("entropy-budgeted top-k, mean k=30", budgeted_stats)

    H = attention_entropy(P)
    print("\nentropy range:", H.min(), H.max())
    print("normalized entropy range:", (H / np.log(N)).min(), (H / np.log(N)).max())
    print("adaptive k range:", adaptive_ks.min(), adaptive_ks.max())
    print("budgeted k range:", budgeted_ks.min(), budgeted_ks.max())
    print("budgeted total k:", budgeted_ks.sum())

    budgets = [4, 8, 16, 24, 32, 48, 64]
    sweep_rows = run_budget_sweep(P, V, budgets, k_min=4, k_max=96)
    print_budget_sweep(sweep_rows)

    analysis_k30 = predictor_analysis(P, fixed_stats)
    print_predictor_analysis("fixed top-k, k=30", analysis_k30)

    q_scale_rows = run_q_scale_sweep([0.25, 0.5, 1.0, 2.0, 4.0, 8.0], N=N, d=d, k=30)
    print_q_scale_sweep(q_scale_rows)

    error_budget_rows = run_error_budget_sweep(P, V, epsilons=[0.1, 0.2, 0.5, 1.0, 2.0])
    print_error_budget_sweep(error_budget_rows)

    dropped_mass_rows = run_dropped_mass_sweep(P, V, taus=[0.1, 0.2, 0.4, 0.6, 0.8])
    print_dropped_mass_sweep(dropped_mass_rows)
