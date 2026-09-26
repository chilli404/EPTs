import itertools

import numpy as np
from scipy.optimize import minimize as _minimize


def compile_multichoice_dp(layer_options, budget, resolution=1000, max_free=None):
    """Per-layer multiple-choice knapsack: pick exactly one option per layer.

    Generalizes `compile_graph_dp`'s 0-1 (skip/keep) choice per layer to N
    options per layer -- e.g. the 6 execution modes, or 3 for ternary.

    `layer_options`: list of length n_layers; layer_options[i] is a dict
    {option_name: (cost, benefit)}. `cost` is non-negative accuracy
    degradation (e.g. delta_bpb); `benefit` is latency saved vs. the
    all-sequential baseline. A "sequential" option with cost=0, benefit=0
    should be included per layer (it is the do-nothing choice; layers with
    no other affordable option fall back to it automatically since it is
    free).

    Options with cost <= 0 (free or quality-*improving*) are taken
    unconditionally per layer (by picking whichever such option has the
    highest benefit) before the DP runs, since there is never a reason not
    to take a strictly non-worse, no-cost improvement in isolation -- this
    matters because at least one measured mode (attn_only) has negative
    single-layer cost. The remaining budget after these free picks is what
    the DP allocates.

    CAUTION -- validated failure mode: single-layer costs are measured in
    isolation and do NOT sum linearly once many are stacked (the same
    non-additivity documented for ternary composition, holdout Pearson only
    0.70, and for teacher-detach collapse). Taking every free option
    unconditionally can silently combine into a catastrophic joint effect
    (observed: 16/20 "free" single-layer skip costs at 430M, individually
    near-zero, jointly +1.18 BPB -- ~240x the additive prediction). Use
    `max_free` to cap how many free picks are taken (ranked by benefit,
    globally across layers); layers whose free option isn't in the top
    `max_free` fall through to the ordinary budgeted DP instead of being
    auto-applied. Always validate the compiled mask against real
    measurements before trusting it -- this cap is an empirical mitigation,
    not a proof of additivity at the chosen count.

    Returns {"choices": [...], "predicted_cost": float, "predicted_benefit":
    float}. Optimal on the discretized cost grid; true optimality gap is at
    most one bin width (remaining_budget / resolution) per layer.
    """
    n_layers = len(layer_options)
    choices = [None] * n_layers
    free_benefit = np.zeros(n_layers)
    remaining_budget = float(budget)

    # Rank each layer's best free option globally by benefit; only the top
    # `max_free` are auto-applied. The rest are dropped entirely (not merely
    # deprioritized) so a layer whose only cheap option is unsafely-stacked
    # `skip` does not silently reappear inside the paid DP either.
    free_candidates = []  # (layer_idx, name, cost, benefit)
    for i, opts in enumerate(layer_options):
        # benefit > 0 excludes the no-op "sequential" (cost=0, benefit=0)
        # from ever occupying a free slot -- otherwise it trivially wins
        # every layer with no better free alternative and wipes out that
        # layer's paid DP options entirely (observed: this broke the DP for
        # every layer whose only cost<=0 option was the do-nothing default).
        free = [(name, cost, benefit) for name, (cost, benefit) in opts.items()
                if cost <= 0 and benefit > 0]
        if free:
            free_candidates.append((i,) + max(free, key=lambda t: t[2]))
    if max_free is not None and len(free_candidates) > max_free:
        free_candidates = sorted(free_candidates, key=lambda t: t[3],
                                 reverse=True)[:max_free]
    applied_free = set()
    for i, name, cost, benefit in free_candidates:
        choices[i] = name
        free_benefit[i] = benefit
        remaining_budget -= cost  # cost <= 0, so this only ever adds budget back
        applied_free.add(i)

    dp_options = []  # per layer, options with cost > 0 only (or all, if capped out)
    for i, opts in enumerate(layer_options):
        if i in applied_free:
            dp_options.append([])
        else:
            dp_options.append([(name, cost, benefit)
                               for name, (cost, benefit) in opts.items() if cost > 0])

    remaining_budget = max(remaining_budget, 0.0)
    step = remaining_budget / resolution if resolution > 0 else 0.0
    if step <= 0:
        predicted_cost = -sum(opts.get(choices[i], (0.0, 0.0))[0]
                              for i, opts in enumerate(layer_options) if choices[i])
        return {
            "choices": [c or "sequential" for c in choices],
            "predicted_cost": float(-min(predicted_cost, 0.0)),
            "predicted_benefit": float(free_benefit.sum()),
        }

    dp = np.full(resolution + 1, -np.inf)
    dp[0] = 0.0
    # trace[i][cap] = index into dp_options[i] chosen to reach this cap, or -1
    trace = [np.full(resolution + 1, -1, dtype=int) for _ in range(n_layers)]

    for i in range(n_layers):
        new_dp = dp.copy()
        for opt_idx, (name, cost, benefit) in enumerate(dp_options[i]):
            int_cost = int(round(cost / step))
            if int_cost > resolution or int_cost <= 0:
                continue
            for cap in range(resolution, int_cost - 1, -1):
                val = dp[cap - int_cost] + benefit
                if val > new_dp[cap]:
                    new_dp[cap] = val
                    trace[i][cap] = opt_idx
        dp = new_dp

    cap = int(np.argmax(dp))
    dp_benefit = float(dp[cap])  # capture before traceback mutates `cap`
    dp_cost = 0.0
    for i in range(n_layers - 1, -1, -1):
        opt_idx = trace[i][cap]
        if opt_idx >= 0:
            name, cost, benefit = dp_options[i][opt_idx]
            choices[i] = name
            int_cost = int(round(cost / step))
            cap -= int_cost
            dp_cost += cost

    free_cost = sum(opts[choices[i]][0] for i, opts in enumerate(layer_options)
                    if choices[i] and opts[choices[i]][0] <= 0)
    return {
        "choices": [c or "sequential" for c in choices],
        "predicted_cost": float(dp_cost + free_cost),
        "predicted_benefit": float(free_benefit.sum() + dp_benefit),
    }


def _dist_bin(d):
    return 0 if d == 1 else (1 if d <= 4 else 2)


AGGRESSIVE_MODES = frozenset({"skip", "attn_only", "ffn_only"})


def evaluate_true_cost(mask, base_costs, alt_modes, alpha, beta, gamma, seq_penalty=None,
                       aggressive_modes=AGGRESSIVE_MODES):
    """The non-separable M2+M3(+sequential-starvation) cost of a full mask --
    ground truth against which any linearized/iterative approximation must
    be checked.

    M2: sum_k alpha_k * A_k(m) * max(n_k(m),1)^-beta_k, where A_k(m) is the
    additive subtotal of base_costs for mode k and n_k(m) its count.
    M3: sum over non-sequential position pairs of gamma[(mode_i,mode_j,dist_bin)],
    keyed by the alphabetically-sorted mode pair.
    `seq_penalty=(gamma_seq, delta_seq)` (optional, default None = no-op):
    a term that penalizes masks with few remaining "safe" layers -- sequential
    PLUS the mild, non-FLOP-reducing alt-modes (`aggressive_modes` excludes
    parallel/reverse by default). Real finding (this session): a full
    all-parallel 1B specialist rewrite (28/28 layers changed) degraded only
    +0.229 BPB, LESS than several partial masks that left 8-10 layers
    sequential but used skip/attn_only heavily for the rest (+0.37 to
    +0.54 BPB) -- severity tracks which modes are used, not how many
    layers are touched. The original version keyed this term on raw
    n_sequential alone, which can't distinguish "28 parallel, 0 sequential"
    (mild, low real cost) from "18 skip/attn_only, 0 sequential" (severe);
    counting mild alt-modes as "safe" alongside sequential fixes this.
    Uses (n_safe + 1) as the base (not max(n_safe,1), which would make 0
    and 1 indistinguishable) so the term is finite at n_safe=0 and strictly
    decreasing as more safe layers remain.
    """
    positions = [(l, m) for l, m in enumerate(mask) if m != "sequential"]
    n_k = {k: 0 for k in alt_modes}
    A_k = {k: 0.0 for k in alt_modes}
    for l, m in positions:
        n_k[m] += 1
        A_k[m] += base_costs.get((l, m), 0.0)

    m2 = 0.0
    for k in alt_modes:
        a = alpha.get(k, 1.0)
        b = beta.get(k, 0.0)
        m2 += a * A_k[k] * max(n_k[k], 1) ** (-b)

    m3 = 0.0
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            li, mi = positions[i]
            lj, mj = positions[j]
            key_modes = tuple(sorted((mi, mj)))
            m3 += gamma.get(key_modes + (_dist_bin(abs(li - lj)),), 0.0)

    seq_term = 0.0
    if seq_penalty is not None:
        gamma_seq, delta_seq = seq_penalty
        n_aggressive = sum(1 for _, m in positions if m in aggressive_modes)
        n_safe = len(mask) - n_aggressive
        seq_term = gamma_seq * (n_safe + 1) ** (-delta_seq)

    return m2 + m3 + seq_term


def _marginal_m3(layer, mode, current_mask, gamma):
    """Marginal M3 contribution of placing `mode` at `layer`, holding every
    other currently-selected position fixed (first-order linearization used
    to build the next DP-solvable cost table)."""
    total = 0.0
    for l, m in enumerate(current_mask):
        if l == layer or m == "sequential":
            continue
        key_modes = tuple(sorted((mode, m)))
        total += gamma.get(key_modes + (_dist_bin(abs(layer - l)),), 0.0)
    return total


def compile_iterative_dp(base_costs, benefits, alt_modes, alpha, beta, gamma,
                          budget, n_layers, resolution=1000, max_free=None,
                          n_passes=3, extra_candidates=None, seq_penalty=None):
    """Compiler for a non-separable (M2+M3) cost model.

    `extra_candidates`: optional list of previously-found masks (e.g. the
    winning masks from smaller budgets already computed by the caller) to
    also check via `consider()`. Required for cross-budget monotonicity --
    a wider budget must never do worse than a tighter one, but the
    coordinate sweep below only finds *self-consistent fixed points* (a
    guessed per-mode count and the DP's own resulting count under that
    guess agree), and a mask that is self-consistent at one budget is not
    guaranteed to be rediscovered as self-consistent at a different budget,
    even though it may still be truly feasible there (real bug found on
    production ternary/6-mode data: an 8-skip mask with predicted_cost
    -0.0004, satisfying every budget >= 0, was found at budget=0.0005 but
    not at the wider budget=0.002). Passing prior winners closes this gap
    without changing the search itself -- a smaller budget's feasible mask
    is always feasible at a larger budget too, so it just needs to be
    re-checked, not re-discovered.

    `compile_multichoice_dp` requires per-layer-independent costs -- M2's
    count attenuation (n_k(m)^-beta_k depends on the WHOLE mask's count of
    mode k, not just this layer) and M3's pairwise term break that. Fix:
    linearize around a current *count estimate* per mode (for M2) and a
    current mask (for M3's marginal contribution), which makes the DP
    separable again, then re-solve.

    A naive fixed-point (re-estimate the count from the DP's own output,
    repeat) was tried first and does not work: on a strongly super-additive
    penalty it can oscillate between under- and over-estimating the count
    without ever visiting the true optimum in between (verified: 3 layers,
    cost ~ n^2, budget affording exactly 2 -- fixed point cycles between
    "solve as if n=1" -> picks all 3 -> "solve as if n=3" -> picks 1 -> repeat,
    never trying n=2). Fixed by coordinate-sweeping: for each mode, explicitly
    try every candidate count 0..n_layers (not just whatever the last DP
    solve happened to produce), solve the DP at each, keep whichever
    TRULY-feasible (real cost <= budget) result has the best true benefit --
    a few passes over all modes lets modes' count estimates settle against
    each other. The plain non-iterative DP on unlinearized base costs is
    always included as a candidate too, so the result is never worse than
    a single ordinary compile_multichoice_dp call.
    """
    def marginal_seq_cost(current_mask):
        """Cost of converting ONE more sequential layer to ANY alt-mode,
        given the current mask's sequential count -- added to every
        alt-mode's effective cost at every layer, since seq_penalty depends
        only on the total sequential count, not which mode replaces it."""
        if seq_penalty is None:
            return 0.0
        gamma_seq, delta_seq = seq_penalty
        n_seq = current_mask.count("sequential")
        before = gamma_seq * (n_seq + 1) ** (-delta_seq)
        after = gamma_seq * (max(n_seq - 1, 0) + 1) ** (-delta_seq)
        return after - before

    def build_layer_options(n_k_guess, current_mask):
        marginal_seq = marginal_seq_cost(current_mask)
        opts = []
        for l in range(n_layers):
            row = {"sequential": (0.0, 0.0)}
            for k in alt_modes:
                if (l, k) not in base_costs:
                    continue
                a = alpha.get(k, 1.0)
                b = beta.get(k, 0.0)
                eff_cost = base_costs[(l, k)] * a * max(n_k_guess[k], 1) ** (-b)
                eff_cost += _marginal_m3(l, k, current_mask, gamma)
                eff_cost += marginal_seq
                row[k] = (eff_cost, benefits.get((l, k), 0.0))
            opts.append(row)
        return opts

    def true_benefit(mask):
        return sum(benefits.get((l, m), 0.0) for l, m in enumerate(mask) if m != "sequential")

    best_mask, best_benefit = ["sequential"] * n_layers, 0.0

    def consider(mask):
        nonlocal best_mask, best_benefit
        cost = evaluate_true_cost(mask, base_costs, alt_modes, alpha, beta, gamma,
                                  seq_penalty=seq_penalty)
        if cost <= budget + 1e-9:
            b = true_benefit(mask)
            if b > best_benefit:
                best_mask, best_benefit = mask, b
                return True
        return False

    # Candidate 0: plain DP on unlinearized base costs (the non-iterative baseline).
    plain_layers = [{"sequential": (0.0, 0.0),
                      **{k: (base_costs[(l, k)], benefits.get((l, k), 0.0))
                         for k in alt_modes if (l, k) in base_costs}}
                     for l in range(n_layers)]
    plain = compile_multichoice_dp(plain_layers, budget, resolution=resolution, max_free=max_free)
    consider(plain["choices"])

    for prior_mask in (extra_candidates or []):
        consider(prior_mask)

    # Real bug found and fixed this session (C30/F103): seeding every mode's
    # count-guess from best_mask (all-sequential if candidate 0 was
    # rejected, giving n=0 for every mode) makes the M2 attenuation
    # multiplier max(0,1)^-beta = 1, the UNATTENUATED per-layer cost, for
    # every mode not currently being swept. Combined with
    # compile_multichoice_dp's unconditional free-pick step, this lets the
    # very first joint solve pack in layers from several modes at once
    # under an overly optimistic linearization, which evaluate_true_cost
    # then correctly rejects as infeasible -- and since nothing is ever
    # accepted, no mode's guess ever moves off 0: a genuine stuck fixed
    # point, not a property of the fitted cost model. Fix: seed every
    # mode's guess PESSIMISTICALLY at n_layers (worst-case attenuation)
    # instead, so the first joint solve is conservative/feasible and the
    # sweep relaxes counts upward from there. Also cap max_free (rather
    # than leaving it unconditional) when the caller didn't specify one --
    # both changes are jointly necessary; either alone still degenerates
    # to all-sequential (verified experimentally this session).
    n_k_guess = {k: n_layers for k in alt_modes}
    if max_free is None:
        max_free = 2
    current_mask = list(best_mask)

    for _ in range(n_passes):
        improved_this_pass = False
        for k in alt_modes:
            best_n_for_k = n_k_guess[k]
            for candidate_n in range(0, n_layers + 1):
                trial_guess = dict(n_k_guess, **{k: candidate_n})
                layer_options = build_layer_options(trial_guess, current_mask)
                result = compile_multichoice_dp(layer_options, budget,
                                                 resolution=resolution, max_free=max_free)
                if consider(result["choices"]):
                    improved_this_pass = True
                    best_n_for_k = result["choices"].count(k)
            n_k_guess[k] = best_n_for_k
            current_mask = list(best_mask)
        if not improved_this_pass:
            break

    return {
        "mask": best_mask,
        "predicted_cost": evaluate_true_cost(best_mask, base_costs, alt_modes, alpha, beta, gamma),
        "predicted_benefit": best_benefit,
    }


def interpolate_bracketed_baseline(brackets, position):
    """Linear interpolation between periodic baseline re-measurements taken
    at known positions during a long sweep -- corrects for continuous
    thermal/clock drift over the sweep's duration, which a one-time upfront
    warmup cannot fix (real finding, this session: CUDA latency drift
    persisted at its original magnitude even with a 40-iteration warmup
    burst before measurement started -- the drift accumulates throughout
    the whole sweep, not just at the start, so it needs a baseline measured
    close in time to each probe, not one fixed value).

    `brackets`: list of (position, baseline_value) pairs, sorted by
    position. `position` is any sweep-order coordinate (e.g. layer index).
    Positions before the first or after the last bracket use the nearest
    bracket's value (no extrapolation past measured data).
    """
    if position <= brackets[0][0]:
        return brackets[0][1]
    if position >= brackets[-1][0]:
        return brackets[-1][1]
    for (p0, v0), (p1, v1) in zip(brackets, brackets[1:]):
        if p0 <= position <= p1:
            frac = (position - p0) / (p1 - p0)
            return v0 + frac * (v1 - v0)
    raise ValueError(f"position {position} not covered by brackets {brackets}")


def fit_additive_composition(masks, targets, n_layers, alt_modes, ridge=1e-6):
    """Least-squares fit of target ~ sum_l cost[l, mode_l].

    Generalizes eval_6mode_masks.py's fit_additive to any number of
    alternative modes per layer (2 for ternary, 5 for 6-mode, N in general) --
    that script hardcoded ALT_MODES to the 6-mode set and had to be
    imported-and-asserted-equal from other scripts to reuse safely. This is
    the single source of truth; `alt_modes` is required, not inferred.

    Design matrix is one-hot over (layer, alt-mode) pairs; "sequential" is the
    reference level and contributes zero, so the fitted coefficients are
    exactly the per-(layer, mode) costs. Returns a flat array of length
    n_layers * len(alt_modes), indexed as `coef[l * len(alt_modes) + j]` for
    the j-th alt_modes entry at layer l.
    """
    idx = {(l, m): l * len(alt_modes) + j
           for l in range(n_layers) for j, m in enumerate(alt_modes)}
    X = np.zeros((len(masks), n_layers * len(alt_modes)))
    for i, mask in enumerate(masks):
        for l, m in enumerate(mask):
            if m != "sequential":
                X[i, idx[(l, m)]] = 1.0
    y = np.asarray(targets, dtype=float)
    A = X.T @ X + ridge * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ y)


def fit_composition_scale(predicted, actual):
    predicted = np.asarray(predicted, dtype=float)
    actual = np.asarray(actual, dtype=float)
    denominator = float(np.dot(predicted, predicted))
    return float(np.dot(predicted, actual) / denominator) if denominator > 0 else 1.0


def fit_count_adjusted_scale(predicted_sums, actual, n_parallel):
    """Fit the count-adjusted composition model: D(m) ≈ α₀ * Σd * |m|^(-β).

    Returns (alpha0, beta) minimizing MSE over the multi-layer graph set.
    """
    predicted_sums = np.asarray(predicted_sums, dtype=float)
    actual = np.asarray(actual, dtype=float)
    n_parallel = np.asarray(n_parallel, dtype=float)

    def loss(params):
        a, b = params
        pred = a * predicted_sums * np.power(np.maximum(n_parallel, 1.0), -b)
        return float(np.mean((actual - pred) ** 2))

    best = None
    for a0 in [0.8, 1.0, 1.2, 1.5]:
        for b0 in [0.1, 0.2, 0.3, 0.4]:
            res = _minimize(loss, [a0, b0], method="Nelder-Mead",
                            options={"xatol": 1e-8, "fatol": 1e-12, "maxiter": 10000})
            if best is None or res.fun < best.fun:
                best = res
    return float(best.x[0]), float(best.x[1])


def single_layer_effects(rows, field):
    n_layers = len(rows[0]["bits"])
    baseline = next(row[field] for row in rows if sum(row["bits"]) == 0)
    effects = np.zeros(n_layers, dtype=float)
    found = np.zeros(n_layers, dtype=bool)
    for row in rows:
        if sum(row["bits"]) == 1:
            layer = row["bits"].index(1)
            effects[layer] = max(0.0, row[field] - baseline)
            found[layer] = True
    if not found.all():
        missing = np.where(~found)[0].tolist()
        raise ValueError(f"Missing single-layer graph effects: {missing}")
    return effects


def additive_effect(bits, effects, scale=1.0):
    return float(scale * np.dot(np.asarray(bits, dtype=float), effects))


def compile_graph(effect_costs, latency_savings, budget, scale=1.0):
    effect_costs = np.maximum(np.asarray(effect_costs, dtype=float), 0.0)
    latency_savings = np.maximum(np.asarray(latency_savings, dtype=float), 0.0)
    if effect_costs.shape != latency_savings.shape:
        raise ValueError("Effect costs and latency savings must have equal shape")
    best = None
    for bits in itertools.product((0, 1), repeat=len(effect_costs)):
        predicted_effect = additive_effect(bits, effect_costs, scale)
        if predicted_effect > budget + 1e-12:
            continue
        predicted_saving = float(np.dot(bits, latency_savings))
        candidate = {
            "bits": list(bits),
            "predicted_effect": predicted_effect,
            "predicted_saving": predicted_saving,
        }
        if best is None or (
            candidate["predicted_saving"], -candidate["predicted_effect"]
        ) > (
            best["predicted_saving"], -best["predicted_effect"]
        ):
            best = candidate
    return best


def compile_graph_greedy(effect_costs, latency_savings, budget, scale=1.0):
    """O(L log L) graph compiler for arbitrary layer counts.

    For uniform latency_savings (all equal), this is provably optimal:
    greedy-by-cost maximizes item count under a weight budget when all
    items have equal value.

    For non-uniform savings, uses greedy-by-ratio (savings/cost),
    which is optimal for the fractional relaxation and a good
    approximation for the 0-1 case. Use compile_graph_dp for
    tighter solutions with non-uniform savings.
    """
    effect_costs = np.maximum(np.asarray(effect_costs, dtype=float), 0.0)
    latency_savings = np.maximum(np.asarray(latency_savings, dtype=float), 0.0)
    if effect_costs.shape != latency_savings.shape:
        raise ValueError("Effect costs and latency savings must have equal shape")
    n_layers = len(effect_costs)

    uniform = np.allclose(latency_savings[latency_savings > 0],
                          latency_savings[latency_savings > 0].mean(),
                          rtol=1e-6) if latency_savings.sum() > 0 else True

    if uniform:
        order = np.argsort(effect_costs)
    else:
        ratios = np.where(effect_costs > 1e-12,
                          latency_savings / effect_costs, np.inf)
        order = np.argsort(-ratios)

    bits = [0] * n_layers
    total_cost = 0.0
    for layer in order:
        if latency_savings[layer] <= 0:
            continue
        new_cost = total_cost + effect_costs[layer]
        if scale * new_cost <= budget + 1e-12:
            bits[int(layer)] = 1
            total_cost = new_cost

    return {
        "bits": bits,
        "predicted_effect": float(scale * total_cost),
        "predicted_saving": float(np.dot(bits, latency_savings)),
    }


def compile_graph_dp(effect_costs, latency_savings, budget, scale=1.0,
                     resolution=1000):
    """Discretized 0-1 knapsack via DP. O(L * resolution) time and space.

    Handles non-uniform latency savings correctly, unlike greedy-by-ratio.
    Scales to L=128+ in milliseconds. The solution is exact for the
    discretized cost grid; continuous costs are rounded to `resolution`
    bins, so the true optimality gap is at most one bin width
    (budget / resolution) per selected layer.
    """
    effect_costs = np.maximum(np.asarray(effect_costs, dtype=float), 0.0)
    latency_savings = np.maximum(np.asarray(latency_savings, dtype=float), 0.0)
    if effect_costs.shape != latency_savings.shape:
        raise ValueError("Effect costs and latency savings must have equal shape")
    n_layers = len(effect_costs)

    scaled_costs = effect_costs * scale
    max_cost = budget
    step = max_cost / resolution
    if step <= 0:
        return {"bits": [0] * n_layers, "predicted_effect": 0.0,
                "predicted_saving": 0.0}

    int_costs = np.round(scaled_costs / step).astype(int)
    int_budget = resolution

    dp = np.full(int_budget + 1, -np.inf)
    dp[0] = 0.0
    choice = np.zeros((n_layers, int_budget + 1), dtype=bool)

    for i in range(n_layers):
        if latency_savings[i] <= 0 or int_costs[i] <= 0:
            continue
        for cap in range(int_budget, int_costs[i] - 1, -1):
            val = dp[cap - int_costs[i]] + latency_savings[i]
            if val > dp[cap]:
                dp[cap] = val
                choice[i, cap] = True

    # Traceback from the capacity with maximum value (not necessarily int_budget)
    bits = [0] * n_layers
    cap = int(np.argmax(dp))
    for i in range(n_layers - 1, -1, -1):
        if choice[i, cap]:
            bits[i] = 1
            cap -= int_costs[i]

    total_cost = sum(effect_costs[i] for i in range(n_layers) if bits[i])
    return {
        "bits": bits,
        "predicted_effect": float(scale * total_cost),
        "predicted_saving": float(np.dot(bits, latency_savings)),
    }
