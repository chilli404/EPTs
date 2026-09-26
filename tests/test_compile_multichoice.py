"""compile_multichoice_dp: per-layer N-ary choice generalizing compile_graph_dp's
binary (skip/keep) choice to the 6-mode (or ternary) program space."""
from __future__ import annotations

import pytest

from fogen.execution_graph import (
    compile_iterative_dp,
    compile_multichoice_dp,
    evaluate_true_cost,
    fit_additive_composition,
    interpolate_bracketed_baseline,
)


def test_picks_free_option_unconditionally():
    """A cost<=0 option should always be taken, even with zero budget."""
    layers = [{"sequential": (0.0, 0.0), "attn_only": (-0.1, 5.0)}]
    result = compile_multichoice_dp(layers, budget=0.0)
    assert result["choices"] == ["attn_only"]
    assert result["predicted_benefit"] == pytest.approx(5.0)
    assert result["predicted_cost"] == pytest.approx(-0.1)


def test_zero_budget_no_free_options_falls_back_to_sequential():
    layers = [{"sequential": (0.0, 0.0), "skip": (1.0, 10.0)}]
    result = compile_multichoice_dp(layers, budget=0.0)
    assert result["choices"] == ["sequential"]
    assert result["predicted_benefit"] == pytest.approx(0.0)


def test_respects_budget_across_layers():
    """Two layers, each with a costly high-benefit option; budget only
    affords one."""
    layers = [
        {"sequential": (0.0, 0.0), "skip": (1.0, 10.0)},
        {"sequential": (0.0, 0.0), "skip": (1.0, 10.0)},
    ]
    result = compile_multichoice_dp(layers, budget=1.0, resolution=100)
    assert result["predicted_cost"] <= 1.0 + 1e-6
    assert result["predicted_benefit"] == pytest.approx(10.0, rel=1e-2)
    assert result["choices"].count("skip") == 1


def test_prefers_better_ratio_within_budget():
    """Given a fixed budget, the DP should not leave a strictly better
    (higher benefit, equal or lower cost) option on the table."""
    layers = [
        {"sequential": (0.0, 0.0), "cheap": (0.5, 3.0), "costly": (0.9, 3.5)},
    ]
    result = compile_multichoice_dp(layers, budget=0.5, resolution=200)
    assert result["choices"] == ["cheap"]


def test_exactly_one_choice_per_layer():
    layers = [
        {"sequential": (0.0, 0.0), "a": (0.3, 1.0), "b": (0.7, 4.0)},
        {"sequential": (0.0, 0.0), "a": (0.2, 0.5)},
        {"sequential": (0.0, 0.0), "attn_only": (-0.05, 2.0)},
    ]
    result = compile_multichoice_dp(layers, budget=1.0, resolution=500)
    assert len(result["choices"]) == 3
    assert all(c is not None for c in result["choices"])
    # layer 2's free option must always be taken
    assert result["choices"][2] == "attn_only"


def test_max_free_caps_simultaneous_free_picks():
    """Regression guard for the 430M compiler failure: 5 layers each have an
    individually-free option; capping at 2 must take only the 2 best by
    benefit and leave the rest at sequential."""
    layers = [
        {"sequential": (0.0, 0.0), "skip": (0.0, float(v))}
        for v in [5, 1, 4, 2, 3]
    ]
    result = compile_multichoice_dp(layers, budget=0.0, max_free=2)
    assert result["choices"].count("skip") == 2
    # the two highest-benefit free options (5 and 4) must be the ones taken
    assert result["choices"][0] == "skip"  # benefit 5
    assert result["choices"][2] == "skip"  # benefit 4
    assert result["predicted_benefit"] == pytest.approx(9.0)


def test_max_free_none_is_unbounded_backward_compatible():
    layers = [{"sequential": (0.0, 0.0), "skip": (0.0, 1.0)} for _ in range(5)]
    result = compile_multichoice_dp(layers, budget=0.0, max_free=None)
    assert result["choices"].count("skip") == 5


def test_dropped_free_option_does_not_reappear_in_paid_dp():
    """A layer whose only affordable option is free-but-capped-out must fall
    back to sequential, not be silently re-admitted as a paid choice."""
    layers = [{"sequential": (0.0, 0.0), "skip": (0.0, 100.0)},
              {"sequential": (0.0, 0.0), "skip": (0.0, 1.0)}]
    result = compile_multichoice_dp(layers, budget=0.0, max_free=1)
    assert result["choices"] == ["skip", "sequential"]


def test_matches_binary_compile_graph_dp_as_special_case():
    """With options {sequential: (0,0), parallel: (cost, benefit)} per layer,
    this must recover the same selection as the binary compile_graph_dp."""
    from fogen.execution_graph import compile_graph_dp

    effect_costs = [0.1, 0.4, 0.2, 0.6]
    latency_savings = [2.0, 5.0, 1.0, 6.0]
    budget = 0.5

    binary = compile_graph_dp(effect_costs, latency_savings, budget, resolution=2000)
    layers = [{"sequential": (0.0, 0.0), "parallel": (c, s)}
              for c, s in zip(effect_costs, latency_savings)]
    multi = compile_multichoice_dp(layers, budget, resolution=2000)

    assert multi["predicted_benefit"] == pytest.approx(
        binary["predicted_saving"], rel=1e-3)
    assert multi["predicted_cost"] <= budget + 1e-6


def test_fit_additive_composition_recovers_known_coefficients():
    """Exact recovery when the data is genuinely additive and noise-free."""
    import numpy as np

    alt_modes = ["parallel", "skip"]
    n_layers = 3
    # true per-(layer, mode) cost table
    true_cost = {
        (0, "parallel"): 0.10, (0, "skip"): 0.30,
        (1, "parallel"): 0.05, (1, "skip"): 0.20,
        (2, "parallel"): 0.15, (2, "skip"): 0.40,
    }
    rng = np.random.default_rng(0)
    masks = []
    targets = []
    for _ in range(500):
        mask = [rng.choice(["sequential"] + alt_modes) for _ in range(n_layers)]
        masks.append(mask)
        targets.append(sum(true_cost.get((l, m), 0.0) for l, m in enumerate(mask)))

    coef = fit_additive_composition(masks, targets, n_layers, alt_modes, ridge=1e-9)
    for l in range(n_layers):
        for j, m in enumerate(alt_modes):
            assert coef[l * len(alt_modes) + j] == pytest.approx(
                true_cost[(l, m)], abs=1e-6)


def test_fit_additive_composition_two_alt_modes_matches_ternary_shape():
    """Ternary has 2 alt-modes (parallel, skip) per layer, not 6-mode's 5 --
    the generalized function must not hardcode a mode count."""
    masks = [["sequential", "parallel"], ["skip", "sequential"],
              ["parallel", "skip"], ["sequential", "sequential"]]
    targets = [0.1, 0.2, 0.3, 0.0]
    coef = fit_additive_composition(masks, targets, n_layers=2,
                                     alt_modes=["parallel", "skip"])
    assert len(coef) == 2 * 2  # n_layers * len(alt_modes)


def test_evaluate_true_cost_matches_hand_computation():
    """cost(m) = sum_k alpha_k * A_k(m) * max(n_k,1)^-beta_k + structural term.
    Single mode, no structural term: 2 of 3 layers use 'skip', c=0.01 each,
    alpha=1, beta=-1 (super-additive: n^-(-1) = n)."""
    mask = ["skip", "skip", "sequential"]
    base_costs = {(0, "skip"): 0.01, (1, "skip"): 0.01, (2, "skip"): 0.01}
    alpha = {"skip": 1.0}
    beta = {"skip": -1.0}
    cost = evaluate_true_cost(mask, base_costs, ["skip"], alpha, beta, gamma={})
    # A_skip = 0.01+0.01 = 0.02, n_skip=2, cost = 1.0 * 0.02 * 2^1 = 0.04
    assert cost == pytest.approx(0.04)


def test_iterative_dp_avoids_the_240x_style_blowup():
    """The exact C15 failure mode in miniature: naive per-layer costs look
    affordable for all 3 layers, but the TRUE (super-additive) cost of all 3
    exceeds budget while a plain multichoice_dp run on unlinearized base
    costs would happily predict it fits. The iterative solver must find the
    true optimum (2 layers), not the naively-predicted-affordable 3."""
    n_layers = 3
    base_costs = {(l, "skip"): 0.01 for l in range(n_layers)}
    benefits = {(l, "skip"): 1.0 for l in range(n_layers)}
    alpha = {"skip": 1.0}
    beta = {"skip": -1.0}  # super-additive: cost(m) = 0.01*n_skip * n_skip = 0.01*n^2...
    # actually A(m)=0.01*n_skip, cost = A*n_skip^1 = 0.01*n_skip^2
    # n=1: 0.01, n=2: 0.04, n=3: 0.09
    budget = 0.05

    result = compile_iterative_dp(
        base_costs, benefits, ["skip"], alpha, beta, gamma={},
        budget=budget, n_layers=n_layers, resolution=1000)

    true_cost = evaluate_true_cost(result["mask"], base_costs, ["skip"], alpha, beta, gamma={})
    assert true_cost <= budget + 1e-6, "solution must be truly feasible, not just naively-predicted feasible"
    assert result["mask"].count("skip") == 2, "true optimum is 2 layers (cost 0.04); naive DP would wrongly pick 3 (predicted 0.03, actual 0.09)"


def test_iterative_dp_never_worse_than_plain_dp():
    """Regression guard: even if the fixed-point iteration doesn't converge
    cleanly, the returned mask must be at least as good (by true cost/benefit)
    as what a single non-iterative DP pass on the base costs would give --
    the 'keep best feasible seen' tracking must never regress below iteration 0."""
    n_layers = 4
    base_costs = {(l, "skip"): 0.02 for l in range(n_layers)}
    benefits = {(l, "skip"): 1.0 for l in range(n_layers)}
    alpha = {"skip": 1.0}
    beta = {"skip": -0.5}
    budget = 0.1

    result = compile_iterative_dp(
        base_costs, benefits, ["skip"], alpha, beta, gamma={},
        budget=budget, n_layers=n_layers, resolution=1000)
    true_cost = evaluate_true_cost(result["mask"], base_costs, ["skip"], alpha, beta, gamma={})
    assert true_cost <= budget + 1e-6

    naive_layers = [{"sequential": (0.0, 0.0), "skip": (base_costs[(l, "skip")], benefits[(l, "skip")])}
                    for l in range(n_layers)]
    naive = compile_multichoice_dp(naive_layers, budget, resolution=1000)
    naive_mask = naive["choices"]
    naive_true_cost = evaluate_true_cost(naive_mask, base_costs, ["skip"], alpha, beta, gamma={})
    naive_true_benefit = sum(benefits[(l, "skip")] for l, m in enumerate(naive_mask) if m == "skip")
    if naive_true_cost <= budget + 1e-6:
        result_benefit = sum(benefits[(l, "skip")] for l, m in enumerate(result["mask"]) if m == "skip")
        assert result_benefit >= naive_true_benefit - 1e-6


def test_evaluate_true_cost_includes_structural_term():
    """Two 'skip' layers adjacent (distance 1) vs far apart (distance 5) must
    get different true costs when gamma penalizes the adjacent-skip-skip bin."""
    base_costs = {(l, "skip"): 0.0 for l in range(6)}  # isolate the structural term
    alpha = {"skip": 1.0}
    beta = {"skip": 0.0}
    gamma = {("skip", "skip", 0): 0.05,  # adjacent (dist=1) pair penalty
             ("skip", "skip", 1): 0.0, ("skip", "skip", 2): 0.0}

    adjacent = ["skip", "skip", "sequential", "sequential", "sequential", "sequential"]
    far = ["skip", "sequential", "sequential", "sequential", "sequential", "skip"]

    cost_adjacent = evaluate_true_cost(adjacent, base_costs, ["skip"], alpha, beta, gamma)
    cost_far = evaluate_true_cost(far, base_costs, ["skip"], alpha, beta, gamma)
    assert cost_adjacent == pytest.approx(0.05)
    assert cost_far == pytest.approx(0.0)


def test_iterative_dp_is_monotonic_across_budgets():
    """Regression for a real bug found on production data: a super-additive
    (negative beta) mode can have a fitted extrapolation quirk where the
    *coordinate sweep only finds a self-consistent fixed point* -- the guessed
    count and the DP's own resulting count under that guess agree -- at some
    budgets but not others, even though the true predicted_cost of that mask
    is negative (satisfies every budget >= 0). A wider budget must never do
    worse than a tighter one; `compile_iterative_dp` alone does not guarantee
    this (each call searches independently) unless prior results are carried
    forward as extra candidates.
    """
    n_layers = 20
    # Mirrors the shape of the real bug: skip has a fitted power-law cost
    # that goes slightly negative around count=8 (extrapolation artifact),
    # rises for smaller and larger counts.
    base_costs = {(l, "skip"): 0.0006 for l in range(n_layers)}
    benefits = {(l, "skip"): 1.0 for l in range(n_layers)}
    alpha = {"skip": 1.0}
    # beta chosen so max(n,1)^-beta dips slightly below 1/n at n=8 -- same
    # qualitative shape (mild negative-cost region) as the real fitted curve.
    beta = {"skip": -1.02}
    budgets = [0.0005, 0.002, 0.005, 0.02]

    prior_masks = []
    best_benefits = []
    for budget in budgets:
        result = compile_iterative_dp(base_costs, benefits, ["skip"], alpha, beta,
                                      gamma={}, budget=budget, n_layers=n_layers,
                                      resolution=1000, extra_candidates=prior_masks)
        true_cost = evaluate_true_cost(result["mask"], base_costs, ["skip"], alpha, beta, gamma={})
        assert true_cost <= budget + 1e-6
        best_benefits.append(result["predicted_benefit"])
        prior_masks.append(result["mask"])

    for i in range(1, len(best_benefits)):
        assert best_benefits[i] >= best_benefits[i - 1] - 1e-9, (
            f"budget {budgets[i]} (benefit {best_benefits[i]}) did worse than "
            f"budget {budgets[i-1]} (benefit {best_benefits[i-1]}) -- monotonicity violated")


def test_evaluate_true_cost_includes_sequential_penalty():
    """Real finding (this session): masks that leave ZERO sequential layers
    collapse catastrophically regardless of which alt-modes fill the rest --
    M2's per-mode-independent terms cannot represent a threshold on the
    TOTAL non-sequential count. seq_penalty = (gamma_seq, delta_seq) adds
    gamma_seq * max(n_sequential,1)^-delta_seq once, globally, not per-mode."""
    n_layers = 4
    base_costs = {(l, k): 0.0 for l in range(n_layers) for k in ["skip", "attn_only"]}
    alpha = {"skip": 1.0, "attn_only": 1.0}
    beta = {"skip": 0.0, "attn_only": 0.0}

    all_non_sequential = ["skip", "attn_only", "skip", "attn_only"]  # n_sequential=0
    one_sequential = ["sequential", "attn_only", "skip", "attn_only"]  # n_sequential=1

    gamma_seq, delta_seq = 1.0, 2.0
    cost_zero_seq = evaluate_true_cost(all_non_sequential, base_costs, ["skip", "attn_only"],
                                       alpha, beta, gamma={}, seq_penalty=(gamma_seq, delta_seq))
    cost_one_seq = evaluate_true_cost(one_sequential, base_costs, ["skip", "attn_only"],
                                      alpha, beta, gamma={}, seq_penalty=(gamma_seq, delta_seq))
    # Uses (n_sequential + 1) as the base, not max(n_sequential, 1) -- the
    # latter would clamp both 0 and 1 to the same value and fail to
    # distinguish them, defeating the entire point of the term.
    # (0+1)^-2 = 1, (1+1)^-2 = 0.25 -- must differ.
    assert cost_zero_seq == pytest.approx(gamma_seq * 1.0 ** (-delta_seq))
    assert cost_one_seq == pytest.approx(gamma_seq * 2.0 ** (-delta_seq))
    assert cost_zero_seq > cost_one_seq, (
        "a mask with zero sequential layers must cost strictly more than "
        "one with at least one sequential layer, all else equal")


def test_evaluate_true_cost_seq_penalty_defaults_to_noop():
    """Backward compatibility: omitting seq_penalty must not change any
    existing cost computation (all prior tests call evaluate_true_cost
    without it)."""
    mask = ["skip", "skip", "sequential"]
    base_costs = {(0, "skip"): 0.01, (1, "skip"): 0.01}
    alpha = {"skip": 1.0}
    beta = {"skip": -1.0}
    with_default = evaluate_true_cost(mask, base_costs, ["skip"], alpha, beta, gamma={})
    with_explicit_none = evaluate_true_cost(mask, base_costs, ["skip"], alpha, beta,
                                            gamma={}, seq_penalty=None)
    assert with_default == pytest.approx(with_explicit_none)


def test_iterative_dp_avoids_zero_sequential_when_penalty_is_severe():
    """The decision-relevant test: given a severe sequential-starvation
    penalty, the compiler must not pick a mask that zeroes out every
    sequential layer, even if the naive per-mode costs alone would look
    affordable -- this is the exact real failure found on 1B production
    data (skip+attn_only combos summing to all 28 layers collapsed by
    0.5-1.2 BPB regardless of the specific mix)."""
    n_layers = 6
    base_costs = {(l, k): 0.001 for l in range(n_layers) for k in ["skip", "attn_only"]}
    benefits = {(l, k): 1.0 for l in range(n_layers) for k in ["skip", "attn_only"]}
    alpha = {"skip": 1.0, "attn_only": 1.0}
    beta = {"skip": 0.0, "attn_only": 0.0}
    # Severe: going from 1 remaining sequential layer to 0 costs +10 BPB.
    seq_penalty = (10.0, 3.0)
    budget = 0.02  # affordable for ALL 6 layers under the naive per-mode cost alone

    result = compile_iterative_dp(base_costs, benefits, ["skip", "attn_only"], alpha, beta,
                                  gamma={}, budget=budget, n_layers=n_layers,
                                  resolution=1000, seq_penalty=seq_penalty)
    n_sequential = result["mask"].count("sequential")
    assert n_sequential >= 1, (
        f"picked {result['mask']} with 0 sequential layers despite a severe "
        f"starvation penalty -- the fix did not take effect")


def test_interpolate_bracketed_baseline_corrects_linear_drift():
    """Real finding, this session: a single upfront baseline plus a fixed
    warmup burst does NOT fix CUDA latency drift -- it's continuous thermal
    accumulation over the sweep's full duration, not a settle-then-plateau
    effect. The fix is to re-measure baseline periodically during the sweep
    and interpolate, not warm up harder beforehand. This test verifies the
    pure interpolation logic against synthetic linear drift: with bracket
    points every 4 "layers", a probe's true (drift-free) delta must be
    recovered even though raw measurements drift substantially."""
    # Synthetic: baseline itself drifts +0.1ms per layer-index of sweep time,
    # e.g. bracket measurements taken at sweep positions 0, 4, 8, 12.
    brackets = [(0, 100.0), (4, 100.4), (8, 100.8), (12, 101.2)]
    # A probe measured at sweep position 6 with a TRUE (drift-free) delta of
    # +2.0ms would show raw_time = interpolated_baseline_at_6 + 2.0.
    interpolated_at_6 = interpolate_bracketed_baseline(brackets, 6)
    assert interpolated_at_6 == pytest.approx(100.6, abs=1e-9)  # halfway between 100.4 and 100.8
    raw_time = interpolated_at_6 + 2.0
    recovered_delta = raw_time - interpolate_bracketed_baseline(brackets, 6)
    assert recovered_delta == pytest.approx(2.0, abs=1e-9)


def test_interpolate_bracketed_baseline_extrapolates_at_edges():
    """A probe measured before the first or after the last bracket point
    must use the nearest bracket's value, not fail or extrapolate wildly."""
    brackets = [(4, 100.4), (8, 100.8)]
    assert interpolate_bracketed_baseline(brackets, 0) == pytest.approx(100.4)
    assert interpolate_bracketed_baseline(brackets, 20) == pytest.approx(100.8)
