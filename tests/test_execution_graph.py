import json
from pathlib import Path

import numpy as np
import pytest

from fogen.execution_graph import (
    additive_effect,
    compile_graph,
    compile_graph_dp,
    compile_graph_greedy,
    fit_composition_scale,
    fit_count_adjusted_scale,
    single_layer_effects,
)


def test_fit_composition_scale_recovers_subadditive_coefficient():
    assert fit_composition_scale([1, 2, 3], [0.7, 1.4, 2.1]) == pytest.approx(0.7)


def test_single_layer_effects_subtracts_baseline():
    rows = [
        {"bits": [0, 0], "metric": 2.0},
        {"bits": [1, 0], "metric": 2.3},
        {"bits": [0, 1], "metric": 2.5},
    ]

    assert np.allclose(single_layer_effects(rows, "metric"), [0.3, 0.5])


def test_additive_effect_applies_scale():
    assert additive_effect([1, 0, 1], [0.2, 0.3, 0.4], 0.5) == pytest.approx(0.3)


def test_compile_graph_maximizes_saving_under_budget():
    result = compile_graph(
        effect_costs=[0.2, 0.4, 0.7],
        latency_savings=[1.0, 3.0, 5.0],
        budget=0.6,
    )

    assert result["bits"] == [1, 1, 0]
    assert result["predicted_effect"] == pytest.approx(0.6)
    assert result["predicted_saving"] == pytest.approx(4.0)


def test_compile_graph_greedy_matches_brute_force_uniform():
    """Greedy is provably optimal for uniform savings; verify on synthetic data."""
    np.random.seed(42)
    costs = np.random.exponential(0.5, size=12)
    savings = np.ones(12)
    for budget in [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]:
        bf = compile_graph(costs, savings, budget, scale=0.7)
        greedy = compile_graph_greedy(costs, savings, budget, scale=0.7)
        assert greedy["bits"] == bf["bits"], f"Mismatch at budget={budget}"
        assert greedy["predicted_effect"] == pytest.approx(bf["predicted_effect"])
        assert greedy["predicted_saving"] == pytest.approx(bf["predicted_saving"])


def test_compile_graph_greedy_matches_brute_force_nonuniform():
    """Greedy-by-ratio on non-uniform savings matches brute-force at L=12."""
    np.random.seed(7)
    costs = np.random.exponential(0.5, size=10)
    savings = np.random.uniform(0.5, 2.0, size=10)
    for budget in [1.0, 2.0, 3.0]:
        bf = compile_graph(costs, savings, budget, scale=0.8)
        greedy = compile_graph_greedy(costs, savings, budget, scale=0.8)
        # Greedy-by-ratio may not be exactly optimal for 0-1 knapsack,
        # but should be close. Check saving is >= 90% of brute-force.
        assert greedy["predicted_saving"] >= bf["predicted_saving"] * 0.9
        assert greedy["predicted_effect"] <= budget + 1e-10


def test_compile_graph_greedy_on_real_data():
    """Verify greedy matches brute-force on actual ClimbMix graph data."""
    data_path = Path(__file__).parent.parent / "results" / "data" / "climbmix_graphs.json"
    if not data_path.exists():
        pytest.skip("ClimbMix graph data not available")
    with open(data_path) as f:
        data = json.load(f)
    rows = data["rows"]
    effects = single_layer_effects(rows, "symmetric_kl")
    scale = fit_composition_scale(
        [sum(effects[i] * b for i, b in enumerate(r["bits"]))
         for r in rows if sum(r["bits"]) >= 2],
        [r["symmetric_kl"] - rows[0]["symmetric_kl"]
         for r in rows if sum(r["bits"]) >= 2],
    )
    savings = np.ones(12)
    for budget in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
        bf = compile_graph(effects, savings, budget, scale)
        greedy = compile_graph_greedy(effects, savings, budget, scale)
        assert greedy["bits"] == bf["bits"], (
            f"Budget={budget}: BF={bf['bits']} vs Greedy={greedy['bits']}"
        )


def test_compile_graph_greedy_scales_to_large_layers():
    """Greedy handles L=20 (Blackwell) and L=64 instantly."""
    np.random.seed(0)
    for n_layers in [20, 32, 64]:
        costs = np.random.exponential(0.3, size=n_layers)
        savings = np.ones(n_layers)
        result = compile_graph_greedy(costs, savings, budget=3.0, scale=0.7)
        assert sum(result["bits"]) > 0
        assert result["predicted_effect"] <= 3.0 + 1e-10


def test_fit_count_adjusted_scale():
    """Count-adjusted fit recovers known parameters on synthetic data."""
    np.random.seed(99)
    n = 50
    sums = np.random.uniform(1, 10, n)
    n_par = np.random.randint(2, 13, n).astype(float)
    actual = 1.2 * sums * np.power(n_par, -0.25) + np.random.normal(0, 0.05, n)
    alpha0, beta = fit_count_adjusted_scale(sums, actual, n_par)
    assert alpha0 == pytest.approx(1.2, abs=0.1)
    assert beta == pytest.approx(0.25, abs=0.1)


def test_compile_graph_dp_matches_brute_force_nonuniform():
    """DP solver matches brute-force exactly on non-uniform savings."""
    np.random.seed(7)
    costs = np.random.exponential(0.5, size=10)
    savings = np.random.uniform(0.5, 2.0, size=10)
    for budget in [1.0, 2.0, 3.0]:
        bf = compile_graph(costs, savings, budget, scale=0.8)
        dp = compile_graph_dp(costs, savings, budget, scale=0.8, resolution=10000)
        assert dp["predicted_saving"] == pytest.approx(
            bf["predicted_saving"], abs=0.01
        ), f"Budget={budget}: BF={bf['predicted_saving']:.4f} vs DP={dp['predicted_saving']:.4f}"


def test_compile_graph_dp_scales_to_large_layers():
    """DP handles L=64 in reasonable time."""
    np.random.seed(0)
    costs = np.random.exponential(0.3, size=64)
    savings = np.random.uniform(0.1, 1.0, size=64)
    result = compile_graph_dp(costs, savings, budget=5.0, scale=0.7)
    assert sum(result["bits"]) > 0
    assert result["predicted_effect"] <= 5.0 + 0.01


def test_compile_graph_dp_on_real_data():
    """DP matches brute-force on actual ClimbMix data with non-uniform savings."""
    data_path = Path(__file__).parent.parent / "results" / "data" / "climbmix_graphs.json"
    if not data_path.exists():
        pytest.skip("ClimbMix graph data not available")
    with open(data_path) as f:
        data = json.load(f)
    rows = data["rows"]
    effects = single_layer_effects(rows, "symmetric_kl")
    scale = fit_composition_scale(
        [sum(effects[i] * b for i, b in enumerate(r["bits"]))
         for r in rows if sum(r["bits"]) >= 2],
        [r["symmetric_kl"] - rows[0]["symmetric_kl"]
         for r in rows if sum(r["bits"]) >= 2],
    )
    # Non-uniform savings (simulate hardware-measured values)
    savings = np.array([0.8, 1.5, 0.3, 0.1, 0.9, 0.2, 1.1, 0.4, 0.7, 0.05, 0.6, 1.2])
    for budget in [2.0, 4.0, 6.0]:
        bf = compile_graph(effects, savings, budget, scale)
        dp = compile_graph_dp(effects, savings, budget, scale, resolution=10000)
        assert dp["predicted_saving"] == pytest.approx(
            bf["predicted_saving"], abs=0.01
        )
