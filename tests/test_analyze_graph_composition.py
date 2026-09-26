from scripts.analyze_graph_composition import composition_metrics


def test_additive_graph_effect_has_perfect_composition_metrics():
    rows = []
    effects = [1.0, 2.0, 3.0]
    for mask in range(8):
        bits = [(mask >> index) & 1 for index in range(3)]
        rows.append({
            "bits": bits,
            "effect": sum(effect for effect, bit in zip(effects, bits) if bit),
        })

    result = composition_metrics(rows, "effect")

    assert result["single_layers_found"] == 3
    assert result["spearman"] == 1.0
    assert result["pearson"] == 1.0
    assert result["slope"] == 1.0
    assert result["relative_rmse"] == 0.0
