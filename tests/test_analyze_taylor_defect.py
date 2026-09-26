import torch

from scripts.analyze_taylor_defect import tensor_metrics


def test_tensor_metrics_identifies_exact_approximation():
    actual = torch.tensor([1.0, 2.0, 3.0])

    result = tensor_metrics(actual, actual)

    assert abs(result["cosine"] - 1.0) < 1e-7
    assert result["relative_error"] == 0.0
    assert result["norm_ratio"] == 1.0
