import numpy as np
import torch

from scripts.eval_prompt_graph_compiler import conformal_scale, mean_token_kl, random_mask


def test_conformal_scale_uses_upper_ratio_quantile():
    scale = conformal_scale([1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0], 0.75)

    assert scale == 4.0


def test_mean_token_kl_is_zero_for_equal_logits():
    logits = torch.tensor([[[1.0, 2.0, 3.0]]])

    assert abs(mean_token_kl(logits, logits)) < 1e-7


def test_random_mask_has_requested_number_of_layers():
    rng = np.random.default_rng(2026)

    mask = random_mask(12, 5, rng)

    assert len(mask) == 12
    assert sum(mask) == 5
