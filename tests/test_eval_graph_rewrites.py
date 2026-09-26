import numpy as np

from scripts.eval_graph_rewrites import graph_masks


def test_four_layer_masks_are_exhaustive():
    masks = graph_masks(4, np.ones(4), 0)

    assert len(masks) == 16
    assert (0, 0, 0, 0) in masks
    assert (1, 1, 1, 1) in masks


def test_twelve_layer_masks_are_deterministic_and_cover_key_graphs():
    defects = np.arange(12, dtype=float)

    first = graph_masks(12, defects, 64)
    second = graph_masks(12, defects, 64)

    assert first == second
    assert tuple([0] * 12) in first
    assert tuple([1] * 12) in first
    assert tuple([1] * 6 + [0] * 6) in first
    assert tuple([0] * 6 + [1] * 6) in first
    assert tuple(index % 2 for index in range(12)) in first
    assert all(tuple(int(index == layer) for index in range(12)) in first
               for layer in range(12))
