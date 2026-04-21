from __future__ import annotations

import numpy as np

from hlda_cv import fit_hlda, prune


def test_fit_hlda_prefers_separating_axis() -> None:
    x_a = [
        [0.0, 0.0, 1.0],
        [0.1, 0.1, 1.1],
        [-0.1, -0.1, 0.9],
        [0.05, 0.0, 1.0],
    ]
    x_b = [
        [2.0, 0.0, 1.0],
        [2.1, 0.1, 1.1],
        [1.9, -0.1, 0.9],
        [2.05, 0.0, 1.0],
    ]

    weights, eigenvalue = fit_hlda(x_a, x_b, ["x", "y", "z"])

    assert eigenvalue > 0
    assert abs(weights["x"]) > abs(weights["y"])
    assert abs(weights["x"]) > abs(weights["z"])


def test_prune_drops_redundant_descriptor() -> None:
    x_a = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.1],
            [2.0, 2.0, -0.1],
            [3.0, 3.0, 0.0],
        ]
    )
    x_b = np.array(
        [
            [4.0, 4.0, 0.0],
            [5.0, 5.0, 0.2],
            [6.0, 6.0, -0.2],
            [7.0, 7.0, 0.0],
        ]
    )

    kept_cols, keep_idx = prune(x_a, x_b, ["x", "x_copy", "noise"], threshold=0.95)

    assert "noise" in kept_cols
    assert len(kept_cols) == 2
    assert len(keep_idx) == 2
    assert sum(col in {"x", "x_copy"} for col in kept_cols) == 1
