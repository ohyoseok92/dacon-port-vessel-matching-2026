"""Blend two fitted row-local LightGBM rankers for shared predict_ids()."""

import numpy as np


class RankerBlendAdapter:
    def __init__(self, base, second, second_weight=0.75):
        if not 0.0 <= second_weight <= 1.0:
            raise ValueError("second_weight must be in [0, 1]")
        self.base = base
        self.second = second
        self.second_weight = second_weight

    def predict_proba(self, matrix):
        a = np.asarray(self.base.predict(matrix), dtype=float)
        b = np.asarray(self.second.predict(matrix), dtype=float)
        scores = (1.0 - self.second_weight) * a + self.second_weight * b
        return np.column_stack((np.zeros_like(scores), scores))
