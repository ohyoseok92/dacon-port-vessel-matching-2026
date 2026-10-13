"""Make a fitted LightGBM ranker usable by the shared row-independent scorer."""

import numpy as np


class RankerAdapter:
    def __init__(self, model):
        self.model = model

    def predict_proba(self, matrix):
        scores = self.model.predict(matrix)
        return np.column_stack((np.zeros_like(scores), scores))
