"""Global fuzzy candidate search over the official vessel catalog only."""
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from baseline_lib import normalize_name
from nonhangul_lib import FuzzyNonHangulIndex
from romanized_lib import RomanIndex


class GlobalFuzzyIndex(FuzzyNonHangulIndex):
    def __init__(self, catalog):
        super().__init__(catalog)
        self._name_cache = {}

    def global_candidates(self, row):
        query = normalize_name(row["vessel_name"])
        if query not in self._name_cache:
            self._name_cache[query] = [i for _, _, i in process.extract(
                query, self.names, scorer=Levenshtein.normalized_similarity, limit=64)]
        return self._name_cache[query]

    def extra_candidates(self, row):
        return list(dict.fromkeys(super().extra_candidates(row) + self.global_candidates(row)))


def make_training_pairs_global(fit, index):
    import numpy as np

    x, y = [], []
    for row in fit.to_dict("records"):
        true = index.id_to_index[row["target_vessel_id"]]
        baseline = [i for i in RomanIndex.candidates(index, row) if i != true][:20]
        v2 = [i for i in FuzzyNonHangulIndex.extra_candidates(index, row)
              if i != true and i not in baseline][:20]
        used = set(baseline + v2 + [true])
        global_negatives = [i for i in index.global_candidates(row) if i not in used][:20]
        for candidate in [true] + baseline + v2 + global_negatives:
            x.append(index.pair_features(row, candidate))
            y.append(int(candidate == true))
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int8)
