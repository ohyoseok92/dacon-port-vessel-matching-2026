"""Official-catalog-only, row-independent non-Hangul vessel matcher."""
from collections import Counter

import numpy as np
from rapidfuzz.distance import Levenshtein

from baseline_lib import grams, is_hangul_name, normalize_name, positive_float, relative_gap
from romanized_lib import RomanIndex

FEATURE_NAMES = ("name_levenshtein_similarity",)


class FuzzyNonHangulIndex(RomanIndex):
    def extra_candidates(self, row):
        query = normalize_name(row["vessel_name"])
        if not query:
            return []
        gross = positive_float(row.get("gross_tonnage"))
        international = positive_float(row.get("international_gross_tonnage"))
        name_counts = Counter()
        for gram in sorted(grams(query, 3)):
            name_counts.update(self.trigram_postings.get(gram, ()))
        top_name = sorted(name_counts, key=lambda i: (-name_counts[i], i))[:256]
        top_name.sort(key=lambda i: (-Levenshtein.normalized_similarity(query, self.names[i]), i))
        near_tonnage = list(dict.fromkeys(self._nearest_gt(gross, 256) + self._nearest_gt(international, 256)))
        near_tonnage.sort(key=lambda i: (
            -(0.7 * Levenshtein.normalized_similarity(query, self.names[i])
              + 0.3 * (1. - min(relative_gap(gross, positive_float(self.gt[i])),
                                  relative_gap(international, positive_float(self.gt[i]))))), i))
        return list(dict.fromkeys(top_name[:32] + near_tonnage[:32]))

    def candidates(self, row, name_limit=16, tonnage_limit=12, max_candidates=40):
        base = super().candidates(row, name_limit, tonnage_limit, max_candidates)
        if is_hangul_name(row.get("vessel_name")):
            return base
        used = set(base)
        for i in self.extra_candidates(row):
            if i not in used:
                base.append(i)
                used.add(i)
        return base

    def pair_features(self, row, candidate_index):
        return super().pair_features(row, candidate_index) + [
            Levenshtein.normalized_similarity(normalize_name(row.get("vessel_name")),
                                              self.names[candidate_index])]


def make_training_pairs(fit, index):
    """Positive plus old and fuzzy negatives, from the fit partition only."""
    features, labels = [], []
    for row in fit.to_dict("records"):
        if is_hangul_name(row["vessel_name"]):
            raise ValueError("Non-Hangul model received a Hangul training row")
        true = index.id_to_index[row["target_vessel_id"]]
        baseline = [i for i in RomanIndex.candidates(index, row) if i != true][:20]
        extra = [i for i in index.extra_candidates(row) if i != true and i not in baseline][:20]
        for candidate in [true] + baseline + extra:
            features.append(index.pair_features(row, candidate))
            labels.append(int(candidate == true))
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int8)
