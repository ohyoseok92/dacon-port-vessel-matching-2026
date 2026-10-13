"""Additional row-local romanized-name similarity ranking features."""

from rapidfuzz import fuzz

from baseline_lib import is_hangul_name, positive_float
from dac_v2_phonetic import romanize
from tonnage_filtered_lib import TonnageFilteredIndex, FEATURE_NAMES as TONNAGE_FEATURE_NAMES


FEATURE_NAMES = TONNAGE_FEATURE_NAMES + ("raw_roman_ratio", "raw_roman_vs_local_best")


class RawRatioIndex(TonnageFilteredIndex):
    def pair_features(self, row, candidate_index):
        base = super().pair_features(row, candidate_index)
        if not is_hangul_name(row.get("vessel_name")):
            return base + [0.0, 0.0]
        raw = romanize(row["vessel_name"])
        score = fuzz.ratio(raw, self.names[candidate_index]) / 100
        local_raw, _ = self.local_name_lists(
            row["vessel_name"], positive_float(row.get("gross_tonnage")),
            positive_float(row.get("international_gross_tonnage"))
        )
        best = fuzz.ratio(raw, self.names[local_raw[0]]) / 100 if local_raw else 0.0
        return base + [score, score - best]
