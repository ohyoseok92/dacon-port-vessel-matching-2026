"""Tonnage-local name retrieval and row-local ranking features."""

from functools import lru_cache

import numpy as np
from rapidfuzz import fuzz

from baseline_lib import is_hangul_name, positive_float
from dac_v2_phonetic import romanize
from rank_position_lib import RankPositionIndex, FEATURE_NAMES as RANK_FEATURE_NAMES


FEATURE_NAMES = RANK_FEATURE_NAMES + (
    "local_raw_present", "local_raw_rank", "local_phonetic_present", "local_phonetic_rank"
)


class TonnageFilteredIndex(RankPositionIndex):
    @lru_cache(maxsize=4096)
    def local_name_lists(self, query: str, gross: float | None, international: float | None):
        values = [v for v in (gross, international) if v is not None]
        if not values:
            return (), ()
        near = np.zeros(len(self.gt), dtype=bool)
        for value in values:
            near |= np.isfinite(self.gt) & (self.gt > 0) & (
                np.abs(self.gt - value) / np.maximum(self.gt, value) <= .08
            )
        pool = np.flatnonzero(near)
        raw = romanize(query)
        phon = self.query_key(query)
        by_raw = sorted(pool, key=lambda i: (-fuzz.ratio(raw, self.names[i]), i))[:32]
        by_phon = sorted(pool, key=lambda i: (-fuzz.ratio(phon, self.phonetic_names[i]), i))[:32]
        return tuple(int(i) for i in by_raw), tuple(int(i) for i in by_phon)

    def local_name_candidates(self, query: str, gross: float | None, international: float | None):
        raw, phon = self.local_name_lists(query, gross, international)
        return raw + phon

    def candidates(self, row, name_limit=16, tonnage_limit=12, max_candidates=40):
        base = super().candidates(row, name_limit, tonnage_limit, max_candidates)
        if not is_hangul_name(row.get("vessel_name")):
            return base
        extra = self.local_name_candidates(
            row["vessel_name"], positive_float(row.get("gross_tonnage")),
            positive_float(row.get("international_gross_tonnage"))
        )
        used = set(base)
        for i in extra:
            if i not in used:
                base.append(i)
                used.add(i)
        return base

    def pair_features(self, row, candidate_index):
        base = super().pair_features(row, candidate_index)
        if not is_hangul_name(row.get("vessel_name")):
            return base + [0.0, 0.0, 0.0, 0.0]
        raw, phon = self.local_name_lists(
            row["vessel_name"], positive_float(row.get("gross_tonnage")),
            positive_float(row.get("international_gross_tonnage"))
        )
        raw_rank = raw.index(candidate_index) if candidate_index in raw else None
        phon_rank = phon.index(candidate_index) if candidate_index in phon else None
        return base + [float(raw_rank is not None), 0.0 if raw_rank is None else 1 - raw_rank / 32,
                       float(phon_rank is not None), 0.0 if phon_rank is None else 1 - phon_rank / 32]
