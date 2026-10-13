"""Row-local phonetic retrieval-rank features for the Hangul LambdaRank model."""

from functools import lru_cache

from rapidfuzz import fuzz

from baseline_lib import is_hangul_name
from dac_v2_phonetic import skeleton, romanize
from phonetic_metadata_lib import CategoryPhoneticIndex


FEATURE_NAMES = ("phonetic_candidate_hit", "phonetic_candidate_rank", "phonetic_ratio_to_top")


class RankPositionIndex(CategoryPhoneticIndex):
    @lru_cache(maxsize=4096)
    def query_key(self, query):
        return skeleton(romanize(query))

    @lru_cache(maxsize=4096)
    def rank_map(self, query):
        return {candidate: rank for rank, candidate in enumerate(self.phonetic_candidates(query))}

    @lru_cache(maxsize=4096)
    def top_ratio(self, query):
        candidates = self.phonetic_candidates(query)
        if not candidates:
            return 0.0
        return fuzz.ratio(self.query_key(query), self.phonetic_names[candidates[0]]) / 100

    def pair_features(self, row, candidate_index):
        base = super().pair_features(row, candidate_index)
        if not is_hangul_name(row.get("vessel_name")):
            return base + [0.0, 0.0, 0.0]
        query = row["vessel_name"]
        rank = self.rank_map(query).get(candidate_index)
        ratio = fuzz.ratio(self.query_key(query), self.phonetic_names[candidate_index]) / 100
        return base + [float(rank is not None), 0.0 if rank is None else 1 - rank / 64,
                       ratio - self.top_ratio(query)]
