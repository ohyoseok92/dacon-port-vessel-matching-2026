"""Official-catalog phonetic retrieval plus train-fitted category priors."""

from collections import Counter, defaultdict

from dac_v2_phonetic import PhoneticIndex


FEATURE_NAMES = (
    "name_exact", "name_bigram_jaccard", "name_trigram_jaccard",
    "name_prefix_ratio", "name_length_ratio", "gross_tonnage_gap",
    "international_gross_tonnage_gap", "best_tonnage_gap",
    "query_tonnage_missing", "catalog_tonnage_missing",
    "roman_bigram_jaccard", "roman_trigram_jaccard", "roman_prefix_ratio",
    "phonetic_ratio", "phonetic_bigram_jaccard",
    "flag_prior", "kind_prior", "flag_prior_over_point_three", "kind_prior_over_point_three",
)


class CategoryPhoneticIndex(PhoneticIndex):
    def __init__(self, catalog, flagmap, kindmap, phonetic_limit=64):
        super().__init__(catalog, phonetic_limit=phonetic_limit)
        self.flags = self.catalog.flag_name.astype(str).tolist()
        self.kinds = self.catalog.vessel_type.astype(str).tolist()
        self.flagmap = defaultdict(Counter, {k: Counter(v) for k, v in flagmap.items()})
        self.kindmap = defaultdict(Counter, {k: Counter(v) for k, v in kindmap.items()})

    @staticmethod
    def prior(table, query, item):
        counts = table.get(str(query), Counter())
        n = sum(counts.values())
        if n < 20:
            return 0.0
        return counts.get(str(item), 0) / (n + 10)

    def pair_features(self, row, candidate_index):
        base = super().pair_features(row, candidate_index)
        flag = self.prior(self.flagmap, row.get("flag_name"), self.flags[candidate_index])
        kind = self.prior(self.kindmap, row.get("vessel_type"), self.kinds[candidate_index])
        return base + [flag, kind, float(flag > .3), float(kind > .3)]
