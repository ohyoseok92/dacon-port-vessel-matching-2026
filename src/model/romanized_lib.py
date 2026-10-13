"""Cross-script candidate retrieval over the contest-provided vessel catalog.

Korean vessel names are romanized approximately for candidate lookup. The
catalog and fitted model remain the only sources of vessel identity.
"""
from collections import Counter, defaultdict

from baseline_lib import CatalogIndex, grams, is_hangul_name, jaccard

INITIAL = ("G", "KK", "N", "D", "TT", "R", "M", "B", "PP", "S", "SS", "", "J", "JJ", "CH", "K", "T", "P", "H")
MEDIAL = ("A", "AE", "YA", "YAE", "EO", "E", "YEO", "YE", "O", "WA", "WAE", "OE", "YO", "U", "WEO", "WE", "WI", "YU", "EU", "UI", "I")
FINAL = ("", "K", "K", "K", "N", "N", "N", "T", "L", "K", "M", "L", "L", "L", "P", "L", "M", "P", "P", "T", "T", "NG", "T", "T", "K", "T", "P", "T")
FEATURE_NAMES = ("roman_bigram_jaccard", "roman_trigram_jaccard", "roman_prefix_ratio")


def romanize(value: object) -> str:
    result = []
    for char in str(value).upper():
        point = ord(char) - 0xAC00
        if 0 <= point < 11172:
            onset, rest = divmod(point, 588)
            nucleus, coda = divmod(rest, 28)
            result.append(INITIAL[onset] + MEDIAL[nucleus] + FINAL[coda])
        elif char.isalnum():
            result.append(char)
    return "".join(result)


class RomanIndex(CatalogIndex):
    def __init__(self, catalog):
        super().__init__(catalog)
        self.roman_postings = defaultdict(list)
        for i, name in enumerate(self.names):
            for gram in grams(name, 3):
                self.roman_postings[gram].append(i)

    def candidates(self, row, name_limit=16, tonnage_limit=12, max_candidates=40):
        base = super().candidates(row, name_limit, tonnage_limit, max_candidates)
        if not is_hangul_name(row.get("vessel_name")):
            return base
        counts = Counter()
        for gram in sorted(grams(romanize(row["vessel_name"]), 3)):
            counts.update(self.roman_postings.get(gram, ()))
        used = set(base)
        for i in sorted(counts, key=lambda i: (-counts[i], i))[:64]:
            if i not in used:
                base.append(i)
                used.add(i)
        return base

    def pair_features(self, row, candidate_index):
        base = super().pair_features(row, candidate_index)
        if not is_hangul_name(row.get("vessel_name")):
            return base + [0.0, 0.0, 0.0]
        query = romanize(row["vessel_name"])
        name = self.names[candidate_index]
        prefix = 0
        for left, right in zip(query, name):
            if left != right:
                break
            prefix += 1
        return base + [
            jaccard(grams(query, 2), self.bigram_sets[candidate_index]),
            jaccard(grams(query, 3), self.trigram_sets[candidate_index]),
            prefix / max(len(query), len(name), 1),
        ]
