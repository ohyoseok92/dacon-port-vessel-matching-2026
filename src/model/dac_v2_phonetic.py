"""Independent phonetic-skeleton retrieval from official vessel catalog names.

No external ship database or validation labels are indexed. Every query is
searched independently; the optional cache stores only query-to-catalog hits.
"""

from functools import lru_cache

from rapidfuzz import fuzz, process

from baseline_lib import grams, is_hangul_name, jaccard
from romanized_lib import RomanIndex, romanize


def skeleton(value: str) -> str:
    text = value.upper().replace("CH", "J").replace("SH", "S").replace("PH", "F")
    text = text.replace("CK", "K").replace("NG", "N")
    translation = str.maketrans({
        "B": "P", "V": "P", "F": "P", "D": "T", "G": "K", "C": "K", "Q": "K",
        "R": "L", "Z": "S", "X": "S", "H": "", "W": "", "Y": "",
        "A": "", "E": "", "I": "", "O": "", "U": "",
    })
    text = text.translate(translation)
    return "".join(char for i, char in enumerate(text) if i == 0 or char != text[i - 1])


class PhoneticIndex(RomanIndex):
    def __init__(self, catalog, phonetic_limit: int = 64):
        super().__init__(catalog)
        self.phonetic_limit = phonetic_limit
        self.phonetic_names = [skeleton(name) for name in self.names]

    @lru_cache(maxsize=4096)
    def phonetic_candidates(self, query: str) -> tuple[int, ...]:
        key = skeleton(romanize(query))
        if not key:
            return ()
        # RapidFuzz searches the immutable public catalog; its output depends
        # solely on this one query and the catalog, not on other test rows.
        hits = process.extract(key, self.phonetic_names, scorer=fuzz.ratio, limit=self.phonetic_limit)
        return tuple(index for _, score, index in hits if score > 0)

    def candidates(self, row, name_limit=16, tonnage_limit=12, max_candidates=40):
        base = super().candidates(row, name_limit, tonnage_limit, max_candidates)
        if not is_hangul_name(row.get("vessel_name")):
            return base
        used = set(base)
        for candidate in self.phonetic_candidates(row["vessel_name"]):
            if candidate not in used:
                base.append(candidate)
                used.add(candidate)
        return base

    def pair_features(self, row, candidate_index):
        base = super().pair_features(row, candidate_index)
        if not is_hangul_name(row.get("vessel_name")):
            return base + [0.0, 0.0]
        query = skeleton(romanize(row["vessel_name"]))
        candidate = self.phonetic_names[candidate_index]
        return base + [fuzz.ratio(query, candidate) / 100.0,
                       jaccard(grams(query, 2), grams(candidate, 2))]
