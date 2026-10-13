"""Generic Korean spoken-letter normalization for row-local vessel matching."""

from functools import lru_cache

from rapidfuzz import fuzz

from baseline_lib import is_hangul_name
from romanized_lib import romanize
from token_feature_lib import RawTokenFeatureIndex, FEATURE_NAMES as TOKEN_FEATURE_NAMES


FEATURE_NAMES = TOKEN_FEATURE_NAMES + ("initialism_detected", "initialism_name_ratio")
SPOKEN_LETTERS = {
    "에이치": "H", "더블유": "W", "에프": "F", "에이": "A", "아이": "I",
    "제이": "J", "케이": "K", "에스": "S", "브이": "V", "엑스": "X",
    "와이": "Y", "제트": "Z", "앤": "N", "비": "B", "씨": "C", "시": "C",
    "디": "D", "이": "E", "지": "G", "엘": "L", "엠": "M", "엔": "N",
    "오": "O", "피": "P", "큐": "Q", "알": "R", "티": "T", "유": "U",
}
ORDERED = sorted(SPOKEN_LETTERS, key=len, reverse=True)


@lru_cache(maxsize=4096)
def spoken_initialism(token):
    letters = []
    index = 0
    while index < len(token):
        choice = next((part for part in ORDERED if token.startswith(part, index)), None)
        if choice is None:
            return None
        letters.append(SPOKEN_LETTERS[choice])
        index += len(choice)
    return "".join(letters) if len(letters) >= 2 else None


class InitialismIndex(RawTokenFeatureIndex):
    @lru_cache(maxsize=4096)
    def initialism_name(self, name):
        tokens = str(name).split()
        initials = [spoken_initialism(token) for token in tokens]
        if not any(initials):
            return None
        return "".join(initial or self.token_map.get(token, romanize(token))
                       for token, initial in zip(tokens, initials))

    def pair_features(self, row, candidate_index):
        base = super().pair_features(row, candidate_index)
        if not is_hangul_name(row.get("vessel_name")):
            return base + [0.0, 0.0]
        query = self.initialism_name(row["vessel_name"])
        if query is None:
            return base + [0.0, 0.0]
        return base + [1.0, fuzz.ratio(query, self.names[candidate_index]) / 100.0]
