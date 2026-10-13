"""Fit-only Hangul token matching feature from the official training catalog."""

from collections import defaultdict
from functools import lru_cache

from rapidfuzz import fuzz

from baseline_lib import is_hangul_name, normalize_name
from dac_v2_phonetic import romanize
from raw_ratio_lib import RawRatioIndex, FEATURE_NAMES as RAW_FEATURE_NAMES


FEATURE_NAMES = RAW_FEATURE_NAMES + ("fit_token_name_ratio",)


def longest_common_substring(values):
    if not values:
        return ""
    shortest = min(values, key=len)
    for width in range(len(shortest), 2, -1):
        for start in range(len(shortest) - width + 1):
            part = shortest[start:start + width]
            if all(part in value for value in values):
                return part
    return ""


def fit_token_map(fit, catalog):
    cat_name = dict(zip(catalog.vessel_id, catalog.vessel_name.map(normalize_name)))
    names_by_token = defaultdict(dict)
    for row in fit.to_dict("records"):
        if not is_hangul_name(row["vessel_name"]):
            continue
        for token in str(row["vessel_name"]).split():
            if len(token) < 2 or not any("\uac00" <= char <= "\ud7a3" for char in token):
                continue
            names_by_token[token][row["target_vessel_id"]] = cat_name[row["target_vessel_id"]]
    return {token: part for token, targets in names_by_token.items() if len(targets) >= 2
            for part in [longest_common_substring(list(targets.values()))] if len(part) >= 3}


class RawTokenFeatureIndex(RawRatioIndex):
    def __init__(self, catalog, flagmap, kindmap, token_map, phonetic_limit=64):
        super().__init__(catalog, flagmap, kindmap, phonetic_limit=phonetic_limit)
        self.token_map = token_map

    @lru_cache(maxsize=4096)
    def translated_name(self, name):
        return "".join(self.token_map.get(token, romanize(token)) for token in str(name).split())

    def pair_features(self, row, candidate_index):
        base = super().pair_features(row, candidate_index)
        if not is_hangul_name(row.get("vessel_name")):
            return base + [0.0]
        query = self.translated_name(row["vessel_name"])
        return base + [fuzz.ratio(query, self.names[candidate_index]) / 100.0]
