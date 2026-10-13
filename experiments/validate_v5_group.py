"""Independent full-pipeline GroupShuffleSplit validation of v4 blend logic.

For each seed, all fitted artifacts (LambdaRank, RF, priors, and call-sign
mapping) come only from the fit partition. The holdout's target IDs are removed
before calling the same v5 script.py used for the candidate package.
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT / "v5-candidate"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PACKAGE / "model"))
from baseline_lib import FEATURE_NAMES as BASE_FEATURES, is_hangul_name, local_weighted_accuracy
from romanized_lib import FEATURE_NAMES as ROMAN_FEATURES, RomanIndex
from nonhangul_lib import FEATURE_NAMES as NONHANGUL_FEATURES
from nonhangul_global_lib import GlobalFuzzyIndex, make_training_pairs_global
from phonetic_metadata_lib import FEATURE_NAMES as HANGUL_FEATURES
from token_feature_lib import FEATURE_NAMES as LOCAL_FEATURES, RawTokenFeatureIndex, fit_token_map
from initialism_lib import FEATURE_NAMES as INITIAL_FEATURES, InitialismIndex
from dac_v3_tonnage_filtered_trial import rank_pairs_with_local
from dac_v3_hard_mining_trial import mine_hard_pairs, new_ranker

spec = importlib.util.spec_from_file_location("v5_candidate_script", PACKAGE / "script.py")
assert spec and spec.loader
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)


def fit_priors(fit, catalog):
    linked = fit.merge(catalog, left_on="target_vessel_id", right_on="vessel_id",
                       suffixes=("_q", "_c"))
    flagmap, kindmap = defaultdict(Counter), defaultdict(Counter)
    for row in linked.to_dict("records"):
        if pd.notna(row["flag_name_q"]):
            flagmap[str(row["flag_name_q"])][str(row["flag_name_c"])] += 1
        if pd.notna(row["vessel_type_q"]):
            kindmap[str(row["vessel_type_q"])][str(row["vessel_type_c"])] += 1
    return dict(flagmap), dict(kindmap)


def make_phonetic_pairs(fit, index):
    features, labels, groups = [], [], []
    retrieved = 0
    for row in fit.to_dict("records"):
        true = index.id_to_index[row["target_vessel_id"]]
        all_candidates = index.candidates(row)
        retrieved += true in all_candidates
        base = RomanIndex.candidates(index, row)
        if is_hangul_name(row["vessel_name"]):
            base_neg = [i for i in base if i != true][:10]
            extra_neg = [i for i in all_candidates if i not in base and i != true][:10]
            negatives = base_neg + extra_neg
            if len(negatives) < 20:
                used = set(negatives)
                negatives += [i for i in base if i != true and i not in used][:20-len(negatives)]
        else:
            negatives = [i for i in base if i != true][:20]
        group_candidates = [true] + negatives
        groups.append(len(group_candidates))
        for candidate in group_candidates:
            features.append(index.pair_features(row, candidate))
            labels.append(candidate == true)
    return (np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int8),
            groups, retrieved / len(fit))


def fit_rf(x, y):
    model = RandomForestClassifier(n_estimators=60, max_depth=12, min_samples_leaf=10,
                                   class_weight="balanced_subsample", n_jobs=4, random_state=42)
    model.fit(x, y)
    return model


def main():
    train = pd.read_csv(ROOT / "open" / "train.csv", dtype={"call_sign": str, "target_vessel_id": str})
    catalog = pd.read_csv(ROOT / "open" / "vessel_catalog.csv", dtype={"vessel_id": str})
    results = []
    for seed in (42, 43, 44, 20261013):
        fi, vi = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed)
                      .split(train, groups=train["call_sign"]))
        fit, valid = train.iloc[fi].copy(), train.iloc[vi].copy()
        assert set(fit["call_sign"]).isdisjoint(valid["call_sign"])
        assert not fit["target_vessel_id"].isna().any()

        flagmap, kindmap = fit_priors(fit, catalog)
        token_map = fit_token_map(fit, catalog)
        han_index = RawTokenFeatureIndex(catalog, flagmap, kindmap, token_map, phonetic_limit=64)
        x_han, y_han, groups_han = rank_pairs_with_local(fit, han_index)
        assert x_han.shape[1] == len(HANGUL_FEATURES + LOCAL_FEATURES)
        ranker_base = new_ranker()
        ranker_base.fit(x_han, y_han, group=groups_han)
        x_hard, y_hard, groups_hard = mine_hard_pairs(
            fit, han_index, ranker_base, x_han, y_han, groups_han
        )
        ranker_second = new_ranker()
        ranker_second.fit(x_hard, y_hard, group=groups_hard)

        initial_index = InitialismIndex(catalog, flagmap, kindmap, token_map, phonetic_limit=64)
        x_initial, y_initial, groups_initial = rank_pairs_with_local(fit, initial_index)
        assert x_initial.shape[1] == len(HANGUL_FEATURES + INITIAL_FEATURES)
        initial_base = new_ranker()
        initial_base.fit(x_initial, y_initial, group=groups_initial)
        x_initial_hard, y_initial_hard, groups_initial_hard = mine_hard_pairs(
            fit, initial_index, initial_base, x_initial, y_initial, groups_initial
        )
        initial_second = new_ranker()
        initial_second.fit(x_initial_hard, y_initial_hard, group=groups_initial_hard)

        fit_non = fit[~fit["vessel_name"].map(is_hangul_name)]
        non_index = GlobalFuzzyIndex(catalog)
        x_non, y_non = make_training_pairs_global(fit_non, non_index)
        assert x_non.shape[1] == len(BASE_FEATURES + ROMAN_FEATURES + NONHANGUL_FEATURES)
        rf_non = fit_rf(x_non, y_non)

        mapping = fit[["call_sign", "target_vessel_id"]].drop_duplicates()
        assert not mapping["call_sign"].duplicated().any()
        with tempfile.TemporaryDirectory(prefix=f"dacon_v5_group_{seed}_") as d:
            tmp = Path(d)
            model_dir = tmp / "model"
            model_dir.mkdir()
            joblib.dump({"base_model": ranker_base, "second_model": ranker_second,
                         "second_weight": 0.75,
                         "feature_names": HANGUL_FEATURES + LOCAL_FEATURES,
                         "flagmap": flagmap, "kindmap": kindmap, "token_map": token_map,
                         "phonetic_limit": 64},
                        model_dir / "ranker_hangul_blend.pkl")
            joblib.dump({"base_model": initial_base, "second_model": initial_second,
                         "second_weight": 0.75,
                         "feature_names": HANGUL_FEATURES + INITIAL_FEATURES,
                         "flagmap": flagmap, "kindmap": kindmap, "token_map": token_map,
                         "phonetic_limit": 64},
                        model_dir / "ranker_initialism_blend.pkl")
            joblib.dump({"model": rf_non,
                         "feature_names": BASE_FEATURES + ROMAN_FEATURES + NONHANGUL_FEATURES},
                        model_dir / "nonhangul_rf.pkl")
            mapping.to_csv(model_dir / "call_sign_map.csv", index=False, encoding="utf-8")
            query = valid.drop(columns=["target_vessel_id"])
            query.to_csv(tmp / "holdout.csv", index=False, encoding="utf-8")
            out_path = v3.run(tmp / "holdout.csv", ROOT / "open" / "vessel_catalog.csv",
                              model_dir, tmp / "pred.csv")
            predicted = pd.read_csv(out_path, dtype=str)
            assert predicted["sample_id"].tolist() == valid["sample_id"].tolist()
            score = local_weighted_accuracy(valid, predicted["target_vessel_id"].tolist())
            if seed == 42:
                sub = query.head(20)
                sub.iloc[::-1].to_csv(tmp / "reverse.csv", index=False, encoding="utf-8")
                reverse_path = v3.run(tmp / "reverse.csv", ROOT / "open" / "vessel_catalog.csv",
                                      model_dir, tmp / "reverse_pred.csv")
                reverse = pd.read_csv(reverse_path, dtype=str)
                original = predicted.set_index("sample_id")["target_vessel_id"]
                assert reverse.set_index("sample_id")["target_vessel_id"].equals(
                    original.reindex(reverse["sample_id"]))

        recall_han = sum(han_index.id_to_index[row["target_vessel_id"]] in han_index.candidates(row)
                         for row in valid.to_dict("records") if is_hangul_name(row["vessel_name"]))
        result = {"seed": seed, "fit_rows": len(fit), "holdout_rows": len(valid),
                  "holdout_hangul_rows": int(valid["vessel_name"].map(is_hangul_name).sum()),
                  "holdout_initialism_rows": sum(initial_index.initialism_name(name) is not None
                                                 for name in valid["vessel_name"]),
                  "hangul_candidate_recall": recall_han, **score}
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    count = len(results)
    print(json.dumps({"mean_weighted_accuracy": sum(r["weighted_accuracy"] for r in results)/count,
                      "mean_hangul_accuracy": sum(r["hangul_accuracy"] for r in results)/count,
                      "mean_nonhangul_accuracy": sum(r["non_hangul_accuracy"] for r in results)/count},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
