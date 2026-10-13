"""Row-independent conditional initialism/v4 Hangul inference with global fuzzy RF."""

from pathlib import Path
import sys

import joblib
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "model"))
from baseline_lib import FEATURE_NAMES as BASE_FEATURES, is_hangul_name, predict_ids
from romanized_lib import FEATURE_NAMES as ROMAN_FEATURES
from nonhangul_lib import FEATURE_NAMES as NONHANGUL_FEATURES
from nonhangul_global_lib import GlobalFuzzyIndex
from phonetic_metadata_lib import FEATURE_NAMES as HANGUL_FEATURES
from token_feature_lib import RawTokenFeatureIndex, FEATURE_NAMES as LOCAL_FEATURES
from initialism_lib import InitialismIndex, FEATURE_NAMES as INITIAL_FEATURES
from ranker_blend_adapter import RankerBlendAdapter


def run(test_path="data/test.csv", catalog_path="data/vessel_catalog.csv",
        model_dir="model", output_path="output/submission.csv"):
    test = pd.read_csv(test_path, encoding="utf-8-sig", dtype={"call_sign": str}).reset_index(drop=True)
    catalog = pd.read_csv(catalog_path, encoding="utf-8-sig", dtype={"vessel_id": str})
    if test["sample_id"].isna().any() or not test["sample_id"].is_unique:
        raise ValueError("sample_id must be present and unique")
    model_dir = Path(model_dir)
    hangul_bundle = joblib.load(model_dir / "ranker_hangul_blend.pkl")
    initial_bundle = joblib.load(model_dir / "ranker_initialism_blend.pkl")
    nonhangul_bundle = joblib.load(model_dir / "nonhangul_rf.pkl")
    if tuple(hangul_bundle["feature_names"]) != HANGUL_FEATURES + LOCAL_FEATURES:
        raise ValueError("Hangul ranker and feature code versions differ")
    if tuple(initial_bundle["feature_names"]) != HANGUL_FEATURES + INITIAL_FEATURES:
        raise ValueError("Initialism ranker and feature code versions differ")
    if tuple(nonhangul_bundle["feature_names"]) != BASE_FEATURES + ROMAN_FEATURES + NONHANGUL_FEATURES:
        raise ValueError("Non-Hangul RF and feature code versions differ")

    sign_map = pd.read_csv(model_dir / "call_sign_map.csv", dtype=str)
    if sign_map["call_sign"].duplicated().any():
        raise ValueError("Ambiguous train call-sign mapping")
    mapped = test["call_sign"].astype(str).map(sign_map.set_index("call_sign")["target_vessel_id"])
    predictions = [None] * len(test)
    for pos, value in mapped.dropna().items():
        predictions[pos] = value

    unknown = mapped.index[mapped.isna()].tolist()
    hangul_positions = [pos for pos in unknown if is_hangul_name(test.at[pos, "vessel_name"])]
    hangul_set = set(hangul_positions)
    nonhangul_positions = [pos for pos in unknown if pos not in hangul_set]
    initial_positions = []
    regular_positions = []

    if hangul_positions:
        initial_index = InitialismIndex(catalog, initial_bundle["flagmap"],
                                       initial_bundle["kindmap"],
                                       initial_bundle["token_map"],
                                       phonetic_limit=initial_bundle["phonetic_limit"])
        initial_positions = [pos for pos in hangul_positions
                             if initial_index.initialism_name(test.at[pos, "vessel_name"]) is not None]
        initial_set = set(initial_positions)
        regular_positions = [pos for pos in hangul_positions if pos not in initial_set]

    if regular_positions:
        index = RawTokenFeatureIndex(catalog, hangul_bundle["flagmap"],
                                     hangul_bundle["kindmap"],
                                     hangul_bundle["token_map"],
                                     phonetic_limit=hangul_bundle["phonetic_limit"])
        ranker = RankerBlendAdapter(hangul_bundle["base_model"],
                                    hangul_bundle["second_model"],
                                    hangul_bundle["second_weight"])
        values = predict_ids(ranker, test.iloc[regular_positions], index)
        for pos, value in zip(regular_positions, values, strict=True):
            predictions[pos] = value

    if initial_positions:
        ranker = RankerBlendAdapter(initial_bundle["base_model"],
                                    initial_bundle["second_model"],
                                    initial_bundle["second_weight"])
        values = predict_ids(ranker, test.iloc[initial_positions], initial_index)
        for pos, value in zip(initial_positions, values, strict=True):
            predictions[pos] = value

    if nonhangul_positions:
        index = GlobalFuzzyIndex(catalog)
        values = predict_ids(nonhangul_bundle["model"], test.iloc[nonhangul_positions], index)
        for pos, value in zip(nonhangul_positions, values, strict=True):
            predictions[pos] = value

    valid_ids = set(catalog["vessel_id"].astype(str))
    if len(predictions) != len(test) or not set(predictions).issubset(valid_ids):
        raise ValueError("Prediction is missing or outside official catalog")
    result = pd.DataFrame({"sample_id": test["sample_id"], "target_vessel_id": predictions})
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Saved {len(result)} predictions")
    return output_path


if __name__ == "__main__":
    run()
