"""Fit two official-data Hangul rankers after four-split blend validation."""

from pathlib import Path
import sys

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT / "v4-candidate"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PACKAGE / "model"))
from dac_v3_lambdarank import fit_priors
from dac_v3_tonnage_filtered_trial import rank_pairs_with_local
from dac_v3_hard_mining_trial import mine_hard_pairs, new_ranker
from phonetic_metadata_lib import FEATURE_NAMES as BASE_FEATURE_NAMES
from token_feature_lib import RawTokenFeatureIndex, FEATURE_NAMES as LOCAL_FEATURE_NAMES, fit_token_map


def main():
    train = pd.read_csv(ROOT / "open" / "train.csv", dtype={"call_sign": str, "target_vessel_id": str})
    catalog = pd.read_csv(ROOT / "open" / "vessel_catalog.csv", dtype={"vessel_id": str})
    flagmap, kindmap = fit_priors(train, catalog)
    token_map = fit_token_map(train, catalog)
    index = RawTokenFeatureIndex(catalog, flagmap, kindmap, token_map, phonetic_limit=64)
    x, y, groups = rank_pairs_with_local(train, index)
    features = BASE_FEATURE_NAMES + LOCAL_FEATURE_NAMES
    if x.shape[1] != len(features) or sum(groups) != len(y):
        raise ValueError("Ranker training feature/group mismatch")
    base = new_ranker()
    base.fit(x, y, group=groups)
    x2, y2, groups2 = mine_hard_pairs(train, index, base, x, y, groups)
    second = new_ranker()
    second.fit(x2, y2, group=groups2)
    path = PACKAGE / "model" / "ranker_hangul_blend.pkl"
    joblib.dump({"base_model": base, "second_model": second, "second_weight": 0.75,
                 "feature_names": features, "flagmap": flagmap,
                 "kindmap": kindmap, "token_map": token_map,
                 "phonetic_limit": 64}, path, compress=3)
    print(f"Fitted rankers: train_rows={len(train)} groups={len(groups)} "
          f"base_pairs={len(y)} hard_pairs={len(y2)} "
          f"features={x.shape[1]} bytes={path.stat().st_size}")


if __name__ == "__main__":
    main()
