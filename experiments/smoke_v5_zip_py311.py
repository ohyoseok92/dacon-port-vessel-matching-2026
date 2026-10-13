"""Extract the actual v4 ZIP and run private 30-row Python 3.11 inference."""

import importlib.util
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
ZIP = ROOT / "v5-candidate-not-sent.zip"

train = pd.read_csv(ROOT / "open" / "train.csv").drop(columns=["target_vessel_id"])
is_hangul = train.vessel_name.astype(str).str.contains("[가-힣]", regex=True)
query = pd.concat([train[is_hangul].head(15), train[~is_hangul].head(15)], ignore_index=True)
query["sample_id"] = [f"SMOKE_{i:02d}" for i in range(30)]
query["call_sign"] = [f"NEVER_SEEN_ZIP_PY311_{i:02d}" for i in range(30)]
with tempfile.TemporaryDirectory(prefix="dacon_v33_py311_zip_") as name:
    temp = Path(name)
    with zipfile.ZipFile(ZIP) as z:
        assert z.testzip() is None
        z.extractall(temp / "pkg")
    package = temp / "pkg"
    spec = importlib.util.spec_from_file_location("v33_zip_script", package / "script.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    query.to_csv(temp / "input.csv", index=False)
    output = module.run(temp / "input.csv", ROOT / "open" / "vessel_catalog.csv",
                        package / "model", temp / "output.csv")
    predictions = pd.read_csv(output, dtype=str)
    catalog_ids = set(pd.read_csv(ROOT / "open" / "vessel_catalog.csv",
                                  usecols=["vessel_id"], dtype=str).vessel_id)
    assert len(predictions) == 30
    assert predictions.columns.tolist() == ["sample_id", "target_vessel_id"]
    assert predictions.sample_id.tolist() == query.sample_id.tolist()
    assert predictions.target_vessel_id.isin(catalog_ids).all()
    print("PASS actual_zip=True rows=30 schema=True catalog=True")
