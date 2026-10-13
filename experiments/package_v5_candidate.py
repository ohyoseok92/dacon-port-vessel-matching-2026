"""Create a deterministic local DACON ZIP; this does not submit it."""

import hashlib
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT / "v5-candidate"
OUTPUT = ROOT / "v5-candidate-not-sent.zip"
FILES = (
    "script.py",
    "requirements.txt",
    "model/baseline_lib.py",
    "model/romanized_lib.py",
    "model/dac_v2_phonetic.py",
    "model/phonetic_metadata_lib.py",
    "model/rank_position_lib.py",
    "model/tonnage_filtered_lib.py",
    "model/raw_ratio_lib.py",
    "model/token_feature_lib.py",
    "model/initialism_lib.py",
    "model/ranker_blend_adapter.py",
    "model/nonhangul_lib.py",
    "model/nonhangul_global_lib.py",
    "model/ranker_hangul_blend.pkl",
    "model/ranker_initialism_blend.pkl",
    "model/nonhangul_rf.pkl",
    "model/call_sign_map.csv",
)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    for name in FILES:
        if not (PACKAGE / name).is_file():
            raise FileNotFoundError(PACKAGE / name)
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name in FILES:
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 13, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, (PACKAGE / name).read_bytes(), compress_type=zipfile.ZIP_DEFLATED,
                       compresslevel=6)
    with zipfile.ZipFile(OUTPUT) as z:
        assert z.testzip() is None
        assert tuple(z.namelist()) == FILES
        for name in FILES:
            assert hashlib.sha256(z.read(name)).hexdigest() == sha256(PACKAGE / name)
    source_h = ROOT / "v4-candidate" / "model" / "ranker_hangul_blend.pkl"
    source_i = ROOT / "v5-prep" / "model" / "ranker_initialism_blend.pkl"
    source_n = ROOT / "nonhangul-v3" / "model" / "nonhangul_rf.pkl"
    assert sha256(source_h) == sha256(PACKAGE / "model/ranker_hangul_blend.pkl")
    assert sha256(source_i) == sha256(PACKAGE / "model/ranker_initialism_blend.pkl")
    assert sha256(source_n) == sha256(PACKAGE / "model/nonhangul_rf.pkl")
    print(f"ZIP PASS files={len(FILES)} bytes={OUTPUT.stat().st_size} sha256={sha256(OUTPUT)}")


if __name__ == "__main__":
    main()
