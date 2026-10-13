# DACON Port Vessel Matching (2026)

Source and local validation notes for the [DACON port AI competition](https://dacon.io/competitions/official/236771/overview/rules). The competition work stopped on 2026-10-14 under a new selection rule: future contests require at least ten confirmed prize-winning teams.

## Submitted versions

| Submission | Date (KST) | Official score at last check |
| --- | --- | ---: |
| #107373, roman | 2026-10-13 | 0.7497777778 |
| #107420, v2 | 2026-10-13 | 0.9025555556 |
| #107436, v3.3 | 2026-10-13 | 0.945 |
| #107483, v5 | 2026-10-14 | Pending; not verified |

Local grouped holdout accuracy for v5 averaged 0.986789 across four call-sign splits. This is **not** the official test score, nor evidence of a prize or payment. See `reports/2026-10-14-dacon-v5-combined-validation.md`.

## Contents

- `src/`: v5 inference source and dependency list, excluding trained models and data-derived maps.
- `experiments/`: selected training, packaging, and grouped-validation scripts.
- `reports/`: submission and validation records.

The official `train.csv`, `vessel_catalog.csv`, test data, trained model files, data-derived `call_sign_map.csv`, and submitted ZIPs are deliberately absent. The [competition rules](https://dacon.io/competitions/official/236771/overview/rules) prohibit unauthorized redistribution of competition data. This repository alone cannot reproduce the submitted predictions; the private archive in the user's MYBOX contains the exact submitted ZIPs.

Generated code was developed with AI assistance. Any execution must follow the official competition rules and use data obtained from DACON through authorized access.
