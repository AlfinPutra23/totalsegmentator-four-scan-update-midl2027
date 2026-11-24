"""vccor/final LCTSC fine-tuning runs and datasets (the plan module lctsc_train.py uses, == word_common's RUNS / DATASETS
interface). F1b only (bsl_common.RECIPES['F1b']: all weights, lr 1e-4, 50 x 250 it, seed 1337, TS init strict); TS parts
291 (lobes + esophagus) and 293 (heart); one run per draw. STOCK is not part of LCTSC (its 250-epoch nnU-Net run took
about 17 GPU-h per draw on WORD; it does not fit before the cutoff).
"""
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import lctsc_common as C  # noqa: E402

PLAN = "lctsc"
NN_ROOT = C.OUT / "nnunet"
MODELS = C.OUT / "models"
PREP = C.OUT / "prep"
DATASETS = {}                          # site-matched draws S1-S3 (Dataset861-873 of the superseded pooled draws stay unused)
for i, d in enumerate(C.DRAW_SEEDS, start=1):
    DATASETS[880 + i] = dict(name=f"Dataset{880 + i}_LCTSC291_{d}", part=291, draw=d)
    DATASETS[890 + i] = dict(name=f"Dataset{890 + i}_LCTSC293_{d}", part=293, draw=d)
RUNS = {f"F1b_{d}": dict(model=f"lctsc_F1b_{d}", recipe="F1b", ts_datasets={291: 880 + i, 293: 890 + i}, stock_dataset=None,
                         ds="lctsc", n_classes=None, labelled=d) for i, d in enumerate(C.DRAW_SEEDS, start=1)}


def recipe_of(run, selection=None):
    return RUNS[run]["recipe"]


def labelled_of(run):
    return C.draw(RUNS[run]["labelled"])


def dataset_name(did):
    return DATASETS[did]["name"]


def verify_datasets(run, recipe, prep=None):
    import lctsc_log as L
    p = Path(prep or PREP) / "DONE_datasets.json"
    if not p.is_file():
        raise SystemExit(f"[REFUSE] {p} missing: run lctsc_prep.py first")
    d = json.load(open(p))
    bad = [f for f, h in d["files_sha256"].items() if L.sha256(f) != h]
    if bad:
        raise SystemExit(f"[REFUSE] LCTSC training dataset files changed since the prep: {bad[:3]}")
    for tid, did in RUNS[run]["ts_datasets"].items():
        if dataset_name(did) not in d["datasets"]:
            raise SystemExit(f"[REFUSE] {dataset_name(did)} missing from the prep record")
    return True
