"""vccor/final plan B (RE-SPECIFIED BASELINES, fix item 5): design shared by the plan writer, prep / train / predict jobs and
the looks. Amends three blocks that are patched into the scheduler and paused (peft, scratch_nm, amos_draws); every amendment
ADDS jobs (insert-only scheduler patch) and changes no registered job, rule or look of those blocks.

  peft A1      dev18 predictions of HEAD / F1B_LR3E5 / STOCK (the registered jobs predict held18 + test10 only); the dev18
               selection among the six recipes (bsl_select.py); the challenger C (= the dev18-best recipe other than F1b; equal
               to the winner W whenever W != F1b) trained on the AMOS K = 4 and MACT K = 4 labelled scans and scored on AMOS-240
               (declared repeat look 2) and MACT-81 (declared repeat look 3) next to F1b's stored FULL_K4 maps; the FLARE look of
               every recipe re-scored under conventions v1
  scratch_nm A1 SCRATCH_NM100 / SCRATCH_NM1000 unchanged (kept); its FLARE look re-scored under conventions v1 (bsl_score.py
               part 'flare'); scratch is a reference baseline, not a recipe candidate
  amos_draws A1 draw 3407 (not trained yet) is REPLACED by the cross-site draw XS: K = 4 AMOS-240 scans of the site / vendor never
               used for training ('people', SIEMENS SOMATOM Force), chosen from metadata only (xs_draw()); the XS scans are
               excluded from every evaluation set of this plan; XS is scored on AMOS-240 (look 2; n = 236) and AMOS-56
No import-time side effects beyond sys.path.
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import final_common as F  # noqa: E402

PLAN = "baselines"
DIR = F.PLAN_DIR[PLAN]
VC = F.VC
NN_ROOT = DIR / "nnunet"
PREP = DIR / "prep"
MODELS = DIR / "models"
SEL = DIR / "selection"
LOOK = DIR / "look"

# ---------------------------------------------------------------------------------------------------- recipes (== peft)
CANDIDATES = ["F1b", "LORA_A", "LORA_B", "HEAD", "F1B_LR3E5", "STOCK"]       # this order breaks exact ties
RECIPES = {
    "F1b": dict(kind="ts", trainable="all", lr=1e-4, label="all weights, lr 1e-4 (tsft_full F1b; the registered recipe)"),
    "LORA_A": dict(kind="ts", trainable="lora", lr=1e-2, r=8, alpha=16, label="LoRA r 8 / alpha 16 on every Conv3d, lr 1e-2"),
    "LORA_B": dict(kind="ts", trainable="lora", lr=1e-3, r=8, alpha=16, label="LoRA r 8 / alpha 16 on every Conv3d, lr 1e-3"),
    "HEAD": dict(kind="ts", trainable="heads", lr=1e-2, label="segmentation heads only (decoder.seg_layers), lr 1e-2"),
    "F1B_LR3E5": dict(kind="ts", trainable="all", lr=3e-5, label="all weights, lr 3e-5"),
    "STOCK": dict(kind="stock", trainable="all", lr=1e-2, epochs=250, trainer="nnUNetTrainer_250epochs_NoMirroring",
                  label="nnU-Net's pretrained fine-tuning recipe: one network on the dataset's own classes, TS-291 init "
                        "without the segmentation layers, lr 1e-2, 250 epochs, no mirroring"),
}
DEV18_MAPS = {   # candidate -> (dev18 raw-map dir, lut kind, DONE file)
    "F1b": (VC / "tsft_full/pred/F1b/raw", "ts", VC / "tsft_full/pred/F1b/DONE.json"),
    "LORA_A": (VC / "peft/pred/LORA_A/raw", "ts", VC / "peft/pred/LORA_A/DONE.json"),
    "LORA_B": (VC / "peft/pred/LORA_B/raw", "ts", VC / "peft/pred/LORA_B/DONE.json"),
    "HEAD": (DIR / "pred/peft_HEAD__flare_dev18/raw", "ts", DIR / "pred/peft_HEAD__flare_dev18/DONE.json"),
    "F1B_LR3E5": (DIR / "pred/peft_F1B_LR3E5__flare_dev18/raw", "ts", DIR / "pred/peft_F1B_LR3E5__flare_dev18/DONE.json"),
    "STOCK": (DIR / "pred/peft_STOCK__flare_dev18/raw", "id", DIR / "pred/peft_STOCK__flare_dev18/DONE.json"),
}
SELECTION_RULE = ("W = argmax over the six candidates of the mean over dev18 (18 scans) of the per-scan PENALISED macro Dice "
                  "(13 FLARE organs; native raw grid, full field; conventions v1); C = argmax over the five candidates other than "
                  "F1b (C = W whenever W != F1b); exact ties (|difference| < 1e-9) go to the earlier candidate in "
                  "F1b, LORA_A, LORA_B, HEAD, F1B_LR3E5, STOCK")

# ---------------------------------------------------------------------------------------------------- labelled sets
AMOS_K4 = ["0017", "0001", "0008", "0048"]        # xdata labelled('amos', 1337)
MACT_K4 = ["0075", "0049", "0037", "0041"]        # xdata labelled('mact', 1337)
XS_SEED = 3407                                     # the seed of the draw XS replaces
XS_SITE = "people"


def xs_eligible():
    """AMOS-240 scans of site 'people' (all SIEMENS SOMATOM Force) whose patient cluster (metadata key over all 300 labelled
    AMOS CT) holds no other scan. Metadata + file listing only: no label is read."""
    import a240_common as A240
    ids, m = A240.ids240(), A240.meta()
    cl = F.amos_patient_clusters()
    size = {}
    for c, k in cl.items():
        size[k] = size.get(k, 0) + 1
    return sorted(c for c in ids if m[c]["Site"] == XS_SITE and size[cl[c]] == 1)


def xs_draw():
    """THE cross-site draw: numpy RandomState(3407).permutation(xs_eligible())[:4] (sorted ids in, order kept)."""
    import numpy as np
    el = xs_eligible()
    perm = np.random.RandomState(XS_SEED).permutation(len(el))
    return [el[i] for i in perm[:4]]


def amos240_eval(xs=None):
    """The AMOS-240 set of this plan's look: AMOS-240 minus the XS scans (patient singletons by construction)."""
    import a240_common as A240
    xs = set(xs if xs is not None else xs_draw())
    return [c for c in A240.ids240() if c not in xs]


# ---------------------------------------------------------------------------------------------------- training runs
DATASETS = {   # nnU-Net datasets of this plan (own root NN_ROOT); built by bsl_prep.py
    811: dict(name="Dataset811_FINAMOS291_K4", ds="amos", part=291, labelled=AMOS_K4, kind="ts", source="xdata s1337_K4_FULL"),
    812: dict(name="Dataset812_FINAMOS293_K4", ds="amos", part=293, labelled=AMOS_K4, kind="ts", source="xdata s1337_K4_FULL"),
    813: dict(name="Dataset813_FINMACT291_K4", ds="mact", part=291, labelled=MACT_K4, kind="ts", source="xdata s1337_K4_FULL"),
    814: dict(name="Dataset814_FINAMOSSTOCK_K4", ds="amos", part=None, labelled=AMOS_K4, kind="stock", source="raw AMOS GT"),
    815: dict(name="Dataset815_FINMACTSTOCK_K4", ds="mact", part=None, labelled=MACT_K4, kind="stock",
              source="Zenodo raw MACT labels, ignore label outside Gibson's box"),
    816: dict(name="Dataset816_FINAMOS291_XS", ds="amos", part=291, labelled="XS", kind="ts", source="new composite labels"),
    817: dict(name="Dataset817_FINAMOS293_XS", ds="amos", part=293, labelled="XS", kind="ts", source="new composite labels"),
}
RUNS = {   # run -> spec; 'recipe' None = the selected challenger C (selection/selection.json)
    "C_amos": dict(model="bslC_amos", recipe=None, ts_datasets={291: 811, 293: 812}, stock_dataset=814, ds="amos",
                   n_classes=13, labelled=AMOS_K4),
    "C_mact": dict(model="bslC_mact", recipe=None, ts_datasets={291: 813}, stock_dataset=815, ds="mact", n_classes=8,
                   labelled=MACT_K4),
    "XS_amos": dict(model="bslXS_amos", recipe="F1b", ts_datasets={291: 816, 293: 817}, stock_dataset=None, ds="amos",
                    n_classes=13, labelled="XS"),
}
TRAIN_H = {"F1b": {"amos": 2.1, "mact": 1.25}, "LORA_A": {"amos": 2.4, "mact": 1.3}, "LORA_B": {"amos": 2.4, "mact": 1.3},
           "HEAD": {"amos": 1.4, "mact": 0.8}, "F1B_LR3E5": {"amos": 2.2, "mact": 1.25}, "STOCK": {"amos": 5.5, "mact": 5.5}}
PRED_JOBS = {   # job -> GPU-h (measured: AMOS-240 F0 5.7 h, FULL_K4 with cache 3.1 h; MACT-81 0.8-0.9 h; AMOS-56 0.55 h)
    "peft_HEAD__flare_dev18": 0.2, "peft_F1B_LR3E5__flare_dev18": 0.2, "peft_STOCK__flare_dev18": 0.2,
    "F0__amos240": 5.8, "bslC_amos__amos240": 3.3, "bslC_amos__amos56": 0.7, "bslC_mact__mact81": 1.0,
    "bslXS_amos__amos240": 3.3, "bslXS_amos__amos56": 0.7,
    "flare_F1b__amos240": 3.3,     # the FLARE-tuned model (another protocol) on AMOS-240: the GENERIC part of the decomposition
}


def recipe_of(run, selection=None):
    r = RUNS[run]["recipe"]
    if r is not None:
        return r
    import json
    sel = selection or json.load(open(SEL / "selection.json"))
    return sel["challenger"]


def labelled_of(run):
    lab = RUNS[run]["labelled"]
    return xs_draw() if lab == "XS" else list(lab)


def dataset_name(did):
    return DATASETS[did]["name"]


def verify_datasets(run, recipe, prep=None):
    """The training datasets of a run hash to their prep DONE record (bsl_prep.py); refuses otherwise."""
    import json
    which = "xs" if run == "XS_amos" else "challenger"
    p = Path(prep or PREP) / f"DONE_{which}.json"
    if not p.is_file():
        raise SystemExit(f"[REFUSE] {F.rel(p)} missing: run bsl_prep.py --which {which} first")
    d = json.load(open(p))
    if which == "challenger" and d["recipe"] != recipe:
        raise SystemExit(f"[REFUSE] the challenger datasets were built for {d['recipe']}, not {recipe}")
    bad = [f for f, h in d["files_sha256"].items() if F.sha256(f) != h]
    if bad:
        raise SystemExit(f"[REFUSE] training dataset files changed since the prep: {bad[:3]}")
    kind = RECIPES[recipe]["kind"]
    need = [dataset_name(x) for x in (RUNS[run]["ts_datasets"].values() if kind == "ts" else [RUNS[run]["stock_dataset"]])]
    missing = sorted(set(need) - set(d["datasets"]))
    if missing:
        raise SystemExit(f"[REFUSE] datasets missing from the prep record: {missing}")
    return True
