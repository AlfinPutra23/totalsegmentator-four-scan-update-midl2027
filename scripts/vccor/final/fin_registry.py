"""vccor/final REGISTRY: the models (zero-shot / adapted) and the target sets of every prediction job of plans A-C, and the
STORED prediction maps (cells) that already exist and are reused. Prediction job name = '<model>__<target>'; its output is
outputs/results/vccor/final/<plan dir>/pred/<model>__<target>/{raw/<cid>.nii.gz, rec/<cid>.json, DONE.json}
(raw = TotalSegmentator 'total' ids on the raw CT grid for 'ts' models, the dataset's own ids for 'stock' models).
No import-time side effects beyond sys.path (and no file is read at import).
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import final_common as F  # noqa: E402

VC = F.VC
TSFT_M, TSFT_P = VC / "tsft_full/models", VC / "tsft_full/pred"
XM, XP = VC / "xdata/models", VC / "xdata/pred"
PEFT_M = VC / "peft/models"
A_DIR, B_DIR, C_DIR = F.PLAN_DIR["transfer"], F.PLAN_DIR["baselines"], F.PLAN_DIR["word"]
TS_EPOCHS = 50

# ---------------------------------------------------------------------------------------------------- models
# kind 'ts'    : TotalSegmentator's own pipeline (frozen tsft_full/ts_pipeline.py) with the listed parts redirected to an
#                export (TS's plans.json + dataset.json, fold_0/checkpoint_final.pth, export.json); 'check' = export.json
#                fields that must match (plus checkpoint sha256 == export.json, num_epochs == 50, smoke False)
# kind 'stock' : one nnU-Net network on the dataset's own classes (nnUNetPredictor, tile 0.5, gaussian, mirroring OFF)
# kind 'export': resolved at run time from <dir>/DONE.json ('ts' parts or 'stock'), written by train_final.py
MODELS = {
    "F0": dict(kind="ts", parts={}, plan_owner=None,
               label="TotalSegmentator 2.13.0 'total', full resolution, --ml: its own weights for all five parts"),
    "flare_F1b": dict(kind="ts", site="flare", parts={291: TSFT_M / "F1b/part291", 293: TSFT_M / "F1b/part293"},
                      check={"variant": "F1b", "trainable": "all", "initial_lr": 1e-4},
                      label="FLARE-tuned F1b (tsft_full; K = 4, seed-1337 draw 0014 0001 0006 0038; all weights, lr 1e-4, 50 x 250 it)"),
    "amos_K4": dict(kind="ts", site="amos", parts={291: XM / "amos_s1337_K4_FULL_F1b/part291", 293: XM / "amos_s1337_K4_FULL_F1b/part293"},
                    check={"run": "amos_s1337_K4_FULL_F1b", "trainable": "all", "initial_lr": 1e-4},
                    label="AMOS-tuned FULL_K4 (xdata E1; K = 4, seed-1337 draw 0017 0001 0008 0048; F1b recipe)"),
    "mact_K4": dict(kind="ts", site="mact", parts={291: XM / "mact_s1337_K4_FULL_F1b/part291"},
                    check={"run": "mact_s1337_K4_FULL_F1b", "trainable": "all", "initial_lr": 1e-4},
                    label="MACT-tuned FULL_K4 (xdata E1; K = 4 pool scans 0075 0049 0037 0041; F1b recipe, part 291)"),
    "peft_HEAD": dict(kind="ts", site="flare", parts={291: PEFT_M / "HEAD/part291", 293: PEFT_M / "HEAD/part293"},
                      check={"variant": "HEAD", "trainable": "heads", "initial_lr": 1e-2},
                      label="peft HEAD (FLARE K = 4; segmentation heads only, lr 1e-2)"),
    "peft_F1B_LR3E5": dict(kind="ts", site="flare", parts={291: PEFT_M / "F1B_LR3E5/part291", 293: PEFT_M / "F1B_LR3E5/part293"},
                           check={"variant": "F1B_LR3E5", "trainable": "all", "initial_lr": 3e-5},
                           label="peft F1B_LR3E5 (FLARE K = 4; all weights, lr 3e-5)"),
    "peft_STOCK": dict(kind="stock", site="flare", dir=PEFT_M / "STOCK/stock", fold="all", n_classes=13, epochs=250,
                       label="peft STOCK (FLARE K = 4; nnU-Net's pretrained fine-tuning recipe, 13 classes, 250 epochs)"),
    "bslC_amos": dict(kind="export", site="amos", dir=B_DIR / "models/C_amos",
                      label="plan B challenger C (the dev18-best recipe other than F1b) trained on AMOS K = 4 (0017 0001 0008 0048)"),
    "bslC_mact": dict(kind="export", site="mact", dir=B_DIR / "models/C_mact",
                      label="plan B challenger C trained on MACT K = 4 (0075 0049 0037 0041)"),
    "bslXS_amos": dict(kind="export", site="amos", dir=B_DIR / "models/XS_amos",
                       label="plan B cross-site AMOS draw XS: F1b recipe on K = 4 AMOS-240 'people' (SIEMENS) scans"),
}
for _d in (1, 2, 3):
    MODELS[f"word_F1b_d{_d}"] = dict(kind="export", site="word", dir=C_DIR / f"models/F1b_d{_d}",
                                     label=f"plan C: F1b recipe (part 291) on WORD K = 4 training draw {_d}")


# ---------------------------------------------------------------------------------------------------- targets
def _flare_img(cid):
    import xdata_common as X
    return X.raw_img("flare", cid)


def _amos_img(cid):
    import xdata_common as X
    return X.amos_raw_img(cid)


def _mact_img(cid):
    import xdata_common as X
    return X.raw_img("mact", cid)


def _amos240_img(cid):
    import a240_common as A240
    return A240.raw_img(cid)


def _word_img(cid):
    import word_common as W
    return W.image_path(cid)


def target_ids(name):
    """Scan ids of a target set (computed lazily: some need metadata / file listings)."""
    import tsft_full_common as Q
    import xdata_common as X
    if name == "flare_eval28":
        return list(Q.HELD18) + list(Q.TEST10)
    if name == "flare_dev18":
        return list(Q.DEV18)
    if name == "amos56":
        return list(X.eval_ids("amos", 1337))
    if name == "mact81":
        return list(X.eval_ids("mact"))
    if name == "amos240":
        import a240_common as A240
        return list(A240.ids240())
    if name == "word_test30":
        import word_common as W
        return list(W.split()["test"])
    raise KeyError(name)


TARGETS = {
    "flare_eval28": dict(ds="flare", image=_flare_img, cache=TSFT_P / "cache_frozen",
                         label="FLARE22 held18 + test10 (raw grid)"),
    "flare_dev18": dict(ds="flare", image=_flare_img, cache=TSFT_P / "cache_frozen", label="FLARE22 dev18 (design data)"),
    "amos56": dict(ds="amos", image=_amos_img, cache=XP / "amos_F0/cache_frozen", label="AMOS-56 (E1 evaluation set)"),
    "mact81": dict(ds="mact", image=_mact_img, cache=XP / "mact_F0/cache_frozen", label="MACT-81 (the sealed set, opened once)"),
    "amos240": dict(ds="amos", image=_amos240_img, cache=B_DIR / "pred/F0__amos240/cache_frozen",
                    label="AMOS-240 (opened once on 2026-10-22; its F0 frozen-part cache was deleted after that look and is "
                          "regenerated by job F0__amos240)"),
    "word_test30": dict(ds="word", image=_word_img, cache=C_DIR / "pred/F0__word_test30/cache_frozen",
                        label="WORD official test set (30 scans; labels never opened)"),
}

# ---------------------------------------------------------------------------------------------------- stored cells (reused)
# (model, target) -> raw-map directory of predictions that already exist (TS 'total' ids on the raw grid, keyed by the
# target's ids; BTCV-28 maps are keyed by MACT id)
STORED = {
    ("F0", "flare"): TSFT_P / "F0/raw",
    ("F0", "amos56"): XP / "amos_F0/raw",
    ("F0", "mact81"): XP / "mact_F0/raw",
    ("F0", "btcv28"): XP / "mact_F0/raw",
    ("flare_F1b", "flare"): TSFT_P / "F1b/raw",
    ("flare_F1b", "amos56"): VC / "e2/pred/forget_amos_F1b/raw",
    ("flare_F1b", "mact81"): VC / "xdata/g1/pred/X1/raw",
    ("flare_F1b", "btcv28"): VC / "e2/pred/forget_btcv_F1b/raw",
    ("amos_K4", "amos56"): XP / "amos_s1337_K4_FULL_F1b/raw",
    ("mact_K4", "mact81"): XP / "mact_s1337_K4_FULL_F1b/raw",
    ("mact_K4", "btcv28"): XP / "mact_s1337_K4_FULL_F1b/raw",
}
STORED_DONE = {k: v.parent / "DONE.json" for k, v in STORED.items()}


def pred_dir(plan, job, root=None):
    return Path(root or (F.PLAN_DIR[plan] / "pred")) / job


def job_parts(job):
    m, t = job.split("__")
    assert m in MODELS and t in TARGETS, job
    return m, t
