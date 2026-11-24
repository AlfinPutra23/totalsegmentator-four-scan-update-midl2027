"""vccor/e2 (2026-10-18): the ADDITIONAL GPU experiments of the pre-registered study of few-shot adaptation of
full-resolution TotalSegmentator (TS-full, 2.13.0 'total', --ml, no --fast, 1.5 mm). PREPARED, NOT LAUNCHED: nothing in
this directory starts a GPU job; the scheduler patch (queue_v8_patch_e2.py) is written, not applied.

Frozen pipelines are imported / read READ-ONLY (never written):
  scripts/vccor/tsft_full   TS-full recipe (trainer, plans, init incl. heads), ts_pipeline (TS's own inference with
                            swappable part weights), composite training labels, the labelled-4 captures + F1b labels,
                            F0/F1a/F1b predictions, the F0 frozen-part cache (292/294/295)
  scripts/vccor/xdata       FLARE/AMOS/MACT frames, LUTs, metrics (score_scan), the MACT pool, E1's AMOS/MACT F0 runs
                            (+ frozen-part caches) and MACT K4 / K1 predictions
  scripts/vccor/fairfight   the K = 1 / K = 2 FULL (F1b recipe) predictions on held18 + test10
  scripts/vccor/audit_ft    the seed-2025 / 3407 K = 4 FULL predictions (draw variability at K = 4)
  scripts/vccor/ssft        TS's own captures (1.5 mm input image + parts 291/293) of the 36 FLARE pool scans

Blocks (each has its own hashed prereg outputs/results/vccor/e2/prereg_e2_<block>.json, written before any training):
  budget_flare  FULL (F1b recipe) at K = 8 and K = 16 on FLARE seed 1337 -> K curve 0/1/2/4/8/16 (held18 primary)
  budget_mact   FULL at K = 8 on MACT (the 8-scan labelled pool) -> MACT curve 0/1/4/8 on the 81 evaluation scans; the
                SECOND MACT look (after E1's), separately pre-registered
  recipe        FLARE K = 4: LR1E2 (default nnU-Net lr 1e-2), EP10 (10 x 250 it), FAST_F1b (3 mm fast model Dataset297,
                F1b recipe) + FAST_F0 (its zero-shot output through the same GPU path)
  scratch       nnU-Net 3d_fullres from scratch on the same 4 FLARE scans (own plans, nnUNetTrainer_100epochs)
  forgetting    TS classes FLARE does not annotate: BTCV portal/splenic vein, AMOS bladder + prostate/uterus (GT) and
                agreement with F0 on every non-target class in FLARE scans (no GT)

LABELLED SETS. FLARE K = 8 / 16 = the trainer's split ORDER (RandomState(1337).shuffle(flare.txt[:40]), the order whose
first 4 are the K = 4 set) restricted to scans that are NOT evaluation scans (labelled 4 + dev18): nested K4 < K8 < K16,
held18 / test10 unchanged and never opened outside score jobs. The plain prefix (shuffled[:8] / [:16]) would put held18
scans 0039 0028 (K8) and + 0035 0022 (K16) into training (documented; not used). MACT K = 8 = xdata's pool (K4 = its
first 4, nested).

This module has NO import-time side effects beyond sys.path and reading the split file (safe before nnunetv2 / torch).
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path("/home/user/Documents/Medical Imagining SSL")
HERE = Path(__file__).resolve().parent
TSFT_DIR = REPO / "scripts/vccor/tsft_full"
XDATA_DIR = REPO / "scripts/vccor/xdata"
for _d in (TSFT_DIR, XDATA_DIR):                    # appended: those dirs also hold smoke_test.py / write_prereg*.py
    if str(_d) not in sys.path:
        sys.path.append(str(_d))
import tsft_full_common as Q  # noqa: E402  (frozen)
import xdata_common as X  # noqa: E402  (frozen)

OUT = REPO / "outputs/results/vccor/e2"
LOGD = REPO / "outputs/logs/vccor"
NN_ROOT = OUT / "nnunet"                             # nnUNet_raw / nnUNet_preprocessed (transient) + results_<run>
PREP = OUT / "prep"                                  # prep/<group>/{DONE.json, datasets.json, ...}; fast captures
LABELS = OUT / "labels"                              # labels/<group>/part<tid>/<PFX>_<cid>.nii.gz + manifest (kept)
MODELS = OUT / "models"                              # models/<run>/part<tid>/ (TS-compatible) or models/<run>/ (scratch)
PRED = OUT / "pred"                                  # pred/<run>/{raw,h5}/<cid> + DONE.json [+ CLEANUP.json]
RES = OUT / "results"                                # results/<block>/{results.json, tables.md, pervol/, maps.json, GT_OPENED}
SMOKE_RES = OUT / "smoke"                            # smoke_results.json (no data)
PREREG_LOG = LOGD / "e2_prereg.log"
BLOCKS = ("budget_flare", "budget_mact", "recipe", "scratch", "forgetting")
FRAME_CACHE = REPO / "outputs/results/extras/ts_finetune/h5_frames"   # FLARE h5 frames (all 50 cached: no GT read)

# ---------------------------------------------------------------------------------------------------- TS models
TS_RESULTS = Q.TS_RESULTS
FAST_TID, FAST_TRAINER, FAST_SPACING = 297, "nnUNetTrainer_4000epochs_NoMirroring", 3.0
TS297_NAME = "Dataset297_TotalSegmentator_total_3mm_1559subj"
TS297_MODEL = TS_RESULTS / TS297_NAME / f"{FAST_TRAINER}__nnUNetPlans__3d_fullres"
TS_MODEL = {**Q.TS_MODEL, FAST_TID: TS297_MODEL}
TS_DS_NAME = {**Q.TS_DS, FAST_TID: TS297_NAME}
TRAINER_OF = {291: Q.TS_TRAINER, 293: Q.TS_TRAINER, FAST_TID: FAST_TRAINER}
STOCK_EPOCHS = {Q.TS_TRAINER: 1000, FAST_TRAINER: 4000, "nnUNetTrainer_100epochs": 100}
PLANS_ID = {291: "TS291Plans", 293: "TS293Plans", FAST_TID: "TS297Plans"}
SCRATCH_TRAINER, SCRATCH_PLANS = "nnUNetTrainer_100epochs", "nnUNetPlans"
CONFIG, FOLD, ITERS, VAL_ITERS, SAVE_EVERY = Q.CONFIG, Q.FOLD, Q.ITERS_PER_EPOCH, Q.VAL_ITERS_PER_EPOCH, Q.SAVE_EVERY
SEED = Q.SEED                                        # 1337

# ---------------------------------------------------------------------------------------------------- sets
LAB4 = list(Q.LAB)                                   # 0014 0001 0006 0038
DEV18, HELD18, TEST10 = list(Q.DEV18), list(Q.HELD18), list(Q.TEST10)
EVAL28 = HELD18 + TEST10
FLARE_SETS = {"held18": HELD18, "test10": TEST10}
SEED_K4_LAB = {"2025": ["0007", "0026", "0032", "0029"], "3407": ["0018", "0030", "0033", "0004"]}   # audit_ft draws
HELD13 = [c for c in HELD18 if c not in {x for v in SEED_K4_LAB.values() for x in v}]


def flare_order():
    """The trainer's split order: RandomState(1337).shuffle(flare.txt[:40]) (== train_main_flare_octree.py)."""
    return X.trainer_draw(X.flare_ids(), 40, SEED, 40)


def flare_labelled(k):
    """Nested FLARE labelled set of size k: the trainer's order, evaluation scans (held18) skipped (see module doc)."""
    pool = set(LAB4) | set(DEV18)
    seq = [c for c in flare_order() if c in pool]
    assert seq[:4] == LAB4 and len(seq) == 22, seq
    assert k <= len(seq)
    return seq[:k]


def flare_prefix(k):
    """The plain trainer prefix (NOT used; documented): shuffled[:k]."""
    return flare_order()[:k]


def mact_k8():
    return list(X.mact_pool())


def mact_eval():
    return list(X.eval_ids("mact"))


def btcv28():
    """{MACT id: BTCV Training id} of the 28 BTCV-Training images in the MACT evaluation set (E1's mact_F0 covers them;
    the 2 other BTCV-Training images, MACT 0049 / 0070, are MACT pool scans)."""
    S = X.mact_sources()
    ev = set(X.eval_ids("mact"))
    out = {m: s["original_id"] for m, s in S.items() if s["source"] == "btcv" and s["btcv_subset"] == "Training" and m in ev}
    assert len(out) == 28, len(out)
    return out


def amos56():
    return list(X.eval_ids("amos"))


BTCV_IMG = X.BTCV_RAW_ROOT / "Training" / "img" / "img{oid}.nii.gz"
BTCV_LBL = X.BTCV_RAW_ROOT / "Training" / "label" / "label{oid}.nii.gz"

# ---------------------------------------------------------------------------------------------------- groups / runs
# A GROUP = the nnU-Net datasets (one per trained part) of one labelled set; several runs may train on one group.
GROUPS = {
    "flare_K4": dict(ds="flare", kind="ts", parts=(291, 293), labels="tsft_full", ids={291: 701, 293: 702}),
    "flare_K8": dict(ds="flare", kind="ts", parts=(291, 293), labels="compose", ids={291: 703, 293: 704}),
    "flare_K16": dict(ds="flare", kind="ts", parts=(291, 293), labels="compose", ids={291: 705, 293: 706}),
    "flare_K4_fast": dict(ds="flare", kind="fast", parts=(FAST_TID,), labels="compose_fast", ids={FAST_TID: 707}),
    "flare_K4_scratch": dict(ds="flare", kind="scratch", parts=(0,), labels="gt_raw", ids={0: 708}),
    "mact_K8": dict(ds="mact", kind="ts", parts=(291,), labels="compose_mact", ids={291: 709}),
    # task B (budget_flare v2, 2026-10-19): the seed-2025 / 3407 draws at K = 8 / 16 (labels: K4 = the audit block's copies)
    "flare_K8_s2025": dict(ds="flare", kind="ts", parts=(291, 293), labels="compose", ids={291: 711, 293: 712}),
    "flare_K16_s2025": dict(ds="flare", kind="ts", parts=(291, 293), labels="compose", ids={291: 713, 293: 714}),
    "flare_K8_s3407": dict(ds="flare", kind="ts", parts=(291, 293), labels="compose", ids={291: 715, 293: 716}),
    "flare_K16_s3407": dict(ds="flare", kind="ts", parts=(291, 293), labels="compose", ids={291: 717, 293: 718}),
}
SEEDS_X = ("2025", "3407")
AUDIT_LABELS = REPO / "MagicNet/data/vccor_audit_ft"   # s<seed>_FULL/part<tid>/FLARE_<cid>.nii.gz (audit_ft, read-only)
AUDIT_PREP_DONE = REPO / "outputs/results/vccor/audit_ft/prep/DONE.json"


def flare_labelled_seed(seed, k):
    """Task B: K4 = the audit block's draw of the seed (trainer split prefix); K8 / K16 = K4 + the first 4 / 12 dev18 scans in
    the seed's trainer shuffle order (RandomState(seed).shuffle(flare.txt[:40]); scans already in K4 skipped); nested."""
    order = X.trainer_draw(X.flare_ids(), 40, int(seed), 40)
    k4 = order[:4]
    assert k4 == SEED_K4_LAB[str(seed)], (seed, k4)
    extra = [c for c in order if c in DEV18 and c not in k4]
    assert k in (4, 8, 16), k
    return k4 + extra[:k - 4]


def group_scans(g):
    if g.endswith(("_s2025", "_s3407")):                 # task B draws
        k, s = g.split("_")[1], g.split("_s")[-1]
        return flare_labelled_seed(s, int(k.lstrip("K")))
    return {"flare_K4": LAB4, "flare_K8": flare_labelled(8), "flare_K16": flare_labelled(16), "flare_K4_fast": LAB4,
            "flare_K4_scratch": LAB4, "mact_K8": mact_k8()}[g]


RUNS = {   # trained runs (train_e2.py + predict_e2.py)
    "flare_K8_F1b": dict(group="flare_K8", lr=1e-4, epochs=50, block="budget_flare", train_h=2.05, pred_h=0.2,
                         what="FULL K=8 (F1b recipe: all weights, lr 1e-4, 50 x 250 it; parts 291+293)"),
    "flare_K16_F1b": dict(group="flare_K16", lr=1e-4, epochs=50, block="budget_flare", train_h=2.05, pred_h=0.2,
                          what="FULL K=16 (F1b recipe; parts 291+293)"),
    "mact_K8_F1b": dict(group="mact_K8", lr=1e-4, epochs=50, block="budget_mact", train_h=1.05, pred_h=0.4,
                        what="MACT FULL K=8 (F1b recipe; part 291 only, as E1)"),
    "flare_K4_LR1E2": dict(group="flare_K4", lr=1e-2, epochs=50, block="recipe", train_h=2.05, pred_h=0.2,
                           what="LR1E2: F1b recipe with nnU-Net's default lr 1e-2 (all weights, 50 x 250 it)"),
    "flare_K4_EP10": dict(group="flare_K4", lr=1e-4, epochs=10, block="recipe", train_h=0.5, pred_h=0.2,
                          what="EP10: F1b recipe with 10 x 250 it (poly lr to 0 over 10 epochs)"),
    "flare_K4_FAST_F1b": dict(group="flare_K4_fast", lr=1e-4, epochs=50, block="recipe", train_h=0.9, pred_h=0.15,
                              what="FAST_F1b: the 3 mm fast TS model (Dataset297) with the F1b recipe"),
    "flare_K4_SCRATCH": dict(group="flare_K4_scratch", lr=1e-2, epochs=100, block="scratch", train_h=2.8, pred_h=0.6,
                             what="SCRATCH: nnU-Net 3d_fullres from scratch (own plans, nnUNetTrainer_100epochs, lr 1e-2)"),
}
for _s in SEEDS_X:                                       # task B (budget_flare v2): F1b recipe on the seed draws at K = 8 / 16
    for _k in ("K8", "K16"):
        RUNS[f"flare_{_k}_F1b_s{_s}"] = dict(group=f"flare_{_k}_s{_s}", lr=1e-4, epochs=50, block="budget_flare", train_h=2.05,
                                             pred_h=0.2, what=f"FULL {_k} seed {_s} (F1b recipe; parts 291+293)")
PRED_RUNS = {   # prediction-only runs (predict_e2.py)
    "flare_FAST_F0": dict(source="ts297", ds="flare", ids="eval28", block="recipe", pred_h=0.15,
                          what="FAST_F0: TS-fast own weights through the same GPU path"),
    "forget_btcv_F1b": dict(source="tsft_full:F1b", ds="btcv", ids="btcv28", block="forgetting", pred_h=0.25,
                            what="F1b (FLARE-tuned) on the 28 BTCV-Training images", e1_f0="xd_mact_F0"),
    "forget_amos_F1b": dict(source="tsft_full:F1b", ds="amos", ids="amos56", block="forgetting", pred_h=0.65,
                            what="F1b (FLARE-tuned) on the 56 AMOS evaluation scans", e1_f0="xd_amos_F0"),
    "forget_btcv_LR1E2": dict(source="e2:flare_K4_LR1E2", ds="btcv", ids="btcv28", block="forgetting", pred_h=0.25,
                              what="LR1E2 (FLARE-tuned, lr 1e-2) on the 28 BTCV-Training images", e1_f0="xd_mact_F0"),
    "forget_amos_LR1E2": dict(source="e2:flare_K4_LR1E2", ds="amos", ids="amos56", block="forgetting", pred_h=0.65,
                              what="LR1E2 (FLARE-tuned, lr 1e-2) on the 56 AMOS evaluation scans", e1_f0="xd_amos_F0"),
}


def run_spec(run):
    if run in RUNS:
        s = dict(RUNS[run])
        g = GROUPS[s["group"]]
        s.update(ds=g["ds"], kind=g["kind"], parts=g["parts"], trained=True,
                 ids=("mact81" if g["ds"] == "mact" else "eval28"),
                 trainer=s.get("trainer", SCRATCH_TRAINER if g["kind"] == "scratch" else TRAINER_OF[g["parts"][0]]))
        return s
    s = dict(PRED_RUNS[run])
    s.update(trained=False, kind=("fast" if s["source"] == "ts297" else "ts"),
             parts=((FAST_TID,) if s["source"] == "ts297" else (291, 293)))
    return s


def consumers(group):
    return [r for r, s in RUNS.items() if s["group"] == group]


def pred_ids(name):
    """Prediction id set of a run: eval28 (FLARE held18 + test10), mact81 (MACT evaluation set), btcv28 (MACT ids of the
    BTCV-Training images in it), amos56 (AMOS evaluation set of E1)."""
    if name == "eval28":
        return list(EVAL28)
    if name == "mact81":
        return mact_eval()
    if name == "btcv28":
        return sorted(btcv28())
    if name == "amos56":
        return amos56()
    raise KeyError(name)


def case_prefix(ds):
    return {"flare": "FLARE", "mact": "MACT"}[ds]


def dataset_id(g, tid):
    return GROUPS[g]["ids"][tid]


def dataset_name(g, tid):
    return f"Dataset{dataset_id(g, tid)}_E2_{g}" + ("" if tid == 0 else f"_{tid}")


def results_root(run, nn_root=None):
    return Path(nn_root or NN_ROOT) / f"results_{run}"


def model_dir(run, tid, models_root=None):
    return Path(models_root or MODELS) / run / (f"part{tid}" if tid else "scratch")


def model_done(run, models_root=None):
    return Path(models_root or MODELS) / run / "DONE.json"


def pred_dir(run, pred_root=None):
    return Path(pred_root or PRED) / run


def pred_done(run, pred_root=None):
    return pred_dir(run, pred_root) / "DONE.json"


LABEL_SET = {"flare_K4": "flare_ts", "flare_K8": "flare_ts", "flare_K16": "flare_ts", "flare_K4_fast": "flare_fast",
             "flare_K4_scratch": "flare_scratch", "mact_K8": "mact_ts"}   # a scan's training label is group-independent
LABEL_SET.update({g: "flare_ts" for g in GROUPS if g.endswith(("_s2025", "_s3407"))})     # task B draws share flare_ts


def labels_dir(g, tid, root=None):
    """labels/<label set>/part<tid>/ (scratch: gt_raw/): one file per scan, shared by the groups of that label set."""
    return Path(root or LABELS) / LABEL_SET[g] / (f"part{tid}" if tid else "gt_raw")


def label_manifest(g, root=None):
    return Path(root or LABELS) / LABEL_SET[g] / "manifest.json"


def prep_done(g, prep_root=None):
    return Path(prep_root or PREP) / g / "DONE.json"


# ---------------------------------------------------------------------------------------------------- label maps
def fast_labels():
    return json.load(open(TS297_MODEL / "dataset.json"))["labels"]


def fast_lut():
    """uint8 LUT Dataset297 class id -> FLARE id. 297's ids ARE TotalSegmentator 'total' ids (asserted by name), so this
    is tsft_common.ts_to_flare_lut('v2') (kidney_cyst_left 23 -> 13, kidney_cyst_right 24 -> 2), the LUT of every TS-full
    arm."""
    sys.path.insert(0, str(REPO / "scripts/extras/ts_finetune"))
    import tsft_common as C
    from totalsegmentator.map_to_binary import class_map
    lab = fast_labels()
    assert {int(v): k for k, v in lab.items() if int(v) > 0} == dict(class_map["total"]), "297 ids != 'total' ids"
    return C.ts_to_flare_lut("v2")


def fast_owned():
    """FLARE organ id -> its canonical Dataset297 class id (all 13 FLARE organs live in the single fast model)."""
    lab = fast_labels()
    return {o: int(lab[Q.FLARE_TS_NAMES[o]]) for o in range(1, 14)}


def composite(G, P, lut, owned, n_labels=14):
    """Training label of one TS model on its own canonical grid, with an explicit LUT (model class id -> dataset id) and
    ownership {dataset organ: canonical model class id}. Rule == tsft_full_common.composite_target (asserted bit-exact for
    parts 291 / 293 in smoke T0):
      GT organ o owned          -> P if lut[P] == o (e.g. a same-side kidney cyst), else canon(o)
      GT organ not owned        -> 0
      GT background, P non-ds   -> P   (the model's own label for every class the dataset does not annotate)
      GT background, P ds class -> 0   (a TS false positive under the dataset's protocol)
    Returns (Y uint8, counts)."""
    import numpy as np
    canon = np.zeros(n_labels, np.uint8)
    own = np.zeros(n_labels, bool)
    for o, j in owned.items():
        canon[o], own[o] = j, True
    G = G.astype(np.uint8)
    P = P.astype(np.uint8)
    assert G.shape == P.shape and int(G.max()) < n_labels
    flp = lut[P]
    Y = np.zeros(P.shape, np.uint8)
    bg = G == 0
    keep = bg & (P > 0) & (flp == 0)
    Y[keep] = P[keep]
    ow = own[G]
    same = ow & (flp == G)
    Y[same] = P[same]
    relab = ow & ~same
    Y[relab] = canon[G[relab]]
    other = (G > 0) & ~ow
    cnt = {"n_vox": int(G.size), "unchanged": int((Y == P).sum()), "gt_owned_relabelled": int(relab.sum()),
           "gt_owned_relabelled_from_nonds": int((relab & (P > 0) & (flp == 0)).sum()),
           "gt_other_part_zeroed": int((other & (P > 0)).sum()), "ts_ds_fp_removed": int((bg & (flp > 0)).sum()),
           "nonds_kept": int(keep.sum()), "cyst_label_kept": int((same & (P != canon[G])).sum())}
    return Y, cnt


def total_name_lut(names_to_ids):
    """uint8 LUT TS 'total' id -> dataset id from {TS class name: dataset id}."""
    import numpy as np
    from totalsegmentator.map_to_binary import class_map
    inv = {v: k for k, v in class_map["total"].items()}
    lut = np.zeros(256, np.uint8)
    for n, i in names_to_ids.items():
        lut[inv[n]] = i
    return lut


# Forgetting look: dataset label spaces on the RAW grid (TS 'total' names -> dataset ids; kidney cysts -> same-side kidney,
# as the v2 LUT of every FLARE arm). BTCV = the native 13-organ scheme (10 = portal + splenic vein); AMOS = the native 15.
BTCV_CLASSES = {1: "spleen", 2: "r.kid", 3: "l.kid", 4: "gall", 5: "esoph", 6: "liver", 7: "stom", 8: "aorta", 9: "ivc",
                10: "portal_splenic_vein", 11: "panc", 12: "r.adr", 13: "l.adr"}
BTCV_TS = {"spleen": 1, "kidney_right": 2, "kidney_cyst_right": 2, "kidney_left": 3, "kidney_cyst_left": 3,
           "gallbladder": 4, "esophagus": 5, "liver": 6, "stomach": 7, "aorta": 8, "inferior_vena_cava": 9,
           "portal_vein_and_splenic_vein": 10, "pancreas": 11, "adrenal_gland_right": 12, "adrenal_gland_left": 13}
AMOS_CLASSES = {1: "spleen", 2: "r.kid", 3: "l.kid", 4: "gall", 5: "esoph", 6: "liver", 7: "stom", 8: "aorta", 9: "ivc",
                10: "panc", 11: "r.adr", 12: "l.adr", 13: "duod", 14: "bladder", 15: "prostate_uterus"}
AMOS_TS = {"spleen": 1, "kidney_right": 2, "kidney_cyst_right": 2, "kidney_left": 3, "kidney_cyst_left": 3,
           "gallbladder": 4, "esophagus": 5, "liver": 6, "stomach": 7, "aorta": 8, "inferior_vena_cava": 9, "pancreas": 10,
           "adrenal_gland_right": 11, "adrenal_gland_left": 12, "duodenum": 13, "urinary_bladder": 14, "prostate": 15}
FORGET_PRIMARY = {"btcv": [10], "amos": [14, 15]}     # the classes FLARE does not annotate (primary forgetting classes)


# ---------------------------------------------------------------------------------------------------- helpers
sha256, rel, write_json, Log, be_nice, cap_threads = Q.sha256, Q.rel, Q.write_json, Q.Log, Q.be_nice, Q.cap_threads


def now():
    return time.strftime("%F %T")


def arr_sha(a):
    import numpy as np
    a = np.ascontiguousarray(a)
    return hashlib.sha256(f"{a.dtype}|{a.shape}|".encode() + a.tobytes()).hexdigest()


def set_nnunet_env(results_root, nn_root=None):
    """Component-owned nnU-Net roots. Must run before nnunetv2 is imported anywhere in the process."""
    if "nnunetv2" in sys.modules:
        raise RuntimeError("set_nnunet_env() must be called before importing nnunetv2")
    root = Path(nn_root) if nn_root else NN_ROOT
    os.environ["nnUNet_raw"] = str(root / "nnUNet_raw")
    os.environ["nnUNet_preprocessed"] = str(root / "nnUNet_preprocessed")
    os.environ["nnUNet_results"] = str(results_root)


def dir_bytes(p):
    p = Path(p)
    if not p.exists():
        return 0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and not f.is_symlink())


def flare_frame_cached(cid):
    """FLARE predictions are mapped with tsft_common.raw_to_h5, whose frame comes from the h5-frame cache; a missing cache
    entry would make it read the case's GT -> prediction jobs refuse instead."""
    return (FRAME_CACHE / f"{cid}.json").is_file()


# ---------------------------------------------------------------------------------------------------- prereg / freeze
def prereg_path(block):
    assert block in BLOCKS, block
    return OUT / f"prereg_e2_{block}.json"


def prereg_sha_logged(path):
    """sha256 logged for a prereg ('PREREG sha256 <hex> <repo-relative path>' lines of e2_prereg.log; last wins)."""
    h, want = None, rel(path)
    if PREREG_LOG.is_file():
        for line in open(PREREG_LOG):
            p = line.split()
            if len(p) >= 6 and p[2] == "PREREG" and p[3] == "sha256" and " ".join(p[5:]) == want:
                h = p[4]
    return h


def frozen_code_hashes():
    """{repo-relative path: sha256} from the CODE / IMPORT lines of e2_prereg.log (the last entry per path wins)."""
    out = {}
    if PREREG_LOG.is_file():
        for line in open(PREREG_LOG):
            p = line.split()
            if len(p) >= 5 and p[2] in ("CODE", "IMPORT") and len(p[3]) == 64:
                out[" ".join(p[4:])] = p[3]
    return out


def check_code_frozen(paths, log=print, strict=True):
    fz = frozen_code_hashes()
    bad = {}
    for pth in paths:
        r = rel(pth)
        got = sha256(REPO / r)
        if fz.get(r) != got:
            bad[r] = (got[:16], (fz.get(r) or "NOT-FROZEN")[:16])
    if bad:
        msg = f"code differs from the CODE hashes in {rel(PREREG_LOG)}: {bad}"
        if strict:
            raise SystemExit("[REFUSE] " + msg)
        log("[warn] " + msg)
    return not bad


def frozen_if_registered(paths, log=print):
    """Real (non-smoke) jobs: strict code check once a CODE freeze exists in the prereg log; before any freeze: warn."""
    if frozen_code_hashes():
        return check_code_frozen(paths, log, strict=True)
    log(f"[warn] no CODE freeze in {rel(PREREG_LOG)} yet: running unfrozen code")
    return False


COMMON_CODE = [HERE / "e2_common.py", TSFT_DIR / "tsft_full_common.py", XDATA_DIR / "xdata_common.py"]
