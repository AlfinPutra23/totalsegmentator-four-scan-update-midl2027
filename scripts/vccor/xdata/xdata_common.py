"""vccor/xdata: the TS-full (TotalSegmentator 2.13.0 'total', --ml, no --fast, 1.5 mm) few-shot fine-tuning pipeline of
scripts/vccor/tsft_full (+ the partial-label composition of audit_ft and the single-scan draws of fairfight), generalised
from FLARE to {flare, amos, mact} for the cross-dataset replications. NO-REGRET PREPARATION (2026-10-17): nothing here
launches a GPU job; the scheduler patch (queue_v8_patch_xdata.py) is written, not applied.

Datasets (label spaces are the trainers' h5 label spaces):
  flare  13 organs, FLARE22 ids liver-first 1..13 (tsft_full_common.NAMES); h5 frame = tsft_common.raw_to_h5 (1.5 mm zoom
         order 0 + flare_h5 GT-bbox crop); raw CT data/FLARE22Train; GT on the raw grid (labelled scans) / flare_h5 (scorer).
  amos   the same 13 organs in FLARE ids (MagicNet/data/amos_ct_h5; AMOS 14 bladder / 15 prostate-uterus -> 0, as
         scripts/prep_amos_ct.py); h5 frame = zoom spacing/1.5 order 0 'nearest' + the GT-bbox crop (-10/+11) recorded per
         case by the cor_feas build (outputs/results/cor_feas/ts_full_amos_btcv/build/amos_<cid>.json, read-only; no GT is
         read to map a prediction); raw CT /media/.../amos_ct; GT = raw labelsTr (labelled scans) / amos_ct_h5 (scorer).
  mact   8 organs, MACT h5 ids 1 spleen, 2 left kidney, 3 gallbladder, 4 oesophagus, 5 liver, 6 stomach, 7 pancreas,
         8 duodenum (Gibson et al. TMI 2018 order with the BTCV-only labels removed; value set {0..8} confirmed on the two
         first labelled-pool scans only). MACT annotates the LEFT kidney only: TS's kidney_right / kidney_cyst_right are
         non-MACT classes (kept as TS's own labels in training targets, 0 in the MACT label space). kidney_cyst_left -> 2
         (as the v2 FLARE LUT puts cysts into the kidney).
         h5 frame (MagicNet's released MACT_h5, reproduced to ~1e-12 from raw CT, image only): raw CT -> LPS array
         (index increases to patient Left / Posterior / Superior) -> Gibson's cropping.csv box [left:right, ant:post,
         inf:sup] -> ITK-style linear resampling to 1 mm (h5 voxel i <-> LPS index start + i * 1/spacing, cval 0) -> clip
         [-125, 275] HU -> z-score. Labels / predictions go through the same grid with nearest neighbour (cval 0).
         Raw CT: MACT 0044-0090 = BTCV img (Training 0001-0040 = 30, Testing 0061-0080 minus 0071-0073 = 17; 47 overlap
         BTCV); MACT 0001-0043 = TCIA Pancreas-CT (public, CC BY 3.0), downloaded + converted with TotalSegmentator's own
         dcm_to_nifti (dicom2nifti, reorient) and stored losslessly as int16 under MagicNet/data/vccor_xdata/mact/raw_ct.
         PANCREAS_0025 (= MACT 0022) is WITHDRAWN from TCIA (TCIA: "same scan as #2" = MACT 0001; the withdrawn series had
         a different in-plane spacing, so it cannot be rebuilt from PANCREAS_0002): no raw CT, no TS-full label; 0022 is
         a duplicate scan of 0001 and is EXCLUDED from the MACT evaluation set (recorded in the split file).
         Outside Gibson's box the MACT reference "may be incomplete" (Zenodo README) and the h5 holds no label there: on
         TS's grid such voxels are UNKNOWN and keep TS's own prediction in the training targets (valid mask).
Parts fine-tuned (derived from TS's class_map_5_parts by organ ownership): flare/amos 291 (organs) + 293 (aorta, IVC);
mact 291 only (all 8 MACT organs are part-1 organs). The other parts keep TS's weights (replayed when cached).

This module has NO import-time side effects beyond sys.path (safe before nnunetv2 / torch).
Everything under scripts/vccor/{tsft_full,audit_ft,fairfight}, scripts/cor_*, outputs/results/vccor/<other blocks> and
outputs/results/cor_* is imported / read ONLY.
"""
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path("/home/user/Documents/Medical Imagining SSL")
HERE = Path(__file__).resolve().parent
TSFT_DIR = REPO / "scripts/vccor/tsft_full"            # FROZEN, read-only (label functions, ts_pipeline, train recipe)
if str(TSFT_DIR) not in sys.path:
    sys.path.append(str(TSFT_DIR))                       # appended: tsft_full also holds smoke_test.py / write_prereg.py
import tsft_full_common as Q  # noqa: E402

OUT = REPO / "outputs/results/vccor/xdata"
LOGD = REPO / "outputs/logs/vccor"
DATA = REPO / "MagicNet/data"
DATA_X = DATA / "vccor_xdata"                           # every NEW data file of this block (except the MACT label set)
SPLITS_DIR = OUT / "splits"
FRAMES_DIR = OUT / "frames"                             # per-dataset frame caches (AMOS crops copied from cor_feas)
NN_ROOT = OUT / "nnunet"
NN_RAW = NN_ROOT / "nnUNet_raw"
NN_PRE = NN_ROOT / "nnUNet_preprocessed"
PREP = OUT / "prep"                                     # prep/<ds>/ts_cpu/<cid>/ captures, prep/<ds>/DONE.json, verify
MODELS = OUT / "models"
PRED = OUT / "pred"
MAPS = OUT / "maps"
RES = OUT / "results"
LABELS_ROOT = DATA_X / "labels"                         # labels/<ds>/<labelcfg>/{gt_ts,part291,part293}/<PFX>_<cid>.nii.gz

# ---- MACT
MACT_H5 = DATA / "MACT_h5"
MACT_X = DATA_X / "mact"
MACT_META = MACT_X / "meta"                             # cropping.csv / index.htm (Zenodo 1169361), TCIA series list
MACT_CROPPING = MACT_META / "cropping.csv"
MACT_CROPPING_SHA = "d6503bda3d776c10698523aae14446409bc9d3722fb405b0025b4a5f59094f8f"
MACT_TCIA_SERIES = MACT_META / "tcia_pancreas_ct_series.json"
MACT_RAW_CT = MACT_X / "raw_ct"                         # tcia/PANCREAS_<oid>.nii.gz
MACT_DL_TMP = MACT_X / "dl_tmp"
MACT_TS_RAW = MACT_X / "ts_full_raw"                    # <rawkey>.nii.gz: TS 'total' ids on the raw CT grid
MACT_PSEUDO = DATA / "mact_pseudo_fullres"              # trainer format (== flare_pseudo_fullres layout)
MACT_JOB_OUT = OUT / "mact_ts"                          # timing/<rawkey>.json, build/<mid>.json, manifest.json
MACT_LOG = LOGD / "xdata_mact_ts_full.log"
MACT_PID = LOGD / "xdata_mact_ts_full.pid"
MACT_LOCK = LOGD / "xdata_mact_ts_full.lock"
BTCV_RAW_ROOT = Path("/media/user/B4864CD4864C98AE/medssl_data/BTCV_raw/RawData")
COR_FEAS_TS_RAW = DATA / "ts_fullres_ext/raw_full"      # cor_feas TS-full outputs (amos/<cid>, btcv/<cid>), read-only
COR_FEAS_TIMING = REPO / "outputs/results/cor_feas/ts_full_amos_btcv/timing"
COR_FEAS_BUILD = REPO / "outputs/results/cor_feas/ts_full_amos_btcv/build"
MACT_WITHDRAWN = {"0022": "PANCREAS_0025 withdrawn from TCIA (duplicate scan of PANCREAS_0002 = MACT 0001; different "
                          "in-plane spacing, so the MACT 0022 frame cannot be rebuilt from PANCREAS_0002)"}
MACT_DUPLICATES = [("0001", "0022")]

# ---- AMOS
AMOS_H5 = DATA / "amos_ct_h5"
AMOS_RAW = Path("/media/user/B4864CD4864C98AE/amos_ct")
AMOS_PSEUDO_FULLRES = DATA / "amos_pseudo_fullres"
AMOS_TO_FLARE = {1: 3, 2: 2, 3: 13, 4: 9, 5: 10, 6: 1, 7: 11, 8: 5, 9: 6, 10: 4, 11: 7, 12: 8, 13: 12}   # prep_amos_ct.py

# ---- FLARE
FLARE_RAW_IMG = REPO / "data/FLARE22Train/images/FLARE22_Tr_{cid}_0000.nii.gz"
FLARE_RAW_LBL = REPO / "data/FLARE22Train/labels/FLARE22_Tr_{cid}.nii.gz"
FLARE_H5 = DATA / "flare_h5"
FLARE_PSEUDO_FULLRES = DATA / "flare_pseudo_fullres"

TS_PARTS = (291, 292, 293, 294, 295)                    # TS merge order
SEEDS = (1337, 2025, 3407)
TS_THREADS = 6                                          # TS inference torch threads (CPU); everything else <= 4

FLARE_NAMES13 = list(Q.NAMES)
FLARE_TS13 = list(Q.FLARE_TS_NAMES[1:])
SPECS = {
    "flare": dict(names=FLARE_NAMES13, ts_names=FLARE_TS13, extra={"kidney_cyst_left": 13, "kidney_cyst_right": 2},
                  h5_spacing=1.5, prefix="FLARE", n_train=40),
    "amos": dict(names=FLARE_NAMES13, ts_names=FLARE_TS13, extra={"kidney_cyst_left": 13, "kidney_cyst_right": 2},
                 h5_spacing=1.5, prefix="AMOS", n_train=40),
    "mact": dict(names=["spleen", "l.kid", "gall", "esoph", "liver", "stom", "panc", "duod"],
                 ts_names=["spleen", "kidney_left", "gallbladder", "esophagus", "liver", "stomach", "pancreas", "duodenum"],
                 extra={"kidney_cyst_left": 2}, h5_spacing=1.0, prefix="MACT", n_train=None),
}
DATASETS = tuple(SPECS)


def n_org(ds):
    return len(SPECS[ds]["names"])


def names(ds):
    return list(SPECS[ds]["names"])


# ---------------------------------------------------------------------------------------------------- helpers
sha256, rel, write_json, Log, be_nice, cap_threads = Q.sha256, Q.rel, Q.write_json, Q.Log, Q.be_nice, Q.cap_threads


def now():
    return time.strftime("%F %T")


def sha16(p):
    return sha256(p)[:16]


def h5_sha(arr):
    import numpy as np
    a = np.ascontiguousarray(arr)
    return hashlib.sha256(f"{a.dtype}|{a.shape}|".encode() + a.tobytes()).hexdigest()


def set_nnunet_env(results_root, nn_root=None):
    """Component-owned nnU-Net roots. Must run before nnunetv2 is imported anywhere in the process."""
    if "nnunetv2" in sys.modules:
        raise RuntimeError("set_nnunet_env() must be called before importing nnunetv2")
    root = Path(nn_root) if nn_root else NN_ROOT
    os.environ["nnUNet_raw"] = str(root / "nnUNet_raw")
    os.environ["nnUNet_preprocessed"] = str(root / "nnUNet_preprocessed")
    os.environ["nnUNet_results"] = str(results_root)


# ---------------------------------------------------------------------------------------------------- label maps
def ts_total_names():
    from totalsegmentator.map_to_binary import class_map
    return dict(class_map["total"])


def total_lut(ds):
    """uint8 LUT TotalSegmentator 'total' id -> dataset id, BY NAME (canonical classes + the dataset's cyst rule).
    flare/amos == tsft_common.ts_to_flare_lut('v2') (asserted in check_luts)."""
    import numpy as np
    inv = {v: k for k, v in ts_total_names().items()}
    lut = np.zeros(256, np.uint8)
    for o, n in enumerate(SPECS[ds]["ts_names"], start=1):
        lut[inv[n]] = o
    for n, o in SPECS[ds]["extra"].items():
        lut[inv[n]] = o
    return lut


def part_classes(tid):
    return Q.part_classes(tid)


def part_lut(ds, tid):
    """uint8 LUT part class id -> dataset id (through TS's 'total' ids, i.e. what TS-full's merged output undergoes)."""
    import numpy as np
    from totalsegmentator.map_to_binary import class_map_5_parts, map_taskid_to_partname_ct
    inv = {v: k for k, v in ts_total_names().items()}
    pmap = class_map_5_parts[map_taskid_to_partname_ct[tid]]
    assert pmap == part_classes(tid), f"TS class_map_5_parts != dataset.json labels for {tid}"
    tl = total_lut(ds)
    lut = np.zeros(256, np.uint8)
    for j, n in pmap.items():
        lut[j] = tl[inv[n]]
    return lut


def owned_organs(ds, tid):
    """dataset organ id -> canonical part class id, for the organs whose canonical TS class lives in part tid."""
    pn = {n: j for j, n in part_classes(tid).items()}
    return {o: pn[n] for o, n in enumerate(SPECS[ds]["ts_names"], start=1) if n in pn}


def parts_ft(ds):
    """The TS parts that own at least one organ of the dataset (= the parts that are fine-tuned)."""
    return tuple(t for t in TS_PARTS if owned_organs(ds, t))


def parts_frozen(ds):
    return tuple(t for t in TS_PARTS if t not in parts_ft(ds))


def check_luts():
    """Cross-checks against the frozen FLARE functions (the regression anchor) and the AMOS/BTCV label-set builders."""
    import numpy as np
    sys.path.insert(0, str(REPO / "scripts/extras/ts_finetune"))
    import tsft_common as C
    assert np.array_equal(total_lut("flare"), C.ts_to_flare_lut("v2")), "flare total LUT != ts_to_flare_lut('v2')"
    for t in TS_PARTS:
        assert np.array_equal(part_lut("flare", t), Q.part_flare_lut(t)), f"flare part LUT {t} != tsft_full part_flare_lut"
        assert owned_organs("flare", t) == Q.owned_organs(t), t
    assert parts_ft("flare") == Q.PARTS_FT and parts_ft("amos") == Q.PARTS_FT and parts_ft("mact") == (291,)
    sys.path.insert(0, str(REPO / "scripts/cor_feas/ts_full_amos_btcv"))
    import ext_common as E
    lut = total_lut("amos")
    assert all(lut[k] == v for k, v in E.TS_TO_FLARE_V2.items()) and int((lut > 0).sum()) == len(E.TS_TO_FLARE_V2)
    assert E.AMOS_TO_FLARE == AMOS_TO_FLARE
    m = total_lut("mact")
    inv = {v: k for k, v in ts_total_names().items()}
    assert m[inv["kidney_right"]] == 0 and m[inv["kidney_cyst_right"]] == 0 and m[inv["kidney_cyst_left"]] == 2
    assert int((m > 0).sum()) == 9 and sorted(set(m[m > 0].tolist())) == list(range(1, 9))
    return True


# ---------------------------------------------------------------------------------------------------- label composition
def composite_target(ds, G, P, tid, lut=None, valid=None):
    """Training label for sub-model tid on TS's 1.5 mm canonical grid (== tsft_full_common.composite_target for FLARE,
    generalised to the dataset's label space; asserted bit-exact for flare in the regression check).
    G: dataset GT (0..n) on that grid; P: TS-tid's own prediction (part class ids) on that grid.
      GT organ o owned by tid       -> P if P already maps to o (e.g. kidney_cyst_left), else canon_tid(o)
      GT organ owned by other part  -> 0
      GT background, P non-dataset  -> P   (TS's knowledge of every class the dataset does not annotate, incl. TS classes
                                            of organs the dataset does not annotate, e.g. kidney_right for MACT)
      GT background, P dataset class -> 0  (a TS false positive under the dataset's protocol)
      valid == False (GT unknown: MACT outside Gibson's box) -> P (TS's own prediction, every class)
    Returns (Y uint8, counts dict)."""
    import numpy as np
    lut = part_lut(ds, tid) if lut is None else lut
    own = owned_organs(ds, tid)
    n = n_org(ds) + 1
    canon = np.zeros(n, np.uint8)
    owned = np.zeros(n, bool)
    for o, j in own.items():
        canon[o], owned[o] = j, True
    G = G.astype(np.uint8)
    P = P.astype(np.uint8)
    assert G.shape == P.shape and int(G.max()) < n, (G.shape, P.shape, int(G.max()))
    flp = lut[P]
    Y = np.zeros(P.shape, np.uint8)
    bg = G == 0
    keep_non = bg & (P > 0) & (flp == 0)
    Y[keep_non] = P[keep_non]
    ow = owned[G]
    same = ow & (flp == G)
    Y[same] = P[same]
    relab = ow & ~same
    Y[relab] = canon[G[relab]]
    other = (G > 0) & ~ow
    cnt = {"n_vox": int(G.size), "unchanged": None,
           "gt_owned_relabelled": int(relab.sum()),
           "gt_owned_relabelled_from_nonds": int((relab & (P > 0) & (flp == 0)).sum()),
           "gt_other_part_zeroed": int((other & (P > 0)).sum()),
           "ts_ds_fp_removed": int((bg & (flp > 0)).sum()),
           "nonds_kept": int(keep_non.sum()),
           "cyst_label_kept": int((same & (P != canon[G])).sum())}
    if valid is not None:
        valid = np.asarray(valid, bool)
        assert valid.shape == G.shape and not (G[~valid] > 0).any(), "GT must be 0 where it is unknown"
        inv_ = ~valid
        cnt["unknown_vox"] = int(inv_.sum())
        cnt["unknown_kept_ts"] = int((inv_ & (P > 0)).sum())
        cnt["unknown_kept_ts_ds_class"] = int((inv_ & (flp > 0)).sum())
        cnt["ts_ds_fp_removed"] = int((bg & valid & (flp > 0)).sum())
        cnt["nonds_kept"] = int((keep_non & valid).sum())
        Y[inv_] = P[inv_]
    cnt["unchanged"] = int((Y == P).sum())
    return Y, cnt


def partial_gt(ds, G, Tf, annotated, valid=None):
    """Partial annotation on TS's 1.5 mm grid (== audit_ft_common.partial_gt, n organs; asserted bit-exact for flare).
      G == o, o annotated                 -> o
      otherwise Tf == u, u NOT annotated  -> u   (TS-full's own label fills every unannotated organ)
      otherwise                           -> 0
    With every organ annotated the result is G itself. valid (MACT) is passed through to composite_target unchanged:
    unknown voxels have G == 0 and get Tf's unannotated organs here, TS's own part prediction there."""
    import numpy as np
    n = n_org(ds) + 1
    ann = np.zeros(n, bool)
    for o in annotated:
        ann[int(o)] = True
    assert not ann[0]
    G = G.astype(np.uint8)
    Tf = Tf.astype(np.uint8)
    Gp = np.where(ann[G], G, 0).astype(np.uint8)
    fill = (Gp == 0) & (Tf > 0) & ~ann[Tf]
    if valid is not None:
        fill &= np.asarray(valid, bool)                 # unknown voxels stay 0 (composite_target gives them TS's own P)
    Gp[fill] = Tf[fill]
    cnt = {"n_vox": int(G.size), "annotated_organs": sorted(int(o) for o in annotated),
           "gt_kept": int(ann[G].sum()), "ts_filled": int(fill.sum()),
           "gt_unannotated_vox": int(((G > 0) & ~ann[G]).sum()),
           "gt_unannotated_vox_ts_agrees": int(((G > 0) & ~ann[G] & (Gp == G)).sum()),
           "ts_annotated_organ_outside_gt_zeroed": int(((Tf > 0) & ann[Tf] & (Gp == 0)).sum()),
           "differs_from_gt": int((Gp != G).sum())}
    return Gp, cnt


def ts_full_ds_map(ds, parts, luts=None):
    """TS-full's merged output on the 1.5 mm grid (parts 291..295 in TS's order; a later part's non-background class
    overwrites), mapped to dataset ids through the part LUTs (== audit_ft_common.ts_full_flare_map for flare)."""
    import numpy as np
    luts = luts or {t: part_lut(ds, t) for t in TS_PARTS}
    Tf = None
    for tid in TS_PARTS:
        P = parts[tid].astype(np.uint8)
        if Tf is None:
            Tf = np.zeros(P.shape, np.uint8)
        nz = P > 0
        Tf[nz] = luts[tid][P[nz]]
    return Tf


def audit_flags(vols, min_sign=3):
    """== audit_ft_common.audit_flags for any number of organs: vols (n_scans, n_org, 2) = (|G_o|, |T_o|). Uses the frozen
    vv2_common.vc_fit (shape-agnostic). flagged = non-zero budget AND same sign on >= min_sign scans."""
    import numpy as np
    sys.path.insert(0, str(REPO / "scripts/cor_salvage/verify_v2"))
    import vv2_common as W
    v = np.asarray(vols, np.float64)
    assert v.ndim == 3 and v.shape[0] >= 2 and v.shape[2] == 2, v.shape
    k = v.shape[1]
    F = W.vc_fit(v)
    r, delta, eps = F["r"], F["delta"], F["eps"]
    n_pos, n_neg = (r > 0).sum(0), (r < 0).sum(0)
    grow = (delta > 0) & (n_pos >= min_sign)
    trim = (eps > 0) & (n_neg >= min_sign)
    assert not (grow & trim).any()
    direction = ["grow" if grow[j] else ("trim" if trim[j] else "off") for j in range(k)]
    budget = [j for j in range(k) if delta[j] > 0 or eps[j] > 0]
    return dict(r=r.tolist(), mean_r=r.mean(0).tolist(), se_r=(r.std(0, ddof=1) / np.sqrt(r.shape[0])).tolist(),
                delta=delta.tolist(), eps=eps.tolist(), n_pos=n_pos.tolist(), n_neg=n_neg.tolist(), min_sign=min_sign,
                direction=direction, flagged=[j + 1 for j in range(k) if grow[j] or trim[j]],
                budget_only_flagged=[j + 1 for j in budget],
                dropped_by_sign_rule=[j + 1 for j in budget if not (grow[j] or trim[j])])


def audit_arm_sets(ds, seed, au, mean_dice):
    """FULL / AUDIT / DICE / RANDOM / ANTI sets of one draw (== audit_ft audit.py rules with n organs)."""
    import numpy as np
    k_all = n_org(ds)
    k = len(au["flagged"])
    assert k >= 1, "no organ flagged"
    order = sorted(range(k_all), key=lambda j: (mean_dice[j], j))
    dice_set = sorted(j + 1 for j in order[:k])
    rng = np.random.default_rng(seed)
    random_set = sorted(int(x) + 1 for x in rng.choice(k_all, k, replace=False))
    anti = [o for o in range(1, k_all + 1) if o not in au["flagged"]]
    return {"FULL": list(range(1, k_all + 1)), "AUDIT": list(au["flagged"]), "DICE": dice_set, "RANDOM": random_set,
            "ANTI": anti}


# ---------------------------------------------------------------------------------------------------- splits
def _ids(txt):
    return ["{}".format(x.strip().split('.')[0][-4:]) for x in open(txt).readlines() if x.strip()]   # trainer parsing


def flare_ids():
    return _ids(DATA / "flare.txt")


def amos_ids():
    return _ids(DATA / "amos.txt")


def mact_ids():
    return [f"{i:04d}" for i in range(1, 91)]


def trainer_draw(ids, n_train, seed, k):
    """== train_main_flare_octree.py / ext_common._split: RandomState(seed).shuffle(ids[:n_train])[:k]."""
    import numpy as np
    tr = list(ids[:n_train])
    np.random.RandomState(int(seed)).shuffle(tr)
    return tr[:k]


MACT_POOL_SEED, MACT_POOL_N = 1337, 8


def mact_pool(n=MACT_POOL_N):
    """THE MACT labelled pool (MACT has no trainer split): RandomState(1337).shuffle(0001..0090) with withdrawn scans
    skipped, first n (= 0075 0049 0037 0041 0070 0087 0011 0080 for n = 8; no withdrawn scan is skipped for 1337)."""
    import numpy as np
    o = list(mact_ids())
    np.random.RandomState(MACT_POOL_SEED).shuffle(o)
    return [c for c in o if c not in MACT_WITHDRAWN][:n]


def labelled(ds, seed, k=4):
    """The K=k labelled scans of a seed. flare/amos: the trainer split replay (flare.txt[:40] / amos.txt[:40]).
    mact: seed 1337 = the pool prefix pool[:k] (K=4 = the first 4 of the 8-scan pool); other seeds = an alternative draw
    WITHIN the same pool, RandomState(seed).shuffle(pool)[:k], so the MACT evaluation set never changes."""
    import numpy as np
    if ds == "flare":
        return trainer_draw(flare_ids(), 40, seed, k)
    if ds == "amos":
        return trainer_draw(amos_ids(), 40, seed, k)
    pool = mact_pool()
    assert k <= len(pool), (k, len(pool))
    if int(seed) == MACT_POOL_SEED:
        return pool[:k]
    o = list(pool)
    np.random.RandomState(int(seed)).shuffle(o)
    return o[:k]


def eval_ids(ds, seed=1337):
    """flare: dev18 + held18 + test10 (tsft_full sets); amos: amos.txt minus the seed's 4 labelled scans (36 pool + 20
    test = 56); mact: every scan outside the 8-scan labelled pool, minus 0022 (no raw CT; duplicate scan of 0001) = 81,
    the same for every seed / K."""
    if ds == "flare":
        return list(Q.EVAL)
    if ds == "amos":
        lab = set(labelled("amos", seed))
        return [c for c in amos_ids() if c not in lab]
    pool = set(mact_pool())
    return [c for c in mact_ids() if c not in pool and c not in MACT_WITHDRAWN]


def draws(ds, seed=1337):
    """Labelled draws of a seed: K4 (the 4-scan set), K1 single-scan draws d1_<c>, K2 consecutive pairs d2_<a>_<b>
    (== fairfight rules), and for MACT also K8 (the whole pool)."""
    lab = labelled(ds, seed, 4)
    out = {"K4": list(lab)}
    for c in lab:
        out[f"d1_{c}"] = [c]
    for a, b in zip(lab[:-1], lab[1:]):
        out[f"d2_{a}_{b}"] = [a, b]
    if ds == "mact" and int(seed) == MACT_POOL_SEED:
        out["K8"] = mact_pool()
    return out


# ---------------------------------------------------------------------------------------------------- MACT sources
def mact_sources():
    """MACT id -> source record from Gibson's cropping.csv (Zenodo 1169361; sha256-pinned). Header typo
    'extent_rightextent_inf' = two columns extent_right, extent_inf."""
    assert sha256(MACT_CROPPING) == MACT_CROPPING_SHA, "cropping.csv changed"
    rows = [r for r in csv.reader(open(MACT_CROPPING))][1:]
    out = {}
    for r in rows:
        if not r:
            continue
        i, pub, dset, oid = int(r[0]), r[1], r[2], r[3]
        a, p, l, rt, inf, sup = map(int, r[4:10])
        src = "tcia" if pub == "TCIA" else "btcv"
        mid = f"{i:04d}"
        if src == "btcv":
            sub = "Training" if int(oid) <= 40 else "Testing"
            raw = BTCV_RAW_ROOT / sub / "img" / f"img{oid}.nii.gz"
            key = f"btcv_{oid}"
        else:
            sub = None
            raw = MACT_RAW_CT / "tcia" / f"PANCREAS_{oid}.nii.gz"
            key = f"tcia_{oid}"
        out[mid] = dict(mid=mid, publisher=pub, dataset=dset, source=src, original_id=oid, btcv_subset=sub,
                        raw_key=key, raw_ct=str(raw), crop_lps=dict(left=l, right=rt, ant=a, post=p, inf=inf, sup=sup),
                        withdrawn=MACT_WITHDRAWN.get(mid))
    assert len(out) == 90 and sorted(out) == mact_ids()
    return out


def btcv_overlap():
    S = mact_sources()
    rows = [dict(mact_id=m, btcv_id=s["original_id"], btcv_subset=s["btcv_subset"]) for m, s in S.items() if s["source"] == "btcv"]
    assert len(rows) == 47
    return rows


def tcia_series_uid(oid):
    d = json.load(open(MACT_TCIA_SERIES))
    m = [x for x in d if x["PatientID"] == f"PANCREAS_{oid}"]
    return m[0] if m else None


def mact_raw_ready(mid_or_src):
    s = mact_sources()[mid_or_src] if isinstance(mid_or_src, str) else mid_or_src
    return Path(s["raw_ct"]).is_file()


def mact_ts_raw(raw_key):
    return MACT_TS_RAW / f"{raw_key}.nii.gz"


def lps_view(img):
    """(array reoriented to LPS (x -> patient Left, y -> Posterior, z -> Superior), its affine, spacing)."""
    import nibabel as nib
    import numpy as np
    tr = nib.orientations.ornt_transform(nib.orientations.io_orientation(img.affine), nib.orientations.axcodes2ornt(("L", "P", "S")))
    arr = nib.orientations.apply_orientation(np.asarray(img.dataobj), tr)
    aff = img.affine @ nib.orientations.inv_ornt_aff(tr, img.shape[:3])
    sp = np.sqrt((aff[:3, :3] ** 2).sum(0))
    return arr, aff, sp, tr


def mact_frame(mid, img=None):
    """The MACT h5 frame of scan mid: crop start (LPS index), LPS spacing, output shape, LPS affine; asserts the shape
    equals the stored MACT_h5 shape (dataset shape only; no voxel of 'label' is read)."""
    import h5py
    import nibabel as nib
    import numpy as np
    s = mact_sources()[mid]
    img = img if img is not None else nib.load(s["raw_ct"])
    tr = nib.orientations.ornt_transform(nib.orientations.io_orientation(img.affine), nib.orientations.axcodes2ornt(("L", "P", "S")))
    aff = img.affine @ nib.orientations.inv_ornt_aff(tr, img.shape[:3])
    sp = np.sqrt((aff[:3, :3] ** 2).sum(0))
    c = s["crop_lps"]
    start = np.array([c["left"], c["ant"], c["inf"]], float)
    size = np.array([c["right"] - c["left"], c["post"] - c["ant"], c["sup"] - c["inf"]], float)
    shape = tuple(int(round(n * z)) for n, z in zip(size, sp))
    with h5py.File(MACT_H5 / f"{mid}.h5", "r") as f:
        h5shape = tuple(f["image"].shape)
    assert shape == h5shape, f"MACT {mid}: frame shape {shape} != MACT_h5 shape {h5shape}"
    return dict(start=start, size=size, spacing=sp, shape=shape, aff_lps=aff, ornt=tr, raw_shape=tuple(img.shape[:3]))


def mact_image_check(mid, img=None):
    """Reproduce the MACT_h5 'image' from the raw CT (no label is read): returns max |difference| (z-score units)."""
    import h5py
    import nibabel as nib
    import numpy as np
    from scipy import ndimage as ndi
    s = mact_sources()[mid]
    img = img if img is not None else nib.load(s["raw_ct"])
    fr = mact_frame(mid, img)
    arr = nib.orientations.apply_orientation(np.asarray(img.dataobj).astype(np.float64), fr["ornt"])
    st = fr["start"].astype(int)
    sz = fr["size"].astype(int)
    C = arr[st[0]:st[0] + sz[0], st[1]:st[1] + sz[1], st[2]:st[2] + sz[2]]
    Z = ndi.affine_transform(C, np.diag(1.0 / fr["spacing"]), offset=0.0, output_shape=fr["shape"], order=1,
                             mode="constant", cval=0.0)
    Z = np.clip(Z, -125.0, 275.0)
    Z = (Z - Z.mean()) / Z.std()
    with h5py.File(MACT_H5 / f"{mid}.h5", "r") as f:
        H = f["image"][:]
    return float(np.abs(Z - H).max()), Z, H


def mact_raw_to_h5(seg_raw, mid, img=None):
    """Raw-grid label map (any ids; shape = raw CT) -> MACT h5 frame by nearest neighbour on the frame's grid (cval 0)."""
    import nibabel as nib
    import numpy as np
    from scipy import ndimage as ndi
    s = mact_sources()[mid]
    img = img if img is not None else nib.load(s["raw_ct"])
    fr = mact_frame(mid, img)
    assert tuple(seg_raw.shape) == fr["raw_shape"], (mid, seg_raw.shape, fr["raw_shape"])
    arr = nib.orientations.apply_orientation(np.asarray(seg_raw).astype(np.uint8), fr["ornt"])
    out = ndi.affine_transform(arr, np.diag(1.0 / fr["spacing"]), offset=fr["start"], output_shape=fr["shape"], order=0,
                               mode="constant", cval=0, prefilter=False)
    return out.astype(np.uint8)


def ts_grid_positions(img_in_shape, img_rsp_shape):
    """Continuous canonical-raw voxel index sampled by TS's change_spacing (scipy zoom, grid_mode False = corner-aligned:
    x_in = o * (n_in - 1) / (n_out - 1)) for every output index o, per axis."""
    import numpy as np
    out = []
    for n_in, n_out in zip(img_in_shape, img_rsp_shape):
        f = (n_in - 1) / (n_out - 1) if n_out > 1 else 1.0
        out.append(np.arange(n_out, dtype=np.float64) * f)
    return out


def mact_ts_grid_coords(mid, img_orig, img_in_shape, img_in_affine, img_rsp_shape):
    """For every voxel of TS's 1.5 mm canonical grid: its continuous MACT-h5 index (the canonical raw position TS's own
    zoom sampled, corner-aligned, mapped through world coordinates into the h5 frame) and whether it lies inside the h5
    box. Returns (grids [3 arrays broadcast to the TS grid shape], valid bool, h5 shape)."""
    import numpy as np
    fr = mact_frame(mid, img_orig)
    canon = ts_grid_positions(img_in_shape[:3], img_rsp_shape[:3])
    # canonical index -> world -> LPS index -> h5 index; all maps are axis-aligned (permutation / flip / scale)
    M = np.linalg.inv(fr["aff_lps"]) @ img_in_affine                 # canonical index -> LPS index (4x4)
    A3, b3 = M[:3, :3], M[:3, 3]
    perm = np.argmax(np.abs(A3), axis=1)                             # LPS axis i draws from canonical axis perm[i]
    assert np.allclose(np.abs(A3)[np.arange(3), perm], np.abs(A3).sum(1), atol=1e-6), "non axis-aligned orientation"
    assert sorted(perm.tolist()) == [0, 1, 2]
    out_shape = tuple(len(c) for c in canon)
    shp = fr["shape"]
    tol = 1e-6
    valid = np.ones(out_shape, bool)
    grids = []
    for i in range(3):
        ax = perm[i]
        lps_i = A3[i, ax] * canon[ax] + b3[i]                       # LPS index along axis i (1-D along canonical axis ax)
        c = (lps_i - fr["start"][i]) * fr["spacing"][i]              # h5 index (h5 voxel j <-> LPS index start + j / spacing)
        sh = [1, 1, 1]
        sh[ax] = len(c)
        valid &= ((c >= -tol) & (c <= shp[i] - 1 + tol)).reshape(sh)
        grids.append(np.broadcast_to(c.astype(np.float32).reshape(sh), out_shape))
    return grids, valid, shp


def mact_h5_to_ts_grid(G_h5, mid, img_orig, img_in, img_rsp, n_labels=None):
    """MACT h5-frame label (0..8) -> TS's 1.5 mm canonical grid: the one-hot channels are linearly interpolated (cval 0)
    at the TS-grid voxels' h5 positions (mact_ts_grid_coords) -> argmax with strict '>' (lowest id wins ties, ==
    tsft_full_common.gt_to_ts_grid). Voxels outside the h5 box are UNKNOWN (valid False, label 0). Returns (G, valid)."""
    import numpy as np
    from scipy import ndimage as ndi
    n_labels = n_labels or (n_org("mact") + 1)
    grids, valid, shp = mact_ts_grid_coords(mid, img_orig, img_in.shape, img_in.affine, img_rsp.shape)
    H = np.asarray(G_h5)
    assert tuple(H.shape) == tuple(shp), (H.shape, shp)
    out_shape = valid.shape
    best = arg = None
    for k in range(n_labels):
        ch = (H == k).astype(np.float32)
        v = ndi.map_coordinates(ch, grids, order=1, mode="constant", cval=0.0, prefilter=False)
        if best is None:
            best, arg = v, np.zeros(out_shape, np.uint8)
        else:
            m = v > best
            best = np.where(m, v, best)
            arg[m] = k
    arg[~valid] = 0
    return arg, valid


# ---------------------------------------------------------------------------------------------------- AMOS / FLARE frames
def amos_frames():
    """AMOS h5 frames (crop, zoomed shape, h5 shape, raw shape) copied from the cor_feas build reports (no GT read);
    cached in outputs/results/vccor/xdata/frames/amos.json with the source report hashes."""
    p = FRAMES_DIR / "amos.json"
    if p.is_file():
        return json.load(open(p))["frames"]
    fr, src = {}, {}
    for c in amos_ids():
        rp = COR_FEAS_BUILD / f"amos_{c}.json"
        d = json.load(open(rp))
        assert d["label_replay_exact"] is True and d["dataset"] == "amos" and d["cid"] == c
        fr[c] = {"crop": d["crop"], "h5_shape": d["h5_shape"], "raw_shape": d["raw_shape"]}
        src[c] = sha256(rp)
    write_json(p, {"frames": fr, "source_reports_sha256": src, "source": rel(COR_FEAS_BUILD),
                   "note": "crop = GT-bbox crop (-10/+11) of prep_amos_ct.py recovered by the cor_feas build (label replay "
                           "proved bit-exact there); copied here so that no GT is read to map a prediction", "written": now()})
    return fr


def amos_raw_img(cid):
    for sub, d in (("train", "imagesTr"), ("valid", "imagesVa")):
        ip = AMOS_RAW / sub / d / f"amos_{cid}.nii.gz"
        if ip.is_file():
            return ip
    raise FileNotFoundError(f"raw AMOS case {cid} not found")


def amos_raw_lbl(cid):
    ip = amos_raw_img(cid)
    return Path(str(ip).replace("imagesTr", "labelsTr").replace("imagesVa", "labelsVa"))


def amos_raw_to_h5(seg_raw, cid, img=None):
    """Raw-grid label map -> AMOS h5 frame (== cor_feas amos_frame_map / prep_amos_ct: zoom spacing/1.5 order 0 'nearest',
    then the recorded crop)."""
    import nibabel as nib
    import numpy as np
    from scipy.ndimage import zoom
    fr = amos_frames()[cid]
    img = img if img is not None else nib.load(str(amos_raw_img(cid)))
    assert tuple(seg_raw.shape) == tuple(fr["raw_shape"]) == tuple(img.shape[:3]), (cid, seg_raw.shape, fr["raw_shape"])
    f = np.sqrt((img.affine[:3, :3] ** 2).sum(0)) / 1.5
    z = zoom(np.asarray(seg_raw).astype(np.uint8), f, order=0, mode="nearest")
    sl = tuple(slice(a, b) for a, b in fr["crop"])
    out = z[sl]
    assert list(out.shape) == list(fr["h5_shape"]), (cid, out.shape, fr["h5_shape"])
    return out.astype(np.uint8)


def flare_raw_to_h5(seg_raw, cid):
    sys.path.insert(0, str(REPO / "scripts/extras/ts_finetune"))
    import tsft_common as C
    return C.raw_to_h5(seg_raw, cid)


def raw_img(ds, cid):
    if ds == "flare":
        return Path(str(FLARE_RAW_IMG).format(cid=cid))
    if ds == "amos":
        return amos_raw_img(cid)
    s = mact_sources()[cid]
    assert not s["withdrawn"], f"MACT {cid}: {s['withdrawn']}"
    return Path(s["raw_ct"])


def raw_to_h5(ds, seg_raw, cid, img=None):
    """Raw-grid label map (dataset ids or TS ids) -> the dataset's h5 frame (nearest neighbour; voxelwise LUTs commute)."""
    if ds == "flare":
        return flare_raw_to_h5(seg_raw, cid)
    if ds == "amos":
        return amos_raw_to_h5(seg_raw, cid, img)
    return mact_raw_to_h5(seg_raw, cid, img)


def h5_shape(ds, cid):
    import h5py
    p = {"flare": FLARE_H5, "amos": AMOS_H5, "mact": MACT_H5}[ds] / f"{cid}.h5"
    with h5py.File(p, "r") as f:
        return tuple(f["image"].shape)


# ---------------------------------------------------------------------------------------------------- metrics
def dice(a, b):
    s = a.sum() + b.sum()
    return float("nan") if b.sum() == 0 else float(2 * (a & b).sum() / s)


def dice_vec(ds, Y, G):
    """Per organ Dice x 100; nan when the organ is absent from GT (present-only, == confirm_common.dice_vec for 13)."""
    return [dice(Y == o, G == o) * 100.0 for o in range(1, n_org(ds) + 1)]


def _bbox(m, pad):
    import numpy as np
    idx = np.argwhere(m)
    lo = np.maximum(idx.min(0) - pad, 0)
    hi = np.minimum(idx.max(0) + pad + 1, m.shape)
    return tuple(slice(lo[i], hi[i]) for i in range(3))


def nsd_hd95(Y, G, o, sp, tol=2.0):
    """== confirm_common.nsd_hd95 with the dataset's h5 spacing: NSD at tol mm (x100) and HD95 (mm). GT absent ->
    (nan, nan); prediction empty with GT present -> (0, nan)."""
    import numpy as np
    from scipy import ndimage
    a, b = Y == o, G == o
    if not b.any():
        return float("nan"), float("nan")
    if not a.any():
        return 0.0, float("nan")
    sl = _bbox(a | b, 2)
    a, b = a[sl], b[sl]
    sa = a & ~ndimage.binary_erosion(a)
    sb = b & ~ndimage.binary_erosion(b)
    da = ndimage.distance_transform_edt(~sa, sampling=sp)
    db = ndimage.distance_transform_edt(~sb, sampling=sp)
    d_ab, d_ba = db[sa], da[sb]
    nsd = float(((d_ab <= tol).sum() + (d_ba <= tol).sum()) / (len(d_ab) + len(d_ba)))
    hd = float(max(np.percentile(d_ab, 95), np.percentile(d_ba, 95)))
    return nsd * 100.0, hd


def score_scan(ds, maps, G, surface=True):
    """{arm: label map} vs GT in the dataset's h5 frame: per-organ Dice / NSD 2 mm / HD95 (present-only: nan when the
    organ is absent from GT), volumes. macro = nanmean over the organs of a scan (computed by the caller)."""
    import numpy as np
    from threadpoolctl import threadpool_limits
    sp = SPECS[ds]["h5_spacing"]
    k = n_org(ds)
    with threadpool_limits(1):
        rec = {"gt_vol": np.bincount(G.ravel(), minlength=256)[1:k + 1].astype(int).tolist(), "present": [],
               "dice": {}, "vol": {}, "nsd": {}, "hd95": {}}
        rec["present"] = [bool((G == o).any()) for o in range(1, k + 1)]
        for a, Y in maps.items():
            assert Y.shape == G.shape, (a, Y.shape, G.shape)
            rec["dice"][a] = dice_vec(ds, Y, G)
            rec["vol"][a] = np.bincount(Y.ravel(), minlength=256)[1:k + 1].astype(int).tolist()
            if surface:
                sh = [nsd_hd95(Y, G, o, sp) for o in range(1, k + 1)]
                rec["nsd"][a] = [s[0] for s in sh]
                rec["hd95"][a] = [s[1] for s in sh]
    return rec


# ---------------------------------------------------------------------------------------------------- nnU-Net naming
DS_BASE = {"flare": 400, "amos": 500, "mact": 600}       # own nn root; never collides with TS 291-297 or other blocks
ARMS = ("FULL", "AUDIT", "DICE", "RANDOM", "ANTI", "MIX2")
RECIPES = {
    "F1b": dict(trainable="all", lr=1e-4, label="= tsft_full F1b: all weights, lr 1e-4, 50 x 250 it, TS init incl. heads"),
    "F3": dict(trainable="norm", lr=1e-2, label="= tsft_full F3: InstanceNorm affines only (heads frozen), lr 1e-2"),
    "F1a": dict(trainable="all", lr=1e-3, label="= tsft_full F1a: all weights, lr 1e-3"),
    "F2": dict(trainable="decoder", lr=1e-3, label="= tsft_full F2: decoder only, lr 1e-3"),
}


def label_configs(ds):
    """Canonical ordered list of (seed, draw, arm) label configurations -> nnU-Net dataset ids (stable)."""
    out = []
    for s in SEEDS:
        dr = draws(ds, s)
        for arm in ARMS:
            out.append((s, "K4", arm))
        for d in dr:
            if d != "K4":
                out.append((s, d, "FULL"))
    return out


def label_cfg_name(seed, draw, arm):
    return f"s{seed}_{draw}_{arm}"


def dataset_id(ds, seed, draw, arm, tid):
    cfgs = label_configs(ds)
    k = cfgs.index((int(seed), draw, arm))
    did = DS_BASE[ds] + 2 * k + (0 if tid == 291 else 1)
    assert did < DS_BASE[ds] + 100, (ds, k, did)
    return did


def dataset_name(ds, seed, draw, arm, tid):
    return f"Dataset{dataset_id(ds, seed, draw, arm, tid)}_X{ds.upper()}{tid}_{label_cfg_name(seed, draw, arm)}"


def run_name(ds, seed, draw, arm, recipe):
    return f"{ds}_{label_cfg_name(seed, draw, arm)}_{recipe}"


def parse_run(run):
    ds, s, draw_and_rest = run.split("_", 2)
    parts = draw_and_rest.rsplit("_", 2)
    draw, arm, recipe = parts[0], parts[1], parts[2]
    return ds, int(s.lstrip("s")), draw, arm, recipe


def labels_dir(ds, seed, draw, arm, root=None):
    return Path(root or LABELS_ROOT) / ds / label_cfg_name(seed, draw, arm)


def case_name(ds, cid):
    return f"{SPECS[ds]['prefix']}_{cid}"


def cap_dir(ds, cid, prep_root=None):
    """Captures (1.5 mm TS input image + the five part predictions + merged raw output) of a labelled scan. flare seed-1337
    labelled 4: the frozen tsft_full prep (read-only); everything else: prep/<ds>/ts_cpu/<cid>/."""
    if ds == "flare" and cid in Q.LAB and prep_root is None:
        return Q.PREP / "ts_cpu" / cid
    return Path(prep_root or PREP) / ds / "ts_cpu" / cid


def capture_ok(d):
    d = Path(d)
    return (d / "capture.json").is_file() and (d / "image_1p5.nii.gz").is_file() and \
        all((d / f"part_{t}.nii.gz").is_file() for t in TS_PARTS)


def results_root(run, nn_root=None):
    return Path(nn_root or NN_ROOT) / f"results_{run}"


def model_dir(run, tid, models_root=None):
    return Path(models_root or MODELS) / run / f"part{tid}"


def model_done(run, models_root=None):
    return Path(models_root or MODELS) / run / "DONE.json"


def pred_dir(run, pred_root=None):
    return Path(pred_root or PRED) / run


def stored_ts_raw(ds, cid):
    """The stored TS-full raw output (TS 'total' ids, raw grid) of a scan, or None."""
    if ds == "flare":
        p = DATA / "ts_fullres/raw_full" / f"{cid}.nii.gz"
    elif ds == "amos":
        p = COR_FEAS_TS_RAW / "amos" / f"{cid}.nii.gz"
    else:
        s = mact_sources()[cid]
        p = mact_ts_raw(s["raw_key"])
    return p if p.is_file() else None


def ts_label_set_dir(ds):
    return {"flare": FLARE_PSEUDO_FULLRES, "amos": AMOS_PSEUDO_FULLRES, "mact": MACT_PSEUDO}[ds]


# ---------------------------------------------------------------------------------------------------- E1 (2026-10-17)
# E1 = the descriptive TS-full few-shot fine-tune baseline look on MACT and AMOS (2026-10-17):
#   MACT: F0, FULL K=4 (F1b recipe), NORM K=4 (F3 recipe), FULL K=1 on each K=4 scan;  AMOS: F0, FULL K=4, NORM K=4.
# Pre-registered in outputs/results/vccor/xdata/prereg_e1_<ds>.json (sha256 + CODE lines in outputs/logs/vccor/xdata_prereg.log).
PREREG_LOG = LOGD / "xdata_prereg.log"
PREREG_E1 = {"mact": OUT / "prereg_e1_mact.json", "amos": OUT / "prereg_e1_amos.json"}
E1_DATASETS = ("amos", "mact")


def e1_arms(ds):
    """E1 arm label -> source ('TS' = the CPU TS-full label set; else a predict_xdata.py run name), in reporting order."""
    arms = {"TS_CPU": "TS", "F0": f"{ds}_F0", "FULL_K4": run_name(ds, 1337, "K4", "FULL", "F1b"),
            "NORM_K4": run_name(ds, 1337, "K4", "FULL", "F3")}
    if ds == "mact":
        for c in labelled("mact", 1337, 4):
            arms[f"FULL_K1@{c}"] = run_name("mact", 1337, f"d1_{c}", "FULL", "F1b")
    return arms


def e1_runs(ds):
    """The fine-tuned E1 runs of a dataset (training + prediction jobs)."""
    return [s for a, s in e1_arms(ds).items() if a not in ("TS_CPU", "F0")]


def e1_draws(ds):
    return ["K4"] + ([f"d1_{c}" for c in labelled("mact", 1337, 4)] if ds == "mact" else [])


def e1_consumers(run):
    """E1 runs that train on the same nnU-Net datasets (same ds / seed / draw / label arm) as run."""
    ds, s, d, a, _ = parse_run(run)
    return [r for r in e1_runs(ds) if parse_run(r)[:4] == (ds, s, d, a)]


def prereg_sha_logged(path):
    """sha256 logged for a prereg file ('PREREG sha256 <hex> <repo-relative path>' lines; the last entry wins)."""
    h, want = None, rel(path)
    if PREREG_LOG.is_file():
        for line in open(PREREG_LOG):
            p = line.split()
            if len(p) >= 6 and p[2] == "PREREG" and p[3] == "sha256" and " ".join(p[5:]) == want:
                h = p[4]
    return h


def frozen_code_hashes():
    """{repo-relative path: sha256} from the CODE / IMPORT lines of xdata_prereg.log (the last entry per path wins)."""
    out = {}
    if PREREG_LOG.is_file():
        for line in open(PREREG_LOG):
            p = line.split()
            if len(p) >= 5 and p[2] in ("CODE", "IMPORT") and len(p[3]) == 64:
                out[" ".join(p[4:])] = p[3]
    return out


def check_code_frozen(paths, log=print, strict=True):
    """Refuse (strict) / warn unless every code file hashes to its logged CODE / IMPORT value."""
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


def dir_bytes(p):
    p = Path(p)
    if not p.exists():
        return 0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and not f.is_symlink())
