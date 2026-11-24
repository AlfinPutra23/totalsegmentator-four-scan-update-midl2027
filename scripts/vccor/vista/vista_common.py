"""vccor/vista: VISTA3D (MONAI bundle 0.5.8, models/vista3d; network vista3d132, MONAI 1.6.0 in .venv) as an
'ALREADY-SEEN PROTOCOL' CONTROL for the TS-full few-shot adaptation findings (prereg v2, 2026-10-19; v1 = a replication
design, superseded before any VISTA3D prediction on an evaluation scan after the contamination check showed that VISTA3D
was trained on FLARE22 / AMOS22 / BTCV / MACT with their own labels). Trimmed v2 scope: FLARE + AMOS, seed 1337 only.
v3 (2026-10-19): every VF run trains 5,000 iterations x batch 1 = 5,000 patches, budget-matched to the TS EP10 arm of e2;
C2's primary contrast uses TS EP10. Pre-registered in outputs/results/vccor/vista/prereg_vista.json (v3; v1 / v2 archived as
prereg_vista.v1.json / .v2.json; sha256 + AMENDMENT + CODEFREEZE in outputs/logs/vccor/vista_prereg.log). PREPARED, NOT
LAUNCHED (patch written, not applied).

Arms (all through the SAME automatic (class-prompt) inference path: the bundle's inference.json preprocessing, the bundle's
Vista3dInferer, prompts = the bundle's 117 'everything_labels', VistaPostTransformd, Invertd to the raw grid; VISTA ids ->
dataset ids BY NAME (kidney cysts -> same-side kidney, as the project's v2 TS LUT); raw -> the dataset's h5 frame with the
project's frozen frame functions (xdata_common.raw_to_h5)):
  V0   VISTA3D's own weights (zero-shot automatic branch)
  VF   automatic-branch fine-tune on the seed-1337 K = 4 labelled scans (FLARE: lr 5e-5 and 5e-6, dev18 selection; AMOS)
  (v1's VS / VK1 / seeds 2025 + 3407 / MACT arms are DROPPED in v2; the MACT helpers below stay as unused library code)
Training labels (raw CT grid, VISTA global ids; == the TS rule tsft_full_common.composite_target with one model that owns
every organ): dataset GT for the dataset's organs (V0's own class kept where V0 already calls that organ, e.g. a kidney
cyst); V0's own prediction for every class the dataset does not annotate where GT is background; 0 where V0 predicts a
dataset organ on GT background; MACT outside Gibson's box (GT unknown): V0's own prediction.

This module has NO import-time side effects beyond sys.path (safe before torch / monai). Everything under
scripts/vccor/{tsft_full,audit_ft,xdata,fairfight,ssft}, scripts/cor_*, models/vista3d and outputs/results/vccor/<other
blocks> is imported / read ONLY.
"""
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path("/home/user/Documents/Medical Imagining SSL")
HERE = Path(__file__).resolve().parent
XDATA_DIR = REPO / "scripts/vccor/xdata"
AFT_DIR = REPO / "scripts/vccor/audit_ft"
for _d in (XDATA_DIR, AFT_DIR):
    if str(_d) not in sys.path:
        sys.path.append(str(_d))                         # appended: those dirs also hold smoke_test.py / write_prereg.py
import xdata_common as X  # noqa: E402  (frozen: frames, label sets, splits, metrics)
Q = X.Q                                                  # tsft_full_common (frozen: FLARE sets, Log, sha256, write_json)

# ---------------------------------------------------------------------------------------------------- layout (ours)
OUT = REPO / "outputs/results/vccor/vista"
LOGD = REPO / "outputs/logs/vccor"
PREREG = OUT / "prereg_vista.json"                      # v3 (the current pre-registration)
PREREG_V1 = OUT / "prereg_vista.v1.json"                # v1, superseded (archived byte-for-byte)
PREREG_V1_SHA = "e4eae7da919147543ae17561b137887ecebab210f3c5b9e61330fcacd4cca52a"
PREREG_V2 = OUT / "prereg_vista.v2.json"                # v2, superseded by v3 (archived byte-for-byte)
PREREG_V2_SHA = "abf86a3be2d9c0a9f0726d479af1c60227ac9d601f4883a4baf662a8013f24f2"
PREREG_LOG = LOGD / "vista_prereg.log"
CONTAM = OUT / "contamination"                           # vista_repo/ (datalists, configs), paper/ (text), exposure.json
EXPOSURE = CONTAM / "exposure.json"
LABELS = OUT / "labels"                                  # labels/<ds>/<cid>.nii.gz (composite, raw grid) + manifest_<ds>.json
MODELS = OUT / "models"                                  # models/<run>/{weights_trainable.pt, export.json, train_log.jsonl}
PRED = OUT / "pred"                                      # pred/<run>/{raw,h5}/<cid> + DONE.json
PROBE = OUT / "probe" / "probe.json"
SELECT = OUT / "selection"                               # lr_selection.json, maps.json, GT_OPENED_dev18
RES = OUT / "results"                                    # results/<look>/{maps.json, GT_OPENED, pervol/, results.json, tables.md}
SMOKE = OUT / "smoke"

BUNDLE = REPO / "models/vista3d"
MODEL_PT = BUNDLE / "models/model.pt"
BUNDLE_SHA = {                                           # pinned at preparation (2026-10-19); checked by every GPU job
    "models/model.pt": "c92bab26d00b4a5d89fa8a383900cdeb88302fd318e5e816df0bbec7106d9a1b",
    "configs/train.json": "3f036e92c7392188b533273c7f7b6f64e77f6462d08261194c3e91d8d0934284",
    "configs/train_continual.json": "ca97ef6422a05340d75933b567cbfcf93a2ef4d1c791f900292a2bbe5b74a3f3",
    "configs/inference.json": "b1f7f98012db40a9069f7d79e97ad80320fa7bfd8247b38747836e39baf77e59",
    "configs/metadata.json": "e857d9fad731d31b1d905192a53812061b17fdfac47981310ec1595e8b1b306b",
    "scripts/inferer.py": "2f0ed6176a24dbb51515096224f47ff85d1808ee492c5c336f129d39e0e2ab88",
    "docs/labels.json": "186226ba214b3e02cc5394427c6395364905129e449a6b726253269117302475",
}
BUNDLE_VERSION = "0.5.8"                                 # configs/metadata.json; HF ref MONAI/vista3d 0.5.8 -> f2ae8ec
VENV = REPO / ".venv/bin/python"

# ---------------------------------------------------------------------------------------------------- VISTA classes
# the bundle's inference.json 'everything_labels' (asserted against the parsed config by the engine)
EXCLUDED_FROM_EVERYTHING = (2, 16, 18, 20, 21, 23, 24, 25, 26, 27, 128, 129, 130, 131, 132)
EVERYTHING = sorted(set(range(1, 133)) - set(EXCLUDED_FROM_EVERYTHING))
assert len(EVERYTHING) == 117
NAN_VALUE = 255                                          # the bundle's nan_to_num marker (point windows only)

# dataset organ (h5 id order) -> VISTA class name; extra VISTA classes folded into a dataset organ (cyst rule)
DS_VISTA = {
    "flare": dict(organs=["liver", "right kidney", "spleen", "pancreas", "aorta", "inferior vena cava", "right adrenal gland",
                          "left adrenal gland", "gallbladder", "esophagus", "stomach", "duodenum", "left kidney"],
                  extra={"left kidney cyst": 13, "right kidney cyst": 2}),
    "amos": dict(organs=["liver", "right kidney", "spleen", "pancreas", "aorta", "inferior vena cava", "right adrenal gland",
                         "left adrenal gland", "gallbladder", "esophagus", "stomach", "duodenum", "left kidney"],
                 extra={"left kidney cyst": 13, "right kidney cyst": 2}),
    "mact": dict(organs=["spleen", "left kidney", "gallbladder", "esophagus", "liver", "stomach", "pancreas", "duodenum"],
                 extra={"left kidney cyst": 2}),
}
# TotalSegmentator class name (xdata SPECS ts_names / extra) -> VISTA class name: the cross-check that both models are
# mapped onto the same anatomical label space
TS_TO_VISTA_NAME = {"liver": "liver", "kidney_right": "right kidney", "spleen": "spleen", "pancreas": "pancreas",
                    "aorta": "aorta", "inferior_vena_cava": "inferior vena cava", "adrenal_gland_right": "right adrenal gland",
                    "adrenal_gland_left": "left adrenal gland", "gallbladder": "gallbladder", "esophagus": "esophagus",
                    "stomach": "stomach", "duodenum": "duodenum", "kidney_left": "left kidney",
                    "kidney_cyst_left": "left kidney cyst", "kidney_cyst_right": "right kidney cyst"}
DATASETS = ("flare", "amos", "mact")
PFX = {"flare": "FLARE", "amos": "AMOS", "mact": "MACT"}


def vista_label_dict():
    """VISTA class name -> id (bundle docs/labels.json == VISTA repo data/jsons/label_dict.json, same sha256)."""
    d = json.load(open(BUNDLE / "docs/labels.json"))
    return {k: int(v) for k, v in d.items()}


def vista_channel_def():
    """VISTA id -> class name (bundle configs/metadata.json channel_def; 0 = background, 1..132)."""
    m = json.load(open(BUNDLE / "configs/metadata.json"))
    return {int(k): v for k, v in m["network_data_format"]["outputs"]["pred"]["channel_def"].items()}


def organ_vista_ids(ds):
    """[VISTA id of dataset organ 1, ..., organ n] (canonical class of every dataset organ)."""
    ld = vista_label_dict()
    return [ld[n] for n in DS_VISTA[ds]["organs"]]


def vista_lut(ds):
    """uint8 LUT VISTA id -> dataset id (canonical organ classes + the dataset's cyst rule; NaN marker 255 -> 0)."""
    import numpy as np
    ld = vista_label_dict()
    lut = np.zeros(256, np.uint8)
    for o, n in enumerate(DS_VISTA[ds]["organs"], start=1):
        lut[ld[n]] = o
    for n, o in DS_VISTA[ds]["extra"].items():
        lut[ld[n]] = o
    lut[NAN_VALUE] = 0
    return lut


def canon_ids(ds):
    """uint8 array dataset id -> canonical VISTA id (0 -> 0)."""
    import numpy as np
    c = np.zeros(X.n_org(ds) + 1, np.uint8)
    for o, v in enumerate(organ_vista_ids(ds), start=1):
        c[o] = v
    return c


def check_luts():
    """Cross-checks: names exist in VISTA's label dict and in everything_labels, the dataset organ order equals the
    project's label spaces (xdata SPECS, via the TS class names), labels.json == metadata channel_def, the cyst rule
    equals the TS one, and the LUT is a bijection on the canonical classes."""
    ld, cd = vista_label_dict(), vista_channel_def()
    assert all(cd[v] == k for k, v in ld.items()), "docs/labels.json != metadata channel_def"
    for ds in DATASETS:
        sp = X.SPECS[ds]
        assert [TS_TO_VISTA_NAME[n] for n in sp["ts_names"]] == DS_VISTA[ds]["organs"], ds
        assert {TS_TO_VISTA_NAME[n]: o for n, o in sp["extra"].items()} == DS_VISTA[ds]["extra"], ds
        ids = organ_vista_ids(ds)
        assert len(set(ids)) == len(ids) == X.n_org(ds) and all(i in EVERYTHING for i in ids), ds
        lut = vista_lut(ds)
        assert all(lut[v] == o for o, v in enumerate(ids, start=1))
        assert int((lut > 0).sum()) == len(ids) + len(DS_VISTA[ds]["extra"]), ds
        assert all(ld[n] in EVERYTHING for n in DS_VISTA[ds]["extra"]), ds
    m = vista_lut("mact")
    assert m[ld["right kidney"]] == 0 and m[ld["right kidney cyst"]] == 0 and m[ld["left kidney cyst"]] == 2
    return True


# ---------------------------------------------------------------------------------------------------- the design
SEEDS = (1337, 2025, 3407)
DEV18, HELD18, TEST10 = list(Q.DEV18), list(Q.HELD18), list(Q.TEST10)
FLARE_ALL = sorted(set(Q.LAB) | set(Q.EVAL))             # all 50 FLARE scans
LR_A, LR_B = 5e-5, 5e-6                                   # bundle train_continual.json learning_rate, and 0.1 x
N_ITER = 5000                                            # v3: 5,000 it x batch 1 = 5,000 patches of 128^3 = TS EP10's
                                                         # budget (10 epochs x 250 it x batch 2 per part; e2 flare_K4_EP10)
N_ITER_V1V2 = 3200                                       # v1 / v2: the bundle README run (320 epochs x 10 scans x 1 patch)
MAX_FOREPROMPT, MAX_BACKPROMPT = 32, 4                   # VISTA's own automatic-branch stages 3/4 (36 prompts)
TRAIN_SEED, INFER_SEED = 0, 123                          # bundle train.json / inference.json set_determinism seeds
KEEP_WEIGHTS = ("flare_s1337_K4_lrA", "flare_s1337_K4_lrB")   # R1 exports kept; others deleted after their predictions
PROBE_RULE = dict(max_mem_gb=22.0, max_train_hours=3.5, warmup=5, timed=30)
K1_DRAWS = ()                                              # v1 only (dropped in v2)


def flare_lab(seed):
    lab = X.labelled("flare", seed, 4)
    import audit_ft_common as A
    assert lab == A.LAB_EXPECTED[seed], (seed, lab)
    return lab


def labelled_of_run(run):
    """Training scans (labelled, GT-supervised) of a fine-tuned run."""
    r = RUNS[run]
    return list(r["lab"])


def _runs():
    """v2 (control, trimmed): FLARE seed 1337 at both learning rates + AMOS seed 1337 at the selected one."""
    R = {}
    lab = X.labelled("flare", 1337, 4)
    R["flare_s1337_K4_lrA"] = dict(ds="flare", seed=1337, lab=lab, pool=[], lr="A", ids="dev18+held18+test10",
                                   what="VF seed 1337, lr 5e-5 (bundle default)")
    R["flare_s1337_K4_lrB"] = dict(ds="flare", seed=1337, lab=lab, pool=[], lr="B", ids="dev18+held18+test10",
                                   what="VF seed 1337, lr 5e-6 (0.1 x bundle default)")
    R["amos_s1337_K4_sel"] = dict(ds="amos", seed=1337, lab=X.labelled("amos", 1337, 4), pool=[], lr="sel", ids="eval",
                                  what="VF AMOS K=4 (E1 labelled scans), selected lr")
    return R


RUNS = _runs()
V0_RUNS = {"V0_flare": "flare", "V0_amos": "amos"}
TRAIN_ORDER = ["flare_s1337_K4_lrA", "flare_s1337_K4_lrB", "amos_s1337_K4_sel"]
assert sorted(TRAIN_ORDER) == sorted(RUNS)
DROPPED_V1 = {"runs": ["flare_s2025_K4_sel", "flare_s3407_K4_sel", "flare_s1337_VS_sel", "flare_s1337_d1_0014_sel",
                       "flare_s1337_d1_0001_sel", "mact_s1337_K4_sel"], "v0_runs": ["V0_mact"],
              "hypotheses": ["H1", "H1_MACT", "H2"], "looks": ["mact"]}
FLARE_LAB4 = list(X.labelled("flare", 1337, 4))         # C1's FLARE scans (== E0's FLARE_LAB4)


def ds_of(run):
    return V0_RUNS[run] if run in V0_RUNS else RUNS[run]["ds"]


def v0_ids(ds):
    """Scans V0 predicts: every evaluation scan + every labelled scan whose training label needs V0's own prediction."""
    if ds == "flare":
        return list(FLARE_ALL)
    lab = X.labelled(ds, 1337, 4)
    return sorted(set(X.eval_ids(ds, 1337)) | set(lab))


def run_ids(run):
    if run in V0_RUNS:
        return v0_ids(V0_RUNS[run])
    r = RUNS[run]
    if r["ids"] == "eval":
        return list(X.eval_ids(r["ds"], 1337))
    sets = {"dev18": DEV18, "held18": HELD18, "test10": TEST10}
    out = []
    for s in r["ids"].split("+"):
        out += sets[s]
    return out


def train_scans(run):
    """(labelled scans with composite labels, pool scans with V0 pseudo-labels)."""
    r = RUNS[run]
    return list(r["lab"]), list(r["pool"])


def label_ids(ds):
    """Every scan that gets a composite (GT-supervised) training label in this block."""
    s = set()
    for r in RUNS.values():
        if r["ds"] == ds:
            s |= set(r["lab"])
    return sorted(s)


# ---------------------------------------------------------------------------------------------------- paths
def raw_img(ds, cid):
    return X.raw_img(ds, cid)


def label_path(ds, cid):
    return LABELS / ds / f"{PFX[ds]}_{cid}.nii.gz"


def model_dir(run):
    return MODELS / run


def weights_path(run):
    return MODELS / run / "weights_trainable.pt"


def model_done(run):
    return MODELS / run / "DONE.json"


def pred_dir(run):
    return PRED / run


def pred_done(run):
    return PRED / run / "DONE.json"


def v0_raw(ds, cid):
    return PRED / f"V0_{ds}" / "raw" / f"{cid}.nii.gz"


# ---------------------------------------------------------------------------------------------------- label composition
def composite(ds, G, P, valid=None):
    """Training label (VISTA ids) on the raw grid. G: dataset GT (0..n); P: V0's own prediction (VISTA ids).
      GT organ o, V0 already calls o (lut[P] == o)  -> P   (V0's own class of that organ, e.g. a kidney cyst)
      GT organ o, otherwise                         -> canonical VISTA id of o
      GT background, P a non-dataset class          -> P   (V0's knowledge of every class the dataset does not annotate)
      GT background, P a dataset class              -> 0   (a V0 false positive under the dataset's protocol)
      valid == False (GT unknown: MACT outside Gibson's box) -> P
    == tsft_full_common.composite_target with one model owning every organ. Returns (Y uint8, counts)."""
    import numpy as np
    lut, canon = vista_lut(ds), canon_ids(ds)
    G = np.asarray(G).astype(np.uint8)
    P = np.asarray(P).astype(np.uint8)
    assert G.shape == P.shape and int(G.max()) <= X.n_org(ds), (G.shape, P.shape, int(G.max()))
    assert not (P == NAN_VALUE).any(), "V0 prediction holds the NaN marker 255"
    flp = lut[P]
    Y = np.zeros(P.shape, np.uint8)
    bg = G == 0
    keep_non = bg & (P > 0) & (flp == 0)
    Y[keep_non] = P[keep_non]
    org = G > 0
    same = org & (flp == G)
    Y[same] = P[same]
    relab = org & ~same
    Y[relab] = canon[G[relab]]
    cnt = {"n_vox": int(G.size), "gt_organ_vox": int(org.sum()), "gt_organ_relabelled": int(relab.sum()),
           "gt_organ_relabelled_from_nonds": int((relab & (P > 0) & (flp == 0)).sum()),
           "v0_ds_fp_removed": int((bg & (flp > 0)).sum()), "nonds_kept": int(keep_non.sum()),
           "v0_own_organ_class_kept_not_canonical": int((same & (P != canon[G])).sum())}
    if valid is not None:
        valid = np.asarray(valid, bool)
        assert valid.shape == G.shape and not (G[~valid] > 0).any(), "GT must be 0 where it is unknown"
        inv = ~valid
        cnt["unknown_vox"] = int(inv.sum())
        cnt["unknown_kept_v0"] = int((inv & (P > 0)).sum())
        cnt["v0_ds_fp_removed"] = int((bg & valid & (flp > 0)).sum())
        cnt["nonds_kept"] = int((keep_non & valid).sum())
        Y[inv] = P[inv]
    cnt["unchanged_vs_v0"] = int((Y == P).sum())
    cnt["classes"] = sorted(int(c) for c in np.unique(Y) if c > 0)
    return Y, cnt


def identity_check(ds, G, P, valid=None):
    """Per dataset organ k present in G: the dataset class j maximising Dice(G == k, lut[P] == j) must be k (the
    amendment-A1 rule). PASS/FAIL only (no overlap value is returned)."""
    import numpy as np
    n = X.n_org(ds)
    D = vista_lut(ds)[np.asarray(P).astype(np.uint8)]
    m_all = np.ones(D.shape, bool) if valid is None else np.asarray(valid, bool)
    vol = np.bincount(D[m_all], minlength=n + 1)[1:n + 1]
    res = {}
    for k in range(1, n + 1):
        m = (G == k) & m_all
        if not m.any():
            continue
        inter = np.bincount(D[m], minlength=n + 1)[1:n + 1]
        dice = 2 * inter / (m.sum() + vol)
        res[X.names(ds)[k - 1]] = bool(dice.argmax() + 1 == k and dice.max() > 0)
    return res


# ---------------------------------------------------------------------------------------------------- GT on the raw grid
def read_train_gt(ds, cid):
    """The ONLY GT reader for training labels: labelled scans of this block's runs only. Returns (GT on the raw CT grid
    in the dataset label space, valid mask or None, raw image nifti)."""
    import h5py
    import nibabel as nib
    import numpy as np
    assert cid in label_ids(ds), f"{ds} {cid} is not a labelled (training) scan of this block"
    img = nib.load(str(raw_img(ds, cid)))
    if ds == "flare":
        lbl = nib.load(str(X.FLARE_RAW_LBL).format(cid=cid))
        lab = np.asarray(lbl.dataobj).astype(np.uint8)
        assert lab.max() <= 13 and lab.shape == img.shape[:3] and np.allclose(lbl.affine, img.affine, atol=1e-3), cid
        return lab, None, img
    if ds == "amos":
        lbl = nib.load(str(X.amos_raw_lbl(cid)))
        raw = np.asarray(lbl.dataobj)
        assert raw.min() >= 0 and raw.max() <= 15 and raw.shape == img.shape[:3], (cid, raw.min(), raw.max())
        assert np.allclose(lbl.affine, img.affine, atol=1e-3), cid
        lab = np.zeros(raw.shape, np.uint8)
        for a, f in X.AMOS_TO_FLARE.items():
            lab[raw == a] = f                            # AMOS 14 bladder / 15 prostate-uterus -> 0 (prep_amos_ct.py)
        return lab, None, img
    with h5py.File(X.MACT_H5 / f"{cid}.h5", "r") as f:
        g = f["label"][:]
    gi = np.rint(g)
    assert np.array_equal(gi, g) and gi.min() >= 0 and gi.max() <= 8, f"MACT {cid}: label not in 0..8"
    G, valid = mact_h5_to_raw(gi.astype(np.uint8), cid, img)
    return G, valid, img


def mact_h5_to_raw(G_h5, mid, img, n_labels=None, tol=1e-6):
    """MACT h5-frame label (0..8) -> the RAW CT grid: every raw voxel's h5 position (inverse of xdata_common.
    mact_raw_to_h5: LPS index p <-> h5 index (p - start) * spacing, axis-aligned) samples the one-hot channels with linear
    interpolation (cval 0) -> argmax with strict '>' (lowest id wins ties, == mact_h5_to_ts_grid). Raw voxels outside the
    h5 box are UNKNOWN (valid False, label 0). Returns (G_raw uint8, valid bool), both in the raw array orientation."""
    import nibabel as nib
    import numpy as np
    from scipy import ndimage as ndi
    n_labels = n_labels or (X.n_org("mact") + 1)
    fr = X.mact_frame(mid, img)
    H = np.asarray(G_h5)
    assert tuple(H.shape) == tuple(fr["shape"]), (mid, H.shape, fr["shape"])
    tr = fr["ornt"]
    lps_shape = nib.orientations.apply_orientation(np.zeros(fr["raw_shape"], np.uint8), tr).shape
    axes = []
    for i in range(3):
        c = (np.arange(lps_shape[i], dtype=np.float64) - fr["start"][i]) * fr["spacing"][i]
        ok = (c >= -tol) & (c <= fr["shape"][i] - 1 + tol)
        idx = np.nonzero(ok)[0]
        axes.append((c, idx))
    G_lps = np.zeros(lps_shape, np.uint8)
    valid_lps = np.zeros(lps_shape, bool)
    if all(len(ix) for _, ix in axes):
        lo = [int(ix[0]) for _, ix in axes]
        hi = [int(ix[-1]) + 1 for _, ix in axes]
        assert all(len(ix) == h - l for (_, ix), l, h in zip(axes, lo, hi)), "the valid range must be contiguous"
        cs = [np.clip(c[l:h], 0, s - 1).astype(np.float32) for (c, _), l, h, s in zip(axes, lo, hi, fr["shape"])]
        sub = tuple(h - l for l, h in zip(lo, hi))
        grids = [np.broadcast_to(cs[i].reshape([-1 if j == i else 1 for j in range(3)]), sub) for i in range(3)]
        best = arg = None
        for k in range(n_labels):
            v = ndi.map_coordinates((H == k).astype(np.float32), grids, order=1, mode="constant", cval=0.0, prefilter=False)
            if best is None:
                best, arg = v, np.zeros(sub, np.uint8)
            else:
                m = v > best
                best = np.where(m, v, best)
                arg[m] = k
        box = tuple(slice(l, h) for l, h in zip(lo, hi))
        G_lps[box] = arg
        valid_lps[box] = True
    inv = nib.orientations.ornt_transform(nib.orientations.axcodes2ornt(("L", "P", "S")),
                                          nib.orientations.io_orientation(img.affine))
    G_raw = np.ascontiguousarray(nib.orientations.apply_orientation(G_lps, inv)).astype(np.uint8)
    valid_raw = np.ascontiguousarray(nib.orientations.apply_orientation(valid_lps, inv)).astype(bool)
    assert G_raw.shape == tuple(fr["raw_shape"]), (mid, G_raw.shape, fr["raw_shape"])
    return G_raw, valid_raw


def to_h5(ds, seg_raw_vista, cid, img=None):
    """VISTA-id raw-grid map -> (dataset-id map in the h5 frame, VISTA-id map in the h5 frame) with the frozen frame
    functions (nearest neighbour; voxelwise LUTs commute). FLARE frames come from the cached h5_frames (no GT is read)."""
    import numpy as np
    if ds == "flare":
        sys.path.insert(0, str(REPO / "scripts/extras/ts_finetune"))
        import tsft_common as C
        assert (C.FRAME_CACHE / f"{cid}.json").is_file(), f"FLARE frame cache missing for {cid} (would read GT)"
    seg = np.asarray(seg_raw_vista).astype(np.uint8)
    raw_h5 = X.raw_to_h5(ds, seg, cid, img)
    return vista_lut(ds)[raw_h5], raw_h5


# ---------------------------------------------------------------------------------------------------- exposure
def exposure():
    """Per-scan VISTA3D training exposure from the official datalists (outputs/results/vccor/vista/contamination/
    exposure.json, written by write_prereg_vista.py and re-derived by the smoke test)."""
    return json.load(open(EXPOSURE))


# ---------------------------------------------------------------------------------------------------- helpers
sha256, rel, write_json, Log, be_nice, cap_threads = Q.sha256, Q.rel, Q.write_json, Q.Log, Q.be_nice, Q.cap_threads


def now():
    return time.strftime("%F %T")


def arr_sha(a):
    import numpy as np
    a = np.ascontiguousarray(a)
    return hashlib.sha256(f"{a.dtype}|{a.shape}|".encode() + a.tobytes()).hexdigest()


def check_bundle(log=print):
    bad = {k: v[:12] for k, v in BUNDLE_SHA.items() if sha256(BUNDLE / k) != v}
    if bad:
        raise SystemExit(f"[REFUSE] VISTA3D bundle files changed since preparation: {bad}")
    log(f"bundle {BUNDLE_VERSION} verified ({len(BUNDLE_SHA)} files incl. model.pt)")
    return True


def save_nii(arr, affine, path):
    import nibabel as nib
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name.replace('.nii.gz', '')}.tmp{os.getpid()}.nii.gz")
    img = nib.Nifti1Image(np.asarray(arr).astype(np.uint8), affine)
    img.set_data_dtype(np.uint8)
    nib.save(img, str(tmp))
    os.replace(tmp, path)
    return sha256(path)


def epochs_equiv(n_scans, n_iter=N_ITER):
    """The bundle's epoch count that ClassesToIndicesd's max_samples_per_class (10 x epochs) is computed from."""
    return int(math.ceil(n_iter / max(1, n_scans)))


# ---------------------------------------------------------------------------------------------------- integrity
def prereg_sha_logged():
    h = None
    if PREREG_LOG.is_file():
        for line in open(PREREG_LOG):
            p = line.split()
            if len(p) >= 6 and p[2] == "PREREG" and p[3] == "sha256" and " ".join(p[5:]) == rel(PREREG):
                h = p[4]
    return h


def frozen_code_hashes():
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
        msg = f"code differs from the CODEFREEZE hashes in {rel(PREREG_LOG)}: {bad}"
        if strict:
            raise SystemExit("[REFUSE] " + msg)
        log("[warn] " + msg)
    return not bad


def check_prereg():
    h = sha256(PREREG)
    if h != prereg_sha_logged():
        raise SystemExit(f"[REFUSE] {rel(PREREG)} sha256 {h[:16]} != logged {str(prereg_sha_logged())[:16]} ({rel(PREREG_LOG)})")
    return json.load(open(PREREG))


def code_files():
    return sorted(HERE.glob("*.py"))


def preflight_real(script, log=print):
    """Every real (non-smoke) job: prereg hash == logged, this block's code == CODEFREEZE, bundle unchanged."""
    check_prereg()
    check_code_frozen([HERE / "vista_common.py", HERE / "vista_engine.py", Path(script)], log)
    check_bundle(log)
