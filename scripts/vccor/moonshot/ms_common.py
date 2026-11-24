"""vccor/moonshot common module (2026-11-03): paths, design constants, data-discipline guards, IO helpers.

Moonshot = "convention head + atlas prior + anchoring" on FROZEN TotalSegmentator-full (TS-full, 1.5 mm, all five parts).
Kill test (pre-registered in outputs/results/vccor/moonshot/prereg.json; hash logged in logs/moonshot_prereg.log):
  K=1 draws = the fair-fight draw rule (seed-1337 labelled four, each scan once): d1_0014, d1_0001, d1_0006, d1_0038.
  Evaluation set = FLARE dev18 ONLY (design data of the study). Comparators on the same 18 scans: zero-shot F0 (TS-full
  weights through the GPU path, tsft_full/pred/F0) and FT-full K=1 (F1b recipe, lr 1e-4, the frozen fair-fight
  train/predict code re-run through ff_launch.py, outputs under outputs/results/vccor/moonshot/ff/).

DATA DISCIPLINE (asserted here; every GT read goes through gt_fit / gt_btcv / gt_eval):
  FLARE labelled 4 (0014 0001 0006 0038): GT allowed for fitting (gt_fit), raw-grid GT never needed.
  BTCV (30, btcv_ours_h5): GT allowed (external cohort = leave-site-out prior source; FLARE never enters its prior).
  FLARE dev18: GT ONLY through gt_eval(), ONLY after results/GT_OPENED_dev18 exists (created O_EXCL by score_ms.py after
               every moonshot label map was written and hashed).
  FLARE held18 / test-10, AMOS (any), MACT (any): REFUSED. No code path of this block reads them.

This module has no import-time side effects beyond reading constants (GPU jobs never import it; ff_launch.py patches the
fair-fight module instead).
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
OUT = REPO / "outputs/results/vccor/moonshot"
LOGD = OUT / "logs"
PREREG = OUT / "prereg.json"
PREREG_LOG = LOGD / "moonshot_prereg.log"
LEDGER = OUT / "look_ledger.jsonl"
DOC = REPO / "docs/MOONSHOT_2026-11-03.md"

PRIOR_DIR = OUT / "prior"
PRIOR_JSON = PRIOR_DIR / "prior_btcv.json"
FIT_DIR = OUT / "fits"
FITS_JSON = FIT_DIR / "fits.json"
MAPS = OUT / "maps"
MAPS_MANIFEST = MAPS / "label_manifest.json"
MAPS_DONE = MAPS / "DONE.json"
FORGET_JSON = MAPS / "forgetting_label_free.json"
RES = OUT / "results"
GT_MARKER = RES / "GT_OPENED_dev18"
STAGE1_JSON = RES / "stage1.json"
VERDICT_JSON = RES / "verdict.json"
FF = OUT / "ff"
CHAIN = OUT / "chain"
STOP_FILE = CHAIN / "STOP_AFTER_CURRENT"

DATA = REPO / "MagicNet/data"
FLARE_H5 = DATA / "flare_h5"
PSEUDO_FULLRES = DATA / "flare_pseudo_fullres"
BTCV_GT = DATA / "btcv_ours_h5"
BTCV_TS = DATA / "btcv_ts_fullres"
F0_PRED = REPO / "outputs/results/vccor/tsft_full/pred/F0"
TS_CAPTURE = REPO / "outputs/results/vccor/tsft_full/prep/ts_cpu"
FRAME_CACHE = REPO / "outputs/results/extras/ts_finetune/h5_frames"
RAW_IMG = REPO / "data/FLARE22Train/images/FLARE22_Tr_{cid}_0000.nii.gz"
FF_SCRIPTS = REPO / "scripts/vccor/fairfight"
TSFT_SCRIPTS = REPO / "scripts/vccor/tsft_full"
CC_DIR = REPO / "scripts/cor_confirm"
TSFT_COMMON_DIR = REPO / "scripts/extras/ts_finetune"

NAMES = ["liver", "r.kid", "spleen", "panc", "aorta", "ivc", "r.adr", "l.adr", "gall", "esoph", "stom", "duod", "l.kid"]
NO = len(NAMES)
BTCV_NAMES = ["spleen", "r.kid", "l.kid", "gall", "esoph", "liver", "stom", "aorta", "ivc", "pv_sv", "panc", "r.adr",
              "l.adr"]
LAB = ["0014", "0001", "0006", "0038"]
DEV18 = ["0002", "0003", "0004", "0005", "0007", "0008", "0009", "0010", "0011", "0012", "0013", "0015", "0016", "0017",
         "0018", "0019", "0020", "0021"]
HELD18 = ["0022", "0023", "0024", "0025", "0026", "0027", "0028", "0029", "0030", "0031", "0032", "0033", "0034", "0035",
          "0036", "0037", "0039", "0040"]
TEST10 = ["0041", "0042", "0043", "0044", "0045", "0046", "0047", "0048", "0049", "0050"]
DRAWS = {f"d1_{c}": [c] for c in LAB}
DRAW_ORDER = list(DRAWS)
BTCV_IDS = [f"{i:04d}" for i in list(range(1, 11)) + list(range(21, 41))]

# TotalSegmentator 'total' id of each FLARE organ (FLARE id = index + 1); kidney cysts 23 -> l.kid, 24 -> r.kid (v2 map)
CANON_TS_ID = [5, 2, 1, 7, 52, 63, 8, 9, 4, 15, 6, 19, 3]

# ------------------------------------------------------------------------------------------------ convention head
H5_SPACING = 1.5
DMAX_MM = 9.0
BBOX_PAD_VOX = 9
TAU_MM = 0.5
PRIOR_SCALE_MM = 2.0
LAMBDA = 0.01
LBFGS_ITERS = 100
NM_ITERS = 200
MIN_ORGAN_VOX = 50
N_THETA = 5
ARMS_MS = {
    "MS": dict(n_params=5, lam=LAMBDA, prior=True, anchoring="strict", primary=True,
               label="PRIMARY: 5-parameter signed-distance offset field (offset, 3 direction terms, concavity term), "
                     "ridge to the BTCV prior (lambda 0.01), strict anchoring"),
    "MS_NP": dict(n_params=5, lam=0.0, prior=False, anchoring="strict", primary=False,
                  label="no prior (lambda 0, init 0), strict anchoring"),
    "MS_OFF": dict(n_params=1, lam=LAMBDA, prior=True, anchoring="strict", primary=False,
                   label="offset-only head (1 parameter per organ), BTCV prior, strict anchoring"),
    "MS_SOFT": dict(n_params=5, lam=LAMBDA, prior=True, anchoring="soft", primary=False,
                    label="as MS but growth may also claim non-target TS classes (forgetting possible)"),
}
ARM_PRIOR_ONLY = "MS_PRIOR"

# ------------------------------------------------------------------------------------------------ kill-test rule
FUTILITY_MIN_GAIN = 0.50
FORGET_MAX_LOSS = 2.0
MIN_FT_DRAWS = 3
GPU_BUDGET_H = 9.0
GPU_DRAW_EST_H = 2.3
NBOOT, BOOT_SEED = 5000, 0
FORGET_MIN_REF_VOX = 100


# ------------------------------------------------------------------------------------------------ helpers
def now():
    return time.strftime("%F %T")


def sha256(path, n=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(n), b""):
            h.update(b)
    return h.hexdigest()


def arr_sha(a):
    import numpy as np
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def rel(p):
    try:
        return str(Path(p).resolve().relative_to(REPO))
    except ValueError:
        return str(p)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(tmp, path)


class Log:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg):
        line = f"[{now()}] {msg}"
        print(line, flush=True)
        with open(self.path, "a") as f:
            f.write(line + "\n")


def ledger(event, **kw):
    """Append-only look ledger (one JSON object per line)."""
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    rec = dict(time=now(), event=event, **kw)
    with open(LEDGER, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    return rec


def cap_threads(n):
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = str(n)


def be_nice(target=15):
    cur = os.nice(0)
    if cur < target:
        os.nice(target - cur)


# ------------------------------------------------------------------------------------------------ prereg / freeze
def prereg_sha_logged():
    h = None
    if PREREG_LOG.is_file():
        for line in open(PREREG_LOG):
            p = line.split()
            if len(p) >= 5 and p[2] == "PREREG" and p[3] == "sha256":
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


def check_prereg():
    h = sha256(PREREG)
    if h != prereg_sha_logged():
        raise SystemExit(f"[REFUSE] prereg.json sha256 {h} != logged {prereg_sha_logged()} ({rel(PREREG_LOG)})")
    return json.load(open(PREREG))


def check_code_frozen(paths, strict=True, log=print):
    fz = frozen_code_hashes()
    bad = {}
    for p in paths:
        r = rel(p)
        got = sha256(REPO / r)
        if fz.get(r) != got:
            bad[r] = (got[:16], (fz.get(r) or "NOT-FROZEN")[:16])
    if bad:
        msg = f"code differs from the CODEFREEZE hashes in {rel(PREREG_LOG)}: {bad}"
        if strict:
            raise SystemExit("[REFUSE] " + msg)
        log("[warn] " + msg)
    return not bad


CODE_FILES = ["ms_common.py", "ms_head.py", "write_prereg.py", "ff_launch.py", "prior_btcv.py", "fit_ms.py", "apply_ms.py",
              "score_ms.py", "gpu_chain.py", "smoke_test.py", "da_bench.py", "freeze.py"]
FF_FILES = [FF_SCRIPTS / f for f in ("ff_common.py", "ff_pipeline.py", "prep_ff.py", "prep_ff_helpers.py", "verify_ff.py",
                                     "train_ff.py", "predict_ff.py")]
IMPORT_FILES = [TSFT_SCRIPTS / "tsft_full_common.py", TSFT_SCRIPTS / "ts_pipeline.py", CC_DIR / "confirm_common.py"]
FF_FROZEN_SHA = {
    "scripts/vccor/fairfight/ff_common.py": "d61f743457752f70a282d591ef13756f53a807485f59a0e211e80ccd4ef4e655",
    "scripts/vccor/fairfight/ff_pipeline.py": "1dc69d656cc6d1ffd6947e1ff7906d25984309de4d87da336f7eab6be04c0932",
    "scripts/vccor/fairfight/prep_ff.py": "27720fd66b2819203549fa227a7d5fb50253c916558090e5a207d2e093b021eb",
    "scripts/vccor/fairfight/prep_ff_helpers.py": "89678b7b4f9e05d7fa2310416f00ae3150003ce43c66b1f602dbcd650b726ca8",
    "scripts/vccor/fairfight/verify_ff.py": "34295cdfb2d3577bbb6a39e7afd7caae9427f55ae6843fbda3b7147ae6155bd4",
    "scripts/vccor/fairfight/train_ff.py": "9854b5f000baed13230a64bda6ba30c80e9c2944594b983547f1b39ca4ee26d1",
    "scripts/vccor/fairfight/predict_ff.py": "59e020e9ff30989606173ebd1fa86296d6d23a00e5a32ad290bf50677ef48df3",
    "scripts/vccor/tsft_full/tsft_full_common.py": "4f3efd44f775708dfe05333845077e89782fa57d242071bab16041caddb13e20",
    "scripts/vccor/tsft_full/ts_pipeline.py": "51bca3e2f545b0c4c0d873abdb16dbd2f978ba135e1def07554ae12e57bb3bd2",
    "scripts/cor_confirm/confirm_common.py": "292bdfe4c38db2fe2e0755cee2e6e685416529e5ade55917e4c3eb0bb49c6de1",
}


# ------------------------------------------------------------------------------------------------ GT discipline
GT_LOG = []
_REFUSED = set(HELD18) | set(TEST10)


def _refuse_sealed(cid):
    if cid in _REFUSED:
        raise PermissionError(f"moonshot: GT of FLARE {cid} (held18 / test-10) is never read in this stage")


def _h5(path, key):
    import h5py
    import numpy as np
    with h5py.File(path, "r") as f:
        a = f[key][:]
    if a.dtype != np.uint8:
        r = np.rint(a)
        assert np.abs(a - r).max() < 1e-6 and r.min() >= 0 and r.max() <= 255, path
        a = r.astype(np.uint8)
    return a


def gt_fit(cid):
    """flare_h5 'label' of a LABELLED scan (fitting only)."""
    _refuse_sealed(cid)
    if cid not in LAB:
        raise PermissionError(f"moonshot: {cid} is not a labelled scan; dev18 GT only via gt_eval() after GT_OPENED")
    GT_LOG.append(("flare_fit", cid))
    return _h5(FLARE_H5 / f"{cid}.h5", "label")


def gt_btcv(cid):
    """BTCV label (BTCV ids 1..13) in the btcv_ours_h5 frame; external prior source only."""
    assert cid in BTCV_IDS, cid
    GT_LOG.append(("btcv", cid))
    a = _h5(BTCV_GT / f"{cid}.h5", "label")
    assert int(a.max()) <= len(BTCV_NAMES), (cid, int(a.max()))
    return a


def gt_eval(cid):
    """flare_h5 'label' of a dev18 scan, ONLY after the GT_OPENED_dev18 marker exists."""
    _refuse_sealed(cid)
    if cid not in DEV18:
        raise PermissionError(f"moonshot: {cid} is not a dev18 scan")
    if not GT_MARKER.is_file():
        raise PermissionError(f"moonshot: {rel(GT_MARKER)} missing: write and hash every label map before opening GT")
    GT_LOG.append(("flare_dev18", cid))
    return _h5(FLARE_H5 / f"{cid}.h5", "label")


# ------------------------------------------------------------------------------------------------ hosts (no GT)
def frame(cid):
    """flare_h5 frame of a scan from the cache (never recomputed here: recomputation would read the GT)."""
    p = FRAME_CACHE / f"{cid}.json"
    if not p.is_file():
        raise SystemExit(f"[REFUSE] h5 frame cache missing for {cid}")
    d = json.load(open(p))
    return {"spacing": tuple(float(x) for x in d["spacing"]), "slices": tuple(slice(a, b) for a, b in d["slices"]),
            "shape": tuple(int(x) for x in d["shape"]), "raw_shape": tuple(int(x) for x in d["raw_shape"])}


def raw_to_h5(arr, cid):
    """== tsft_common.raw_to_h5 geometry (scipy zoom spacing / 1.5, order 0, mode nearest, then the flare_h5 crop)."""
    import numpy as np
    from scipy.ndimage import zoom
    fr = frame(cid)
    assert tuple(arr.shape) == fr["raw_shape"], (cid, arr.shape, fr["raw_shape"])
    out = zoom(arr.astype(np.uint8), np.asarray(fr["spacing"], float) / H5_SPACING, order=0, mode="nearest")[fr["slices"]]
    assert out.shape == fr["shape"], (cid, out.shape, fr["shape"])
    return np.ascontiguousarray(out)


_LUT = None


def lut_v2():
    """TS 'total' id -> FLARE id (tsft_common.ts_to_flare_lut('v2'): kidney cysts mapped to the same-side kidney)."""
    global _LUT
    if _LUT is None:
        sys.path.insert(0, str(TSFT_COMMON_DIR))
        import tsft_common as C
        _LUT = C.ts_to_flare_lut("v2")
        sys.path.remove(str(TSFT_COMMON_DIR))
    return _LUT


def target_ts_ids():
    import numpy as np
    return [int(k) for k in np.nonzero(lut_v2())[0]]


def host_labelled(cid):
    """TS-full (CPU) output of a labelled scan in the h5 frame: (flare ids, raw TS ids). Raw ids come from the frozen
    tsft_full prep capture raw_total.nii.gz (TS 'total' ids, raw FLARE grid) through raw_to_h5; the FLARE map is checked
    voxel-identical to MagicNet/data/flare_pseudo_fullres 'pseudo' (the host of every earlier correction)."""
    import nibabel as nib
    import numpy as np
    assert cid in LAB, cid
    seg = np.asarray(nib.load(str(TS_CAPTURE / cid / "raw_total.nii.gz")).dataobj)
    assert seg.min() >= 0 and seg.max() < 256
    raw = raw_to_h5(seg.astype(np.uint8), cid)
    fl = lut_v2()[raw]
    ref = _h5(PSEUDO_FULLRES / f"{cid}.h5", "pseudo")
    assert np.array_equal(fl, ref), f"{cid}: LUT(raw_total -> h5) != flare_pseudo_fullres pseudo"
    return fl.astype(np.uint8), raw.astype(np.uint8)


def host_f0(cid):
    """Zero-shot F0 (TS-full weights through the GPU path) in the h5 frame: (flare ids 'pseudo', raw TS ids 'raw_h5')."""
    import h5py
    import numpy as np
    assert cid in DEV18, cid
    with h5py.File(F0_PRED / "h5" / f"{cid}.h5", "r") as f:
        fl, raw = f["pseudo"][:].astype(np.uint8), f["raw_h5"][:].astype(np.uint8)
    assert np.array_equal(lut_v2()[raw], fl), f"{cid}: F0 pseudo != LUT(raw_h5)"
    return fl, raw


def btcv_pair(cid):
    """BTCV GT and TS-full host, both remapped to FLARE ids (pv_sv -> 255 = other structure; duodenum absent)."""
    import numpy as np
    G = gt_btcv(cid)
    T = _h5(BTCV_TS / f"{cid}.h5", "pseudo")
    assert G.shape == T.shape, (cid, G.shape, T.shape)
    m = np.zeros(256, np.uint8)
    for i, n in enumerate(BTCV_NAMES, start=1):
        m[i] = NAMES.index(n) + 1 if n in NAMES else 255
    return m[G], m[T]


def ft_pred_h5(draw, cid):
    """FT-full K=1 prediction of a draw (moonshot re-run) in the h5 frame: (flare ids, raw TS ids)."""
    import h5py
    import numpy as np
    p = FF / "pred" / "FULL" / draw / "h5" / f"{cid}.h5"
    with h5py.File(p, "r") as f:
        return f["pseudo"][:].astype(np.uint8), f["raw_h5"][:].astype(np.uint8)


def nontarget_ft_classes():
    """Non-FLARE TS classes of the fine-tuned parts 291 + 293 (== tsft_full score.forgetting class set)."""
    sys.path.append(str(TSFT_SCRIPTS))
    import tsft_full_common as Q
    from totalsegmentator.map_to_binary import class_map
    v2 = lut_v2()
    inv = {v: k for k, v in class_map["total"].items()}
    klass = sorted({inv[n] for t in Q.PARTS_FT for n in Q.part_classes(t).values() if v2[inv[n]] == 0})
    return {k: class_map["total"][k] for k in klass}
