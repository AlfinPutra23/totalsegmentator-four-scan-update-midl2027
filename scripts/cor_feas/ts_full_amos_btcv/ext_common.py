"""cor_feas / ts_full_amos_btcv: shared paths and conventions for FULL-RESOLUTION TotalSegmentator on AMOS-CT (60) and
BTCV (30), and for mapping its raw outputs to the trainer-format teacher label sets used by the existing --fast pipelines.

Teacher: TotalSegmentator 2.13.0 (.venv), task 'total', --ml, NO --fast (Dataset291-295, 1.5 mm, 5 nnU-Nets merged),
device cpu, 6 torch threads, nr_thr_resamp 1, nr_thr_saving 1 -- i.e. scripts/ts_fullres/run_ts.py::run_ts_one(fast=False),
the exact call that produced the FLARE full-res labels (flare_pseudo_fullres, FLARE test 88.65).

Layout (new paths only; nothing existing is modified)
  scripts/cor_feas/ts_full_amos_btcv/           this code
  MagicNet/data/ts_fullres_ext/raw_full/<ds>/   raw-grid TS 'total' multilabel NIfTI, ds in {amos, btcv}
  MagicNet/data/amos_pseudo_fullres/<cid>.h5    AMOS trainer-format teacher (== scripts/amos_prep/build_ts_pseudo.py layout:
                                                'pseudo' uint8 FLARE ids, 'conf', 'protocol_mask' zeros, gzip 4)
  MagicNet/data/btcv_ts_fullres/<cid>.h5        BTCV trainer-format teacher (== method_candidates/btcv_ts_v2 layout: BTCV ids)
  outputs/results/cor_feas/ts_full_amos_btcv/   timing/, build/ (per-case reports), proof_fast_*.json, validate_*.json
  outputs/logs/cor_feas/ts_full_amos_btcv.log   progress log of the background TS job (+ .pid, .lock)

Frames (replayed exactly like the existing pipelines; GT is read here ONLY to recover the GT-bbox crop of the h5 frame and to
prove the replay reproduces the stored h5 label bit-exactly -- never for fitting; scoring only in validate on train volumes):
  AMOS : scripts/prep_amos_ct.py (== scripts/extras/external_rescore/ts_direct.py::replay): zoom spacing/1.5 order 0
         'nearest', AMOS->FLARE remap, crop bbox(remapped label) -10/+11; TS ids -> FLARE ids with TS_TO_FLARE_V2.
  BTCV : scripts/build_btcv_real_pipeline.py / build_btcv_pseudo.py (== scripts/btcv_cor/build_ts_v2.py replay): zoom
         spacing/1.5 order 0, trim to min(image zoom shape, label, seg), crop bbox(label) -10/+11; TS ids -> BTCV ids TS2B+CYST.
"""
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True   # importing repo modules must never drop __pycache__ next to them

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DATA = ROOT / "MagicNet/data"
TS_FULLRES_DIR = ROOT / "scripts/ts_fullres"               # finished workflow: read/reuse only
VENV_PY = ROOT / ".venv/bin/python"

RAW_ROOT = DATA / "ts_fullres_ext"
OUT = ROOT / "outputs/results/cor_feas/ts_full_amos_btcv"
TIMING = OUT / "timing"
BUILD = OUT / "build"
LOGD = ROOT / "outputs/logs/cor_feas"
LOG = LOGD / "ts_full_amos_btcv.log"
PIDF = LOGD / "ts_full_amos_btcv.pid"
LOCKF = LOGD / "ts_full_amos_btcv.lock"

LABEL_DIR = {"amos": DATA / "amos_pseudo_fullres", "btcv": DATA / "btcv_ts_fullres"}

# ---- raw CT (exactly what the existing pipelines read)
AMOS_RAW = Path("/media/user/B4864CD4864C98AE/amos_ct")                                    # prep_amos_ct.py AMOS
BTCV_RAW = Path("/media/user/B4864CD4864C98AE/medssl_data/BTCV_raw/RawData/Training")      # build_btcv_real_pipeline.py RAW

# ---- existing --fast artefacts used as bit-exact controls of the build path
AMOS_FAST_RAW_SMOKE = ROOT / "outputs/results/extras/external_rescore/smoke/ts_raw/amos"   # ts_direct.py --stage ts (smoke, 0001)
AMOS_FAST_ALIGNED_SMOKE = ROOT / "outputs/results/extras/external_rescore/smoke/amos_ts"   # ts_direct.py --stage align (ts_v1/ts_v2)
AMOS_FAST_PSEUDO_SMOKE = Path("/tmp/claude-1000/-home-user-Documents-Medical-Imagining-SSL/"
                              "e0eae168-45fa-47e8-b092-dccd1465a55b/scratchpad/amos_smoke/ts")  # build_ts_pseudo.py --out (smoke)
AMOS_FAST_RAW_FULLSET = ROOT / "outputs/results/extras/external_rescore/ts_raw/amos"       # queue_v8 er_ts_direct (not run yet)
AMOS_FAST_PSEUDO_FULLSET = DATA / "amos_ts_v2_pseudo"                                       # build_ts_pseudo.py default out
BTCV_FAST_RAW = ROOT / "outputs/results/seg_btcv_raw"                                       # TS --ml --fast cache (30)
BTCV_FAST_V1 = DATA / "btcv_ours_pseudo"                                                    # build_btcv_pseudo.py (no cyst fix)
BTCV_FAST_V2 = DATA / "method_candidates/btcv_ts_v2"                                        # build_ts_v2.py (cyst fix)
BTCV_FULL_CHECK_0001 = DATA / "ts_fullres/btcv_check/full_0001.nii.gz"                     # ts_fullres stage btcv (same settings)

# ---- label maps
# TotalSegmentator v2 'total' ids -> FLARE ids (== er_common.TS_TO_FLARE_V2 == tsft_common.ts_to_flare_lut('v2'))
TS_TO_FLARE_V1 = {1: 3, 2: 2, 3: 13, 4: 9, 5: 1, 6: 11, 7: 4, 8: 7, 9: 8, 15: 10, 19: 12, 52: 5, 63: 6}
TS_TO_FLARE_V2 = {**TS_TO_FLARE_V1, 23: 13, 24: 2}
# TotalSegmentator 'total' ids -> BTCV ids (== scripts/btcv_cor/build_ts_v2.py TS2B + CYST)
TS2B = {1: 1, 2: 2, 3: 3, 4: 4, 15: 5, 5: 6, 6: 7, 52: 8, 63: 9, 64: 10, 7: 11, 8: 12, 9: 13}
CYST_B = {23: 3, 24: 2}
TS_NAMES_USED = {1: "spleen", 2: "kidney_right", 3: "kidney_left", 4: "gallbladder", 5: "liver", 6: "stomach", 7: "pancreas",
                 8: "adrenal_gland_right", 9: "adrenal_gland_left", 15: "esophagus", 19: "duodenum", 23: "kidney_cyst_left",
                 24: "kidney_cyst_right", 52: "aorta", 63: "inferior_vena_cava", 64: "portal_vein_and_splenic_vein"}
AMOS_TO_FLARE = {1: 3, 2: 2, 3: 13, 4: 9, 5: 10, 6: 1, 7: 11, 8: 5, 9: 6, 10: 4, 11: 7, 12: 8, 13: 12}   # prep_amos_ct.py

FLARE_NAMES = ['liver', 'r.kid', 'spleen', 'panc', 'aorta', 'ivc', 'r.adr', 'l.adr', 'gall', 'esoph', 'stom', 'duod', 'l.kid']
BTCV_NAMES = ['spleen', 'r.kid', 'l.kid', 'gall', 'esoph', 'liver', 'stom', 'aorta', 'ivc', 'pv_sv', 'panc', 'r.adr', 'l.adr']
ISO, MARGIN = 1.5, 10


# ------------------------------------------------------------------------------------------------- case lists / split
def _ids(txt):
    return ["{}".format(x.strip().split('.')[0][-4:]) for x in open(txt).readlines() if x.strip()]   # trainer parsing


def amos_ids():
    return _ids(DATA / "amos.txt")


def btcv_ids():
    return _ids(DATA / "btcv.txt")


def _split(ids, n_train, seed):
    """Replay of train_main_flare_octree.py (== amos_common.recover_split / common_btcv.recover_split)."""
    import numpy as np
    tr, te = list(ids[:n_train]), list(ids[n_train:])
    np.random.RandomState(int(str(seed).lstrip('s'))).shuffle(tr)
    return tr, te


SEEDS = ("s1337", "s2025", "s3407")
SPLIT = {"amos": dict(n_train=40, labelnum=4, ids=amos_ids), "btcv": dict(n_train=18, labelnum=5, ids=btcv_ids)}


def labelled(ds, seed):
    s = SPLIT[ds]
    return _split(s["ids"](), s["n_train"], seed)[0][:s["labelnum"]]


def test_ids(ds):
    s = SPLIT[ds]
    return s["ids"]()[s["n_train"]:]


def train_ids(ds):
    s = SPLIT[ds]
    return s["ids"]()[:s["n_train"]]


def priority_order(ds):
    """Processing order: the seeds' labelled volumes first (what a 4-scan predictor / COR fit needs; AMOS 0001 first because
    a cached --fast output exists for it), then the test set (TS-direct test row), then the rest of the training pool."""
    first = []
    if ds == "amos":
        first.append("0001")
    for s in SEEDS:
        first += labelled(ds, s)
    order = []
    for c in first + test_ids(ds) + train_ids(ds):
        if c not in order:
            order.append(c)
    assert sorted(order) == sorted(SPLIT[ds]["ids"]()), ds
    return order


def job_order():
    return [("amos", c) for c in priority_order("amos")] + [("btcv", c) for c in priority_order("btcv")]


# --------------------------------------------------------------------------------------------------------- raw paths
def raw_img(ds, cid):
    if ds == "amos":
        for sub, d in (("train", "imagesTr"), ("valid", "imagesVa")):        # == ts_direct.amos_raw
            ip = AMOS_RAW / sub / d / f"amos_{cid}.nii.gz"
            if ip.is_file():
                return ip
        raise FileNotFoundError(f"raw AMOS case {cid} not found")
    ip = BTCV_RAW / "img" / f"img{cid}.nii.gz"
    if not ip.is_file():
        raise FileNotFoundError(ip)
    return ip


def raw_lbl(ds, cid):
    ip = raw_img(ds, cid)
    if ds == "amos":
        return Path(str(ip).replace("imagesTr", "labelsTr").replace("imagesVa", "labelsVa"))
    return BTCV_RAW / "label" / f"label{cid}.nii.gz"


def raw_full(ds, cid):
    return RAW_ROOT / "raw_full" / ds / f"{cid}.nii.gz"


def timing_json(ds, cid):
    return TIMING / f"{ds}_{cid}.json"


def raw_done(ds, cid):
    return raw_full(ds, cid).is_file() and timing_json(ds, cid).is_file()


def h5_ref(ds, cid):
    return (DATA / "amos_ct_h5" / f"{cid}.h5") if ds == "amos" else (DATA / "btcv_ours_h5" / f"{cid}.h5")


def cap_threads(n):
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = str(n)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
