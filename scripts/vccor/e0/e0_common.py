"""vccor/e0 common: paths, organ maps, data-discipline guards (CPU only, nice 15, <= 4 threads).

E0 = pre-registered, CPU-only protocol-compatibility map deciding GO / NO-GO for "protocol-compatibility-gated
fine-tuning" (fine-tune TS-full on all organs, but only on label sources whose per-organ protocol matches the target's).

Read-only inputs (never written): MagicNet/data/{flare_h5, flare_pseudo_fullres, amos_ct_h5, amos_pseudo_fullres,
btcv_ours_h5, btcv_ts_fullres, MACT_h5, mact_pseudo_fullres}, raw CT folders, outputs/results/vccor/{fairfight,
tsft_full, audit_ft, xdata/splits}. Writes only outputs/results/vccor/e0/, outputs/plots/vccor/e0_*, outputs/logs/vccor/e0_*.

GT DISCIPLINE (asserted in gt_read):
  flare : 'label' read ONLY for the seed-1337 labelled 4 (0014 0001 0006 0038)  -- never dev18 / held18 / test10
  mact  : 'label' read ONLY for the xdata seed-1337 K=4 (0075 0049 0037 0041)   -- MACT is sealed otherwise
  amos  : all 60 (external label source)        btcv : all 30 (external label source)
Image-only reads (overlap check) never open a 'label' dataset.
"""
import json
import os
import sys
import time
import hashlib
from pathlib import Path

N_THREADS = 4
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = str(N_THREADS)
sys.dont_write_bytecode = True

REPO = Path("/home/user/Documents/Medical Imagining SSL")
DATA = REPO / "MagicNet/data"
OUT = REPO / "outputs/results/vccor/e0"
PLOTS = REPO / "outputs/plots/vccor"
LOGD = REPO / "outputs/logs/vccor"
PREREG = OUT / "prereg.json"
PREREG_LOG = LOGD / "e0_prereg.log"
XDATA_SPLITS = REPO / "outputs/results/vccor/xdata/splits"

FLARE_NAMES = ["liver", "r.kid", "spleen", "panc", "aorta", "ivc", "r.adr", "l.adr", "gall", "esoph", "stom", "duod",
               "l.kid"]                                                        # FLARE ids 1..13 (liver-first)
BTCV_NAMES = ["spleen", "r.kid", "l.kid", "gall", "esoph", "liver", "stom", "aorta", "ivc", "pv_sv", "panc", "r.adr",
              "l.adr"]                                                         # BTCV ids 1..13 (no duodenum)
MACT_NAMES = ["spleen", "l.kid", "gall", "esoph", "liver", "stom", "panc", "duod"]   # MACT ids 1..8

# dataset -> (GT h5 dir, GT key, TS-full h5 dir, TS key, label names (id = index + 1), h5 voxel spacing mm)
DS = {
    "flare": dict(gt=DATA / "flare_h5", gt_key="label", ts=DATA / "flare_pseudo_fullres", ts_key="pseudo",
                  names=FLARE_NAMES, spacing=1.5),
    "amos": dict(gt=DATA / "amos_ct_h5", gt_key="label", ts=DATA / "amos_pseudo_fullres", ts_key="pseudo",
                 names=FLARE_NAMES, spacing=1.5),
    "btcv": dict(gt=DATA / "btcv_ours_h5", gt_key="label", ts=DATA / "btcv_ts_fullres", ts_key="pseudo",
                 names=BTCV_NAMES, spacing=1.5),
    "mact": dict(gt=DATA / "MACT_h5", gt_key="label", ts=DATA / "mact_pseudo_fullres", ts_key="pseudo",
                 names=MACT_NAMES, spacing=1.0),
}

FLARE_LAB4 = ["0014", "0001", "0006", "0038"]       # seed 1337 trainer split replay (flare.txt[:40], RandomState(1337))
MACT_LAB4 = ["0075", "0049", "0037", "0041"]        # xdata splits.md: seed-1337 K4 = pool[:4]


def ids_of(ds):
    """Scans whose GT this experiment may read."""
    if ds == "flare":
        return list(FLARE_LAB4)
    if ds == "mact":
        return list(MACT_LAB4)
    return sorted(p.stem for p in DS[ds]["gt"].glob("*.h5"))


_ALLOWED = {"flare": set(FLARE_LAB4), "mact": set(MACT_LAB4)}


def gt_read(ds, cid):
    """THE only GT reader of E0 (asserts the data discipline). Returns uint8 label array."""
    import h5py
    import numpy as np
    if ds in _ALLOWED:
        assert cid in _ALLOWED[ds], f"DATA DISCIPLINE: {ds} GT {cid} is not in the allowed labelled set"
    with h5py.File(DS[ds]["gt"] / f"{cid}.h5", "r") as f:
        a = f[DS[ds]["gt_key"]][:]
    if a.dtype != np.uint8:
        r = np.rint(a)
        assert np.abs(a - r).max() < 1e-6 and r.min() >= 0 and r.max() <= 255, (ds, cid)
        a = r.astype(np.uint8)
    assert int(a.max()) <= len(DS[ds]["names"]), (ds, cid, int(a.max()))
    return a


def ts_read(ds, cid):
    import h5py
    import numpy as np
    with h5py.File(DS[ds]["ts"] / f"{cid}.h5", "r") as f:
        a = f[DS[ds]["ts_key"]][:]
    assert a.dtype == np.uint8 and int(a.max()) <= len(DS[ds]["names"]), (ds, cid, a.dtype, int(a.max()))
    return a


def ts_ready(ds, cid):
    return (DS[ds]["ts"] / f"{cid}.h5").is_file()


def organ_id(ds, name):
    """dataset label id of a canonical organ name, or None if the dataset does not annotate it."""
    n = DS[ds]["names"]
    return n.index(name) + 1 if name in n else None


def sha256(path, n=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(n), b""):
            h.update(b)
    return h.hexdigest()


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
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(self.path, "a") as f:
            f.write(line + "\n")


def be_nice(target=15):
    cur = os.nice(0)
    if cur < target:
        os.nice(target - cur)


def now():
    return time.strftime("%F %T")


def rel(p):
    try:
        return str(Path(p).resolve().relative_to(REPO))
    except ValueError:
        return str(p)
