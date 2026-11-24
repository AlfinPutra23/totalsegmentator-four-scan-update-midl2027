# cor_salvage/verify_v2: independent verifier of the v2 VC_COR dev18 screen. Shared helpers, written from the
# pre-registration text (outputs/results/cor_salvage/prereg.json). Imports NO code from the v2 tester, design3_reframe,
# diag1, cor36 or ts_fullres modules: loaders, fits, operators and scoring are re-implemented here.
# DATA DISCIPLINE: gt() opens ground truth only for the 4 labelled seed-1337 scans and the dev18 scans.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import numpy as np
import h5py
from scipy import ndimage

REPO = "/home/user/Documents/Medical Imagining SSL"
DATA = f"{REPO}/MagicNet/data"
MC = f"{DATA}/method_candidates"
OUT = f"{REPO}/outputs/results/cor_salvage/verify_v2"
PREREG = json.load(open(f"{REPO}/outputs/results/cor_salvage/prereg.json"))
NAMES = ["liver", "r.kid", "spleen", "panc", "aorta", "ivc", "r.adr", "l.adr", "gall", "esoph", "stom", "duod", "l.kid"]
DCAP = 8
S_FN, S_TRIM = 0.3, -0.3

# ---- split, rebuilt from flare.txt (first 40 = train, last 10 = test; seed-1337 labelled 4 from the prereg)
_ids = [l.strip().split(".")[0][-4:] for l in open(f"{DATA}/flare.txt") if l.strip()]
TRAIN, TEST = _ids[:40], _ids[40:]
LAB = list(PREREG["data_discipline"]["fit_scans"])
POOL = [c for c in TRAIN if c not in LAB]
DEV18 = POOL[:18]
HELD18 = POOL[18:]
assert DEV18 == PREREG["data_discipline"]["screen_scans_dev18"], DEV18
assert HELD18 == [f"{i:04d}" for i in list(range(22, 38)) + [39, 40]], HELD18
assert TEST == [f"{i:04d}" for i in range(41, 51)], TEST
_ALLOWED = frozenset(LAB) | frozenset(DEV18)
GT_LOG = []          # every GT open is recorded (audit)
OPENED = []          # every h5 path opened by this process (audit)


def h5read(path, key):
    OPENED.append(path)
    with h5py.File(path, "r") as f:
        return f[key][:]


def gt(c):
    if c not in _ALLOWED:
        raise PermissionError(f"verify_v2: GT of {c} is forbidden (held18/test)")
    GT_LOG.append(c)
    return h5read(f"{DATA}/flare_h5/{c}.h5", "label")


def host(c, X):
    d = "flare_pseudo_fullres" if X == "full" else "flare_pseudo_v2"
    return h5read(f"{DATA}/{d}/{c}.h5", "pseudo").astype(np.uint8)


def carrier(c):
    d = f"{MC}/carrier_basestd16_s1337_lab" if c in LAB else f"{MC}/carrier_basestd16_s1337"
    OPENED.append(f"{d}/{c}.h5")
    with h5py.File(f"{d}/{c}.h5", "r") as f:
        return f["pred"][:].astype(np.uint8), f["prob"][:].astype(np.float32)


def edt_out(M):
    """Euclidean distance (voxel units) from every voxel to the nearest voxel of M (0 inside M)."""
    return ndimage.distance_transform_edt(~M)


def edt_in(M):
    """Inner EDT: for voxels of M, distance to the nearest voxel outside M (0 outside M)."""
    return ndimage.distance_transform_edt(M)


# ---------------------------------------------------------------- COR fit (labelled 4 only), from the written text
def cor_fit(pairs):
    """pairs: list of (G, T) for the labelled scans. Per organ: signed bias, P90 FN distance to T_o, P90 FP inner
    distance, allowed source labels (>= 5 % of FN voxels, excluding o). Organs with |G_o| < 20 on a scan skip that scan."""
    out = {}
    for o in range(1, 14):
        s, fnd, fpd = [], [], []
        lab = np.zeros(14, np.int64)
        for G, T in pairs:
            Go, To = G == o, T == o
            if Go.sum() < 20:
                continue
            FN, FP = Go & ~To, To & ~Go
            nfn, nfp = int(FN.sum()), int(FP.sum())
            s.append((nfn - nfp) / max(nfn + nfp, 1))
            fnd.append(edt_out(To)[FN])
            fpd.append(edt_in(To)[FP])
            lab += np.bincount(T[FN], minlength=14)[:14]
        fnd, fpd = np.concatenate(fnd), np.concatenate(fpd)
        share = lab / max(lab.sum(), 1)
        out[NAMES[o - 1]] = dict(signed=float(np.mean(s)), fn_d90=float(np.percentile(fnd, 90)) if fnd.size else 0.0,
                                 fp_d90=float(np.percentile(fpd, 90)) if fpd.size else 0.0,
                                 allowed=[int(l) for l in range(14) if share[l] >= 0.05 and l != o])
    return out


def vc_fit(vols):
    """vols: (n, 13, 2) = (|G_o|, |T_o|) on the labelled scans."""
    v = np.asarray(vols, np.float64)
    r = (v[..., 0] - v[..., 1]) / v[..., 1]
    m = r.mean(0)
    se = r.std(0, ddof=1) / np.sqrt(r.shape[0])
    delta = np.where(m - se > 0.02, m - se, 0.0)
    eps = np.where(-m - se > 0.02, -m - se, 0.0)
    return dict(delta=delta, eps=eps, r=r)


def sources(fit, o):
    """A_o = fitted allowed sources minus labels l>0 whose fitted signed bias is >= +0.3."""
    return [l for l in fit[NAMES[o - 1]]["allowed"] if l == 0 or fit[NAMES[l - 1]]["signed"] < S_FN]


def rank_desc_prob_then_dist(prob, dist):
    """order: carrier prob descending, ties by smaller distance, remaining ties by raster index (stable)."""
    o1 = np.argsort(dist, kind="stable")
    o2 = np.argsort(-prob[o1], kind="stable")
    return o1[o2]


# ---------------------------------------------------------------- operators
def vc_apply(T, Ca, Pr, fit, V):
    Y = T.copy()
    Tf, Caf, Prf = T.ravel(), Ca.ravel(), Pr.ravel()
    Yf = Y.ravel()
    for o in range(1, 14):
        To = T == o
        nT = int(To.sum())
        if nT == 0:
            continue
        f = fit[NAMES[o - 1]]
        if V["delta"][o - 1] > 0:
            Do = min(int(np.ceil(f["fn_d90"])), DCAP)
            d = edt_out(To)
            cand = (d > 0) & (d <= Do) & np.isin(T, sources(fit, o)) & (Ca == o)
            idx = np.flatnonzero(cand)
            B = int(round(float(V["delta"][o - 1]) * nT))
            if idx.size > B:
                idx = idx[rank_desc_prob_then_dist(Prf[idx], d.ravel()[idx])[:B]]
            Yf[idx] = o
        if V["eps"][o - 1] > 0:
            Ro = min(int(np.ceil(f["fp_d90"])), DCAP)
            di = edt_in(To)
            cand = To & (di <= Ro) & (Ca != o) & (Ca != 255)
            idx = np.flatnonzero(cand)
            E = int(round(float(V["eps"][o - 1]) * nT))
            if idx.size > E:
                idx = idx[np.argsort(-Prf[idx], kind="stable")[:E]]
            Yf[idx] = Caf[idx]
    return Y


def cor_apply(T, Ca, fit):
    """Plain COR (no cap): add L_o voxels where carrier == o; trim inner band of organs with s_o <= -0.3 where carrier
    disagrees (carrier label). Zones from unedited T, organs 1..13 in order."""
    Y = T.copy()
    for o in range(1, 14):
        To = T == o
        if not To.any():
            continue
        f = fit[NAMES[o - 1]]
        Do = min(int(np.ceil(f["fn_d90"])), DCAP)
        d = edt_out(To)
        L = (d > 0) & (d <= Do) & np.isin(T, sources(fit, o))
        Y[L & (Ca == o)] = o
        if f["signed"] <= S_TRIM:
            Ro = min(int(np.ceil(f["fp_d90"])), DCAP)
            band = To & (edt_in(To) <= Ro)
            r = band & (Ca != o) & (Ca != 255)
            Y[r] = Ca[r]
    return Y


def dice13(Y, G):
    out = []
    for o in range(1, 14):
        g, y = G == o, Y == o
        ng = int(g.sum())
        out.append(float("nan") if ng == 0 else 2.0 * int((g & y).sum()) / (ng + int(y.sum())))
    return out
