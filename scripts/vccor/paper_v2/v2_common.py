"""vccor/paper_v2 (blind-panel fix items 2 and 6, 2026-10-22): CORRECTED / POST-HOC re-analyses of looks already made, under
outputs/results/vccor/paper_tables_v2/analysis_conventions_v1.json (hash logged as a PREREG line in
outputs/logs/vccor/paper_v2.log BEFORE any number under these conventions was computed). CPU only, nice 15, <= 6 workers,
no GPU, no network inference: every arm is a STORED prediction map or a deterministic CPU post-processing of stored maps
(RECAL, MERGE). Existing results files are read only; every output of this block is new and lives in paper_tables_v2/.

This module: paths, the conventions guard, the v1 metric rules (1 mL presence on both sides, FP organs, one HD95 convention),
statistics (patient-cluster bootstrap, crossed two-level bootstrap via m2_cluster) and small helpers. No import-time side
effects beyond sys.path.
"""
import hashlib
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path("/home/user/Documents/Medical Imagining SSL")
HERE = Path(__file__).resolve().parent
for _p in ("scripts/vccor/xdata", "scripts/vccor/xdata/a3", "scripts/vccor/m2", "scripts/vccor/audit_ft",
           "scripts/vccor/tsft_full", "scripts/vccor/e2", "scripts/vccor/amos240"):
    if str(REPO / _p) not in sys.path:
        sys.path.append(str(REPO / _p))

VC = REPO / "outputs/results/vccor"
OUT = VC / "paper_tables_v2"
PERVOL = OUT / "pervol"
CONV = OUT / "analysis_conventions_v1.json"
LOGF = REPO / "outputs/logs/vccor/paper_v2.log"
LABEL = ("CORRECTED / POST-HOC re-analysis under analysis_conventions_v1 (looks already opened; not confirmatory; "
         "internally specified, not externally deposited)")
FLOOR_ML = 1.0
NBOOT = 10000
SEED_CLUSTER = 20261022
SEED_TWO_LEVEL = 20261020
MAX_WORKERS = 6


def now():
    return time.strftime("%F %T")


def sha256(p, n=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(n), b""):
            h.update(b)
    return h.hexdigest()


def rel(p):
    p = Path(p)
    try:
        return str(p.resolve().relative_to(REPO))
    except ValueError:
        return str(p)


def log(msg, also_print=True):
    LOGF.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{now()}] {msg}"
    with open(LOGF, "a") as fh:
        fh.write(line + "\n")
    if also_print:
        print(line, flush=True)


def _default(o):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def clean_nan(o):
    """JSON-safe copy: float nan / inf -> None (recursively)."""
    if isinstance(o, float):
        return None if not math.isfinite(o) else o
    if isinstance(o, dict):
        return {str(k): clean_nan(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean_nan(v) for v in o]
    try:
        import numpy as np
        if isinstance(o, np.ndarray):
            return clean_nan(o.tolist())
        if isinstance(o, np.floating):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
    except ImportError:
        pass
    return o


def write_json(path, obj):
    """Atomic write of a NEW output of this block (never used on a pre-existing results file of another block)."""
    path = Path(path)
    assert str(path.resolve()).startswith(str(OUT.resolve())), f"paper_v2 writes only under {OUT}: {path}"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with open(tmp, "w") as fh:
        json.dump(clean_nan(obj), fh, indent=1, default=_default)
    os.replace(tmp, path)


def write_text(path, txt):
    path = Path(path)
    assert str(path.resolve()).startswith(str(OUT.resolve())), f"paper_v2 writes only under {OUT}: {path}"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    open(tmp, "w").write(txt)
    os.replace(tmp, path)


def be_nice(target=15):
    try:
        cur = os.nice(0)
        if cur < target:
            os.nice(target - cur)
    except OSError:
        pass


def cap_threads():
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


# ---------------------------------------------------------------------------------------------------- conventions guard
def logged_prereg_sha(path=CONV):
    """The last 'PREREG sha256 <hex> <repo-relative path>' line of paper_v2.log for path."""
    want, h = rel(path), None
    if LOGF.is_file():
        for line in open(LOGF):
            p = line.split()
            if len(p) >= 6 and p[2] == "PREREG" and p[3] == "sha256" and p[5] == want:
                h = p[4]
    return h


def conventions():
    """Refuse unless the conventions file hashes to its logged PREREG line (no number is computed otherwise)."""
    if not CONV.is_file():
        raise SystemExit(f"[REFUSE] {rel(CONV)} missing: write the conventions (step 0) first")
    h, lg = sha256(CONV), logged_prereg_sha()
    if lg is None or h != lg:
        raise SystemExit(f"[REFUSE] {rel(CONV)} sha256 {h[:16]} != logged PREREG {str(lg)[:16]}")
    return json.load(open(CONV)), h


def log_code(paths, tag):
    for p in paths:
        log(f"CODE {sha256(p)} {rel(p)} ({tag})", also_print=False)


# ---------------------------------------------------------------------------------------------------- v1 metric rules
def nanmean(x, axis=None):
    import numpy as np
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmean(x, axis=axis)


def nanmedian(x, axis=None):
    import numpy as np
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmedian(x, axis=axis)


def joint_counts(Y, G, k):
    """(k+1) x (k+1) joint histogram [G, Y] (values 0..k)."""
    import numpy as np
    n = k + 1
    return np.bincount(G.astype(np.int64).ravel() * n + Y.astype(np.int64).ravel(), minlength=n * n).reshape(n, n)


def hd95_nsd(Y, G, o, sp):
    """The ONE surface convention (== a3_common.nsd_hd95_kd == xdata_common.nsd_hd95): surface = mask minus its 6-connected
    erosion; nearest surface-voxel-centre distances (KD-tree) in mm with the frame's spacing; HD95 = max of the two directed
    95th percentiles; NSD = share of surface points within 2 mm (x100). Both masks must be non-empty."""
    import a3_common as A3
    return A3.nsd_hd95_kd(Y, G, o, sp, tol=2.0)


def score_arm(Y, G, k, sp, vox_ml, diag, surface=True, H=None):
    """v1 per-organ record of one arm vs the reference (both label maps 0..k on the same grid).
    Presence: volume >= 1 mL on BOTH sides (a mask below 1 mL counts as empty for every metric).
      R & P: Dice from voxels, HD95 / NSD from surfaces;  R & ~P: Dice 0, NSD 0, HD95 = diag (missed organ);
      ~R & P: FP organ (Dice 0 / NSD 0 / HD95 = diag in the penalised statistics);  ~R & ~P: undefined.
    Also returns the raw voxel counts / voxel Dice (old present-only convention, GT > 0 voxels) for reproduction checks."""
    import numpy as np
    H = joint_counts(Y, G, k) if H is None else H
    g, y = H.sum(1)[1:].astype(np.int64), H.sum(0)[1:].astype(np.int64)
    inter = np.array([H[o, o] for o in range(1, k + 1)], np.int64)
    gml, yml = g * vox_ml, y * vox_ml
    R, P = gml >= FLOOR_ML, yml >= FLOOR_ML
    dice = np.full(k, np.nan)
    hd = np.full(k, np.nan)
    nsd = np.full(k, np.nan)
    status = []
    for j in range(k):
        if R[j] and P[j]:
            dice[j] = 200.0 * inter[j] / (g[j] + y[j])
            status.append("TP")
            if surface:
                n_, h_ = hd95_nsd(Y, G, j + 1, sp)
                nsd[j], hd[j] = n_, h_
        elif R[j]:
            dice[j], nsd[j], hd[j] = 0.0, 0.0, diag
            status.append("FN")
        elif P[j]:
            dice[j], nsd[j], hd[j] = 0.0, 0.0, diag
            status.append("FP")
        else:
            status.append("TN")
    dice_vox = [float(200.0 * inter[j] / (g[j] + y[j])) if g[j] > 0 else float("nan") for j in range(k)]
    return {"vox": y.tolist(), "inter": inter.tolist(), "ml": [float(v) for v in yml], "present": P.tolist(),
            "status": status, "dice": dice.tolist(), "hd95": hd.tolist(), "nsd": nsd.tolist(), "dice_vox_old": dice_vox}


def ref_record(G, k, vox_ml):
    import numpy as np
    g = bincount_chunked(G, 256)[1:k + 1].astype(np.int64)
    return {"vox": g.tolist(), "ml": [float(v * vox_ml) for v in g], "present": [bool(v * vox_ml >= FLOOR_ML) for v in g]}


def diag_mm(shape, sp):
    import numpy as np
    return float(np.sqrt(((np.asarray(shape[:3], float) * np.asarray(sp, float)) ** 2).sum()))


def bincount_chunked(A, minlength, step=24):
    """np.bincount of a (large) uint8 volume, slab by slab along axis 0 (bounded memory)."""
    import numpy as np
    out = np.zeros(minlength, np.int64)
    for i in range(0, A.shape[0], step):
        out += np.bincount(A[i:i + step].ravel(), minlength=minlength)[:minlength]
    return out


def joint_counts_chunked(Y, G, k, step=24):
    """(k+1) x (k+1) joint histogram [G, Y] slab by slab (== joint_counts)."""
    import numpy as np
    n = k + 1
    H = np.zeros(n * n, np.int64)
    for i in range(0, Y.shape[0], step):
        idx = G[i:i + step].astype(np.int32) * n + Y[i:i + step].astype(np.int32)
        H += np.bincount(idx.ravel(), minlength=n * n)
    return H.reshape(n, n)


def agreement_counts(Z, F, step=24):
    """Per TS class (0..255): voxels in Z (zero-shot), in F (fine-tuned) and in both (no reference involved)."""
    import numpy as np
    nz, nf, it = (np.zeros(256, np.int64) for _ in range(3))
    for i in range(0, Z.shape[0], step):
        a, b = Z[i:i + step], F[i:i + step]
        nz += np.bincount(a.ravel(), minlength=256)
        nf += np.bincount(b.ravel(), minlength=256)
        it += np.bincount(a[a == b].ravel(), minlength=256)
    return nz, nf, it


def _surface_points_lab(L, o, sl):
    """== a3_common._surface_points(L == o) using the label's find_objects slice (bbox padded by 2, 6-connected erosion)."""
    import numpy as np
    from scipy import ndimage
    lo = [max(s.start - 2, 0) for s in sl]
    hi = [min(s.stop + 2, n) for s, n in zip(sl, L.shape)]
    c = L[tuple(slice(a, b) for a, b in zip(lo, hi))] == o
    s = c & ~ndimage.binary_erosion(c)
    return np.argwhere(s) + np.asarray(lo)


class Scorer:
    """v1 scoring of many arms against ONE reference label map (values 0..k) on one grid, with the reference's surfaces
    cached per organ. score(Y) == score_arm(Y, G, ...) (HD95 / NSD == a3_common.nsd_hd95_kd)."""

    def __init__(self, G, k, sp, vox_ml=None, diag=None):
        import numpy as np
        from scipy import ndimage
        self.G, self.k, self.sp = G, int(k), np.asarray(sp, float)
        self.vox_ml = float(np.prod(self.sp)) / 1000.0 if vox_ml is None else float(vox_ml)
        self.diag = diag_mm(G.shape, self.sp) if diag is None else float(diag)
        self.ref = ref_record(G, self.k, self.vox_ml)
        self.slices = ndimage.find_objects(G, max_label=self.k)
        self._tree = {}

    def _ref_tree(self, o):
        from scipy.spatial import cKDTree
        if o not in self._tree:
            p = _surface_points_lab(self.G, o, self.slices[o - 1]) * self.sp
            self._tree[o] = (p, cKDTree(p))
        return self._tree[o]

    def score(self, Y, surface=True, tol=2.0):
        import numpy as np
        from scipy import ndimage
        from scipy.spatial import cKDTree
        k = self.k
        assert Y.shape == self.G.shape, (Y.shape, self.G.shape)
        H = joint_counts_chunked(Y, self.G, k)
        g, y = H.sum(1)[1:].astype(np.int64), H.sum(0)[1:].astype(np.int64)
        inter = np.array([H[o, o] for o in range(1, k + 1)], np.int64)
        gml, yml = g * self.vox_ml, y * self.vox_ml
        R, P = gml >= FLOOR_ML, yml >= FLOOR_ML
        dice, hd, nsd = np.full(k, np.nan), np.full(k, np.nan), np.full(k, np.nan)
        status = []
        ysl = ndimage.find_objects(Y, max_label=k) if surface else None
        for j in range(k):
            o = j + 1
            if R[j] and P[j]:
                dice[j] = 200.0 * inter[j] / (g[j] + y[j])
                status.append("TP")
                if surface:
                    pb, tb = self._ref_tree(o)
                    pa = _surface_points_lab(Y, o, ysl[j]) * self.sp
                    d_ab = tb.query(pa, k=1)[0]
                    d_ba = cKDTree(pa).query(pb, k=1)[0]
                    nsd[j] = 100.0 * float(((d_ab <= tol).sum() + (d_ba <= tol).sum()) / (len(d_ab) + len(d_ba)))
                    hd[j] = float(max(np.percentile(d_ab, 95), np.percentile(d_ba, 95)))
            elif R[j] or P[j]:
                dice[j], nsd[j], hd[j] = 0.0, 0.0, self.diag
                status.append("FN" if R[j] else "FP")
            else:
                status.append("TN")
        dice_vox = [float(200.0 * inter[j] / (g[j] + y[j])) if g[j] > 0 else float("nan") for j in range(k)]
        return {"vox": y.tolist(), "inter": inter.tolist(), "ml": [float(v) for v in yml], "present": P.tolist(),
                "status": status, "dice": dice.tolist(), "hd95": hd.tolist(), "nsd": nsd.tolist(), "surface": bool(surface),
                "dice_vox_old": dice_vox}

    def frame_record(self, extra=None):
        r = {"shape": list(self.G.shape), "spacing": [float(s) for s in self.sp], "vox_ml": self.vox_ml, "diag_mm": self.diag,
             "ref": self.ref, "arms": {}}
        if extra:
            r.update(extra)
        return r


# ---------------------------------------------------------------------------------------------------- item 6 operators
def merge_maps(Z, F, isT):
    """Class-wise merge of two TS-id maps (conventions item6.class_wise_merge). isT: bool LUT (256) of the site's classes.
    Returns (MERGE_T, MERGE_N)."""
    zt, ft = isT[Z], isT[F]
    base = Z.copy()
    base[zt] = 0
    mt = base.copy()
    mt[ft] = F[ft]
    w = ft & ((Z == 0) | zt)
    mn = base
    mn[w] = F[w]
    return mt, mn


def recal_map(Y, k, sp, scales, pad_mm=40.0, pad_try_mm=10.0):
    """RECAL (conventions item6.recalibration_baseline) on a dataset-label map Y (0..k) of F0 on the native grid.
    GROW: F0 background voxels within d_out <= t (t among the distinct d_out values <= pad_mm, or none) closest to the
    target; TRIM: remove organ voxels with d_in <= t (never all). Conflicts between growing organs: smallest d_out, then the
    lower organ id. The growth search first uses a crop padded by pad_try_mm and falls back to pad_mm when the target lies
    beyond it -- identical result (every voxel with d_out <= t lies within t mm of the organ's bounding box).
    Returns (Y2, info)."""
    import math as _m
    import numpy as np
    from scipy import ndimage
    sp = np.asarray(sp, float)
    vox_ml = float(np.prod(sp)) / 1000.0
    Y2 = Y.copy()
    objs = ndimage.find_objects(Y, max_label=k)
    cnt = bincount_chunked(Y, 256)
    claims, info = [], {}
    for o in range(1, k + 1):
        s = float(scales[o - 1])
        sl = objs[o - 1]
        n0 = int(cnt[o])
        rec = {"scale": s, "f0_vox": n0, "target_vox": s * n0}
        if sl is None or n0 * vox_ml < FLOOR_ML or s == 1.0:
            rec["action"] = "none (" + ("absent" if sl is None else ("F0 < 1 mL" if n0 * vox_ml < FLOOR_ML else "s = 1")) + ")"
            info[o] = rec
            continue
        if s > 1.0:
            need = (s - 1.0) * n0
            done = None
            for pmm in (pad_try_mm, pad_mm):
                pad = [int(_m.ceil(pmm / sp[i])) + 1 for i in range(3)]
                lo = [max(sl[i].start - pad[i], 0) for i in range(3)]
                hi = [min(sl[i].stop + pad[i], Y.shape[i]) for i in range(3)]
                cr = tuple(slice(a, b) for a, b in zip(lo, hi))
                Yc = Y[cr]
                d = ndimage.distance_transform_edt(Yc != o, sampling=sp)
                cand = (Yc == 0) & (d <= pmm)
                lev, c = np.unique(d[cand], return_counts=True)
                cum = np.cumsum(c)
                if (len(cum) and cum[-1] >= need) or pmm == pad_mm:
                    done = (cr, lo, d, cand, lev, cum, pmm)
                    break
            cr, lo, d, cand, lev, cum, pmm = done
            opts = np.concatenate([[0], cum]).astype(float)
            best = int(np.argmin(np.abs(opts - need)))
            rec.update(action="grow", search_pad_mm=pmm, capped=bool(best == len(opts) - 1 and opts[-1] < need))
            if best > 0:
                t = float(lev[best - 1])
                sel = cand & (d <= t)
                ii = np.nonzero(sel)
                gidx = np.ravel_multi_index(tuple(ii[a] + lo[a] for a in range(3)), Y.shape)
                claims.append((gidx, d[sel], np.full(len(gidx), o, np.int16)))
                rec.update(t_mm=t, claimed_vox=int(len(gidx)))
            else:
                rec.update(t_mm=None, claimed_vox=0)
        else:
            need = (1.0 - s) * n0
            lo = [max(sl[i].start - 2, 0) for i in range(3)]
            hi = [min(sl[i].stop + 2, Y.shape[i]) for i in range(3)]
            cr = tuple(slice(a, b) for a, b in zip(lo, hi))
            M = Y[cr] == o
            d = ndimage.distance_transform_edt(M, sampling=sp)
            lev, c = np.unique(d[M], return_counts=True)
            cum = np.cumsum(c)
            opts = np.concatenate([[0], cum[:-1]]).astype(float)          # never remove every voxel
            best = int(np.argmin(np.abs(opts - need)))
            rec.update(action="trim", capped=bool(best == len(opts) - 1 and opts[-1] < need))
            if best > 0:
                t = float(lev[best - 1])
                rem = M & (d <= t)
                view = Y2[cr]
                view[rem] = 0
                rec.update(t_mm=t, removed_vox=int(rem.sum()))
            else:
                rec.update(t_mm=None, removed_vox=0)
        info[o] = rec
    if claims:
        gi = np.concatenate([c[0] for c in claims])
        gd = np.concatenate([c[1] for c in claims])
        go = np.concatenate([c[2] for c in claims])
        order = np.lexsort((go, gd, gi))
        gi, go = gi[order], go[order]
        first = np.r_[True, gi[1:] != gi[:-1]]
        flat = Y2.reshape(-1)
        flat[gi[first]] = go[first].astype(Y2.dtype)
    cnt2 = bincount_chunked(Y2, 256)
    for o, rec in info.items():
        rec["achieved_vox"] = int(cnt2[o])
        rec["achieved_ratio"] = float(cnt2[o] / rec["f0_vox"]) if rec["f0_vox"] else None
    return Y2, info


# ---------------------------------------------------------------------------------------------------- derived per-pair values
def pair_values(rec, arm, j):
    """(status, dice_pen, dice_refpresent, hd95_pen, v_ref_ml (floored), v_pred_ml (floored), sve) for organ j."""
    ref = rec["ref"]
    a = rec["arms"][arm]
    st = a["status"][j]
    vr = ref["ml"][j] if ref["present"][j] else 0.0
    vp = a["ml"][j] if a["present"][j] else 0.0
    d = a["dice"][j]
    dref = d if st in ("TP", "FN") else float("nan")
    sve = 100.0 * (vp - vr) / vr if ref["present"][j] else float("nan")
    return st, d, dref, a["hd95"][j], vr, vp, sve


# ---------------------------------------------------------------------------------------------------- statistics
def t_ci95(d):
    import numpy as np
    from scipy import stats
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return [None, None]
    se = d.std(ddof=1) / np.sqrt(len(d))
    q = stats.t.ppf(0.975, len(d) - 1)
    return [float(d.mean() - q * se), float(d.mean() + q * se)]


def cluster_boot(values, clusters, n_boot=NBOOT, seed=SEED_CLUSTER, stat="mean"):
    """Percentile bootstrap CI (95%) of the mean over scans of `values` (nan = no pair) when the clusters (patients) are
    resampled with replacement; with singleton clusters it is the ordinary scan bootstrap. Returns [lo, hi]."""
    import numpy as np
    v = np.asarray(values, float)
    c = np.asarray(clusters)
    ok = np.isfinite(v)
    v, c = v[ok], c[ok]
    if len(v) == 0:
        return [None, None]
    uc, inv = np.unique(c, return_inverse=True)
    S = np.bincount(inv, weights=v, minlength=len(uc))
    N = np.bincount(inv, minlength=len(uc)).astype(float)
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    m = len(uc)
    for b0 in range(0, n_boot, 2000):
        b1 = min(n_boot, b0 + 2000)
        W = rng.multinomial(m, np.full(m, 1.0 / m), size=b1 - b0).astype(float)
        num, den = W @ S, W @ N
        with np.errstate(invalid="ignore", divide="ignore"):
            out[b0:b1] = num / den
    out = out[np.isfinite(out)]
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def paired_summary(d, clusters):
    """Mean paired difference, patient-cluster bootstrap CI (primary), scan-level t CI (descriptive), scans up / down."""
    import numpy as np
    d = np.asarray(d, float)
    ok = np.isfinite(d)
    dd = d[ok]
    cl = np.asarray(clusters)[ok]
    return {"n": int(len(dd)), "n_clusters": int(len(set(cl.tolist()))), "mean": float(dd.mean()) if len(dd) else None,
            "cluster_boot_ci95": cluster_boot(dd, cl), "t_ci95": t_ci95(dd), "up": int((dd > 0).sum()),
            "down": int((dd < 0).sum()), "zero": int((dd == 0).sum())}


def two_level(per_draw):
    """FLARE multi-draw contrast: {draw: {cid: value}} -> m2_cluster.clustered (crossed scan x draw bootstrap)."""
    import m2_cluster as MC
    clean = {s: {c: v for c, v in d.items() if v is not None and math.isfinite(v)} for s, d in per_draw.items()}
    clean = {s: d for s, d in clean.items() if len(d) >= 1}
    if not clean:
        return None
    draws = sorted(clean)
    cl = MC.clustered(clean, draws, NBOOT, SEED_TWO_LEVEL, keep_per_scan=False)
    dm = {s: float(sum(d.values()) / len(d)) for s, d in clean.items()}
    return {"M": cl["M_mean_of_draw_means"], "two_level_ci95": cl["boot2_ci95"], "scan_cluster_ci95": cl["scan_cluster_boot_ci95"],
            "draw_level_t_ci95": cl["draw_level_t_ci95"], "draw_means": dm, "n_scans": cl["n_scans"], "n_pairs": cl["n_pairs"],
            "scans_up": cl["scans_up"], "scans_down": cl["scans_down"]}


def excludes0(ci):
    return bool(ci and ci[0] is not None and (ci[0] > 0 or ci[1] < 0))


def fmt(v, nd=2, sign=True):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "-"
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def fci(ci, nd=2):
    return "-" if not ci or ci[0] is None else f"[{fmt(ci[0], nd)}, {fmt(ci[1], nd)}]"
