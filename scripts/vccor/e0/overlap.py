"""vccor/e0 STEP 3: image-overlap (near-duplicate) check, IMAGES ONLY (no 'label' dataset is ever opened).

Sets: FLARE 50 raw images; MACT 90 MACT_h5 'image'; BTCV raw Training 30 + Testing 20 (positive control vs MACT);
AMOS 60 raw images of the amos_ct_h5 set (extra). Protocol = prereg.json 'overlap_check':
  LPS -> window [-125, 275] HU (window units 0..1; MACT_h5 = z-scored window, mapped back by its min/max) -> Gaussian
  anti-alias -> linear zoom to 8 mm (stage 1, all pairs) / 4 mm (stage 2, union of every volume's top-2 candidates)
  -> masked NCC (Padfield 2012, FFT) over all integer translations, overlap >= 40 % of the smaller volume.
  near-duplicate iff NCC4 >= 0.90. Header fingerprints (raw shape + spacing) reported alongside.

usage: CUDA_VISIBLE_DEVICES="" nice -n 15 .venv/bin/python -B scripts/vccor/e0/overlap.py [--cache DIR]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e0_common as E  # noqa: E402
import numpy as np  # noqa: E402

LO_HU, HI_HU = -125.0, 275.0
MIN_OVERLAP = 0.40
NEAR_DUP = 0.90
TOPK = 2
FLARE_RAW = E.REPO / "data/FLARE22Train/images"
BTCV_RAW = Path("/media/user/B4864CD4864C98AE/medssl_data/BTCV_raw/RawData")
OUTD = E.OUT / "overlap"
LOG = E.Log(E.LOGD / "e0_overlap.log")


def volumes():
    sys.path.insert(0, str(E.REPO / "scripts/vccor/xdata"))
    import xdata_common as X
    V = {}
    for p in sorted(FLARE_RAW.glob("FLARE22_Tr_*_0000.nii.gz")):
        V[f"FLARE:{p.name[11:15]}"] = dict(kind="raw", path=str(p))
    for i in range(1, 91):
        V[f"MACT:{i:04d}"] = dict(kind="mact_h5", path=str(E.DATA / "MACT_h5" / f"{i:04d}.h5"))
    for sub in ("Training", "Testing"):
        for p in sorted((BTCV_RAW / sub / "img").glob("img*.nii.gz")):
            V[f"BTCV:{p.name[3:7]}"] = dict(kind="raw", path=str(p), subset=sub)
    for c in sorted(q.stem for q in (E.DATA / "amos_ct_h5").glob("*.h5")):
        V[f"AMOS:{c}"] = dict(kind="raw", path=str(X.amos_raw_img(c)))
    return V, X


def lps_raw(path):
    import nibabel as nib
    img = nib.load(path)
    tr = nib.orientations.ornt_transform(nib.orientations.io_orientation(img.affine),
                                         nib.orientations.axcodes2ornt(("L", "P", "S")))
    arr = nib.orientations.apply_orientation(np.asarray(img.dataobj, dtype=np.float32), tr)
    aff = img.affine @ nib.orientations.inv_ornt_aff(tr, img.shape[:3])
    sp = np.sqrt((aff[:3, :3] ** 2).sum(0))
    return arr, sp, tuple(int(x) for x in img.shape[:3]), [float(x) for x in np.sqrt((img.affine[:3, :3] ** 2).sum(0))]


def window_units(item):
    """full-res array in window units 0..1 (LPS) + spacing + header info."""
    import h5py
    if item["kind"] == "raw":
        arr, sp, shp, sp_native = lps_raw(item["path"])
        w = (np.clip(arr, LO_HU, HI_HU) - LO_HU) / (HI_HU - LO_HU)
        return w.astype(np.float32), sp, dict(raw_shape=shp, raw_spacing=sp_native)
    with h5py.File(item["path"], "r") as f:
        z = f["image"][:]                                   # ONLY the image dataset
    zmin, zmax = float(z.min()), float(z.max())
    fr_lo, fr_hi = float((z <= zmin + 1e-9).mean()), float((z >= zmax - 1e-9).mean())
    w = ((z - zmin) / (zmax - zmin)).astype(np.float32)
    return w, np.array([1.0, 1.0, 1.0]), dict(h5_shape=tuple(int(x) for x in z.shape), frac_at_min=fr_lo,
                                               frac_at_max=fr_hi)


def coarse(w, sp, t):
    from scipy import ndimage as ndi
    f = sp / t
    sig = np.maximum(0.5 * (1.0 / f - 1.0), 0.0)
    g = ndi.gaussian_filter(w, sigma=sig, mode="nearest") if np.any(sig > 0) else w
    return ndi.zoom(g, f, order=1, mode="nearest", grid_mode=False).astype(np.float32)


def prep_one(args):
    key, item, cache = args
    p8, p4 = Path(cache) / f"{key.replace(':', '_')}_8.npy", Path(cache) / f"{key.replace(':', '_')}_4.npy"
    pj = Path(cache) / f"{key.replace(':', '_')}.json"
    if p8.is_file() and p4.is_file() and pj.is_file():
        return key, json.load(open(pj))
    t0 = time.time()
    w, sp, info = window_units(item)
    c8, c4 = coarse(w, sp, 8.0), coarse(w, sp, 4.0)
    hist = np.histogram(w, bins=40, range=(0.0, 1.0))[0].tolist()
    np.save(p8, c8)
    np.save(p4, c4)
    info.update(lps_spacing=[float(x) for x in sp], fullres_shape=[int(x) for x in w.shape], shape8=list(c8.shape),
                shape4=list(c4.shape), hist40_fullres=hist, sec=round(time.time() - t0, 1))
    json.dump(info, open(pj, "w"))
    return key, info


def zs(a):
    a = a.astype(np.float64)
    return (a - a.mean()) / (a.std() + 1e-12)


class Spec:
    """One volume for masked NCC at a fixed FFT shape: z-scored array (box sums) + its rFFT (cross term)."""

    def __init__(self, a, shape, dtype):
        from scipy import fft
        self.n = a.size
        self.shape_v = a.shape
        self.v = zs(a)                                           # float64 for the box sums
        self.F = fft.rfftn(self.v.astype(dtype), s=shape, workers=E.N_THREADS)


def _intervals(na, nb, S):
    """per circular index x of one axis: overlap interval in A coords and in B coords (empty if no overlap)."""
    x = np.arange(S)
    u = np.where(x <= na - 1, x, x - S)                         # signed shift (A(x+u) vs B(x))
    loA, hiA = np.maximum(u, 0), np.minimum(nb + u, na)
    loB, hiB = np.maximum(0, -u), np.minimum(nb, na - u)
    bad = (hiA <= loA) | (x > na - 1) & (x < S - (nb - 1))
    for arr in (loA, hiA, loB, hiB):
        arr[bad] = 0
    return loA, hiA, loB, hiB


def _boxsum(Y, lohi):
    """sum of Y over the per-axis intervals, for every (x0, x1, x2) -> array (S0, S1, S2) (separable cumsums)."""
    out = Y
    for d, (lo, hi) in enumerate(lohi):
        C = np.cumsum(out, axis=d)
        C = np.concatenate([np.zeros_like(np.take(C, [0], axis=d)), C], axis=d)
        out = np.take(C, hi, axis=d) - np.take(C, lo, axis=d)
    return out


FFT_WORKERS = E.N_THREADS     # set to 1 inside pool workers


def masked_ncc(A, B, shape, min_frac=MIN_OVERLAP):
    """max masked NCC (Padfield 2012 with box masks) over all translations; corr(u) = sum_x A(x+u) B(x).
    Exact pruning: a shift can reach overlap >= min_frac * min(|A|, |B|) only if along every axis d its overlap length
    L_d >= min_frac * min(|A|, |B|) / prod_{d' != d} max L_d'; the statistics are evaluated on that sub-grid only."""
    from scipy import fft
    iv = [_intervals(na, nb, S) for na, nb, S in zip(A.shape_v, B.shape_v, shape)]
    L = [(t[1] - t[0]).astype(np.float64) for t in iv]
    need = min_frac * min(A.n, B.n)
    mx = [l.max() for l in L]
    keep = [np.nonzero(L[d] >= need / np.prod([mx[e] for e in range(3) if e != d]) - 1e-9)[0] for d in range(3)]
    if any(k.size == 0 for k in keep):
        return -1.0, [0, 0, 0], 0.0
    ivk = [tuple(a[k] for a in t) for t, k in zip(iv, keep)]
    n = np.einsum("i,j,k->ijk", *[(t[1] - t[0]).astype(np.float64) for t in ivk])
    SA = _boxsum(A.v, [(t[0], t[1]) for t in ivk])
    SA2 = _boxsum(A.v * A.v, [(t[0], t[1]) for t in ivk])
    SB = _boxsum(B.v, [(t[2], t[3]) for t in ivk])
    SB2 = _boxsum(B.v * B.v, [(t[2], t[3]) for t in ivk])
    SAB = fft.irfftn(A.F * np.conj(B.F), s=shape, workers=FFT_WORKERS)[np.ix_(*keep)]
    ok = n >= need
    nn = np.where(ok, n, 1.0)
    num = SAB - SA * SB / nn
    va = np.maximum(SA2 - SA * SA / nn, 0.0)
    vb = np.maximum(SB2 - SB * SB / nn, 0.0)
    den = np.sqrt(va * vb)
    ok &= den > 1e-6 * nn
    if not ok.any():
        return -1.0, [0, 0, 0], 0.0
    ncc = np.where(ok, num / np.where(ok, den, 1.0), -1.0)
    i = int(np.argmax(ncc))
    ii = np.unravel_index(i, ncc.shape)
    u = [int(keep[d][ii[d]]) for d in range(3)]
    # circular index -> signed shift: valid u in [0, nA-1] (positive) or [S-(nB-1), S-1] (negative); S >= nA+nB-1
    shift = [int(x) if x <= na - 1 else int(x - s) for x, s, na in zip(u, shape, A.shape_v)]
    return float(ncc.flat[i]), shift, float(n.flat[i] / min(A.n, B.n))


def fast_shape(shapes_a, shapes_b):
    from scipy import fft
    ma, mb = np.max(shapes_a, 0), np.max(shapes_b, 0)
    return tuple(int(fft.next_fast_len(int(x + y - 1), real=True)) for x, y in zip(ma, mb))


def overlap_hist_corr(a, b, shift):
    """intensity-histogram correlation (40 bins, window units) inside the overlap of a (shifted by shift) and b (4 mm)."""
    sl_a, sl_b = [], []
    for u, na, nb in zip(shift, a.shape, b.shape):
        x0, x1 = max(0, -u), min(nb, na - u)            # b coords
        sl_b.append(slice(x0, x1))
        sl_a.append(slice(x0 + u, x1 + u))
    pa, pb = a[tuple(sl_a)], b[tuple(sl_b)]
    if pa.size < 10:
        return None
    ha = np.histogram(np.clip(pa, 0, 1), bins=40, range=(0, 1))[0].astype(float)
    hb = np.histogram(np.clip(pb, 0, 1), bins=40, range=(0, 1))[0].astype(float)
    return float(np.corrcoef(ha, hb)[0, 1])


_G = {}                       # fork-inherited state of the pool workers


def _init_worker():
    global FFT_WORKERS
    FFT_WORKERS = 1


def _row_worker(i):
    A = _G["SA"][i]
    return i, [masked_ncc(A, B, _G["shp"])[0] for B in _G["SB"]]


def _load(cache, k, r):
    return np.load(Path(cache) / f"{k.replace(':', '_')}_{r}.npy")


def _pair_worker(args):
    ka, kb, cache = args
    a4, b4 = _load(cache, ka, 4), _load(cache, kb, 4)
    shp4 = fast_shape([a4.shape], [b4.shape])
    ncc4, shift, ov = masked_ncc(Spec(a4, shp4, np.float64), Spec(b4, shp4, np.float64), shp4)
    return dict(a=ka, b=kb, ncc4=ncc4, shift4_vox=shift, overlap_frac=ov,
                hist_corr_overlap=overlap_hist_corr(a4, b4, shift))


def compare(name, keysA, keysB, cache, LOG, pairs_extra=()):
    """stage 1 (8 mm, all pairs) + stage 2 (4 mm, union of every volume's top-2)."""
    from multiprocessing import Pool
    t0 = time.time()
    A8 = [_load(cache, k, 8) for k in keysA]
    B8 = [_load(cache, k, 8) for k in keysB]
    shp = fast_shape([a.shape for a in A8], [b.shape for b in B8])
    _G.update(SA=[Spec(a, shp, np.float32) for a in A8], SB=[Spec(b, shp, np.float32) for b in B8], shp=shp)
    S1 = np.full((len(keysA), len(keysB)), -1.0)
    with Pool(E.N_THREADS, initializer=_init_worker) as p:
        for c, (i, row) in enumerate(p.imap_unordered(_row_worker, range(len(keysA)), chunksize=1)):
            S1[i] = row
            if (c + 1) % 10 == 0:
                LOG(f"  [{name}] stage1 {c + 1}/{len(keysA)} rows ({time.time() - t0:.0f}s)")
    _G.clear()
    cand = set()
    for i in range(len(keysA)):
        for j in np.argsort(-S1[i])[:TOPK]:
            cand.add((i, int(j)))
    for j in range(len(keysB)):
        for i in np.argsort(-S1[:, j])[:TOPK]:
            cand.add((int(i), j))
    for ka, kb in pairs_extra:
        cand.add((keysA.index(ka), keysB.index(kb)))
    cand = sorted(cand)
    rows = []
    with Pool(E.N_THREADS, initializer=_init_worker) as p:
        for r in p.imap(_pair_worker, [(keysA[i], keysB[j], cache) for i, j in cand], chunksize=1):
            rows.append(r)
    for (i, j), r in zip(cand, rows):
        r.update(ncc8=float(S1[i, j]), rank8_in_row=int((S1[i] > S1[i, j]).sum()) + 1,
                 rank8_in_col=int((S1[:, j] > S1[i, j]).sum()) + 1)
    LOG(f"  [{name}] {len(keysA)}x{len(keysB)} stage1 + {len(rows)} stage-2 pairs in {time.time() - t0:.0f}s "
        f"(fft {shp})")
    return S1, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(Path("/tmp/claude-1000/-home-user-Documents-Medical-Imagining-SSL/"
                                                "e0eae168-45fa-47e8-b092-dccd1465a55b/scratchpad/e0/overlap_cache")))
    ap.add_argument("--only", nargs="*", default=None, help="subset of comparisons")
    args = ap.parse_args()
    E.be_nice()
    Path(args.cache).mkdir(parents=True, exist_ok=True)
    OUTD.mkdir(parents=True, exist_ok=True)
    V, X = volumes()
    LOG(f"overlap start: {len(V)} volumes; cache {args.cache}")
    from multiprocessing import Pool
    import os
    t0 = time.time()
    os.environ["OMP_NUM_THREADS"] = "1"
    info = {}
    with Pool(E.N_THREADS) as p:
        for k, inf in p.imap_unordered(prep_one, [(k, it, args.cache) for k, it in V.items()], chunksize=1):
            info[k] = inf
    LOG(f"  fingerprints ready ({time.time() - t0:.0f}s)")
    # header fingerprints of MACT raw CT (BTCV raw / downloaded TCIA NIfTI headers; else TCIA slice count)
    import nibabel as nib
    S = X.mact_sources()
    mhead = {}
    for mid, s in S.items():
        rec = dict(source=s["source"], original_id=s["original_id"], btcv_subset=s["btcv_subset"],
                   withdrawn=bool(s["withdrawn"]))
        if Path(s["raw_ct"]).is_file():
            im = nib.load(s["raw_ct"])
            rec.update(raw_shape=[int(x) for x in im.shape[:3]],
                       raw_spacing=[float(x) for x in np.sqrt((im.affine[:3, :3] ** 2).sum(0))])
        elif s["source"] == "tcia":
            t = X.tcia_series_uid(s["original_id"])
            rec.update(tcia_image_count=t["ImageCount"] if t else None)
        mhead[mid] = rec
    E.write_json(OUTD / "fingerprints.json", dict(volumes=info, mact_raw_headers=mhead, written=E.now()))

    def hdr_matches(keysA):
        out = []
        for ka in keysA:
            ia = info[ka]
            for mid, h in mhead.items():
                if "raw_shape" in h:
                    same = ia["raw_shape"] == h["raw_shape"] and np.allclose(ia["raw_spacing"], h["raw_spacing"], atol=1e-3)
                    if same:
                        out.append(dict(a=ka, mact=mid, kind="shape+spacing", raw_shape=h["raw_shape"],
                                        raw_spacing=h["raw_spacing"]))
                elif h.get("tcia_image_count") is not None and ia["raw_shape"][2] == h["tcia_image_count"]:
                    out.append(dict(a=ka, mact=mid, kind="slice count only (TCIA header not downloaded)",
                                    n_slices=h["tcia_image_count"]))
        return out

    keys = {s: [k for k in V if k.startswith(s + ":")] for s in ("FLARE", "MACT", "BTCV", "AMOS")}
    comps = [("FLARE_vs_MACT", keys["FLARE"], keys["MACT"], ()),
             ("BTCV_vs_MACT", keys["BTCV"], keys["MACT"], ()),
             ("FLARE_vs_AMOS", keys["FLARE"], keys["AMOS"], ()),
             ("AMOS_vs_MACT", keys["AMOS"], keys["MACT"], ()),
             ("MACT_0001_vs_0022", ["MACT:0001"], ["MACT:0022"], ())]
    res = {}
    for name, ka, kb, extra in comps:
        if args.only and name not in args.only:
            continue
        S1, rows = compare(name, ka, kb, args.cache, LOG, extra)
        res[name] = dict(n_a=len(ka), n_b=len(kb), stage1_ncc8=dict(rows=ka, cols=kb, M=np.round(S1, 4).tolist()),
                         stage2=sorted(rows, key=lambda r: -r["ncc4"]),
                         near_duplicates=[r for r in sorted(rows, key=lambda r: -r["ncc4"]) if r["ncc4"] >= NEAR_DUP])
        if name.startswith("FLARE") or name.startswith("BTCV") or name.startswith("AMOS_vs"):
            res[name]["header_matches_vs_mact"] = hdr_matches(ka) if name.endswith("MACT") else None
        E.write_json(OUTD / "pairs.json", dict(results=res, written=E.now()))
        nd = res[name]["near_duplicates"]
        LOG(f"  [{name}] near-duplicates (NCC4 >= {NEAR_DUP}): {len(nd)}; top NCC4: "
            + ", ".join(f"{r['a']}~{r['b']} {r['ncc4']:.3f}" for r in res[name]["stage2"][:5]))
    LOG("overlap done")


if __name__ == "__main__":
    main()
