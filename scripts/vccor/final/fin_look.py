"""vccor/final LOOK ENGINE (CPU; shared by the looks of plans A-C): per-scan v1 scoring of stored raw prediction maps against
one reference standard in the conventions' frames, and the aggregate statistics every plan pre-specifies.

Frames (analysis_conventions_v1): flare / amos / btcv / word = 'NATIVE' (the raw CT grid, full field of view, the
reference file's own spacing); mact = 'NATIVE_BOX' (PRIMARY: raw grid reoriented to LPS, inside Gibson's box, vs the Zenodo
label; n = 77 verified), 'NATIVE_FULL' (whole LPS raw grid; descriptive) and 'H5' (the sealed look's registered 1 mm frame vs
MagicNet MACT_h5 'label'; predictions via xdata_common.mact_raw_to_h5 == every MACT scorer).
Every arm map is a stored raw map in TS 'total' ids (lut 'ts') or in the dataset's own ids (lut 'id', STOCK models).
References are read only after the look's GT_OPENED marker exists (final_common.require_marker).
No import-time side effects beyond sys.path.
"""
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import final_common as F  # noqa: E402

FRAMES = {"flare": ["NATIVE"], "amos": ["NATIVE"], "btcv": ["NATIVE"], "word": ["NATIVE"],
          "mact": ["NATIVE_BOX", "NATIVE_FULL", "H5"]}
PRIMARY_FRAME = {"flare": "NATIVE", "amos": "NATIVE", "btcv": "NATIVE", "word": "NATIVE", "mact": "NATIVE_BOX"}
_LUTS = {}


def lut(ds, kind):
    import numpy as np
    key = (ds, kind)
    if key not in _LUTS:
        if kind == "ts":
            _LUTS[key] = F.ts_lut(ds) if ds != "word" else __import__("word_common").ts_lut()
        elif kind == "id":
            _LUTS[key] = F.identity_lut(F.n_org(ds) if ds != "word" else __import__("word_common").N_ORG)
        elif kind == "vista":
            if ds == "word":
                _LUTS[key] = __import__("word_common").vista_lut()
            else:
                sys.path.insert(0, str(F.REPO / "scripts/vccor/vista"))
                _LUTS[key] = __import__("vista_common").vista_lut(ds)        # frozen vista block LUT (by name, cysts -> kidney)
        elif kind == "mrseg":
            if ds == "word":
                _LUTS[key] = __import__("word_common").mrseg_lut()
            else:
                _LUTS[key] = __import__("m2_common").mrseg_lut(ds)            # frozen m2 block LUT (by name)
        else:
            raise KeyError(kind)
        assert _LUTS[key].dtype == np.uint8 and _LUTS[key].shape == (256,)
    return _LUTS[key]


def read_raw(path):
    import nibabel as nib
    import numpy as np
    im = nib.load(str(path))
    a = np.asarray(im.dataobj)
    assert a.min() >= 0 and a.max() <= 255, path
    return a.astype(np.uint8), im


def n_org(ds):
    if ds == "word":
        return __import__("word_common").N_ORG
    return F.n_org(ds)


def references(ds, cid, marker, frames=None):
    """{frame: (G, spacing)} + context for mapping arms. Reads the reference ONLY under an existing GT_OPENED marker."""
    import numpy as np
    frames = frames or FRAMES[ds]
    ctx = {"ds": ds, "cid": cid}
    out = {}
    if ds == "flare":
        G, sp = F.flare_ref(cid, marker)
        out["NATIVE"] = (G, sp)
    elif ds == "amos":
        G, sp = F.amos_ref(cid, marker)
        out["NATIVE"] = (G, sp)
    elif ds == "btcv":
        G, sp = F.btcv_ref(cid, marker)
        out["NATIVE"] = (G, sp)
    elif ds == "word":
        import word_common as W
        G, sp = W.reference(cid, marker)
        out["NATIVE"] = (G, sp)
    elif ds == "mact":
        import h5py
        import xdata_common as X
        nat = F.mact_native(cid, marker)
        if nat is not None:
            fr, Z, box, sp, img = nat
            ctx.update(fr=fr, box=box, img=img)
            if "NATIVE_BOX" in frames:
                out["NATIVE_BOX"] = (np.ascontiguousarray(Z[box]), sp)
            if "NATIVE_FULL" in frames:
                out["NATIVE_FULL"] = (Z, sp)
        else:
            import nibabel as nib
            ctx.update(img=nib.load(str(X.raw_img("mact", cid))), unverified=True)
        if "H5" in frames:
            F.require_marker(marker)
            with h5py.File(X.MACT_H5 / f"{cid}.h5", "r") as f:
                g = f["label"][:]
            gi = np.rint(g)
            assert np.array_equal(gi, g) and gi.min() >= 0 and gi.max() <= 8, cid
            out["H5"] = (gi.astype(np.uint8), (1.0, 1.0, 1.0))
    else:
        raise KeyError(ds)
    return out, ctx


def arm_in_frame(ds, frame, seg_ds, ctx):
    """An arm's dataset-id map on the raw grid -> the frame's grid."""
    import numpy as np
    if ds != "mact" or frame == "NATIVE":
        return seg_ds
    import xdata_common as X
    if frame == "H5":
        return X.mact_raw_to_h5(seg_ds, ctx["cid"], ctx["img"])
    L = F.mact_pred_lps(seg_ds, ctx["fr"])
    return np.ascontiguousarray(L[ctx["box"]]) if frame == "NATIVE_BOX" else L


def score_scan(ds, cid, arms, marker, shas=None, frames=None, surface=True):
    """arms {arm: (raw path, lut kind)}. shas {arm: sha256 of the raw file} re-checked before use. Returns
    {frame: {'ref': ref_record, 'shape', 'spacing', 'vox_ml', 'diag_mm', 'arms': {arm: v1 record}}}."""
    refs, ctx = references(ds, cid, marker, frames)
    maps = {}
    for a, (p, lk) in arms.items():
        if shas is not None and F.sha256(p) != shas[a]:
            raise SystemExit(f"[REFUSE] {a}/{cid}: {p} changed after it was hashed")
        seg, im = read_raw(p)
        if "img" in ctx:
            assert tuple(seg.shape) == tuple(ctx["img"].shape[:3]), (a, cid, seg.shape)
        maps[a] = lut(ds, lk)[seg]
    out = {}
    k = n_org(ds)
    for fr, (G, sp) in refs.items():
        sc = F.Scorer(G, k, sp)
        rec = sc.frame_record()
        for a, Y in maps.items():
            Yf = arm_in_frame(ds, fr, Y, ctx)
            assert Yf.shape == G.shape, (a, cid, fr, Yf.shape, G.shape)
            rec["arms"][a] = sc.score(Yf, surface=surface)
        out[fr] = rec
    if ctx.get("unverified"):
        out["_unverified"] = True
    return out


# ---------------------------------------------------------------------------------------------------- aggregation
def macro_delta(R, frame, arm, base, organs, ids, penalised=True):
    """{cid: macro(arm) - macro(base)} over the organs (1-based) on the scans of `ids` that have the frame."""
    import numpy as np
    out = {}
    for c in ids:
        f = R.get(c, {}).get(frame)
        if f is None:
            continue
        a = F.macro(f["arms"][arm], f["ref"], organs, penalised)
        b = F.macro(f["arms"][base], f["ref"], organs, penalised)
        if np.isfinite(a) and np.isfinite(b):
            out[c] = a - b
    return out


def macro_values(R, frame, arm, organs, ids, penalised=True):
    return {c: F.macro(R[c][frame]["arms"][arm], R[c][frame]["ref"], organs, penalised) for c in ids if frame in R.get(c, {})}


def organ_delta(R, frame, arm, base, o, ids):
    import numpy as np
    out = {}
    for c in ids:
        f = R.get(c, {}).get(frame)
        if f is None or not f["ref"]["present"][o - 1]:
            continue
        a, b = F.organ_dice(f["arms"][arm], f["ref"], o), F.organ_dice(f["arms"][base], f["ref"], o)
        if np.isfinite(a) and np.isfinite(b):
            out[c] = a - b
    return out


def fp_events(R, frame, arm, organs, ids):
    ev = []
    for c in ids:
        f = R.get(c, {}).get(frame)
        if f is None:
            continue
        for o in organs:
            if f["arms"][arm]["status"][o - 1] == "FP":
                ev.append([c, o])
    return ev


def failures(R, frame, arm, base, organs, ids):
    """conventions metrics.failure: failure = (R and Dice < 50) or FP organ; fixed / created vs the base arm."""
    fixed, created = [], []

    def fail(rec, ref, j):
        return (ref["present"][j] and rec["dice"][j] < 50) or rec["status"][j] == "FP"

    for c in ids:
        f = R.get(c, {}).get(frame)
        if f is None:
            continue
        for o in organs:
            j = o - 1
            fb, fa = fail(f["arms"][base], f["ref"], j), fail(f["arms"][arm], f["ref"], j)
            if fb and not fa:
                fixed.append([c, o, f["arms"][base]["dice"][j], f["arms"][arm]["dice"][j]])
            elif fa and not fb:
                created.append([c, o, f["arms"][base]["dice"][j], f["arms"][arm]["dice"][j]])
    return {"fixed": fixed, "created": created, "n_fixed": len(fixed), "n_created": len(created)}


def surface_summary(R, frame, arm, organs, ids, metric):
    """Mean incl. the FN/FP penalties, number penalised, median (conventions metrics.hd95)."""
    import numpy as np
    v, pen = [], 0
    for c in ids:
        f = R.get(c, {}).get(frame)
        if f is None:
            continue
        for o in organs:
            j = o - 1
            st = f["arms"][arm]["status"][j]
            if st == "TN":
                continue
            v.append(f["arms"][arm][metric][j])
            pen += st in ("FN", "FP")
    v = np.array(v, float)
    return {"mean": float(np.mean(v)) if len(v) else None, "median": float(np.median(v)) if len(v) else None,
            "n_pairs": int(len(v)), "n_penalised": int(pen)}


def contrast(R, frame, arm, base, organs, ids, clusters=None, margin=F.NI_MARGIN, per_organ=True):
    """The pre-specified block of statistics of one paired contrast arm - base on one set / frame."""
    d = macro_delta(R, frame, arm, base, organs, ids, True)
    out = {"arm": arm, "base": base, "frame": frame, "organs": organs, "n_scans": len(d),
           "macro_penalised": F.cluster_boot(d, clusters), "macro_penalised_t": F.t_ci(list(d.values())),
           "macro_reference_present": F.cluster_boot(macro_delta(R, frame, arm, base, organs, ids, False), clusters),
           "fp_events": {"arm": fp_events(R, frame, arm, organs, ids), "base": fp_events(R, frame, base, organs, ids)},
           "failures": failures(R, frame, arm, base, organs, ids)}
    if per_organ:
        po = {}
        for o in organs:
            dd = organ_delta(R, frame, arm, base, o, ids)
            cb = F.cluster_boot(dd, clusters)
            po[str(o)] = dict(cb, ni=F.ni_class(cb["ci"], margin),
                              ni_sensitivity={str(m): F.ni_class(cb["ci"], m) for m in F.NI_SENSITIVITY},
                              harm=F.harm_class(cb["ci"]), organ_floor_flag=bool(cb["mean"] is not None and cb["mean"] < F.ORGAN_FLOOR))
        out["per_organ"] = po
        out["all_organs_non_inferior"] = bool(po) and all(v["ni"] == "NON-INFERIOR" for v in po.values())
        out["any_organ_inferior"] = any(v["ni"] == "INFERIOR" for v in po.values())
    for m in ("hd95", "nsd"):
        out[m] = {"arm": surface_summary(R, frame, arm, organs, ids, m), "base": surface_summary(R, frame, base, organs, ids, m)}
    return out


def maps_manifest(arm_files):
    """{arm: {cid: path}} -> {arm: {cid: sha256}} (hashed BEFORE any reference is opened)."""
    return {a: {c: F.sha256(p) for c, p in files.items()} for a, files in arm_files.items()}


def done_raw_sha(done_file, cid):
    """The recorded sha256 of a scan's raw map: DONE per_scan_raw_sha256 (this block) / per_scan raw_sha256 (other blocks) /
    the per-scan record h5/<cid>.json 'raw_file_sha256' or 'raw_sha256' (the VISTA3D V0 jobs record it there)."""
    d = json.load(open(done_file))
    if "per_scan_raw_sha256" in d:
        return d["per_scan_raw_sha256"].get(cid)
    ps = d.get("per_scan", {}).get(cid)
    if isinstance(ps, dict) and ps.get("raw_sha256"):
        return ps["raw_sha256"]
    rp = Path(done_file).parent / "h5" / f"{cid}.json"
    if rp.is_file():
        r = json.load(open(rp))
        return r.get("raw_file_sha256") or r.get("raw_sha256")
    return None
