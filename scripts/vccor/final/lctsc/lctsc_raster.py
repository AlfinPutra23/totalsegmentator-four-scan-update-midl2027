"""vccor/final LCTSC RTSTRUCT RASTERISER (runs in external/lctsc_env: pydicom 2.4.4, rt-utils 1.2.7 geometry helpers, OpenCV;
review SF2). Used for the training labels (lctsc_convert.py) and the test labels (lctsc_look_labels.py) alike.

Geometry == rt-utils 1.2.7: the CT series sorted by slice position (image_helper.load_sorted_image_series), the
patient-to-pixel matrix (get_patient_to_pixel_transformation_matrix), vertices rounded to the nearest pixel (np.around) and
filled with cv2.fillPoly (boundary pixels included). The FILL RULE differs from rt-utils (which fills all contours of a
slice in ONE fillPoly call = even-odd, so two overlapping contours of one ROI cancel out):
  each contour is filled on its own; on a slice, a contour whose filled area lies strictly inside another contour of the
  same ROI is a HOLE (subtracted); every other contour is added (UNION). Identical duplicates are added once.
Slice association: a contour's ContourImageSequence SOP UID when present; otherwise the slice whose position along the
slice normal is within half a slice spacing of the contour's mean position (recorded as 'matched by position').
Per-ROI outcomes (never an exception; each recorded in the result):
  ok | skipped contours (fewer than 3 points: POINT / one- or two-point contours) | contour SOP UID not in the CT |
  no contour on the CT | duplicate structure name (two ROIs map to one structure: that structure is MISSING for the
  patient) | wrong referenced series (the RTSTRUCT references another CT series: every structure is MISSING) |
  conversion error (that structure is MISSING).
A structure without an ROI, or MISSING by an outcome above, is 'not evaluable' for that patient (registered scoring rule:
excluded from every arm's macro and from every per-structure contrast for that patient).
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lctsc_common as C  # noqa: E402


def referenced_series_uids(rs):
    out = set()
    for fr in getattr(rs, "ReferencedFrameOfReferenceSequence", []) or []:
        for st in getattr(fr, "RTReferencedStudySequence", []) or []:
            for se in getattr(st, "RTReferencedSeriesSequence", []) or []:
                out.add(str(se.SeriesInstanceUID))
    return out


DUP_FRAC = 0.5                           # review NS2: a nested contour >= 50 % of its container is a duplicate


def fill_slice(polys, shape):
    """Contours filled one by one and classified by nesting, largest first (registered rule, review NS2):
      a contour inside no other contour                          -> FILL
      a contour whose smallest container is c:
        area >= 50 % of c's area                                 -> a DUPLICATE of c: same class as c
        area <  50 % of c's area, c is FILL                      -> HOLE (subtracted)
        area <  50 % of c's area, c is HOLE                      -> ISLAND (filled again)
    Identical duplicates are counted once. Records the hole area fractions."""
    import cv2
    import numpy as np
    masks = []
    for p in polys:
        m = np.zeros(shape, np.uint8)
        cv2.fillPoly(m, [p], 1)
        masks.append(m.astype(bool))
    keyset, uniq = set(), []
    for m in masks:
        k = m.tobytes()
        if k not in keyset:
            keyset.add(k)
            uniq.append(m)
    order = sorted(range(len(uniq)), key=lambda i: -int(uniq[i].sum()))
    cls, out = {}, np.zeros(shape, bool)
    hole_fracs, n_dup, n_island = [], 0, 0
    for i in order:
        m, a = uniq[i], int(uniq[i].sum())
        conts = [j for j in cls if a < int(uniq[j].sum()) and not (m & ~uniq[j]).any()]
        if not conts:
            cls[i] = "fill"
        else:
            c = min(conts, key=lambda j: int(uniq[j].sum()))
            frac = a / float(uniq[c].sum())
            if frac >= DUP_FRAC:
                cls[i] = cls[c]
                n_dup += 1
            elif cls[c] == "fill":
                cls[i] = "hole"
                hole_fracs.append(round(frac, 4))
            else:
                cls[i] = "fill"
                n_island += 1
        if cls[i] == "fill":
            out |= m
        else:
            out &= ~m
    return out, {"n_contours": len(polys), "n_unique": len(uniq), "n_holes": len(hole_fracs),
                 "hole_area_fractions": hole_fracs, "n_nested_duplicates": n_dup, "n_islands": n_island}


def _missing_all(rec, why):
    for o in range(1, C.N_ORG + 1):
        rec["structures"][C.ORGANS[o - 1]] = f"MISSING: {why}"
    rec["missing"] = [C.ORGANS[o - 1] for o in range(1, C.N_ORG + 1)]
    rec["overlap_vox_overwritten"] = {}
    return rec


def rasterise(ct_dir, rs_path, ct_series_uid, sitk_files, shape_zyx):
    """-> (label uint8 (z, y, x) in SimpleITK slice order, record). sitk_files: the CT file names in SimpleITK's order.
    Never raises for a malformed RTSTRUCT (review N4): an unreadable RTSTRUCT makes every structure MISSING."""
    import cv2
    import numpy as np
    import pydicom
    from rt_utils import image_helper as H
    rec = {"roi": {}, "structures": {}, "referenced_series_ok": None}
    lab = np.zeros(shape_zyx, np.uint8)
    series = H.load_sorted_image_series(str(ct_dir))                 # the CT (already converted and checked): may raise
    uid_k = {str(ds.SOPInstanceUID): k for k, ds in enumerate(series)}
    order = [uid_k[str(pydicom.dcmread(f, stop_before_pixels=True).SOPInstanceUID)] for f in sitk_files]
    pos = np.array([H.get_slice_position(ds) for ds in series])
    half = abs(float(H.get_spacing_between_slices(series))) / 2.0
    T = H.get_patient_to_pixel_transformation_matrix(series)
    rows, cols = int(series[0].Rows), int(series[0].Columns)
    _, _, normal = H.get_slice_directions(series[0])
    try:
        rs = pydicom.dcmread(str(rs_path))
        refs = referenced_series_uids(rs)
        rec["referenced_series_ok"] = (ct_series_uid in refs) if refs else None
        names = {int(r.ROINumber): str(r.ROIName) for r in getattr(rs, "StructureSetROISequence", [])}
        contours = {int(rc.ReferencedROINumber): rc for rc in getattr(rs, "ROIContourSequence", [])}
    except Exception as e:  # noqa: BLE001  (registered: an unreadable RTSTRUCT makes every structure MISSING)
        rec["rtstruct_error"] = f"{type(e).__name__}: {str(e)[:160]}"
        return lab, _missing_all(rec, "RTSTRUCT unreadable")
    # review N3: a referenced series other than the CT is FLAGGED only; contours are still placed by their own SOP UIDs
    # (contours without a ContourImageSequence are then NOT placed by position); a structure none of whose contours
    # lands on the CT is MISSING ('no contour on the CT').
    by_struct = {}
    for n, nm in names.items():
        o = C.ROI_NAMES.get(C.norm_roi(nm))
        if o is None:
            rec["roi"][nm] = {"outcome": "unknown ROI name (ignored)"}
            continue
        by_struct.setdefault(o, []).append((n, nm))
    masks = {}
    for o, lst in by_struct.items():
        sname = C.ORGANS[o - 1]
        if len(lst) > 1:
            for _, nm in lst:
                rec["roi"][nm] = {"outcome": f"duplicate structure name ({sname}): structure MISSING"}
            rec["structures"][sname] = "MISSING: duplicate structure name"
            continue
        n, nm = lst[0]
        try:
            rc = contours.get(n)
            seq = list(getattr(rc, "ContourSequence", []) or []) if rc is not None else []
            per_slice, skipped, not_in_ct, by_pos = {}, 0, 0, 0
            for c in seq:
                pts = np.reshape(np.array(c.ContourData, float), (-1, 3))
                if len(pts) < 3:
                    skipped += 1
                    continue
                k = None
                cis = getattr(c, "ContourImageSequence", None)
                if cis:
                    u = str(cis[0].ReferencedSOPInstanceUID)
                    if u in uid_k:
                        k = uid_k[u]
                    else:
                        not_in_ct += 1
                        continue
                elif rec["referenced_series_ok"] is False:
                    not_in_ct += 1
                    continue
                else:
                    pz = float(np.dot(normal, pts.mean(0)))
                    j = int(np.argmin(np.abs(pos - pz)))
                    if abs(pos[j] - pz) <= half + 1e-6:
                        k, by_pos = j, by_pos + 1
                    else:
                        not_in_ct += 1
                        continue
                px = H.apply_transformation_to_3d_points(pts, T)
                per_slice.setdefault(k, []).append(np.around(px[:, :2]).astype(np.int32))
            m = np.zeros((rows, cols, len(series)), bool)             # (y, x, slice); fillPoly points are (x, y)
            multi, holes, eo_diff, fracs, ndup, nisl = 0, 0, 0, [], 0, 0
            for k, polys in per_slice.items():
                sm, srec = fill_slice(polys, (rows, cols))
                m[:, :, k] = sm
                if srec["n_contours"] > 1:
                    multi += 1
                    eo = np.zeros((rows, cols), np.uint8)
                    cv2.fillPoly(eo, polys, 1)
                    eo_diff += int((eo.astype(bool) != sm).sum())
                holes += srec["n_holes"]
                fracs += srec["hole_area_fractions"]
                ndup += srec["n_nested_duplicates"]
                nisl += srec["n_islands"]
            if not m.any():
                rec["roi"][nm] = {"outcome": "no contour on the CT: structure MISSING", "skipped_contours": skipped,
                                  "contours_not_in_ct": not_in_ct}
                rec["structures"][sname] = "MISSING: no contour on the CT"
                continue
            masks[o] = np.stack([m[:, :, k] for k in order], 0)
            rec["roi"][nm] = {"outcome": "ok" if not (skipped or not_in_ct) else "ok with skipped contours",
                              "structure": sname, "contoured_slices": len(per_slice), "slices_multi_contour": multi,
                              "holes": holes, "hole_area_fractions": fracs, "nested_duplicates": ndup, "islands": nisl,
                              "union_vs_even_odd_pixels": eo_diff, "skipped_contours": skipped,
                              "contours_not_in_ct": not_in_ct, "matched_by_position": by_pos}
            rec["structures"][sname] = "ok"
        except Exception as e:  # noqa: BLE001  (registered: a conversion error makes the structure MISSING)
            rec["roi"][nm] = {"outcome": f"conversion error ({type(e).__name__}: {str(e)[:120]}): structure MISSING"}
            rec["structures"][sname] = "MISSING: conversion error"
    for o in range(1, C.N_ORG + 1):
        rec["structures"].setdefault(C.ORGANS[o - 1], "MISSING: no ROI")
    overlap = {}
    for o in reversed(C.PRIORITY):
        if o in masks:
            assert masks[o].shape == tuple(shape_zyx), (masks[o].shape, shape_zyx)
            ov = (lab > 0) & masks[o]
            if ov.any():
                overlap[C.ORGANS[o - 1]] = int(ov.sum())
            lab[masks[o]] = o
    rec["overlap_vox_overwritten"] = overlap
    rec["missing"] = [s for s, v in rec["structures"].items() if v != "ok"]
    return lab, rec


def sanity(lab, hu, img):
    """Descriptive per-structure record (ml, mean HU, contoured slices) + checks: grid, lungs < -500 HU, heart / cord
    > -100 HU, left lung on the patient's left (LPS x larger)."""
    import numpy as np
    vox_ml = float(np.prod(img.GetSpacing())) / 1000.0
    st = {}
    for o in range(1, C.N_ORG + 1):
        m = lab == o
        st[C.ORGANS[o - 1]] = {"present": bool(m.any()), "ml": round(float(m.sum()) * vox_ml, 2),
                               "mean_hu": round(float(hu[m].mean()), 1) if m.any() else None,
                               "slices": int(m.any(axis=(1, 2)).sum())}
    chk = {"grid_equal": tuple(lab.shape) == tuple(hu.shape),
           "lungs_below_-500": all(st[k]["mean_hu"] is None or st[k]["mean_hu"] < -500 for k in ("lung_left", "lung_right")),
           "heart_cord_above_-100": all(st[k]["mean_hu"] is None or st[k]["mean_hu"] > -100 for k in ("heart", "spinal_cord")),
           "left_lung_on_patient_left": None}
    if st["lung_left"]["present"] and st["lung_right"]["present"]:
        def cx(o):
            idx = np.argwhere(lab == o).mean(0)[::-1]
            return img.TransformContinuousIndexToPhysicalPoint([float(v) for v in idx])[0]
        chk["left_lung_on_patient_left"] = bool(cx(3) > cx(4))
    chk["all_ok"] = bool(chk["grid_equal"] and chk["lungs_below_-500"] and chk["heart_cord_above_-100"]
                         and chk["left_lung_on_patient_left"] is not False)
    return st, chk
