"""vccor/final LCTSC CONVERT (CPU; runs in external/lctsc_env: pydicom 2.4.4, SimpleITK 2.3.1, rt-utils 1.2.7, numpy 1.26.4,
nibabel 5.2.1; idempotent).

Stages:
  images   every downloaded CT zip -> data_lctsc/dicom/<pid>_CT/ (verified zip, member count == TCIA ImageCount) ->
           data_lctsc/images/<pid>.nii.gz with SimpleITK's GDCM series reader (one series per patient asserted; HU int16;
           SimpleITK writes the NIfTI geometry). Records shape, spacing, origin, direction and the series' DICOM header
           fields (manufacturer, model, kernel, kVp, slice thickness, contrast agent) per patient -> K_lctsc/images.json.
  labels   the TRAINING RTSTRUCT of the 12 draw patients only (lctsc_common.draw_patients(); a test patient is refused):
           rt-utils 1.2.7 RTStructBuilder fills each ROI's contours per slice (its own polygon fill); slices are matched to
           SimpleITK's slice order by SOPInstanceUID; ROI names -> analysis ids by lctsc_common.ROI_NAMES (unknown ROIs are
           recorded and ignored, a missing structure is recorded); overlaps resolved by lctsc_common.PRIORITY (counts
           recorded) -> data_lctsc/labels_train_v2/<pid>.nii.gz (uint8, the image's exact geometry) + K_lctsc/labels_train_v2.json
           [v2: lctsc_raster, union / hole fill rule, sanity before write (review SF2 / SF4)]
           (every RTSTRUCT opened is listed). Label sanity: ids in 0..5, the label grid == the image grid, per-structure
           volume (mL) and mean HU (lungs must be < -500 HU, heart and cord > -100 HU: an orientation / fill check).
  dups     image-only near-duplicate check, test vs training (all 60 CT): header fingerprint (shape, spacing, origin) and
           Pearson r of the images resampled to 64^3 (HU clipped to [-1000, 1000]); pairs with r >= 0.999 or equal
           fingerprints are flagged -> K_lctsc/dups.json. No label is involved.

usage:  external/lctsc_env/bin/python -I -B scripts/vccor/final/lctsc/lctsc_convert.py --stage images|labels|dups

Version of 2026-11-13.
"""
import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lctsc_common as C  # noqa: E402
import lctsc_log as L  # noqa: E402

HDR_FIELDS = ("Manufacturer", "ManufacturerModelName", "ConvolutionKernel", "KVP", "SliceThickness", "ContrastBolusAgent",
              "InstitutionName", "StudyDescription", "PatientSex")


def zip_of(pid, mod):
    return C.ZIPS / f"{pid}_{mod}.zip"


def unpack(pid, mod):
    rec = json.load(open(C.DOWNLOAD))["files"][f"{pid}_{mod}.zip"]
    d = C.DICOM / f"{pid}_{mod}"
    if d.is_dir() and len(list(d.rglob("*.dcm"))) == rec["n_dcm"]:
        return d
    z = zip_of(pid, mod)
    assert L.sha256(z) == rec["sha256"], f"{z}: changed since download"
    tmp = C.DICOM / f".tmp_{pid}_{mod}"
    if tmp.exists():
        shutil.rmtree(tmp)
    with zipfile.ZipFile(z) as zf:
        for m in zf.infolist():
            assert not m.filename.startswith("/") and ".." not in Path(m.filename).parts, m.filename
        zf.extractall(tmp)
    n = len(list(tmp.rglob("*.dcm")))
    assert n == rec["n_dcm"], (pid, mod, n, rec["n_dcm"])
    tmp.rename(d)
    return d


def series_files(d):
    import SimpleITK as sitk
    dirs = sorted({p.parent for p in d.rglob("*.dcm")})
    assert len(dirs) == 1, f"{d}: {len(dirs)} folders"
    ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(dirs[0]))
    assert len(ids) == 1, f"{d}: {len(ids)} series"
    return dirs[0], list(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(dirs[0]), ids[0]))


def stage_images():
    import numpy as np
    import pydicom
    import SimpleITK as sitk
    C.IMAGES.mkdir(parents=True, exist_ok=True)
    out_p = C.OUT / "images.json"
    out = json.load(open(out_p)) if out_p.is_file() else {"patients": {}}
    pids = C.patients("train") + C.patients("test")
    have = json.load(open(C.DOWNLOAD))["files"]
    for pid in pids:
        if pid in out["patients"] and C.image_path(pid).is_file():
            continue
        if f"{pid}_CT.zip" not in have:
            L.event(f"[images] {pid}: CT not downloaded yet, skipped")
            continue
        d = unpack(pid, "CT")
        sdir, files = series_files(d)
        r = sitk.ImageSeriesReader()
        r.SetFileNames(files)
        img = r.Execute()
        arr = sitk.GetArrayFromImage(img)
        assert arr.min() >= -32768 and arr.max() <= 32767, (pid, int(arr.min()), int(arr.max()))   # int16-safe; values kept as stored
        img16 = sitk.Cast(img, sitk.sitkInt16)
        tmp = C.IMAGES / f".{pid}.tmp.nii.gz"
        sitk.WriteImage(img16, str(tmp), useCompression=True)
        tmp.rename(C.image_path(pid))
        ds = pydicom.dcmread(files[len(files) // 2], stop_before_pixels=True)
        zs = [float(pydicom.dcmread(f, stop_before_pixels=True).ImagePositionPatient[2]) for f in files]
        dz = np.diff(sorted(zs))
        out["patients"][pid] = {"n_slices": len(files), "size_xyz": list(img.GetSize()), "spacing": list(img.GetSpacing()),
                                "origin": list(img.GetOrigin()), "direction": list(img.GetDirection()),
                                "slice_gap_min_max": [float(dz.min()), float(dz.max())] if len(dz) else None,
                                "hu_min_max": [int(arr.min()), int(arr.max())],
                                "dicom": {k: (str(getattr(ds, k)) if hasattr(ds, k) else None) for k in HDR_FIELDS},
                                "sha256": L.sha256(C.image_path(pid)), "site": C.site(pid), "split": "test" if C.is_test(pid) else "train"}
        L.write_json(out_p, out)
        L.event(f"[images] {pid} {img.GetSize()} sp {tuple(round(s, 3) for s in img.GetSpacing())}")
    out["finished"] = L.now()
    L.write_json(out_p, out)


def stage_labels():
    """Training labels of the draw patients with lctsc_raster (union / hole fill rule; review SF2), written to
    labels_train_v2/ ONLY after the sanity check passes (review SF4); a rerun re-checks the stored record and file hash.
    v1 (labels_train/, rt-utils even-odd fill, pooled draws) is kept as a superseded record."""
    import pydicom
    import SimpleITK as sitk
    import lctsc_raster as RA
    C.LABELS_TR.mkdir(parents=True, exist_ok=True)
    out_p = C.OUT / "labels_train_v2.json"
    out = json.load(open(out_p)) if out_p.is_file() else {"patients": {}, "rtstruct_opened": []}
    for pid in C.draw_patients():
        assert not C.is_test(pid), pid
        r = out["patients"].get(pid)
        if r and r.get("sanity", {}).get("all_ok") and C.train_label_path(pid).is_file() \
                and L.sha256(C.train_label_path(pid)) == r["sha256"]:
            continue
        if not C.image_path(pid).is_file():
            L.event(f"[labels] {pid}: CT image not converted yet, skipped")
            continue
        ct_dir, files = series_files(C.DICOM / f"{pid}_CT")
        ct_uid = str(pydicom.dcmread(files[0], stop_before_pixels=True).SeriesInstanceUID)
        rs_d = unpack(pid, "RTSTRUCT")
        rs = sorted(rs_d.rglob("*.dcm"))
        assert len(rs) == 1, (pid, len(rs))
        out["rtstruct_opened"] = sorted(set(out["rtstruct_opened"]) | {pid})
        L.write_json(out_p, out)
        L.event(f"[gt] LCTSC training RTSTRUCT {pid} opened (draw patient; rasteriser v2)")
        img = sitk.ReadImage(str(C.image_path(pid)))
        hu = sitk.GetArrayFromImage(img)
        lab, rrec = RA.rasterise(ct_dir, rs[0], ct_uid, files, hu.shape)
        stats, sane = RA.sanity(lab, hu, img)
        rec = {"raster": rrec, "stats": stats, "sanity": sane, "rtstruct_sha256": L.sha256(rs[0]), "ct_series_uid": ct_uid,
               "converter": "lctsc_raster (rt-utils 1.2.7 geometry, union / hole fill); SimpleITK 2.3.1"}
        if not sane["all_ok"] or rrec["missing"]:
            out["patients"][pid] = dict(rec, sha256=None, written=False)
            L.write_json(out_p, out)
            raise SystemExit(f"{pid}: training label sanity failed or a structure is missing ({sane}, {rrec['missing']}); "
                             f"nothing written")
        L_img = sitk.GetImageFromArray(lab)
        L_img.CopyInformation(img)
        tmp = C.LABELS_TR / f".{pid}.tmp.nii.gz"
        sitk.WriteImage(L_img, str(tmp), useCompression=True)
        tmp.rename(C.train_label_path(pid))
        out["patients"][pid] = dict(rec, sha256=L.sha256(C.train_label_path(pid)), written=True)
        L.write_json(out_p, out)
        L.event(f"[labels] {pid} rois {list(rrec['roi'])} missing {rrec['missing']} sanity {sane}")
    out["finished"] = L.now()
    L.write_json(out_p, out)


def stage_dups():
    import numpy as np
    import SimpleITK as sitk
    feats = {}
    for pid in C.patients("train") + C.patients("test"):
        img = sitk.ReadImage(str(C.image_path(pid)))
        rs = sitk.ResampleImageFilter()
        rs.SetSize([64, 64, 64])
        sz, sp = img.GetSize(), img.GetSpacing()
        rs.SetOutputSpacing([sz[i] * sp[i] / 64 for i in range(3)])
        rs.SetOutputOrigin(img.GetOrigin())
        rs.SetOutputDirection(img.GetDirection())
        rs.SetInterpolator(sitk.sitkLinear)
        rs.SetDefaultPixelValue(-1000)
        a = np.clip(sitk.GetArrayFromImage(rs.Execute(sitk.Cast(img, sitk.sitkFloat32))), -1000, 1000).ravel()
        feats[pid] = (a - a.mean()) / (a.std() + 1e-6), (tuple(sz), tuple(round(s, 3) for s in sp),
                                                          tuple(round(o, 1) for o in img.GetOrigin()))
    tr, te = C.patients("train"), C.patients("test")
    pairs, top = [], []
    for t in te:
        best = max(((float(np.mean(feats[t][0] * feats[s][0])), s) for s in tr))
        top.append({"test": t, "max_r_train": round(best[0], 5), "argmax": best[1]})
        for s in tr:
            r = float(np.mean(feats[t][0] * feats[s][0]))
            if r >= 0.999 or feats[t][1] == feats[s][1]:
                pairs.append({"test": t, "train": s, "r": round(r, 5), "same_fingerprint": feats[t][1] == feats[s][1]})
    within = []
    for i, a in enumerate(te):
        for b in te[i + 1:]:
            r = float(np.mean(feats[a][0] * feats[b][0]))
            if r >= 0.999:
                within.append({"a": a, "b": b, "r": round(r, 5)})
    rec = {"rule": "Pearson r of 64^3 resampled images (HU clipped [-1000, 1000]) >= 0.999, or equal (size, spacing, origin)",
           "flagged_test_vs_train": pairs, "flagged_within_test": within, "per_test_max": top, "finished": L.now()}
    L.write_json(C.OUT / "dups.json", rec)
    L.event(f"[dups] flagged test-train {len(pairs)}, within-test {len(within)}; max r {max(x['max_r_train'] for x in top):.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["images", "labels", "dups"])
    a = ap.parse_args()
    {"images": stage_images, "labels": stage_labels, "dups": stage_dups}[a.stage]()
