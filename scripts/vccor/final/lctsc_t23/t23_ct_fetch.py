"""vccor/final LCTSC-T23 CT FETCH (CPU, network; runs in external/lctsc_env; idempotent; before the freeze; IMAGES ONLY).

Why: the look's rasteriser (frozen lctsc_raster, as in the d6ayj look) places the RTSTRUCT contours with the CT DICOM series'
geometry; the training CT DICOM folders were deleted after the LCTSC freeze (approved, 2026-11-13 18:17; the CTs are
kept as NIfTI in data_lctsc/images, sha256 in K_lctsc/images.json). This script fetches again, through the public NBIA v1
getImage endpoint, the CT series (exact SeriesInstanceUID of K_lctsc/tcia_manifest.json) of the 23 LCTSC-T23 patients and of
the 4 rehearsal stand-ins (t23_common.STANDINS: site-draw patients whose labels were read on 2026-11-13).

Per patient: zip -> data_lctsc/t23_ct_zips/<pid>_CT.zip (via .part); checks: readable zip, CRCs, dcm member count == TCIA
ImageCount, uncompressed dcm bytes == TCIA FileSize; zip sha256 compared with the 2026-11-13 record (K_lctsc/download.json;
recorded, not required: TCIA may re-zip); extracted to data_lctsc/dicom/<pid>_CT/ (via a .tmp folder); then the series is read
with SimpleITK exactly as lctsc_convert.stage_images did (GDCM series reader, one series, int16) and compared with the stored
NIfTI (data_lctsc/images/<pid>.nii.gz, whose sha256 must equal K_lctsc/images.json): voxel arrays equal and spacing,
origin and direction equal (numpy allclose, atol 1e-6 with its default rtol 1e-5). A patient that fails any check is
recorded as FAILED and the freeze refuses. The re-downloaded zips are not byte-identical to the 2026-11-13 zips (TCIA
re-packages series); the claim is voxel-level and geometry equality. Each run rewrites 'finished' and appends an IMAGES line
(not idempotent in that sense); it refuses after the T23 freeze (review S4/N8).
Refuses any RTSTRUCT series and any LCTSC-Test patient. No label is read. Record: K_lctsc_t23/t23_ct.json and an 'IMAGES'
line in final_prereg.log.

usage:  external/lctsc_env/bin/python -I -B scripts/vccor/final/lctsc_t23/t23_ct_fetch.py

Version of 2026-11-17.
"""
import json
import os
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lctsc"))
sys.path.insert(0, str(HERE))
import lctsc_common as C  # noqa: E402  (frozen)
import t23_common as T  # noqa: E402
import t23_log as L  # noqa: E402

API = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
MIN_FREE_GIB = 27.0


def ct_row(pid):
    rows = [r for r in C.manifest()["series"] if r["PatientID"] == pid and r["Modality"] == "CT"]
    assert len(rows) == 1, (pid, len(rows))
    return rows[0]


def check_zip(path, row):
    with zipfile.ZipFile(path) as z:
        bad = z.testzip()
        dcm = [i for i in z.infolist() if not i.is_dir() and i.filename.lower().endswith(".dcm")]
        for m in z.infolist():
            assert not m.filename.startswith("/") and ".." not in Path(m.filename).parts, m.filename
    if bad is not None:
        return f"CRC error in {bad}"
    n, usize = len(dcm), sum(i.file_size for i in dcm)
    if n != int(row["ImageCount"]) or usize != int(row["FileSize"]):
        return f"{n} dcm / {usize} B vs manifest {row['ImageCount']} dcm / {row['FileSize']} B"
    return None


def download(pid, row):
    T.CT_ZIPS.mkdir(parents=True, exist_ok=True)
    dst, part = T.CT_ZIPS / f"{pid}_CT.zip", T.CT_ZIPS / f".{pid}_CT.part"
    if dst.is_file() and check_zip(dst, row) is None:
        return dst
    for attempt in range(1, 5):
        free = shutil.disk_usage("/").free / 1024 ** 3
        if free < MIN_FREE_GIB:
            raise SystemExit(f"refuse: / has {free:.1f} GiB free (< {MIN_FREE_GIB})")
        try:
            req = urllib.request.Request(f"{API}/getImage?SeriesInstanceUID={row['SeriesInstanceUID']}",
                                         headers={"User-Agent": "lctsc-t23-ct"})
            with urllib.request.urlopen(req, timeout=600) as r, open(part, "wb") as f:
                while True:
                    b = r.read(1 << 22)
                    if not b:
                        break
                    f.write(b)
            why = check_zip(part, row)
            if why is None:
                os.replace(part, dst)
                return dst
            L.event(f"[ct-fetch] {pid}: attempt {attempt}: {why}")
        except Exception as e:  # noqa: BLE001  (network / zip faults are retried)
            L.event(f"[ct-fetch] {pid}: attempt {attempt}: {type(e).__name__}: {str(e)[:120]}")
        if part.exists():
            part.unlink()
        time.sleep(15 * attempt)
    raise SystemExit(f"CT download failed: {pid}")


def extract(pid, zp, row):
    d = C.DICOM / f"{pid}_CT"
    if d.is_dir() and len(list(d.rglob("*.dcm"))) == int(row["ImageCount"]):
        return d
    tmp = C.DICOM / f".tmp_{pid}_CT"
    if tmp.exists():
        shutil.rmtree(tmp)
    with zipfile.ZipFile(zp) as z:
        z.extractall(tmp)
    n = len(list(tmp.rglob("*.dcm")))
    assert n == int(row["ImageCount"]), (pid, n, row["ImageCount"])
    tmp.rename(d)
    return d


def verify(pid, d):
    import numpy as np
    import SimpleITK as sitk
    import lctsc_convert as K  # frozen: series_files() as used by the d6ayj look
    rec_img = json.load(open(C.OUT / "images.json"))["patients"][pid]
    if L.sha256(C.image_path(pid)) != rec_img["sha256"]:
        return {"ok": False, "why": "stored NIfTI differs from K_lctsc/images.json"}
    sdir, files = K.series_files(d)
    r = sitk.ImageSeriesReader()
    r.SetFileNames(files)
    img = sitk.Cast(r.Execute(), sitk.sitkInt16)
    ref = sitk.ReadImage(str(C.image_path(pid)))
    a, b = sitk.GetArrayFromImage(img), sitk.GetArrayFromImage(ref)
    geo = all(np.allclose(x, y, atol=1e-6) for x, y in ((img.GetSpacing(), ref.GetSpacing()), (img.GetOrigin(), ref.GetOrigin()),
                                                         (img.GetDirection(), ref.GetDirection())))
    same = a.shape == b.shape and bool(np.array_equal(a, b))
    return {"ok": bool(same and geo), "voxels_equal": same, "geometry_equal": bool(geo), "shape_zyx": list(a.shape),
            "n_files": len(files), "nifti_sha256": rec_img["sha256"]}


def main():
    if (T.OUT / "code_freeze.json").exists():            # review S4 (added after the 2026-11-17 run; as_run/ holds that copy)
        raise SystemExit("an LCTSC-T23 freeze exists: t23_ct.json is frozen and is not rewritten")
    out = json.load(open(T.CT_RECORD)) if T.CT_RECORD.is_file() else {"patients": {}}
    pids = list(T.STANDINS) + T.ids()
    assert not any(C.is_test(p) for p in pids), "LCTSC-T23 never fetches a test patient"
    dl = json.load(open(C.DOWNLOAD))["files"]
    L.event(f"[ct-fetch] start: {len(pids)} CT series (23 LCTSC-T23 + {len(T.STANDINS)} stand-ins); images only")
    for pid in pids:
        if out["patients"].get(pid, {}).get("ok"):
            continue
        row = ct_row(pid)
        zp = download(pid, row)
        d = extract(pid, zp, row)
        v = verify(pid, d)
        old = dl.get(f"{pid}_CT.zip", {})
        v.update(uid=row["SeriesInstanceUID"], zip_sha256=L.sha256(zp), zip_bytes=zp.stat().st_size,
                 zip_equal_to_2026_11_13_record=(old.get("sha256") == L.sha256(zp)), role=("stand-in" if pid in T.STANDINS
                                                                                          else "LCTSC-T23"), checked=L.now())
        out["patients"][pid] = v
        L.write_json(T.CT_RECORD, out)
        L.event(f"[ct-fetch] {pid}: ok {v['ok']} (voxels {v.get('voxels_equal')}, geometry {v.get('geometry_equal')}, "
                f"zip equal to 2026-11-13 {v['zip_equal_to_2026_11_13_record']})")
    out["all_ok"] = all(out["patients"].get(p, {}).get("ok") for p in pids)
    out["finished"] = L.now()
    L.write_json(T.CT_RECORD, out)
    L.prereg_line("IMAGES", T.CT_RECORD, f"lctsc_t23: CT DICOM re-downloaded for {len(pids)} patients (23 LCTSC-T23 + "
                                         f"{len(T.STANDINS)} stand-ins) for the rasteriser's geometry; all reproduce the stored "
                                         f"NIfTI voxel for voxel: {out['all_ok']}; no label read")
    return 0 if out["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
