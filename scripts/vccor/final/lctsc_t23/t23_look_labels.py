"""vccor/final LCTSC-T23 LOOK, part 1 of 2: the labels of the 23 LCTSC-T23 patients (CPU, no network; runs in
external/lctsc_env; ONE look). DERIVED from the frozen lctsc_look_labels.py by derive_t23.py (every change is listed there).

Gates, in this order, before any of the 23 RTSTRUCT zips is extracted (a new look):
  1. no second look: look/GT_OPENED and look/results.json must not exist;
  2. the freeze (t23_gates.freeze_ok) and the public gate (t23_gates.gate_public: upload gate, logged deposit holding the
     current freeze, public after the upload and before the public deadline Thu 2026-11-19 12:00 KST, before the stop rule
     Fri 2026-11-20 16:00 KST);
  3. ARMS: BEFORE the cutoff (Fri 2026-11-20 10:00 KST) every arm must be DONE in mode 'real' or declared NOT RUN by the
     prediction runner's binding NOTRUN line; AFTER the cutoff an arm not DONE by the cutoff is NOT RUN; F0 and at least
     one F1b arm must be DONE;
  4. the arms' maps are hashed into look/maps.json (each raw map against its DONE record).
Then look/GT_OPENED is created O_EXCL and a 'GT_OPENED sha256 ...' line is appended (re-appended by --continue if a crash
left it out). The look is defined by the GT_OPENED time: --continue may finish it after the stop rule.

Labels (NO download; the one change of substance from the LCTSC look): per patient, the RTSTRUCT zip downloaded on
2026-11-13 with the other training series (data_lctsc/tcia_zips/<pid>_RTSTRUCT.zip) is used. Before it is extracted it
must equal its K_lctsc/download.json record (sha256 and size) and the TCIA manifest (dcm member count and bytes, zip CRCs),
and no extracted copy may exist in data_lctsc/dicom; anything else logs GT_ABORTED and stops the look for good. The zip is
extracted under look/rtstruct/<pid>/ only after GT_OPENED. Conversion == the LCTSC look: lctsc_raster (frozen) on the CT
DICOM series re-downloaded on 2026-11-17 (data_lctsc/dicom/<pid>_CT; t23_ct.json shows each reproduces the stored NIfTI
voxel for voxel); the RTSTRUCT's referenced series is checked against the CT's SeriesInstanceUID; per patient a descriptive
sanity record; MISSING structures are listed (registered scoring rule: not evaluable for that patient). labels.json and a
'LABELS sha256 ...' line close the step.

usage:  external/lctsc_env/bin/python -I -B scripts/vccor/final/lctsc_t23/t23_look_labels.py [--check | --continue]

Version of 2026-11-13; LCTSC-T23 derivation 2026-11-17.
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lctsc"))
sys.path.insert(0, str(HERE))
import lctsc_common as C  # noqa: E402
import t23_gates as G  # noqa: E402
import t23_log as L  # noqa: E402
import t23_common as T  # noqa: E402

API = None                           # LCTSC-T23: no download in the look
ZDIR = C.ZIPS                        # the training RTSTRUCT zips of 2026-11-13 (K_lctsc/download.json); read only
LDIR = G.LOOK / "labels"
MARKER = G.LOOK / "GT_OPENED"
RETRY_S = [60, 300, 900]
PERSISTENT_N = 3
FAULTS = os.environ.get("LCTSC_T23_REHEARSAL_FAULTS", "") if G.REHEARSAL else ""    # rehearsal-only fault injection


class Temporary(Exception):
    pass


def fault(kind, pid=None):
    for f in FAULTS.split(";"):
        parts = f.split(":")
        if parts and parts[0] == kind and (pid is None or (len(parts) > 1 and parts[1] == pid)):
            return parts
    return None


def aborted():
    if not G.PREREG_LOG.is_file():
        return False
    return any(" GT_ABORTED lctsc_t23 " in line for line in open(G.PREREG_LOG))


def not_completed():
    if not G.PREREG_LOG.is_file():
        return False
    return any(" NOTE lctsc_t23 look not completed" in line for line in open(G.PREREG_LOG))


def notrun_declared(arm):
    return G.notrun_declared(arm)


def arm_states():
    st = {a: G.arm_done(a) for a in G.ARMS}
    modes = ("real", "rehearsal") if G.REHEARSAL else ("real",)
    # review NS1: a binding NOTRUN line beats a DONE.json (an arm declared NOT RUN never enters the look)
    done = [a for a, (s, d) in st.items() if s == "DONE" and (d or {}).get("mode") in modes and not notrun_declared(a)]
    pending = [a for a in G.ARMS if a not in done and not notrun_declared(a)]
    if G.before_cutoff() and pending:
        raise G.Refuse(f"before the cutoff every arm must be DONE (mode real) or declared NOT RUN by the runner; pending: {pending}")
    not_run = {}
    for a in G.ARMS:
        if a in done:
            continue
        not_run[a] = "declared NOT RUN by the prediction runner" if notrun_declared(a) else "not DONE by the cutoff"
    if "F0" not in done or not any(a.startswith("F1b_") for a in done):
        raise G.Refuse(f"arms DONE {done}: F0 and at least one F1b arm are required")
    return done, not_run


def maps_record(done):
    rec = {}
    for a in done:
        d = json.load(open(G.pred_dir(a) / "DONE.json"))
        for pid in G.test_ids():
            h = G.sha256(G.pred_dir(a) / "raw" / f"{pid}.nii.gz")
            if h != d["per_scan_raw_sha256"][pid]:
                raise G.Refuse(f"{a} {pid}: raw map differs from its DONE record")
            rec.setdefault(a, {})[pid] = h
    return rec


def precheck_zips():
    """LCTSC-T23 (review S1): before GT_OPENED, by file metadata only (no byte of any RTSTRUCT is read): every look
    patient's stored zip exists with the size of its K_lctsc/download.json record and (real look) no extracted copy or
    partial extraction exists in data_lctsc/dicom; else the look is refused before it starts."""
    dl = json.load(open(C.DOWNLOAD))["files"]
    for pid in G.test_ids():
        if G.REHEARSAL:
            assert pid in T.STANDINS and pid not in T.ids() and not C.is_test(pid), pid
        rec, z = dl.get(f"{pid}_RTSTRUCT.zip"), ZDIR / f"{pid}_RTSTRUCT.zip"
        if rec is None or not z.is_file() or z.stat().st_size != rec["bytes"] or fault("precheck_size", pid):
            raise G.Refuse(f"{pid}: stored RTSTRUCT zip missing or its size differs from K_lctsc/download.json: no look")
        if not G.REHEARSAL and any((C.DICOM / n).exists() for n in (f"{pid}_RTSTRUCT", f".tmp_{pid}_RTSTRUCT")):
            raise G.Refuse(f"{pid}: an extracted RTSTRUCT copy exists in data_lctsc/dicom: no look")


def gates(cont=False):
    G.kst_gate()
    if (G.LOOK / "results.json").exists():
        raise G.Refuse("results.json exists: no second look")
    if MARKER.exists() and not cont:
        raise G.Refuse("GT_OPENED exists: no second look (use --continue to finish the same look)")
    if cont and (not MARKER.exists() or aborted()):
        raise G.Refuse("--continue needs GT_OPENED and no GT_ABORTED line")
    if cont and not_completed():                     # review diff-2 N1: a look reported NOT COMPLETED stays so
        raise G.Refuse("the look was reported NOT COMPLETED (stop rule): no --continue")
    G.freeze_ok()
    if cont:
        G.gate_public(allow_after_stop=True)        # N8: the look began at GT_OPENED, before the stop rule
        m = json.load(open(G.LOOK / "maps.json"))
        return m["arms_done"], m["not_run"]
    G.gate_public()
    precheck_zips()
    return arm_states()


def series_of(pid):
    rows = [r for r in C.manifest()["series"] if r["PatientID"] == pid and r["Modality"] == "RTSTRUCT"]
    assert len(rows) == 1 and (pid in T.ids() or G.REHEARSAL), pid
    return rows[0]


def check_zip(path, row):
    """-> None if the zip matches the manifest; raises Temporary (unreadable / CRC) or returns a mismatch text."""
    try:
        with zipfile.ZipFile(path) as z:
            bad = z.testzip()
            dcm = [i for i in z.infolist() if i.filename.lower().endswith(".dcm")]
    except (zipfile.BadZipFile, OSError, EOFError) as e:
        raise Temporary(f"not a readable zip ({type(e).__name__})")
    if bad is not None:
        raise Temporary(f"CRC error in {bad}")
    n, usize = len(dcm), sum(i.file_size for i in dcm)
    if n != int(row["ImageCount"]) or usize != int(row["FileSize"]):
        return f"{n} dcm / {usize} B vs manifest {row['ImageCount']} dcm / {row['FileSize']} B"
    return None


def fetch(pid):
    """LCTSC-T23: NO download. The RTSTRUCT zip downloaded on 2026-11-13 with the other training series is used only if it
    equals its K_lctsc/download.json record (sha256 and size) and the TCIA manifest (dcm member count and uncompressed
    bytes, zip CRCs), and (real look) no extracted copy exists in data_lctsc/dicom; anything else logs GT_ABORTED."""
    if G.REHEARSAL:                         # review S2: rehearsal stand-ins only (site-draw patients read on 2026-11-13)
        assert pid in T.STANDINS and pid not in T.ids() and not C.is_test(pid), pid
    row = series_of(pid)
    dst = ZDIR / f"{pid}_RTSTRUCT.zip"
    rec = json.load(open(C.DOWNLOAD))["files"].get(f"{pid}_RTSTRUCT.zip")
    why = None
    if rec is None or not dst.is_file():
        why = "no stored zip or no K_lctsc/download.json record"
    elif not G.REHEARSAL and any((C.DICOM / n).exists() for n in (f"{pid}_RTSTRUCT", f".tmp_{pid}_RTSTRUCT")):
        why = "an extracted copy exists in data_lctsc/dicom (the label was opened before the look)"
    elif fault("mismatch", pid) or dst.stat().st_size != rec["bytes"] or G.sha256(dst) != rec["sha256"]:
        why = "sha256 or size differ from the K_lctsc/download.json record of 2026-11-13"
    else:
        try:
            why = check_zip(dst, row)
        except Temporary as e:
            why = str(e)
    if why:
        L.prereg_note("GT_ABORTED", f"RTSTRUCT {pid}: {why}; the look is aborted")
        raise G.Refuse(f"{pid}: {why}: look aborted")
    return dst, row


def convert(pid, zpath):
    import pydicom
    import SimpleITK as sitk
    import lctsc_convert as K
    import lctsc_raster as RA
    d = G.LOOK / "rtstruct" / pid
    if not d.is_dir():
        tmp = G.LOOK / "rtstruct" / f".tmp_{pid}"
        with zipfile.ZipFile(zpath) as z:
            for m in z.infolist():
                assert not m.filename.startswith("/") and ".." not in Path(m.filename).parts, m.filename
            z.extractall(tmp)
        tmp.rename(d)
    rs = sorted(d.rglob("*.dcm"))
    assert len(rs) == 1, (pid, len(rs))
    ct_dir, files = K.series_files(C.DICOM / f"{pid}_CT")
    ct_uid = str(pydicom.dcmread(files[0], stop_before_pixels=True).SeriesInstanceUID)
    img = sitk.ReadImage(str(C.image_path(pid)))
    hu = sitk.GetArrayFromImage(img)
    lab, rrec = RA.rasterise(ct_dir, rs[0], ct_uid, files, hu.shape)
    drop = fault("drop_roi", pid)
    if drop:                                          # rehearsal only: a stand-in with a structure missing
        o = C.ORGANS.index(drop[2]) + 1
        lab[lab == o] = 0
        rrec["structures"][drop[2]] = "MISSING: no ROI (rehearsal fault)"
        rrec["missing"] = sorted(set(rrec["missing"]) | {drop[2]})
    stats, sane = RA.sanity(lab, hu, img)
    if int(lab.max()) > C.N_ORG or not sane["grid_equal"]:
        L.prereg_note("GT_ABORTED", f"label {pid}: id out of range or grid differs from the CT")
        raise G.Refuse("label grid / id check failed: look aborted")
    out = sitk.GetImageFromArray(lab)
    out.CopyInformation(img)
    LDIR.mkdir(parents=True, exist_ok=True)
    tmp = LDIR / f".{pid}.tmp.nii.gz"
    sitk.WriteImage(out, str(tmp), useCompression=True)
    os.replace(tmp, LDIR / f"{pid}.nii.gz")
    return {"raster": rrec, "missing": rrec["missing"], "stats": stats, "sanity_descriptive": sane, "ct_series_uid": ct_uid,
            "label_sha256": G.sha256(LDIR / f"{pid}.nii.gz"), "rtstruct_sha256": G.sha256(rs[0])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--continue", dest="cont", action="store_true")
    a = ap.parse_args()
    done, not_run = gates(a.cont)
    if a.check:
        print(f"gates OK: arms DONE {done}, not run {not_run}")
        return 0
    G.LOOK.mkdir(parents=True, exist_ok=True)
    if not a.cont:
        maps = {"arms_done": done, "not_run": not_run, "maps": maps_record(done), "written": G.now()}
        if not G.before_stop():                   # LCTSC-T23 (review N3): no GT_OPENED at or after the stop rule
            raise G.Refuse("stop rule passed (Fri 2026-11-20 16:00 KST): no look")
        G.write_json(G.LOOK / "maps.json", maps, mode=0o444)
        fd = os.open(MARKER, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
        os.write(fd, json.dumps({"opened": G.now(), "maps_sha256": G.sha256(G.LOOK / "maps.json")}).encode())
        os.close(fd)
    if not G.logged("GT_OPENED", r"K_lctsc_t23/(rehearsal_\d+/)?look/GT_OPENED$"):
        L.prereg_line("GT_OPENED", MARKER, f"lctsc_t23 look: the 23 RTSTRUCT zips are extracted only after this line; arms DONE {done}; "
                                           f"not run {sorted(not_run)}")
    lp = G.LOOK / "labels.json"
    rec = json.load(open(lp)) if lp.is_file() else {"patients": {}}
    for i, pid in enumerate(G.test_ids()):
        r = rec["patients"].get(pid)
        if r and (LDIR / f"{pid}.nii.gz").is_file() and G.sha256(LDIR / f"{pid}.nii.gz") == r["label_sha256"]:
            continue
        if fault("crash_after") and i == int(fault("crash_after")[1]) and not (G.LOOK / ".crashed_once").exists():
            (G.LOOK / ".crashed_once").touch()
            raise SystemExit(70)                       # rehearsal only: a crash in the middle of the look
        z, row = fetch(pid)
        r = convert(pid, z)
        r.update(zip_sha256=G.sha256(z), zip_bytes=z.stat().st_size, uid=row["SeriesInstanceUID"])
        rec["patients"][pid] = r
        G.write_json(lp, rec)
        L.event(f"[look-labels] {pid} missing {r['missing']} sanity {r['sanity_descriptive']}")
    rec["finished"] = G.now()
    G.write_json(lp, rec)
    os.chmod(lp, 0o444)
    L.prereg_line("LABELS", lp, f"lctsc_t23 look: {len(rec['patients'])} labels converted (lctsc_raster); missing structures "
                                f"{ {p: v['missing'] for p, v in rec['patients'].items() if v['missing']} }")
    return 0


if __name__ == "__main__":
    sys.exit(main())
