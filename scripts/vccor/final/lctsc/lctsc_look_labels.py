"""vccor/final LCTSC LOOK, part 1 of 2: the TEST labels (CPU, network; runs in external/lctsc_env; ONE look).
Revised after the independent review (B1, B2, SF2, SF4, N7, N8).

Gates, in this order, before any test RTSTRUCT byte is requested (a new look):
  1. no second look: look/GT_OPENED and look/results.json must not exist;
  2. the freeze (lctsc_gates.freeze_ok) and the public gate (lctsc_gates.gate_public: upload gate, logged deposit holding the
     current freeze, public after the upload, before the stop rule Sat 2026-11-21 12:00 KST);
  3. ARMS (B2): BEFORE the cutoff (Thu 2026-11-19 18:00 KST) every arm must be DONE in mode 'real' or declared NOT RUN by
     the prediction runner's binding NOTRUN line (two genuine failures, or its training draw NOT RUN); AFTER the cutoff an
     arm not DONE by the cutoff is NOT RUN; F0 and at least one F1b arm must be DONE;
  4. the arms' maps are hashed into look/maps.json (each raw map against its DONE record).
Then look/GT_OPENED is created O_EXCL and a 'GT_OPENED sha256 ...' line is appended (re-appended by --continue if a crash
left it out; N7). The look is defined by the GT_OPENED time: --continue may finish it after the stop rule (N8).

Download (B1): per test patient, the RTSTRUCT series of tcia_manifest.json (exact SeriesInstanceUID) via NBIA v1 getImage
into <pid>.part; the .part is verified BEFORE it is renamed: a readable zip, CRCs OK, exactly the manifest's ImageCount
.dcm members whose uncompressed bytes equal the TCIA FileSize. A network / HTTP error, an empty or HTML body, a BadZipFile
or a CRC error is TEMPORARY: retried after 60 s, 300 s, 900 s, then every 900 s until the stop rule (Sat 2026-11-21
12:00 KST); if the download still fails then, the look is reported as NOT COMPLETED (a NOTE line) and ends. Only a PERSISTENT
manifest mismatch -- a well-formed zip whose member count or byte total differs from the manifest on 3 consecutive downloads
-- logs GT_ABORTED (the data differ from the registered manifest) and stops the look for good. A stored zip is re-verified
on --continue (a bad one is renamed aside as <pid>.bad_<n>, never deleted, and fetched again).
Conversion (SF2): lctsc_raster (rt-utils 1.2.7 geometry, union / hole fill rule, per-ROI outcomes; never an exception for a
bad contour); the RTSTRUCT's referenced series is checked against the CT's SeriesInstanceUID; per patient a descriptive
sanity record (grid, ml, mean HU, slices, laterality: lctsc_raster.sanity) -- recorded, never an abort; MISSING structures
are listed (registered scoring rule: not evaluable for that patient). The label file is written, then its record (SF4);
a rerun re-verifies the file against the record. labels.json and a 'LABELS sha256 ...' line close the step.

usage:  external/lctsc_env/bin/python -I -B scripts/vccor/final/lctsc/lctsc_look_labels.py [--check | --continue]

Version of 2026-11-13.
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
sys.path.insert(0, str(HERE))
import lctsc_common as C  # noqa: E402
import lctsc_gates as G  # noqa: E402
import lctsc_log as L  # noqa: E402

API = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
ZDIR = (Path(os.environ["LCTSC_REHEARSAL_ROOT"]) / "zips") if G.REHEARSAL else C.DATA / "test_rtstruct_zips"
LDIR = G.LOOK / "labels"
MARKER = G.LOOK / "GT_OPENED"
RETRY_S = [60, 300, 900]
PERSISTENT_N = 3
FAULTS = os.environ.get("LCTSC_REHEARSAL_FAULTS", "") if G.REHEARSAL else ""    # rehearsal-only fault injection


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
    return any(" GT_ABORTED lctsc " in line for line in open(G.PREREG_LOG))


def not_completed():
    if not G.PREREG_LOG.is_file():
        return False
    return any(" NOTE lctsc look not completed" in line for line in open(G.PREREG_LOG))


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
    return arm_states()


def series_of(pid):
    rows = [r for r in C.manifest()["series"] if r["PatientID"] == pid and r["Modality"] == "RTSTRUCT"]
    assert len(rows) == 1 and (C.is_test(pid) or G.REHEARSAL), pid
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


def download_part(pid, row, part):
    if G.REHEARSAL:                                   # stand-in: the TRAINING RTSTRUCT zip already on disk
        assert not C.is_test(pid) and pid in C.draw_patients(), pid
        src = C.ZIPS / f"{pid}_RTSTRUCT.zip"
        n = int(fault("corrupt", pid)[2]) if fault("corrupt", pid) else 0
        k = len(list(ZDIR.glob(f".{pid}.attempt*")))
        Path(ZDIR / f".{pid}.attempt{k}").touch()
        if k < n:
            part.write_bytes(b"<html>maintenance</html>")
            return
        if fault("mismatch", pid):
            data = bytearray(src.read_bytes())
            import io
            zi = zipfile.ZipFile(io.BytesIO(bytes(data)))
            with zipfile.ZipFile(part, "w") as zo:
                for m in zi.infolist():
                    zo.writestr(m, zi.read(m) + b"\0")
            return
        part.write_bytes(src.read_bytes())
        return
    req = urllib.request.Request(f"{API}/getImage?SeriesInstanceUID={row['SeriesInstanceUID']}", headers={"User-Agent": "lctsc-look"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r, open(part, "wb") as f:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b)
    except Exception as e:  # noqa: BLE001  (any network / HTTP fault is temporary)
        raise Temporary(f"download error {type(e).__name__}: {str(e)[:120]}")


def fetch(pid):
    row = series_of(pid)
    ZDIR.mkdir(parents=True, exist_ok=True)
    dst = ZDIR / f"{pid}_RTSTRUCT.zip"
    if dst.is_file():
        try:
            if check_zip(dst, row) is None:
                return dst, row
        except Temporary:
            pass
        k = 1
        while (ZDIR / f"{pid}.bad_{k}").exists():
            k += 1
        os.replace(dst, ZDIR / f"{pid}.bad_{k}")
        L.event(f"[look-labels] {pid}: stored zip failed its check; renamed aside to {pid}.bad_{k}")
    mismatches, waits = 0, 0
    while True:
        part = ZDIR / f".{pid}.part"
        try:
            download_part(pid, row, part)
            mm = check_zip(part, row)
            if mm is None:
                os.replace(part, dst)
                return dst, row
            mismatches += 1
            L.event(f"[look-labels] {pid}: manifest mismatch {mismatches}/{PERSISTENT_N} ({mm})")
            if mismatches >= PERSISTENT_N:
                L.prereg_note("GT_ABORTED", f"test RTSTRUCT {pid}: {PERSISTENT_N} consecutive well-formed downloads differ from "
                                            f"the registered manifest ({mm}); the look is aborted")
                raise G.Refuse(f"{pid}: persistent manifest mismatch: look aborted")
        except Temporary as e:
            mismatches = 0
            L.event(f"[look-labels] {pid}: temporary fault ({e}); retrying")
        if part.exists():
            os.replace(part, ZDIR / f".{pid}.part_failed")
        if not G.before_stop():                       # review N1: retries end at the stop rule
            L.prereg_note("NOTE", f"look not completed: the test RTSTRUCT of {pid} could not be downloaded before the stop rule "
                                  f"(temporary faults only; the look stays unfinished and is reported)")
            raise G.Refuse("stop rule passed while the download was failing: the look is reported as not completed")
        delay = RETRY_S[min(waits, len(RETRY_S) - 1)]
        waits += 1
        time.sleep(1 if G.REHEARSAL else delay)


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
        L.prereg_note("GT_ABORTED", f"test label {pid}: id out of range or grid differs from the CT")
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
        G.write_json(G.LOOK / "maps.json", maps, mode=0o444)
        fd = os.open(MARKER, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
        os.write(fd, json.dumps({"opened": G.now(), "maps_sha256": G.sha256(G.LOOK / "maps.json")}).encode())
        os.close(fd)
    if not G.logged("GT_OPENED", r"K_lctsc/(rehearsal_\d+/)?look/GT_OPENED$"):
        L.prereg_line("GT_OPENED", MARKER, f"lctsc look: test RTSTRUCTs are requested only after this line; arms DONE {done}; "
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
    L.prereg_line("LABELS", lp, f"lctsc look: {len(rec['patients'])} test labels converted (lctsc_raster); missing structures "
                                f"{ {p: v['missing'] for p, v in rec['patients'].items() if v['missing']} }")
    return 0


if __name__ == "__main__":
    sys.exit(main())
