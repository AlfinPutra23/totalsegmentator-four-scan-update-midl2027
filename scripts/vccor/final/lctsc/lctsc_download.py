"""vccor/final LCTSC DOWNLOAD (CPU, network; idempotent; standard library only).

Fetches the public TCIA series list of LCTSC (Lung CT Segmentation Challenge 2017; CC BY 3.0;
DOI 10.7937/K9/TCIA.2017.3R3FVZ08) and writes the series manifest K_lctsc/tcia_manifest.json (every series: UID,
patient, modality, image count, TCIA-listed file size, time stamp) BEFORE downloading anything. Then downloads, through the
public NBIA v1 getImage endpoint, the CT series of all 60 patients and the RTSTRUCT series of the 36 TRAINING patients
only. The 24 test RTSTRUCT series are never requested here (asserted per request): they are fetched only by the look,
after the public-gate file exists.

Each download is a zip in data_lctsc/tcia_zips/<PatientID>_<Modality>.zip; it is checked (zip CRCs, member count ==
TCIA ImageCount) and its sha256 and size go to K_lctsc/download.json. Nothing is extracted here.

usage:  external/lctsc_env/bin/python -I -B scripts/vccor/final/lctsc/lctsc_download.py [--list-only]

Version of 2026-11-13.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
import zipfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

REPO = Path("/home/user/Documents/Medical Imagining SSL")
OUT = REPO / "outputs/results/vccor/final/K_lctsc"
ZIPS = REPO / "data_lctsc/tcia_zips"
LOG = REPO / "outputs/logs/vccor/final_lctsc_download.log"
API = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
KST = timezone(timedelta(hours=9))


def now():
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def wj(path, obj):
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    json.dump(obj, open(tmp, "w"), indent=1)
    os.replace(tmp, path)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def get_json(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "lctsc-download"}), timeout=120) as r:
        return json.loads(r.read().decode())


def is_test(pid):
    return pid.startswith("LCTSC-Test-")


def manifest():
    series = get_json(f"{API}/getSeries?Collection=LCTSC&format=json")
    keep = ("SeriesInstanceUID", "StudyInstanceUID", "Modality", "PatientID", "ImageCount", "FileSize", "TimeStamp",
            "LicenseName", "SeriesDescription", "SeriesNumber", "Manufacturer", "ManufacturerModelName")
    rows = sorted(({k: s.get(k) for k in keep} for s in series), key=lambda r: (r["PatientID"], r["Modality"]))
    path = OUT / "tcia_manifest.json"
    if path.exists():
        old = json.load(open(path))
        if old["series"] != rows:
            raise SystemExit("TCIA series list differs from the stored manifest: stop and report")
        return old
    m = {"fetched_kst": now(), "endpoint": f"{API}/getSeries?Collection=LCTSC&format=json", "n_series": len(rows),
         "n_patients": len({r["PatientID"] for r in rows}), "series": rows,
         "note": "written before any download; the 24 test RTSTRUCT series are identified here by UID, TCIA ImageCount and "
                 "TCIA FileSize and are not downloaded before the public gate"}
    wj(path, m)
    log(f"MANIFEST {len(rows)} series, sha256 {sha256(path)}")
    return m


def download(row, rec):
    pid, mod, uid = row["PatientID"], row["Modality"], row["SeriesInstanceUID"]
    if mod == "RTSTRUCT" and is_test(pid):
        raise SystemExit(f"refuse: test RTSTRUCT {pid}")
    name = f"{pid}_{mod}.zip"
    if name in rec["files"]:
        return
    dst = ZIPS / name
    tmp = ZIPS / f".{name}.part"
    for attempt in range(1, 4):
        try:
            req = urllib.request.Request(f"{API}/getImage?SeriesInstanceUID={uid}", headers={"User-Agent": "lctsc-download"})
            with urllib.request.urlopen(req, timeout=600) as r, open(tmp, "wb") as f:
                while True:
                    b = r.read(1 << 22)
                    if not b:
                        break
                    f.write(b)
            with zipfile.ZipFile(tmp) as z:
                bad = z.testzip()
                n = len([i for i in z.infolist() if not i.is_dir() and i.filename.lower().endswith(".dcm")])
            if bad is not None:
                raise RuntimeError(f"bad member {bad}")
            if n != int(row["ImageCount"]):
                raise RuntimeError(f"{n} dcm members, TCIA ImageCount {row['ImageCount']}")
            os.replace(tmp, dst)
            rec["files"][name] = {"uid": uid, "patient": pid, "modality": mod, "n_dcm": n, "bytes": dst.stat().st_size,
                                  "sha256": sha256(dst), "tcia_filesize": row["FileSize"], "downloaded_kst": now()}
            wj(OUT / "download.json", rec)
            log(f"OK {name} {n} dcm {dst.stat().st_size} B")
            return
        except Exception as e:  # noqa: BLE001
            log(f"RETRY {name} attempt {attempt}: {e}")
            if tmp.exists():
                tmp.unlink()
            time.sleep(10 * attempt)
    raise SystemExit(f"download failed: {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-only", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ZIPS.mkdir(parents=True, exist_ok=True)
    m = manifest()
    if a.list_only:
        return
    rp = OUT / "download.json"
    rec = json.load(open(rp)) if rp.exists() else {"started_kst": now(), "files": {}}
    todo = [r for r in m["series"] if r["Modality"] == "CT" or (r["Modality"] == "RTSTRUCT" and not is_test(r["PatientID"]))]
    todo.sort(key=lambda r: (r["Modality"] != "RTSTRUCT", is_test(r["PatientID"]), r["PatientID"]))
    log(f"START {len(todo)} series ({sum(int(r['FileSize']) for r in todo) / 1e9:.2f} GB TCIA-listed)")
    for r in todo:
        st = os.statvfs("/")
        if st.f_bavail * st.f_frsize / 1024 ** 3 < 27:
            raise SystemExit("refuse: / under 27 GiB free")
        download(r, rec)
    rec["finished_kst"] = now()
    wj(rp, rec)
    log(f"DONE {len(rec['files'])} files")


if __name__ == "__main__":
    main()
