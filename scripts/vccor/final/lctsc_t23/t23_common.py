"""vccor/final LCTSC-T23: shared constants (standard library only; both the SSL .venv and external/lctsc_env import it).

LCTSC-T23 = one registered single look on the 23 LCTSC TRAINING patients whose labels were never opened, with the frozen
LCTSC models of osf.io/d6ayj (F0 = TotalSegmentator 2.13.0 zero-shot; F1b_S1..S3 = the site-matched gentle fine-tunes of
2026-11-13). No training. Approved on 2026-11-17.

The 23 patients = LCTSC's 36 training patients (K_lctsc/tcia_manifest.json) minus the 13 whose RTSTRUCT was read before
2026-11-17: the 12 site-draw patients and LCTSC-Train-S2-003 (records: K_lctsc/labels_train.json, labels_train_v2.json,
draws_site.json, draws.json). derive_ids() applies this rule to the frozen LCTSC records; the list is fixed in
K_lctsc_t23/ids_t23.json before the freeze, and test_t23.py checks that the two agree.

Their RTSTRUCT zips were downloaded on 2026-11-13 with the other training series (sha256 and size in K_lctsc/download.json)
and were never extracted or parsed; the look verifies each zip against that record before it extracts it.

Version of 2026-11-17.
"""
import json
from datetime import datetime
from pathlib import Path

REPO = Path("/home/user/Documents/Medical Imagining SSL")
SRC = REPO / "outputs/results/vccor/final/K_lctsc"          # the frozen LCTSC (d6ayj) records and models
OUT = REPO / "outputs/results/vccor/final/K_lctsc_t23"
IDS_FILE = OUT / "ids_t23.json"
PLAN = OUT / "plan_lctsc_t23.json"
CT_RECORD = OUT / "t23_ct.json"                             # the CT DICOM re-download record (geometry for the rasteriser)
DATA = REPO / "data_lctsc"
CT_ZIPS = DATA / "t23_ct_zips"                              # CT series re-downloaded on 2026-11-17 (images only)
TAG = "lctsc_t23"

PLAN_FREEZE_DEADLINE = "2026-11-18 18:00:00"                # go/no-go gate 1 (plan, code and deposit frozen)
PUBLIC_DEADLINE = "2026-11-19 12:00:00"                     # gate 2: the public gate must be checked before this time
CUTOFF = "2026-11-20 10:00:00"                              # no prediction starts at or after it; a later DONE = NOT RUN
CUTOFF_DT = datetime(2026, 11, 20, 10, 0)
STOP = "2026-11-20 16:00:00"                                # no GT_OPENED at or after it (gate 3: look done on Fri 20 Nov)

STANDINS = ["LCTSC-Train-S1-003", "LCTSC-Train-S1-004", "LCTSC-Train-S2-004", "LCTSC-Train-S3-002"]   # rehearsal / smoke
# stand-ins: site-draw TRAINING patients whose labels were read on 2026-11-13 for training (never an LCTSC-T23 patient)


def _train_ids():
    m = json.load(open(SRC / "tcia_manifest.json"))
    return sorted({r["PatientID"] for r in m["series"] if r["PatientID"].startswith("LCTSC-Train-")})


def read_before():
    """Training patients whose RTSTRUCT was read before 2026-11-17, from the frozen LCTSC records."""
    read = set()
    for f in ("labels_train.json", "labels_train_v2.json"):
        d = json.load(open(SRC / f))
        read |= set(d.get("rtstruct_opened", [])) | set(d.get("patients", {}))
    ds = json.load(open(SRC / "draws_site.json"))
    read |= set(ds.get("already_read_training_labels_at_fixing", []))
    for v in ds["draws"].values():
        read |= set(v)
    for v in json.load(open(SRC / "draws.json"))["draws"].values():
        read |= set(v)
    return read


def derive_ids():
    rb = read_before()
    return [p for p in _train_ids() if p not in rb]


def ids():
    return list(json.load(open(IDS_FILE))["ids"])


def site(pid):
    return pid.split("-")[2]
