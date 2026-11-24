"""vccor/final LCTSC gates and arms (standard library only, so that both the SSL .venv and external/lctsc_env can import
it). == FLARE-T50's t50_common gates, with LCTSC's paths and dates.

  freeze        K_lctsc/code_freeze.json lists the sha256 of every study file (code, frozen imports, records); freeze_ok()
                re-hashes them (a change refuses every gated step).
  upload gate   K_lctsc/gate_osf_uploaded.json, written only after the OSF upload, keys osf_url,
                uploaded_kst, written_by, evidence, deposit_zip_sha256, freeze_sha256; the deposit sha must equal a
                'DEPOSIT sha256 <h> ...lctsc...zip' line of final_prereg.log, the freeze sha the current code_freeze.json,
                and the upload must be after the freeze. Required before any test-image prediction.
  public gate   K_lctsc/gate_osf_public.json (written by hand): keys osf_url, public_checked_kst, api_http_status (200),
                public (true), written_by, evidence; checked after the upload and before the stop rule. Required before any
                test RTSTRUCT is downloaded.
  cutoff        Thu 2026-11-19 18:00 KST: no prediction starts at or after it; a DONE.json written at or after it makes the
                arm NOT RUN. Stop rule Sat 2026-11-21 12:00 KST: no look after it.
  look          K_lctsc/look/GT_OPENED (O_EXCL) is created before the first test RTSTRUCT byte is requested; one look.
"""
import hashlib
import json
import os
import re
import shutil
import time
from datetime import datetime
from pathlib import Path

REPO = Path("/home/user/Documents/Medical Imagining SSL")
OUT = REPO / "outputs/results/vccor/final/K_lctsc"
PRED = OUT / "pred"
LOOK = OUT / "look"
MODELS = OUT / "models"
FREEZE = OUT / "code_freeze.json"
GATE_UPLOADED = OUT / "gate_osf_uploaded.json"
GATE_PUBLIC = OUT / "gate_osf_public.json"
PREREG_LOG = REPO / "outputs/logs/vccor/final_prereg.log"
REHEARSAL_ROOT = os.environ.get("LCTSC_REHEARSAL_ROOT")
REHEARSAL = bool(REHEARSAL_ROOT)
if REHEARSAL:                         # the look rehearsal: every path of the look under a scratch root (stand-in patients)
    _R = Path(REHEARSAL_ROOT)
    assert _R.resolve() != OUT.resolve() and "rehearsal" in _R.name, "the rehearsal root must be a separate 'rehearsal' folder"
    PRED, LOOK, FREEZE = _R / "pred", _R / "look", _R / "code_freeze.json"
    GATE_UPLOADED, GATE_PUBLIC, PREREG_LOG = _R / "gate_osf_uploaded.json", _R / "gate_osf_public.json", _R / "prereg.log"
CUTOFF = "2026-11-19 18:00:00"
STOP = "2026-11-21 12:00:00"
ARMS = {"F0": {"kind": "ts", "run": None}, "F1b_S1": {"kind": "ts", "run": "F1b_S1"},     # site-matched draws S1-S3
        "F1b_S2": {"kind": "ts", "run": "F1b_S2"}, "F1b_S3": {"kind": "ts", "run": "F1b_S3"}}
DRAW_ARM = {"S1": "F1b_S1", "S2": "F1b_S2", "S3": "F1b_S3"}
CAPTURE = (292, 294, 295)            # the parts every F1b arm keeps from TS: F0 captures them, the F1b arms replay them
TS_EPOCHS = 50


class Refuse(SystemExit):
    pass


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ts(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def kst_gate():
    if time.strftime("%Z") not in ("KST",) and time.timezone != -9 * 3600:
        raise Refuse("the machine clock is not KST")


def _clock():
    """The current time; in the rehearsal only, LCTSC_REHEARSAL_NOW may set it (to rehearse the after-cutoff rules)."""
    if REHEARSAL and os.environ.get("LCTSC_REHEARSAL_NOW"):
        return os.environ["LCTSC_REHEARSAL_NOW"]
    return now()


def before_cutoff(t=None):
    return ts(t or _clock()) < ts(CUTOFF)


def before_stop(t=None):
    return ts(t or _clock()) < ts(STOP)


def disk_gate(need_gib, what, floor=25.0):
    free = shutil.disk_usage("/").free / 1024 ** 3
    if free - need_gib < floor:
        raise Refuse(f"{what}: / has {free:.1f} GiB free; {need_gib} GiB needed above the {floor} GiB floor")
    return free


def freeze_ok():
    if not FREEZE.is_file():
        raise Refuse("no code freeze")
    fz = json.load(open(FREEZE))
    bad = [p for p, h in fz["files"].items() if not Path(p if p.startswith("/") else REPO / p).is_file()
           or sha256(p if p.startswith("/") else REPO / p) != h]
    if bad:
        raise Refuse(f"frozen files changed: {bad[:5]}")
    return fz


def logged_path(kind, pattern):
    """path of the last final_prereg.log line '[t] KIND sha256 <h> <path>' whose path matches pattern."""
    hit = None
    if PREREG_LOG.is_file():
        for line in open(PREREG_LOG):
            m = re.match(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] (\S+) sha256 ([0-9a-f]{64}) (\S+)", line)
            if m and m.group(2) == kind and re.search(pattern, m.group(4)):
                hit = m.group(4)
    if hit is None:
        raise Refuse(f"no {kind} line matching {pattern}")
    return hit


def logged(kind, pattern):
    """(time, sha) of the last final_prereg.log line '[t] KIND sha256 <h> <path...>' whose path matches pattern."""
    hit = None
    if PREREG_LOG.is_file():
        for line in open(PREREG_LOG):
            m = re.match(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] (\S+) sha256 ([0-9a-f]{64}) (\S+)", line)
            if m and m.group(2) == kind and re.search(pattern, m.group(4)):
                hit = (m.group(1), m.group(3))
    return hit


def gate_uploaded():
    if not GATE_UPLOADED.is_file():
        raise Refuse("upload gate missing (the coordinator writes it after the owner's OSF upload)")
    g = json.load(open(GATE_UPLOADED))
    need = ("osf_url", "uploaded_kst", "written_by", "evidence", "deposit_zip_sha256", "freeze_sha256")
    miss = [k for k in need if not g.get(k)]
    if miss:
        raise Refuse(f"upload gate lacks {miss}")
    dep = logged("DEPOSIT", r"lctsc.*\.zip$")
    if not dep or dep[1] != g["deposit_zip_sha256"]:
        raise Refuse("upload gate: deposit sha256 does not match the logged LCTSC DEPOSIT line")
    if g["freeze_sha256"] != sha256(FREEZE):
        raise Refuse("upload gate: freeze sha256 differs from code_freeze.json")
    fz = json.load(open(FREEZE))
    if ts(g["uploaded_kst"]) <= ts(fz["frozen_kst"]):
        raise Refuse("upload gate: the upload is not after the freeze")
    zp = REPO / logged_path("DEPOSIT", r"lctsc.*\.zip$")              # review N6: the deposit holds THIS freeze
    import zipfile
    with zipfile.ZipFile(zp) as z:
        hits = [n for n in z.namelist() if n.endswith("code_freeze.json")]
        if len(hits) != 1 or hashlib.sha256(z.read(hits[0])).hexdigest() != g["freeze_sha256"]:
            raise Refuse("upload gate: the deposit zip does not hold the current code_freeze.json")
    return g


def gate_public(allow_after_stop=False):
    """allow_after_stop: --continue of a look whose GT_OPENED came before the stop rule (review N8)."""
    g = gate_uploaded()
    if not GATE_PUBLIC.is_file():
        raise Refuse("public gate missing (the coordinator writes it once OSF shows the registration public)")
    p = json.load(open(GATE_PUBLIC))
    need = ("osf_url", "public_checked_kst", "api_http_status", "public", "written_by", "evidence")
    miss = [k for k in need if p.get(k) in (None, "")]
    if miss:
        raise Refuse(f"public gate lacks {miss}")
    if p["api_http_status"] != 200 or p["public"] is not True:
        raise Refuse("public gate: the OSF API did not report the registration public")
    if p["osf_url"].rstrip("/") != g["osf_url"].rstrip("/"):
        raise Refuse("public gate: OSF URL differs from the upload gate")
    if ts(p["public_checked_kst"]) < ts(g["uploaded_kst"]):
        raise Refuse("public gate: checked before the upload")
    if not before_stop(p["public_checked_kst"]) or (not before_stop() and not allow_after_stop):
        raise Refuse("stop rule passed (Sat 2026-11-21 12:00 KST): no look")
    return g, p


def test_ids():
    """The look's patients: LCTSC's 24 test patients; in the rehearsal, the stand-in TRAINING patients of standin.json."""
    if REHEARSAL:
        return json.load(open(Path(REHEARSAL_ROOT) / "standin.json"))["ids"]
    m = json.load(open(OUT / "tcia_manifest.json"))
    return sorted({r["PatientID"] for r in m["series"] if r["PatientID"].startswith("LCTSC-Test-")})


def notrun_declared(arm):
    """True iff final_prereg.log holds a binding 'NOTRUN lctsc <arm> ...' line (review NS1: it beats a DONE.json)."""
    if not PREREG_LOG.is_file():
        return False
    return any(f" NOTRUN lctsc {arm} " in line for line in open(PREREG_LOG))


def frozen_draw_status(arm):
    """The frozen training status of an arm's draw (code_freeze.json 'draws_status'); 'DONE' for F0."""
    run = ARMS[arm]["run"]
    if run is None:
        return "DONE"
    fz = json.load(open(FREEZE))
    return fz["draws_status"][run]["status"]


def look_made():
    return (LOOK / "GT_OPENED").exists() or (LOOK / "results.json").exists()


def pred_dir(arm, root=None):
    return Path(root or PRED) / arm


def arm_done(arm, root=None):
    """(state, record): 'DONE' only for a DONE.json written before the cutoff; 'LATE' (= NOT RUN) at or after it."""
    p = pred_dir(arm, root) / "DONE.json"
    if not p.is_file():
        return "MISSING", None
    d = json.load(open(p))
    return ("DONE" if before_cutoff(d["finished"]) else "LATE"), d


def write_json(path, obj, mode=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    json.dump(obj, open(tmp, "w"), indent=1, default=str)
    os.replace(tmp, path)
    if mode is not None:
        os.chmod(path, mode)
    return sha256(path)
