"""vccor/final WORD-N: shared constants, arm registry, gates and helpers.

WORD-N is GO-2 of the design of 2026-11-09 (AC.md, section GO-2; the
volume idea from R1, E1), adopted on 2026-11-13 (with a VISTA3D tier; overshoot reading split, option A). One registered look at the 20 official WORD validation scans
(never opened: their images and labels were never extracted; final_prereg.log approval lines 2026-11-13 09:31:37 and
09:49:49), scored with the EXISTING WORD models (tier 1: F0, F1b d1-d3, STOCK d1-d3 and the CPU arms MERGE_N / LABELMAP),
plus tier 2 (VISTA3D: V0 and VF5k d1-d3 trained on the 12 already-read draw labels, and VF5k_lrA_d1). The 88 unused WORD
training scans stay sealed (their images and labels are never extracted); the WORD test labels are never read again.

Order (binding):
  before the registration: code, unit tests, the look rehearsal (stand-in data under /tmp), the VISTA3D smoke test (draw
    scans or synthetic data only), the archive metadata pin (member names / sizes, no content), the freeze, the deposit;
  after the gate_osf_uploaded.json record (written once the OSF registration was initiated; it names the deposit zip and
    the code freeze): the whitelist extraction of the 20 validation IMAGES, the registered duplicate rule, every
    prediction and tier-2 training;
  after the gate_osf_public.json record (OSF API: public): the single look (GT_OPENED marker, then the 20
    labelsVal members extracted, size-checked, hashed and archived, then scoring).
Cutoff Thu 2026-11-19 18:00 KST: an arm not DONE by then is NOT RUN. Stop rule: if the registration is not public by
Sat 2026-11-21 12:00 KST the validation labels stay sealed (the images may have been extracted at the upload gate).
Write area (binding): outputs/results/vccor/final/I_wordn/ (records, predictions, look, event logs under I_wordn/logs/),
the WORD image folder MagicNet/data/word_v010/images/ (the 20 validation images only), outputs/logs/vccor/final_prereg.log
(registration lines), and the one deposit outputs/results/vccor/osf_deposit_<date>_wordn/ and its .zip.
GPU (binding): every GPU entry point (wn_predict real mode, wn_vista --smoke / --train) refuses unless CUDA_VISIBLE_DEVICES
is GPU 1's UUID (bus 00000000:05:00.0) and takes the shared flock itself. Every wait after the upload gate (the lock, GPU 1
idle, disk space) is bounded by the cutoff only (no 12 h timeout); the pre-registration smoke test keeps a 12 h bound.

use_root(ROOT) redirects every study path (incl. the archive, the image folder and the TotalSegmentator root) under ROOT
(the look rehearsal and the unit tests only). No import-time side effects beyond sys.path (no file is read or written).
Version of 2026-11-13.
"""
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
FINAL = HERE.parent
for _p in (str(FINAL), str(HERE), str(FINAL / "word_a1"), str(FINAL / "flaret50"), str(FINAL.parent / "m2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import final_common as F  # noqa: E402
import word_common as W  # noqa: E402

REPO = F.REPO
VC = F.VC
VENV = F.VENV
TAG = "wordn"

VAL_IDS = ["0001", "0007", "0015", "0025", "0031", "0035", "0039", "0045", "0048", "0066", "0075", "0080", "0083", "0085",
           "0098", "0112", "0137", "0139", "0141", "0149"]
ARCHIVE_SHA256 = "1ef1e48f8b41d72d733c5cdd91e1238d9b345d88c0de0c902e356310901f3115"      # C_word/download.json
ARCHIVE_BYTES = 6809125051
SEALED_TWIN = {"0083": "0095"}           # header twin (prep/overlap.json); 0095 is one of the 88 sealed scans: never used

CUTOFF = datetime(2026, 11, 19, 18, 0)
STOP_RULE = datetime(2026, 11, 21, 12, 0)
FLOOR_GIB = 25.0
GPU_LOCK = Path("/home/user/Documents/Handpose_project/runs/method_pilot3/gpu1.lock")
GPU_BUS = "00000000:05:00.0"
GPU_IDLE_MIB = 3000
GPU_WAIT_S = 12 * 3600                  # only for waits that are NOT cutoff-aware (the pre-registration smoke test)
DISK_POLL_S = 300
GPU_POLL_S = 60
GPU1_UUID = "GPU-1a7b7614-499a-6540-55fd-5920dcee8340"     # nvidia-smi: index 1, bus 00000000:05:00.0
GPU_ENV = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": GPU1_UUID,
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
CPU_ENV = {"CUDA_VISIBLE_DEVICES": ""}
TS_EPOCHS = 50
TS_VERSION = "2.13.0"
CAPTURE = (292, 293, 294, 295)
VERSION_PACKAGES = ["numpy", "scipy", "nibabel", "torch", "nnunetv2", "TotalSegmentator", "monai"]
DUP_R = 0.999                            # Plan C / A1 image rule: Pearson r on a 4-voxel stride grid
LABEL_MAX_RAW = 16                       # WORD raw label ids 0..16

N_ORG = W.N_ORG                          # 15 analysis organs (word_common)
PRIMARY = list(W.PRIMARY)                # 12 primary organs
SOLID = [1, 2, 3, 4, 8]                  # liver, spleen, kidney_left, kidney_right, pancreas (analysis ids)
DUODENUM, ESOPHAGUS = 9, 7

# ---------------------------------------------------------------------------------------------------- paths (use_root)
REHEARSAL = False


def _bind(out, prereg_log, events_log, archive, images, ts_root, c_word):
    g = globals()
    g.update(OUT=out, PRED=out / "pred", LOOK=out / "look", RUN=out / "run", INTAKE=out / "intake",
             VISTA=out / "vista", VSMOKE=out / "vsmoke", PLAN=out / "plan_wordn.json",
             MEMBERS=out / "archive_members.json", IMAGES_RECORD=out / "intake" / "images_record.json",
             EXCLUSIONS=out / "intake" / "exclusions.json", GATE_UPLOADED=out / "gate_osf_uploaded.json",
             GATE_PUBLIC=out / "gate_osf_public.json", FREEZE=out / "code_freeze.json",
             REHEARSAL_RECORD=out / "rehearsal.json", UNIT_TESTS=out / "unit_tests.json", PREREG_LOG=prereg_log,
             EVENTS_LOG=events_log, ARCHIVE=archive, IMAGES=images, TS_ROOT=ts_root, C_WORD=c_word)
    g["LBL_DIR"] = out / "look" / "labels"                                     # created only inside the look


_bind(F.OUT / "I_wordn", F.PREREG_LOG, F.OUT / "I_wordn" / "logs" / "events.log", W.ARCHIVE, W.IMAGES, Path.home() / ".totalsegmentator",
      W.DIR)


def use_root(root, ts_root=None, c_word=None):
    """Redirect every study path under root (look rehearsal / unit tests). Sets REHEARSAL = True."""
    global REHEARSAL
    root = Path(root)
    assert str(root.resolve()).startswith("/tmp/"), f"use_root only under /tmp: {root}"
    _bind(root / "I_wordn", root / "logs" / "final_prereg.log", root / "I_wordn" / "logs" / "events.log",
          root / "archive" / "WORD-V0.1.0.zip", root / "images", Path(ts_root) if ts_root else root / "totalsegmentator",
          Path(c_word) if c_word else root / "C_word")
    REHEARSAL = True
    (root / "logs").mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------------------------------- arms
def _cw(*p):
    return Path(C_WORD, "models", *p)


DRAWS = ["d1", "d2", "d3"]
VF_RUNS = {"VF5k_d1": dict(draw="d1", lr=5e-6, n_iter=5000), "VF5k_d2": dict(draw="d2", lr=5e-6, n_iter=5000),
           "VF5k_d3": dict(draw="d3", lr=5e-6, n_iter=5000), "VF5k_lrA_d1": dict(draw="d1", lr=5e-5, n_iter=5000)}
VISTA_RECIPE = W.VISTA_RECIPE            # "R1" (the vista block's probe decision; Plan C)


def arms():
    """The arm registry (paths resolved at call time so use_root applies).
    kind 'ts': TotalSegmentator 2.13.0 'total' with part 291 redirected to the WORD F1b export (as in A1);
    kind 'nnunet': the A1.2 STOCK network (WORD's 15 analysis ids, fold 'all', 250 epochs);
    kind 'cpu': MERGE_N / LABELMAP built from stored maps (A1.2 operators, unchanged);
    kind 'vista': VISTA3D 0.5.8 automatic branch (V0 = the bundle's weights; VF = a WORD K = 4 export)."""
    A = {"F0": dict(kind="ts", tier=1, parts={}, lut="ts", role="zero-shot",
                    label="TotalSegmentator 2.13.0 'total', its own weights (captures 292-295)")}
    for d in DRAWS:
        A[f"F1b_{d}"] = dict(kind="ts", tier=1, lut="ts", role="F1b", draw=d, parts={291: _cw(f"F1b_{d}", "part291")},
                             check={"run": f"F1b_{d}", "variant": "F1b", "trainable": "all", "initial_lr": 1e-4,
                                    "labelled_ids": W.draw(d)},
                             label=f"WORD F1b, K = 4, draw {d} (A1, part 291)")
    for d in DRAWS:
        A[f"STOCK_{d}"] = dict(kind="nnunet", tier=1, lut="id", role="STOCK", draw=d, dir=_cw(f"STOCK_{d}", "stock"),
                               fold="all", n_classes=N_ORG, epochs=250,
                               check={"run": f"STOCK_{d}", "variant": "STOCK", "labelled_ids": W.draw(d)},
                               label=f"WORD STOCK (A1.2, nnU-Net's stock fine-tune), draw {d}")
    for d in DRAWS:
        A[f"MERGE_N_{d}"] = dict(kind="cpu", tier=1, lut="ts", role="MERGE_N", draw=d, inputs=["F0", f"F1b_{d}"],
                                 label=f"A1.2 MERGE_N (non-target classes from F0), draw {d}")
    for d in DRAWS:
        A[f"LABELMAP_{d}"] = dict(kind="cpu", tier=1, lut="id", role="LABELMAP", draw=d, inputs=["F0"],
                                  rules=Path(C_WORD, "pred", f"LABELMAP_{d}__word_test30", "rules.json"),
                                  label=f"A1.2 LABELMAP (F0 relabelled by draw {d}'s frozen rules)")
    A["V0"] = dict(kind="vista", tier=2, lut="vista", role="V0", weights=None,
                   label="VISTA3D 0.5.8 zero-shot automatic branch")
    for r, s in VF_RUNS.items():
        A[r] = dict(kind="vista", tier=2, lut="vista", role="VF_lrA" if "lrA" in r else "VF5k", draw=s["draw"],
                    run=r, weights=Path(VISTA, "models", r, "weights_trainable.pt"),
                    label=f"VISTA3D R1 recipe, lr {s['lr']:g}, {s['n_iter']} it, WORD draw {s['draw']}")
    return A


TIER1 = (["F0"] + [f"F1b_{d}" for d in DRAWS] + [f"STOCK_{d}" for d in DRAWS] + [f"MERGE_N_{d}" for d in DRAWS]
         + [f"LABELMAP_{d}" for d in DRAWS])
TIER2 = ["V0", "VF5k_d1", "VF5k_d2", "VF5k_d3", "VF5k_lrA_d1"]
RUN_ORDER = TIER1 + TIER2
REQUIRED = ["F0", "F1b_d1"]
TRAIN_JOBS = list(VF_RUNS)               # tier-2 trainings (GPU), each before its own prediction


def lut_kind(arm):
    return arms()[arm]["lut"]


# ---------------------------------------------------------------------------------------------------- errors / logs
class Refuse(Exception):
    """A gate refused. code 1 = an ordinary failure, 2 = a gate or configuration error, 3 = a resource refusal (only the
    pre-registration smoke test's 12 h bounds and its disk gate), 4 = the cutoff passed while waiting (CUTOFF_CODE)."""

    def __init__(self, msg, code=2):
        super().__init__(msg)
        self.msg = msg
        self.code = code


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def sha256(p, n=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(n)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


_HCACHE = {}


def sha256_cached(p):
    st = os.stat(p)
    k = (str(Path(p).resolve()), st.st_size, st.st_mtime_ns)
    if k not in _HCACHE:
        _HCACHE[k] = sha256(p)
    return _HCACHE[k]


def rel(p):
    p = Path(p)
    try:
        return str(p.resolve().relative_to(REPO))
    except ValueError:
        return str(p)


def log_event(msg, path=None):
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    p = Path(path or EVENTS_LOG)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(line + "\n")


def prereg_line(kind, path, note, log_path=None):
    """'[time] KIND sha256 <h> <rel path> (note)' appended to final_prereg.log (the project's registration log)."""
    h = sha256(path)
    line = f"[{now()}] {kind} sha256 {h} {rel(path)} ({note})"
    p = Path(log_path or PREREG_LOG)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)
    return h


def prereg_note(kind, text, log_path=None):
    line = f"[{now()}] {kind} {TAG} {text}"
    p = Path(log_path or PREREG_LOG)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def prereg_lines():
    p = Path(PREREG_LOG)
    return open(p).read().splitlines() if p.is_file() else []


def write_json(path, obj, mode=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not os.access(path, os.W_OK):
        raise Refuse(f"{rel(path)} is read-only: not rewritten")
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(F.clean_nan(obj), f, indent=1, default=F._default)
    os.replace(tmp, path)
    if mode is not None:
        os.chmod(path, mode)


def code_hashes(names):
    return {n: sha256(HERE / n) for n in names}


def load_plan():
    if not PLAN.is_file():
        raise Refuse(f"{rel(PLAN)} missing: run wn_plan.py first")
    return json.load(open(PLAN))


def overshoot_mode(plan=None):
    """The registered switch: 'SPLIT' (option A, the author's choice of 2026-11-13) or 'AC_COMBINED' (the design's
    intersection rule). Refuses if the plan does not fix it."""
    m = (plan or load_plan()).get("overshoot_mode")
    if m not in ("SPLIT", "AC_COMBINED"):
        raise Refuse(f"plan overshoot_mode {m!r} is not fixed to SPLIT or AC_COMBINED")
    return m


# ---------------------------------------------------------------------------------------------------- versions
def versions():
    import importlib.metadata as md
    return {p: md.version(p) for p in VERSION_PACKAGES}


def check_versions(frozen=None):
    v = versions()
    if v["TotalSegmentator"] != TS_VERSION:
        raise Refuse(f"TotalSegmentator {v['TotalSegmentator']} is installed, the study needs {TS_VERSION}")
    if frozen is not None and v != frozen:
        diff = {k: (frozen.get(k), v.get(k)) for k in set(v) | set(frozen) if v.get(k) != frozen.get(k)}
        raise Refuse(f"package versions differ from the freeze: {diff}")
    return v


# ---------------------------------------------------------------------------------------------------- archive (whitelist)
SEVEN_Z = W.SEVEN_Z
PASSWORD = W.PASSWORD


def member(kind, cid):
    """Whitelisted archive members: imagesVal / labelsVal of a VALIDATION scan only (the 88 stay sealed)."""
    assert kind in ("imagesVal", "labelsVal"), kind
    assert cid in VAL_IDS, f"{cid} is not a WORD validation scan"
    return f"WORD-V0.1.0/{kind}/word_{cid}.nii.gz"


def list_members(names):
    """7z technical listing (-slt) of the named members: size, packed size, modified, method, encrypted. Central-directory
    metadata only: no member content is decrypted or read."""
    r = subprocess.run([SEVEN_Z, "l", "-slt", f"-p{PASSWORD}", str(ARCHIVE)] + list(names), capture_output=True, text=True)
    if r.returncode != 0:
        raise Refuse(f"7z listing failed: {r.stderr[-300:]}")
    out, cur = {}, None
    for ln in r.stdout.splitlines():
        if ln.startswith("Path = WORD-V0.1.0/"):
            cur = ln.split(" = ", 1)[1]
            out[cur] = {}
        elif cur and " = " in ln:
            k, v = ln.split(" = ", 1)
            if k in ("Size", "Packed Size", "Modified", "Method", "Encrypted", "CRC"):
                out[cur][k] = v
    for n in names:
        if n not in out:
            raise Refuse(f"archive member {n} not listed")
    return out


def extract_member_bytes(name):
    """One whitelisted member, decrypted into memory (7z e -so). Only imagesVal / labelsVal names reach here."""
    assert re.match(r"^WORD-V0\.1\.0/(imagesVal|labelsVal)/word_\d{4}\.nii\.gz$", name), name
    assert name[-11:-7] in VAL_IDS, name
    r = subprocess.run([SEVEN_Z, "e", "-so", f"-p{PASSWORD}", str(ARCHIVE), name], capture_output=True)
    if r.returncode != 0 or not r.stdout:
        raise Refuse(f"7z extraction of {name} failed: {r.stderr[-300:]!r}")
    return r.stdout


def archive_ok():
    """The main WORD archive is the downloaded one (size and sha256 pinned in C_word/download.json)."""
    if REHEARSAL:
        pinned = json.load(open(MEMBERS)).get("archive_sha256") if MEMBERS.is_file() else None
        if pinned and sha256_cached(ARCHIVE) != pinned:
            raise Refuse("rehearsal archive differs from its pinned sha256")
        return True
    st = os.stat(ARCHIVE)
    if st.st_size != ARCHIVE_BYTES or sha256_cached(ARCHIVE) != ARCHIVE_SHA256:
        raise Refuse("the WORD archive differs from its download record (size / sha256)")
    return True


def image_path(cid):
    return Path(IMAGES) / f"word_{cid}.nii.gz"


# ---------------------------------------------------------------------------------------------------- gates
def kst_gate():
    if time.strftime("%Z") != "KST":
        raise Refuse(f"local time zone is {time.strftime('%Z')}, not KST")


def disk_free_gib(path=None):
    st = os.statvfs(str(path or REPO))
    return st.f_bavail * st.f_frsize / 1024 ** 3


def disk_gate(need_gib, what):
    free = disk_free_gib()
    if free - float(need_gib) < FLOOR_GIB:
        raise Refuse(f"{what}: / has {free:.2f} GiB free; with {need_gib:.2f} GiB more it would fall below the "
                     f"{FLOOR_GIB:.0f} GiB floor", code=3)
    return free


def wait_disk(need_gib, what, log=print, poll_s=None):
    """A cutoff-aware wait for disk space (NB1): returns once / keeps the floor with need_gib more; Refuse code 4 once the
    cutoff passes. Never a resource refusal."""
    said = False
    while True:
        try:
            return disk_gate(need_gib, what)
        except Refuse as e:
            if not before_cutoff():
                raise Refuse(f"the cutoff passed while waiting for disk space ({e.msg})", code=CUTOFF_CODE)
            if not said:
                log(f"{e.msg}: waiting for space")
                said = True
            time.sleep(poll_s or DISK_POLL_S)


def before_cutoff(t=None):
    return (t or datetime.now()) < CUTOFF


def osf_url_ok(u):
    return bool(re.match(r"^https://osf\.io/[A-Za-z0-9]{5,}/?$", str(u or "")))


def _kst(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def _log_gate_once(path, what):
    h = sha256(path)
    if not any(f"GATE sha256 {h} " in ln for ln in prereg_lines()):
        prereg_line("GATE", path, f"wordn {what}: first use")


def deposit_zip_logged(zsha):
    return deposit_zip_path(zsha) is not None


def deposit_zip_path(zsha):
    """The path on the DEPOSIT line of this zip sha256 (a wordn deposit zip), or None."""
    for ln in prereg_lines():
        head = ln.split(" (")[0]
        if f"DEPOSIT sha256 {zsha} " in ln and head.endswith("_wordn.zip"):
            p = Path(head.split(f"DEPOSIT sha256 {zsha} ", 1)[1].strip())
            return p if p.is_absolute() else REPO / p
    return None


def deposit_holds_freeze(zsha, freeze_sha):
    """The deposit zip (sha256 == its DEPOSIT line) contains code_freeze.json with exactly this sha256."""
    import zipfile
    p = deposit_zip_path(zsha)
    if p is None or not p.is_file() or sha256(p) != zsha:
        return False
    with zipfile.ZipFile(p) as z:
        if "code_freeze.json" not in z.namelist():
            return False
        return hashlib.sha256(z.read("code_freeze.json")).hexdigest() == freeze_sha


def gate_uploaded():
    """The record that the OSF registration was uploaded (initiated) with this freeze's deposit.
    Required before any validation image is extracted and before any prediction or tier-2 training."""
    if not GATE_UPLOADED.is_file():
        raise Refuse(f"{rel(GATE_UPLOADED)} missing: nothing on the validation set before the registration is initiated")
    g = json.load(open(GATE_UPLOADED))
    for k in ("osf_url", "uploaded_kst", "written_by", "evidence", "deposit_zip_sha256", "freeze_sha256"):
        if not g.get(k):
            raise Refuse(f"{rel(GATE_UPLOADED)} lacks '{k}'")
    if not osf_url_ok(g["osf_url"]):
        raise Refuse(f"{rel(GATE_UPLOADED)}: bad osf_url {g['osf_url']!r}")
    t = _kst(g["uploaded_kst"])
    if t > datetime.now():
        raise Refuse(f"{rel(GATE_UPLOADED)}: uploaded_kst {g['uploaded_kst']} is in the future")
    if not FREEZE.is_file() or sha256(FREEZE) != g["freeze_sha256"]:
        raise Refuse("gate_osf_uploaded.json names another code freeze than code_freeze.json")
    fz = json.load(open(FREEZE))
    if t < _kst(fz["written"]):
        raise Refuse(f"the upload ({g['uploaded_kst']}) precedes the freeze ({fz['written']})")
    if not deposit_zip_logged(g["deposit_zip_sha256"]):
        raise Refuse("gate_osf_uploaded.json names a deposit zip with no wordn DEPOSIT line in final_prereg.log")
    if not deposit_holds_freeze(g["deposit_zip_sha256"], g["freeze_sha256"]):
        raise Refuse("the deposit zip does not contain this code_freeze.json (or differs from its DEPOSIT line)")
    _log_gate_once(GATE_UPLOADED, "upload gate (registration initiated)")
    return g


def gate_public():
    """The record that the OSF API shows the registration public. Required before any validation label is
    extracted or read; refuses if the registration was first seen public at or after the stop rule."""
    up = gate_uploaded()
    if not GATE_PUBLIC.is_file():
        raise Refuse(f"{rel(GATE_PUBLIC)} missing: no validation label before the registration is public")
    g = json.load(open(GATE_PUBLIC))
    for k in ("osf_url", "public_checked_kst", "api_http_status", "public", "written_by", "evidence"):
        if g.get(k) in (None, ""):
            raise Refuse(f"{rel(GATE_PUBLIC)} lacks '{k}'")
    if g["osf_url"].rstrip("/") != up["osf_url"].rstrip("/"):
        raise Refuse("gate_osf_public.json names another registration than gate_osf_uploaded.json")
    if int(g["api_http_status"]) != 200 or g["public"] is not True:
        raise Refuse("gate_osf_public.json does not record HTTP 200 with public = true")
    t = _kst(g["public_checked_kst"])
    if t < _kst(up["uploaded_kst"]) or t > datetime.now():
        raise Refuse("gate_osf_public.json: public_checked_kst must lie between the upload and now")
    if t >= STOP_RULE:
        raise Refuse(f"the registration was first seen public at {g['public_checked_kst']}, at or after the stop rule "
                     f"({STOP_RULE:%Y-%m-%d %H:%M} KST): the validation labels stay sealed")
    _log_gate_once(GATE_PUBLIC, "public gate (registration public)")
    return g


def look_made():
    return (LOOK / "GT_OPENED").exists() or (LOOK / "results.json").exists()


# ---------------------------------------------------------------------------------------------------- frozen files
def ts_weight_files():
    out = []
    for tid in (291, 292, 293, 294, 295):
        ds = sorted(Path(TS_ROOT, "nnunet", "results").glob(f"Dataset{tid}_*"))
        if len(ds) != 1:
            raise Refuse(f"TotalSegmentator part {tid}: expected one dataset folder under {TS_ROOT}, found {len(ds)}")
        d = ds[0] / "nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres"
        out += [d / "fold_0" / "checkpoint_final.pth", d / "plans.json", d / "dataset.json"]
    return out


def vista_bundle_files():
    sys.path.insert(0, str(REPO / "scripts/vccor/vista"))
    import vista_common as V
    return [V.BUNDLE / k for k in sorted(V.BUNDLE_SHA)]


def freeze_ok():
    if not FREEZE.is_file():
        raise Refuse("code_freeze.json missing: nothing runs on WORD-N before the freeze")
    fz = json.load(open(FREEZE))
    bad = {p: h[:16] for p, h in fz["files"].items() if not (REPO / p).is_file() or sha256_cached(REPO / p) != h}
    if bad:
        raise Refuse(f"files changed after the freeze: {bad}")
    return fz


def frozen_model_hashes(arm):
    fz = json.load(open(FREEZE))
    if arm in fz.get("models", {}):
        return fz["models"][arm]
    for p in sorted(OUT.glob("freeze_addendum_*.json")):
        ad = json.load(open(p))
        if arm in ad.get("models", {}):
            return ad["models"][arm]
    return None


def _check_hashes(h, what):
    bad = [p for p, x in h.items() if not (REPO / p).is_file() or sha256_cached(REPO / p) != x]
    if bad:
        raise Refuse(f"{what}: files changed after the freeze: {bad}")


def check_frozen_model(arm):
    """An arm's model files equal their frozen hashes; every 'ts' arm also needs TotalSegmentator's own weights unchanged,
    every 'vista' arm the VISTA3D bundle; a 'cpu' arm its frozen rules (LABELMAP) and its inputs' models."""
    fz = json.load(open(FREEZE))
    a = arms()[arm]
    if a["kind"] == "ts":
        _check_hashes(fz["models"]["TS_WEIGHTS"], "TS_WEIGHTS")
    if a["kind"] == "vista":
        _check_hashes(fz["models"]["VISTA_BUNDLE"], "VISTA_BUNDLE")
    if a["kind"] == "cpu":
        for i in a["inputs"]:
            check_frozen_model(i)
        if a["role"] == "LABELMAP":
            _check_hashes(fz["models"][f"RULES_{a['draw']}"], f"LABELMAP rules {a['draw']}")
        return
    if arm in ("F0", "V0"):
        return
    h = frozen_model_hashes(arm)
    if h is None:
        raise Refuse(f"{arm}: its model is hashed in neither the freeze nor a freeze addendum")
    _check_hashes(h, arm)


def frozen_checkpoint_hashes(arm):
    """The frozen sha256 of the arm's weight files (what DONE.json's model_sha256 must list)."""
    a = arms()[arm]
    if arm == "V0":
        return []
    h = frozen_model_hashes(arm) or {}
    key = "weights_trainable.pt" if a["kind"] == "vista" else "checkpoint_final.pth"
    return sorted(v for p, v in h.items() if p.endswith(key))


# ---------------------------------------------------------------------------------------------------- GPU
CUTOFF_CODE = 4                          # Refuse code: the cutoff passed while waiting (the job is NOT RUN)
INFRA_POLL_S = 300                       # GPU-infrastructure outages are polled every 300 s (NB2)


class InfraDown(Exception):
    """nvidia-smi missing / failing / timing out / printing an error, GPU 1 or its UUID not listed: not available YET."""


def nvsmi_rows():
    """[(uuid, bus, used MiB)] from nvidia-smi; InfraDown on any failure (never a Refuse)."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=uuid,pci.bus_id,memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as e:
        raise InfraDown(f"nvidia-smi did not run: {type(e).__name__}: {e}")
    if r.returncode != 0:
        raise InfraDown(f"nvidia-smi returned {r.returncode}: {(r.stdout + r.stderr).strip()[:200]}")
    rows = []
    try:
        for line in r.stdout.strip().splitlines():
            u, bus, used = [x.strip() for x in line.split(",")]
            rows.append((u, bus, int(used)))
    except ValueError:
        raise InfraDown(f"nvidia-smi printed an unexpected line: {r.stdout.strip()[:200]}")
    return rows


def _gpu1_row(rows):
    for u, bus, used in rows:
        if bus.upper().endswith(GPU_BUS.upper()[-12:]):
            return u, bus, used
    raise InfraDown(f"GPU {GPU_BUS} not listed by nvidia-smi")


def wait_infra(fn, what, log=print, cutoff_aware=True, poll_s=None, wait_s=GPU_WAIT_S):
    """Call fn() until it does not raise InfraDown: a cutoff-aware wait (code 4 once the cutoff passes; NB2). Without
    cutoff_aware (the pre-registration smoke test) it gives up after wait_s with code 3."""
    t0, said = time.time(), False
    while True:
        try:
            return fn()
        except InfraDown as e:
            if cutoff_aware and not before_cutoff():
                raise Refuse(f"the cutoff passed while waiting for {what} ({e})", code=CUTOFF_CODE)
            if not cutoff_aware and time.time() - t0 > wait_s:
                raise Refuse(f"{what} not available for {wait_s / 3600:.0f} h ({e})", code=3)
            if not said:
                log(f"{what} not available ({e}); waiting (polled every {poll_s or INFRA_POLL_S} s)")
                said = True
            time.sleep(poll_s or INFRA_POLL_S)


def gpu1_used_mib():
    """memory.used of GPU 1 (bus 05:00.0); InfraDown if nvidia-smi or GPU 1 is not available."""
    return _gpu1_row(nvsmi_rows())[2]


def wait_gpu1_idle(wait_s=GPU_WAIT_S, poll_s=GPU_POLL_S, log=print, used_fn=None, cutoff_aware=True):
    """Wait until GPU 1 uses < 3000 MiB. An nvidia-smi outage is waited out like a busy GPU (NB2)."""
    used_fn = used_fn or gpu1_used_mib
    t0 = time.time()
    while True:
        try:
            u = used_fn()
        except InfraDown as e:
            u, why = None, str(e)
        if u is not None and u < GPU_IDLE_MIB:
            return u
        if cutoff_aware and not before_cutoff():
            raise Refuse("the cutoff passed while waiting for GPU 1 to be idle", code=CUTOFF_CODE)
        if not cutoff_aware and time.time() - t0 > wait_s:
            raise Refuse(f"GPU 1 not idle for {wait_s / 3600:.0f} h (memory.used {u} MiB)", code=3)
        log((f"GPU 1 busy ({u} MiB)" if u is not None else f"GPU 1 not available ({why})") + "; waiting")
        time.sleep(poll_s)


def gpu_identity_gate(log=print, cutoff_aware=True, poll_s=None):
    """A real GPU entry point runs only on GPU 1. CUDA_VISIBLE_DEVICES must be GPU 1's UUID: otherwise a configuration
    error (Refuse code 2, a failure). nvidia-smi must list that UUID on bus 00000000:05:00.0: an nvidia-smi outage or a
    missing UUID is waited out (cutoff-aware, NB2); the UUID on another bus is a configuration error."""
    v = os.environ.get("CUDA_VISIBLE_DEVICES")
    if v != GPU1_UUID:
        raise Refuse(f"CUDA_VISIBLE_DEVICES is {v!r}, not GPU 1's UUID {GPU1_UUID}: real GPU jobs run on GPU 1 only")

    def find():
        for u, bus, _used in nvsmi_rows():
            if u == GPU1_UUID:
                return bus
        raise InfraDown(f"GPU 1's UUID {GPU1_UUID} not listed by nvidia-smi")
    bus = wait_infra(find, "GPU 1 (nvidia-smi)", log=log, cutoff_aware=cutoff_aware, poll_s=poll_s)
    if not bus.upper().endswith(GPU_BUS.upper()[-12:]):
        raise Refuse(f"GPU 1's UUID is on bus {bus}, not {GPU_BUS}")
    return True


def acquire_gpu_lock(path=None, wait_s=GPU_WAIT_S, poll_s=30, log=print, cutoff_aware=True, missing_poll_s=None):
    """Exclusive flock on the shared GPU-1 lock file (opened read-only; never created or written). A missing lock file is
    waited for (it belongs to the Handpose project's tooling and is never created here; NB2); a held lock is waited for.
    Both waits are cutoff-aware (code 4)."""
    path = Path(path or GPU_LOCK)

    def opened():
        if not path.is_file():
            raise InfraDown(f"the GPU-1 lock file {path} is missing (it belongs to the Handpose project's tooling; "
                            f"WORD-N never creates it)")
        try:
            return os.open(str(path), os.O_RDONLY)
        except OSError as e:
            raise InfraDown(f"the GPU-1 lock file {path} cannot be opened: {e}")
    fd = wait_infra(opened, "the GPU-1 lock file", log=log, cutoff_aware=cutoff_aware, poll_s=missing_poll_s, wait_s=wait_s)
    t0, said = time.time(), False
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except BlockingIOError:
            if cutoff_aware and not before_cutoff():
                os.close(fd)
                raise Refuse("the cutoff passed while waiting for the GPU-1 lock", code=CUTOFF_CODE)
            if not cutoff_aware and time.time() - t0 > wait_s:
                os.close(fd)
                raise Refuse(f"GPU-1 lock held elsewhere for {wait_s / 3600:.0f} h", code=3)
            if not said:
                log("GPU-1 lock held elsewhere; waiting")
                said = True
            time.sleep(poll_s)


def release_gpu_lock(fd):
    if fd is None:
        return
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def gpu_session(log=print, cutoff_aware=True, poll_s=None, idle_poll_s=None, lock_poll_s=30):
    """For a real GPU entry point: the GPU identity gate; then wait for GPU 1 to be idle WITHOUT the lock, take the shared
    flock, re-check idleness, and release and repeat if it is busy again (N-E1). Every wait is cutoff-aware. Returns the
    lock fd (release with release_gpu_lock)."""
    gpu_identity_gate(log=log, cutoff_aware=cutoff_aware, poll_s=poll_s)
    ip = idle_poll_s or GPU_POLL_S
    while True:
        wait_gpu1_idle(log=log, cutoff_aware=cutoff_aware, poll_s=ip)
        fd = acquire_gpu_lock(log=log, cutoff_aware=cutoff_aware, poll_s=lock_poll_s, missing_poll_s=poll_s)
        try:
            u = gpu1_used_mib()
        except InfraDown:
            u = None
        if u is not None and u < GPU_IDLE_MIB:
            return fd
        release_gpu_lock(fd)
        log(f"GPU 1 busy again after the lock was taken ({u} MiB); lock released, waiting")
        if cutoff_aware and not before_cutoff():
            raise Refuse("the cutoff passed while waiting for GPU 1", code=CUTOFF_CODE)
        time.sleep(ip)


# ---------------------------------------------------------------------------------------------------- models
def _check_field(man, k, v):
    if isinstance(v, float):
        return man.get(k) is not None and abs(float(man.get(k)) - v) < 1e-12
    if isinstance(v, list):
        return list(man.get(k) or []) == list(v)
    return man.get(k) == v


def resolve_model(arm, strict=True):
    """-> ('ts', {'redirect', 'shas'}) / ('nnunet', {...}) / ('vista', {...}). Asserts every checkpoint against its export
    and the arm's registered checks (strict: full, non-smoke runs only)."""
    a = arms()[arm]
    if a["kind"] == "ts":
        red, shas = {}, {}
        for tid, md in sorted(a["parts"].items()):
            md = Path(md)
            man = json.load(open(md / "export.json"))
            h = sha256_cached(md / "fold_0" / "checkpoint_final.pth")
            if h != man["checkpoint_sha256"]:
                raise Refuse(f"{rel(md)}: checkpoint changed after export")
            if strict:
                if man.get("num_epochs") != TS_EPOCHS or man.get("smoke"):
                    raise Refuse(f"{rel(md)} is not a full training run")
                for k, v in (a.get("check") or {}).items():
                    if not _check_field(man, k, v):
                        raise Refuse(f"{rel(md)}: export.json {k} = {man.get(k)!r}, registered {v!r}")
            red[int(tid)], shas[int(tid)] = str(md), h
        return "ts", {"redirect": red, "shas": shas}
    if a["kind"] == "nnunet":
        md = Path(a["dir"])
        man = json.load(open(md / "export.json"))
        h = sha256_cached(md / f"fold_{a['fold']}" / "checkpoint_final.pth")
        if h != man["checkpoint_sha256"]:
            raise Refuse(f"{rel(md)}: checkpoint changed after export")
        if strict:
            if man.get("smoke") or int(man.get("num_epochs", -1)) != int(a["epochs"]):
                raise Refuse(f"{rel(md)} is not the registered full run")
            for k, v in (a.get("check") or {}).items():
                if not _check_field(man, k, v):
                    raise Refuse(f"{rel(md)}: export.json {k} = {man.get(k)!r}, registered {v!r}")
        return "nnunet", {"dir": str(md), "fold": a["fold"], "n_classes": int(a["n_classes"]), "shas": {"nnunet": h}}
    if a["kind"] == "vista":
        if a["weights"] is None:
            return "vista", {"weights": None, "shas": {"vista": "bundle"}}
        w = Path(a["weights"])
        d = json.load(open(w.parent / "DONE.json"))
        h = sha256_cached(w)
        if h != d["weights_sha256"]:
            raise Refuse(f"{rel(w)}: weights changed after training")
        if strict:
            s = VF_RUNS[a["run"]]
            if d.get("smoke") or d.get("n_iter") != s["n_iter"] or abs(float(d.get("lr", -1)) - s["lr"]) > 1e-15 \
                    or d.get("draw") != s["draw"]:
                raise Refuse(f"{rel(w)} is not the registered {a['run']} run")
        return "vista", {"weights": str(w), "shas": {"vista": h}}
    raise Refuse(f"{arm}: kind {a['kind']} has no model")


def model_files(arm):
    a = arms()[arm]
    if a["kind"] == "ts":
        out = []
        for tid, md in sorted(a["parts"].items()):
            md = Path(md)
            out += [md / "fold_0" / "checkpoint_final.pth", md / "export.json", md / "plans.json", md / "dataset.json"]
        return out
    if a["kind"] == "nnunet":
        md = Path(a["dir"])
        return [md / f"fold_{a['fold']}" / "checkpoint_final.pth", md / "export.json", md / "plans.json",
                md / "dataset.json"]
    if a["kind"] == "vista" and a["weights"] is not None:
        w = Path(a["weights"])
        return [w, w.parent / "export.json", w.parent / "DONE.json"]
    return []


def pred_dir(arm, root=None):
    return Path(root or PRED) / arm


def done_file(arm, root=None):
    return pred_dir(arm, root) / "DONE.json"


def label_path(cid):
    return Path(LBL_DIR) / f"word_{cid}.nii.gz"


def run_state(arm):
    p = RUN / f"{arm}.json"
    return json.load(open(p)) if p.is_file() else {}


def declared_not_run(arm):
    """The runner's binding NOT RUN declaration of an arm (or None)."""
    return run_state(arm).get("not_run")


def smoke_passed():
    p = VSMOKE / "smoke.json"
    return p.is_file() and json.load(open(p)).get("decision") == "VSMOKE_PASS"


def arm_allowed(arm):
    """Real jobs refuse an arm the runner declared NOT RUN, and any tier-2 arm without VSMOKE_PASS."""
    r = declared_not_run(arm)
    if r:
        raise Refuse(f"{arm} was declared NOT RUN by the runner ({r}): binding, no job runs for it")
    if arm in TIER2 and not smoke_passed():
        raise Refuse(f"{arm}: the VISTA3D smoke test has not passed: tier 2 is NOT RUN")


def included_ids():
    """The scans of the look: VAL_IDS minus the registered duplicate exclusions (fixed from the images after the upload
    gate, before any prediction)."""
    if not EXCLUSIONS.is_file():
        raise Refuse(f"{rel(EXCLUSIONS)} missing: the intake has not applied the duplicate rule")
    ex = json.load(open(EXCLUSIONS))
    if ex.get("decision") != "INTAKE_PASS":
        raise Refuse("the intake did not pass")
    drop = set(ex.get("excluded", {}))
    return [c for c in VAL_IDS if c not in drop]
