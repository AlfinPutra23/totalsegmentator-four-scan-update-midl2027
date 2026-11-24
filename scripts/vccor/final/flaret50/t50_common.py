"""vccor/final FLARE-T50: shared constants, arm registry, gates and helpers.

FLARE-T50 is GO-1 of the design of 2026-11-09 (AC.md, sections 2.0 and
GO-1): one registered look at the 50 public validation cases of the FLARE 2024-25 Task 2 release (inferred to be FLARE22's
tuning set), scored with frozen FLARE models, plus tier 2 (FULL K=1 retrained on the four fair-fight draws). Fixed
deviations from AC.md: images on / (not the external drive, which is under a write-nothing rule); GO-2
(WORD-N) is not part of this code.

Order (binding): images-only steps (download, hashing, K1b overlap, K1c timing on the two registered images) may run before
the OSF registration; no tier-1 prediction before gate_osf_uploaded.json (it names the deposit
zip and the code freeze, both checked here); no FLARE-T50 label byte before gate_osf_public.json (written
after the OSF API shows the registration public). Arms not DONE by the cutoff (Fri 2026-11-13 18:00 KST) are NOT RUN. If the
registration is not public by Tue 2026-11-17 12:00 KST the set stays sealed (stop rule).

use_root(ROOT) redirects every path of the study under ROOT (the look rehearsal and the unit tests only).
No import-time side effects beyond sys.path (no file is read or written at import).
Version of 2026-11-09; revised after the independent code review of 2026-11-09 (M1b).
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
for _p in (str(FINAL), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import final_common as F  # noqa: E402

REPO = F.REPO
VC = F.VC
VENV = F.VENV
TAG = "flaret50"

HF_REPO = "FLARE-MedFM/FLARE-Task2-LaptopSeg"
HF_REVISION = "d918fba87eec740e53f78b980eb9efefd8dfb68d"
HF_IMG_PREFIX = "validation/Validation-Public-Images/"
HF_LBL_PREFIX = "validation/Validation-Public-Labels/"
IDS = [f"FLARETs_{i:04d}" for i in range(1, 51)]
K1C_IDS = ["FLARETs_0030", "FLARETs_0050"]

CUTOFF = datetime(2026, 11, 13, 18, 0)
STOP_RULE = datetime(2026, 11, 17, 12, 0)
FLOOR_GIB = 35.0
GPU_LOCK = Path("/home/user/Documents/Handpose_project/runs/method_pilot3/gpu1.lock")
GPU_BUS = "00000000:05:00.0"
GPU_IDLE_MIB = 3000
GPU_WAIT_S = 12 * 3600
GPU_POLL_S = 60
GPU_ENV = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "1",
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
CPU_ENV = {"CUDA_VISIBLE_DEVICES": ""}
HF_ENV = {"HF_HUB_DISABLE_TELEMETRY": "1", "HF_XET_CHUNK_CACHE_SIZE_BYTES": "0"}
TS_EPOCHS = 50
TS_VERSION = "2.13.0"
CAPTURE = (292, 293, 294, 295)
VERSION_PACKAGES = ["numpy", "scipy", "nibabel", "torch", "nnunetv2", "TotalSegmentator", "huggingface_hub"]

N_ORG = 13
ORGANS = list(range(1, N_ORG + 1))                     # FLARE ids: final_common.FLARE13
SOLID = {"liver": 1, "spleen": 3, "kidney_right": 2, "kidney_left": 13, "pancreas": 4}
LABEL_MAX = 13

# ---------------------------------------------------------------------------------------------------- paths (use_root)
REHEARSAL = False
LABEL_SOURCE = None                    # rehearsal only: a local folder of stand-in labels instead of the Hub


def _bind(out, data, prereg_log, events_log, ff_models, ts_root):
    g = globals()
    g.update(OUT=out, DATA=data, HF_LOCAL=data / "hf", PRED=out / "pred", LOOK=out / "look", TIER2=out / "tier2",
             K1B=out / "k1b", K1C=out / "k1c", RUN=out / "run", MANIFEST=out / "hf_manifest.json",
             IMAGES_RECORD=out / "images_record.json", PLAN=out / "plan_flaret50.json", EXCLUSIONS=out / "exclusions.json",
             GATE_UPLOADED=out / "gate_osf_uploaded.json", GATE_PUBLIC=out / "gate_osf_public.json",
             FREEZE=out / "code_freeze.json", REHEARSAL_RECORD=out / "rehearsal.json", PREREG_LOG=prereg_log,
             EVENTS_LOG=events_log, FF_MODELS=ff_models, TS_ROOT=ts_root)
    g["IMG_DIR"] = g["HF_LOCAL"] / "validation" / "Validation-Public-Images"     # images only, before registration
    g["LBL_DIR"] = g["HF_LOCAL"] / "validation" / "Validation-Public-Labels"     # created only inside the look


_bind(VC / "final" / "I_flaret50", REPO / "data_flaret50", F.PREREG_LOG, F.LOGD / "final_flaret50.log",
      VC / "moonshot/ff/models/FULL", Path.home() / ".totalsegmentator")


def use_root(root, ts_root=None):
    """Redirect every study path under root (look rehearsal / unit tests). Sets REHEARSAL = True."""
    global REHEARSAL
    root = Path(root)
    _bind(root / "I_flaret50", root / "data_flaret50", root / "logs" / "final_prereg.log", root / "logs" / "events.log",
          root / "ff_models", Path(ts_root) if ts_root else root / "totalsegmentator")
    REHEARSAL = True
    (root / "logs").mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------------------------------- arms
TSFT, AUD, E2M, SSFT = VC / "tsft_full/models", VC / "audit_ft/models", VC / "e2/models", VC / "ssft/models"
XM, PEFT, SNM, CW = VC / "xdata/models", VC / "peft/models", VC / "scratch_nm/models", VC / "final/C_word/models"
FF_DIR = VC / "moonshot/ff/models/FULL"
S1337, S2025, S3407 = ["0014", "0001", "0006", "0038"], ["0007", "0026", "0032", "0029"], ["0018", "0030", "0033", "0004"]

# kind 'ts': TotalSegmentator 2.13.0's own 'total' pipeline (frozen tsft_full/ts_pipeline.py) with the listed parts
#   redirected to the export; 'check' = export.json fields that must match (plus checkpoint sha256 == export.json,
#   num_epochs == 50, smoke False; 'labelled_prefix' = the first labelled ids). kind 'nnunet': one nnU-Net network on
#   FLARE's 13 classes (nnUNetPredictor, tile 0.5, gaussian, mirroring OFF), fold 'all'.
ARMS = {
    "F0": dict(kind="ts", parts={}, tier=1, role="zero-shot",
               label="TotalSegmentator 2.13.0 'total', full resolution, its own weights for all five parts (captures 292-295)"),
    "FULL_s1337": dict(kind="ts", tier=1, role="K4", draw="s1337",
                       parts={291: TSFT / "F1b/part291", 293: TSFT / "F1b/part293"},
                       check={"variant": "F1b", "trainable": "all", "initial_lr": 1e-4, "labelled_ids": S1337},
                       label="F1b, K = 4, draw s1337 (0014 0001 0006 0038), tsft_full"),
    "FULL_s2025": dict(kind="ts", tier=1, role="K4", draw="s2025",
                       parts={291: AUD / "s2025_FULL/part291", 293: AUD / "s2025_FULL/part293"},
                       check={"run": "s2025_FULL", "trainable": "all", "initial_lr": 1e-4, "labelled_ids": S2025},
                       label="F1b recipe, K = 4, draw s2025, audit_ft"),
    "FULL_s3407": dict(kind="ts", tier=1, role="K4", draw="s3407",
                       parts={291: AUD / "s3407_FULL/part291", 293: AUD / "s3407_FULL/part293"},
                       check={"run": "s3407_FULL", "trainable": "all", "initial_lr": 1e-4, "labelled_ids": S3407},
                       label="F1b recipe, K = 4, draw s3407, audit_ft"),
    "K16_s1337": dict(kind="ts", tier=1, role="K16", draw="s1337",
                      parts={291: E2M / "flare_K16_F1b/part291", 293: E2M / "flare_K16_F1b/part293"},
                      check={"run": "flare_K16_F1b", "initial_lr": 1e-4, "labelled_prefix": S1337},
                      label="F1b recipe, K = 16, draw s1337, e2 budget"),
    "K16_s2025": dict(kind="ts", tier=1, role="K16", draw="s2025",
                      parts={291: E2M / "flare_K16_F1b_s2025/part291", 293: E2M / "flare_K16_F1b_s2025/part293"},
                      check={"run": "flare_K16_F1b_s2025", "initial_lr": 1e-4, "labelled_prefix": S2025},
                      label="F1b recipe, K = 16, draw s2025, e2 budget"),
    "K16_s3407": dict(kind="ts", tier=1, role="K16", draw="s3407",
                      parts={291: E2M / "flare_K16_F1b_s3407/part291", 293: E2M / "flare_K16_F1b_s3407/part293"},
                      check={"run": "flare_K16_F1b_s3407", "initial_lr": 1e-4, "labelled_prefix": S3407},
                      label="F1b recipe, K = 16, draw s3407, e2 budget"),
    "SSFT_B": dict(kind="ts", tier=1, role="pseudo", parts={291: SSFT / "B/part291", 293: SSFT / "B/part293"},
                   check={"trainable": "all", "initial_lr": 1e-4, "arm": "B", "pool_source": "TS"},
                   label="ssft B: F1b recipe on the 4 labelled s1337 scans + 36 scans with TotalSegmentator pseudo-labels"),
    "SSFT_D": dict(kind="ts", tier=1, role="pseudo", parts={291: SSFT / "D/part291", 293: SSFT / "D/part293"},
                   check={"trainable": "all", "initial_lr": 1e-4, "arm": "D", "pool_source": "F1b"},
                   label="ssft D: F1b recipe on the 4 labelled s1337 scans + the same 36 scans with self-labels"),
    "STOCK": dict(kind="nnunet", tier=1, role="stock", dir=PEFT / "STOCK/stock", fold="all", n_classes=13, epochs=250,
                  label="peft STOCK: nnU-Net's pretrained fine-tuning recipe, 13 classes, 250 epochs, K = 4 s1337"),
    "SCRATCH_NM1000": dict(kind="nnunet", tier=1, role="scratch", dir=SNM / "SCRATCH_NM1000/scratch", fold="all",
                           n_classes=13, epochs=1000,
                           label="nnU-Net from scratch, no mirroring, 1000 epochs, K = 4 s1337"),
    "BORROW_amos": dict(kind="ts", tier=1, role="borrowed", cohort="amos",
                        parts={291: XM / "amos_s1337_K4_FULL_F1b/part291", 293: XM / "amos_s1337_K4_FULL_F1b/part293"},
                        check={"run": "amos_s1337_K4_FULL_F1b", "trainable": "all", "initial_lr": 1e-4},
                        label="AMOS-tuned FULL_K4 (xdata E1, F1b recipe)"),
    "BORROW_mact": dict(kind="ts", tier=1, role="borrowed", cohort="mact",
                        parts={291: XM / "mact_s1337_K4_FULL_F1b/part291"},
                        check={"run": "mact_s1337_K4_FULL_F1b", "trainable": "all", "initial_lr": 1e-4},
                        label="MACT-tuned FULL_K4 (xdata E1, F1b recipe, part 291)"),
    "BORROW_word": dict(kind="ts", tier=1, role="borrowed", cohort="word", parts={291: CW / "F1b_d1/part291"},
                        check={"run": "F1b_d1", "variant": "F1b", "trainable": "all", "initial_lr": 1e-4},
                        label="WORD-tuned F1b draw d1 (plan C word A1, part 291)"),
}
TIER1_ARMS = list(ARMS)
TIER2_DRAWS = ["d1_0014", "d1_0001", "d1_0006", "d1_0038"]
for _d in TIER2_DRAWS:
    ARMS[f"K1_{_d[3:]}"] = dict(kind="ts", tier=2, role="K1", draw=_d,
                                parts={291: FF_DIR / _d / "part291", 293: FF_DIR / _d / "part293"},
                                check={"trainable": "all", "initial_lr": 1e-4},
                                label=f"FULL K = 1 on fair-fight draw {_d} (frozen fair-fight code via moonshot/ff_launch.py)")
TIER2_ARMS = [a for a in ARMS if ARMS[a]["tier"] == 2]
K4 = {"s1337": "FULL_s1337", "s2025": "FULL_s2025", "s3407": "FULL_s3407"}
K16 = {"s1337": "K16_s1337", "s2025": "K16_s2025", "s3407": "K16_s3407"}
BORROWED = {"amos": "BORROW_amos", "mact": "BORROW_mact", "word": "BORROW_word"}
RUN_ORDER = TIER1_ARMS + TIER2_ARMS                     # F0 first (it builds the frozen-part cache)
LUT_KIND = {a: ("ts" if ARMS[a]["kind"] == "ts" else "id") for a in ARMS}

# ---------------------------------------------------------------------------------------------------- errors / logs


class Refuse(Exception):
    """A gate refused. code 2 = gate, 3 = resource (disk, GPU busy, lock)."""

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
    """sha256 cached per process by (path, size, mtime_ns): large frozen files are hashed once per process."""
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
    with open(p, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)
    return h


def prereg_note(kind, text, log_path=None):
    line = f"[{now()}] {kind} {TAG} {text}"
    with open(Path(log_path or PREREG_LOG), "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def prereg_lines():
    p = Path(PREREG_LOG)
    return open(p).read().splitlines() if p.is_file() else []


def write_json(path, obj, mode=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(F.clean_nan(obj), f, indent=1, default=F._default)
    os.replace(tmp, path)
    if mode is not None:
        os.chmod(path, mode)


def code_hashes(names):
    """sha256 of the named FLARE-T50 code files (recorded in every pre-registration record that the code produced)."""
    return {n: sha256(HERE / n) for n in names}


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


# ---------------------------------------------------------------------------------------------------- LFS pointers
def lfs_pointer_text(sha256_hex, size):
    return f"version https://git-lfs.github.com/spec/v1\noid sha256:{sha256_hex}\nsize {int(size)}\n"


def git_blob_oid(data: bytes):
    """git's object id of a blob (sha1 of 'blob <len>\\0' + data)."""
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def pointer_oid(sha256_hex, size):
    """The git oid of the LFS pointer file of a content with this sha256 and size: what the Hub's public tree listing
    shows as 'oid' for an LFS file. It pins the content without access to the gated files."""
    return git_blob_oid(lfs_pointer_text(sha256_hex, size).encode())


def load_manifest():
    if not MANIFEST.is_file():
        raise Refuse(f"{rel(MANIFEST)} missing: run t50_intake.py --metadata first")
    return json.load(open(MANIFEST))


# ---------------------------------------------------------------------------------------------------- gates
def kst_gate():
    tz = time.strftime("%Z")
    if tz != "KST":
        raise Refuse(f"local time zone is {tz}, not KST")


def disk_free_gib(path=None):
    st = os.statvfs(str(path or REPO))
    return st.f_bavail * st.f_frsize / 1024 ** 3


def disk_gate(need_gib, what):
    free = disk_free_gib()
    if free - float(need_gib) < FLOOR_GIB:
        raise Refuse(f"{what}: / has {free:.2f} GiB free; with {need_gib:.2f} GiB more it would fall below the "
                     f"{FLOOR_GIB:.0f} GiB floor", code=3)
    return free


def before_cutoff(t=None):
    return (t or datetime.now()) < CUTOFF


def osf_url_ok(u):
    return bool(re.match(r"^https://osf\.io/[A-Za-z0-9]{5,}/?$", str(u or "")))


def _kst(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def _log_gate_once(path, what):
    h = sha256(path)
    if not any(f"GATE sha256 {h} " in ln for ln in prereg_lines()):
        prereg_line("GATE", path, f"flaret50 {what}: first use")


def deposit_zip_logged(zsha):
    """True iff final_prereg.log has a 'DEPOSIT sha256 <zsha> ...flaret50.zip' line."""
    return any(f"DEPOSIT sha256 {zsha} " in ln and ln.split(" (")[0].endswith("_flaret50.zip") for ln in prereg_lines())


def gate_uploaded():
    """The record that the OSF registration was uploaded (initiated) with this freeze's deposit.
    Required before any tier-1 prediction on FLARE-T50."""
    if not GATE_UPLOADED.is_file():
        raise Refuse(f"{rel(GATE_UPLOADED)} missing: no tier-1 prediction before the registration is initiated")
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
        raise Refuse("gate_osf_uploaded.json names a deposit zip with no flaret50 DEPOSIT line in final_prereg.log")
    _log_gate_once(GATE_UPLOADED, "upload gate (registration initiated)")
    return g


def gate_public():
    """The record that the OSF API shows the registration public. Required before any FLARE-T50 label byte
    is downloaded or read; refuses if the registration was first seen public at or after the stop rule."""
    up = gate_uploaded()
    if not GATE_PUBLIC.is_file():
        raise Refuse(f"{rel(GATE_PUBLIC)} missing: no FLARE-T50 label before the registration is public")
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
                     f"({STOP_RULE:%Y-%m-%d %H:%M} KST): the set stays sealed")
    _log_gate_once(GATE_PUBLIC, "public gate (registration public)")
    return g


def look_made():
    return (LOOK / "GT_OPENED").exists() or (LOOK / "results.json").exists()


# ---------------------------------------------------------------------------------------------------- frozen files
def ts_weight_files():
    """TotalSegmentator 2.13.0's own weights of the five 'total' parts (checkpoint, plans, dataset json)."""
    out = []
    for tid in (291, 292, 293, 294, 295):
        ds = sorted(Path(TS_ROOT, "nnunet", "results").glob(f"Dataset{tid}_*"))
        if len(ds) != 1:
            raise Refuse(f"TotalSegmentator part {tid}: expected one dataset folder under {TS_ROOT}, found {len(ds)}")
        d = ds[0] / "nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres"
        out += [d / "fold_0" / "checkpoint_final.pth", d / "plans.json", d / "dataset.json"]
    return out


def freeze_ok():
    """Every non-model file of the freeze (code, frozen imports, records) still has its hash."""
    if not FREEZE.is_file():
        raise Refuse("code_freeze.json missing: nothing runs on FLARE-T50 before the freeze")
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
    """An arm's model files equal their hashes in the freeze or a freeze addendum; every 'ts' arm (F0 included) also needs
    TotalSegmentator's own weights unchanged (models['TS_WEIGHTS'])."""
    fz = json.load(open(FREEZE))
    if ARMS[arm]["kind"] == "ts":
        tsw = fz.get("models", {}).get("TS_WEIGHTS")
        if not tsw:
            raise Refuse("TotalSegmentator's weights are not hashed in the freeze")
        _check_hashes(tsw, "TS_WEIGHTS")
    if arm == "F0":
        return
    h = frozen_model_hashes(arm)
    if h is None:
        raise Refuse(f"{arm}: its model is hashed in neither the freeze nor a freeze addendum")
    _check_hashes(h, arm)


def frozen_checkpoint_hashes(arm):
    """The frozen sha256 of the arm's checkpoint files (what DONE.json's model_sha256 must list)."""
    h = frozen_model_hashes(arm) or {}
    return sorted(v for p, v in h.items() if p.endswith("checkpoint_final.pth"))


# ---------------------------------------------------------------------------------------------------- GPU
def gpu1_used_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=pci.bus_id,memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=60).stdout
    for line in out.strip().splitlines():
        bus, used = [x.strip() for x in line.split(",")]
        if bus.upper().endswith(GPU_BUS.upper()[-12:]):
            return int(used)
    raise Refuse(f"GPU {GPU_BUS} not found by nvidia-smi", code=3)


def wait_gpu1_idle(wait_s=GPU_WAIT_S, poll_s=GPU_POLL_S, log=print, used_fn=gpu1_used_mib):
    t0 = time.time()
    while True:
        u = used_fn()
        if u < GPU_IDLE_MIB:
            return u
        if time.time() - t0 > wait_s:
            raise Refuse(f"GPU 1 not idle for {wait_s / 3600:.0f} h (memory.used {u} MiB)", code=3)
        log(f"GPU 1 busy ({u} MiB); waiting")
        time.sleep(poll_s)


def acquire_gpu_lock(path=GPU_LOCK, wait_s=GPU_WAIT_S, poll_s=30, log=print):
    """Exclusive flock on the shared GPU-1 lock file (opened read-only; never created or written)."""
    if not Path(path).is_file():
        raise Refuse(f"GPU-1 lock file {path} missing", code=3)
    fd = os.open(str(path), os.O_RDONLY)
    t0 = time.time()
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except BlockingIOError:
            if time.time() - t0 > wait_s:
                os.close(fd)
                raise Refuse(f"GPU-1 lock held elsewhere for {wait_s / 3600:.0f} h", code=3)
            log("GPU-1 lock held elsewhere; waiting")
            time.sleep(poll_s)


def release_gpu_lock(fd):
    if fd is None:
        return
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------------------------------- models
def checkpoint_path(arm, tid=None):
    a = ARMS[arm]
    if a["kind"] == "ts":
        return Path(a["parts"][tid]) / "fold_0" / "checkpoint_final.pth"
    return Path(a["dir"]) / f"fold_{a['fold']}" / "checkpoint_final.pth"


def _check_field(man, k, v):
    if k == "labelled_prefix":
        return list(man.get("labelled_ids") or [])[:len(v)] == list(v)
    if isinstance(v, float):
        return man.get(k) is not None and abs(float(man.get(k)) - v) < 1e-12
    return man.get(k) == v


def resolve_model(arm, strict=True):
    """-> ('ts', {'redirect': {tid: dir}, 'shas': {tid: sha256}}) or ('nnunet', {...}). Asserts every checkpoint against
    its export.json and the arm's registered checks (strict: full, non-smoke runs only)."""
    a = ARMS[arm]
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
    md = Path(a["dir"])
    man = json.load(open(md / "export.json"))
    h = sha256_cached(md / f"fold_{a['fold']}" / "checkpoint_final.pth")
    if h != man["checkpoint_sha256"]:
        raise Refuse(f"{rel(md)}: checkpoint changed after export")
    if strict and (man.get("smoke") or int(man.get("num_epochs", -1)) != int(a["epochs"])):
        raise Refuse(f"{rel(md)} is not the registered full run")
    return "nnunet", {"dir": str(md), "fold": a["fold"], "n_classes": int(a["n_classes"]), "shas": {"nnunet": h}}


def model_files(arm):
    """Every file of an arm's model that is hashed in the freeze (checkpoint + export / plans / dataset json)."""
    a = ARMS[arm]
    if a["kind"] == "ts":
        out = []
        for tid, md in sorted(a["parts"].items()):
            md = Path(md)
            out += [md / "fold_0" / "checkpoint_final.pth", md / "export.json", md / "plans.json", md / "dataset.json"]
        return out
    md = Path(a["dir"])
    return [md / f"fold_{a['fold']}" / "checkpoint_final.pth", md / "export.json", md / "plans.json", md / "dataset.json"]


def pred_dir(arm, root=None):
    return Path(root or PRED) / arm


def done_file(arm, root=None):
    return pred_dir(arm, root) / "DONE.json"


def image_path(cid, root=None):
    return Path(root or IMG_DIR) / f"{cid}_0000.nii.gz"


def label_path(cid, root=None):
    return Path(root or LBL_DIR) / f"{cid}.nii.gz"


def included_ids():
    """The scans of the look: IDS minus the exclusions fixed before registration (K1b near-duplicates, unloadable files)."""
    if not EXCLUSIONS.is_file():
        raise Refuse(f"{rel(EXCLUSIONS)} missing: K1b has not fixed the exclusions")
    ex = json.load(open(EXCLUSIONS))
    drop = set(ex.get("excluded", {}))
    return [c for c in IDS if c not in drop]
