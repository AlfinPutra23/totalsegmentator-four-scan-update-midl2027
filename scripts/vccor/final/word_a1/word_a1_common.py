"""vccor/final plan C, amendments word A1 (outputs/results/vccor/final/amend_word_A1.json), A1.1 (amend_word_A1_1.json) and
A1.2 (amend_word_A1_2.json): shared constants and gates of the TS-only scorer (word_a1_score.py), the gated runner of the WORD
jobs (word_a1_run.py), the A1.2 STOCK wrapper (word_a1_stock.py), the A1.2 CPU arms (word_a1_cpu_arms.py) and the pre-look
rehearsal (word_a1_rehearse.py).

A1 keeps TotalSegmentator only: arms F0 (zero-shot) and F1b on draws d1 / d2 / d3. A1.2 adds three secondary arms per draw
(P1 and P2 unchanged): STOCK_dX (nnU-Net's stock pretrained fine-tune of TS part 291, the registered STOCK recipe),
MERGE_N_dX (the class-wise merge of F0 and F1b_dX, non-target classes from F0; conventions v1 item6) and LABELMAP_dX (F0
relabelled with class-merge rules fixed from the draw's four training scans). This module holds
  * the pinned identities: A1's, A1.1's and A1.2's sha256, the superseded v1 / v2 / v3 / v4 deposit zips, the v5 deposit
    zip, the cutoff (2026-11-27 00:00 KST);
  * the 22 jobs in A1.2's registered run order: the seven GPU jobs of A1 (the command lines of queue_v8.py's pfinC rows,
    unchanged), then the six CPU arm jobs (MERGE_N d1 / d2 / d3, then LABELMAP d1 / d2 / d3), then STOCK d1 / d2 / d3 (CPU
    prep, GPU train, GPU predict);
  * the scheduler rows A1 neutralises (outputs/logs/q8/decisions.json 'drop'; no queue_v8 row runs an A1.2 job);
  * the OSF gate: C_word/A1_osf_registration.txt must hold the public osf.io URL and the v5 zip's logged sha256 (never the
    v1, v2, v3 or v4 zip), and final_prereg.log must hold a matching OSF-REGISTRATION line written after the v5 DEPOSIT
    line;
  * NOT RUN declarations (final_prereg.log 'NOTRUN word_A1 <arm|EXPERIMENT> ...' lines);
  * integrity checks of the real run (plan C PREREG line, CODE / IMPORT lines, A1 / A1.1 / A1.2 / code-amendment AMENDMENT
    lines);
  * the shared GPU-1 flock (A1.1): before every GPU job the runner takes an exclusive flock on
    /home/user/Documents/Handpose_project/runs/method_pilot3/gpu1.lock (waits up to 12 h) and holds it until the job's main
    process exits. The job inherits the locked descriptor, but the runner then releases the lock (LOCK_UN on the shared
    open file description), so a grandchild that outlives the job does not keep it; the next GPU launch's idle check (GPU 1
    memory.used < 3000 MiB) covers such a survivor. The file is opened read-only and is never created, written or truncated
    here; CPU jobs take no lock;
  * a Ctx object that points every path at the real project, at a scratch folder under /tmp (unit tests), or (rehearsal)
    reads the real prediction maps and writes only under /tmp.

Log line formats (outputs/logs/vccor/final_prereg.log, whitespace-separated, parsed by position):
  [date time] OSF-REGISTRATION <url> sha256 <sha256 of the gate file> outputs/results/vccor/final/C_word/A1_osf_registration.txt
  [date time] NOTRUN word_A1 <arm|EXPERIMENT> <reason ...>          (arm: F0, F1b_dX, STOCK_dX, MERGE_N_dX, LABELMAP_dX)
  [date time] HANDSTOP word_A1 <job> <details ...>                  (the runner was stopped by hand; not a failed attempt)
  [date time] HANDSTOP_REASON word_A1 <job> <reason ...>            (the reason for that hand stop)
  [date time] GT_OPENED sha256 <marker sha256> outputs/results/vccor/final/C_word/look/GT_OPENED (maps.json sha256 <hex>; ...)
  [date time] GT_ABORTED sha256 <zip sha256> <zip path> (<details>)
  [date time] GT_CRASHED <exception type> <message ...>             (any exception after the marker; the look is not over)
  [date time] GT_CONTINUED sha256 <marker sha256> <marker path> (<details>)   (A1.1 continuation of the same single look)
  [date time] SECONDARY_ERROR word_A1 <type>: <message ...> (<details>)   (A1.2: the secondary analysis raised; the primary
                                                                     results and the RESULTS line are still written)
  [date time] READINESS sha256 <hex> <path> (<details>)             (CPU-only --check record before registration)

Version of 2026-11-05 (A1, A1.1) and 2026-11-06 (A1.2 and its
v5 revision). No import-time side effects beyond sys.path.
"""
import errno
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
FINAL = HERE.parent
if str(FINAL) not in sys.path:
    sys.path.insert(0, str(FINAL))
import final_common as F
import word_common as W

REPO = F.REPO
VC = F.VC
A1_JSON = F.OUT / "amend_word_A1.json"
A1_SHA256 = "5722f7edd782e3655bf2960f931c5affdc28f9abbf79026e86a86a0128f796a0"
A1_1_JSON = F.OUT / "amend_word_A1_1.json"
A1_1_SHA256 = "1f3008f362d15a6831808632450eab35dcbdfda6d94ccb0d8c2046502eda7226"
A1_2_JSON = F.OUT / "amend_word_A1_2.json"
A1_2_SHA256 = "27b9faf7bff7d1ae5101a6dac1792c711a9ccf499a19d621210b86dcbb7d602b"
A1_CODE_JSON = F.OUT / "amend_word_A1_code_v5.json"
A1_CODE_V4_JSON = F.OUT / "amend_word_A1_code_v4.json"
A1_CODE_V3_JSON = F.OUT / "amend_word_A1_code_v3.json"
A1_CODE_V2_JSON = F.OUT / "amend_word_A1_code.json"
DEPOSIT_V1_ZIP = VC / "osf_deposit_2026-11-05_word.zip"
DEPOSIT_V1_SHA256 = "ce765b1a01f53d04618332179e972d32a688a05dd6bf769282ff6957b462eed1"
DEPOSIT_V2_ZIP = VC / "osf_deposit_2026-11-05_word_v2.zip"
DEPOSIT_V2_SHA256 = "866bef3314371bd94303d91ebdcae5152bb26f13370ba913955f5ed9e8ace0cf"
DEPOSIT_V3_ZIP = VC / "osf_deposit_2026-11-05_word_v3.zip"
DEPOSIT_V3_SHA256 = "65b6b3f9f1044f866ea331322d8cd871c73584d56758ac48cd976c18f0a611f3"
DEPOSIT_V4_ZIP = VC / "osf_deposit_2026-11-06_word_v4.zip"
DEPOSIT_V4_SHA256 = "92a9b5a397b5ad0f9b915506dfc95f2049ca45b39d8321e106e0b80f57811fd7"
DEPOSIT_ZIP = VC / "osf_deposit_2026-11-06_word_v5.zip"
DEPOSIT_VERSION = "v5"
SUPERSEDED_DEPOSITS = {"v1": (DEPOSIT_V1_ZIP.name, DEPOSIT_V1_SHA256), "v2": (DEPOSIT_V2_ZIP.name, DEPOSIT_V2_SHA256),
                       "v3": (DEPOSIT_V3_ZIP.name, DEPOSIT_V3_SHA256), "v4": (DEPOSIT_V4_ZIP.name, DEPOSIT_V4_SHA256)}
OSF_FILE_NAME = "A1_osf_registration.txt"
OSF_URL_RE = re.compile(r"^https://osf\.io/[A-Za-z0-9]{5,}/?$")
CUTOFF = datetime(2026, 11, 27, 0, 0)
CUTOFF_TEXT = "a draw whose predictions are not DONE by 2026-11-27 00:00 KST is declared NOT RUN"
CUTOFF_A1_1 = "2026-11-27 00:00 KST"
LABELS_TS_BYTES = 4148364
MIN_FREE_GB = 30.0
GPU_IDLE_MIB = 3000
GPU_IDLE_WAIT_S = 12 * 3600
GPU_POLL_S = 300
GPU_LOCK = Path("/home/user/Documents/Handpose_project/runs/method_pilot3/gpu1.lock")
GPU_LOCK_WAIT_S = 12 * 3600
GPU_LOCK_POLL_S = 30
MAX_ATTEMPTS = 2

REAL_LOOK = F.PLAN_DIR["word"] / "look"
REAL_WORD_DATA = F.DATA / "word_v010"
REAL_LABELS_ZIP = REAL_WORD_DATA / "archive_labelsTs" / "WORD_V0.1.0_labelsTs.zip"
REAL_LABEL_PATHS = [REAL_WORD_DATA / "archive_labelsTs", REAL_WORD_DATA / "archive"]

DRAWS = ["d1", "d2", "d3"]
ARMS = ["F0"] + [f"F1b_{d}" for d in DRAWS]
REQUIRED = ["F0", "F1b_d1"]
NOTRUN_TAG = "word_A1"

STOCK_ARMS = [f"STOCK_{d}" for d in DRAWS]
MERGE_ARMS = [f"MERGE_N_{d}" for d in DRAWS]
LABELMAP_ARMS = [f"LABELMAP_{d}" for d in DRAWS]
A12_ARMS = STOCK_ARMS + MERGE_ARMS + LABELMAP_ARMS
ALL_ARMS = ARMS + A12_ARMS
ARM_LUT = dict({a: "ts" for a in ARMS}, **{a: "id" for a in STOCK_ARMS}, **{a: "ts" for a in MERGE_ARMS},
               **{a: "id" for a in LABELMAP_ARMS})
STOCK_DATASETS = {"d1": 824, "d2": 825, "d3": 826}
STOCK_N_CLASSES = W.N_ORG
STOCK_EST_MAX_H = {"prep": 0.5, "train": 7.0, "predict": 1.0}
STOCK_CHAIN_MAX_H = {"prep": 8.5, "train": 8.0, "predict": 1.0}

DROPPED_JOBS = [
    "pfinC_vista_v0_draws", "pfinC_vista_v0_test30", "pfinC_vista_labels", "pfinC_vista_train_VF_d1",
    "pfinC_vista_pred_VF_d1", "pfinC_vista_train_VF_d2", "pfinC_vista_pred_VF_d2", "pfinC_vista_train_VF_d3",
    "pfinC_vista_pred_VF_d3", "pfinC_vista_train_VF5k_d1", "pfinC_vista_pred_VF5k_d1", "pfinC_mrseg_m0_draws",
    "pfinC_mrseg_m0_test30", "pfinC_mrseg_prep", "pfinC_mrseg_train_MF_d1", "pfinC_mrseg_pred_MF_d1",
    "pfinC_mrseg_train_MF_d2", "pfinC_mrseg_pred_MF_d2", "pfinC_mrseg_train_MF_d3", "pfinC_mrseg_pred_MF_d3",
]
TS_JOBS = [
    "pfinC_pred_F0__word_test30",
    "pfinC_train_F1b_d1", "pfinC_pred_word_F1b_d1__word_test30",
    "pfinC_train_F1b_d2", "pfinC_pred_word_F1b_d2__word_test30",
    "pfinC_train_F1b_d3", "pfinC_pred_word_F1b_d3__word_test30",
]
A12_CPU_JOBS = [f"a12_cpu_MERGE_N_{d}" for d in DRAWS] + [f"a12_cpu_LABELMAP_{d}" for d in DRAWS]
A12_STOCK_JOBS = [n for d in DRAWS for n in (f"a12_prep_STOCK_{d}", f"a12_train_STOCK_{d}",
                                              f"a12_pred_STOCK_{d}__word_test30")]
A12_JOBS = A12_CPU_JOBS + A12_STOCK_JOBS
RUN_JOBS = TS_JOBS + A12_JOBS
OLD_SCORER_JOB = "pfinC_score"
SCHEDULER_NEUTRALISED = DROPPED_JOBS + TS_JOBS + [OLD_SCORER_JOB]
PREP_DONE_Q8 = ["pfinC_prep_front", "pfinC_prep_capture", "pfinC_prep_back"]

CODE_FILES = [HERE / "word_a1_common.py", HERE / "word_a1_score.py", HERE / "word_a1_run.py",
              HERE / "word_a1_rehearse.py", HERE / "word_a1_stock.py", HERE / "word_a1_cpu_arms.py",
              HERE / "test_word_a1.py", HERE / "ssl-word-a1-run.service", HERE / "ssl-word-a1-look.service"]
V2_COMMON = REPO / "scripts/vccor/paper_v2/v2_common.py"
V2_COMMON_SHA256 = "079e529ec78ec23f7b6cb1de7bdc2445dad113220fe780e47ddfe7baa98c6c63"
FROZEN_EXTRA = [V2_COMMON]
ONTOLOGY_PLAN = VC / "ontology_oracle/plan_ontology_oracle.json"
ONTOLOGY_PLAN_SHA256 = "44f7d2b70d94aee19d6ca6df8bca1c93bb6ab1447c034872fa427d9377f288d8"
ONTOLOGY_CODE = REPO / "scripts/vccor/review_fix_2026_11_06/ontology_oracle.py"
ONTOLOGY_CODE_SHA256 = "7271769257a7097d1c301be4911ad193f21cfadc59f2f23fc5feba243f6440e2"

GPU_ENV = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "1",
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
CPU_ENV = {"CUDA_VISIBLE_DEVICES": ""}


class Refuse(Exception):
    """A gate refused. code 2 = gate, 3 = resource (disk, GPU busy, lock), 4 = archive changed after the marker."""

    def __init__(self, msg, code=2):
        super().__init__(msg)
        self.msg = msg
        self.code = code


# ---------------------------------------------------------------------------------------------------- the jobs
def ts_jobs(pred_dir=None, models_dir=None):
    """The seven GPU jobs in A1's run order. Command lines == queue_v8.py's pfinC rows (checked by test_word_a1.py)."""
    P = Path(pred_dir) if pred_dir else W.DIR / "pred"
    M = Path(models_dir) if models_dir else W.DIR / "models"
    py = str(F.VENV)
    jobs = [dict(name="pfinC_pred_F0__word_test30", arm="F0", arms=["F0"], draw=None, kind="predict", gpu=True,
                 interp=py,
                 args=["-B", "scripts/vccor/final/predict_final.py", "--plan", "word", "--job", "F0__word_test30",
                       "--device", "gpu"],
                 done_file=str(P / "F0__word_test30" / "DONE.json"), deps=[], est_h=0.7)]
    for d in DRAWS:
        jobs.append(dict(name=f"pfinC_train_F1b_{d}", arm=f"F1b_{d}", arms=[f"F1b_{d}"], draw=d, kind="train", gpu=True,
                         interp=py,
                         args=["-B", "scripts/vccor/final/train_final.py", "--plan", "word", "--run", f"F1b_{d}",
                               "--device", "cuda", "--n_proc_da", "8"],
                         done_file=str(M / f"F1b_{d}" / "DONE.json"), deps=[], est_h=1.2))
        jobs.append(dict(name=f"pfinC_pred_word_F1b_{d}__word_test30", arm=f"F1b_{d}", arms=[f"F1b_{d}"], draw=d,
                         kind="predict", gpu=True, interp=py,
                         args=["-B", "scripts/vccor/final/predict_final.py", "--plan", "word", "--job",
                               f"word_F1b_{d}__word_test30", "--device", "gpu"],
                         done_file=str(P / f"word_F1b_{d}__word_test30" / "DONE.json"),
                         deps=[f"pfinC_train_F1b_{d}", "pfinC_pred_F0__word_test30"], est_h=0.3))
    assert [j["name"] for j in jobs] == TS_JOBS
    return jobs


def stock_prep_dir():
    return W.DIR / "prep_stock"


def a12_jobs(pred_dir=None, models_dir=None, prep_stock_dir=None):
    """The 15 jobs A1.2 adds, in its registered order (after every A1 job): the six CPU arm jobs MERGE_N d1-d3, then
    LABELMAP d1-d3 (minutes each; they need only F0 and F1b, so the long STOCK chain cannot delay them), then STOCK d1, d2,
    d3 (CPU prep, GPU train, GPU predict; lowest priority, last). Each STOCK job carries est_max_h, the registered maximum
    run time of the rest of its STOCK chain (prep 0.5 h + train 7.0 h + predict 1.0 h): it is launched only if that chain
    can finish before the cutoff."""
    P = Path(pred_dir) if pred_dir else W.DIR / "pred"
    M = Path(models_dir) if models_dir else W.DIR / "models"
    S = Path(prep_stock_dir) if prep_stock_dir else stock_prep_dir()
    py = str(F.VENV)
    st = "scripts/vccor/final/word_a1/word_a1_stock.py"
    ca = "scripts/vccor/final/word_a1/word_a1_cpu_arms.py"
    jobs = []
    for d in DRAWS:
        arm = f"MERGE_N_{d}"
        jobs.append(dict(name=f"a12_cpu_MERGE_N_{d}", arm=arm, arms=[arm], draw=d, kind="cpu_arm", gpu=False, interp=py,
                         args=["-B", ca, "--arm", "MERGE_N", "--draw", d],
                         done_file=str(P / f"MERGE_N_{d}__word_test30" / "DONE.json"),
                         deps=["pfinC_pred_F0__word_test30", f"pfinC_pred_word_F1b_{d}__word_test30"], est_h=0.1))
    for d in DRAWS:
        arm = f"LABELMAP_{d}"
        jobs.append(dict(name=f"a12_cpu_LABELMAP_{d}", arm=arm, arms=[arm], draw=d, kind="cpu_arm", gpu=False, interp=py,
                         args=["-B", ca, "--arm", "LABELMAP", "--draw", d],
                         done_file=str(P / f"LABELMAP_{d}__word_test30" / "DONE.json"),
                         deps=["pfinC_pred_F0__word_test30"], est_h=0.1))
    for d in DRAWS:
        arm = f"STOCK_{d}"
        jobs.append(dict(name=f"a12_prep_STOCK_{d}", arm=arm, arms=[arm], draw=d, kind="prep", gpu=False, interp=py,
                         args=["-B", st, "--stage", "prep", "--draw", d], done_file=str(S / f"DONE_stock_{d}.json"),
                         deps=[], est_h=0.25, est_max_h=STOCK_CHAIN_MAX_H["prep"]))
        jobs.append(dict(name=f"a12_train_STOCK_{d}", arm=arm, arms=[arm], draw=d, kind="train", gpu=True, interp=py,
                         args=["-B", st, "--stage", "train", "--draw", d, "--device", "cuda", "--n_proc_da", "8"],
                         done_file=str(M / f"STOCK_{d}" / "DONE.json"), deps=[f"a12_prep_STOCK_{d}"], est_h=5.4,
                         est_max_h=STOCK_CHAIN_MAX_H["train"]))
        jobs.append(dict(name=f"a12_pred_STOCK_{d}__word_test30", arm=arm, arms=[arm], draw=d, kind="predict", gpu=True,
                         interp=py, args=["-B", st, "--stage", "predict", "--draw", d, "--device", "gpu"],
                         done_file=str(P / f"word_STOCK_{d}__word_test30" / "DONE.json"), deps=[f"a12_train_STOCK_{d}"],
                         est_h=0.4, est_max_h=STOCK_CHAIN_MAX_H["predict"]))
    assert [j["name"] for j in jobs] == A12_JOBS
    return jobs


def run_jobs(pred_dir=None, models_dir=None, prep_stock_dir=None):
    """Every job of the run, in A1.2's registered order (A1's seven first, unchanged)."""
    jobs = ts_jobs(pred_dir, models_dir) + a12_jobs(pred_dir, models_dir, prep_stock_dir)
    assert [j["name"] for j in jobs] == RUN_JOBS
    return jobs


def job_arms(job):
    return list(job.get("arms") or [job["arm"]])


def arm_draw(arm):
    m = re.match(r"^.*_(d[123])$", arm)
    return m.group(1) if m else None


def arm_dirs(pred_dir=None):
    """arm -> prediction dir (raw maps + DONE.json); the TS arms == word_score.arms()."""
    P = Path(pred_dir) if pred_dir else W.DIR / "pred"
    out = {"F0": P / "F0__word_test30"}
    for d in DRAWS:
        out[f"F1b_{d}"] = P / f"word_F1b_{d}__word_test30"
    for d in DRAWS:
        out[f"STOCK_{d}"] = P / f"word_STOCK_{d}__word_test30"
        out[f"MERGE_N_{d}"] = P / f"MERGE_N_{d}__word_test30"
        out[f"LABELMAP_{d}"] = P / f"LABELMAP_{d}__word_test30"
    return out


# ---------------------------------------------------------------------------------------------------- context
def under_tmp(p):
    return str(Path(p).resolve()).startswith("/tmp/")


def touches_real_labels_or_look(p):
    """True if p is (inside) the real labelsTs archive folder, the real main archive folder or the real look directory.
    The protected paths are compared as normalised strings and are never stat-ed or resolved; only p is resolved, and only
    after its plain absolute form has been checked."""
    prot = [os.path.normpath(str(q)) for q in REAL_LABEL_PATHS + [REAL_LOOK]]

    def inside(s):
        return any(s == q or s.startswith(q + os.sep) for q in prot)

    a = os.path.normpath(os.path.abspath(str(p)))
    if inside(a):
        return True
    return inside(os.path.realpath(a))


class Ctx:
    """Every path a gate or a job touches. mode 'real' = the project; 'scratch' = unit tests (everything under /tmp);
    'rehearsal' = word_a1_rehearse.py (reads the prediction maps, writes only under /tmp, never the labels or the look)."""

    WRITE_PATHS = ("prereg_log", "osf_file", "look_dir", "state_dir", "event_log", "score_log", "job_log_dir",
                   "labels_zip")

    def __init__(self, real, prereg_log, osf_file, look_dir, pred_dir, models_dir, state_dir, event_log, score_log,
                 job_log_dir, decisions, q8_dir, deposit_rel, labels_zip, labels_zip_bytes, labels_zip_sha256=None,
                 cutoff=CUTOFF, min_free_gb=MIN_FREE_GB, gpu_lock=GPU_LOCK, code_files=None, mode=None, prep_dir=None):
        self.real = bool(real)
        self.mode = mode or ("real" if self.real else "scratch")
        assert self.mode in ("real", "scratch", "rehearsal") and (self.mode == "real") == self.real, self.mode
        self.prereg_log = Path(prereg_log)
        self.osf_file = Path(osf_file)
        self.look_dir = Path(look_dir)
        self.pred_dir = Path(pred_dir)
        self.models_dir = Path(models_dir)
        self.state_dir = Path(state_dir)
        self.event_log = Path(event_log)
        self.score_log = Path(score_log)
        self.job_log_dir = Path(job_log_dir)
        self.decisions = Path(decisions)
        self.q8_dir = Path(q8_dir)
        self.deposit_rel = str(deposit_rel)
        self.labels_zip = Path(labels_zip)
        self.labels_zip_bytes = labels_zip_bytes
        self.labels_zip_sha256 = labels_zip_sha256
        self.cutoff = cutoff
        self.min_free_gb = float(min_free_gb)
        self.gpu_lock = Path(gpu_lock)
        self.code_files = [Path(p) for p in (code_files or [])]
        self.prep_dir = Path(prep_dir) if prep_dir is not None else self.pred_dir.parent / "prep"
        if self.mode == "scratch":
            for p in self.paths():
                assert under_tmp(p), f"scratch context must stay under /tmp: {p}"
        if self.mode == "rehearsal":
            for k in self.WRITE_PATHS:
                p = getattr(self, k)
                assert under_tmp(p) and not touches_real_labels_or_look(p), f"rehearsal path {k} not a /tmp scratch: {p}"
            for p in (self.pred_dir, self.models_dir):
                assert not touches_real_labels_or_look(p), p

    def paths(self):
        return [self.prereg_log, self.osf_file, self.look_dir, self.pred_dir, self.models_dir, self.state_dir,
                self.event_log, self.score_log, self.job_log_dir, self.decisions, self.q8_dir, self.labels_zip,
                self.gpu_lock, self.prep_dir] + list(self.code_files)

    @classmethod
    def real_ctx(cls):
        return cls(True, F.PREREG_LOG, W.DIR / OSF_FILE_NAME, W.LOOK, W.DIR / "pred", W.DIR / "models", W.DIR / "a1_run",
                   F.LOGD / "final_word_a1_run.log", F.LOGD / "final_word_a1_score.log", REPO / "outputs/logs",
                   REPO / "outputs/logs/q8/decisions.json", REPO / "outputs/logs/q8", F.rel(DEPOSIT_ZIP),
                   W.LABELS_TS_ZIP, LABELS_TS_BYTES, None, prep_dir=W.PREP)

    @classmethod
    def scratch_ctx(cls, root, **over):
        """Unit tests only. The scorer also uses word_common's module paths (W.LOOK, W.DIR, W.LABELS_TS_ZIP, W.IMAGES,
        W.PREP, W.DOWNLOAD): the test must point those at the same scratch root first (asserted)."""
        root = Path(root).resolve()
        assert str(root).startswith("/tmp/"), root
        kw = dict(prereg_log=root / "logs/final_prereg.log", osf_file=root / "C_word" / OSF_FILE_NAME,
                  look_dir=root / "C_word/look", pred_dir=root / "C_word/pred", models_dir=root / "C_word/models",
                  state_dir=root / "C_word/a1_run", event_log=root / "logs/final_word_a1_run.log",
                  score_log=root / "logs/final_word_a1_score.log", job_log_dir=root / "logs/jobs",
                  decisions=root / "q8/decisions.json", q8_dir=root / "q8",
                  deposit_rel="outputs/results/vccor/osf_deposit_2026-11-06_word_v5.zip",
                  labels_zip=root / "archive_labelsTs/WORD_V0.1.0_labelsTs.zip", labels_zip_bytes=None,
                  labels_zip_sha256=None, gpu_lock=root / "gpu1.lock", prep_dir=root / "C_word/prep")
        kw.update(over)
        return cls(False, **kw)

    def check_word_paths(self):
        """The scorer reads references through word_common: those module paths must match this context."""
        if self.mode == "real":
            assert W.LOOK == self.look_dir and W.DIR / "pred" == self.pred_dir and W.LABELS_TS_ZIP == self.labels_zip
            return
        if self.mode == "scratch":
            for p in (W.LOOK, W.DIR, W.LABELS_TS_ZIP, W.IMAGES, W.PREP, W.DOWNLOAD):
                assert under_tmp(p), f"scratch run with a real word_common path: {p}"
        else:
            for p in (W.LOOK, W.LABELS_TS_ZIP):
                assert under_tmp(p) and not touches_real_labels_or_look(p), f"rehearsal with a real label / look path: {p}"
        assert Path(W.LOOK).resolve() == self.look_dir.resolve(), (W.LOOK, self.look_dir)
        assert (Path(W.DIR) / "pred").resolve() == self.pred_dir.resolve(), (W.DIR, self.pred_dir)
        assert Path(W.LABELS_TS_ZIP).resolve() == self.labels_zip.resolve(), (W.LABELS_TS_ZIP, self.labels_zip)

    def rel(self, p):
        return F.rel(p)

    def plog(self, msg):
        F.Log(self.prereg_log, also_print=False)(msg)

    def log_lines(self):
        if not self.prereg_log.is_file():
            return []
        return [ln.split() for ln in open(self.prereg_log)]

    def log_text_lines(self):
        if not self.prereg_log.is_file():
            return []
        return [ln.rstrip("\n") for ln in open(self.prereg_log)]


# ---------------------------------------------------------------------------------------------------- log parsing
def logged_tag_sha(ctx, tag, relpath):
    """(sha256, line index) of the last '<tag> sha256 <hex> <relpath>' line, or (None, None)."""
    h, idx = None, None
    for i, p in enumerate(ctx.log_lines()):
        if len(p) >= 6 and p[2] == tag and p[3] == "sha256" and len(p[4]) == 64 and p[5] == relpath:
            h, idx = p[4], i
    return h, idx


def notrun_declared(ctx):
    """{arm or 'EXPERIMENT': reason} from the NOTRUN word_A1 lines of final_prereg.log."""
    out = {}
    for p in ctx.log_lines():
        if len(p) >= 5 and p[2] == "NOTRUN" and p[3] == NOTRUN_TAG:
            out[p[4]] = " ".join(p[5:])
    return out


def declare_not_run(ctx, arm, reason):
    """Append 'NOTRUN word_A1 <arm> <reason>' once (before the look; refused after the marker)."""
    assert arm in ALL_ARMS or arm == "EXPERIMENT", arm
    if (ctx.look_dir / "GT_OPENED").exists():
        raise Refuse(f"cannot declare {arm} NOT RUN: the look has been made")
    if arm in notrun_declared(ctx):
        return False
    ctx.plog(f"NOTRUN {NOTRUN_TAG} {arm} {reason}")
    return True


def frozen_code_mismatch(ctx, files):
    """{path: (sha256 now, last CODE / IMPORT sha256 in ctx.prereg_log)} for every file that differs from its last line."""
    fz = {}
    for p in ctx.log_lines():
        if len(p) >= 5 and p[2] in ("CODE", "IMPORT") and len(p[3]) == 64:
            fz[" ".join(p[4:])] = p[3]
    bad = {}
    for f in files:
        k = ctx.rel(f)
        h = F.sha256(f) if Path(f).is_file() else None
        if h is None or fz.get(k) != h:
            bad[k] = (str(h)[:16], str(fz.get(k) or "NOT-FROZEN")[:16])
    return bad


# ---------------------------------------------------------------------------------------------------- gates
def osf_gate(ctx):
    """(url, record) if the OSF gate holds, else raises Refuse. The gate file's first line is the public osf.io URL; the
    file names the v5 zip's sha256 as logged on its DEPOSIT line and no superseded zip (v1, v2, v3, v4; name or sha256);
    final_prereg.log holds 'OSF-REGISTRATION <url> sha256 <gate file sha256> <gate file path>' after the v5 DEPOSIT line."""
    f = ctx.osf_file
    if not f.is_file():
        raise Refuse(f"OSF gate: {ctx.rel(f)} does not exist (the owner writes it after the OSF registration is public)")
    txt = f.read_text()
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    url = lines[0] if lines else ""
    if not OSF_URL_RE.match(url):
        raise Refuse(f"OSF gate: the first line of {ctx.rel(f)} is not an https://osf.io/<id> URL: {url[:80]!r}")
    for ver, (name, h) in SUPERSEDED_DEPOSITS.items():
        if h in txt or name in txt:
            raise Refuse(f"OSF gate: the gate file names the {ver} zip ({name}, {h[:8]}...), which is superseded and "
                         f"was never uploaded; only the {DEPOSIT_VERSION} zip may be registered")
    dep_sha, dep_idx = logged_tag_sha(ctx, "DEPOSIT", ctx.deposit_rel)
    if dep_sha is None:
        raise Refuse(f"OSF gate: no DEPOSIT line for {ctx.deposit_rel} in {ctx.rel(ctx.prereg_log)}")
    if dep_sha in [h for _, h in SUPERSEDED_DEPOSITS.values()]:
        raise Refuse(f"OSF gate: the DEPOSIT line of {ctx.deposit_rel} carries a superseded zip sha256 {dep_sha[:8]}...")
    if dep_sha not in txt:
        raise Refuse(f"OSF gate: the gate file does not name the {DEPOSIT_VERSION} zip sha256 {dep_sha[:16]}... (its "
                     f"DEPOSIT line)")
    fsha = F.sha256(f)
    hit = None
    for i, p in enumerate(ctx.log_lines()):
        if (len(p) >= 7 and p[2] == "OSF-REGISTRATION" and p[3].rstrip("/") == url.rstrip("/") and p[4] == "sha256"
                and p[5] == fsha and p[6] == ctx.rel(f)):
            hit = i
    if hit is None:
        raise Refuse(f"OSF gate: no 'OSF-REGISTRATION {url} sha256 {fsha[:16]}... {ctx.rel(f)}' line in "
                     f"{ctx.rel(ctx.prereg_log)} (the line must match the gate file as it is now)")
    if hit < dep_idx:
        raise Refuse(f"OSF gate: the OSF-REGISTRATION line precedes the {DEPOSIT_VERSION} DEPOSIT line")
    return url, {"url": url, "gate_file": ctx.rel(f), "gate_file_sha256": fsha, "deposit_zip": ctx.deposit_rel,
                 "deposit_zip_sha256": dep_sha, "log_line_index": hit}


def _pinned_amendment(ctx, path, pinned, name):
    if not Path(path).is_file():
        raise Refuse(f"integrity: {F.rel(path)} missing")
    h = F.sha256(path)
    lg, _ = logged_tag_sha(ctx, "AMENDMENT", F.rel(path))
    if not (h == pinned == lg):
        raise Refuse(f"integrity: {name} sha256 {h[:16]} / pinned {pinned[:16]} / logged {str(lg)[:16]}")
    return h


def integrity(ctx, script, log=print, extra=()):
    """Real run only: plan C hashes to its PREREG line; every code file of scripts/vccor/final/*.py, the word_a1 files and
    the frozen imports (incl. paper_v2/v2_common.py, the MERGE_N operator) hash to their CODE / IMPORT lines; A1, A1.1, A1.2
    and the v5 code amendment hash to their AMENDMENT lines; A1's, A1.1's and A1.2's cutoff is the constant above. Returns
    (plan, record)."""
    if not ctx.real:
        return None, {"skipped": f"{ctx.mode} context"}
    try:
        plan = F.preflight("word", script, log, extra=[FINAL / "fin_look.py", FINAL / "word_common.py",
                                                        FINAL / "dec_common.py", FINAL / "word_score.py"]
                           + list(CODE_FILES) + list(FROZEN_EXTRA) + list(extra))
    except SystemExit as e:
        raise Refuse(f"integrity: {e}")
    h = _pinned_amendment(ctx, A1_JSON, A1_SHA256, "amend_word_A1.json")
    a1 = json.load(open(A1_JSON))
    if not str(a1["stopping_rule"]["cutoff"]).startswith(CUTOFF_TEXT):
        raise Refuse("integrity: A1's cutoff text differs from the runner's constant")
    h11 = _pinned_amendment(ctx, A1_1_JSON, A1_1_SHA256, "amend_word_A1_1.json")
    a11 = json.load(open(A1_1_JSON))
    if a11["decisions"]["cutoff"]["cutoff"] != CUTOFF_A1_1:
        raise Refuse("integrity: A1.1's cutoff differs from the runner's constant")
    h12 = _pinned_amendment(ctx, A1_2_JSON, A1_2_SHA256, "amend_word_A1_2.json")
    a12 = json.load(open(A1_2_JSON))
    if a12["stopping_rule_A1_2"]["cutoff"] != CUTOFF_A1_1:
        raise Refuse("integrity: A1.2's cutoff differs from the runner's constant")
    if V2_COMMON_SHA256 != F.sha256(V2_COMMON):
        raise Refuse("integrity: scripts/vccor/paper_v2/v2_common.py (the MERGE_N operator) changed")
    if not A1_CODE_JSON.is_file():
        raise Refuse(f"integrity: {F.rel(A1_CODE_JSON)} missing")
    hc = F.sha256(A1_CODE_JSON)
    lc, _ = logged_tag_sha(ctx, "AMENDMENT", F.rel(A1_CODE_JSON))
    if hc != lc:
        raise Refuse(f"integrity: {A1_CODE_JSON.name} sha256 {hc[:16]} != logged AMENDMENT {str(lc)[:16]}")
    return plan, {"plan_word_sha256": F.sha256(F.plan_path("word")), "a1_sha256": h, "a1_1_sha256": h11,
                  "a1_2_sha256": h12, "a1_code_v5_sha256": hc}


def runs_queue_v8(argv):
    """True iff argv is a python interpreter executing queue_v8.py (script path or -m queue_v8). An editor, pager or
    'python -c ... queue_v8.py' is not matched."""
    if not argv or not re.match(r"^python[0-9.]*$", os.path.basename(argv[0])):
        return False
    i = 1
    while i < len(argv):
        a = argv[i]
        if a == "-m":
            return i + 1 < len(argv) and argv[i + 1] == "queue_v8"
        if a in ("-W", "-X", "-Q"):
            i += 2
            continue
        if a.startswith("-") and a != "-":
            if not a.startswith("--") and "c" in a[1:]:
                return False
            if not a.startswith("--") and a.endswith("m"):
                return i + 1 < len(argv) and argv[i + 1] == "queue_v8"
            i += 1
            continue
        return os.path.basename(a) == "queue_v8.py"
    return False


def queue_v8_pids():
    """pids of python processes executing queue_v8.py (other than this process)."""
    me, out = os.getpid(), []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit() or int(d.name) == me:
            continue
        try:
            argv = [a.decode("utf-8", "replace") for a in (d / "cmdline").read_bytes().split(b"\0") if a]
        except OSError:
            continue
        if runs_queue_v8(argv):
            out.append(int(d.name))
    return out


def scheduler_gate(ctx, pids_fn=queue_v8_pids):
    """queue_v8.py must not be running, and decisions.json must drop every pfinC row A1 neutralises (no queue_v8 row runs an
    A1.2 job: test_word_a1.py T10 / T30 check the live table)."""
    pids = pids_fn()
    if pids:
        raise Refuse(f"queue_v8.py is running (pids {pids}); A1 forbids restarting it (see amend_word_A1_code_v5.json)")
    try:
        d = json.load(open(ctx.decisions))
    except (OSError, ValueError) as e:
        raise Refuse(f"cannot read {ctx.rel(ctx.decisions)}: {e}")
    missing = [n for n in SCHEDULER_NEUTRALISED if n not in set(d.get("drop", []))]
    if missing:
        raise Refuse(f"{ctx.rel(ctx.decisions)} 'drop' lacks the A1 rows {missing[:4]}{'...' if len(missing) > 4 else ''}")
    return {"queue_v8_running": False, "neutralised_rows": len(SCHEDULER_NEUTRALISED)}


def a12_inputs_gate(ctx):
    """A1.2's CPU inputs exist (metadata records only; no label file is opened or stat-ed here): the WORD prep's labels
    manifest and TS-capture record (LABELMAP rules), and the prep's image record (STOCK datasets)."""
    need = [ctx.prep_dir / "labels" / "manifest.json", ctx.prep_dir / "ts_cpu" / "CAPTURE_DONE.json",
            ctx.prep_dir / "images.json"]
    miss = [ctx.rel(p) for p in need if not p.is_file()]
    if miss:
        raise Refuse(f"A1.2 inputs missing: {miss}")
    return "prep/labels/manifest.json, prep/ts_cpu/CAPTURE_DONE.json and prep/images.json present"


def look_not_made(ctx):
    for n in ("GT_OPENED", "results.json"):
        if (ctx.look_dir / n).exists():
            raise Refuse(f"the WORD look has already been made ({ctx.rel(ctx.look_dir / n)} exists): no second look")
    return True


def kst_gate():
    if time.strftime("%z") != "+0900":
        raise Refuse(f"local time zone is {time.strftime('%z')}, not KST (+0900): A1's cutoff is in KST")
    return True


def disk_gate(ctx):
    free = shutil.disk_usage("/").free / 1e9
    if free < ctx.min_free_gb:
        raise Refuse(f"/ has {free:.1f} GB free (< {ctx.min_free_gb:.0f} GB): NOT LAUNCHED", code=3)
    return free


def nvidia_gpu1_used_mib():
    """memory.used of GPU 1 (PCI bus order, == nvidia-smi's index) in MiB, or None if the query fails."""
    try:
        r = subprocess.run(["nvidia-smi", "-i", "1", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=60)
        return int(r.stdout.strip().splitlines()[0]) if r.returncode == 0 and r.stdout.strip() else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def wait_gpu1_idle(used_fn=nvidia_gpu1_used_mib, wait_s=GPU_IDLE_WAIT_S, poll_s=GPU_POLL_S, log=print):
    """Wait until GPU 1 uses < 3000 MiB (as queue_v8 and the gap-stage chains); Refuse(code 3) after wait_s."""
    t0 = time.time()
    while True:
        used = used_fn()
        if used is not None and used < GPU_IDLE_MIB:
            return used
        if time.time() - t0 >= wait_s:
            raise Refuse(f"GPU 1 not idle for {wait_s / 3600:.1f} h (memory.used {used} MiB): NOT LAUNCHED", code=3)
        log(f"GPU 1 busy ({used} MiB used); waiting")
        time.sleep(poll_s)


def acquire_gpu_lock(path, wait_s=GPU_LOCK_WAIT_S, poll_s=GPU_LOCK_POLL_S, log=print):
    """An fd holding an exclusive flock on the shared GPU-1 lock file (A1.1). The file is opened read-only: it is never
    created, written or truncated. Refuse(code 3) if it is missing or still held by another job after wait_s."""
    p = Path(path)
    if not p.is_file():
        raise Refuse(f"the shared GPU-1 lock file {p} does not exist (it is never created here): NOT LAUNCHED", code=3)
    fd = os.open(p, os.O_RDONLY | os.O_CLOEXEC)
    t0, said = time.time(), False
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except OSError as e:
            if e.errno not in (errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES):
                os.close(fd)
                raise Refuse(f"flock on {p} failed: {e}: NOT LAUNCHED", code=3)
        if time.time() - t0 >= wait_s:
            os.close(fd)
            raise Refuse(f"the shared GPU-1 lock {p} was held by another job for {wait_s / 3600:.1f} h: NOT LAUNCHED",
                         code=3)
        if not said:
            log(f"the shared GPU-1 lock {p} is held by another job; waiting (up to {wait_s / 3600:.1f} h)")
            said = True
        time.sleep(poll_s)


def release_gpu_lock(fd):
    if fd is None:
        return
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------------------------------- A1.2 helpers
def real_job_guard(script, log=print):
    """A real (non-test) A1.2 job: no look exists, the OSF gate holds and the word_a1 code + frozen imports hash to their
    CODE / IMPORT lines. The runner checks the same before every launch; this repeats it in the job's own process, so a job
    started by hand before the registration refuses too."""
    ctx = Ctx.real_ctx()
    look_not_made(ctx)
    osf_gate(ctx)
    integrity(ctx, script, log)
    return ctx
