"""vccor/final LCTSC GPU-1 / disk resource primitives (review B3 / B4). COPIED from the reviewed WORD-N code
scripts/vccor/final/wordn/wn_common.py (InfraDown, nvsmi_rows, _gpu1_row, wait_infra, gpu1_used_mib, wait_gpu1_idle,
gpu_identity_gate, acquire_gpu_lock, release_gpu_lock, gpu_session, disk_gate / wait_disk) and wn_run.py (backoff), NOT
imported from it, with one change: every wait is bounded by an explicit DEADLINE (a datetime) instead of WORD-N's cutoff
-- the training queue passes TRAIN_DEADLINE (Sun 2026-11-15 18:00 KST), the prediction runner the registered cutoff
(Thu 2026-11-19 18:00 KST). Rules (as WORD-N):
  * real GPU jobs run on GPU 1 only: CUDA_VISIBLE_DEVICES must be GPU 1's UUID, listed by nvidia-smi on bus 05:00.0;
  * the shared lock file /home/user/Documents/Handpose_project/runs/method_pilot3/gpu1.lock is opened read-only and NEVER
    created; a missing lock file is waited for;
  * wait for GPU 1 to be idle (< 3000 MiB) WITHOUT holding the lock, then take the lock, re-check idleness, and release and
    repeat if GPU 1 is busy again; the lock is held by the GPU job itself until it exits;
  * nvidia-smi runs with a 60 s timeout; any nvidia-smi fault is an outage (InfraDown) that is waited out, never a failure;
  * disk space below the floor is waited for, never a refusal;
  * every wait ends only at the deadline (Refuse code 4 = DEADLINE_CODE).
Standard library only.
"""
import fcntl
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

GPU1_UUID = "GPU-1a7b7614-499a-6540-55fd-5920dcee8340"
GPU_BUS = "00000000:05:00.0"
GPU_LOCK = Path("/home/user/Documents/Handpose_project/runs/method_pilot3/gpu1.lock")
GPU_IDLE_MIB = 3000
GPU_POLL_S = 60
INFRA_POLL_S = 300
DISK_POLL_S = 300
LOCK_POLL_S = 30
FLOOR_GIB = 25.0
DEADLINE_CODE = 4
BACKOFF_S = 600
TRAIN_DEADLINE = datetime(2026, 11, 15, 18, 0)
CUTOFF = datetime(2026, 11, 19, 18, 0)
GPU_ENV = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": GPU1_UUID,
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
LOCK_HELD_ENV = "LCTSC_GPU1_LOCK_HELD"


class Refuse(SystemExit):
    def __init__(self, msg, code=2):
        super().__init__(code)
        self.msg, self.code = msg, code
        import sys
        print(f"REFUSE (code {code}): {msg}", file=sys.stderr, flush=True)       # review N2: the reason reaches the job log

    def __str__(self):
        return self.msg


class InfraDown(Exception):
    """nvidia-smi missing / failing / timing out / printing an error, GPU 1 or its UUID not listed: not available YET."""


def before(deadline, t=None):
    return (t or datetime.now()) < deadline


def disk_free_gib(path="/"):
    return shutil.disk_usage(str(path)).free / 1024 ** 3


def disk_short(need_gib, what):
    """None if / keeps the floor with need_gib more, else the reason (no Refuse object, so nothing is printed)."""
    free = disk_free_gib()
    if free - float(need_gib) < FLOOR_GIB:
        return (f"{what}: / has {free:.2f} GiB free; with {need_gib:.2f} GiB more it would fall below the "
                f"{FLOOR_GIB:.0f} GiB floor")
    return None


def disk_gate(need_gib, what):
    why = disk_short(need_gib, what)
    if why:
        raise Refuse(why, code=3)
    return disk_free_gib()


def wait_disk(need_gib, what, deadline, log=print, poll_s=None):
    """Cutoff/deadline-aware wait for disk space; logs once per wait (review diff-2 N2)."""
    said = False
    while True:
        why = disk_short(need_gib, what)
        if why is None:
            return disk_free_gib()
        if not before(deadline):
            raise Refuse(f"the deadline {deadline} passed while waiting for disk space ({why})", code=DEADLINE_CODE)
        if not said:
            log(f"{why}: waiting for space")
            said = True
        time.sleep(poll_s or DISK_POLL_S)


def nvsmi_rows():
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


def wait_infra(fn, what, deadline, log=print, poll_s=None):
    said = False
    while True:
        try:
            return fn()
        except InfraDown as e:
            if not before(deadline):
                raise Refuse(f"the deadline {deadline} passed while waiting for {what} ({e})", code=DEADLINE_CODE)
            if not said:
                log(f"{what} not available ({e}); waiting (polled every {poll_s or INFRA_POLL_S} s)")
                said = True
            time.sleep(poll_s or INFRA_POLL_S)


def gpu1_used_mib():
    return _gpu1_row(nvsmi_rows())[2]


def wait_gpu1_idle(deadline, poll_s=GPU_POLL_S, log=print, used_fn=None):
    used_fn = used_fn or gpu1_used_mib
    said = None
    while True:
        try:
            u, why = used_fn(), None
        except InfraDown as e:
            u, why = None, str(e)
        if u is not None and u < GPU_IDLE_MIB:
            return u
        if not before(deadline):
            raise Refuse(f"the deadline {deadline} passed while waiting for GPU 1 to be idle", code=DEADLINE_CODE)
        msg = (f"GPU 1 busy ({u} MiB)" if u is not None else f"GPU 1 not available ({why})") + "; waiting"
        if msg.split(" (")[0] != said:
            log(msg)
            said = msg.split(" (")[0]
        time.sleep(poll_s)


def gpu_identity_gate(deadline, log=print, poll_s=None):
    v = os.environ.get("CUDA_VISIBLE_DEVICES")
    if v != GPU1_UUID:
        raise Refuse(f"CUDA_VISIBLE_DEVICES is {v!r}, not GPU 1's UUID {GPU1_UUID}: real GPU jobs run on GPU 1 only")

    def find():
        for u, bus, _used in nvsmi_rows():
            if u == GPU1_UUID:
                return bus
        raise InfraDown(f"GPU 1's UUID {GPU1_UUID} not listed by nvidia-smi")
    bus = wait_infra(find, "GPU 1 (nvidia-smi)", deadline, log=log, poll_s=poll_s)
    if not bus.upper().endswith(GPU_BUS.upper()[-12:]):
        raise Refuse(f"GPU 1's UUID is on bus {bus}, not {GPU_BUS}")
    return True


def acquire_gpu_lock(deadline, path=None, poll_s=LOCK_POLL_S, log=print, missing_poll_s=None):
    path = Path(path or GPU_LOCK)

    def opened():
        if not path.is_file():
            raise InfraDown(f"the GPU-1 lock file {path} is missing (it belongs to the Handpose project's tooling; "
                            f"LCTSC never creates it)")
        try:
            return os.open(str(path), os.O_RDONLY)
        except OSError as e:
            raise InfraDown(f"the GPU-1 lock file {path} cannot be opened: {e}")
    fd = wait_infra(opened, "the GPU-1 lock file", deadline, log=log, poll_s=missing_poll_s)
    said = False
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except BlockingIOError:
            if not before(deadline):
                os.close(fd)
                raise Refuse(f"the deadline {deadline} passed while waiting for the GPU-1 lock", code=DEADLINE_CODE)
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


def gpu_session(deadline, log=print, poll_s=None, idle_poll_s=None, lock_poll_s=LOCK_POLL_S):
    """GPU identity gate; wait for GPU 1 idle WITHOUT the lock; take the lock; re-check idleness; release and repeat if busy
    again. Returns the lock fd; the caller holds it until it exits (release_gpu_lock or process exit)."""
    gpu_identity_gate(deadline, log=log, poll_s=poll_s)
    ip = idle_poll_s or GPU_POLL_S
    while True:
        wait_gpu1_idle(deadline, log=log, poll_s=ip)
        fd = acquire_gpu_lock(deadline, log=log, poll_s=lock_poll_s, missing_poll_s=poll_s)
        try:
            u = gpu1_used_mib()
        except InfraDown:
            u = None
        if u is not None and u < GPU_IDLE_MIB:
            return fd
        release_gpu_lock(fd)
        log(f"GPU 1 busy again after the lock was taken ({u} MiB); lock released, waiting")
        if not before(deadline):
            raise Refuse(f"the deadline {deadline} passed while waiting for GPU 1", code=DEADLINE_CODE)
        time.sleep(ip)


def backoff(deadline, seconds=BACKOFF_S):
    """Wait before a retry; False if the deadline passes meanwhile (== wn_run.backoff)."""
    t_end = time.time() + seconds
    while time.time() < t_end:
        if not before(deadline):
            return False
        time.sleep(min(30, max(0.0, t_end - time.time())))
    return before(deadline)
