"""vccor/final LCTSC training queue (run by the systemd user unit ssl-lctsc-train): lctsc_train.py --run F1b_S1, F1b_S2,
F1b_S3 (site-matched draws), in this order. Rebuilt after the independent review (B4) on the WORD-N runner pattern
(scripts/vccor/final/wordn/wn_run.py run_job / run_until_done / backoff), COPIED, not imported:

  * the queue never touches GPU 1 or the lock: each training job takes GPU 1 itself (lctsc_infra.gpu_session: identity
    gate, idle wait WITHOUT the lock, lock, re-check) and holds the shared flock until it exits; so the lock is taken per
    job and released between jobs, and the lock file is never created;
  * before each launch the queue waits for disk space (/ >= 25 GiB with 2 GiB more), bounded by the training deadline
    TRAIN_DEADLINE = Sun 2026-11-15 18:00 KST; there is no 12 h give-up;
  * an attempt is recorded only after Popen succeeded;
  * exit 0 with DONE.json = DONE; exit 3 (a resource refusal) or 4 (the deadline passed while waiting) is NOT a failure;
    any other exit is a failure; a failure waits a 10-minute back-off before attempt 2; two failures = FAILED;
  * FAILED is reset only by an explicit, logged command (--reset RUN --reason TEXT, a 'RESET lctsc' line), never by a
    restart;
  * a hand stop (SIGTERM / SIGHUP / SIGINT) stops the running job's process group and is logged (HANDSTOP), not a failure.
Registered consequence (plan_lctsc.json 'training_failure_rule'): a draw FAILED twice, or not DONE by TRAIN_DEADLINE, is
NOT RUN; P1 then runs on the draws that remain (WORD A1 / FLARE-T50 rule: two-level over >= 2 draws, the single-draw form
for one; the combination over the draws run); no draw trained = no look (NOTRUN EXPERIMENT).
State: K_lctsc/train_state.json; events: outputs/logs/vccor/final_lctsc.log; job logs K_lctsc/train_logs/.

usage:  .venv/bin/python -I -B scripts/vccor/final/lctsc/lctsc_train_queue.py [--runs F1b_S1,F1b_S2,F1b_S3]
        .venv/bin/python -I -B scripts/vccor/final/lctsc/lctsc_train_queue.py --reset F1b_S2 --cause infrastructure --reason "..."

Version of 2026-11-13.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lctsc_infra as I  # noqa: E402
import lctsc_log as L  # noqa: E402
import lctsc_runs as R  # noqa: E402

PY = "/home/user/Documents/Medical Imagining SSL/.venv/bin/python"
STATE = R.C.OUT / "train_state.json"
MAX_FAIL = 2
_CHILD = {"p": None}
ST = None


class HandStop(SystemExit):
    pass


def _kill(p):
    if p is None or p.poll() is not None:
        return
    try:
        os.killpg(p.pid, signal.SIGTERM)
        p.wait(timeout=300)
    except (OSError, subprocess.TimeoutExpired):
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except OSError:
            pass


def _on_signal(signum, _frm):
    _kill(_CHILD["p"])
    L.event(f"[train-queue] HANDSTOP signal {signum}")
    raise HandStop(128 + signum)


def load():
    return json.load(open(STATE)) if STATE.is_file() else {"runs": {}}


def save():
    L.write_json(STATE, ST)


def run_once(run, rs, deadline):
    """-> 'done' / 'failed' / 'resource' / 'deadline'."""
    try:
        I.wait_disk(2.0, f"training {run}", deadline, log=L.event)
    except I.Refuse:
        return "deadline"
    if not I.before(deadline):
        return "deadline"
    n = len(rs["attempts"]) + 1
    logf = R.C.OUT / "train_logs" / f"{run}_attempt{n}.log"
    logf.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, **I.GPU_ENV, PYTHONDONTWRITEBYTECODE="1")
    p = None
    try:
        with open(logf, "a") as out:
            p = subprocess.Popen([PY, "-I", "-B", str(HERE / "lctsc_train.py"), "--run", run, "--device", "cuda",
                                  "--n_proc_da", "8"], stdout=out, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            _CHILD["p"] = p
            att = {"attempt": n, "start": L.now(), "log": L.rel(logf), "pid": p.pid}      # recorded only after the launch
            rs["attempts"].append(att)
            save()
            L.event(f"[train-queue] START {run} attempt {n} (pid {p.pid}; the job takes GPU 1 and the lock itself)")
            t0 = time.time()
            rc = p.wait()
            _CHILD["p"] = None
        att.update(end=L.now(), rc=rc, wall_h=round((time.time() - t0) / 3600, 3))
        if rc == 0 and (R.MODELS / run / "DONE.json").is_file():
            return "done"
        if rc == I.DEADLINE_CODE:
            return "deadline"
        if rc == 3:
            att["resource_refusal"] = True
            return "resource"
        rs["failures"] = rs.get("failures", 0) + 1
        return "failed"
    finally:
        _kill(p)
        save()


def handle(run, deadline):
    rs = ST["runs"].setdefault(run, {"attempts": [], "failures": 0})
    if (R.MODELS / run / "DONE.json").is_file():
        rs["status"] = "DONE"
        save()
        return
    if rs.get("status") == "FAILED":
        L.event(f"[train-queue] {run} is FAILED (reset only with --reset): skipped")
        return
    try:
        R.verify_datasets(run, "F1b")
    except SystemExit as e:                          # review N6: an integrity refusal fails only this draw
        rs["status"] = "INTEGRITY REFUSED"
        rs["integrity"] = str(e)[:300]
        save()
        L.event(f"[train-queue] {run}: dataset integrity refused ({e}): NOT RUN under the registered rule; next draw")
        return
    while True:
        r = run_once(run, rs, deadline)
        if r == "done":
            rs["status"] = "DONE"
            rs["done_sha256"] = L.sha256(R.MODELS / run / "DONE.json")
            save()
            L.event(f"[train-queue] DONE {run}")
            return
        if r == "deadline":
            rs["status"] = "NOT DONE BY THE TRAINING DEADLINE"
            save()
            L.event(f"[train-queue] {run}: training deadline {deadline} passed: NOT RUN under the registered rule")
            return
        if r == "failed" and rs["failures"] >= MAX_FAIL:
            rs["status"] = "FAILED"
            save()
            L.event(f"[train-queue] FAILED {run} after {MAX_FAIL} failed attempts: NOT RUN under the registered rule")
            return
        L.event(f"[train-queue] {run}: {r}; back-off {I.BACKOFF_S} s before the next attempt")
        if not I.backoff(deadline):
            rs["status"] = "NOT DONE BY THE TRAINING DEADLINE"
            save()
            return


def main():
    global ST
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="F1b_S1,F1b_S2,F1b_S3")
    ap.add_argument("--reset", default=None)
    ap.add_argument("--reason", default=None)
    ap.add_argument("--cause", default=None, choices=["infrastructure"])
    a = ap.parse_args()
    ST = load()
    if a.reset:
        # registered RESET policy (plan_lctsc.json 'training_reset_policy'): only before the training deadline, only for a
        # FAILED state whose failures were infrastructure faults (not the model or its data), with a reason; each use is
        # logged ('RESET lctsc') and disclosed
        assert a.reason and a.cause == "infrastructure", "--reset needs --reason and --cause infrastructure"
        assert I.before(I.TRAIN_DEADLINE), "no RESET after the training deadline"
        assert not (R.C.OUT / "code_freeze.json").exists(), "no RESET after the freeze (review NB2)"
        rs = ST["runs"].get(a.reset)
        assert rs and rs.get("status") == "FAILED", f"{a.reset} is not FAILED"
        rs.setdefault("resets", []).append({"time": L.now(), "reason": a.reason, "failures_before": rs.get("failures", 0)})
        rs["status"], rs["failures"] = "RESET", 0
        save()
        L.prereg_note("RESET", f"training {a.reset}: FAILED state reset by an explicit command (cause: infrastructure); "
                               f"reason: {a.reason}")
        return 0
    for s in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(s, _on_signal)
    L.event(f"[train-queue] start {a.runs}; training deadline {I.TRAIN_DEADLINE}")
    for run in a.runs.split(","):
        handle(run, I.TRAIN_DEADLINE)
    L.event("[train-queue] finished: " + ", ".join(f"{r} {ST['runs'].get(r, {}).get('status')}" for r in a.runs.split(",")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
