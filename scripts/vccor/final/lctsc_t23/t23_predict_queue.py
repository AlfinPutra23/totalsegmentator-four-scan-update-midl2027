"""vccor/final LCTSC-T23 PREDICTION RUNNER (run by the systemd user unit ssl-lctsc-t23-predict). DERIVED from the frozen
lctsc_predict_queue.py by derive_t23.py (every change is listed there). The registered order: F0, then F1b_S1, F1b_S2,
F1b_S3 (the frozen d6ayj models), each one job of t23_predict.py (real mode) on the 23 LCTSC-T23 CTs. Rules as LCTSC:

  * the runner waits (cutoff-aware, polled every 5 min) until the upload gate holds, then checks the freeze
    and that no look exists before every job;
  * each GPU job takes GPU 1 itself (GPU-1 UUID identity gate, idle wait WITHOUT the lock, lock, re-check; every wait
    bounded by the cutoff Fri 2026-11-20 10:00 KST) and holds the shared flock until it exits;
  * exit 0 + DONE.json = DONE; exit 3 = a resource refusal (retried after the back-off until the cutoff); exit 4 = the
    cutoff passed while waiting; any other exit = a failure; a failure waits a 10-minute back-off before its second
    attempt; a second failure declares the arm NOT RUN ('NOTRUN lctsc_t23 <arm> ...', binding);
  * a job still running at the cutoff is stopped; after the cutoff every arm not DONE is declared NOT RUN;
  * an F1b arm needs F0 DONE (F0 NOT RUN -> 'NOTRUN lctsc_t23 EXPERIMENT', no look);
  * a hand stop (SIGTERM / SIGHUP / SIGINT) stops the running job and is logged (HANDSTOP).
The look is NOT started here. State: K_lctsc_t23/run/<arm>.json; events: outputs/logs/vccor/final_lctsc_t23.log.

usage:  .venv/bin/python -I -B scripts/vccor/final/lctsc_t23/t23_predict_queue.py [--check]

Version of 2026-11-13; LCTSC-T23 derivation 2026-11-17.
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
sys.path.insert(0, str(HERE.parent / "lctsc"))
sys.path.insert(0, str(HERE))
import t23_gates as G  # noqa: E402
import lctsc_infra as I  # noqa: E402
import t23_log as L  # noqa: E402
import t23_common as T  # noqa: E402

PY = "/home/user/Documents/Medical Imagining SSL/.venv/bin/python"
RUN = G.OUT / "run"
ORDER = ["F0", "F1b_S1", "F1b_S2", "F1b_S3"]
MAX_FAIL = 2
GATE_POLL_S = 300
_CHILD = {"p": None}


class HandStop(SystemExit):
    pass


def _kill(p):
    if p is None or p.poll() is not None:
        return
    try:
        os.killpg(p.pid, signal.SIGTERM)
        p.wait(timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except OSError:
            pass


def _on_signal(signum, _frm):
    _kill(_CHILD["p"])
    L.event(f"[predict-run] HANDSTOP signal {signum}")
    raise HandStop(128 + signum)


def state(arm):
    p = RUN / f"{arm}.json"
    return json.load(open(p)) if p.is_file() else {"arm": arm, "attempts": [], "failures": 0}


def save(arm, st):
    G.write_json(RUN / f"{arm}.json", st)


def declared_not_run(arm):
    return bool(state(arm).get("not_run")) or G.notrun_declared(arm)


def declare_not_run(arm, reason):
    st = state(arm)
    if st.get("not_run") or G.arm_done(arm)[0] == "DONE":
        return
    st["not_run"] = reason
    save(arm, st)
    L.prereg_note("NOTRUN", f"{arm} {reason}")
    L.event(f"[predict-run] NOT RUN {arm}: {reason}")


def wait_upload_gate():
    said = False
    while True:
        try:
            return G.gate_uploaded()
        except G.Refuse as e:
            if not G.before_cutoff():
                raise
            if not said:
                L.event(f"[predict-run] waiting for the upload gate ({e})")
                said = True
            time.sleep(GATE_POLL_S)


def gates():
    G.kst_gate()
    G.freeze_ok()
    G.gate_uploaded()
    if G.look_made():
        raise G.Refuse("a look exists: nothing runs after the look")


def training_not_run(arm):
    """review NB1: the FROZEN training status (code_freeze.json draws_status), never the live train_state."""
    if G.ARMS[arm]["run"] is None:
        return None
    st = G.frozen_draw_status(arm)
    return None if st == "DONE" else f"its training draw {G.ARMS[arm]['run']} is {st} in the freeze"


def run_job(arm):
    """-> 'done' / 'failed' / 'resource' / 'cutoff'."""
    try:
        I.wait_disk(0.5, f"prediction {arm}", T.CUTOFF_DT, log=L.event)
    except I.Refuse:
        return "cutoff"
    if not G.before_cutoff():
        return "cutoff"
    st = state(arm)
    n = len(st["attempts"]) + 1
    lp = RUN / "logs" / f"{arm}_attempt{n}.log"
    lp.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, **I.GPU_ENV, PYTHONDONTWRITEBYTECODE="1")
    p = None
    try:
        with open(lp, "a") as lf:
            p = subprocess.Popen([PY, "-I", "-B", str(HERE / "t23_predict.py"), "--arm", arm, "--device", "gpu"],
                                 cwd=str(G.REPO), env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
            _CHILD["p"] = p
            att = {"attempt": n, "start": G.now(), "log": L.rel(lp), "pid": p.pid}
            st["attempts"].append(att)
            save(arm, st)
            L.event(f"[predict-run] START {arm} attempt {n}")
            cut = False
            while p.poll() is None:
                time.sleep(10)
                if not G.before_cutoff():
                    cut = True
                    _kill(p)
                    break
            rc = p.wait()
            _CHILD["p"] = None
        att.update(end=G.now(), rc=rc, stopped_at_cutoff=cut)
        if cut or rc == I.DEADLINE_CODE or not G.before_cutoff():
            return "cutoff"
        if rc == 0 and G.arm_done(arm)[0] == "DONE":
            return "done"
        if rc == 3:
            att["resource_refusal"] = True
            return "resource"
        st["failures"] += 1
        return "failed"
    finally:
        _kill(p)
        save(arm, st)


def handle(arm):
    if G.arm_done(arm)[0] == "DONE" or declared_not_run(arm):
        return
    why = training_not_run(arm)
    if why:
        declare_not_run(arm, why)
        return
    if arm != "F0" and G.arm_done("F0")[0] != "DONE":
        declare_not_run(arm, "F0 is not DONE (its frozen-part cache feeds the replay)")
        return
    while G.arm_done(arm)[0] != "DONE":
        if state(arm)["failures"] >= MAX_FAIL:
            declare_not_run(arm, f"{MAX_FAIL} failed attempts")
            return
        gates()
        r = run_job(arm)
        if r == "cutoff":
            return
        if r == "done":
            L.event(f"[predict-run] DONE {arm}")
            return
        if r == "failed" and state(arm)["failures"] >= MAX_FAIL:
            continue
        L.event(f"[predict-run] {arm}: {r}; back-off {I.BACKOFF_S} s")
        if not I.backoff(T.CUTOFF_DT):
            return


def after_cutoff():
    if G.before_cutoff():
        return
    for a in ORDER:
        if G.arm_done(a)[0] != "DONE":
            declare_not_run(a, "not DONE by the cutoff (Fri 2026-11-20 10:00 KST)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        gates()
        print({x: G.arm_done(x)[0] for x in ORDER}, {x: training_not_run(x) for x in ORDER})
        return 0
    for s in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(s, _on_signal)
    G.kst_gate()
    wait_upload_gate()
    gates()
    L.event("[predict-run] start: " + ", ".join(ORDER))
    for arm in ORDER:
        handle(arm)
    if G.arm_done("F0")[0] != "DONE" and declared_not_run("F0"):
        L.prereg_note("NOTRUN", "EXPERIMENT: F0 NOT RUN, no look")
    after_cutoff()
    L.event("[predict-run] finished: " + ", ".join(f"{x} {G.arm_done(x)[0]}" for x in ORDER))
    return 0


if __name__ == "__main__":
    sys.exit(main())
