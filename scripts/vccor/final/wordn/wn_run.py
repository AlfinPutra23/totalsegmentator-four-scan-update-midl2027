"""vccor/final WORD-N RUNNER (GPU 1 under the shared flock): every job of the registered run, in this order:
  1. the intake (CPU): the 20 validation images, whitelist-extracted; the duplicate rule (wn_intake.py --run);
  2. tier 1 GPU predictions: F0 (captures the frozen TS parts), F1b_d1, F1b_d2, F1b_d3, STOCK_d1, STOCK_d2, STOCK_d3;
  3. tier 1 CPU arms: MERGE_N_d1-d3, LABELMAP_d1-d3;
  4. tier 2 (only if vsmoke/smoke.json says VSMOKE_PASS; otherwise every tier-2 arm is declared NOT RUN): V0, then per run
     VF5k_d1, VF5k_d2, VF5k_d3, VF5k_lrA_d1: training (GPU), a freeze addendum hashing the new weights, prediction (GPU).
The look is NOT started here.

Refuses to start (nothing launched) unless: KST; the freeze holds; the gate_osf_uploaded.json record holds; no look.
Before EVERY job: the same gates; the cutoff (Thu 2026-11-19 18:00 KST) has not passed; / keeps >= 25 GiB free. A GPU job
is launched with CUDA_VISIBLE_DEVICES = GPU 1's UUID and handles the shared flock
/home/user/Documents/Handpose_project/runs/method_pilot3/gpu1.lock ITSELF: it waits for GPU 1 to be idle without the lock,
takes the lock, re-checks idleness, and releases and repeats if GPU 1 is busy again; every wait is cutoff-aware (exit code
4 = the cutoff passed while waiting). Waits for disk space, GPU-infrastructure outages
(nvidia-smi failing, GPU 1 or its UUID not listed, the lock file missing), the lock and GPU 1's idleness happen inside the
job and end only at the cutoff; exit 3 (a resource refusal) is kept as a code but no post-upload job raises it.

Stopping rule: a failed job is rerun once, unchanged (predictions resume per scan); a second failure declares its arm NOT
RUN ('NOTRUN wordn <arm> ...'). A resource refusal (exit 3, or a Refuse code 3 in the runner) is NOT a failure: the job is
retried after a 10-minute back-off until the cutoff, and it never declares an arm NOT RUN by itself (NB1); disk space,
GPU-infrastructure outages, the shared lock and GPU 1's idleness are waited for, bounded by the cutoff only (NB2). A failed
job waits a 10-minute back-off before its second attempt, so one passing outage cannot use both attempts. A job still running at the cutoff is
stopped; after the cutoff every arm not DONE is declared NOT RUN. Any other error while handling one arm declares that
arm NOT RUN (unless its DONE already exists: then the DONE is kept) and the runner goes on, so it never exits with arms
undeclared (except on a gate refusal, a hand stop or a required-arm failure; the look then counts every arm not DONE by the
cutoff as NOT RUN). An arm whose inputs are NOT RUN is NOT RUN (CPU arms; F1b needs F0); the VF arms are NOT RUN when V0 is
NOT RUN (V-P1 needs V0). F0 or F1b_d1 NOT RUN -> 'NOTRUN wordn EXPERIMENT' (no look). A DONE arm is never re-run. A hand
stop (SIGTERM / SIGHUP / SIGINT) stops the running job and is logged ('HANDSTOP').
State: I_wordn/run/<arm>.json; events: I_wordn/logs/run.log.

usage:  .venv/bin/python -I -B scripts/vccor/final/wordn/wn_run.py           (normally via the ssl-wordn-run unit)
        .venv/bin/python -I -B scripts/vccor/final/wordn/wn_run.py --check   (gates and status; writes nothing except one
            GATE line in final_prereg.log on the upload gate file's first use)

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
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import wn_common as T  # noqa: E402

def evlog():
    return T.OUT / "logs" / "run.log"          # resolved at call time (use_root in the unit tests)
MAX_ATTEMPTS = 2
BACKOFF_S = 600
_CHILD = {"p": None}


class HandStop(SystemExit):
    def __init__(self, signum):
        super().__init__(128 + signum)
        self.signum = signum


def log(m):
    T.log_event(m, evlog())


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


def _on_signal(signum, frame):
    _kill(_CHILD["p"])
    raise HandStop(signum)


def state(key):
    p = T.RUN / f"{key}.json"
    return json.load(open(p)) if p.is_file() else {"arm": key, "attempts": 0, "failures": 0, "history": []}


def save(key, st):
    T.write_json(T.RUN / f"{key}.json", st)


def is_done(arm):
    return T.done_file(arm).is_file()


def not_run(arm):
    return bool(state(arm).get("not_run"))


def declare_not_run(arm, reason):
    st = state(arm)
    if st.get("not_run"):
        return
    st["not_run"] = reason
    save(arm, st)
    T.prereg_note("NOTRUN", f"{arm} {reason}")
    log(f"NOT RUN {arm}: {reason}")


def gates():
    T.kst_gate()
    T.freeze_ok()
    g = T.gate_uploaded()
    if T.look_made():
        raise T.Refuse("a look exists: nothing runs after the look")
    T.overshoot_mode()
    return g


def smoke_passed():
    return T.smoke_passed()


def job_cmd(key):
    """(command, gpu, done_check) of a job key: 'intake', '<arm>' (predict / CPU arm) or 'train:<run>'."""
    py = [str(T.VENV), "-I", "-B"]
    if key == "intake":
        return py + [str(HERE / "wn_intake.py"), "--run"], False, lambda: T.EXCLUSIONS.is_file()
    if key.startswith("train:"):
        r = key.split(":", 1)[1]
        return (py + [str(HERE / "wn_vista.py"), "--train", "--run", r], True,
                lambda: (T.VISTA / "models" / r / "DONE.json").is_file())
    if T.arms()[key]["kind"] == "cpu":
        return py + [str(HERE / "wn_cpu_arms.py"), "--arm", key], False, lambda: is_done(key)
    return py + [str(HERE / "wn_predict.py"), "--arm", key, "--device", "gpu"], True, lambda: is_done(key)


def run_job(key):
    """-> 'done' / 'failed' / 'cutoff' / 'resource'. The GPU job takes the shared lock itself (SF3)."""
    cmd, gpu, done = job_cmd(key)
    T.wait_disk(0.5 if gpu else 1.5, key, log=log)          # a cutoff-aware wait for space (NB1), never a refusal
    st = state(key)
    p = None
    try:
        if not T.before_cutoff():
            return "cutoff"
        st["attempts"] += 1
        n = st["attempts"]
        lp = T.RUN / "logs" / f"{key.replace(':', '_')}_attempt{n}.log"
        lp.parent.mkdir(parents=True, exist_ok=True)
        st["history"].append({"attempt": n, "start": T.now(), "cmd": cmd, "log": T.rel(lp)})
        save(key, st)
        log(f"START {key} attempt {n}")
        t0 = time.time()
        env = dict(os.environ, **(T.GPU_ENV if gpu else T.CPU_ENV), PYTHONDONTWRITEBYTECODE="1")
        with open(lp, "a") as lf:
            p = subprocess.Popen(cmd, cwd=str(T.REPO), env=env, stdout=lf, stderr=subprocess.STDOUT,
                                 start_new_session=True)
            _CHILD["p"] = p
            cut = False
            while p.poll() is None:
                time.sleep(10)
                if not T.before_cutoff():
                    cut = True
                    _kill(p)
                    break
            rc = p.wait()
            _CHILD["p"] = None
        st = state(key)
        st["history"][-1].update(end=T.now(), rc=rc, wall_s=round(time.time() - t0, 1), stopped_at_cutoff=cut)
        if cut:
            save(key, st)
            return "cutoff"
        if rc == 0 and done():
            save(key, st)
            log(f"DONE {key} in {(time.time() - t0) / 60:.1f} min")
            return "done"
        if rc == T.CUTOFF_CODE or not T.before_cutoff():
            save(key, st)
            return "cutoff"
        if rc == 3:
            st["resource_refusals"] = st.get("resource_refusals", 0) + 1
            save(key, st)
            log(f"RESOURCE REFUSAL {key} attempt {n} (rc 3): retried after a back-off, not counted as a failure")
            return "resource"
        st["failures"] += 1
        save(key, st)
        log(f"FAILED {key} attempt {n} rc {rc}")
        return "failed"
    finally:
        _kill(p)


def run_until_done(key, arm):
    """Run one job (rerun once). -> 'done' / 'not_run' / 'cutoff'."""
    _, _, done = job_cmd(key)
    while not done():
        if not_run(arm):
            return "not_run"
        if state(key)["failures"] >= MAX_ATTEMPTS:
            declare_not_run(arm, f"{MAX_ATTEMPTS} failed attempts of job {key}")
            return "not_run"
        gates()
        if key == arm:                    # a prediction / CPU-arm job (a training job hashes its model afterwards)
            try:
                T.check_frozen_model(arm)
            except T.Refuse as e:
                declare_not_run(arm, f"its model failed the freeze check: {e.msg}")
                return "not_run"
        try:
            r = run_job(key)
        except T.Refuse as e:
            if e.code == T.CUTOFF_CODE:
                return "cutoff"
            if e.code != 3:
                raise
            r = "resource"
            log(f"RESOURCE REFUSAL {key}: {e.msg}")
        if r == "cutoff":
            return "cutoff"
        if r == "resource" and not backoff():      # NB1: never a NOT RUN by itself; retried until the cutoff
            return "cutoff"
        if r == "failed" and state(key)["failures"] < MAX_ATTEMPTS and not backoff():   # NB2: back-off before attempt 2
            return "cutoff"
    return "done"


def backoff(seconds=BACKOFF_S):
    """Wait before retrying a job after a resource refusal; False if the cutoff passes meanwhile."""
    t_end = time.time() + seconds
    while time.time() < t_end:
        if not T.before_cutoff():
            return False
        time.sleep(min(30, max(0.0, t_end - time.time())))
    return T.before_cutoff()


def after_cutoff_declarations():
    if T.before_cutoff():                          # N-E3: only after the cutoff
        log("after_cutoff_declarations called before the cutoff: nothing declared")
        return
    for a in T.RUN_ORDER:
        if not is_done(a) and not not_run(a):
            declare_not_run(a, "not DONE by the cutoff (Thu 2026-11-19 18:00 KST)")


def inputs_not_run(arm):
    a = T.arms()[arm]
    need = list(a.get("inputs") or [])
    if a["kind"] == "ts" and arm != "F0":
        need.append("F0")
    return [i for i in need if not_run(i) or not is_done(i)]


def handle_arm(arm):
    """Everything for one arm; any error other than a gate refusal or a hand stop declares the arm NOT RUN (SF1)."""
    try:
        if arm in T.TIER2 and not smoke_passed():
            declare_not_run(arm, "the VISTA3D smoke test did not pass before the look (tier 2 NOT RUN)")
            return "not_run"
        if arm != "F0" and arm != "V0":
            miss = inputs_not_run(arm)
            if miss:
                declare_not_run(arm, f"its inputs are not DONE: {miss}")
                return "not_run"
        arm_def = T.arms()[arm]
        if arm_def["kind"] == "vista" and arm != "V0":
            if not_run("V0") or not is_done("V0"):
                declare_not_run(arm, "V0 is NOT RUN: V-P1 cannot be read, so the VF training is skipped (N7)")
                return "not_run"
            r = run_until_done(f"train:{arm}", arm)
            if r != "done":
                return r
            if T.frozen_model_hashes(arm) is None:
                import wn_freeze as FZ
                try:
                    FZ.addendum()
                except Exception as e:                    # noqa: BLE001  (the arm then fails its freeze check: NOT RUN)
                    log(f"{arm}: freeze addendum not written: {type(e).__name__}: {e}")
        return run_until_done(arm, arm)
    except HandStop:
        raise
    except T.Refuse as e:
        if e.code == 2:
            raise
        if e.code == T.CUTOFF_CODE:
            return "cutoff"
        if is_done(arm):
            log(f"{arm}: refusal after its DONE ({e.msg}); the DONE is kept")
            return "done"
        declare_not_run(arm, f"refusal while handling the arm: {e.msg}")
        return "not_run"
    except Exception as e:                                # noqa: BLE001
        if is_done(arm):                                  # N-D2: a valid DONE is kept; the runner error is only logged
            log(f"{arm}: runner error after its DONE ({type(e).__name__}: {str(e)[:200]}); the DONE is kept")
            return "done"
        declare_not_run(arm, f"error while handling the arm: {type(e).__name__}: {str(e)[:200]}")
        return "not_run"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    try:
        g = gates()
    except T.Refuse as e:
        if a.check:
            print(f"[REFUSE] {e.msg}")
            return e.code
        raise
    if a.check:
        print(f"gates OK (registration {g['osf_url']} uploaded {g['uploaded_kst']}); smoke passed {smoke_passed()}")
        for arm in T.RUN_ORDER:
            print(f"  {arm}: done={is_done(arm)} not_run={state(arm).get('not_run')}")
        return 0
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, _on_signal)
    T.RUN.mkdir(parents=True, exist_ok=True)
    log(f"WORD-N runner start pid {os.getpid()}; registration {g['osf_url']} uploaded {g['uploaded_kst']}")
    try:
        if not T.EXCLUSIONS.is_file():
            r = run_until_done("intake", "F0")
            if r != "done":
                if r == "cutoff":
                    after_cutoff_declarations()
                T.prereg_note("NOTRUN", "EXPERIMENT the intake did not finish: no look")
                return 0
        for arm in T.TIER1 + T.TIER2:
            if is_done(arm) or not_run(arm):
                continue
            if not T.before_cutoff():
                after_cutoff_declarations()
                break
            r = handle_arm(arm)
            if r == "cutoff":
                after_cutoff_declarations()
                break
            if arm in T.REQUIRED and not is_done(arm):
                T.prereg_note("NOTRUN", f"EXPERIMENT required arm {arm} NOT RUN: no look")
                log(f"EXPERIMENT NOT RUN ({arm})")
                return 0
        fin = {a_: ("DONE" if is_done(a_) else f"NOT RUN: {state(a_).get('not_run')}") for a_ in T.RUN_ORDER}
        T.write_json(T.RUN / "RUN_FINISHED.json", {"arms": fin, "finished": T.now()})
        log("WORD-N runner finished: " + ", ".join(f"{k} {v.split(':')[0]}" for k, v in fin.items()))
        return 0
    except HandStop as e:
        T.prereg_note("HANDSTOP", f"WORD-N runner stopped by signal {e.signum}")
        raise


if __name__ == "__main__":
    try:
        sys.exit(main())
    except T.Refuse as e:
        print(f"[REFUSE] {e.msg}", file=sys.stderr)
        sys.exit(e.code)
