"""vccor/final FLARE-T50 TIER-1 RUNNER (GPU 1): predicts every arm on the included FLARE-T50 scans, in
t50_common.RUN_ORDER (F0 first: it captures the frozen parts the other 'ts' arms replay; then the 13 other tier-1 arms;
then the four tier-2 K = 1 arms once their models exist). The look is NOT started here.

Refuses to start (nothing launched) unless: KST; the gate_osf_uploaded.json record holds (the OSF registration was
initiated); no look exists; exclusions.json exists (K1b); images_record.json exists; k1c.json says K1C_PASS.
Before EVERY launch: the same gates; the cutoff (Fri 2026-11-13 18:00 KST) has not passed; / keeps >= 35 GiB free; an
exclusive flock on the shared GPU-1 lock (held until the job has exited; waits up to 12 h); GPU 1 idle (< 3000 MiB; waits up
to 12 h); the cutoff again.

Stopping rule: a failed job is rerun once, unchanged (t50_predict.py resumes per scan); a second failure declares the arm
NOT RUN ('NOTRUN flaret50 <arm> ...' in final_prereg.log). A job still running at the cutoff is stopped and its arm declared
NOT RUN; after the cutoff every arm not DONE is declared NOT RUN. F0 or FULL_s1337 NOT RUN -> 'NOTRUN flaret50 EXPERIMENT'
(no look). A K = 1 arm whose model was declared NOT RUN by the tier-2 runner is declared NOT RUN without a launch; a K = 1
arm whose training is still running is waited for (polled every 5 min) until it is DONE, declared NOT RUN, or the cutoff.
A DONE arm is never re-run. A K = 1 model trained after the freeze is hashed by a freeze addendum before
its own arm's prediction (its time relative to the upload is disclosed); a K = 1 arm whose model fails the freeze check is
declared NOT RUN. A hand stop (SIGTERM / SIGHUP / SIGINT) stops the running job and is logged ('HANDSTOP').
State: I_flaret50/run/<arm>.json; events: outputs/logs/vccor/final_flaret50_run.log.

usage:  .venv/bin/python -I -B scripts/vccor/final/flaret50/t50_run.py          (normally via the ssl-flaret50-run unit)
        .venv/bin/python -I -B scripts/vccor/final/flaret50/t50_run.py --check  (gates and status; writes nothing
            except one GATE line in final_prereg.log on the upload gate file's first use)

Version of 2026-11-09.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import t50_common as T  # noqa: E402

EVLOG = T.F.LOGD / "final_flaret50_run.log"
REQUIRED = ["F0", "FULL_s1337"]
MAX_ATTEMPTS = 2
_CHILD = {"p": None}


class HandStop(SystemExit):
    def __init__(self, signum):
        super().__init__(128 + signum)
        self.signum = signum


def log(m):
    T.log_event(m, EVLOG)


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


def state(arm):
    p = T.RUN / f"{arm}.json"
    return json.load(open(p)) if p.is_file() else {"arm": arm, "attempts": 0, "failures": 0, "history": []}


def save(arm, st):
    T.write_json(T.RUN / f"{arm}.json", st)


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


def tier2_model_status(arm):
    d = T.ARMS[arm]["draw"]
    if (T.FF_MODELS / d / "DONE.json").is_file():
        return "done"
    nr = T.TIER2 / "NOT_RUN.json"
    st = T.TIER2 / "state.json"
    if nr.is_file():
        return "not_run"
    if st.is_file() and json.load(open(st))["draws"].get(d, {}).get("not_run"):
        return "not_run"
    return "pending"


def gates():
    T.kst_gate()
    T.freeze_ok()
    g = T.gate_uploaded()
    if T.look_made():
        raise T.Refuse("a look exists: no prediction after the look")
    for p in (T.EXCLUSIONS, T.IMAGES_RECORD, T.K1C / "k1c.json"):
        if not p.is_file():
            raise T.Refuse(f"{T.rel(p)} missing")
    if json.load(open(T.K1C / "k1c.json")).get("decision") != "K1C_PASS":
        raise T.Refuse("K1c did not pass")
    if json.load(open(T.EXCLUSIONS)).get("decision") != "K1B_PASS":
        raise T.Refuse("K1b did not pass")
    return g


def run_job(arm):
    st = state(arm)
    T.disk_gate(0.3, f"prediction {arm}")
    fd = T.acquire_gpu_lock(log=log)
    p = None
    try:
        T.wait_gpu1_idle(log=log)
        if not T.before_cutoff():
            return "cutoff"
        st["attempts"] += 1
        n = st["attempts"]
        lp = T.RUN / "logs" / f"{arm}_attempt{n}.log"
        lp.parent.mkdir(parents=True, exist_ok=True)
        cmd = [str(T.VENV), "-I", "-B", str(HERE / "t50_predict.py"), "--arm", arm, "--device", "gpu"]
        st["history"].append({"attempt": n, "start": T.now(), "cmd": cmd, "log": T.rel(lp)})
        save(arm, st)
        log(f"START {arm} attempt {n}")
        t0 = time.time()
        with open(lp, "a") as lf:
            p = subprocess.Popen(cmd, cwd=str(T.REPO), env=dict(os.environ, **T.GPU_ENV, PYTHONDONTWRITEBYTECODE="1"),
                                 stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
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
        st = state(arm)
        st["history"][-1].update(end=T.now(), rc=rc, wall_s=round(time.time() - t0, 1), stopped_at_cutoff=cut)
        if cut:
            save(arm, st)
            return "cutoff"
        if rc == 0 and is_done(arm):
            st["history"][-1]["done_sha256"] = T.sha256(T.done_file(arm))
            save(arm, st)
            log(f"DONE {arm} in {(time.time() - t0) / 60:.1f} min")
            return "done"
        st["failures"] += 1
        save(arm, st)
        log(f"FAILED {arm} attempt {n} rc {rc}")
        return "failed"
    finally:
        _kill(p)
        T.release_gpu_lock(fd)


def after_cutoff_declarations():
    for a in T.RUN_ORDER:
        if not is_done(a) and not not_run(a):
            declare_not_run(a, "not DONE by the cutoff (Fri 2026-11-13 18:00 KST)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    try:
        g = gates()
    except T.Refuse as e:
        if a.check:
            print(f"[REFUSE] {e.msg}")
            for arm in T.RUN_ORDER:
                print(f"  {arm}: done={is_done(arm)} not_run={state(arm).get('not_run')}")
            return e.code
        raise
    if a.check:
        print(f"gates OK (registration {g['osf_url']} uploaded {g['uploaded_kst']})")
        for arm in T.RUN_ORDER:
            print(f"  {arm}: done={is_done(arm)} not_run={state(arm).get('not_run')}")
        return 0
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, _on_signal)
    T.RUN.mkdir(parents=True, exist_ok=True)
    log(f"tier-1 runner start pid {os.getpid()}; registration {g['osf_url']} uploaded {g['uploaded_kst']}")
    try:
        for arm in T.RUN_ORDER:
            if is_done(arm) or not_run(arm):
                continue
            if not T.before_cutoff():
                after_cutoff_declarations()
                break
            if T.ARMS[arm]["tier"] == 2:
                while True:
                    s = tier2_model_status(arm)
                    if s != "pending" or not T.before_cutoff():
                        break
                    log(f"{arm}: its K = 1 model is still training; waiting")
                    time.sleep(300)
                if s == "not_run":
                    declare_not_run(arm, "its K = 1 training was declared NOT RUN by the tier-2 runner")
                    continue
                if s == "pending":
                    after_cutoff_declarations()
                    break
            if T.ARMS[arm]["tier"] == 2 and T.frozen_model_hashes(arm) is None:
                import t50_freeze as FZ
                try:
                    FZ.addendum()
                except T.Refuse as e:
                    log(f"{arm}: freeze addendum not written: {e.msg}")
            while not is_done(arm) and not not_run(arm):
                if state(arm)["failures"] >= MAX_ATTEMPTS:
                    declare_not_run(arm, f"{MAX_ATTEMPTS} failed attempts")
                    break
                gates()
                try:
                    T.check_frozen_model(arm)
                except T.Refuse as e:
                    if T.ARMS[arm]["tier"] != 2:
                        raise
                    declare_not_run(arm, f"its model failed the freeze check: {e.msg}")
                    break
                r = run_job(arm)
                if r == "cutoff":
                    after_cutoff_declarations()
                    break
            if arm in REQUIRED and not is_done(arm):
                T.prereg_note("NOTRUN", f"EXPERIMENT required arm {arm} NOT RUN: no look")
                log(f"EXPERIMENT NOT RUN ({arm})")
                return 0
        fin = {a_: ("DONE" if is_done(a_) else f"NOT RUN: {state(a_).get('not_run')}") for a_ in T.RUN_ORDER}
        T.write_json(T.RUN / "RUN_FINISHED.json", {"arms": fin, "finished": T.now()})
        log("tier-1 runner finished: " + ", ".join(f"{k} {v.split(':')[0]}" for k, v in fin.items()))
        return 0
    except HandStop as e:
        T.prereg_note("HANDSTOP", f"tier-1 runner stopped by signal {e.signum}")
        raise


if __name__ == "__main__":
    try:
        sys.exit(main())
    except T.Refuse as e:
        print(f"[REFUSE] {e.msg}", file=sys.stderr)
        sys.exit(e.code)
