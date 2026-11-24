"""vccor/final plan C, amendments word A1, A1.1 and A1.2: GATED RUNNER of the 22 WORD jobs, in A1.2's registered run order
(amend_word_A1_2.json run_order_with_A1_2; A1's seven GPU jobs first and unchanged):
  A1      1 pfinC_pred_F0__word_test30   2 pfinC_train_F1b_d1   3 pfinC_pred_word_F1b_d1__word_test30
          4 pfinC_train_F1b_d2           5 pfinC_pred_word_F1b_d2__word_test30
          6 pfinC_train_F1b_d3           7 pfinC_pred_word_F1b_d3__word_test30
  A1.2    8-13 the CPU arms a12_cpu_MERGE_N_d1..d3, then a12_cpu_LABELMAP_d1..d3;
          14-22 STOCK, lowest priority (last): a12_prep_STOCK_dX (CPU), a12_train_STOCK_dX (GPU),
          a12_pred_STOCK_dX__word_test30 (GPU) for d1, d2, d3
A1's jobs are queue_v8.py's own command lines for those rows (word_a1_common.ts_jobs; equality checked by test_word_a1.py);
A1.2's jobs run word_a1_stock.py (the frozen train_final.py / predict_final.py, unchanged) and word_a1_cpu_arms.py; no
queue_v8 row runs any of them. Each runs with the project venv from the repo root; stdout / stderr to
outputs/logs/<job>.log (appended, as queue_v8). A job is DONE when it exits 0 and its done file exists. The look is NOT
started here: after this runner has finished, the author runs the rehearsal (word_a1_rehearse.py) and then word_a1_score.py
(or the ssl-word-a1-look unit).

Refuses to start (nothing launched, nothing written) unless
  * no WORD look exists (C_word/look/GT_OPENED, results.json);
  * the OSF gate holds (C_word/A1_osf_registration.txt + its OSF-REGISTRATION line; word_a1_common.osf_gate; v5 deposit);
  * plan C, every code file and frozen import, A1, A1.1, A1.2 and the v5 code amendment hash to their logged lines;
  * queue_v8.py is not running (a python process executing it) and outputs/logs/q8/decisions.json drops every pfinC row A1
    neutralises;
  * the local time zone is KST; Plan C's three prep jobs are done (q8 markers); no 'NOTRUN word_A1 EXPERIMENT' line;
  * every HANDSTOP line has its HANDSTOP_REASON line (A1.1); A1.2's CPU inputs are recorded (prep records).
Before EVERY launch (a refusal here stops the runner, exit 3, and counts no attempt): the gates above again; / has >= 30 GB
free. A GPU job then takes the shared GPU-1 lock (A1.1): an exclusive flock on /home/user/Documents/Handpose_project/runs/
method_pilot3/gpu1.lock (waits up to 12 h; the file is opened read-only and never created or written), held until the job's
main process exits: the job inherits the locked descriptor, and the runner releases the lock (LOCK_UN) as soon as that
process has exited, even if a grandchild still holds the descriptor; the next GPU launch's idle check covers such a
survivor. Then GPU 1 must be idle (memory.used < 3000 MiB; waits up to 12 h); then the cutoff is checked AGAIN (A1.1: after
the waits). GPU jobs see only GPU 1 (CUDA_DEVICE_ORDER=PCI_BUS_ID, CUDA_VISIBLE_DEVICES=1) with PYTORCH_CUDA_ALLOC_CONF=
expandable_segments:True. CPU jobs (STOCK prep, MERGE_N, LABELMAP) take no lock and see no GPU (CUDA_VISIBLE_DEVICES='').

A1's stopping rule, with A1.1 and A1.2:
  * a failed job is rerun once, unchanged (train_final.py resumes from its checkpoint); a second failure -> its arm is declared
    NOT RUN in final_prereg.log ('NOTRUN word_A1 <arm> ...');
  * nothing is launched at or after 2026-11-27 00:00 KST (checked before and after the lock / idle waits); a job still
    running then is stopped; every arm not DONE by then is declared NOT RUN (run the runner once after the cutoff to write
    the declarations if it was not running);
  * A1.2, STOCK (lowest priority: after every A1 job and the six CPU arm jobs): a STOCK job is launched only if the
    registered maximum run time of the rest of its draw's STOCK chain (prep 0.5 h + train 7.0 h + predict 1.0 h: 8.5 / 8.0
    / 1.0 h) ends before the cutoff (checked before and after the waits); otherwise its STOCK draw is declared NOT RUN
    without a launch;
  * A1.2: an A1.2 arm whose input arm was declared NOT RUN (MERGE_N_dX without F1b_dX) is declared NOT RUN;
  * F0 or F1b_d1 NOT RUN -> 'NOTRUN word_A1 EXPERIMENT ...' and the runner stops (no look); F0 not DONE by the cutoff is
    therefore the experiment NOT RUN; an A1.2 arm NOT RUN never stops the runner;
  * a DONE job is never re-run (A1.1): it is skipped, and the runner refuses if its DONE record changed after the runner
    recorded it, or if a job the runner saw finish is no longer DONE;
  * a hand stop (SIGTERM / SIGHUP / SIGINT to the runner, e.g. 'systemctl --user stop', or a logout while linger is off)
    stops the running job and is NOT a failed attempt; the runner logs 'HANDSTOP word_A1 <job> ...' in final_prereg.log and
    will not start again until the reason has been logged with --hand-stop-reason "<text>". Killing a job process
    directly (not the runner) counts as a failed attempt;
  * no draw is replaced, re-drawn or added.
State: C_word/a1_run/<job>.json (attempts, failures, history, done_sha256); single instance (flock on
C_word/a1_run/runner.lock). Events: outputs/logs/vccor/final_word_a1_run.log.

usage:  .venv/bin/python -B scripts/vccor/final/word_a1/word_a1_run.py            (normally via the ssl-word-a1-run unit)
        .venv/bin/python -B scripts/vccor/final/word_a1/word_a1_run.py --check    (gates and job status; writes nothing)
        .venv/bin/python -B scripts/vccor/final/word_a1/word_a1_run.py --hand-stop-reason "<why the runner was stopped>"

Version of 2026-11-05 (A1, A1.1) and 2026-11-06 (A1.2).
"""
import argparse
import fcntl
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
import word_a1_common as A
import final_common as F

_CHILD = {"p": None}


class HandStop(SystemExit):
    """The runner received SIGTERM / SIGHUP / SIGINT (a hand stop)."""

    def __init__(self, signum):
        super().__init__(128 + signum)
        self.signum = signum


def _on_term(signum, frame):
    p = _CHILD["p"]
    if p is not None and p.poll() is None:
        try:
            os.killpg(p.pid, signal.SIGTERM)
            p.wait(timeout=120)
        except (OSError, subprocess.TimeoutExpired):
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except OSError:
                pass
    raise HandStop(signum)


# ---------------------------------------------------------------------------------------------------- state
def state_file(ctx, name):
    return ctx.state_dir / f"{name}.json"


def load_state(ctx, name):
    p = state_file(ctx, name)
    if p.is_file():
        return json.load(open(p))
    return {"job": name, "attempts": 0, "failures": 0, "history": []}


def save_state(ctx, name, st):
    F.write_json(state_file(ctx, name), st)


def is_done(job):
    return Path(job["done_file"]).is_file()


def job_arm(job):
    arms = A.job_arms(job)
    assert len(arms) == 1, job["name"]
    return arms[0]


def guard_not_done(job, st):
    """A1.1: the runner never re-runs a DONE job."""
    if is_done(job):
        raise A.Refuse(f"{job['name']} is DONE ({job['done_file']}): the runner never re-runs a DONE job (A1.1); "
                       f"re-running would rewrite its DONE record")
    ok = [h for h in st.get("history", []) if h.get("ok")]
    if ok:
        raise A.Refuse(f"{job['name']} finished OK at attempt {ok[-1].get('attempt')} but {job['done_file']} is missing: "
                       f"the runner never re-runs a DONE job (A1.1); restore the record or declare the arm NOT RUN")


# ---------------------------------------------------------------------------------------------------- hand stops
def hand_stops_pending(ctx):
    """[job, ...] of HANDSTOP word_A1 lines without a later HANDSTOP_REASON line for the same job (in log order)."""
    pend = []
    for p in ctx.log_lines():
        if len(p) >= 5 and p[2] == "HANDSTOP" and p[3] == A.NOTRUN_TAG:
            pend.append(p[4])
        elif len(p) >= 6 and p[2] == "HANDSTOP_REASON" and p[3] == A.NOTRUN_TAG and p[4] in pend:
            pend.remove(p[4])
    return pend


def log_hand_stop_reason(ctx, reason):
    reason = " ".join(str(reason).split())
    if not reason:
        raise A.Refuse("--hand-stop-reason needs a non-empty reason")
    pend = hand_stops_pending(ctx)
    if not pend:
        raise A.Refuse("no HANDSTOP line is waiting for a reason")
    ctx.plog(f"HANDSTOP_REASON {A.NOTRUN_TAG} {pend[0]} {reason}")
    return pend[0]


def record_hand_stop(ctx, job, st, phase, signum, t0, log):
    nm = signal.Signals(signum).name
    if phase == "running":
        st["history"].append({"attempt": st["attempts"], "rc": None, "ok": False, "hand_stopped": True,
                              "signal": nm, "start": datetime.fromtimestamp(t0).strftime("%Y-%m-%d %H:%M:%S"),
                              "wall_h": round((time.time() - t0) / 3600, 3)})
        save_state(ctx, job["name"], st)
        what = f"attempt {st['attempts']} stopped by {nm}; not counted as a failed attempt (A1.1)"
    else:
        what = f"runner stopped by {nm} while {phase}; no job was running"
    ctx.plog(f"HANDSTOP {A.NOTRUN_TAG} {job['name']} {what}; reason pending: word_a1_run.py --hand-stop-reason \"...\"")
    log(f"HANDSTOP {job['name']}: {what}")


# ---------------------------------------------------------------------------------------------------- gates
def static_gates(ctx, pids_fn=A.queue_v8_pids, check=False, log=print):
    """The start gates. check=False: raise A.Refuse on the first failure; check=True: collect every result."""
    rec, fails = {}, []

    def step(name, fn):
        try:
            rec[name] = {"ok": True, "detail": fn()}
        except A.Refuse as e:
            rec[name] = {"ok": False, "detail": e.msg}
            if not check:
                raise
            fails.append(name)

    def prep():
        miss = [n for n in A.PREP_DONE_Q8 if not (ctx.q8_dir / f"{n}.done").is_file()]
        if miss:
            raise A.Refuse(f"Plan C prep jobs not done (q8 markers): {miss}")
        return "pfinC_prep_front / capture / back done"

    def experiment():
        nr = A.notrun_declared(ctx)
        if "EXPERIMENT" in nr:
            raise A.Refuse(f"the experiment was declared NOT RUN ({nr['EXPERIMENT']})")
        return {"declared_not_run": nr}

    def hand_stops():
        pend = hand_stops_pending(ctx)
        if pend:
            raise A.Refuse(f"hand stop(s) of {pend} have no HANDSTOP_REASON line: log the reason with "
                           f"word_a1_run.py --hand-stop-reason \"...\" first (A1.1)")
        return "every hand stop has a reason"

    step("1_no_look", lambda: A.look_not_made(ctx))
    step("2_osf_gate", lambda: A.osf_gate(ctx)[1])
    step("3_integrity", lambda: A.integrity(ctx, HERE / "word_a1_run.py", log)[1])
    step("4_scheduler", lambda: A.scheduler_gate(ctx, pids_fn))
    step("5_kst", A.kst_gate)
    step("6_prep_done", prep)
    step("7_experiment_not_declared_not_run", experiment)
    step("8_hand_stops_have_reasons", hand_stops)
    step("9_a12_inputs", lambda: A.a12_inputs_gate(ctx))
    rec["all_ok"] = not fails
    rec["failed"] = fails
    return rec


def launch_gates(ctx, pids_fn):
    """Before every launch (before the lock and idle waits). Any refusal stops the runner without counting an attempt."""
    A.look_not_made(ctx)
    A.osf_gate(ctx)
    A.scheduler_gate(ctx, pids_fn)
    return A.disk_gate(ctx)


# ---------------------------------------------------------------------------------------------------- one job
def launch(ctx, job, timeout_s, log, lock_fd=None):
    """Run one job (a GPU job on GPU 1 inherits the locked GPU-1 descriptor; a CPU job sees no GPU); returns
    (rc, stopped_at_cutoff)."""
    env = dict(os.environ)
    gpu = job.get("gpu", True)
    env.update(A.GPU_ENV if gpu else A.CPU_ENV)
    ctx.job_log_dir.mkdir(parents=True, exist_ok=True)
    cmd = [job["interp"]] + list(job["args"])
    with open(ctx.job_log_dir / f"{job['name']}.log", "a") as logf:
        p = subprocess.Popen(cmd, cwd=str(job.get("cwd", A.REPO)), env=env, stdout=logf, stderr=subprocess.STDOUT,
                             start_new_session=True, pass_fds=(lock_fd,) if lock_fd is not None else ())
        _CHILD["p"] = p
        where = "GPU1, shared GPU-1 lock held" if gpu else "CPU, no GPU, no lock"
        log(f"START {job['name']} ({where}, pid {p.pid}, timeout {timeout_s / 3600:.2f} h = the cutoff): {' '.join(cmd)}")
        stopped = False
        try:
            rc = p.wait(timeout=max(timeout_s, 1.0))
        except subprocess.TimeoutExpired:
            stopped = True
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
            rc = p.returncode
        _CHILD["p"] = None
    return rc, stopped


def declare(ctx, arm, reason, log):
    if A.declare_not_run(ctx, arm, reason):
        log(f"NOTRUN {arm}: {reason}")
    if arm in A.REQUIRED:
        if A.declare_not_run(ctx, "EXPERIMENT", f"required arm {arm} NOT RUN; no look (A1 stopping_rule.required)"):
            log("NOTRUN EXPERIMENT: required arm not run")
        return True
    return False


def cutoff_reason(ctx, job, what):
    return f"not DONE by the cutoff {ctx.cutoff:%Y-%m-%d %H:%M} KST ({job['name']} {what})"


def too_late(ctx, job, now):
    """A1.2: a STOCK job whose registered maximum remaining chain run time would pass the cutoff is not launched."""
    m = job.get("est_max_h")
    return m is not None and (ctx.cutoff - now).total_seconds() < float(m) * 3600


def estimate_reason(ctx, job, when):
    return (f"not DONE by the cutoff {ctx.cutoff:%Y-%m-%d %H:%M} KST ({job['name']} not launched {when}: the registered "
            f"maximum run time of the rest of its STOCK chain, {job['est_max_h']} h, would pass the cutoff; A1.2 STOCK rule)")


# ---------------------------------------------------------------------------------------------------- the run
def run(ctx, jobs=None, pids_fn=A.queue_v8_pids, used_fn=A.nvidia_gpu1_used_mib, now_fn=datetime.now,
        wait_s=A.GPU_IDLE_WAIT_S, poll_s=A.GPU_POLL_S, lock_wait_s=A.GPU_LOCK_WAIT_S, lock_poll_s=A.GPU_LOCK_POLL_S,
        log=print):
    """Returns 0 (finished), 1 (experiment NOT RUN), 2 (gate refusal), 3 (resource refusal: rerun later)."""
    jobs = jobs if jobs is not None else A.run_jobs(ctx.pred_dir, ctx.models_dir)
    assert [j["name"] for j in jobs] == A.RUN_JOBS, "A1.2's run order"
    by_name = {j["name"]: j for j in jobs}
    static_gates(ctx, pids_fn, check=False, log=log)
    ctx.state_dir.mkdir(parents=True, exist_ok=True)
    fd = os.open(ctx.state_dir / "runner.lock", os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        raise A.Refuse("another word A1 runner holds the lock", code=3)
    cur = {"job": None, "phase": "starting", "st": None, "t0": time.time()}
    try:
        log(f"word A1 runner armed: {len(jobs)} jobs in A1.2 order; cutoff {ctx.cutoff}; shared GPU-1 lock {ctx.gpu_lock}")
        for job in jobs:
            arm = job_arm(job)
            gpu = job.get("gpu", True)
            nr = A.notrun_declared(ctx)
            if "EXPERIMENT" in nr:
                log("experiment declared NOT RUN: stopping")
                return 1
            if arm in nr:
                log(f"SKIP {job['name']}: {arm} declared NOT RUN ({nr[arm]})")
                continue
            st = load_state(ctx, job["name"])
            if is_done(job):
                h = F.sha256(job["done_file"])
                if st.get("done_sha256") and st["done_sha256"] != h:
                    raise A.Refuse(f"{job['name']}: its DONE record changed after the runner recorded it "
                                   f"({st['done_sha256'][:16]} -> {h[:16]}): a DONE job must not be re-run (A1.1)")
                log(f"SKIP {job['name']}: already DONE")
                continue
            guard_not_done(job, st)
            if st["failures"] >= A.MAX_ATTEMPTS:
                if declare(ctx, arm, f"{job['name']} failed twice", log):
                    return 1
                continue
            if job["name"] in A.A12_JOBS:
                dep_nr = [(d, job_arm(by_name[d])) for d in job["deps"] if job_arm(by_name[d]) in nr]
                if dep_nr:
                    d0, a0 = dep_nr[0]
                    declare(ctx, arm, f"its input arm {a0} was declared NOT RUN ({d0}; A1.2)", log)
                    continue
            bad = [d for d in job["deps"] if not is_done(by_name[d])]
            if bad:
                raise A.Refuse(f"{job['name']}: dependencies not DONE and not declared NOT RUN: {bad}")
            while True:
                cur.update(job=job, phase="checking the cutoff", st=st)
                now = now_fn()
                if now >= ctx.cutoff:
                    if declare(ctx, arm, cutoff_reason(ctx, job, "not finished"), log):
                        return 1
                    break
                if too_late(ctx, job, now):
                    declare(ctx, arm, estimate_reason(ctx, job, "before the waits"), log)
                    break
                free = launch_gates(ctx, pids_fn)
                lock_fd, used = None, None
                if gpu:
                    cur["phase"] = "waiting for the shared GPU-1 lock"
                    lock_fd = A.acquire_gpu_lock(ctx.gpu_lock, lock_wait_s, lock_poll_s, log)
                try:
                    if gpu:
                        cur["phase"] = "waiting for GPU 1 to be idle"
                        used = A.wait_gpu1_idle(used_fn, wait_s, poll_s, log)
                    now = now_fn()
                    if now >= ctx.cutoff:
                        log(f"cutoff {ctx.cutoff} passed during the lock / idle wait: {job['name']} NOT LAUNCHED")
                        if declare(ctx, arm, cutoff_reason(ctx, job, "not launched: the cutoff passed during the GPU "
                                                                    "lock / idle wait"), log):
                            return 1
                        break
                    if too_late(ctx, job, now):
                        declare(ctx, arm, estimate_reason(ctx, job, "after the GPU lock / idle wait"), log)
                        break
                    guard_not_done(job, st)
                    left = (ctx.cutoff - now_fn()).total_seconds()
                    st["attempts"] += 1
                    t0 = time.time()
                    cur.update(phase="running", t0=t0)
                    rc, stopped = launch(ctx, job, left, log, lock_fd)
                    cur["phase"] = "after the job"
                finally:
                    A.release_gpu_lock(lock_fd)
                ok = rc == 0 and is_done(job) and not stopped
                st["history"].append({"attempt": st["attempts"], "rc": rc, "ok": ok, "stopped_at_cutoff": stopped,
                                      "start": datetime.fromtimestamp(t0).strftime("%Y-%m-%d %H:%M:%S"),
                                      "wall_h": round((time.time() - t0) / 3600, 3), "free_gb_before": round(free, 1),
                                      "gpu1_mib_before": used, "gpu1_lock": str(ctx.gpu_lock) if gpu else None,
                                      "gpu": bool(gpu)})
                if ok:
                    st["done_sha256"] = F.sha256(job["done_file"])
                if not ok and not stopped:
                    st["failures"] += 1
                save_state(ctx, job["name"], st)
                log(f"END {job['name']} rc={rc} {'OK' if ok else ('STOPPED AT CUTOFF' if stopped else 'FAILED')} "
                    f"attempt {st['attempts']} failures {st['failures']}")
                if ok:
                    break
                if stopped:
                    if declare(ctx, arm, cutoff_reason(ctx, job, "stopped"), log):
                        return 1
                    break
                if st["failures"] >= A.MAX_ATTEMPTS:
                    if declare(ctx, arm, f"{job['name']} failed twice (rc {rc})", log):
                        return 1
                    break
                log(f"RERUN {job['name']} once, unchanged (A1 stopping_rule.failed_job)")
        nr = A.notrun_declared(ctx)
        log(f"word A1 runner finished: DONE {[j['name'] for j in jobs if is_done(j) and job_arm(j) not in nr]}; "
            f"NOT RUN {sorted(nr)}; next: the rehearsal (word_a1_rehearse.py), then the single look (word_a1_score.py)")
        return 0
    except HandStop as e:
        if cur["job"] is not None:
            record_hand_stop(ctx, cur["job"], cur["st"], "running" if cur["phase"] == "running" else cur["phase"],
                             e.signum, cur["t0"], log)
        raise
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def status(ctx):
    nr = A.notrun_declared(ctx)
    out = {}
    for j in A.run_jobs(ctx.pred_dir, ctx.models_dir):
        st = load_state(ctx, j["name"]) if state_file(ctx, j["name"]).is_file() else None
        out[j["name"]] = {"done": is_done(j), "arm_declared_not_run": nr.get(job_arm(j)), "gpu": bool(j.get("gpu", True)),
                          "attempts": st["attempts"] if st else 0, "failures": st["failures"] if st else 0}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="evaluate the start gates and show the jobs; write nothing")
    ap.add_argument("--hand-stop-reason", default=None, help="log the reason of the oldest hand stop without one")
    a = ap.parse_args(argv)
    ctx = A.Ctx.real_ctx()
    if a.hand_stop_reason is not None:
        try:
            job = log_hand_stop_reason(ctx, a.hand_stop_reason)
        except A.Refuse as e:
            print(f"[REFUSE] {e.msg}", file=sys.stderr, flush=True)
            return e.code
        print(f"HANDSTOP_REASON logged for {job}")
        return 0
    if a.check:
        rec = static_gates(ctx, check=True, log=lambda m: print(m, file=sys.stderr))
        try:
            rec["disk_free_gb"] = {"ok": True, "detail": A.disk_gate(ctx)}
        except A.Refuse as e:
            rec["disk_free_gb"] = {"ok": False, "detail": e.msg}
        rec["gpu1_lock_file"] = {"ok": ctx.gpu_lock.is_file(),
                                 "detail": f"{ctx.gpu_lock} {'exists' if ctx.gpu_lock.is_file() else 'MISSING'} "
                                           f"(not locked by --check; flocked before every GPU launch)"}
        rec["gpu1_idle"] = {"ok": None, "detail": "not queried in --check (checked before every GPU launch)"}
        rec["jobs"] = status(ctx)
        print(json.dumps(rec, indent=1, default=str))
        return 0 if rec["all_ok"] and rec["disk_free_gb"]["ok"] and rec["gpu1_lock_file"]["ok"] else 2
    try:
        A.look_not_made(ctx)
        A.osf_gate(ctx)
    except A.Refuse as e:
        print(f"[REFUSE] {e.msg}", file=sys.stderr, flush=True)
        return e.code
    for s in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(s, _on_term)
    log = F.Log(ctx.event_log)
    try:
        return run(ctx, log=log)
    except A.Refuse as e:
        log(f"[REFUSE] {e.msg}")
        return e.code


if __name__ == "__main__":
    sys.exit(main())
