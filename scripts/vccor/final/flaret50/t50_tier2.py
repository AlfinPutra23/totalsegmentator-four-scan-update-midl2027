"""vccor/final FLARE-T50 TIER 2 runner (GPU 1): FULL K = 1 retrained on the four fair-fight draws d1_0014, d1_0001,
d1_0006, d1_0038, in this order, with the FROZEN fair-fight code through scripts/vccor/moonshot/ff_launch.py (byte-identical
to the fair-fight CODEFREEZE of 2026-10-16, asserted by the launcher; recipe F1b: all weights, lr 1e-4, 50 x 250 it, seed
1337; datasets prepared and verified in outputs/results/vccor/moonshot/ff/ on 2026-11-03). Exports land in
outputs/results/vccor/moonshot/ff/models/FULL/<draw>/. Training touches no FLARE-T50 data: each draw reads only its own
labelled FLARE22 training scan, already used by the fair fight of 2026-10-17. The moonshot of 2026-11-03 stopped at its
futility gate and did not train these models; they are trained now only for FLARE-T50 tier 2 (AC.md GO-1).

Rules (AC.md GO-1 tier 2, with the brief of 2026-11-09):
  * kill rule: if the first training (d1_0014) has not finished within 3 h of wall time (all its attempts together, an
    interrupted attempt included), K = 1 is NOT RUN: the runner stops it, writes tier2/NOT_RUN.json and a
    'NOTRUN flaret50 K1 ...' line, and starts nothing more;
  * a failed attempt (non-zero exit) is rerun once, unchanged (train_ff.py resumes from its checkpoint); a second failure
    declares that draw NOT RUN ('NOTRUN flaret50 K1_<draw> ...');
  * cutoff: nothing is launched unless the attempt's registered maximum (3.0 h) ends at least 1 h before Fri 2026-11-13
    18:00 KST (the tier-1 predictions of the K = 1 arms still need GPU time before the cutoff);
  * before every launch: KST; / keeps >= 35 GiB free after a 2 GiB allowance; an exclusive flock on the shared GPU-1 lock
    (/home/user/Documents/Handpose_project/runs/method_pilot3/gpu1.lock, waits up to 12 h), held by this runner until the
    attempt's process group has exited; GPU 1 idle (memory.used < 3000 MiB; waits up to 12 h);
  * GPU 1 only (CUDA_DEVICE_ORDER=PCI_BUS_ID, CUDA_VISIBLE_DEVICES=1, PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True);
  * a DONE draw (models/FULL/<draw>/DONE.json) is never re-run;
  * wall-time accounting (code review SF4, revision of 2026-11-09): an attempt's elapsed time is charged in a finally
    block, so a hand stop or an exception also counts; at start-up every attempt without an end (a logout, reboot or kill
    of the runner) is charged from its start to its log's last write and marked 'interrupted';
  * a hand stop (SIGTERM / SIGHUP / SIGINT, e.g. 'systemctl --user stop ssl-flaret50-tier2') stops the attempt's process
    group and logs 'HANDSTOP flaret50 ...'; it is not a failed attempt.
DISCLOSURE: the runner started at 11:09:08 KST on 2026-11-09 runs the version of this file and of t50_common.py hashed at
11:08:59 (tier2/code_tier2.json; t50_common as tier2/t50_common_a75b24a5.py); that version charged wall time only after a
normal exit. The accounting above applies from the next start.
State: outputs/results/vccor/final/I_flaret50/tier2/state.json; events: outputs/logs/vccor/final_flaret50_tier2.log;
attempt logs: tier2/logs/.

usage:  .venv/bin/python -I -B scripts/vccor/final/flaret50/t50_tier2.py           (normally via the ssl-flaret50-tier2 unit)
        .venv/bin/python -I -B scripts/vccor/final/flaret50/t50_tier2.py --check   (gates and status; writes nothing)

Version of 2026-11-09.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import t50_common as T  # noqa: E402

LAUNCHER = T.REPO / "scripts/vccor/moonshot/ff_launch.py"
FIRST_BUDGET_S = 3 * 3600
ATTEMPT_MAX_S = 3 * 3600
CUTOFF_RESERVE = timedelta(hours=1)
DISK_ALLOWANCE_GIB = 2.0
MAX_ATTEMPTS = 2
_CHILD = {"p": None}


def state_path():
    return T.TIER2 / "state.json"


def not_run_path():
    return T.TIER2 / "NOT_RUN.json"


def evlog():
    return T.F.LOGD / "final_flaret50_tier2.log" if not T.REHEARSAL else T.TIER2 / "tier2.log"


class HandStop(SystemExit):
    def __init__(self, signum):
        super().__init__(128 + signum)
        self.signum = signum


def log(msg):
    T.log_event(msg, evlog())


def _kill_group(p, grace=120):
    if p is None or p.poll() is not None:
        return
    try:
        os.killpg(p.pid, signal.SIGTERM)
        p.wait(timeout=grace)
    except (OSError, subprocess.TimeoutExpired):
        try:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait(timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            pass


def _on_signal(signum, frame):
    _kill_group(_CHILD["p"])
    raise HandStop(signum)


def load_state():
    if state_path().is_file():
        return json.load(open(state_path()))
    return {"draws": {d: {"attempts": 0, "failures": 0, "wall_s": 0.0, "history": []} for d in T.TIER2_DRAWS}}


def save_state(st):
    T.write_json(state_path(), st)


def done(draw):
    return (T.FF_MODELS / draw / "DONE.json").is_file()


def not_run_all():
    return not_run_path().is_file() and json.load(open(not_run_path())).get("scope") == "K1"


def declared_not_run(st, draw):
    return bool(st["draws"][draw].get("not_run"))


def declare_not_run(st, draw, reason, all_k1=False):
    scope = "K1" if all_k1 else f"K1_{draw[3:]}"
    st["draws"][draw]["not_run"] = reason
    save_state(st)
    if all_k1:
        T.write_json(not_run_path(), {"scope": "K1", "reason": reason, "written": T.now()})
    T.prereg_note("NOTRUN", f"{scope} {reason}")
    log(f"NOT RUN {scope}: {reason}")


def charge_interrupted(st):
    """SF4: every attempt without an end (runner killed, logout, reboot) is charged from its start to its log's last write."""
    changed = False
    for d, ds in st["draws"].items():
        for h in ds["history"]:
            if "end" in h:
                continue
            start = datetime.strptime(h["start"], "%Y-%m-%d %H:%M:%S")
            lp = T.REPO / h["log"] if not Path(h["log"]).is_absolute() else Path(h["log"])
            last = datetime.fromtimestamp(lp.stat().st_mtime) if lp.is_file() else start
            el = max(0.0, (last - start).total_seconds())
            h.update(end=last.strftime("%Y-%m-%d %H:%M:%S"), wall_s=round(el, 1), interrupted=True)
            ds["wall_s"] += el
            changed = True
            log(f"{d} attempt {h['attempt']}: no end recorded; charged {el / 60:.1f} min (interrupted)")
    if changed:
        save_state(st)


def launch_ok_by_cutoff(max_s):
    return datetime.now() + timedelta(seconds=max_s) + CUTOFF_RESERVE <= T.CUTOFF


def gates(check=False):
    T.kst_gate()
    if not LAUNCHER.is_file():
        raise T.Refuse(f"{T.rel(LAUNCHER)} missing")
    r = subprocess.run([str(T.VENV), "-I", "-B", str(LAUNCHER), "--check"], cwd=str(T.REPO), capture_output=True,
                       text=True, timeout=300, env=dict(os.environ, **T.CPU_ENV))
    if r.returncode != 0 or "TRAIN_RUNS" not in r.stdout:
        raise T.Refuse(f"ff_launch.py --check failed (rc {r.returncode}): {r.stderr.strip()[-400:]}")
    for d in T.TIER2_DRAWS:
        if f"('FULL', '{d}')" not in r.stdout:
            raise T.Refuse(f"ff_launch.py does not list ('FULL', '{d}') as a training run")
    return True


def train_cmd(draw):
    return [str(T.VENV), "-I", "-B", str(LAUNCHER), "train_ff.py", "--recipe", "FULL", "--draw", draw, "--device", "cuda"]


def run_attempt(st, draw, budget_s):
    """One attempt under the lock; returns 'done' / 'failed' / 'timeout' / 'cutoff'. Its elapsed time is charged in the
    finally block whatever ends it (SF4)."""
    T.disk_gate(DISK_ALLOWANCE_GIB, f"tier-2 training {draw}")
    fd = T.acquire_gpu_lock(log=log)
    p, t0, rec = None, None, None
    try:
        T.wait_gpu1_idle(log=log)
        if not launch_ok_by_cutoff(min(budget_s, ATTEMPT_MAX_S)):
            return "cutoff"
        T.disk_gate(DISK_ALLOWANCE_GIB, f"tier-2 training {draw}")
        ds = st["draws"][draw]
        ds["attempts"] += 1
        n = ds["attempts"]
        logp = T.TIER2 / "logs" / f"train_{draw}_attempt{n}.log"
        logp.parent.mkdir(parents=True, exist_ok=True)
        cmd = train_cmd(draw)
        env = dict(os.environ, **T.GPU_ENV, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
        rec = {"attempt": n, "start": T.now(), "cmd": cmd, "log": T.rel(logp)}
        ds["history"].append(rec)
        save_state(st)
        log(f"START {draw} attempt {n}: budget {budget_s / 3600:.2f} h; log {T.rel(logp)}")
        t0 = time.time()
        with open(logp, "a") as lf:
            p = subprocess.Popen(cmd, cwd=str(T.REPO), env=env, stdout=lf, stderr=subprocess.STDOUT,
                                 start_new_session=True)
            _CHILD["p"] = p
            timed_out = False
            while p.poll() is None:
                time.sleep(10)
                if time.time() - t0 > budget_s:
                    timed_out = True
                    log(f"{draw} attempt {n}: budget of {budget_s / 3600:.2f} h reached; stopping it")
                    _kill_group(p)
                    break
            rc = p.wait()
            _CHILD["p"] = None
        rec.update(rc=rc, timed_out=timed_out)
        if timed_out:
            return "timeout"
        if rc == 0 and done(draw):
            rec["done_sha256"] = T.sha256(T.FF_MODELS / draw / "DONE.json")
            log(f"DONE {draw} attempt {n} in {(time.time() - t0) / 3600:.2f} h")
            return "done"
        ds["failures"] += 1
        log(f"FAILED {draw} attempt {n}: rc {rc}, {(time.time() - t0) / 60:.1f} min")
        return "failed"
    finally:
        _kill_group(p)
        if t0 is not None and rec is not None and "end" not in rec:
            wall = time.time() - t0
            rec.update(end=T.now(), wall_s=round(wall, 1))
            if "rc" not in rec:
                rec["interrupted"] = True
            st["draws"][draw]["wall_s"] += wall
            save_state(st)
        T.release_gpu_lock(fd)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    gates(check=args.check)
    st = load_state()
    if args.check:
        for d in T.TIER2_DRAWS:
            s = st["draws"][d]
            print(f"{d}: done={done(d)} attempts={s['attempts']} failures={s['failures']} wall_h={s['wall_s'] / 3600:.2f} "
                  f"not_run={s.get('not_run')}")
        print(f"K1 NOT RUN: {not_run_all()}; free {T.disk_free_gib():.2f} GiB; cutoff {T.CUTOFF}")
        return 0
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, _on_signal)
    T.TIER2.mkdir(parents=True, exist_ok=True)
    log(f"tier-2 runner start pid {os.getpid()}")
    charge_interrupted(st)
    try:
        if not_run_all():
            log("K1 is NOT RUN (kill rule or cutoff); nothing to do")
            return 0
        for i, d in enumerate(T.TIER2_DRAWS):
            if done(d):
                log(f"{d}: DONE, skipped")
                continue
            if declared_not_run(st, d):
                log(f"{d}: declared NOT RUN, skipped")
                continue
            first = i == 0
            while not done(d):
                ds = st["draws"][d]
                if first:
                    budget = FIRST_BUDGET_S - ds["wall_s"]
                    if budget <= 0:
                        declare_not_run(st, d, "the first training (d1_0014) did not finish within 3 h (kill rule)",
                                        all_k1=True)
                        return 0
                else:
                    budget = ATTEMPT_MAX_S
                if ds["failures"] >= MAX_ATTEMPTS:
                    declare_not_run(st, d, f"{MAX_ATTEMPTS} failed attempts", all_k1=first)
                    if first:
                        return 0
                    break
                res = run_attempt(st, d, budget)
                if res == "cutoff":
                    for e in T.TIER2_DRAWS[i:]:
                        if not done(e) and not declared_not_run(st, e):
                            declare_not_run(st, e, "a training could not finish before the cutoff (Fri 2026-11-13 "
                                                   "18:00 KST, minus 1 h for its predictions)", all_k1=(e == T.TIER2_DRAWS[0]))
                    return 0
                if res == "timeout":
                    if first:
                        declare_not_run(st, d, "the first training (d1_0014) did not finish within 3 h (kill rule)",
                                        all_k1=True)
                        return 0
                    ds["failures"] += 1
                    save_state(st)
                if res == "failed" and ds["failures"] >= MAX_ATTEMPTS:
                    declare_not_run(st, d, f"{MAX_ATTEMPTS} failed attempts", all_k1=first)
                    if first:
                        return 0
                    break
        log("tier-2 runner finished: " + ", ".join(f"{d} {'DONE' if done(d) else 'NOT RUN'}" for d in T.TIER2_DRAWS))
        return 0
    except HandStop as e:
        T.prereg_note("HANDSTOP", f"tier-2 runner stopped by signal {e.signum}; attempt process group stopped")
        log(f"HANDSTOP signal {e.signum}")
        raise
    except T.Refuse as e:
        log(f"REFUSE: {e.msg}")
        return e.code


if __name__ == "__main__":
    sys.exit(main())
