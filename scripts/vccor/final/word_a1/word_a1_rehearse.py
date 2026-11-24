"""vccor/final plan C, amendment word A1.1: REHEARSAL of the single WORD look on real-size volumes, with NO label.

Run it after word_a1_run.py has finished and before the look. It scores the real prediction maps of every DONE arm (A1's TS
arms and A1.2's STOCK, MERGE_N and LABELMAP arms, each with its own label map) against a STAND-IN reference built from F0's
own predictions (F0's raw map through the TS label map, shifted by one voxel along the
first axis so that F0's zero-shot error is not 0), in a scratch look directory under /tmp, with the real worker count and the scorer's own code path
(word_a1_score.pre-marker steps, score_all, finish): the duplicate rule on the real images, the maps manifest, a scratch
GT_OPENED marker and log, the worker pool with its per-result timeout, the per-scan records, the analysis, results.json, and
a re-analysis from the per-scan records (the --continue path). It exists to catch out-of-memory, run-time, path and JSON
problems before the look. Its numbers mean nothing about WORD and are never reported as results. As in the look, an error
in the A1.2 secondary analysis does not stop it (A1.2: a secondary-analysis error never blocks the primary results); the
report names it in 'a12_secondary_error' (null when the secondary analysis ran).

It never touches the real label archives or the real look directory, and refuses if either would be touched:
  * --scratch must resolve under /tmp/, must not be inside the project, and must be new or empty;
  * every path it writes (scratch look, scratch log, stand-in zip path, state) is checked to be under the scratch folder and
    outside outputs/results/vccor/final/C_word/look and MagicNet/data/word_v010/archive{,_labelsTs};
  * word_common.LOOK / LABELS_TS_ZIP / reference are pointed at the scratch folder and the stand-in for the run;
  * an audit hook (inherited by the worker processes) raises PermissionError on ANY open / listdir / scandir / mkdir /
    rename / remove of a path inside the real label archive folders or the real look directory, and on any write-mode open
    inside the project. Reading the prediction maps, the images, prep/overlap.json and download.json is allowed.
It writes nothing in the project and nothing in final_prereg.log; its report is <scratch>/rehearsal_report.json.

usage:  taskset -c 8-15 nice -n 15 env CUDA_VISIBLE_DEVICES= .venv/bin/python -B \\
            scripts/vccor/final/word_a1/word_a1_rehearse.py --scratch /tmp/word_a1_rehearsal --workers 6

Version of 2026-11-05 (A1, A1.1) and 2026-11-06 (A1.2 and its
v5 revision).
"""
import argparse
import contextlib
import json
import os
import resource
import sys
import time
import traceback
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import word_a1_common as A
import word_a1_score as S
import final_common as F
import fin_look as L
import word_common as W
import word_score as WS

PROTECTED = [os.path.normpath(str(p)) for p in A.REAL_LABEL_PATHS + [A.REAL_LOOK]]
REPO_STR = os.path.normpath(str(A.REPO))
_GUARD = {"on": False, "protected": list(PROTECTED), "no_write_root": REPO_STR}


def _inside(path, roots):
    p = os.path.normpath(os.path.abspath(path))
    return any(p == r or p.startswith(r + os.sep) for r in roots)


def _is_write(mode, flags):
    if isinstance(mode, str) and any(ch in mode for ch in "wax+"):
        return True
    if isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
        return True
    return False


def _audit(event, args):
    if not _GUARD["on"]:
        return
    if event == "open":
        path, mode, flags = (list(args) + [None, None, None])[:3]
        if isinstance(path, int):
            return
        path = os.fsdecode(path)
        if _inside(path, _GUARD["protected"]):
            raise PermissionError(f"word A1 rehearsal: refused to open a protected path {path}")
        if _is_write(mode, flags) and _inside(path, [_GUARD["no_write_root"]]):
            raise PermissionError(f"word A1 rehearsal: refused to write inside the project: {path}")
    elif event in ("os.listdir", "os.scandir", "os.mkdir", "os.remove", "os.rmdir", "os.rename", "os.chmod",
                   "shutil.rmtree", "os.truncate"):
        for a in args:
            if isinstance(a, (str, bytes, os.PathLike)):
                p = os.fsdecode(a)
                if _inside(p, _GUARD["protected"]):
                    raise PermissionError(f"word A1 rehearsal: refused {event} on a protected path {p}")
                if event != "os.listdir" and event != "os.scandir" and _inside(p, [_GUARD["no_write_root"]]):
                    raise PermissionError(f"word A1 rehearsal: refused {event} inside the project: {p}")


def install_guard(protected=None, no_write_root=None):
    """Arm the audit hook (once per process; inherited by forked workers). Tests may pass scratch stand-ins."""
    if protected is not None:
        _GUARD["protected"] = [os.path.normpath(str(p)) for p in protected]
    if no_write_root is not None:
        _GUARD["no_write_root"] = os.path.normpath(str(no_write_root))
    if not _GUARD.get("hooked"):
        sys.addaudithook(_audit)
        _GUARD["hooked"] = True
    _GUARD["on"] = True


def check_scratch(scratch):
    """Refuse unless scratch is a new or empty folder under /tmp, outside the project and the protected paths."""
    s = os.path.normpath(os.path.abspath(str(scratch)))
    if not s.startswith("/tmp/"):
        raise A.Refuse(f"rehearsal: --scratch must be under /tmp/: {s}")
    if _inside(s, PROTECTED) or _inside(s, [REPO_STR]):
        raise A.Refuse(f"rehearsal: --scratch would touch the project, the label archives or the real look: {s}")
    rs = str(Path(s).resolve())
    if not rs.startswith("/tmp/") or A.touches_real_labels_or_look(rs):
        raise A.Refuse(f"rehearsal: --scratch resolves outside /tmp or into a protected path: {rs}")
    if Path(rs).exists() and any(Path(rs).iterdir()):
        raise A.Refuse(f"rehearsal: --scratch {rs} exists and is not empty")
    return Path(rs)


def rehearsal_ctx(scratch, pred_dir):
    root = Path(scratch)
    for p in (root / "C_word/look", root / "no_labels/WORD_V0.1.0_labelsTs.zip", root / "logs/final_prereg.log"):
        if not A.under_tmp(p) or _inside(str(p), PROTECTED) or A.touches_real_labels_or_look(p):
            raise A.Refuse(f"rehearsal: path {p} would touch a real label archive or the real look")
    if _inside(str(pred_dir), PROTECTED):
        raise A.Refuse(f"rehearsal: pred dir {pred_dir} is a protected path")
    return A.Ctx(False, prereg_log=root / "logs/final_prereg.log", osf_file=root / "C_word" / A.OSF_FILE_NAME,
                 look_dir=root / "C_word/look", pred_dir=pred_dir, models_dir=root / "models",
                 state_dir=root / "C_word/a1_run", event_log=root / "logs/final_word_a1_run.log",
                 score_log=root / "logs/final_word_a1_score.log", job_log_dir=root / "logs/jobs",
                 decisions=root / "q8/decisions.json", q8_dir=root / "q8", deposit_rel="rehearsal (no deposit)",
                 labels_zip=root / "no_labels/WORD_V0.1.0_labelsTs.zip", labels_zip_bytes=None,
                 labels_zip_sha256=None, gpu_lock=root / "gpu1.lock", mode="rehearsal")


def stand_in_reference(f0_raw_dir):
    """reference(cid, marker) replacement: F0's raw map in the analysis label space, shifted by one voxel along the first
    axis (np.roll), with F0's spacing. No label is read. The shift keeps F0's zero-shot error above 0: with F0 itself as
    the reference, E0 = 0 and the frozen word_score.boot_ratio (S) fails on an all-NaN bootstrap, which real labels never
    give."""
    import numpy as np
    f0 = Path(f0_raw_dir)

    def ref(cid, marker):
        F.require_marker(marker)
        seg, im = L.read_raw(f0 / f"{cid}.nii.gz")
        return np.roll(L.lut("word", "ts")[seg], 1, axis=0), tuple(float(s) for s in im.header.get_zooms()[:3])

    return ref


@contextlib.contextmanager
def stand_in_word(ctx):
    keep = {k: getattr(W, k) for k in ("LOOK", "LABELS_TS_ZIP", "reference")}
    try:
        W.LOOK = ctx.look_dir
        W.LABELS_TS_ZIP = ctx.labels_zip
        W.reference = stand_in_reference(A.arm_dirs(ctx.pred_dir)["F0"] / "raw")
        yield
    finally:
        for k, v in keep.items():
            setattr(W, k, v)


def rehearse(scratch, pred_dir=None, workers=6, result_timeout_s=S.RESULT_TIMEOUT_S, log=print):
    """The rehearsal; returns its report (also written to <scratch>/rehearsal_report.json)."""
    root = check_scratch(scratch)
    pred_dir = Path(pred_dir) if pred_dir else W.DIR / "pred"
    ctx = rehearsal_ctx(root, pred_dir)
    t_all = time.time()
    with stand_in_word(ctx):
        ctx.check_word_paths()
        dirs = A.arm_dirs(ctx.pred_dir)
        if not (dirs["F0"] / "DONE.json").is_file():
            raise A.Refuse("rehearsal: F0 is not DONE (run it after the runner has finished)")
        have = [a_ for a_ in A.ALL_ARMS if (dirs[a_] / "DONE.json").is_file()]
        if "F1b_d1" not in have:
            log("rehearsal: F1b_d1 is not DONE: the analysis step is skipped (the look itself would refuse)")
        ids = list(W.split()["test"])
        dup, dropped = WS.duplicates(ids, log)
        ids = [c for c in ids if c not in dropped]
        files = {}
        for arm in have:
            bad = [c for c in ids if L.done_raw_sha(dirs[arm] / "DONE.json", c)
                   != F.sha256(dirs[arm] / "raw" / f"{c}.nii.gz")]
            if bad:
                raise A.Refuse(f"rehearsal: arm {arm}: maps do not match their DONE record ({bad[:3]})")
            files[arm] = {c: str(dirs[arm] / "raw" / f"{c}.nii.gz") for c in ids}
        shas = L.maps_manifest(files)
        S.a12_inputs_consistent(have, dirs)
        lut = {a_: A.ARM_LUT[a_] for a_ in have}
        done_sha = {a_: F.sha256(dirs[a_] / "DONE.json") for a_ in have}
        rules_sha = {a_: F.sha256(dirs[a_] / "rules.json") for a_ in have if a_ in A.LABELMAP_ARMS}
        extras = S.a12_extras(have, dirs)
        ctx.look_dir.mkdir(parents=True, exist_ok=True)
        mjf = ctx.look_dir / "maps.json"
        F.write_json(mjf, {"sha256": shas, "arms": have, "lut": lut, "done_sha256": done_sha, "rules_sha256": rules_sha,
                           "not_run": {}, "ids": ids, "dropped_as_duplicates": dropped, "duplicates_check": dup,
                           "scorer": "word A1 REHEARSAL (stand-in reference)", "written": F.now()})
        st = {"grec": {"rehearsal": True}, "url": "rehearsal (no OSF gate)", "pinned": None, "ids": ids, "dup": dup,
              "dropped": dropped, "have": have, "notes": {}, "files": files, "shas": shas, "lut": lut,
              "done_sha": done_sha, "extras": extras, "mj": F.sha256(mjf)}
        mk = F.open_marker(ctx.look_dir / "GT_OPENED", {"rehearsal": True,
                                                        "stand_in_reference": "F0 predictions, shifted one voxel",
                                                        "ids": ids, "maps_sha256": st["mj"], "arms": have,
                                                        "osf_registration": st["url"]})
        ctx.plog(f"GT_OPENED sha256 {F.sha256(mk)} {ctx.rel(mk)} (maps.json sha256 {st['mj']}; REHEARSAL, stand-in "
                 f"reference = F0 predictions shifted one voxel; no label)")
        R, D, sc = S.score_all(ctx, st, mk, workers, result_timeout_s, False, log)
        rss_kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        rep = {"rehearsal": "word A1.1 pre-look rehearsal (stand-in reference = F0's predictions shifted by one voxel; "
                            "no label read)",
               "written": F.now(), "scratch": str(root), "pred_dir": str(pred_dir), "arms": have, "n_scans": len(ids),
               "dropped_as_duplicates": dropped, "workers": workers, "result_timeout_s": result_timeout_s,
               "scoring_wall_s": sc["wall_s"], "mean_s_per_scan_per_worker": round(sc["wall_s"] * workers / max(len(ids), 1),
                                                                                    1),
               "peak_child_rss_gb": round(rss_kb / 1e6, 2), "peak_self_rss_gb":
                   round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2),
               "pervol_done_records": len(list((ctx.look_dir / "pervol_done").glob("*.json"))),
               "a12_secondary_error": None}
        if "F1b_d1" in have:
            res = S.finish(ctx, st, R, D, "rehearsal (no zip)", sc, None, log)
            back = json.loads((ctx.look_dir / "results.json").read_text())
            R2, D2 = {}, {}
            for c in ids:
                ok, why, pay = S.check_pervol(ctx, c, have, shas, F.sha256(mk), st["mj"])
                if not ok:
                    raise A.Refuse(f"rehearsal: per-scan record {c} fails its integrity check: {why}")
                R2[c], D2[c] = pay["frames"], pay["dec"]
            res2 = S.analyse_all(R2, ids, set(have), D2, not_run={}, extras=extras)
            a = json.dumps(F.clean_nan({k: res[k] for k in res2}), sort_keys=True, default=F._default)
            b = json.dumps(F.clean_nan(res2), sort_keys=True, default=F._default)
            rep.update(results_json_bytes=(ctx.look_dir / "results.json").stat().st_size,
                       results_json_reparsed=bool(back.get("outcome")), continuation_reanalysis_equal=a == b,
                       stand_in_outcome_meaningless=res["outcome"], a12_secondary_error=S.secondary_error(res))
            if a != b:
                raise A.Refuse("rehearsal: re-analysis from the per-scan records differs from the look's analysis")
        rep["total_wall_s"] = round(time.time() - t_all, 1)
        F.write_json(root / "rehearsal_report.json", rep)
    log(f"rehearsal done: {json.dumps(rep, default=str)}")
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--result-timeout", type=float, default=S.RESULT_TIMEOUT_S)
    ap.add_argument("--pred-dir", default=None, help="tests only: a synthetic prediction folder under /tmp")
    a = ap.parse_args(argv)
    assert 1 <= a.workers <= F.MAX_WORKERS
    try:
        check_scratch(a.scratch)
        if a.pred_dir is not None and not A.under_tmp(a.pred_dir):
            raise A.Refuse("rehearsal: --pred-dir is for tests and must be under /tmp")
    except A.Refuse as e:
        print(f"[REFUSE] {e.msg}", file=sys.stderr, flush=True)
        return e.code
    install_guard()
    F.be_nice(15)
    try:
        rehearse(a.scratch, a.pred_dir, a.workers, a.result_timeout,
                 log=lambda m: print(f"[{F.now()}] {m}", file=sys.stderr, flush=True))
    except A.Refuse as e:
        print(f"[REFUSE] {e.msg}", file=sys.stderr, flush=True)
        return e.code
    except BaseException as e:
        try:
            F.write_json(Path(a.scratch).resolve() / "rehearsal_FAILED.json",
                         {"failed": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-4000:],
                          "written": F.now(), "meaning": "fix before the look; the look must not start"})
        except BaseException:
            pass
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
