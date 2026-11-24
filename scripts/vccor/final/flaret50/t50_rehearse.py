"""vccor/final FLARE-T50 LOOK REHEARSAL (CPU; synthetic stand-in data only; code review B2).

Runs the real t50_score.look() end to end on a synthetic stand-in world: 8 scans (one excluded), all 18 arms with maps
(K1_0038 declared NOT RUN), a synthetic Hub manifest that pins the stand-in labels' pointer oids, a synthetic freeze (fake
checkpoint files, TotalSegmentator weights and versions), a synthetic deposit line and synthetic upload / public gates.
Every study path is redirected with t50_common.use_root(); labels come from a local stand-in folder (t50_common.LABEL_SOURCE)
instead of the Hub. Each scenario runs in its own child process (python -I) with an audit hook that blocks any write outside
its scratch root and any read of the project's data folders, the real FLARE-T50 folder or outputs/.

Scenarios and the outcome each must show:
  ok                         GT_OPENED -> LABELS -> RESULTS, two panels
  crash_scoring              a worker exception -> GT_CRASHED; --continue -> RESULTS (finished scans reused)
  crash_before_labels_json   a stop between the archive and labels.json -> GT_CRASHED; --continue repairs labels.json -> RESULTS
  label_swapped_after_archive a crash after the archive, then a label file altered -> --continue restores it from the archive
                             (GT_NOTE) -> RESULTS identical to 'ok'
  corrupt_label              a label that matches its pin but cannot be read -> GT_ABORTED; --continue refuses
  float_label_non_integer    a float label holding 12.99995 -> GT_ABORTED
  float_label_exact          a float label holding exact integers -> RESULTS
  pointer_mismatch           a downloaded label that does not match its pinned pointer oid -> GT_ABORTED
  sigint                     SIGINT to the look during scoring -> GT_CRASHED; --continue -> RESULTS
  cutoff_done                an arm whose DONE.json was written at the cutoff -> that arm NOT RUN; RESULTS
  second_look                a second look after RESULTS -> refused
Result: I_flaret50/rehearsal.json (all_pass, every scenario, the code's sha256) and a REHEARSAL line in final_prereg.log.
After the freeze, rehearsal.json is a frozen record and is never rewritten: a later run writes a new read-only
rehearsal_postfreeze_<n>.json with its own REHEARSAL line instead (diff check F1).

usage:  CUDA_VISIBLE_DEVICES="" nice -n 15 taskset -c 16-23 .venv/bin/python -I -B scripts/vccor/final/flaret50/t50_rehearse.py
        (optional --root DIR for the scratch worlds; default a new temporary folder, removed afterwards)

Version of 2026-11-09 (code review fix B2).
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

SCENARIOS = ["ok", "crash_scoring", "crash_before_labels_json", "label_swapped_after_archive", "corrupt_label",
             "float_label_non_integer", "float_label_exact", "pointer_mismatch", "sigint", "cutoff_done", "second_look"]
N = 8
EXCL = "FLARETs_0006"
SHRINK = {"F0": 4, "FULL_s1337": 0, "FULL_s2025": 0, "FULL_s3407": 1, "K16_s1337": 0, "K16_s2025": 0, "K16_s3407": 0,
          "SSFT_B": 3, "SSFT_D": 1, "STOCK": 5, "SCRATCH_NM1000": 8, "BORROW_amos": 3, "BORROW_mact": 3,
          "BORROW_word": 4, "K1_0014": 2, "K1_0001": 2, "K1_0006": 3}           # K1_0038 declared NOT RUN


# ---------------------------------------------------------------------------------------------------- child side
def guard(root):
    project = "/home/user/Documents/Medical Imagining SSL"
    blocked = (project + "/data/", project + "/data_flaret50", project + "/MagicNet/data", project + "/outputs",
               "/media/user")
    allowed = (str(root),)

    def _p(a):
        try:
            return os.path.abspath(os.fsdecode(a))
        except TypeError:
            return ""

    def hook(event, args):
        if event == "open":
            path, mode, flags = args
            if path is None or isinstance(path, int):
                return
            p = _p(path)
            w = (mode is not None and any(ch in str(mode) for ch in "wax+")) or \
                (flags is not None and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)))
            if w and not p.startswith(allowed) and not p.startswith(("/dev/", "/proc/")):
                raise PermissionError(f"REHEARSAL GUARD: write blocked: {p}")
            if p.startswith(blocked):
                raise PermissionError(f"REHEARSAL GUARD: read blocked: {p}")
        elif event in ("os.rename", "os.remove", "os.rmdir", "os.mkdir", "os.chmod", "shutil.rmtree", "shutil.move",
                       "os.symlink", "os.link", "os.truncate"):
            for a in args:
                if isinstance(a, (str, bytes, os.PathLike)):
                    p = _p(a)
                    if not p.startswith(allowed) and not p.startswith(("/dev/", "/proc/")):
                        raise PermissionError(f"REHEARSAL GUARD: {event} blocked: {p}")
    sys.addaudithook(hook)


def synth(rng, np):
    G = np.zeros((40, 40, 30), np.uint8)
    boxes = {}
    for o in range(1, 14):
        x, y = (o - 1) % 4 * 9 + 2, (o - 1) // 4 * 9 + 2
        boxes[o] = (slice(x, x + 7), slice(y, y + 7), slice(5, 25))
        G[boxes[o]] = o
    return G, boxes


def build_world(T, SC, scen):
    import nibabel as nib
    import numpy as np
    import final_common as F
    ids = [f"FLARETs_{i:04d}" for i in range(1, N + 1)]
    T.IDS[:] = ids
    rng = np.random.default_rng(3)
    aff = np.diag([1.5, 1.5, 3.0, 1.0])
    lut_inv = {v: k for k, v in enumerate(F.ts_lut("flare")) if v}
    stage = T.OUT.parent / "stage_labels"
    for d in (T.IMG_DIR, stage):
        d.mkdir(parents=True, exist_ok=True)
    per = {a: {} for a in SHRINK}
    for c in ids:
        G, boxes = synth(rng, np)
        lab = G
        if c == "FLARETs_0002" and scen == "float_label_non_integer":
            lab = G.astype(np.float32)
            lab[lab == 13] = 12.99995
        if c == "FLARETs_0002" and scen == "float_label_exact":
            lab = G.astype(np.float32)
        nib.save(nib.Nifti1Image(lab, aff), str(stage / f"{c}.nii.gz"))
        if c == "FLARETs_0002" and scen == "corrupt_label":
            (stage / f"{c}.nii.gz").write_bytes(b"\x1f\x8b\x08\x00garbage-not-a-nifti")
        nib.save(nib.Nifti1Image(rng.integers(-200, 300, G.shape).astype(np.int16), aff), str(T.image_path(c)))
        for a, s in SHRINK.items():
            Y = np.zeros_like(G)
            for o, (sx, sy, sz) in boxes.items():
                k = int(rng.integers(0, 2)) + s
                Y[sx, sy, sz.start + k:sz.stop] = (lut_inv[o] if T.LUT_KIND[a] == "ts" else o)
            p = T.pred_dir(a) / "raw" / f"{c}.nii.gz"
            p.parent.mkdir(parents=True, exist_ok=True)
            nib.save(nib.Nifti1Image(Y, aff), str(p))
            per[a][c] = T.sha256(p)
    inc = [c for c in ids if c != EXCL]
    models, done_sha = {}, {}
    for a in SHRINK:
        if a == "F0":
            done_sha[a] = {}
            continue
        ck = T.OUT.parent / "models" / a / "fold_0" / "checkpoint_final.pth"
        ck.parent.mkdir(parents=True, exist_ok=True)
        ck.write_bytes(os.urandom(256))
        models[a] = {str(ck): T.sha256(ck)}
        done_sha[a] = {"291": T.sha256(ck)}
    tsw = {}
    for tid in (291, 292, 293, 294, 295):
        d = Path(T.TS_ROOT) / "nnunet" / "results" / f"Dataset{tid}_stand_in" / "nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres"
        (d / "fold_0").mkdir(parents=True, exist_ok=True)
        for f in ("fold_0/checkpoint_final.pth", "plans.json", "dataset.json"):
            (d / f).write_bytes(os.urandom(64))
    for p in T.ts_weight_files():
        tsw[str(p)] = T.sha256(p)
    models["TS_WEIGHTS"] = tsw
    before = (T.CUTOFF - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
    for a in SHRINK:
        fin = T.CUTOFF.strftime("%Y-%m-%d %H:%M:%S") if (scen == "cutoff_done" and a == "STOCK") else before
        T.write_json(T.done_file(a), {"arm": a, "ids": inc, "per_scan_raw_sha256": {c: per[a][c] for c in inc},
                                      "model_sha256": done_sha[a], "mode": "real", "finished": fin})
    T.write_json(T.RUN / "K1_0038.json", {"arm": "K1_0038", "not_run": "rehearsal: declared NOT RUN"})
    T.write_json(T.RUN / "RUN_FINISHED.json", {"arms": {}, "finished": before})
    T.write_json(T.EXCLUSIONS, {"excluded": {EXCL: {"reason": "rehearsal"}}, "decision": "K1B_PASS"})
    now = datetime.now().replace(microsecond=0)
    T.write_json(T.FREEZE, {"files": {}, "models": models, "pending_models": [], "versions": T.versions(),
                            "written": (now - timedelta(hours=4)).strftime("%Y-%m-%d %H:%M:%S")})
    lab = {}
    for c in ids:
        p = stage / f"{c}.nii.gz"
        h, s = T.sha256(p), p.stat().st_size
        oid = T.pointer_oid(h, s) if not (scen == "pointer_mismatch" and c == "FLARETs_0003") else "0" * 40
        lab[c] = {"path": f"{T.HF_LBL_PREFIX}{c}.nii.gz", "size": s, "pointer_oid": oid}
    T.write_json(T.MANIFEST, {"repo": T.HF_REPO, "revision": T.HF_REVISION, "labels": lab, "images": {}})
    z = T.OUT.parent / "osf_deposit_rehearsal_flaret50.zip"
    z.write_bytes(os.urandom(128))
    T.prereg_line("DEPOSIT", z, "rehearsal deposit")
    T.write_json(T.GATE_UPLOADED, {"osf_url": "https://osf.io/abcde", "uploaded_kst": (now - timedelta(hours=3)).strftime(
        "%Y-%m-%d %H:%M:%S"), "written_by": "rehearsal", "evidence": "synthetic", "deposit_zip_sha256": T.sha256(z),
        "freeze_sha256": T.sha256(T.FREEZE)})
    T.write_json(T.GATE_PUBLIC, {"osf_url": "https://osf.io/abcde", "public_checked_kst": (now - timedelta(hours=1)).strftime(
        "%Y-%m-%d %H:%M:%S"), "api_http_status": 200, "public": True, "written_by": "rehearsal", "evidence": "synthetic"})
    T.LABEL_SOURCE = stage
    return stage


_REAL = {}


def _boom(args):
    """crash_scoring: one worker exception (module level, so the worker pool can pickle it)."""
    flag = Path(os.environ["T50_REH_FLAG"])
    if args[0] == "FLARETs_0003" and not flag.exists():
        flag.touch()
        raise RuntimeError("rehearsal: simulated worker crash")
    return _REAL["score_one"](args)


def _stop(args):
    """sigint: one SIGINT to the look process while a worker scores a scan."""
    import signal as _s
    flag = Path(os.environ["T50_REH_FLAG"])
    if args[0] == "FLARETs_0004" and not flag.exists():
        flag.touch()
        os.kill(os.getppid(), _s.SIGINT)
    return _REAL["score_one"](args)


def attempt(SC, T, cont):
    try:
        return f"rc {SC.look(cont=cont)}"
    except T.Refuse as e:
        return f"REFUSE {e.msg}"
    except BaseException as e:                       # noqa: BLE001
        return f"RAISED {type(e).__name__}: {str(e)[:160]}"


def child(scen, root):
    root = Path(root)
    os.environ["MPLCONFIGDIR"] = str(root / "mpl")
    os.environ["TMPDIR"] = str(root / "tmp")
    (root / "tmp").mkdir(parents=True, exist_ok=True)
    guard(root)
    import t50_common as T
    T.use_root(root)
    import t50_score as SC
    build_world(T, SC, scen)
    rep = {"scenario": scen}
    if scen in ("ok", "corrupt_label", "float_label_non_integer", "float_label_exact", "pointer_mismatch", "cutoff_done"):
        rep["first"] = attempt(SC, T, False)
        if scen == "corrupt_label":
            rep["continue"] = attempt(SC, T, True)
    elif scen == "second_look":
        rep["first"] = attempt(SC, T, False)
        rep["second"] = attempt(SC, T, False)
    elif scen == "crash_scoring":
        os.environ["T50_REH_FLAG"] = str(root / "crashed_once")
        _REAL["score_one"] = SC.score_one
        SC.score_one = _boom
        rep["first"] = attempt(SC, T, False)
        SC.score_one = _REAL["score_one"]
        rep["continue"] = attempt(SC, T, True)
    elif scen == "crash_before_labels_json":
        real_wj, state = T.write_json, {"done": False}

        def wj(path, obj, mode=None):
            if Path(path).name == "labels.json" and not state["done"]:
                state["done"] = True
                raise KeyboardInterrupt("rehearsal: a stop between the archive and labels.json")
            return real_wj(path, obj, mode)
        T.write_json = wj
        rep["first"] = attempt(SC, T, False)
        T.write_json = real_wj
        rep["continue"] = attempt(SC, T, True)
    elif scen == "label_swapped_after_archive":
        import nibabel as nib
        import numpy as np
        real_sg, state = SC.sanity_gate, {"n": 0}

        def sg(ids):
            state["n"] += 1
            if state["n"] == 1:
                raise RuntimeError("rehearsal: crash after the archive")
            return real_sg(ids)
        SC.sanity_gate = sg
        rep["first"] = attempt(SC, T, False)
        p = T.label_path("FLARETs_0001")
        G = np.asarray(nib.load(str(p)).dataobj).copy()
        G[G == 1] = 0
        os.chmod(p, 0o644)
        nib.save(nib.Nifti1Image(G, np.diag([1.5, 1.5, 3.0, 1.0])), str(p))
        rep["continue"] = attempt(SC, T, True)
    elif scen == "sigint":
        os.environ["T50_REH_FLAG"] = str(root / "sigint_once")
        _REAL["score_one"] = SC.score_one
        SC.score_one = _stop
        rep["first"] = attempt(SC, T, False)
        SC.score_one = _REAL["score_one"]
        rep["continue"] = attempt(SC, T, True)
    kinds = [ln.split()[2] for ln in T.prereg_lines() if len(ln.split()) > 2]
    rep["prereg_kinds"] = kinds
    q = SC.P()
    rep["results_exists"] = q.results.is_file()
    if q.results.is_file():
        res = json.load(open(q.results))
        rep["summary"] = __import__("t50_analysis").summary(res)
        rep["R1_M"] = res["R1"]["M"]
        rep["n_scans"] = res["n_scans"]
        rep["arms_not_run"] = sorted(res["arms_not_run"])
        rep["S3"] = res["secondary_holm_family"]["contrasts"]["S3"].get("reading")
        rep["panels"] = sorted(k for k, v in res.get("panels_rendered", {}).items() if isinstance(v, dict) and v.get("png"))
    print("REHEARSAL_JSON " + json.dumps(rep))


# ---------------------------------------------------------------------------------------------------- parent side
def expect(r, ok_m):
    s, k = r["scenario"], r.get("prereg_kinds", [])
    first, cont = r.get("first", ""), r.get("continue", "")
    seq_ok = [x for x in k if x in ("GT_OPENED", "LABELS", "RESULTS")] == ["GT_OPENED", "LABELS", "RESULTS"]
    if s == "ok":
        return first == "rc 0" and r["results_exists"] and seq_ok and r.get("panels") == ["largest_loss", "median_gain"]
    if s == "crash_scoring":
        return first.startswith("RAISED RuntimeError") and cont == "rc 0" and "GT_CRASHED" in k and \
            "GT_CONTINUED" in k and r["results_exists"] and abs(r["R1_M"] - ok_m) < 1e-12
    if s == "crash_before_labels_json":
        return first.startswith("RAISED KeyboardInterrupt") and cont == "rc 0" and "GT_CRASHED" in k and \
            r["results_exists"] and abs(r["R1_M"] - ok_m) < 1e-12
    if s == "label_swapped_after_archive":
        return first.startswith("RAISED RuntimeError") and cont == "rc 0" and "GT_NOTE" in k and r["results_exists"] \
            and abs(r["R1_M"] - ok_m) < 1e-12
    if s == "corrupt_label":
        return first == "rc 5" and "GT_ABORTED" in k and cont.startswith("REFUSE") and not r["results_exists"]
    if s in ("float_label_non_integer", "pointer_mismatch"):
        return first == "rc 5" and "GT_ABORTED" in k and not r["results_exists"]
    if s == "float_label_exact":
        return first == "rc 0" and r["results_exists"] and abs(r["R1_M"] - ok_m) < 1e-12
    if s == "sigint":
        return first.startswith("RAISED Terminated") and "GT_CRASHED" in k and cont == "rc 0" and r["results_exists"]
    if s == "cutoff_done":
        return first == "rc 0" and "STOCK" in r.get("arms_not_run", []) and r.get("S3") == "NOT RUN"
    if s == "second_look":
        return first == "rc 0" and r.get("second", "").startswith("REFUSE")
    return False


def record_result(out, counts):
    """Before the freeze: I_flaret50/rehearsal.json (a frozen record once the freeze exists) and a REHEARSAL line. After
    the freeze: a new read-only rehearsal_postfreeze_<n>.json with its own REHEARSAL line; the frozen rehearsal.json is
    never touched (diff check F1). -> the path written."""
    import t50_common as T
    if T.FREEZE.exists():
        n = 1
        while (T.OUT / f"rehearsal_postfreeze_{n}.json").exists():
            n += 1
        p = T.OUT / f"rehearsal_postfreeze_{n}.json"
        T.write_json(p, out, mode=0o444)
        T.prereg_line("REHEARSAL", p, f"flaret50 post-freeze look rehearsal {n} on stand-in labels: all_pass "
                                      f"{out['all_pass']} ({counts}); the frozen rehearsal.json is unchanged")
        return p
    T.write_json(T.REHEARSAL_RECORD, out)
    T.prereg_line("REHEARSAL", T.REHEARSAL_RECORD, f"flaret50 look rehearsal on stand-in labels: all_pass "
                                                   f"{out['all_pass']} ({counts})")
    return T.REHEARSAL_RECORD


def parent(root_arg):
    import t50_common as T
    base = Path(root_arg) if root_arg else Path(tempfile.mkdtemp(prefix="t50_rehearsal_"))
    reports = []
    for s in SCENARIOS:
        root = base / s
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        r = subprocess.run([sys.executable, "-I", "-B", __file__, "--child", s, str(root)], capture_output=True, text=True,
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
        line = [ln for ln in r.stdout.splitlines() if ln.startswith("REHEARSAL_JSON ")]
        rep = json.loads(line[-1][len("REHEARSAL_JSON "):]) if line else {"scenario": s, "error": r.stderr[-1500:]}
        rep["child_rc"] = r.returncode
        reports.append(rep)
        print(f"{s}: {rep.get('first', rep.get('error', ''))[:80]} | {rep.get('continue', '')[:60]}", flush=True)
    ok_m = next((r.get("R1_M") for r in reports if r["scenario"] == "ok"), None)
    for r in reports:
        r["pass"] = bool(ok_m is not None and "error" not in r and expect(r, ok_m))
    out = {"all_pass": all(r["pass"] for r in reports), "scenarios": reports, "run": T.now(),
           "code_sha256": T.code_hashes(["t50_rehearse.py", "t50_score.py", "t50_analysis.py", "t50_common.py"]),
           "note": "synthetic stand-in world only; every study path redirected (t50_common.use_root); audit hook in every "
                   "child blocked writes outside its scratch root and reads of project data"}
    record_result(out, f"{sum(r['pass'] for r in reports)}/{len(reports)}")
    if not root_arg:
        shutil.rmtree(base, ignore_errors=True)
    print(f"ALL_PASS {out['all_pass']}")
    return 0 if out["all_pass"] else 1


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--child":
        child(sys.argv[2], sys.argv[3])
    else:
        ra = sys.argv[sys.argv.index("--root") + 1] if "--root" in sys.argv else None
        sys.exit(parent(ra))
