"""vccor/final LCTSC FREEZE (CPU; once, before the deposit): K_lctsc/code_freeze.json (read-only) with the sha256 of every
study file, then the deposit zip K_lctsc/osf_deposit_lctsc_<date>.zip and a 'FREEZE' and a 'DEPOSIT' line in
final_prereg.log. Refuses if unit_tests.json or rehearsal.json is missing or not all_pass, if a training run is not DONE,
or if a freeze already exists.

Frozen: every scripts/vccor/final/lctsc/*.py and unit file; the frozen modules they import (final_common, fin_nn, fin_look,
bsl_common, flaret50/t50_analysis, m2/m2_cluster, tsft_full/tsft_full_common, tsft_full/ts_pipeline, peft/lora); the
records (tcia_manifest, draws (superseded) and draws_site, download, images, labels_train_v2, dups, prep/DONE_datasets, prep/labels/manifest, unit_tests,
rehearsal, plan_lctsc); every F1b export (checkpoint + export.json + DONE.json); TotalSegmentator's own checkpoints of the
five parts; package versions of the SSL .venv and external/lctsc_env.

The deposit zip holds the code, the records, the plan and the OSF guide; no image, label or model file (models are listed
by hash).

usage:  .venv/bin/python -I -B scripts/vccor/final/lctsc/lctsc_freeze.py [--check]
"""
import argparse
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lctsc_common as C  # noqa: E402
import lctsc_gates as G  # noqa: E402
import lctsc_log as L  # noqa: E402
import lctsc_runs as R  # noqa: E402

FINAL = HERE.parent
FROZEN_IMPORTS = [FINAL / "final_common.py", FINAL / "fin_nn.py", FINAL / "fin_look.py", FINAL / "bsl_common.py",
                  FINAL / "flaret50/t50_analysis.py", FINAL.parent / "m2/m2_cluster.py", FINAL.parent / "tsft_full/tsft_full_common.py",
                  FINAL.parent / "tsft_full/ts_pipeline.py", FINAL.parent / "peft/lora.py"]
RECORDS = [C.MANIFEST, C.DRAWS, C.DOWNLOAD, C.OUT / "images.json", C.OUT / "labels_train_v2.json", C.DRAWS_SITE, C.OUT / "dups.json",
           R.PREP / "DONE_datasets.json", R.PREP / "labels/manifest.json", C.OUT / "unit_tests.json", C.OUT / "rehearsal.json",
           C.OUT / "plan_lctsc.json", C.OUT / "train_state_at_freeze.json"]    # review NB2: the SNAPSHOT, never the live file
DOCS = [C.OUT / "OSF_GUIDE_LCTSC.md", C.OUT / "REGISTRATION_SKELETON_LCTSC.md"]


def cv2_record():
    """review N2: the one OpenCV build rt-utils / lctsc_raster use in external/lctsc_env (version + .so sha256)."""
    r = subprocess.run([str(G.REPO / "external/lctsc_env/bin/python"), "-I", "-c",
                        "import cv2, glob, hashlib; f = glob.glob(cv2.__path__[0] + '/cv2*.so')[0]; "
                        "print(cv2.__version__); print(f); print(hashlib.sha256(open(f, 'rb').read()).hexdigest())"],
                       capture_output=True, text=True, check=True)
    v, f, h = r.stdout.strip().split("\n")              # one value per line (the path contains spaces)
    return {"version": v, "so": f, "sha256": h}


def versions():
    out = {}
    for name, py in (("ssl_venv", G.REPO / ".venv/bin/python"), ("lctsc_env", G.REPO / "external/lctsc_env/bin/python")):
        r = subprocess.run([str(py), "-I", "-m", "pip", "freeze"], capture_output=True, text=True)
        out[name] = sorted(r.stdout.split())
    return out


def training_alive(cmdlines, unit_active):
    """review NB2: True if the training unit is active or any process runs lctsc_train.py / lctsc_train_queue.py."""
    return bool(unit_active) or any(("lctsc_train.py" in c or "lctsc_train_queue.py" in c) for c in cmdlines)


def _live_training():
    cmd = []
    for d in Path("/proc").iterdir():
        if d.name.isdigit() and int(d.name) != os.getpid():
            try:
                cmd.append((d / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace"))
            except OSError:
                pass
    r = subprocess.run(["systemctl", "--user", "is-active", "ssl-lctsc-train.service"], capture_output=True, text=True)
    return training_alive(cmd, r.stdout.strip() in ("active", "activating", "deactivating"))


def classify_draws(train_state, now_kst, done_files, deadline=None):
    """Registered training-failure rule (review NB1): per training draw -> DONE / NOT RUN / PENDING.
      DONE     DONE.json exists and its 'finished' is before the training deadline (Sun 2026-11-15 18:00 KST);
      NOT RUN  DONE.json finished at or after the deadline; or the queue's state is FAILED, 'NOT DONE BY THE TRAINING
               DEADLINE' or 'INTEGRITY REFUSED'; or the deadline has passed without a DONE.json;
      PENDING  otherwise (still training, queued, or RESET before the deadline)."""
    from datetime import datetime
    import lctsc_infra as I
    dl = deadline or I.TRAIN_DEADLINE
    t = datetime.strptime(now_kst, "%Y-%m-%d %H:%M:%S")
    out = {}
    for run, df in done_files.items():
        q = train_state.get("runs", {}).get(run, {}).get("status")
        if Path(df).is_file():
            fin = datetime.strptime(json.load(open(df))["finished"], "%Y-%m-%d %H:%M:%S")
            out[run] = ({"status": "DONE", "finished": str(fin)} if fin < dl else
                        {"status": "NOT RUN", "reason": f"DONE.json finished {fin}, at or after the training deadline {dl}"})
        elif q in ("FAILED", "NOT DONE BY THE TRAINING DEADLINE", "INTEGRITY REFUSED"):
            out[run] = {"status": "NOT RUN", "reason": f"training queue state {q}"}
        elif t >= dl:
            out[run] = {"status": "NOT RUN", "reason": f"no model by the training deadline {dl}"}
        else:
            out[run] = {"status": "PENDING", "reason": f"queue state {q}"}
    return out


def ts_weights():
    import tsft_full_common as Q  # noqa: F401
    return {str(t): str(Q.TS_MODEL[t] / "fold_0" / "checkpoint_final.pth") for t in Q.TS_MODEL}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    for p in (FINAL, FINAL.parent / "tsft_full"):
        sys.path.insert(0, str(p))
    G.kst_gate()                                                # review diff-2 N3
    if G.FREEZE.exists():
        raise G.Refuse("a freeze exists")
    if _live_training():                                        # review NB2: no freeze while a training can still write
        raise G.Refuse("ssl-lctsc-train or a training process is still alive: the freeze waits until it has ended")
    cur = {f.name: G.sha256(f) for f in sorted(HERE.glob("*.py")) + sorted(HERE.glob("*.service"))}
    for rec in ("unit_tests.json", "rehearsal.json"):         # review SF1: both records must be of the CURRENT code
        p = C.OUT / rec
        if not p.is_file() or not json.load(open(p)).get("all_pass"):
            raise G.Refuse(f"{rec} missing or not all_pass")
        if json.load(open(p)).get("code_sha256") != cur:
            raise G.Refuse(f"{rec} was recorded on other code than the current files: re-run it")
    st = json.load(open(R.C.OUT / "train_state.json")) if (R.C.OUT / "train_state.json").is_file() else {"runs": {}}
    if not a.check:                                             # review NB2: the frozen snapshot of the queue state
        L.write_json(R.C.OUT / "train_state_at_freeze.json", dict(st, snapshot_kst=G.now()), readonly=True)
    status = classify_draws(st, G.now(), {spec["run"]: (R.MODELS / spec["run"] / "DONE.json") for spec in G.ARMS.values()
                                           if spec["run"]})
    pending = [r for r, v in status.items() if v["status"] == "PENDING"]
    if pending:
        raise G.Refuse(f"training draws still PENDING: {pending} (the freeze waits until each is DONE or NOT RUN)")
    if not any(v["status"] == "DONE" for v in status.values()):
        raise G.Refuse("no training draw is DONE: no look (NOTRUN EXPERIMENT)")
    models = {}
    for arm, spec in G.ARMS.items():
        if spec["run"] is None or status[spec["run"]]["status"] != "DONE":
            continue
        d = json.load(open(R.MODELS / spec["run"] / "DONE.json"))
        models[arm] = {str(t): G.sha256(Path(md) / "fold_0" / "checkpoint_final.pth") for t, md in d["resolved"]["parts"].items()}
    files = sorted(HERE.glob("*.py")) + sorted(HERE.glob("*.service")) + FROZEN_IMPORTS + RECORDS + DOCS
    missing = [str(f) for f in files if not Path(f).is_file()
               and not (a.check and Path(f).name == "train_state_at_freeze.json")]   # written only by the real freeze
    if missing:
        raise G.Refuse(f"missing files: {missing}")
    model_files = []
    for arm, spec in G.ARMS.items():
        if spec["run"] and status[spec["run"]]["status"] == "DONE":
            d = json.load(open(R.MODELS / spec["run"] / "DONE.json"))
            model_files.append(R.MODELS / spec["run"] / "DONE.json")
            for md in d["resolved"]["parts"].values():
                model_files += [Path(md) / "fold_0" / "checkpoint_final.pth", Path(md) / "export.json"]
    tsw = ts_weights()
    if a.check:
        print(f"freeze check OK: {len(files)} files, {len(model_files)} model files, TS weights {len(tsw)}")
        return 0
    fz = {"frozen_kst": G.now(), "files": {L.rel(f): G.sha256(f) for f in files + model_files},
          "ts_weights": {t: G.sha256(p) for t, p in tsw.items()}, "models": models, "draws_status": status,
          "versions": versions(),
          "opencv": cv2_record(),
          "cutoff_kst": G.CUTOFF, "stop_kst": G.STOP}
    for t, p in tsw.items():
        fz["files"][p] = fz["ts_weights"][t]
    G.write_json(G.FREEZE, fz, mode=0o444)
    L.prereg_line("FREEZE", G.FREEZE, f"lctsc code / records / models frozen: {len(fz['files'])} files; draws "
                                       f"{ {r: v['status'] for r, v in status.items()} }")
    for r, v in status.items():
        if v["status"] == "NOT RUN":
            L.prereg_note("NOTRUN", f"{G.DRAW_ARM[r.split('_')[1]]} its training draw {r} is NOT RUN ({v['reason']}); "
                                    f"registered training-failure rule")
    zp = C.OUT / f"osf_deposit_lctsc_{G.now()[:10]}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files + [G.FREEZE]:
            z.write(f, f"lctsc_deposit/{L.rel(f)}")
    L.prereg_line("DEPOSIT", zp, "lctsc OSF deposit zip (code, records, plan, guide; no image, label or model file)")
    print(zp, G.sha256(zp))
    return 0


if __name__ == "__main__":
    sys.exit(main())
