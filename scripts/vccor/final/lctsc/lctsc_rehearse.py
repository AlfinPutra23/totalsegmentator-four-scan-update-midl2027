"""vccor/final LCTSC LOOK REHEARSAL (CPU; before the freeze). Revised after the independent review (SF1, B1, B2, N7, SF2).
The whole look -- lctsc_look_labels.py (external/lctsc_env) then lctsc_score.py (SSL .venv) -- runs unchanged under scratch
roots (LCTSC_REHEARSAL_ROOT = K_lctsc/rehearsal_<n>/) on STAND-IN patients: the 12 site-draw TRAINING patients, whose labels
were already read for training. No test file is touched. Each root has its own prereg.log, a stand-in freeze inside a
stand-in deposit zip, stand-in upload / public gate files, standin.json and pred/<arm>/ (F0 = TS's own merged output from the
prep capture; F1b_S1 = F0; F1b_S2 = F0 without the esophagus on every second slice; F1b_S3 = F0), so the real gate code runs.

Scenarios (each in its own root):
  A  main path with faults: two HTML bodies before a good zip for one patient (B1 temporary fault); a structure dropped
     from one stand-in reference (SF2 missing-ROI rule); a crash of the label step after 3 patients, then --continue; a
     crash of scoring after 5 patients, then a rerun; a crash between results.json and its RESULTS line, then a rerun
     (N7). Checks:
       a1 every converted label equals the training label v2 (same rasteriser) except the dropped structure;
       a2 every pervol Dice equals an independent numpy Dice of the stored maps vs the labels;
       a3 F1b_S1 / F1b_S3 deltas are exactly 0; F1b_S2's esophagus change is < 0; the dropped structure is not evaluable;
       a4 GT_OPENED, LABELS and RESULTS lines exist once each; a second look and a second scoring are refused.
  B  persistent manifest mismatch for one patient: GT_ABORTED is logged; --continue and scoring are refused.
  C  an F1b arm without DONE: before the cutoff the look is refused (B2); with the rehearsal clock after the cutoff the
     look runs with that arm 'not DONE by the cutoff' and P1 on the two remaining draws.
  D  every arm DONE but a binding NOTRUN line for F1b_S3: the NOTRUN line wins (review NS1); P1 on S1 and S2.
Writes K_lctsc/rehearsal.json (all_pass, code_sha256 of every *.py and *.service) and discloses rehearsal_1 (aborted by its
old zip-size check, superseded) and rehearsal_2 (pooled-draw code, superseded).

usage:  .venv/bin/python -I -B scripts/vccor/final/lctsc/lctsc_rehearse.py
"""
import json
import os
import subprocess
import sys
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent), str(HERE.parent.parent / "tsft_full"), str(HERE.parent / "flaret50"),
           str(HERE.parent.parent / "m2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import lctsc_common as C  # noqa: E402
import lctsc_log as L  # noqa: E402
import lctsc_runs as R  # noqa: E402

VENV = "/home/user/Documents/Medical Imagining SSL/.venv/bin/python"
LENV = "/home/user/Documents/Medical Imagining SSL/external/lctsc_env/bin/python"
ARMS = {"F0": None, "F1b_S1": "same", "F1b_S2": "eso_half", "F1b_S3": "same"}


def tm(minutes):
    return (datetime.now() - timedelta(minutes=minutes)).strftime("%Y-%m-%d %H:%M:%S")


def new_root():
    n = 1
    while (C.OUT / f"rehearsal_{n}").exists():
        n += 1
    root = C.OUT / f"rehearsal_{n}"
    root.mkdir()
    return root


def setup(root, ids, arms=ARMS):
    import nibabel as nib
    import numpy as np
    json.dump({"ids": ids, "note": "stand-in TRAINING site-draw patients"}, open(root / "standin.json", "w"))
    lut = C.ts_total_lut()
    eso = [i for i in range(256) if lut[i] == 1]
    for arm, how in arms.items():
        pd = root / "pred" / arm
        (pd / "raw").mkdir(parents=True)
        per = {}
        for pid in ids:
            im = nib.load(str(R.PREP / "ts_cpu" / pid / "raw_total.nii.gz"))
            a = np.asarray(im.dataobj).astype(np.uint8)
            if how == "eso_half":
                keep = np.isin(a, eso)
                keep[:, :, 1::2] = False
                a = a.copy()
                a[np.isin(a, eso) & ~keep] = 0
            nib.save(nib.Nifti1Image(a, im.affine), str(pd / "raw" / f"{pid}.nii.gz"))
            per[pid] = L.sha256(pd / "raw" / f"{pid}.nii.gz")
        json.dump({"arm": arm, "per_scan_raw_sha256": per, "finished": tm(60), "mode": "rehearsal"}, open(pd / "DONE.json", "w"))
    fz = {"frozen_kst": tm(50), "files": {"scripts/vccor/final/lctsc/lctsc_common.py": L.sha256(HERE / "lctsc_common.py")}}
    json.dump(fz, open(root / "code_freeze.json", "w"))
    dz = root / "rehearsal_lctsc_deposit.zip"
    with zipfile.ZipFile(dz, "w") as z:
        z.write(root / "code_freeze.json", "lctsc_deposit/code_freeze.json")
    with open(root / "prereg.log", "a") as f:
        f.write(f"[{tm(49)}] DEPOSIT sha256 {L.sha256(dz)} {L.rel(dz)} (rehearsal stand-in)\n")
    json.dump({"osf_url": "https://osf.io/rehearsal", "uploaded_kst": tm(40), "written_by": "rehearsal", "evidence": "stand-in",
               "deposit_zip_sha256": L.sha256(dz), "freeze_sha256": L.sha256(root / "code_freeze.json")},
              open(root / "gate_osf_uploaded.json", "w"))
    json.dump({"osf_url": "https://osf.io/rehearsal", "public_checked_kst": tm(30), "api_http_status": 200, "public": True,
               "written_by": "rehearsal", "evidence": "stand-in"}, open(root / "gate_osf_public.json", "w"))


def run(py, script, root, faults="", now=None, args=()):
    env = dict(os.environ, LCTSC_REHEARSAL_ROOT=str(root), CUDA_VISIBLE_DEVICES="", LCTSC_REHEARSAL_FAULTS=faults)
    if now:
        env["LCTSC_REHEARSAL_NOW"] = now
    r = subprocess.run([py, "-I", "-B", str(HERE / script), *args], env=env, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr)[-800:]


def count(root, kind):
    return sum(1 for line in open(root / "prereg.log") if f" {kind} " in line)


def scenario_a(ids):
    import nibabel as nib
    import numpy as np
    import lctsc_analysis as A
    root = new_root()
    setup(root, ids)
    faults = f"corrupt:{ids[0]}:2;drop_roi:{ids[1]}:esophagus;crash_after:3;score_crash_after:5;crash_before_results_line"
    out = {"root": L.rel(root), "steps": []}
    for script, py, args in (("lctsc_look_labels.py", LENV, ()), ("lctsc_look_labels.py", LENV, ("--continue",)),
                             ("lctsc_score.py", VENV, ()), ("lctsc_score.py", VENV, ()), ("lctsc_score.py", VENV, ())):
        rc, tail = run(py, script, root, faults, args=args)
        out["steps"].append({"script": script, "args": list(args), "rc": rc, "tail": tail[-300:]})
    out["rc_as_expected"] = [s["rc"] for s in out["steps"]] == [70, 0, 70, 70, 0]
    chk = {}
    lab = json.load(open(root / "look/labels.json"))
    same = True
    for p in ids:
        a = np.asarray(nib.load(str(root / "look/labels" / f"{p}.nii.gz")).dataobj)
        b = np.asarray(nib.load(str(C.train_label_path(p))).dataobj).copy()
        if p == ids[1]:
            b[b == 1] = 0
        same &= bool(np.array_equal(a, b))
    chk["a1_labels_equal_training_v2"] = bool(same and lab["patients"][ids[1]]["missing"] == ["esophagus"])
    lut = C.ts_total_lut()
    bad = []
    for p in ids:
        pv = json.load(open(root / "look/pervol" / f"{p}.json"))["NATIVE"]
        Gt = np.asarray(nib.load(str(root / "look/labels" / f"{p}.nii.gz")).dataobj).astype(np.uint8)
        for arm in ("F0", "F1b_S2"):
            Y = lut[np.asarray(nib.load(str(root / "pred" / arm / "raw" / f"{p}.nii.gz")).dataobj).astype(np.uint8)]
            for o in range(1, C.N_ORG + 1):
                if pv["arms"][arm]["status"][o - 1] == "TP":
                    g, y = Gt == o, Y == o
                    d = 200.0 * (g & y).sum() / (g.sum() + y.sum())
                    if abs(d - pv["arms"][arm]["dice"][o - 1]) > 1e-9:
                        bad.append((p, arm, o))
    chk["a2_dice_matches_numpy"] = not bad
    res = json.load(open(root / "look/results.json"))
    RR = A.mask_missing({p: {"NATIVE": json.load(open(root / "look/pervol" / f"{p}.json"))["NATIVE"]} for p in ids},
                        {ids[1]: ["esophagus"]})
    z = all(abs(v) < 1e-12 for d in ("S1", "S3") for v in A.md(RR, f"F1b_{d}", "F0", ids).values())
    eso = res["organ_costs"]["per_organ"]["esophagus"]
    chk["a3_known_changes_and_missing_rule"] = bool(z and eso["M"] is not None and eso["M"] < 0
                                                    and eso["n_scans"] == len(ids) - 1
                                                    and res["not_evaluable_structures"] == {ids[1]: ["esophagus"]})
    rc1, _ = run(LENV, "lctsc_look_labels.py", root)
    rc2, _ = run(VENV, "lctsc_score.py", root)
    chk["a4_lines_once_and_second_look_refused"] = bool(count(root, "GT_OPENED") == 1 and count(root, "LABELS") == 1
                                                        and count(root, "RESULTS") == 1 and rc1 != 0 and rc2 != 0)
    out["checks"] = chk
    out["temporary_faults_logged"] = sum(1 for line in open(root / "events.log") if "temporary fault" in line)
    out["P1_reading_on_standins"] = res["P1"]["reading"]
    return out


def scenario_b(ids):
    root = new_root()
    setup(root, ids[:3])
    rc1, t1 = run(LENV, "lctsc_look_labels.py", root, faults=f"mismatch:{ids[1]}")
    rc2, t2 = run(LENV, "lctsc_look_labels.py", root, args=("--continue",))
    rc3, t3 = run(VENV, "lctsc_score.py", root)
    ok = rc1 != 0 and count(root, "GT_ABORTED") == 1 and rc2 != 0 and rc3 != 0
    return {"root": L.rel(root), "rc": [rc1, rc2, rc3], "checks": {"b1_persistent_mismatch_aborts_for_good": bool(ok)},
            "tails": [t1[-200:], t2[-200:], t3[-200:]]}


def scenario_c(ids):
    import shutil
    root = new_root()
    setup(root, ids[:4])
    shutil.move(str(root / "pred" / "F1b_S3"), str(root / "F1b_S3_never_done"))
    rc1, t1 = run(LENV, "lctsc_look_labels.py", root, args=("--check",))
    after = "2026-11-19 18:30:00"
    rc2, t2 = run(LENV, "lctsc_look_labels.py", root, now=after)
    rc3, t3 = run(VENV, "lctsc_score.py", root, now=after)
    ok = rc1 != 0 and "before the cutoff every arm must be DONE" in t1 and rc2 == 0 and rc3 == 0
    if ok:
        res = json.load(open(root / "look/results.json"))
        ok = res["draws_run"] == ["S1", "S2"] and res["arms_not_run"].get("F1b_S3") == "not DONE by the cutoff"
    return {"root": L.rel(root), "rc": [rc1, rc2, rc3], "checks": {"c1_missing_arm_refused_before_cutoff_not_run_after": bool(ok)},
            "tails": [t1[-200:], t2[-200:], t3[-200:]]}


def scenario_d(ids):
    """review NS1 / N7: every arm DONE, but a binding NOTRUN line for F1b_S3 (the runner's second failure): the look runs
    before the cutoff, leaves F1b_S3 out ('declared NOT RUN') and reads P1 on S1 and S2."""
    root = new_root()
    setup(root, ids[:4])
    with open(root / "prereg.log", "a") as f:
        f.write(f"[{tm(5)}] NOTRUN lctsc F1b_S3 2 failed attempts (rehearsal)\n")
    rc1, t1 = run(LENV, "lctsc_look_labels.py", root)
    rc2, t2 = run(VENV, "lctsc_score.py", root)
    ok = rc1 == 0 and rc2 == 0
    if ok:
        res = json.load(open(root / "look/results.json"))
        ok = res["draws_run"] == ["S1", "S2"] and res["arms_not_run"].get("F1b_S3", "").startswith("declared NOT RUN")
    return {"root": L.rel(root), "rc": [rc1, rc2], "checks": {"d1_notrun_line_beats_done": bool(ok)},
            "tails": [t1[-200:], t2[-200:]]}


def main():
    assert "LCTSC_REHEARSAL_ROOT" not in os.environ
    ids = C.draw_patients()
    res = {"ids": ids, "scenarios": {}}
    for name, fn in (("A", scenario_a), ("B", scenario_b), ("C", scenario_c), ("D", scenario_d)):
        try:
            res["scenarios"][name] = fn(ids)
        except Exception as e:  # noqa: BLE001
            res["scenarios"][name] = {"error": f"{type(e).__name__}: {e}", "checks": {f"{name}_ran": False}}
    checks = {k: v for s in res["scenarios"].values() for k, v in s.get("checks", {}).items()}
    res["checks"] = checks
    res["all_pass"] = bool(checks) and all(checks.values()) and bool(res["scenarios"]["A"].get("rc_as_expected", False))
    res["superseded_rehearsals"] = {
        "rehearsal_1": "2026-11-13 11:42, pooled-draw code; aborted by design at its old RTSTRUCT size check (zip bytes vs TCIA "
                       "FileSize = uncompressed bytes); the check was fixed; superseded",
        "rehearsal_2": "2026-11-13 11:44, pooled-draw code before the review fixes; all checks passed then; superseded"}
    res["code_sha256"] = {f.name: L.sha256(f) for f in sorted(HERE.glob("*.py")) + sorted(HERE.glob("*.service"))}
    res["written"] = L.now()
    L.write_json(C.OUT / "rehearsal.json", res)
    print(json.dumps({"all_pass": res["all_pass"], "checks": checks,
                      "A_rc": [s["rc"] for s in res["scenarios"]["A"].get("steps", [])]}, indent=1))
    return 0 if res["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
