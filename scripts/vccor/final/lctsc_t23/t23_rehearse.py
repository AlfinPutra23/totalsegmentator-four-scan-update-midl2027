"""vccor/final LCTSC-T23 LOOK REHEARSAL (CPU; before the freeze). ADAPTED from the frozen lctsc_rehearse.py: the whole look
-- t23_look_labels.py (external/lctsc_env) then t23_score.py (SSL .venv) -- runs unchanged under scratch roots
(LCTSC_T23_REHEARSAL_ROOT = K_lctsc_t23/rehearsal_<n>/) on the 4 STAND-IN patients of t23_common.STANDINS: site-draw TRAINING
patients whose labels were read on 2026-11-13 for training. No LCTSC-T23 or test label is touched. Each root has its own
prereg.log, a stand-in freeze inside a stand-in deposit zip, stand-in upload / public gate files, standin.json and
pred/<arm>/ (F0 = TS's own merged output from the LCTSC prep capture; F1b_S1 = F0; F1b_S2 = F0 without the esophagus on
every second slice; F1b_S3 = F0), so the real gate code runs. The stand-ins' CT DICOM is the 2026-11-17 re-download
(t23_ct_fetch.py), so check a1 also tests that the re-downloaded geometry reproduces the 2026-11-13 training labels.

Scenarios (each in its own root):
  A  main path with faults: a structure dropped from one stand-in reference (missing-ROI rule); a crash of the label step
     after 2 patients, then --continue; a crash of scoring after 3 patients, then a rerun; a crash between results.json
     and its RESULTS line, then a rerun. Checks:
       a1 every converted label equals the training label v2 of 2026-11-13 (same rasteriser) except the dropped structure;
       a2 every pervol Dice equals an independent numpy Dice of the stored maps vs the labels;
       a3 F1b_S1 / F1b_S3 deltas are exactly 0 (heart-free and 4-structure); F1b_S2's esophagus change is < 0; the dropped
          structure is not evaluable;
       a4 GT_OPENED, LABELS and RESULTS lines exist once each; a second look and a second scoring are refused.
  B  a stored zip that differs from its download record (fault 'mismatch'): GT_ABORTED is logged; --continue and scoring
     are refused.
  C  an F1b arm without DONE: before the cutoff the look is refused; with the rehearsal clock after the cutoff (and before
     the stop rule) the look runs with that arm 'not DONE by the cutoff' and the readings on the two remaining draws.
  D  every arm DONE but a binding NOTRUN line for F1b_S3: the NOTRUN line wins; readings on S1 and S2.
  F  a public gate checked after the public deadline (Thu 2026-11-19 12:00 KST): the look is refused.
Writes K_lctsc_t23/rehearsal.json (all_pass, code_sha256 of every *.py and *.service).

usage:  .venv/bin/python -I -B scripts/vccor/final/lctsc_t23/t23_rehearse.py

Version of 2026-11-17.
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
for _p in (str(HERE.parent.parent / "tsft_full"), str(HERE.parent.parent / "m2"), str(HERE.parent / "flaret50"),
           str(HERE.parent), str(HERE.parent / "lctsc"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import lctsc_common as C  # noqa: E402  (frozen)
import lctsc_runs as R  # noqa: E402  (frozen: PREP)
import t23_common as T  # noqa: E402
import t23_log as L  # noqa: E402

VENV = "/home/user/Documents/Medical Imagining SSL/.venv/bin/python"
LENV = "/home/user/Documents/Medical Imagining SSL/external/lctsc_env/bin/python"
ARMS = {"F0": None, "F1b_S1": "same", "F1b_S2": "eso_half", "F1b_S3": "same"}


def tm(minutes):
    return (datetime.now() - timedelta(minutes=minutes)).strftime("%Y-%m-%d %H:%M:%S")


def new_root():
    n = 1
    while (T.OUT / f"rehearsal_{n}").exists():
        n += 1
    root = T.OUT / f"rehearsal_{n}"
    root.mkdir()
    return root


D6AYJ_LINES = [   # review N13: d6ayj-format lines in every rehearsal log; none may satisfy or block an LCTSC-T23 check
    "[2026-11-13 18:16:26] DEPOSIT sha256 {h} outputs/results/vccor/final/K_lctsc/osf_deposit_lctsc_2026-11-13.zip (d6ayj)",
    "[2026-11-16 11:27:57] GT_OPENED sha256 {h} outputs/results/vccor/final/K_lctsc/look/GT_OPENED (lctsc look)",
    "[2026-11-16 11:30:41] LABELS sha256 {h} outputs/results/vccor/final/K_lctsc/look/labels.json (lctsc look)",
    "[2026-11-16 11:31:49] RESULTS sha256 {h} outputs/results/vccor/final/K_lctsc/look/results.json (lctsc)",
    "[2026-11-16 11:40:00] NOTRUN lctsc F1b_S3 a d6ayj-format line (rehearsal): must not bind LCTSC-T23",
    "[2026-11-16 11:41:00] GT_ABORTED lctsc a d6ayj-format line (rehearsal): must not abort LCTSC-T23"]


def standin_deposit(root):
    """review B1 / N13: the stand-in deposit has the REAL deposit layout (t23_freeze.deposit_files of the existing frozen
    files) with the root's stand-in freeze in place of the T23 freeze, so the real upload-gate predicate runs on it."""
    import t23_freeze as Z
    real_freeze = T.OUT / "code_freeze.json"
    members = [(f, Z.deposit_member(f)) for f in Z.deposit_files([p for p in Z.frozen_files() if Path(p).is_file()])
               if Path(f) != real_freeze]
    members.append((root / "code_freeze.json", "lctsc_t23_deposit/" + L.rel(root / "code_freeze.json")))
    dz = root / "rehearsal_lctsc_t23_deposit.zip"
    with zipfile.ZipFile(dz, "w", zipfile.ZIP_DEFLATED) as z:
        for f, name in members:
            z.write(f, name)
    return dz


def setup(root, ids, arms=ARMS, public_checked=None):
    import nibabel as nib
    import numpy as np
    with open(root / "prereg.log", "a") as f:
        for ln in D6AYJ_LINES:
            f.write(ln.format(h="0" * 64) + "\n")
    json.dump({"ids": ids, "note": "stand-in TRAINING site-draw patients (labels read on 2026-11-13)"}, open(root / "standin.json", "w"))
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
    lfz = json.load(open(T.SRC / "code_freeze.json"))
    fz = {"frozen_kst": tm(50), "files": {"scripts/vccor/final/lctsc_t23/t23_common.py": L.sha256(HERE / "t23_common.py")},
          "models": lfz["models"], "draws_status": lfz["draws_status"]}
    json.dump(fz, open(root / "code_freeze.json", "w"))
    dz = standin_deposit(root)
    with open(root / "prereg.log", "a") as f:
        f.write(f"[{tm(49)}] DEPOSIT sha256 {L.sha256(dz)} {L.rel(dz)} (rehearsal stand-in)\n")
    json.dump({"osf_url": "https://osf.io/rehearsal", "uploaded_kst": tm(40), "written_by": "rehearsal", "evidence": "stand-in",
               "deposit_zip_sha256": L.sha256(dz), "freeze_sha256": L.sha256(root / "code_freeze.json")},
              open(root / "gate_osf_uploaded.json", "w"))
    json.dump({"osf_url": "https://osf.io/rehearsal", "public_checked_kst": public_checked or tm(30), "api_http_status": 200,
               "public": True, "written_by": "rehearsal", "evidence": "stand-in"}, open(root / "gate_osf_public.json", "w"))


def run(py, script, root, faults="", now=None, args=()):
    env = dict(os.environ, LCTSC_T23_REHEARSAL_ROOT=str(root), CUDA_VISIBLE_DEVICES="", LCTSC_T23_REHEARSAL_FAULTS=faults)
    if now:
        env["LCTSC_T23_REHEARSAL_NOW"] = now
    r = subprocess.run([py, "-I", "-B", str(HERE / script), *args], env=env, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr)[-800:]


def count(root, kind):
    return sum(1 for line in open(root / "prereg.log") if f" {kind} " in line)


def scenario_a(ids):
    import nibabel as nib
    import numpy as np
    import lctsc_analysis as LA
    root = new_root()
    setup(root, ids)
    faults = f"drop_roi:{ids[1]}:esophagus;crash_after:2;score_crash_after:3;crash_before_results_line"
    out = {"root": L.rel(root), "steps": []}
    rcq, tq = run(VENV, "t23_predict_queue.py", root, args=("--check",))      # review N13: the runner's own gates
    out["queue_check"] = {"rc": rcq, "tail": tq[-300:]}
    for script, py, args in (("t23_look_labels.py", LENV, ()), ("t23_look_labels.py", LENV, ("--continue",)),
                             ("t23_score.py", VENV, ()), ("t23_score.py", VENV, ()), ("t23_score.py", VENV, ())):
        rc, tail = run(py, script, root, faults, args=args)
        out["steps"].append({"script": script, "args": list(args), "rc": rc, "tail": tail[-300:]})
    out["rc_as_expected"] = [s["rc"] for s in out["steps"]] == [70, 0, 70, 70, 0]
    chk = {}
    lab = json.load(open(root / "look/labels.json"))
    same, diff = True, {}
    for p in ids:
        a = np.asarray(nib.load(str(root / "look/labels" / f"{p}.nii.gz")).dataobj)
        b = np.asarray(nib.load(str(C.train_label_path(p))).dataobj).copy()
        if p == ids[1]:
            b[b == 1] = 0
        eq = bool(np.array_equal(a, b))
        diff[p] = int((a != b).sum()) if a.shape == b.shape else "shape"
        same &= eq
    chk["a1_labels_equal_training_v2"] = bool(same and lab["patients"][ids[1]]["missing"] == ["esophagus"])
    out["a1_voxels_differing"] = diff
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
    RR = LA.mask_missing({p: {"NATIVE": json.load(open(root / "look/pervol" / f"{p}.json"))["NATIVE"]} for p in ids},
                         {ids[1]: ["esophagus"]})
    z = all(abs(v) < 1e-12 for d in ("S1", "S3") for org in ([1, 3, 4], [1, 2, 3, 4])
            for v in LA.md(RR, f"F1b_{d}", "F0", ids, organs=org).values())
    eso = res["organ_costs"]["per_organ"]["esophagus"]
    chk["a3_known_changes_and_missing_rule"] = bool(z and eso["M"] is not None and eso["M"] < 0
                                                    and eso["n_scans"] == len(ids) - 1
                                                    and res["not_evaluable_structures"] == {ids[1]: ["esophagus"]}
                                                    and res["P1_HF"]["n_scans"] == len(ids))
    rc1, _ = run(LENV, "t23_look_labels.py", root)
    rc2, _ = run(VENV, "t23_score.py", root)
    chk["a4_lines_once_and_second_look_refused"] = bool(count(root, "GT_OPENED") == 2 and count(root, "LABELS") == 2
                                                        and count(root, "RESULTS") == 2 and rc1 != 0 and rc2 != 0)
    # (each count includes the one preseeded d6ayj-format line of D6AYJ_LINES)
    chk["a0_queue_check_passes_on_real_deposit_layout"] = bool(rcq == 0)
    chk["a5_d6ayj_lines_do_not_bind"] = bool(res["draws_run"] == ["S1", "S2", "S3"] and not res["arms_not_run"])
    out["checks"] = chk
    out["readings_on_standins"] = {"P1_HF": res["P1_HF"]["reading"], "S_4S": res["S_4S"]["reading"]}
    return out


def scenario_b(ids):
    root = new_root()
    setup(root, ids[:3])
    rc1, t1 = run(LENV, "t23_look_labels.py", root, faults=f"mismatch:{ids[1]}")
    rc2, t2 = run(LENV, "t23_look_labels.py", root, args=("--continue",))
    rc3, t3 = run(VENV, "t23_score.py", root)
    ok = rc1 != 0 and count(root, "GT_ABORTED") == 2 and rc2 != 0 and rc3 != 0     # 1 preseeded d6ayj-format line + 1
    return {"root": L.rel(root), "rc": [rc1, rc2, rc3], "checks": {"b1_record_mismatch_aborts_for_good": bool(ok)},
            "tails": [t1[-200:], t2[-200:], t3[-200:]]}


def scenario_c(ids):
    import shutil
    root = new_root()
    setup(root, ids[:4])
    shutil.move(str(root / "pred" / "F1b_S3"), str(root / "F1b_S3_never_done"))
    rc1, t1 = run(LENV, "t23_look_labels.py", root, args=("--check",))
    after = "2026-11-20 10:30:00"
    rc2, t2 = run(LENV, "t23_look_labels.py", root, now=after)
    rc3, t3 = run(VENV, "t23_score.py", root, now=after)
    ok = rc1 != 0 and "before the cutoff every arm must be DONE" in t1 and rc2 == 0 and rc3 == 0
    if ok:
        res = json.load(open(root / "look/results.json"))
        ok = res["draws_run"] == ["S1", "S2"] and res["arms_not_run"].get("F1b_S3") == "not DONE by the cutoff"
    return {"root": L.rel(root), "rc": [rc1, rc2, rc3], "checks": {"c1_missing_arm_refused_before_cutoff_not_run_after": bool(ok)},
            "tails": [t1[-200:], t2[-200:], t3[-200:]]}


def scenario_d(ids):
    root = new_root()
    setup(root, ids[:4])
    with open(root / "prereg.log", "a") as f:
        f.write(f"[{tm(5)}] NOTRUN lctsc_t23 F1b_S3 2 failed attempts (rehearsal)\n")
    rc1, t1 = run(LENV, "t23_look_labels.py", root)
    rc2, t2 = run(VENV, "t23_score.py", root)
    ok = rc1 == 0 and rc2 == 0
    if ok:
        res = json.load(open(root / "look/results.json"))
        ok = res["draws_run"] == ["S1", "S2"] and res["arms_not_run"].get("F1b_S3", "").startswith("declared NOT RUN")
    return {"root": L.rel(root), "rc": [rc1, rc2], "checks": {"d1_notrun_line_beats_done": bool(ok)},
            "tails": [t1[-200:], t2[-200:]]}


def scenario_f(ids):
    root = new_root()
    setup(root, ids[:2], public_checked="2026-11-19 12:00:01")
    rc1, t1 = run(LENV, "t23_look_labels.py", root, args=("--check",))
    ok = rc1 != 0 and "public deadline" in t1 and not (root / "look" / "GT_OPENED").exists()
    return {"root": L.rel(root), "rc": [rc1], "checks": {"f1_public_after_deadline_refused": bool(ok)}, "tails": [t1[-200:]]}


def scenario_g(ids):
    """review N13: a new look at or after the stop rule (Fri 2026-11-20 16:00 KST) is refused; no GT_OPENED."""
    root = new_root()
    setup(root, ids[:2])
    rc1, t1 = run(LENV, "t23_look_labels.py", root, now="2026-11-20 16:00:01")
    ok = rc1 != 0 and "stop rule passed" in t1 and not (root / "look" / "GT_OPENED").exists()
    return {"root": L.rel(root), "rc": [rc1], "checks": {"g1_new_look_after_stop_refused": bool(ok)}, "tails": [t1[-200:]]}


def scenario_h(ids):
    """review S1: a stored zip whose size differs from its download record is refused BEFORE GT_OPENED (metadata only)."""
    root = new_root()
    setup(root, ids[:2])
    rc1, t1 = run(LENV, "t23_look_labels.py", root, faults=f"precheck_size:{ids[0]}")
    ok = (rc1 != 0 and "size differs" in t1 and not (root / "look" / "GT_OPENED").exists()
          and count(root, "GT_ABORTED") == 1)            # only the preseeded d6ayj-format line
    return {"root": L.rel(root), "rc": [rc1], "checks": {"h1_zip_size_refused_before_gt_opened": bool(ok)}, "tails": [t1[-200:]]}


def main():
    assert "LCTSC_T23_REHEARSAL_ROOT" not in os.environ
    if (T.OUT / "code_freeze.json").exists():            # review S4
        raise SystemExit("an LCTSC-T23 freeze exists: no new rehearsal record")
    ids = list(T.STANDINS)
    res = {"ids": ids, "scenarios": {}}
    for name, fn in (("A", scenario_a), ("B", scenario_b), ("C", scenario_c), ("D", scenario_d), ("F", scenario_f),
                     ("G", scenario_g), ("H", scenario_h)):
        try:
            res["scenarios"][name] = fn(ids)
        except Exception as e:  # noqa: BLE001
            res["scenarios"][name] = {"error": f"{type(e).__name__}: {e}", "checks": {f"{name}_ran": False}}
    checks = {k: v for s in res["scenarios"].values() for k, v in s.get("checks", {}).items()}
    res["checks"] = checks
    res["all_pass"] = bool(checks) and all(checks.values()) and bool(res["scenarios"]["A"].get("rc_as_expected", False))
    res["code_sha256"] = {f.name: L.sha256(f) for f in sorted(HERE.glob("*.py")) + sorted(HERE.glob("*.service"))}
    res["written"] = L.now()
    L.write_json(T.OUT / "rehearsal.json", res)
    print(json.dumps({"all_pass": res["all_pass"], "checks": checks,
                      "A_rc": [s["rc"] for s in res["scenarios"]["A"].get("steps", [])],
                      "A_voxels_differing": res["scenarios"]["A"].get("a1_voxels_differing")}, indent=1))
    return 0 if res["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
