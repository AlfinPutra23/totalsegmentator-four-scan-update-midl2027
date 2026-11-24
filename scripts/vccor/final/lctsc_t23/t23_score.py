"""vccor/final LCTSC-T23 LOOK, part 2 of 2: SCORING and ANALYSIS (CPU, <= 6 workers, nice 15; SSL .venv; the same look).
DERIVED from the frozen lctsc_score.py by derive_t23.py (every change is listed there).

Requires (else refuses): look/GT_OPENED (from t23_look_labels.py) and no GT_ABORTED line; look/labels.json finished, with
a logged 'LABELS sha256' line equal to its hash; every label file equal to labels.json; the freeze unchanged
(t23_gates.freeze_ok); every arm map equal to look/maps.json; results.json absent (--continue reuses finished per-patient
records whose map and label hashes match).
Per patient: the reference = look/labels/<pid>.nii.gz (LCTSC's 5 ids, native CT grid); every arm's raw map (TS total ids)
-> LCTSC ids by lctsc_common.ts_total_lut(); final_common.Scorer (conventions v1; Dice, HD95, NSD 2 mm) ->
look/pervol/<pid>.json; t23_analysis.analyse -> look/results.json and one 'RESULTS sha256 ...' line.

usage:  CUDA_VISIBLE_DEVICES="" nice -n 15 .venv/bin/python -I -B scripts/vccor/final/lctsc_t23/t23_score.py [--continue]

Version of 2026-11-13; LCTSC-T23 derivation 2026-11-17.
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent / "lctsc"), str(HERE.parent), str(HERE.parent / "flaret50"),
           str(HERE.parent.parent / "m2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import lctsc_common as C  # noqa: E402
import t23_gates as G  # noqa: E402
import t23_log as L  # noqa: E402

MARKER = G.LOOK / "GT_OPENED"


def _fault(kind):
    """Rehearsal-only fault injection (LCTSC_T23_REHEARSAL_FAULTS 'kind:n'); None outside the rehearsal."""
    import os
    if not G.REHEARSAL:
        return None
    for f in os.environ.get("LCTSC_T23_REHEARSAL_FAULTS", "").split(";"):
        parts = f.split(":")
        if parts[0] == kind:
            return int(parts[1]) if len(parts) > 1 and parts[1] else 0
    return None
PERVOL = G.LOOK / "pervol"


def results_line_missing():
    return (G.LOOK / "results.json").exists() and not G.logged("RESULTS", r"K_lctsc_t23/(rehearsal_\d+/)?look/results\.json$")


def checks(cont):
    G.kst_gate()
    if (G.LOOK / "results.json").exists():
        raise G.Refuse("results.json exists: the look is finished")
    if not MARKER.exists():
        raise G.Refuse("no GT_OPENED marker: run t23_look_labels.py first")
    if any(" GT_ABORTED lctsc_t23 " in line for line in open(G.PREREG_LOG)):
        raise G.Refuse("the look was aborted (GT_ABORTED)")
    G.freeze_ok()
    lp = G.LOOK / "labels.json"
    lab = json.load(open(lp))
    if "finished" not in lab:
        raise G.Refuse("labels.json is not finished")
    hit = G.logged("LABELS", r"K_lctsc_t23/(rehearsal_\d+/)?look/labels\.json$")
    if not hit or hit[1] != G.sha256(lp):
        raise G.Refuse("labels.json differs from its LABELS line")
    for pid, r in lab["patients"].items():
        if G.sha256(G.LOOK / "labels" / f"{pid}.nii.gz") != r["label_sha256"]:
            raise G.Refuse(f"{pid}: label file changed since labels.json")
    maps = json.load(open(G.LOOK / "maps.json"))
    for a, per in maps["maps"].items():
        for pid, h in per.items():
            if G.sha256(G.pred_dir(a) / "raw" / f"{pid}.nii.gz") != h:
                raise G.Refuse(f"{a} {pid}: map changed since maps.json")
    return lab, maps


def score_one(job):
    pid, arms, lab_sha, map_sha = job
    import nibabel as nib
    import numpy as np
    import final_common as F
    out = PERVOL / f"{pid}.json"
    if out.is_file():
        r = json.load(open(out))
        if r.get("label_sha256") == lab_sha and r.get("map_sha256") == map_sha:
            return pid, "reused"
    li = nib.load(str(G.LOOK / "labels" / f"{pid}.nii.gz"))
    Gt = np.rint(np.asanyarray(li.dataobj)).astype(np.uint8)
    assert Gt.max() <= C.N_ORG
    sp = [float(z) for z in li.header.get_zooms()[:3]]
    sc = F.Scorer(Gt, C.N_ORG, sp)
    rec = sc.frame_record()
    lut = C.ts_total_lut()
    for a in arms:
        mi = nib.load(str(G.pred_dir(a) / "raw" / f"{pid}.nii.gz"))
        assert mi.shape[:3] == li.shape[:3] and np.allclose(mi.affine, li.affine, atol=1e-3), (pid, a, "grid")
        Y = lut[np.asarray(mi.dataobj).astype(np.uint8)]
        rec["arms"][a] = sc.score(Y)
    G.write_json(out, {"NATIVE": rec, "label_sha256": lab_sha, "map_sha256": map_sha, "written": G.now()})
    return pid, "scored"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--continue", dest="cont", action="store_true")
    a = ap.parse_args()
    if results_line_missing():                       # review N7: a crash between results.json and its line
        r = json.load(open(G.LOOK / "results.json"))
        L.prereg_line("RESULTS", G.LOOK / "results.json", f"lctsc_t23: P1_HF {r['P1_HF']['reading']} M {r['P1_HF']['M']:+.2f} "
                                                          f"(line re-appended after a crash)")
        return 0
    lab, maps = checks(a.cont)
    arms = maps["arms_done"]
    ids = G.test_ids()
    PERVOL.mkdir(parents=True, exist_ok=True)
    jobs = [(pid, arms, lab["patients"][pid]["label_sha256"], {x: maps["maps"][x][pid] for x in arms}) for pid in ids]
    with ProcessPoolExecutor(max_workers=6) as ex:
        for k, (pid, how) in enumerate(ex.map(score_one, jobs), start=1):
            L.event(f"[look-score] {pid} {how}")
            if _fault("score_crash_after") == k and not (G.LOOK / ".score_crashed_once").exists():
                (G.LOOK / ".score_crashed_once").touch()          # rehearsal only
                raise SystemExit(70)
    import t23_analysis as A
    R = {pid: {"NATIVE": json.load(open(PERVOL / f"{pid}.json"))["NATIVE"]} for pid in ids}
    missing = {pid: lab["patients"][pid].get("missing", []) for pid in ids}
    res = A.analyse(R, ids, arms, maps["not_run"], missing=missing)
    res["look"] = {"gt_opened": json.load(open(MARKER))["opened"], "labels_sha256": G.sha256(G.LOOK / "labels.json"),
                   "maps_sha256": G.sha256(G.LOOK / "maps.json"), "finished": G.now(),
                   "code_sha256": {f: G.sha256(HERE / f) for f in ("t23_score.py", "t23_analysis.py", "t23_gates.py",
                                                                    "t23_common.py", "t23_look_labels.py")}}
    G.write_json(G.LOOK / "results.json", res, mode=0o444)
    if _fault("crash_before_results_line") is not None and not (G.LOOK / ".results_crashed_once").exists():
        (G.LOOK / ".results_crashed_once").touch()           # rehearsal only (review N7)
        raise SystemExit(70)
    p1, s4 = res["P1_HF"], res["S_4S"]
    L.prereg_line("RESULTS", G.LOOK / "results.json",
                  f"lctsc_t23: P1_HF {p1['reading']} M {p1['M']:+.2f}; S_4S {s4['reading']} M {s4['M']:+.2f}; "
                  f"per-draw P1_HF {res['P1_HF_combination_descriptive']['combination']} (descriptive); organs "
                  f"{res['organ_costs']['reading']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
