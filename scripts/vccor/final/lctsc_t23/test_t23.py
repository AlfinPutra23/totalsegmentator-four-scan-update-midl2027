"""vccor/final LCTSC-T23 unit tests (CPU; synthetic records; reads only LCTSC / LCTSC-T23 JSON records and code).

  t1 the 23 patients: ids_t23.json == the derivation rule on the frozen LCTSC records; 23 = 8 / 7 / 8; none was read before,
     none is in a draw, none is a test patient; the 4 stand-ins were all read before (site-draw patients)
  t2 derivation: the six derived scripts still equal derived.json (no edit after the derivation) and their LCTSC sources
     still equal the LCTSC freeze
  t3 analysis on synthetic records: heart +10 and the other structures +2 -> P1_HF GAIN (M ~ 2), S_4S GAIN (M ~ 4),
     heart share ~ 0.625; heart +10 only -> P1_HF NO_GAIN while S_4S GAINs (the primary excludes the heart);
     need_up = 16 of 23
  t4 missing-structure rule: an esophagus missing from one reference is not evaluable (22 patients for its organ cost;
     the patient's heart-free macro uses the lungs); a NOT RUN draw leaves two draws
  t5 gates: cutoff, stop and public-deadline constants equal t23_common; DONE at / after the cutoff is LATE; the deposit
     pattern matches the LCTSC-T23 deposit only; a NOTRUN line binds only with the lctsc_t23 tag
  t6 refusals: predict smoke refuses an LCTSC-T23 id and the real image folder; the look's series_of refuses a patient
     outside the 23 (no file is opened)
  t7 deposit layout (review B1): exactly one member named code_freeze.json (the T23 freeze); no image, label or model file
  t8 (review N12) plan dates and patients equal the code; d6ayj-format log lines never satisfy an LCTSC-T23 check
Refuses to rewrite unit_tests.json after the T23 freeze (review S4).

usage:  .venv/bin/python -I -B scripts/vccor/final/lctsc_t23/test_t23.py   (writes K_lctsc_t23/unit_tests.json)

Version of 2026-11-17.
"""
import json
import re
import sys
import tempfile
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
for _p in (str(HERE.parent.parent / "tsft_full"), str(HERE.parent.parent / "m2"), str(HERE.parent / "flaret50"),
           str(HERE.parent), str(HERE.parent / "lctsc"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import lctsc_common as C  # noqa: E402
import t23_common as T  # noqa: E402
import t23_gates as G  # noqa: E402
import t23_log as L  # noqa: E402


def t1():
    ids = T.ids()
    assert ids == T.derive_ids() and len(ids) == 23, len(ids)
    assert {s: sum(T.site(p) == s for p in ids) for s in ("S1", "S2", "S3")} == {"S1": 8, "S2": 7, "S3": 8}
    rb = T.read_before()
    assert len(rb) == 13 and not (set(ids) & rb)
    draws = {p for v in json.load(open(T.SRC / "draws_site.json"))["draws"].values() for p in v}
    assert not (set(ids) & draws) and not any(C.is_test(p) for p in ids)
    assert set(T.STANDINS) <= rb and set(T.STANDINS) <= draws


def t2():
    der = json.load(open(HERE / "derived.json"))
    for f, h in der["outputs"].items():
        assert L.sha256(HERE / f) == h, f"{f} edited after the derivation"
    fz = json.load(open(T.SRC / "code_freeze.json"))["files"]
    for f, h in der["sources"].items():
        assert fz[f"scripts/vccor/final/lctsc/{f}"] == h == L.sha256(HERE.parent / "lctsc" / f), f


def synth(gain_heart, gain_other, n=23, seed=0, eso_missing_for=None):
    import numpy as np
    rng = np.random.RandomState(seed)
    ids = T.ids()[:n]
    R = {}
    for c in ids:
        ref = {"present": [True] * 5, "ml": [50.0, 600.0, 2500.0, 2800.0, 80.0], "vox": [1] * 5}
        base = np.array([75.0, 80.0, 95.0, 95.0, 85.0]) + rng.normal(0, 1, 5)
        arms = {}
        for a in ("F0", "F1b_S1", "F1b_S2", "F1b_S3"):
            v = base.copy()
            if a != "F0":
                v[[0, 2, 3]] += gain_other + rng.normal(0, 0.3, 3)
                v[1] += gain_heart + rng.normal(0, 0.3)
            arms[a] = {"dice": v.tolist(), "present": [True] * 5, "status": ["TP"] * 5, "ml": [50.0, 600.0, 2500.0, 2800.0, 80.0],
                       "hd95": [3.0] * 5, "nsd": [90.0] * 5, "vox": [1] * 5, "inter": [1] * 5}
        R[c] = {"NATIVE": {"ref": ref, "arms": arms, "shape": [1, 1, 1], "spacing": [1, 1, 1], "vox_ml": 1e-3, "diag_mm": 1.0}}
    missing = {eso_missing_for: ["esophagus"]} if eso_missing_for else None
    return R, ids, missing


HAVE = ["F0", "F1b_S1", "F1b_S2", "F1b_S3"]


def t3():
    import t23_analysis as A
    R, ids, _ = synth(10.0, 2.0)
    r = A.analyse(R, ids, HAVE)
    assert r["P1_HF"]["reading"] == "GAIN" and abs(r["P1_HF"]["M"] - 2.0) < 0.3, r["P1_HF"]["M"]
    assert r["S_4S"]["reading"] == "GAIN" and abs(r["S_4S"]["M"] - 4.0) < 0.3, r["S_4S"]["M"]
    assert abs(r["heart_share_descriptive"]["share"] - 0.625) < 0.05, r["heart_share_descriptive"]
    assert r["P1_HF"]["need_up"] == 16 and r["P1_HF"]["n_scans"] == 23
    assert r["P1_HF_combination_descriptive"]["combination"] == "GAIN_EVERY_DRAW"
    assert not r["claims_allowed"]["external or independent validation"]
    R, ids, _ = synth(10.0, 0.0, seed=1)
    r = A.analyse(R, ids, HAVE)
    assert r["P1_HF"]["reading"] == "NO_GAIN" and r["S_4S"]["reading"] == "GAIN", (r["P1_HF"]["M"], r["S_4S"]["M"])
    assert abs(r["spinal_cord_descriptive"]["F0"] - r["spinal_cord_descriptive"]["F1b_S1"]) < 1e-9


def t4():
    import t23_analysis as A
    pid = T.ids()[0]
    R, ids, miss = synth(10.0, 2.0, seed=2, eso_missing_for=pid)
    r = A.analyse(R, ids, HAVE, missing=miss)
    assert r["organ_costs"]["per_organ"]["esophagus"]["n_scans"] == 22
    assert r["not_evaluable_structures"] == {pid: ["esophagus"]}
    r2 = A.analyse(R, ids, ["F0", "F1b_S1", "F1b_S3"], {"F1b_S2": "declared NOT RUN"}, missing=miss)
    assert r2["draws_run"] == ["S1", "S3"] and r2["P1_HF_combination_descriptive"]["k_of_n_run"].endswith("of 2")
    import numpy as np
    sv = r["P1_HF"]["M"]
    assert np.isfinite(sv)


def t5():
    assert (G.CUTOFF, G.STOP, G.PUBLIC_DEADLINE) == (T.CUTOFF, T.STOP, T.PUBLIC_DEADLINE)
    assert G.before_cutoff("2026-11-20 09:59:59") and not G.before_cutoff("2026-11-20 10:00:00")
    assert G.before_stop("2026-11-20 15:59:59") and not G.before_stop("2026-11-20 16:00:00")
    with tempfile.TemporaryDirectory() as td:
        for arm, t in (("F0", "2026-11-20 09:00:00"), ("F1b_S1", "2026-11-20 10:00:00")):
            (Path(td) / arm).mkdir()
            json.dump({"finished": t}, open(Path(td) / arm / "DONE.json", "w"))
        assert G.arm_done("F0", td)[0] == "DONE" and G.arm_done("F1b_S1", td)[0] == "LATE"
        assert G.arm_done("F1b_S2", td)[0] == "MISSING"
        old = G.PREREG_LOG
        try:
            G.PREREG_LOG = Path(td) / "prereg.log"
            G.PREREG_LOG.write_text("[2026-11-17 10:00:00] NOTRUN lctsc F1b_S3 old LCTSC line\n")
            assert not G.notrun_declared("F1b_S3")
            G.PREREG_LOG.write_text("[2026-11-17 10:00:00] NOTRUN lctsc_t23 F1b_S3 2 failed attempts\n")
            assert G.notrun_declared("F1b_S3")
        finally:
            G.PREREG_LOG = old
    assert re.search(G.DEPOSIT_RX, "outputs/results/vccor/final/K_lctsc_t23/osf_deposit_lctsc_t23_2026-11-17.zip")
    assert not re.search(G.DEPOSIT_RX, "outputs/results/vccor/final/K_lctsc/osf_deposit_lctsc_2026-11-13.zip")
    assert G.MODELS == T.SRC / "models" and G.OUT == T.OUT


def t6():
    import t23_predict as P
    for argv in (["--arm", "F0", "--smoke", "--root", "/tmp/x", "--images", "/tmp/y", "--ids", T.ids()[0]],
                 ["--arm", "F0", "--smoke", "--root", "/tmp/x", "--images", str(C.IMAGES), "--ids", T.STANDINS[0]]):
        try:
            P.parse(argv)
            raise AssertionError(f"predict smoke not refused: {argv}")
        except AssertionError as e:
            if "not refused" in str(e):
                raise
    import t23_look_labels as LL
    for pid in ("LCTSC-Test-S1-101", T.STANDINS[0]):
        try:
            LL.series_of(pid)
            raise AssertionError(f"series_of not refused: {pid}")
        except AssertionError as e:
            if "not refused" in str(e):
                raise
    assert LL.series_of(T.ids()[0])["Modality"] == "RTSTRUCT"


def t7():
    """review B1: the deposit holds exactly one member named code_freeze.json (the T23 freeze), which is what the upload
    gate's predicate requires; no image, label or model file is deposited."""
    import t23_freeze as Z
    dep = Z.deposit_files(Z.frozen_files())
    names = [Z.deposit_member(f) for f in dep]
    hits = [n for n in names if n.endswith("code_freeze.json")]
    assert hits == [Z.deposit_member(G.FREEZE)], hits
    assert len(set(names)) == len(names)
    bad = [n for n in names if n.lower().endswith((".nii", ".nii.gz", ".gz", ".pth", ".zip", ".tar", ".npz", ".npy", ".dcm"))]
    assert not bad, bad
    assert Z.LCTSC_FREEZE in Z.frozen_files() and Z.LCTSC_FREEZE not in dep


def t8():
    """review N12: dates and patients of the plan equal the code; d6ayj-format log lines never satisfy an LCTSC-T23 check."""
    assert T.CUTOFF_DT == G.ts(T.CUTOFF)
    plan = json.load(open(T.PLAN))
    assert plan["dates_kst"] == {"plan_freeze_deadline": T.PLAN_FREEZE_DEADLINE, "public_deadline": T.PUBLIC_DEADLINE,
                                 "cutoff": T.CUTOFF, "stop_rule": T.STOP}
    assert plan["data"]["patients"] == T.ids()
    import t23_look_labels as LL
    old = G.PREREG_LOG
    with tempfile.TemporaryDirectory() as td:
        try:
            G.PREREG_LOG = Path(td) / "prereg.log"
            h = "0" * 64
            G.PREREG_LOG.write_text(
                f"[2026-11-16 11:27:57] GT_OPENED sha256 {h} outputs/results/vccor/final/K_lctsc/look/GT_OPENED (lctsc look)\n"
                f"[2026-11-16 11:30:41] LABELS sha256 {h} outputs/results/vccor/final/K_lctsc/look/labels.json (lctsc look)\n"
                f"[2026-11-16 11:31:49] RESULTS sha256 {h} outputs/results/vccor/final/K_lctsc/look/results.json (lctsc)\n"
                f"[2026-11-13 18:16:26] DEPOSIT sha256 {h} outputs/results/vccor/final/K_lctsc/osf_deposit_lctsc_2026-11-13.zip (x)\n"
                "[2026-11-16 11:00:00] GT_ABORTED lctsc test RTSTRUCT x\n[2026-11-16 11:00:00] NOTE lctsc look not completed x\n")
            for kind, rx in (("GT_OPENED", r"K_lctsc_t23/(rehearsal_\d+/)?look/GT_OPENED$"),
                             ("LABELS", r"K_lctsc_t23/(rehearsal_\d+/)?look/labels\.json$"),
                             ("RESULTS", r"K_lctsc_t23/(rehearsal_\d+/)?look/results\.json$"), ("DEPOSIT", G.DEPOSIT_RX)):
                assert G.logged(kind, rx) is None, kind
            assert not LL.aborted() and not LL.not_completed()
        finally:
            G.PREREG_LOG = old


def main():
    res = {}
    for name in ("t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"):
        try:
            globals()[name]()
            res[name] = "PASS"
        except Exception:  # noqa: BLE001
            res[name] = "FAIL: " + traceback.format_exc(limit=3)
        print(name, res[name].splitlines()[0], flush=True)
    code = {f.name: L.sha256(f) for f in sorted(HERE.glob("*.py")) + sorted(HERE.glob("*.service"))}
    out = {"results": res, "all_pass": all(v == "PASS" for v in res.values()), "code_sha256": code, "written": L.now()}
    if (T.OUT / "code_freeze.json").exists():            # review S4: a frozen record is never rewritten
        print("an LCTSC-T23 freeze exists: results printed only, unit_tests.json not rewritten")
        print("all_pass", out["all_pass"])
        return 0 if out["all_pass"] else 1
    L.write_json(T.OUT / "unit_tests.json", out)
    print("all_pass", out["all_pass"])
    return 0 if out["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
