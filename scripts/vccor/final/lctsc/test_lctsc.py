"""vccor/final LCTSC unit tests (CPU, synthetic data only; no LCTSC file is read except the TCIA manifest and draws.json).

  u1 draws: the three draws equal draws.json and the project rule; no draw holds a test patient
  u2 TS mapping: every LCTSC structure maps from the intended TS classes (total and part LUTs agree)
  u3 composite 291: GT lungs split into TS lobes (own lobe kept, others to the nearest lobe of the same side); esophagus
     relabelled; TS false-positive lobe / esophagus voxels removed; other TS classes kept; no adapted GT voxel left at 0
  u4 composite 293: GT heart -> TS heart; TS atrial appendage inside GT heart kept; TS heart outside GT heart removed
  u5 analysis on synthetic records: a clear gain passes P1 with GAIN_EVERY_DRAW; a null fails; an organ with a large loss
     reads INFERIOR; claims follow the readings
  u6 gates: cutoff / stop comparisons; arm_done marks a DONE.json written at or after the cutoff as LATE
  u7 refusals: the download script refuses a test RTSTRUCT; predict smoke refuses a test id and the real image folder

usage:  .venv/bin/python -I -B scripts/vccor/final/lctsc/test_lctsc.py   (writes K_lctsc/unit_tests.json)
"""
import json
import sys
import tempfile
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent), str(HERE.parent / "flaret50"), str(HERE.parent.parent / "m2"),
           str(HERE.parent.parent / "tsft_full")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import lctsc_common as C  # noqa: E402
import lctsc_gates as G  # noqa: E402
import lctsc_log as L  # noqa: E402


def u1():
    d = json.load(open(C.DRAWS_SITE))
    for k in C.DRAW_SEEDS:
        assert d["draws"][k] == C.draw(k), k
        assert not any(C.is_test(p) for p in C.draw(k))
        assert len(C.draw(k)) == 4 and all(C.site(p) == k for p in C.draw(k)), k
        assert C.site_draw_rule(k, d["already_read_training_labels_at_fixing"]) == C.draw(k), "rule reproduces the file"
    assert len(C.patients("train")) == 36 and len(C.patients("test")) == 24
    assert {s: sum(C.site(p) == s for p in C.patients("test")) for s in C.DRAW_SEEDS} == {"S1": 8, "S2": 8, "S3": 8}
    old = json.load(open(C.DRAWS))                    # the superseded pooled draws stay reproducible
    assert all(old["draws"][k] == C.draw_pooled(k) for k in C.DRAW_SEEDS_POOLED)


def u2():
    import numpy as np
    from totalsegmentator.map_to_binary import class_map
    inv = {v: k for k, v in class_map["total"].items()}
    tl = C.ts_total_lut()
    for o, names in C.TS_NAMES.items():
        for n in names:
            assert tl[inv[n]] == o, n
    assert int((tl > 0).sum()) == sum(len(v) for v in C.TS_NAMES.values())
    for t in (291, 293, 294):
        pl = C.part_lut(t)
        assert set(np.unique(pl)) - {0} <= set(C.TS_NAMES)
    assert C.part_lut(291)[C.part_class(291, "lung_middle_lobe_right")] == 4
    assert C.part_lut(293)[C.part_class(293, "atrial_appendage_left")] == 2


def u3():
    import numpy as np
    ul, ll = C.part_class(291, "lung_upper_lobe_left"), C.part_class(291, "lung_lower_lobe_left")
    ur, mr, lr = (C.part_class(291, n) for n in ("lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"))
    eso, tra = C.part_class(291, "esophagus"), C.part_class(291, "trachea")
    G_ = np.zeros((20, 20, 20), np.uint8)
    P = np.zeros_like(G_)
    G_[2:8, 2:18, 2:18] = 3            # GT left lung block
    P[2:8, 2:10, 2:18] = ul            # TS upper left lobe covers half of it
    P[2:8, 10:12, 2:18] = ll           # TS lower left lobe a strip
    G_[12:18, 2:18, 2:18] = 4          # GT right lung, TS has only a middle lobe strip
    P[12:18, 2:4, 2:18] = mr
    G_[9:11, 9:11, 2:18] = 1           # GT esophagus; TS calls nothing there
    P[9:11, 2:4, 2:18] = eso           # TS false-positive esophagus elsewhere
    P[0, 0, 0:5] = tra                 # TS trachea in GT background: kept
    P[19, 0, 0] = ll                   # TS false-positive left lower lobe outside the GT (far away): removed
    Y, rec = C.composite(291, G_, P)
    assert (Y[2:8, 2:10, 2:18] == ul).all()
    lungL = (G_ == 3)
    assert set(np.unique(Y[lungL])) <= {ul, ll}
    assert (Y[2:8, 12:18, 2:18] == ll).all(), "voxels nearest to the lower-lobe strip take the lower lobe"
    assert (Y[G_ == 4] == mr).all(), "the only right lobe TS calls"
    assert (Y[9:11, 9:11, 2:18] == eso).all()
    assert (Y[9:11, 2:4, 2:18] == 0).all(), "TS esophagus outside the GT is a false positive"
    assert (Y[0, 0, 0:5] == tra).all()
    assert Y[19, 0, 0] == 0
    assert not ((G_ > 0) & (Y == 0)).any()
    assert rec["lung_split"]["lung_right"]["rule"] == "nearest TS lobe"


def u4():
    import numpy as np
    h, aa, ao = C.part_class(293, "heart"), C.part_class(293, "atrial_appendage_left"), C.part_class(293, "aorta")
    G_ = np.zeros((10, 10, 10), np.uint8)
    P = np.zeros_like(G_)
    G_[2:8, 2:8, 2:8] = 2
    P[2:5, 2:8, 2:8] = h
    P[5:6, 2:8, 2:8] = aa
    P[8:10, 2:8, 2:8] = h              # TS heart outside the GT heart
    P[0, 0, :] = ao
    G_[0:2, 0:2, 0:2] = 3              # a GT lung voxel: not part 293's business
    Y, _ = C.composite(293, G_, P)
    assert (Y[5:6, 2:8, 2:8] == aa).all()
    assert (Y[6:8, 2:8, 2:8] == h).all()
    assert (Y[8:10, 2:8, 2:8] == 0).all()
    assert (Y[0, 0, 2:] == ao).all()
    assert not ((G_ == 2) & (Y == 0)).any()


def synth(gain, loss_organ=None, n=24, seed=0):
    import numpy as np
    rng = np.random.RandomState(seed)
    ids = C.patients("test")[:n]
    R = {}
    for c in ids:
        ref = {"present": [True] * 5, "ml": [50.0, 600.0, 2500.0, 2800.0, 80.0], "vox": [1] * 5}
        arms = {}
        base = np.array([75.0, 88.0, 97.0, 97.0, 85.0]) + rng.normal(0, 1, 5)
        for a in ("F0", "F1b_S1", "F1b_S2", "F1b_S3"):
            v = base.copy()
            if a != "F0":
                v[:4] += gain + rng.normal(0, 0.5, 4)
                if loss_organ is not None:
                    v[loss_organ - 1] -= 6.0
                v[4] = base[4]
            arms[a] = {"dice": v.tolist(), "present": [True] * 5, "status": ["TP"] * 5, "ml": [50.0, 600.0, 2500.0, 2800.0, 80.0],
                       "hd95": [3.0] * 5, "nsd": [90.0] * 5, "vox": [1] * 5, "inter": [1] * 5}
        R[c] = {"NATIVE": {"ref": ref, "arms": arms, "shape": [1, 1, 1], "spacing": [1, 1, 1], "vox_ml": 1e-3, "diag_mm": 1.0}}
    return R, ids


def u5():
    import lctsc_analysis as A
    have = ["F0", "F1b_S1", "F1b_S2", "F1b_S3"]
    R, ids = synth(2.0)
    r = A.analyse(R, ids, have)
    assert r["P1"]["pass"] and r["P1_combination"]["combination"] == "GAIN_EVERY_DRAW", r["P1"]
    assert r["claims_allowed"]["for each of the three draws"] and not r["claims_allowed"]["site adaptation (a gain specific to the own institution)"]
    R, ids = synth(0.0, seed=1)
    r = A.analyse(R, ids, have)
    assert not r["P1"]["pass"] and r["P1"]["reading"] == "NO_GAIN"
    R, ids = synth(2.0, loss_organ=1, seed=2)
    r = A.analyse(R, ids, have)
    assert r["organ_costs"]["per_organ"]["esophagus"]["ni"] == "INFERIOR", r["organ_costs"]["per_organ"]["esophagus"]
    assert "esophagus" in r["claims_allowed"]["structures to name as a cost"]
    assert abs(r["spinal_cord_descriptive"]["F0"] - r["spinal_cord_descriptive"]["F1b_S1"]) < 1e-9
    r2 = A.analyse(R, ids, ["F0", "F1b_S1", "F1b_S3"], {"F1b_S2": "training FAILED twice"})   # a NOT RUN draw
    assert r2["draws_run"] == ["S1", "S3"] and r2["P1_combination"]["k_of_n_run"].endswith("of 2")


def u6():
    assert G.before_cutoff("2026-11-19 17:59:59") and not G.before_cutoff("2026-11-19 18:00:00")
    assert G.before_stop("2026-11-21 11:59:59") and not G.before_stop("2026-11-21 12:00:00")
    with tempfile.TemporaryDirectory() as td:
        for arm, t in (("F0", "2026-11-19 17:00:00"), ("F1b_S1", "2026-11-19 18:00:00")):
            (Path(td) / arm).mkdir()
            json.dump({"finished": t}, open(Path(td) / arm / "DONE.json", "w"))
        assert G.arm_done("F0", td)[0] == "DONE" and G.arm_done("F1b_S1", td)[0] == "LATE" and G.arm_done("F1b_S2", td)[0] == "MISSING"


def u7():
    import importlib.util
    spec = importlib.util.spec_from_file_location("dl", HERE / "lctsc_download.py")
    dl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dl)
    try:
        dl.download({"PatientID": "LCTSC-Test-S1-101", "Modality": "RTSTRUCT", "SeriesInstanceUID": "x", "ImageCount": 1}, {"files": {}})
        raise AssertionError("download of a test RTSTRUCT was not refused")
    except SystemExit as e:
        assert "refuse" in str(e)
    import lctsc_predict as P
    for argv in (["--arm", "F0", "--smoke", "--root", "/tmp/x", "--images", "/tmp/y", "--ids", "LCTSC-Test-S1-101"],
                 ["--arm", "F0", "--smoke", "--root", "/tmp/x", "--images", str(C.IMAGES), "--ids", "LCTSC-Train-S1-001"]):
        try:
            P.parse(argv)
            raise AssertionError(f"predict smoke not refused: {argv}")
        except AssertionError as e:
            if "not refused" in str(e):
                raise


def u8():
    """own-vs-other (descriptive): a built-in own-site effect is found; a pure site main effect cancels."""
    import numpy as np
    import lctsc_analysis as A
    ids = C.patients("test")
    rng = np.random.RandomState(3)
    per = {d: {c: (2.0 if C.site(c) == d else 0.0) + rng.normal(0, 0.3) for c in ids} for d in ("S1", "S2", "S3")}
    r = A.own_vs_other(per, nboot=500)
    assert 1.6 < r["estimate"] < 2.4 and r["ci"][0] > 0, r
    eff = {"S1": 3.0, "S2": 0.0, "S3": -3.0}
    per = {d: {c: eff[C.site(c)] for c in ids} for d in ("S1", "S2", "S3")}
    assert abs(A.own_vs_other(per, nboot=50)["estimate"]) < 1e-9


def u9():
    """registered missing-structure rule: a structure missing from the reference is not evaluable for every arm."""
    import lctsc_analysis as A
    R, ids = synth(2.0, seed=4)
    pid = ids[0]
    R[pid]["NATIVE"]["arms"]["F1b_S1"]["present"][0] = False         # F1b drops the esophagus on this patient
    R[pid]["NATIVE"]["arms"]["F1b_S1"]["dice"][0] = 0.0
    m = A.mask_missing(R, {pid: ["esophagus"]})
    import final_common as F
    f = m[pid]["NATIVE"]
    a = F.macro(f["arms"]["F1b_S1"], f["ref"], A.ADAPTED)
    b = F.macro(f["arms"]["F0"], f["ref"], A.ADAPTED)
    import numpy as np
    exp_a = np.mean(R[pid]["NATIVE"]["arms"]["F1b_S1"]["dice"][1:4])
    exp_b = np.mean(R[pid]["NATIVE"]["arms"]["F0"]["dice"][1:4])
    assert abs(a - exp_a) < 1e-9 and abs(b - exp_b) < 1e-9
    r = A.analyse(R, ids, ["F0", "F1b_S1", "F1b_S2", "F1b_S3"], missing={pid: ["esophagus"]})
    assert r["organ_costs"]["per_organ"]["esophagus"]["n_scans"] == len(ids) - 1


def u10():
    """lctsc_raster fill rule (run in external/lctsc_env, which has OpenCV): overlapping contours of one ROI -> union;
    a contour inside another -> hole; an identical duplicate does not cancel."""
    import subprocess
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "import numpy as np, lctsc_raster as RA\n"
        "a = np.array([[2, 2], [10, 2], [10, 10], [2, 10]], np.int32)\n"
        "b = np.array([[6, 6], [14, 6], [14, 14], [6, 14]], np.int32)\n"
        "h = np.array([[4, 4], [7, 4], [7, 7], [4, 7]], np.int32)\n"
        "u, r = RA.fill_slice([a, b], (20, 20)); assert u[8, 8] and u[3, 3] and u[13, 13]\n"
        "v, r = RA.fill_slice([a, h], (20, 20)); assert (not v[5, 5]) and v[9, 9] and r['n_holes'] == 1\n"
        "w, r = RA.fill_slice([a, a.copy()], (20, 20)); assert w[5, 5]\n"
        "d = np.array([[3, 3], [9, 3], [9, 9], [3, 9]], np.int32)\n"
        "x, r = RA.fill_slice([a, d], (20, 20)); assert x[6, 6] and r['n_holes'] == 0 and r['n_nested_duplicates'] == 1\n"
        "big = np.array([[0, 0], [19, 0], [19, 19], [0, 19]], np.int32); hole = np.array([[4, 4], [14, 4], [14, 14], [4, 14]], np.int32)\n"
        "isl = np.array([[8, 8], [10, 8], [10, 10], [8, 10]], np.int32)\n"
        "y, r = RA.fill_slice([big, hole, isl], (20, 20)); assert y[1, 1] and (not y[5, 5]) and y[9, 9] and r['n_islands'] == 1\n"
        "assert r['hole_area_fractions'] and r['hole_area_fractions'][0] < 0.5\n"
        "print('OK')\n") % str(HERE)
    r = subprocess.run([str(C.REPO / "external/lctsc_env/bin/python"), "-I", "-B", "-c", code], capture_output=True, text=True)
    assert r.returncode == 0 and "OK" in r.stdout, r.stderr[-500:]


def u11():
    """B2: before the cutoff the look refuses unless every arm is DONE (real) or declared NOT RUN; B4/B3 primitives."""
    import lctsc_infra as I
    src = (HERE / "lctsc_look_labels.py").read_text()
    assert "before the cutoff every arm must be DONE" in src
    assert "os.O_RDONLY" in (HERE / "lctsc_infra.py").read_text() and "O_CREAT" not in (HERE / "lctsc_infra.py").read_text()
    q = (HERE / "lctsc_train_queue.py").read_text()
    assert "12 * 3600" not in q and "open(LOCK" not in q and "recorded only after the launch" in q
    with tempfile.TemporaryDirectory() as td:
        try:
            I.acquire_gpu_lock(I.datetime(2000, 1, 1), path=Path(td) / "missing.lock", missing_poll_s=0.01)
            raise AssertionError("a missing lock file must be waited for, then refused at the deadline")
        except I.Refuse as e:
            assert e.code == I.DEADLINE_CODE
        assert not (Path(td) / "missing.lock").exists(), "the lock file is never created"
    import os
    old = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    try:
        I.gpu_identity_gate(I.TRAIN_DEADLINE)
        raise AssertionError("a non-GPU-1 device must be refused")
    except I.Refuse as e:
        assert e.code == 2
    finally:
        if old is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old


def u12():
    """review NB1: the freeze's training-draw classification."""
    import lctsc_freeze as FZ
    with tempfile.TemporaryDirectory() as td:
        def done(name, fin):
            p = Path(td) / f"{name}.json"
            json.dump({"finished": fin}, open(p, "w"))
            return p
        files = {"F1b_S1": done("a", "2026-11-14 10:00:00"), "F1b_S2": Path(td) / "none.json", "F1b_S3": done("c", "2026-11-15 19:30:00")}
        st = {"runs": {"F1b_S2": {"status": "FAILED"}}}
        r = FZ.classify_draws(st, "2026-11-14 12:00:00", files)
        assert r["F1b_S1"]["status"] == "DONE" and r["F1b_S2"]["status"] == "NOT RUN" and r["F1b_S3"]["status"] == "NOT RUN", r
        r = FZ.classify_draws({"runs": {"F1b_S2": {"status": "RUNNING"}}}, "2026-11-14 12:00:00", files)
        assert r["F1b_S2"]["status"] == "PENDING"
        r = FZ.classify_draws({"runs": {"F1b_S2": {"status": "RUNNING"}}}, "2026-11-15 18:00:00", files)
        assert r["F1b_S2"]["status"] == "NOT RUN"
        r = FZ.classify_draws({"runs": {"F1b_S2": {"status": "INTEGRITY REFUSED"}}}, "2026-11-14 12:00:00", files)
        assert r["F1b_S2"]["status"] == "NOT RUN"
        r = FZ.classify_draws({"runs": {"F1b_S2": {"status": "RESET"}}}, "2026-11-14 12:00:00", files)
        assert r["F1b_S2"]["status"] == "PENDING"
    # review NB2: no freeze while the unit or a training process is alive; the freeze snapshots train_state
    assert FZ.training_alive([], True) and FZ.training_alive(["python lctsc_train.py --run F1b_S3"], False)
    assert not FZ.training_alive(["python lctsc_freeze.py"], False)
    src = (HERE / "lctsc_freeze.py").read_text()
    assert 'train_state_at_freeze.json"]' in src and 'C.OUT / "train_state.json"]' not in src
    assert "no RESET after the freeze" in (HERE / "lctsc_train_queue.py").read_text()
    assert "look was reported NOT COMPLETED" in (HERE / "lctsc_look_labels.py").read_text()


def u13():
    """review NS1: a NOTRUN line beats a DONE.json at the look; lctsc_predict real mode checks it."""
    import lctsc_look_labels as LL
    with tempfile.TemporaryDirectory() as td:
        old = (G.PREREG_LOG, G.PRED)
        try:
            G.PREREG_LOG, G.PRED = Path(td) / "prereg.log", Path(td) / "pred"
            for a in G.ARMS:
                (G.PRED / a).mkdir(parents=True)
                json.dump({"finished": "2026-11-16 12:00:00", "mode": "real"}, open(G.PRED / a / "DONE.json", "w"))
            open(G.PREREG_LOG, "w").write("[2026-11-16 13:00:00] NOTRUN lctsc F1b_S2 2 failed attempts\n")
            done, nr = LL.arm_states()
            assert "F1b_S2" not in done and nr["F1b_S2"].startswith("declared NOT RUN"), (done, nr)
        finally:
            G.PREREG_LOG, G.PRED = old
    assert "notrun_declared(arm)" in (HERE / "lctsc_predict.py").read_text()


def main():
    res = {}
    for name in ("u1", "u2", "u3", "u4", "u5", "u6", "u7", "u8", "u9", "u10", "u11", "u12", "u13"):
        try:
            globals()[name]()
            res[name] = "PASS"
        except Exception:  # noqa: BLE001
            res[name] = "FAIL: " + traceback.format_exc(limit=3)
        print(name, res[name].splitlines()[0], flush=True)
    code = {f.name: L.sha256(f) for f in sorted(HERE.glob("*.py")) + sorted(HERE.glob("*.service"))}
    out = {"results": res, "all_pass": all(v == "PASS" for v in res.values()), "code_sha256": code, "written": L.now()}
    L.write_json(C.OUT / "unit_tests.json", out)
    print("all_pass", out["all_pass"])
    return 0 if out["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
