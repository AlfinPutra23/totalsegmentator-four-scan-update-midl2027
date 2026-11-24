"""vccor/final FLARE-T50 UNIT TESTS (CPU; synthetic or stand-in data only; NO FLARE-T50 label and no project label is read).

Covers: LFS pointer oids (against 'git hash-object'); the bootstrap replicate generators against the frozen
final_common.cluster_boot / m2_cluster.boot2 CIs (bit-identical percentiles); the two-family bootstrap (random and fixed
second family); two-sided bootstrap p-values; Holm; WORD A1's P1 rule as reused; the full analysis on synthetic scored scans
(all 18 arms, and with arms missing); the scorer's per-scan function and the label sanity gate on synthetic NIfTI volumes
(unreadable, non-integer and exact-integer float labels); the gates (upload gate tied to the freeze and the deposit line,
public gate ordering, stop rule, disk); the freeze checks (code, models, TotalSegmentator weights); the cutoff rule at the look;
the tier-2 wall-time accounting under a hand stop and after an interrupted run; the model registry against the real exports
(checkpoint sha256 == export.json, registered fields incl. labelled ids and the SSFT identities); t50_predict.py's argument
guards; after the freeze, rehearsal.json and smoke_predict.json are never rewritten (a post-freeze rehearsal writes its own
file; the smoke recorder refuses). Tests that touch study paths run under t50_common.use_root in a temporary folder and
restore the real paths after.
Writes I_flaret50/unit_tests.json (with --record; refused once code_freeze.json exists) or prints the table.

usage:  CUDA_VISIBLE_DEVICES="" nice -n 15 taskset -c 16-23 .venv/bin/python -I -B scripts/vccor/final/flaret50/test_t50.py [--record]

Version of 2026-11-09; revised after the code review (M1b).
"""
import contextlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
import t50_common as T  # noqa: E402
import t50_analysis as AN  # noqa: E402
import final_common as F  # noqa: E402

TESTS = []
BOUND = ["OUT", "DATA", "HF_LOCAL", "PRED", "LOOK", "TIER2", "K1B", "K1C", "RUN", "MANIFEST", "IMAGES_RECORD", "PLAN",
         "EXCLUSIONS", "GATE_UPLOADED", "GATE_PUBLIC", "FREEZE", "REHEARSAL_RECORD", "PREREG_LOG", "EVENTS_LOG",
         "FF_MODELS", "TS_ROOT", "IMG_DIR", "LBL_DIR", "REHEARSAL", "LABEL_SOURCE"]


@contextlib.contextmanager
def rooted():
    saved = {k: getattr(T, k) for k in BOUND}
    with tempfile.TemporaryDirectory() as td:
        T.use_root(td)
        try:
            yield Path(td)
        finally:
            for k, v in saved.items():
                setattr(T, k, v)


def test(fn):
    TESTS.append(fn)
    return fn


@test
def u01_pointer_oid_matches_git():
    with tempfile.TemporaryDirectory() as td:
        data = os.urandom(12345)
        import hashlib
        h = hashlib.sha256(data).hexdigest()
        p = Path(td) / "ptr"
        p.write_text(T.lfs_pointer_text(h, len(data)))
        g = subprocess.run(["git", "hash-object", str(p)], capture_output=True, text=True).stdout.strip()
        assert g == T.pointer_oid(h, len(data)), (g, T.pointer_oid(h, len(data)))
        assert T.pointer_oid(h, len(data) + 1) != g


@test
def u02_scan_boot_reps_equal_cluster_boot():
    rng = np.random.default_rng(7)
    vals = {f"c{i:02d}": float(x) for i, x in enumerate(rng.normal(0.8, 1.3, 50))}
    m, reps, _ = AN.scan_boot_reps(vals)
    cb = F.cluster_boot(vals)
    assert abs(m - cb["mean"]) < 1e-12
    ci = AN.pct_ci(reps)
    assert abs(ci[0] - cb["ci"][0]) < 1e-12 and abs(ci[1] - cb["ci"][1]) < 1e-12, (ci, cb["ci"])


@test
def u03_two_level_reps_equal_boot2():
    rng = np.random.default_rng(11)
    per = {d: {f"c{i:02d}": float(rng.normal(1.0 + 0.2 * j, 1.0)) for i in range(50)} for j, d in enumerate("abc")}
    per["b"].pop("c03")
    per["c"]["c07"] = float("nan")
    M, reps = AN.two_level_reps(per)
    tl = F.two_level(per)
    assert abs(M - tl["M"]) < 1e-12
    ci = AN.pct_ci(reps)
    assert abs(ci[0] - tl["ci"][0]) < 1e-12 and abs(ci[1] - tl["ci"][1]) < 1e-12, (ci, tl["ci"])
    one = {"a": per["a"]}
    _, r1 = AN.two_level_reps(one)
    t1 = F.two_level(one)
    assert abs(AN.pct_ci(r1)[0] - t1["ci"][0]) < 1e-12


@test
def u04_two_family():
    rng = np.random.default_rng(3)
    base = {f"c{i:02d}": float(rng.normal(1.0, 1.0)) for i in range(50)}
    A = {d: {c: v + rng.normal(0, 0.1) for c, v in base.items()} for d in "abc"}
    est0, reps0, n = AN.two_family_reps(A, A)
    assert abs(est0) < 1e-12 and n == 50
    assert AN.pct_ci(reps0)[0] <= 0 <= AN.pct_ci(reps0)[1]
    B = {e: {c: v - 1.0 for c, v in A[d].items()} for e, d in zip(("x", "y", "z"), "abc")}
    est, reps, _ = AN.two_family_reps(A, B)
    assert abs(est - 1.0) < 1e-9
    assert AN.pct_ci(reps)[0] > 0 and AN.p_two_sided(reps) < 0.01


@test
def u05_p_value_and_holm():
    reps = np.full(9999, 0.5)
    assert abs(AN.p_two_sided(reps) - 2.0 / 10000) < 1e-15
    assert AN.p_two_sided(np.array([-1.0, 1.0])) == 1.0
    adj = AN.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": None})
    assert abs(adj["a"] - 0.03) < 1e-12 and abs(adj["c"] - 0.06) < 1e-12 and abs(adj["b"] - 0.06) < 1e-12
    assert adj["d"] is None
    assert AN.direction_reading(-1.0, 0.01, "<0") == "DETECTED"
    assert AN.direction_reading(1.0, 0.01, "<0") == "REVERSED"
    assert AN.direction_reading(-1.0, 0.2, "<0") == "NOT DETECTED"


@test
def u06_p1_rule():
    rng = np.random.default_rng(5)
    good = {d: {f"c{i:02d}": float(rng.normal(1.5, 0.8)) for i in range(50)} for d in ("s1", "s2", "s3")}
    assert AN.p1_pooled(good)["pass"]
    small = {d: {c: v - 1.3 for c, v in good[d].items()} for d in good}
    r = AN.p1_pooled(small)
    assert not r["M_ge_0_30"] and not r["pass"]
    assert AN.combination(True, 3, 3) == "GAIN_EVERY_DRAW" and AN.combination(False, 0, 3) == "NO_GAIN"


def synth_case(shape=(40, 40, 30)):
    """A synthetic reference with the 13 organs as boxes (FLARE ids); arm maps are derived by shifting boxes."""
    G = np.zeros(shape, np.uint8)
    boxes = {}
    for o in range(1, 14):
        x, y = (o - 1) % 4 * 9 + 2, (o - 1) // 4 * 9 + 2
        boxes[o] = (slice(x, x + 7), slice(y, y + 7), slice(5, 25))
        G[boxes[o]] = o
    return G, boxes


def arm_map(G, boxes, shrink, rng):
    Y = np.zeros_like(G)
    for o, (sx, sy, sz) in boxes.items():
        k = int(rng.integers(0, 2)) + shrink
        Y[sx.start:sx.stop, sy.start:sy.stop, sz.start + k:sz.stop] = o
    return Y


def synth_R(n=24, seed=1):
    """Arms scored with final_common.Scorer: F0 shrinks boxes by 4-5 slices, FULL / K16 by 0-1 (better), SSFT_B by 3-4
    (pull-back), SSFT_D by 1-2, STOCK by 5-6, borrowed by 3-4, K1 by 2-3."""
    rng = np.random.default_rng(seed)
    shrink = {"F0": 4, "FULL_s1337": 0, "FULL_s2025": 0, "FULL_s3407": 1, "K16_s1337": 0, "K16_s2025": 0, "K16_s3407": 0,
              "SSFT_B": 3, "SSFT_D": 1, "STOCK": 5, "SCRATCH_NM1000": 8, "BORROW_amos": 3, "BORROW_mact": 3,
              "BORROW_word": 4, "K1_0014": 2, "K1_0001": 2, "K1_0006": 3, "K1_0038": 2}
    R, ids = {}, [f"FLARETs_{i:04d}" for i in range(1, n + 1)]
    for c in ids:
        G, boxes = synth_case()
        sc = F.Scorer(G, 13, (1.5, 1.5, 3.0))
        rec = sc.frame_record()
        for a, s in shrink.items():
            rec["arms"][a] = sc.score(arm_map(G, boxes, s, rng), surface=True)
        R[c] = {"NATIVE": rec}
    return R, ids, list(shrink)


@test
def u07_analysis_synthetic():
    R, ids, arms = synth_R()
    res = AN.analyse(R, ids, arms)
    assert res["R1"]["pass"] and res["R1"]["reading"] == "REPLICATED", res["R1"]
    S = res["secondary_holm_family"]["contrasts"]
    for s in ("S1", "S2", "S3", "S5", "S6"):
        assert S[s]["reading"] == "DETECTED", (s, S[s])
    assert S["S5"]["second_family_fixed"] is True
    assert res["secondary_holm_family"]["m"] == 6
    assert res["tier2"]["K1_minus_K4"]["reading"] == "DETECTED"
    assert res["tier2"]["K1_minus_K4"]["second_family_fixed"] is False
    assert res["panels"]["median_gain_scan"] in ids and res["panels"]["largest_loss_scan"] in ids
    assert res["stock_vs_F0"]["reading"] in ("STOCK_GAINS", "STOCK_NO_GAIN")
    json.dumps(F.clean_nan(res), default=F._default)
    AN.summary(res)


@test
def u08_analysis_missing_arms():
    R, ids, arms = synth_R(n=12, seed=2)
    have = [a for a in arms if not a.startswith("K1_") and a not in ("SSFT_D", "BORROW_word", "K16_s3407")]
    res = AN.analyse(R, ids, have, {"SSFT_D": "test"})
    S = res["secondary_holm_family"]["contrasts"]
    assert S["S2"]["reading"] == "NOT RUN" and res["secondary_holm_family"]["m"] == 5
    assert res["tier2"]["reading"] == "NOT RUN"
    assert S["S4"]["draws"] == ["s1337", "s2025"]


@test
def u09_score_one_and_sanity_gate():
    import nibabel as nib
    import t50_score as SC
    with rooted() as td:
        G, _ = synth_case()
        aff = np.diag([1.5, 1.5, 3.0, 1.0])
        T.LBL_DIR.mkdir(parents=True)
        T.IMG_DIR.mkdir(parents=True)
        nib.save(nib.Nifti1Image(G, aff), str(T.label_path("FLARETs_0001")))
        nib.save(nib.Nifti1Image(np.zeros(G.shape, np.int16), aff), str(T.image_path("FLARETs_0001")))
        lut_inv = {v: k for k, v in enumerate(F.ts_lut("flare")) if v}
        Yts = np.zeros_like(G)
        for o in range(1, 14):
            Yts[G == o] = lut_inv[o]
        p_ts, p_id = td / "ts.nii.gz", td / "id.nii.gz"
        nib.save(nib.Nifti1Image(Yts, aff), str(p_ts))
        nib.save(nib.Nifti1Image(G.copy(), aff), str(p_id))
        _, rec = SC.score_one(("FLARETs_0001", {"F0": (str(p_ts), T.sha256(p_ts), "ts"),
                                                "STOCK": (str(p_id), T.sha256(p_id), "id")}))
        for a in ("F0", "STOCK"):
            assert all(abs(x - 100.0) < 1e-9 for x in rec["NATIVE"]["arms"][a]["dice"]), a
        bad, diag = SC.sanity_gate(["FLARETs_0001"])
        assert bad == {} and diag["FLARETs_0001"]["affine_max_abs_diff"] == 0.0
        Gf = G.astype(np.float32)
        nib.save(nib.Nifti1Image(Gf, aff), str(T.label_path("FLARETs_0001")))
        assert SC.sanity_gate(["FLARETs_0001"])[0] == {}
        Gf[G == 13] = 12.99995
        nib.save(nib.Nifti1Image(Gf, aff), str(T.label_path("FLARETs_0001")))
        assert "FLARETs_0001" in SC.sanity_gate(["FLARETs_0001"])[0]
        G2 = G.copy()
        G2[0, 0, 0] = 14
        nib.save(nib.Nifti1Image(G2, aff), str(T.label_path("FLARETs_0001")))
        assert "FLARETs_0001" in SC.sanity_gate(["FLARETs_0001"])[0]
        T.label_path("FLARETs_0001").write_bytes(b"\x1f\x8b\x08\x00garbage")
        assert SC.sanity_gate(["FLARETs_0001"])[0]["FLARETs_0001"].startswith("unreadable")


def write_gates(td, freeze_written, upload, public, deposit=True):
    T.write_json(T.FREEZE, {"files": {}, "models": {}, "pending_models": [], "written": freeze_written})
    z = td / "osf_deposit_x_flaret50.zip"
    z.write_bytes(os.urandom(32))
    if deposit:
        T.prereg_line("DEPOSIT", z, "test")
    json.dump({"osf_url": "https://osf.io/abcde", "uploaded_kst": upload, "written_by": "test", "evidence": "test",
               "deposit_zip_sha256": T.sha256(z), "freeze_sha256": T.sha256(T.FREEZE)}, open(T.GATE_UPLOADED, "w"))
    if public:
        json.dump({"osf_url": "https://osf.io/abcde", "public_checked_kst": public, "api_http_status": 200, "public": True,
                   "written_by": "test", "evidence": "test"}, open(T.GATE_PUBLIC, "w"))


def refused(fn):
    try:
        fn()
        return False
    except T.Refuse:
        return True


@test
def u10_gates():
    now = datetime.now().replace(microsecond=0)
    fmt = (lambda t: t.strftime("%Y-%m-%d %H:%M:%S"))
    with rooted() as td:
        T.OUT.mkdir(parents=True)
        assert refused(T.gate_uploaded) and refused(T.gate_public)
        write_gates(td, fmt(now - timedelta(hours=3)), fmt(now - timedelta(hours=2)), fmt(now - timedelta(hours=1)))
        assert T.gate_uploaded()["osf_url"] == "https://osf.io/abcde"
        if now < T.STOP_RULE:
            assert T.gate_public()["public"] is True
        assert sum(" GATE sha256 " in ln for ln in T.prereg_lines()) == 2
        T.gate_public()
        assert sum(" GATE sha256 " in ln for ln in T.prereg_lines()) == 2, "a gate file is logged once"
        write_gates(td, fmt(now - timedelta(hours=1)), fmt(now - timedelta(hours=2)), None)
        assert refused(T.gate_uploaded), "upload before the freeze accepted"
        write_gates(td, fmt(now - timedelta(hours=3)), fmt(now - timedelta(hours=2)), None, deposit=False)
        assert refused(T.gate_uploaded), "deposit without a DEPOSIT line accepted"
        write_gates(td, fmt(now - timedelta(hours=3)), fmt(now - timedelta(hours=2)), fmt(now - timedelta(hours=2, minutes=5)))
        assert refused(T.gate_public), "public before the upload accepted"
        write_gates(td, fmt(now - timedelta(hours=3)), fmt(now - timedelta(hours=2)), fmt(now + timedelta(hours=1)))
        assert refused(T.gate_public), "public in the future accepted"
        g = json.load(open(T.GATE_UPLOADED))
        json.dump(dict(g, freeze_sha256="0" * 64), open(T.GATE_UPLOADED, "w"))
        assert refused(T.gate_uploaded), "another freeze accepted"
        json.dump(dict(g, osf_url="http://example.org/x"), open(T.GATE_UPLOADED, "w"))
        assert refused(T.gate_uploaded), "bad url accepted"
    assert refused(lambda: T.disk_gate(1e6, "test"))
    sr = T.STOP_RULE
    try:
        T.STOP_RULE = now - timedelta(minutes=30)
        with rooted() as td:
            T.OUT.mkdir(parents=True)
            write_gates(td, fmt(now - timedelta(hours=3)), fmt(now - timedelta(hours=2)), fmt(now - timedelta(minutes=10)))
            assert refused(T.gate_public), "stop rule not enforced"
    finally:
        T.STOP_RULE = sr


@test
def u11_registry_against_exports():
    for a in T.TIER1_ARMS:
        kind, R = T.resolve_model(a)
        if a == "F0":
            assert kind == "ts" and R["redirect"] == {}
        elif T.ARMS[a]["kind"] == "ts":
            assert set(R["redirect"]) == set(T.ARMS[a]["parts"]) and all(len(h) == 64 for h in R["shas"].values())
        else:
            assert kind == "nnunet" and R["n_classes"] == 13
    for a in ("STOCK", "SCRATCH_NM1000"):
        labels = json.load(open(Path(T.ARMS[a]["dir"]) / "dataset.json"))["labels"]
        names = ["liver", "kidney_right", "spleen", "pancreas", "aorta", "inferior_vena_cava", "adrenal_gland_right",
                 "adrenal_gland_left", "gallbladder", "esophagus", "stomach", "duodenum", "kidney_left"]
        assert [labels[n] for n in names] == list(range(1, 14)), labels
    assert len(T.ts_weight_files()) == 15 and all(p.is_file() for p in T.ts_weight_files())
    assert T.check_versions()["TotalSegmentator"] == T.TS_VERSION
    swapped = dict(T.ARMS["SSFT_B"], check=dict(T.ARMS["SSFT_B"]["check"], arm="D"))
    old = T.ARMS["SSFT_B"]
    T.ARMS["SSFT_B"] = swapped
    try:
        assert refused(lambda: T.resolve_model("SSFT_B")), "a swapped SSFT identity accepted"
    finally:
        T.ARMS["SSFT_B"] = old


@test
def u12_predict_guards():
    import t50_predict as P
    real_img = str(T.IMG_DIR)
    bad = [["--arm", "STOCK", "--timing", "--ids", "FLARETs_0030,FLARETs_0050"],
           ["--arm", "F0", "--timing", "--ids", "FLARETs_0001,FLARETs_0002"],
           ["--arm", "F0", "--timing", "--ids", "FLARETs_0030"],
           ["--arm", "F0", "--ids", "FLARETs_0001"],
           ["--arm", "F0", "--device", "cpu"],
           ["--arm", "F0", "--smoke", "--images", real_img, "--root", "/tmp/x", "--ids", "STANDIN_0001"],
           ["--arm", "F0", "--smoke", "--images", "/tmp/x", "--root", "/tmp/y", "--ids", "FLARETs_0001"]]
    for argv in bad:
        try:
            P.parse(argv)
            raise AssertionError(f"accepted {argv}")
        except (AssertionError, SystemExit) as e:
            if isinstance(e, AssertionError) and str(e).startswith("accepted"):
                raise
    P.parse(["--arm", "F0", "--timing", "--ids", "FLARETs_0050,FLARETs_0030"])
    P.parse(["--arm", "STOCK", "--smoke", "--images", "/tmp/x", "--root", "/tmp/y", "--ids", "STANDIN_0001", "--device",
             "cpu"])


@test
def u13_luts():
    lut = F.ts_lut("flare")
    from totalsegmentator.map_to_binary import class_map
    inv = {v: k for k, v in class_map["total"].items()}
    assert lut[inv["liver"]] == 1 and lut[inv["kidney_left"]] == 13 and lut[inv["duodenum"]] == 12
    assert lut[inv["kidney_cyst_right"]] == 2
    assert int(lut.max()) == 13
    assert list(F.identity_lut(13)[:15]) == list(range(14)) + [0]


@test
def u14_freeze_checks():
    with rooted() as td:
        a, b = td / "a.py", td / "x" / "fold_0" / "checkpoint_final.pth"
        b.parent.mkdir(parents=True)
        a.write_text("x = 1\n")
        b.write_bytes(os.urandom(64))
        tsd = Path(T.TS_ROOT) / "nnunet" / "results"
        for tid in (291, 292, 293, 294, 295):
            d = tsd / f"Dataset{tid}_t" / "nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres" / "fold_0"
            d.mkdir(parents=True)
            for f in (d / "checkpoint_final.pth", d.parent / "plans.json", d.parent / "dataset.json"):
                f.write_bytes(os.urandom(32))
        tsw = {str(p): T.sha256(p) for p in T.ts_weight_files()}
        T.OUT.mkdir(parents=True)
        fz = {"files": {str(a): T.sha256(a)}, "models": {"FULL_s1337": {str(b): T.sha256(b)}, "TS_WEIGHTS": tsw},
              "pending_models": []}
        json.dump(fz, open(T.FREEZE, "w"))
        T.freeze_ok()
        T.check_frozen_model("FULL_s1337")
        T.check_frozen_model("F0")
        assert T.frozen_checkpoint_hashes("FULL_s1337") == [T.sha256(b)]
        assert refused(lambda: T.check_frozen_model("SSFT_B")), "unhashed model accepted"
        json.dump({"addendum": 1, "models": {"SSFT_B": {str(b): T.sha256(b)}}}, open(T.OUT / "freeze_addendum_1.json", "w"))
        T.check_frozen_model("SSFT_B")
        w = T.ts_weight_files()[0]
        w.write_bytes(os.urandom(32))
        T._HCACHE.clear()
        assert refused(lambda: T.check_frozen_model("F0")), "changed TotalSegmentator weights accepted"
        json.dump(dict(fz, models={"FULL_s1337": fz["models"]["FULL_s1337"]}), open(T.FREEZE, "w"))
        assert refused(lambda: T.check_frozen_model("F0")), "missing TS_WEIGHTS accepted"
        b.write_bytes(os.urandom(64))
        T._HCACHE.clear()
        json.dump(fz, open(T.FREEZE, "w"))
        assert refused(lambda: T.check_frozen_model("FULL_s1337")), "changed model accepted"
        a.write_text("x = 2\n")
        T._HCACHE.clear()
        assert refused(T.freeze_ok), "changed code accepted"


@test
def u15_tier2_accounting():
    import t50_tier2 as M
    with rooted() as td:
        saved = (M.gates, T.acquire_gpu_lock, T.release_gpu_lock, T.wait_gpu1_idle, T.disk_gate, M.train_cmd,
                 M.FIRST_BUDGET_S)
        old_handlers = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)}
        try:
            M.gates = lambda check=False: True
            T.acquire_gpu_lock = lambda *a, **k: None
            T.release_gpu_lock = lambda fd: None
            T.wait_gpu1_idle = lambda *a, **k: 0
            T.disk_gate = lambda *a, **k: 99.0
            M.train_cmd = lambda draw: ["/bin/sleep", "60"]
            threading.Timer(4.0, os.kill, (os.getpid(), signal.SIGTERM)).start()
            try:
                M.main([])
                outcome = "returned"
            except SystemExit as e:
                outcome = f"SystemExit {e.code}"
            st = json.load(open(M.state_path()))["draws"]["d1_0014"]
            assert outcome.startswith("SystemExit"), outcome
            assert st["wall_s"] > 2.0 and "end" in st["history"][-1] and st["history"][-1].get("interrupted"), st
            first_wall = st["wall_s"]
            full = json.load(open(M.state_path()))
            full["draws"]["d1_0001"]["history"].append({"attempt": 1, "start": (datetime.now() - timedelta(minutes=30)).strftime(
                "%Y-%m-%d %H:%M:%S"), "log": str(td / "fake.log")})
            (td / "fake.log").write_text("x")
            T.write_json(M.state_path(), full)
            st2 = M.load_state()
            M.charge_interrupted(st2)
            d2 = st2["draws"]["d1_0001"]
            assert 1700 < d2["wall_s"] < 1900 and d2["history"][-1]["interrupted"], d2
            assert abs(st2["draws"]["d1_0014"]["wall_s"] - first_wall) < 1e-9
        finally:
            (M.gates, T.acquire_gpu_lock, T.release_gpu_lock, T.wait_gpu1_idle, T.disk_gate, M.train_cmd,
             M.FIRST_BUDGET_S) = saved
            for s, h in old_handlers.items():
                signal.signal(s, h)


@test
def u16_s5_fixed_borrowed():
    rng = np.random.default_rng(1)
    shared = rng.normal(0, 1.2, 50)
    ids = [f"FLARETs_{i:04d}" for i in range(1, 51)]

    def fam(means):
        return {f"d{j}": {c: float(m + shared[i] + rng.normal(0, 0.4)) for i, c in enumerate(ids)} for j, m in enumerate(means)}
    A, B = fam([2.0, 1.8, 2.2]), fam([2.1, 0.5, -1.0])
    est_r, reps_r, _ = AN.two_family_reps(A, B)
    est_f, reps_f, _ = AN.two_family_reps(A, B, resample_b=False)
    assert abs(est_r - est_f) < 1e-12
    w_r, w_f = np.diff(AN.pct_ci(reps_r))[0], np.diff(AN.pct_ci(reps_f))[0]
    assert w_f < w_r / 3, (w_f, w_r)
    assert AN.pct_ci(reps_f)[0] > 0 and AN.p_two_sided(reps_f) < 0.01


@test
def u17_cutoff_at_look():
    import t50_score as SC
    with rooted():
        for a, fin in (("FULL_s1337", (T.CUTOFF - timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S")),
                       ("STOCK", T.CUTOFF.strftime("%Y-%m-%d %H:%M:%S"))):
            T.write_json(T.done_file(a), {"arm": a, "finished": fin})
        have, nr = SC.arm_status()
        assert "FULL_s1337" in have and "STOCK" not in have and "STOCK" in nr, (have, nr)


@test
def u18_records_not_rewritten_after_freeze():
    import t50_rehearse as RH
    import t50_smoke_record as SR
    with rooted() as td:
        pred, img = td / "smoke" / "pred", td / "smoke" / "images"
        (pred / "F0" / "rec").mkdir(parents=True)
        img.mkdir()
        (img / "STANDIN_0001_0000.nii.gz").write_bytes(os.urandom(32))
        T.write_json(pred / "F0" / "DONE.json", {"arm": "F0", "kind": "ts", "ids": ["STANDIN_0001"], "n": 1,
                                                 "model_sha256": {}, "replay_misses_this_run": 0, "captured_parts": [292],
                                                 "finished": T.now(), "mode": "smoke"})
        T.write_json(pred / "F0" / "rec" / "STANDIN_0001.json", {"fg_vox": 7, "parts": [{"tid": 291, "source": "ts"}]})
        assert refused(lambda: SR.record(T.OUT / "pred", img, "test")), "a smoke root inside the study folder accepted"
        smoke = SR.record(pred, img, "test")
        assert json.load(open(smoke))["arms"]["F0"]["sources"] == {"291": "ts"}
        out = {"all_pass": True, "scenarios": []}
        assert RH.record_result(out, "0/0") == T.REHEARSAL_RECORD
        h_reh, h_smoke = T.sha256(T.REHEARSAL_RECORD), T.sha256(smoke)
        T.write_json(T.FREEZE, {"files": {}, "models": {}, "pending_models": []})
        p1, p2 = RH.record_result(dict(out, all_pass=False), "0/1"), RH.record_result(out, "1/1")
        assert (p1.name, p2.name) == ("rehearsal_postfreeze_1.json", "rehearsal_postfreeze_2.json"), (p1, p2)
        assert json.load(open(p1))["all_pass"] is False and os.stat(p2).st_mode & 0o777 == 0o444
        assert refused(lambda: SR.record(pred, img, "test")), "smoke_predict.json rewritten after the freeze"
        assert T.sha256(T.REHEARSAL_RECORD) == h_reh and T.sha256(smoke) == h_smoke
        lines = T.prereg_lines()
        assert sum(" REHEARSAL sha256 " in ln for ln in lines) == 3 and sum(" SMOKE sha256 " in ln for ln in lines) == 1


def main():
    record = "--record" in sys.argv
    if record and T.FREEZE.exists():
        print("[REFUSE] code_freeze.json exists: unit_tests.json is frozen and is not re-recorded")
        return 2
    rows, ok_all = [], True
    for fn in TESTS:
        t0 = time.time()
        try:
            fn()
            rows.append({"test": fn.__name__, "ok": True, "s": round(time.time() - t0, 1)})
            print(f"PASS {fn.__name__}")
        except Exception:                            # noqa: BLE001
            ok_all = False
            rows.append({"test": fn.__name__, "ok": False, "error": traceback.format_exc()[-1500:]})
            print(f"FAIL {fn.__name__}\n{traceback.format_exc()}")
    out = {"all_pass": ok_all, "n": len(rows), "tests": rows, "run": T.now(),
           "code_sha256": {p.name: T.sha256(p) for p in sorted(HERE.glob("*.py"))}}
    if record:
        T.write_json(T.OUT / "unit_tests.json", out)
    print(f"ALL_PASS {ok_all} ({sum(r['ok'] for r in rows)}/{len(rows)})")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
