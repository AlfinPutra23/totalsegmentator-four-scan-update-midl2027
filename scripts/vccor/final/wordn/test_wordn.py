"""vccor/final WORD-N UNIT TESTS (CPU; scratch folders under /tmp only; no real image, label or prediction is read; the real
models are only hashed / their export.json read for the registry test).

  u01 bootstrap helpers: two-level contrast CI == final_common.two_level; scan contrast CI == cluster_boot
  u02 analysis on a synthetic world (SPLIT): P-V REDUCED, OS-D / OS-E OVERSEGMENTED, family m = 4, readings consistent
  u03 the overshoot switch: AC_COMBINED puts H-D / H-E (signed + accuracy cost) in the family; the plan must fix the mode
  u04 a null synthetic world: P-V NOT_SHOWN, no OVERSEGMENTED, P1-fresh NO_GAIN
  u05 required arms only (F0, F1b_d1): the analysis runs; S1-fresh and tier 2 NOT RUN
  u06 gates: upload / public gate files (missing, wrong freeze, upload before freeze, deposit not logged, other URL, public
      after the stop rule, not HTTP 200)
  u07 the archive whitelist: only imagesVal / labelsVal of the 20 validation ids; labelsTr / labelsTs / imagesTr refused
  u08 the arm registry against the real A1 / A1.2 exports (strict checks pass; labelled ids in draw order; hashes match)
  u09 predict guards: smoke refuses validation ids and the WORD image folder; real mode refuses smoke options
  u10 label maps: every arm's lut kind; fin_look.lut('word', kind) shapes; analysis_lut folds rectum into colon
  u11 the cutoff at the look: a DONE written at / after the cutoff is NOT RUN; before it, DONE
  u12 CPU arms on stand-in maps: MERGE_N keeps F0's non-target classes; LABELMAP applies the frozen rules' lut
  u13 the intake's duplicate rule on synthetic images: val-val (higher id dropped) and val-test (val dropped)
  u14 the runner: job commands, GPU flag, tier-2 NOT RUN without the smoke pass, input checks
  u15 records are not rewritten once read-only (write_json refuses)
  u16 acceptance check: an arm with a clear gain and no organ loss is ACCEPTED; a losing arm is not
  u17 B1: a missing Holm member enters with p = 1.0 (m stays 4); the evaluable members are reported
  u18 SF2: arm_allowed refuses a declared NOT RUN arm and tier 2 without VSMOKE_PASS; the look's accounting is binding
  u19 SF1 / SF3: the GPU identity gate refuses anything but GPU 1's UUID; the lock wait is cutoff-aware (code 4)
  u21 NB2: GPU-infrastructure outages (nvidia-smi errors, an unlisted UUID, a missing lock file) are waited out
      (cutoff-aware); a wrong CUDA_VISIBLE_DEVICES stays a configuration error
  u22 N-F6: gpu_session releases the lock and repeats when GPU 1 is busy again after the lock was taken
  u20 SF1 / NB1: a resource refusal is retried after a back-off (never a NOT RUN by itself; the cutoff ends it); an
      unexpected error declares the arm NOT RUN unless its DONE exists (N-D2); VF arms are NOT RUN when V0 is NOT RUN

usage:  .venv/bin/python -I -B scripts/vccor/final/wordn/test_wordn.py [--record]

Version of 2026-11-13.
"""
import json
import os
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime, timedelta
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import wn_common as T  # noqa: E402

TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


def synth_R(n=18, effect=True, seed=1, arms_=None):
    """A synthetic R in fin_look record form: 15 organs; F0 under-segments; F1b fits solid organs and overshoots the
    tubular ones (effect=True) or equals F0 plus noise (effect=False)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    arms_ = arms_ or ["F0", "F1b_d1", "F1b_d2", "F1b_d3", "STOCK_d1", "STOCK_d2", "STOCK_d3", "V0", "VF5k_d1",
                      "VF5k_d2", "VF5k_d3"]
    R = {}
    for i in range(n):
        c = f"{i:04d}"
        ml = rng.uniform(20, 400, 15)
        ref = {"ml": ml.tolist(), "present": [True] * 15, "vox": [int(x * 100) for x in ml]}
        rec = {"ref": ref, "arms": {}}
        for a in arms_:
            f = np.ones(15)
            if a == "F0" or a == "V0":
                f = 0.85 + rng.normal(0, 0.03, 15)
                dice = 80 + rng.normal(0, 2, 15)
            elif effect and (a.startswith("F1b") or a.startswith("VF")):
                f = 1.0 + rng.normal(0, 0.02, 15)
                f[[6, 8]] = 1.4 + rng.normal(0, 0.05, 2)
                dice = 84 + rng.normal(0, 2, 15)
            elif effect and a.startswith("STOCK"):
                f = 1.0 + rng.normal(0, 0.03, 15)
                dice = 78 + rng.normal(0, 2, 15)
            else:
                f = 0.85 + rng.normal(0, 0.03, 15)
                dice = 80 + rng.normal(0, 2, 15)
            rec["arms"][a] = {"ml": (ml * f).tolist(), "present": [True] * 15, "dice": dice.tolist(),
                              "status": ["TP"] * 15, "hd95": [5.0] * 15, "nsd": [80.0] * 15}
        R[c] = {"NATIVE": rec}
    return R


@test
def u01_bootstrap_helpers():
    import final_common as F
    import t50_analysis as TA
    import numpy as np
    rng = np.random.default_rng(3)
    per = {d: {f"{i:03d}": float(rng.normal(1, 1)) for i in range(15)} for d in ("d1", "d2", "d3")}
    a = TA.two_level_contrast(per, ">0")
    b = F.two_level(per)
    assert np.allclose(a["ci"], b["ci"]) and abs(a["estimate"] - b["M"]) < 1e-12
    v = per["d1"]
    c = TA.single_contrast(v, ">0")
    assert np.allclose(c["ci"], F.cluster_boot(v)["ci"])


@test
def u02_analysis_split():
    import wn_analysis as AN
    R = synth_R()
    ids = sorted(R)
    res = AN.analyse(R, ids, set(R[ids[0]]["NATIVE"]["arms"]), {}, mode="SPLIT")
    assert res["P_V"]["reading"] == "VOLUME_ERROR_REDUCED", res["P_V"]
    fam = res["holm_family"]["results"]
    assert set(fam) == {"OS-D", "OS-E", "P1-fresh", "S1-fresh"} and res["holm_family"]["m"] == 4
    assert fam["OS-D"]["reading"] == "OVERSEGMENTED" and fam["OS-E"]["reading"] == "OVERSEGMENTED", fam
    assert fam["P1-fresh"]["reading"] == "GAIN" and fam["S1-fresh"]["reading"] == "F1B_BETTER", fam
    for k, v in fam.items():
        assert v["p_holm"] >= v["p_iut"]
    tv = res["tubular_volumes"]["duodenum"]
    assert tv["accuracy_cost_reading_descriptive"] in ("COST_SHOWN", "NO_COST_SHOWN")
    assert res["tier2"]["V_P1"]["reading"].startswith("V_GAIN")
    assert "A1_readings_fresh_descriptive" in res and "error" not in res["acceptance_check_descriptive"]


@test
def u03_overshoot_switch():
    import wn_analysis as AN
    R = synth_R()
    ids = sorted(R)
    res = AN.analyse(R, ids, set(R[ids[0]]["NATIVE"]["arms"]), {}, mode="AC_COMBINED")
    fam = res["holm_family"]["results"]
    assert set(fam) == {"H-D", "H-E", "P1-fresh", "S1-fresh"}
    assert fam["H-D"]["parts"] == ["signed_e_F1b > 0", "|e_F1b| - |e_F0| > 0"]
    try:
        AN.analyse(R, ids, set(R[ids[0]]["NATIVE"]["arms"]), {}, mode="OTHER")
        raise RuntimeError("bad mode accepted")
    except AssertionError:
        pass
    with tempfile.TemporaryDirectory(dir="/tmp") as td:
        T.use_root(td)
        T.write_json(T.PLAN, {"overshoot_mode": None})
        assert refused(lambda: T.overshoot_mode())
        T.write_json(T.PLAN, {"overshoot_mode": "SPLIT"})
        assert T.overshoot_mode() == "SPLIT"


@test
def u04_null_world():
    import wn_analysis as AN
    R = synth_R(effect=False, seed=5)
    ids = sorted(R)
    res = AN.analyse(R, ids, set(R[ids[0]]["NATIVE"]["arms"]), {}, mode="SPLIT")
    assert res["P_V"]["reading"] == "NOT_SHOWN", res["P_V"]
    fam = res["holm_family"]["results"]
    assert fam["OS-D"]["reading"] == "NOT_CONFIRMED" and fam["P1-fresh"]["reading"] == "NO_GAIN", fam


@test
def u05_required_only():
    import wn_analysis as AN
    R = synth_R(arms_=["F0", "F1b_d1"])
    ids = sorted(R)
    res = AN.analyse(R, ids, {"F0", "F1b_d1"}, {"F1b_d2": "x"}, mode="SPLIT")
    assert res["holm_family"]["results"]["S1-fresh"]["reading"] == "NOT RUN"
    assert res["tier2"]["V_P1"]["reading"] == "NOT RUN"
    assert res["draws_run"] == ["d1"]


def refused(fn):
    try:
        fn()
    except T.Refuse:
        return True
    return False


@test
def u06_gates():
    with tempfile.TemporaryDirectory(dir="/tmp") as td:
        T.use_root(td)
        T.write_json(T.FREEZE, {"written": (datetime.now() - timedelta(hours=5)).strftime("%Y-%m-%d %H:%M:%S"), "files": {}})
        fz = T.sha256(T.FREEZE)
        import zipfile
        zp = Path(td) / "osf_deposit_2026-11-15_wordn.zip"
        with zipfile.ZipFile(zp, "w") as zz:
            zz.write(T.FREEZE, "code_freeze.json")
        z = T.sha256(zp)
        good = {"osf_url": "https://osf.io/abcde", "uploaded_kst": (datetime.now() - timedelta(hours=2)).strftime(
            "%Y-%m-%d %H:%M:%S"), "written_by": "t", "evidence": "t", "deposit_zip_sha256": z, "freeze_sha256": fz}
        assert refused(T.gate_uploaded)                                         # missing
        T.write_json(T.GATE_UPLOADED, good)
        assert refused(T.gate_uploaded)                                         # no DEPOSIT line
        with open(T.PREREG_LOG, "a") as f:
            f.write(f"[x] DEPOSIT sha256 {z} {zp} (t)\n")
        assert T.gate_uploaded()["osf_url"] == "https://osf.io/abcde"
        T.write_json(T.GATE_UPLOADED, dict(good, freeze_sha256="b" * 64))
        assert refused(T.gate_uploaded)                                         # other freeze
        zp2 = Path(td) / "other_wordn.zip"
        with zipfile.ZipFile(zp2, "w") as zz:
            zz.writestr("code_freeze.json", "{}")
        with open(T.PREREG_LOG, "a") as f:
            f.write(f"[x] DEPOSIT sha256 {T.sha256(zp2)} {zp2} (t)\n")
        T.write_json(T.GATE_UPLOADED, dict(good, deposit_zip_sha256=T.sha256(zp2)))
        assert refused(T.gate_uploaded)                                         # zip holds another freeze (N5)
        T.write_json(T.GATE_UPLOADED, dict(good, uploaded_kst=(datetime.now() - timedelta(hours=9)).strftime(
            "%Y-%m-%d %H:%M:%S")))
        assert refused(T.gate_uploaded)                                         # upload before the freeze
        T.write_json(T.GATE_UPLOADED, good)
        pub = {"osf_url": "https://osf.io/abcde", "public_checked_kst": (datetime.now() - timedelta(hours=1)).strftime(
            "%Y-%m-%d %H:%M:%S"), "api_http_status": 200, "public": True, "written_by": "t", "evidence": "t"}
        assert refused(T.gate_public)                                           # missing
        T.write_json(T.GATE_PUBLIC, pub)
        assert T.gate_public()["public"] is True
        T.write_json(T.GATE_PUBLIC, dict(pub, api_http_status=401))
        assert refused(T.gate_public)
        T.write_json(T.GATE_PUBLIC, dict(pub, osf_url="https://osf.io/zzzzz"))
        assert refused(T.gate_public)
        old = T.STOP_RULE
        try:
            T.STOP_RULE = datetime.now() - timedelta(hours=1, minutes=30)
            T.write_json(T.GATE_PUBLIC, pub)
            assert refused(T.gate_public)                                       # public after the stop rule
        finally:
            T.STOP_RULE = old
        assert sum("GATE sha256" in ln for ln in T.prereg_lines()) >= 2


@test
def u07_whitelist():
    assert T.member("labelsVal", "0001").endswith("labelsVal/word_0001.nii.gz")
    for bad in (lambda: T.member("labelsTr", "0001"), lambda: T.member("labelsVal", "0004"),
                lambda: T.member("imagesVal", "0095"), lambda: T.extract_member_bytes("WORD-V0.1.0/labelsTr/word_0004.nii.gz"),
                lambda: T.extract_member_bytes("WORD-V0.1.0/imagesTr/word_0095.nii.gz"),
                lambda: T.extract_member_bytes("labelsTs/word_0014.nii.gz")):
        try:
            bad()
            raise RuntimeError("not refused")
        except AssertionError:
            pass
    assert len(T.VAL_IDS) == 20 and sorted(T.VAL_IDS) == sorted(T.W.split()["val"])
    assert not set(T.VAL_IDS) & (set(T.W.split()["test"]) | set(T.W.draw_scans()))


@test
def u08_registry_against_exports():
    import importlib
    importlib.reload(T)                       # real paths (earlier tests used use_root)
    for d in T.DRAWS:
        k, R = T.resolve_model(f"F1b_{d}", strict=True)
        assert k == "ts" and list(R["redirect"]) == [291]
        man = json.load(open(T.arms()[f"F1b_{d}"]["parts"][291] / "export.json"))
        assert man["labelled_ids"] == T.W.draw(d)
        k, R = T.resolve_model(f"STOCK_{d}", strict=True)
        assert k == "nnunet" and R["n_classes"] == 15
        assert Path(T.arms()[f"LABELMAP_{d}"]["rules"]).is_file()
    assert T.TIER1[0] == "F0" and T.REQUIRED == ["F0", "F1b_d1"]
    assert all(T.arms()[a]["tier"] == 2 for a in T.TIER2)


@test
def u09_predict_guards():
    import wn_predict as PR
    for argv in (["--arm", "F0", "--smoke", "--ids", "0001", "--root", "/tmp/x", "--images", "/tmp/y"],
                 ["--arm", "F0", "--smoke", "--ids", "0004", "--root", "/tmp/x", "--images", str(T.W.IMAGES)],
                 ["--arm", "F0", "--ids", "0004"], ["--arm", "F0", "--device", "cpu"]):
        try:
            PR.parse(argv)
            raise RuntimeError(f"not refused: {argv}")
        except (AssertionError, SystemExit):
            pass
    a = PR.parse(["--arm", "F0", "--smoke", "--ids", "0004", "--root", "/tmp/x", "--images", "/tmp/y"])
    assert a.smoke
    try:
        PR.parse(["--arm", "MERGE_N_d1"])
        raise RuntimeError("a CPU arm accepted by the predictor")
    except SystemExit:
        pass


@test
def u10_luts():
    import fin_look as L
    import numpy as np
    kinds = {a: T.lut_kind(a) for a in T.RUN_ORDER}
    assert kinds["F0"] == "ts" and kinds["STOCK_d1"] == "id" and kinds["LABELMAP_d1"] == "id"
    assert kinds["MERGE_N_d1"] == "ts" and kinds["V0"] == "vista" and kinds["VF5k_lrA_d1"] == "vista"
    for k in set(kinds.values()):
        lut = L.lut("word", k)
        assert lut.shape == (256,) and lut.dtype == np.uint8 and lut.max() <= 15
    a = T.W.analysis_lut()
    assert a[13] == 10 and a[10] == 10 and a[14] == 13


@test
def u11_cutoff_at_look():
    import wn_score as SC
    with tempfile.TemporaryDirectory(dir="/tmp") as td:
        T.use_root(td)
        for a, fin in (("F0", T.CUTOFF - timedelta(minutes=1)), ("F1b_d1", T.CUTOFF), ("F1b_d2", T.CUTOFF + timedelta(1))):
            T.write_json(T.done_file(a), {"finished": fin.strftime("%Y-%m-%d %H:%M:%S"), "mode": "real"})
        have, nr, _ign = SC.arm_status()
        assert have == ["F0"] and {"F1b_d1", "F1b_d2"} <= set(nr) and "after the cutoff" not in nr.get("F1b_d3", "")
        assert "at or after the cutoff" in nr["F1b_d1"] and set(nr) - {"F1b_d1", "F1b_d2"} == set(T.TIER2)


def _stand_in_maps(root, ids):
    import numpy as np
    import nibabel as nib
    rng = np.random.default_rng(2)
    t = T.W.ts_lut()
    ts_of = {o: int(np.flatnonzero(t == o)[0]) for o in range(1, 14)}
    for a in ("F0", "F1b_d1"):
        per = {}
        for c in ids:
            Y = np.zeros((20, 20, 10), np.uint8)
            Y[2:8, 2:8, 2:6] = ts_of[1]
            Y[10:14, 10:14, 2:6] = ts_of[9] if a == "F1b_d1" else 0
            Y[15:18, 15:18, 1:3] = 52 if a == "F0" else 0      # a non-target TS class (vertebra-like)
            p = Path(root) / a / "raw" / f"word_{c}.nii.gz"
            p.parent.mkdir(parents=True, exist_ok=True)
            nib.save(nib.Nifti1Image(Y, np.eye(4)), str(p))
            per[c] = T.sha256(p)
        T.write_json(Path(root) / a / "DONE.json", {"per_scan_raw_sha256": per})
    _ = rng


@test
def u12_cpu_arms():
    import importlib
    import nibabel as nib
    import numpy as np
    importlib.reload(T)
    import wn_cpu_arms as CPU
    with tempfile.TemporaryDirectory(dir="/tmp") as td:
        ids = ["9001", "9002"]
        _stand_in_maps(td, ids)
        assert CPU.main(["--arm", "MERGE_N_d1", "--smoke-root", td, "--ids", ",".join(ids)]) == 0
        mn = np.asarray(nib.load(str(Path(td) / "MERGE_N_d1" / "raw" / "word_9001.nii.gz")).dataobj)
        f0 = np.asarray(nib.load(str(Path(td) / "F0" / "raw" / "word_9001.nii.gz")).dataobj)
        assert (mn[15:18, 15:18, 1:3] == f0[15:18, 15:18, 1:3]).all()       # non-target class from F0
        assert CPU.main(["--arm", "LABELMAP_d1", "--smoke-root", td, "--ids", ",".join(ids)]) == 0
        rules, _ = CPU.frozen_rules("d1")
        lm = np.asarray(nib.load(str(Path(td) / "LABELMAP_d1" / "raw" / "word_9001.nii.gz")).dataobj)
        assert np.array_equal(lm, np.asarray(rules["lut"], np.uint8)[f0])
        d = json.load(open(Path(td) / "MERGE_N_d1" / "DONE.json"))
        assert set(d["inputs"]) == {"F0_done_sha256", "F1b_d1_done_sha256"}


@test
def u13_duplicates():
    import nibabel as nib
    import numpy as np
    import wn_intake as IN
    with tempfile.TemporaryDirectory(dir="/tmp") as td:
        T.use_root(td)
        rng = np.random.default_rng(4)
        imgs = {c: rng.normal(0, 100, (24, 24, 12)).astype(np.int16) for c in ("0001", "0007", "0014", "0004")}
        imgs["0015"] = imgs["0007"].copy()            # val duplicate of val 0007 -> 0015 dropped
        imgs["0025"] = imgs["0014"].copy()            # val duplicate of test 0014 -> 0025 dropped
        imgs["0031"] = rng.normal(0, 100, (24, 24, 13)).astype(np.int16)   # other shape: never compared
        for c, a in imgs.items():
            Path(T.IMAGES).mkdir(parents=True, exist_ok=True)
            nib.save(nib.Nifti1Image(a, np.eye(4)), str(T.image_path(c)))
        pairs, excl = IN.duplicates(["0001", "0007", "0015", "0025", "0031"], ["0014", "0004"], log=lambda m: None)
        assert set(excl) == {"0015", "0025"}, excl
        assert not any("0031" in p["pair"] for p in pairs)


@test
def u14_runner():
    import importlib
    importlib.reload(T)
    import wn_run as RN
    cmd, gpu, _ = RN.job_cmd("F0")
    assert gpu and "wn_predict.py" in cmd[3]
    cmd, gpu, _ = RN.job_cmd("MERGE_N_d1")
    assert not gpu and "wn_cpu_arms.py" in cmd[3]
    cmd, gpu, _ = RN.job_cmd("train:VF5k_d2")
    assert gpu and "wn_vista.py" in cmd[3] and cmd[-1] == "VF5k_d2"
    cmd, gpu, _ = RN.job_cmd("intake")
    assert not gpu and cmd[-1] == "--run"
    with tempfile.TemporaryDirectory(dir="/tmp") as td:
        T.use_root(td)
        assert not RN.smoke_passed()
        assert RN.inputs_not_run("MERGE_N_d2") == ["F0", "F1b_d2"]
        assert RN.inputs_not_run("F1b_d3") == ["F0"]
        assert T.RUN_ORDER.index("F0") < T.RUN_ORDER.index("F1b_d1") < T.RUN_ORDER.index("MERGE_N_d1") < \
            T.RUN_ORDER.index("V0")


@test
def u15_read_only_records():
    with tempfile.TemporaryDirectory(dir="/tmp") as td:
        p = Path(td) / "r.json"
        T.write_json(p, {"a": 1}, mode=0o444)
        assert refused(lambda: T.write_json(p, {"a": 2}))
        assert json.load(open(p)) == {"a": 1}


@test
def u16_acceptance():
    import wn_analysis as AN
    R = synth_R()
    ids = sorted(R)
    acc = AN.acceptance_check(R, ids, set(R[ids[0]]["NATIVE"]["arms"]))
    dec = acc["all"]["decisions"]
    assert dec["F1b_d1"]["accept"] and not dec["STOCK_d1"]["accept"]
    assert acc["all"]["k_correct"] == 6 and len(acc["first_10_seeded"]["decisions"]) == 6


@test
def u17_holm_missing_member():
    import wn_analysis as AN
    R = synth_R(arms_=["F0", "F1b_d1", "F1b_d2", "F1b_d3"])
    ids = sorted(R)
    res = AN.analyse(R, ids, {"F0", "F1b_d1", "F1b_d2", "F1b_d3"}, {}, mode="SPLIT")
    h = res["holm_family"]
    assert h["m"] == 4 and h["m_used_in_holm"] == 4 and "S1-fresh" not in h["evaluable_members"]
    r = h["results"]
    assert r["S1-fresh"]["reading"] == "NOT RUN" and r["S1-fresh"]["p_holm"] == 1.0
    ps = sorted(v["p_iut"] for k, v in r.items() if k != "S1-fresh")
    assert abs(min(v["p_holm"] for v in r.values()) - min(1.0, 4 * ps[0])) < 1e-12      # smallest p times m = 4


@test
def u18_not_run_binding():
    import wn_score as SC
    with tempfile.TemporaryDirectory(dir="/tmp") as td:
        T.use_root(td)
        T.write_json(T.RUN / "F1b_d2.json", {"not_run": "2 failed attempts"})
        assert refused(lambda: T.arm_allowed("F1b_d2")) and refused(lambda: T.arm_allowed("V0"))
        T.write_json(T.VSMOKE / "smoke.json", {"decision": "VSMOKE_PASS"})
        T.arm_allowed("V0")
        ok = (T.CUTOFF - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
        for a in ("F0", "F1b_d1", "F1b_d2", "V0"):
            T.write_json(T.done_file(a), {"finished": ok, "mode": "real"})
        T.write_json(T.done_file("F1b_d3"), {"finished": ok, "mode": "smoke"})
        have, nr, ign = SC.arm_status()
        assert have == ["F0", "F1b_d1", "V0"] and "F1b_d2" in ign and "F1b_d3" in ign, (have, ign)
        T.write_json(T.VSMOKE / "smoke.json", {"decision": "VSMOKE_FAIL"})
        have, nr, ign = SC.arm_status()
        assert "V0" not in have and "V0" in ign
        old = T.CUTOFF
        try:
            T.CUTOFF = datetime.now() - timedelta(minutes=1)
            have, nr, ign = SC.arm_status()
            assert nr.get("STOCK_d1", "").startswith("not DONE by the cutoff")
        finally:
            T.CUTOFF = old


@test
def u19_gpu_identity_and_cutoff_waits():
    import fcntl
    old_env = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        for v in ("", "1", "0", "GPU-ac2e0ec4-177f-2ddf-61d7-17ca51e21e39"):
            os.environ["CUDA_VISIBLE_DEVICES"] = v
            assert refused(T.gpu_identity_gate), v
    finally:
        os.environ["CUDA_VISIBLE_DEVICES"] = old_env if old_env is not None else ""
    with tempfile.TemporaryDirectory(dir="/tmp") as td:
        lk = Path(td) / "gpu1.lock"
        lk.write_text("")
        fd = os.open(str(lk), os.O_RDONLY)
        fcntl.flock(fd, fcntl.LOCK_EX)
        old = T.CUTOFF
        try:
            T.CUTOFF = datetime.now() - timedelta(minutes=1)
            try:
                T.acquire_gpu_lock(lk, poll_s=0.01)
                raise RuntimeError("lock acquired while held")
            except T.Refuse as e:
                assert e.code == T.CUTOFF_CODE
            try:
                T.wait_gpu1_idle(poll_s=0.01, used_fn=lambda: 9999)
                raise RuntimeError("idle wait returned")
            except T.Refuse as e:
                assert e.code == T.CUTOFF_CODE
        finally:
            T.CUTOFF = old
            os.close(fd)


@test
def u20_runner_declares_not_run():
    import importlib
    importlib.reload(T)
    import wn_run as RN
    with tempfile.TemporaryDirectory(dir="/tmp") as td:
        T.use_root(td)
        orig_gates, orig_run, orig_check, orig_backoff = RN.gates, RN.run_job, T.check_frozen_model, RN.backoff
        try:
            RN.gates = lambda: {}
            T.check_frozen_model = lambda a: None

            calls = {"n": 0, "backoff": 0}

            def res_then_done(key):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise T.Refuse("disk floor", code=3)
                T.write_json(T.done_file("F0"), {"finished": T.now(), "mode": "real"})
                return "done"

            def fake_backoff(seconds=0):
                calls["backoff"] += 1
                return True
            RN.run_job, RN.backoff = res_then_done, fake_backoff
            assert RN.handle_arm("F0") == "done" and not RN.not_run("F0") and calls == {"n": 2, "backoff": 1}
            RN.run_job = lambda key: "resource"
            RN.backoff = lambda seconds=0: False                 # the cutoff passed during the back-off
            assert RN.handle_arm("F1b_d1") == "cutoff" and not RN.not_run("F1b_d1")
            seen = {"backoff": 0, "n": 0}

            def fail_then_done(key):          # a failed attempt (e.g. a passing outage) -> back-off -> attempt 2 (NB2)
                seen["n"] += 1
                st = RN.state(key)
                if seen["n"] == 1:
                    st["failures"] = st.get("failures", 0) + 1
                    RN.save(key, st)
                    return "failed"
                T.write_json(T.done_file("STOCK_d1"), {"finished": T.now(), "mode": "real"})
                return "done"

            def count_backoff(seconds=0):
                seen["backoff"] += 1
                return True
            RN.run_job, RN.backoff = fail_then_done, count_backoff
            assert RN.handle_arm("STOCK_d1") == "done" and seen == {"backoff": 1, "n": 2}
            RN.backoff = orig_backoff

            def boom(key):
                raise ValueError("unexpected")
            RN.run_job = boom
            assert RN.handle_arm("F1b_d2") == "not_run" and "unexpected" in RN.state("F1b_d2")["not_run"]

            def done_then_boom(key):
                T.write_json(T.done_file("F1b_d3"), {"finished": T.now(), "mode": "real"})
                raise ValueError("state write failed")
            RN.run_job = done_then_boom
            assert RN.handle_arm("F1b_d3") == "done" and not RN.not_run("F1b_d3")
            T.write_json(T.VSMOKE / "smoke.json", {"decision": "VSMOKE_PASS"})
            T.write_json(T.RUN / "V0.json", {"not_run": "x"})
            assert RN.handle_arm("VF5k_d1") == "not_run" and "V0 is NOT RUN" in RN.state("VF5k_d1")["not_run"]
        finally:
            RN.gates, RN.run_job, T.check_frozen_model, RN.backoff = orig_gates, orig_run, orig_check, orig_backoff


@test
def u21_gpu_infrastructure_outage_is_waited_out():
    """NB2: an nvidia-smi outage, an unlisted UUID and a missing lock file are waited for (cutoff-aware), never a
    failure; a wrong CUDA_VISIBLE_DEVICES stays a configuration error."""
    import threading
    import types
    orig_rows, orig_env, orig_run = T.nvsmi_rows, os.environ.get("CUDA_VISIBLE_DEVICES"), T.subprocess.run
    try:
        n = {"k": 0}

        def flaky():
            n["k"] += 1
            if n["k"] <= 2:
                raise T.InfraDown("Failed to initialize NVML: Driver/library version mismatch")
            if n["k"] == 3:
                return [("GPU-ac2e0ec4-177f-2ddf-61d7-17ca51e21e39", "00000000:03:00.0", 10)]     # UUID not listed yet
            return [(T.GPU1_UUID, "00000000:05:00.0", 10)]
        T.nvsmi_rows = flaky
        os.environ["CUDA_VISIBLE_DEVICES"] = T.GPU1_UUID
        assert T.gpu_identity_gate(log=lambda m: None, poll_s=0.01) is True and n["k"] == 4
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        try:
            T.gpu_identity_gate(log=lambda m: None, poll_s=0.01)
            raise RuntimeError("wrong CUDA_VISIBLE_DEVICES accepted")
        except T.Refuse as e:
            assert e.code == 2
        T.nvsmi_rows = orig_rows
        T.subprocess.run = lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="Failed to initialize NVML", stderr="")
        try:
            T.nvsmi_rows()
            raise RuntimeError("garbage parsed")
        except T.InfraDown:
            pass
        T.subprocess.run = orig_run
        old = T.CUTOFF
        try:
            T.CUTOFF = datetime.now() - timedelta(minutes=1)
            T.nvsmi_rows = lambda: (_ for _ in ()).throw(T.InfraDown("down"))
            os.environ["CUDA_VISIBLE_DEVICES"] = T.GPU1_UUID
            try:
                T.gpu_identity_gate(log=lambda m: None, poll_s=0.01)
                raise RuntimeError("outage after the cutoff not refused")
            except T.Refuse as e:
                assert e.code == T.CUTOFF_CODE
        finally:
            T.CUTOFF = old
            T.nvsmi_rows = orig_rows
        with tempfile.TemporaryDirectory(dir="/tmp") as td:
            lk = Path(td) / "gpu1.lock"
            threading.Timer(0.2, lambda: lk.write_text("")).start()           # the other project's tooling recreates it
            fd = T.acquire_gpu_lock(lk, poll_s=0.01, missing_poll_s=0.02, log=lambda m: None)
            T.release_gpu_lock(fd)
            assert lk.read_text() == ""                                        # never written by WORD-N
    finally:
        T.nvsmi_rows, T.subprocess.run = orig_rows, orig_run
        os.environ["CUDA_VISIBLE_DEVICES"] = orig_env if orig_env is not None else ""


@test
def u22_gpu_session_release_and_repeat():
    import fcntl
    orig = (T.gpu_identity_gate, T.gpu1_used_mib, T.GPU_LOCK)
    with tempfile.TemporaryDirectory(dir="/tmp") as td:
        lk = Path(td) / "gpu1.lock"
        lk.write_text("")
        seq = iter([10, 9000, 10, 10])        # idle wait: idle; re-check: busy -> release; idle wait: idle; re-check: idle
        held_during_busy = {}

        def used():
            v = next(seq)
            if v >= T.GPU_IDLE_MIB:
                fd2 = os.open(str(lk), os.O_RDONLY)
                try:                          # the session holds the lock while it re-checks
                    fcntl.flock(fd2, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    held_during_busy["held"] = False
                    fcntl.flock(fd2, fcntl.LOCK_UN)
                except BlockingIOError:
                    held_during_busy["held"] = True
                finally:
                    os.close(fd2)
            return v
        try:
            T.gpu_identity_gate = lambda **k: True
            T.gpu1_used_mib = used
            T.GPU_LOCK = lk
            msgs = []
            fd = T.gpu_session(log=msgs.append, idle_poll_s=0.01, lock_poll_s=0.01, poll_s=0.01)
            fd2 = os.open(str(lk), os.O_RDONLY)
            try:
                fcntl.flock(fd2, fcntl.LOCK_EX | fcntl.LOCK_NB)
                raise RuntimeError("the session does not hold the lock")
            except BlockingIOError:
                pass
            finally:
                os.close(fd2)
            T.release_gpu_lock(fd)
            assert held_during_busy == {"held": True}
            assert any("lock released" in m for m in msgs), msgs
            assert next(seq, None) is None                                     # exactly four readings
        finally:
            T.gpu_identity_gate, T.gpu1_used_mib, T.GPU_LOCK = orig


def main():
    out, ok = {}, True
    for fn in TESTS:
        try:
            fn()
            out[fn.__name__] = "PASS"
        except Exception:                            # noqa: BLE001
            out[fn.__name__] = "FAIL: " + traceback.format_exc()[-1500:]
            ok = False
        print(f"{fn.__name__}: {out[fn.__name__][:300]}", flush=True)
    if sys.argv[1:] == ["--record"]:
        import importlib
        importlib.reload(T)
        rec = {"all_pass": ok, "tests": out, "n": len(out), "written": T.now(),
               "code_sha256": {p.name: T.sha256(p) for p in sorted(HERE.glob("*.py"))},
               "python": sys.version.split()[0]}
        if T.FREEZE.exists():
            raise SystemExit("[REFUSE] the freeze exists: unit_tests.json is not rewritten")
        T.write_json(T.UNIT_TESTS, rec)
        print(f"recorded {T.rel(T.UNIT_TESTS)}")
    print(f"all_pass {ok} ({sum(v == 'PASS' for v in out.values())}/{len(out)})")
    return 0 if ok else 1


if __name__ == "__main__":
    _ = subprocess
    sys.exit(main())
