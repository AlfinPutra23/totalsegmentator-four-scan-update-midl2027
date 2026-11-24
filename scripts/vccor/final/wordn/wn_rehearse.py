"""vccor/final WORD-N LOOK REHEARSAL (CPU; before the freeze). Runs the real look code (wn_score.py, wn_analysis.py,
wn_intake.py) end to end under wn_common.use_root in a scratch world under /tmp: a stand-in AES-zip archive built with 7z
(the WORD password; 20 synthetic validation images and labels with the real validation ids), synthetic draw and test
images, stand-in prediction maps of every arm in its own label space (TS / analysis / VISTA ids) with DONE records, stand-in
freeze, plan and gate files. No real image, label, prediction or model is read; the real look folder, archive and log are
never touched (asserted).

Scenarios (each in a fresh child process and world):
  ok                         the full look: one GT_OPENED, LABELS and RESULTS line; P-V REDUCED and OS-D OVERSEGMENTED
                             (built into the synthetic world); the intake drops the planted duplicate 0149 (= test 0016)
  crash_scoring              scoring crashes after 3 scans -> GT_CRASHED; --continue -> RESULTS; the analysis equals 'ok'
  crash_before_labels_json   crash before labels.json -> GT_CRASHED; --continue re-extracts -> RESULTS
  label_swapped_after_archive  a label file changed after the archive -> --continue restores it (GT_NOTE) -> RESULTS
  corrupt_label              a label with a value outside 0..16 -> GT_ABORTED; --continue refused
  archive_changed            the archive differs from the pinned listing -> refused BEFORE the marker
  missing_arm                an arm neither DONE nor NOT RUN -> refused before the marker
  done_after_cutoff          a DONE written after the cutoff -> that arm NOT RUN; the look runs
  second_look                after a finished look -> refused
  ac_combined_mode           the plan's overshoot_mode AC_COMBINED -> the family holds H-D / H-E
  not_run_binding            F1b_d3 declared NOT RUN by the runner although a DONE exists -> ignored (logged), NOT RUN;
                             MERGE_N_d3 (built on F1b_d3) is demoted too (N-D1)
  smoke_failed               smoke.json says VSMOKE_FAIL -> every tier-2 arm NOT RUN, V-P1 NOT RUN; tier 1 unaffected
  runner_dead_after_cutoff   an arm neither DONE nor declared and no RUN_FINISHED, cutoff passed -> NOT RUN; the look runs

usage:  .venv/bin/python -I -B scripts/vccor/final/wordn/wn_rehearse.py [--root /tmp/...]

Version of 2026-11-13.
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
             "archive_changed", "missing_arm", "done_after_cutoff", "second_look", "ac_combined_mode", "not_run_binding",
             "smoke_failed", "runner_dead_after_cutoff"]
SHAPE = (40, 40, 28)
DUP_VAL, DUP_TEST = "0149", "0016"
# raw WORD ids of the synthetic organs and their boxes (x0, x1, y0, y1, z0, z1) before per-scan jitter
BOXES = {1: (4, 18, 4, 18, 4, 14), 2: (22, 30, 4, 12, 4, 10), 3: (22, 28, 14, 20, 12, 18), 4: (4, 10, 22, 28, 12, 18),
         5: (12, 20, 22, 30, 4, 10), 6: (31, 37, 3, 9, 3, 9), 7: (32, 36, 13, 17, 2, 24), 8: (12, 20, 32, 36, 14, 20),
         9: (22, 30, 24, 30, 18, 26), 11: (4, 14, 30, 38, 20, 26), 12: (32, 36, 22, 26, 15, 20), 14: (16, 22, 10, 16, 20, 26)}
SOLID_RAW = (1, 2, 3, 4, 8)
TUBE_RAW = (7, 9)


def guard(root):
    root = Path(root).resolve()
    assert str(root).startswith("/tmp/"), root
    return root


# ---------------------------------------------------------------------------------------------------- synthetic world
def boxes_for(rng, k):
    out = {}
    for o, (x0, x1, y0, y1, z0, z1) in BOXES.items():
        j = rng.integers(-1, 2, size=6)
        out[o] = (x0 + j[0], x1 + j[1], y0 + j[2], y1 + j[3], z0 + j[4] // 2, z1 + j[5] // 2)
    return out


def paint(bx, grow=None, shape=SHAPE):
    import numpy as np
    G = np.zeros(shape, np.uint8)
    for o, (x0, x1, y0, y1, z0, z1) in bx.items():
        g = (grow or {}).get(o, 0)
        G[max(0, x0 - g):x1 + g, max(0, y0 - g):y1 + g, max(0, z0 - g):z1 + g] = o
    return G


def image_of(G, rng):
    import numpy as np
    return (G.astype(np.float32) * 40.0 - 500.0 + rng.normal(0, 30, G.shape)).astype(np.int16)


def save(arr, path, dtype=None):
    import nibabel as nib
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    aff = np.diag([1.5, 1.5, 3.0, 1.0])
    im = nib.Nifti1Image(np.asarray(arr), aff)
    if dtype is not None:
        im.set_data_dtype(dtype)
    nib.save(im, str(path))
    return path


def inv(lut):
    import numpy as np
    out = np.zeros(256, np.uint8)
    for o in range(1, 16):
        hits = np.flatnonzero(lut == o)
        out[o] = hits[0] if len(hits) else 0
    return out


def arm_map(T, W, arm, G_raw, rng):
    """A stand-in prediction in the arm's own label space: from the raw reference (analysis ids) with arm-specific
    shrink / grow: F0 under-segments the solid and tubular organs; F1b fits the solid organs and over-segments the duodenum
    and the oesophagus; STOCK and VISTA arms lie in between."""
    import numpy as np
    from scipy import ndimage
    A = W.analysis_lut()[G_raw]
    out = np.zeros_like(A)
    role = T.arms()[arm]["role"]
    for o in range(1, 16):
        m = A == o
        if not m.any():
            continue
        r = rng.random()
        if role in ("zero-shot", "LABELMAP", "V0"):
            m2 = ndimage.binary_erosion(m) if (r < 0.8) else m
        elif role in ("F1b", "MERGE_N"):
            if o in (7, 9):
                m2 = ndimage.binary_dilation(m, iterations=1 if r < 0.8 else 2)
            else:
                m2 = m if r < 0.7 else ndimage.binary_erosion(m)
        else:
            m2 = m if r < 0.5 else ndimage.binary_erosion(m)
        out[m2 & (out == 0)] = o
    kind = T.lut_kind(arm)
    if kind == "ts":
        return inv(W.ts_lut())[out]
    if kind == "vista":
        return inv(W.vista_lut())[out]
    return out


def build_world(root, scen):
    import numpy as np
    import wn_common as T
    root = guard(root)
    T.use_root(root, c_word=root / "C_word")
    if scen == "runner_dead_after_cutoff":
        T.CUTOFF = datetime.now() - timedelta(hours=1)
    W = T.W
    for p in (T.ARCHIVE, T.LOOK, T.PREREG_LOG):
        assert str(Path(p)).startswith(str(root)), p
    rng = np.random.default_rng(20261113)
    stage = root / "stage" / "WORD-V0.1.0"
    others = sorted(set(W.draw_scans()) | set(W.split()["test"]))
    test_imgs = {}
    for c in others:
        bx = boxes_for(rng, c)
        img = image_of(paint(bx), rng)
        save(img, T.image_path(c), np.int16)
        test_imgs[c] = img
    refs = {}
    for c in T.VAL_IDS:
        bx = boxes_for(rng, c)
        G = paint(bx)
        if scen == "corrupt_label" and c == "0007":
            G[0, 0, 0] = 99
        img = test_imgs[DUP_TEST] if c == DUP_VAL else image_of(G, rng)
        save(img, stage / "imagesVal" / f"word_{c}.nii.gz", np.int16)
        save(G, stage / "labelsVal" / f"word_{c}.nii.gz", np.uint8)
        refs[c] = G
    T.ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["/usr/bin/7z", "a", "-tzip", f"-p{T.PASSWORD}", "-mem=AES128", str(T.ARCHIVE), "WORD-V0.1.0"],
                       cwd=str(stage.parent), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    shutil.rmtree(root / "stage")
    # plan (overshoot mode) and the member pin
    import wn_plan
    plan = dict(wn_plan.PLAN, overshoot_mode="AC_COMBINED" if scen == "ac_combined_mode" else "SPLIT")
    T.write_json(T.PLAN, plan)
    import wn_intake
    wn_intake.pin()
    # stand-in freeze (empty model hash sets; versions as installed) and gate files
    T.write_json(T.FREEZE, {"written": (datetime.now() - timedelta(hours=4)).strftime("%Y-%m-%d %H:%M:%S"), "files": {},
                            "models": {"TS_WEIGHTS": {}, "VISTA_BUNDLE": {}, "RULES_d1": {}, "RULES_d2": {},
                                       "RULES_d3": {}, **{a: {} for a in T.RUN_ORDER}},
                            "pending_models": [], "versions": T.versions()})
    import zipfile
    zp = root / "osf_deposit_2026-11-15_wordn.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.write(T.FREEZE, "code_freeze.json")
    zsha = T.sha256(zp)
    with open(T.PREREG_LOG, "a") as f:
        f.write(f"[{T.now()}] DEPOSIT sha256 {zsha} {zp} (rehearsal)\n")
    T.write_json(T.VSMOKE / "smoke.json", {"decision": "VSMOKE_FAIL" if scen == "smoke_failed" else "VSMOKE_PASS"})
    up = (datetime.now() - timedelta(hours=3)).strftime("%Y-%m-%d %H:%M:%S")
    T.write_json(T.GATE_UPLOADED, {"osf_url": "https://osf.io/abcde", "uploaded_kst": up, "written_by": "rehearsal",
                                   "evidence": "rehearsal", "deposit_zip_sha256": zsha, "freeze_sha256": T.sha256(T.FREEZE)})
    T.write_json(T.GATE_PUBLIC, {"osf_url": "https://osf.io/abcde", "api_http_status": 200, "public": True,
                                 "public_checked_kst": (datetime.now() - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"),
                                 "written_by": "rehearsal", "evidence": "rehearsal"})
    # the intake on the stand-in archive (drops the planted duplicate)
    wn_intake.run()
    ids = T.included_ids()
    # stand-in predictions of every arm, DONE records
    fin = (T.CUTOFF + timedelta(minutes=5)) if scen == "done_after_cutoff" else (datetime.now() - timedelta(hours=2))
    for a in T.RUN_ORDER:
        if scen in ("missing_arm", "runner_dead_after_cutoff") and a == "VF5k_lrA_d1":
            continue
        if a == "VF5k_lrA_d1" and scen != "done_after_cutoff":
            T.write_json(T.RUN / f"{a}.json", {"arm": a, "attempts": 0, "failures": 0, "history": [],
                                               "not_run": "rehearsal: declared NOT RUN"})
            continue
        pd = T.pred_dir(a)
        per = {}
        for c in ids:
            Y = arm_map(T, W, a, refs[c], rng)
            p = save(Y.astype(np.uint8), pd / "raw" / f"word_{c}.nii.gz", np.uint8)
            per[c] = T.sha256(p)
        done = {"arm": a, "ids": ids, "per_scan_raw_sha256": per, "model_sha256": {}, "mode": "real",
                "finished": (fin if a == "VF5k_d3" else datetime.now() - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S")}
        if T.arms()[a]["kind"] == "cpu":
            done["inputs"] = {}
        T.write_json(pd / "DONE.json", done)
    for a in T.RUN_ORDER:                       # CPU arms name their inputs' DONE records
        if T.arms()[a]["kind"] == "cpu" and T.done_file(a).is_file():
            d = json.load(open(T.done_file(a)))
            d["inputs"] = {f"{i}_done_sha256": T.sha256(T.done_file(i)) for i in T.arms()[a]["inputs"]}
            if a.startswith("LABELMAP"):
                d["merged"] = {}
            if a.startswith("MERGE_N"):
                d["per_scan_merge"] = {c: {"nontarget_f0_f1b_both": {}, "changed_vs_f1b": {}, "n_changed_vs_f1b": 0,
                                           "vox_ml": 0.00675} for c in ids}
            T.write_json(T.done_file(a), d)
    if scen == "not_run_binding":
        T.write_json(T.RUN / "F1b_d3.json", {"arm": "F1b_d3", "attempts": 2, "failures": 2, "history": [],
                                            "not_run": "rehearsal: 2 failed attempts"})
    if scen != "runner_dead_after_cutoff":
        T.write_json(T.RUN / "RUN_FINISHED.json", {"finished": T.now()})
    if scen == "archive_changed":
        extra = root / "stage2" / "WORD-V0.1.0" / "labelsVal" / "word_0001.nii.gz"
        save(refs["0001"] * 0, extra, np.uint8)
        subprocess.run(["/usr/bin/7z", "a", "-tzip", f"-p{T.PASSWORD}", "-mem=AES128", str(T.ARCHIVE), "WORD-V0.1.0"],
                       cwd=str(root / "stage2"), capture_output=True, check=True)
    return T, refs


# ---------------------------------------------------------------------------------------------------- child
def lines(T, kind):
    return [ln for ln in T.prereg_lines() if f"] {kind} " in ln]


def child(scen, root):
    T, _refs = build_world(root, scen)
    import wn_score as SC
    out = {"scenario": scen}

    def attempt(cont=False):
        try:
            return SC.look(cont=cont)
        except T.Refuse as e:
            return f"REFUSE: {e.msg}"
        except BaseException as e:                 # noqa: BLE001
            return f"CRASH: {type(e).__name__}: {e}"

    if scen in ("ok", "done_after_cutoff", "ac_combined_mode", "not_run_binding", "smoke_failed",
                "runner_dead_after_cutoff"):
        out["first"] = attempt()
    elif scen == "crash_scoring":
        real = SC.score_one
        n = {"k": 0}
        import multiprocessing.pool as MP
        orig_imap = MP.Pool.imap_unordered

        def imap(self, fn, it, chunksize=1):
            for i, r in enumerate(orig_imap(self, fn, it, chunksize)):
                if i == 3:
                    raise RuntimeError("rehearsal crash during scoring")
                yield r
        MP.Pool.imap_unordered = imap
        out["first"] = attempt()
        MP.Pool.imap_unordered = orig_imap
        _ = (real, n)
        out["pervol_after_crash"] = len(list((T.LOOK / "pervol").glob("*.json")))
        out["second"] = attempt(cont=True)
    elif scen == "crash_before_labels_json":
        orig = SC.finish_labels

        def boom(got):
            raise RuntimeError("rehearsal crash before labels.json")
        SC.finish_labels = boom
        out["first"] = attempt()
        SC.finish_labels = orig
        out["second"] = attempt(cont=True)
    elif scen == "label_swapped_after_archive":
        orig = SC.sanity_gate

        def stop(ids):
            raise RuntimeError("rehearsal stop after the archive")
        SC.sanity_gate = stop
        out["first"] = attempt()
        SC.sanity_gate = orig
        p = T.label_path("0001")
        os.chmod(p, 0o644)
        p.write_bytes(p.read_bytes()[:-10] + b"0123456789")
        out["second"] = attempt(cont=True)
    elif scen == "corrupt_label":
        out["first"] = attempt()
        out["second"] = attempt(cont=True)
    elif scen in ("archive_changed", "missing_arm"):
        out["first"] = attempt()
    elif scen == "second_look":
        out["first"] = attempt()
        out["second"] = attempt()
    out["marker"] = (T.LOOK / "GT_OPENED").exists()
    out["results"] = (T.LOOK / "results.json").exists()
    out["n_lines"] = {k: len(lines(T, k)) for k in ("GT_OPENED", "LABELS", "RESULTS", "GT_CRASHED", "GT_ABORTED",
                                                    "GT_CONTINUED", "GT_NOTE")}
    if out["results"]:
        res = json.load(open(T.LOOK / "results.json"))
        out["readings"] = {"P_V": res["P_V"]["reading"], "P_V_M": res["P_V"]["estimate"],
                           "family": {k: v["reading"] for k, v in res["holm_family"]["results"].items()},
                           "tier2": res["tier2"]["V_P1"]["reading"], "n_scans": res["n_scans"],
                           "arms_not_run": sorted(res["arms_not_run"]), "panels": res.get("panels"),
                           "panels_rendered": sorted((res.get("panels_rendered") or {}).keys()),
                           "m_used": res["holm_family"]["m_used_in_holm"]}
        out["ignored_done"] = sorted(json.load(open(T.LOOK / "maps.json")).get("ignored_done", {}))
        out["excluded"] = sorted(json.load(open(T.EXCLUSIONS))["excluded"])
    print("REHEARSAL_RESULT " + json.dumps(out, default=str))


def expect(r):
    """-> list of failed expectations for one scenario result."""
    s, bad = r["scenario"], []
    nl = r.get("n_lines", {})

    def want(c, msg):
        if not c:
            bad.append(msg)
    if s in ("ok", "done_after_cutoff", "ac_combined_mode", "crash_scoring", "crash_before_labels_json",
             "label_swapped_after_archive", "second_look", "not_run_binding", "smoke_failed", "runner_dead_after_cutoff"):
        want(r["results"], "results.json missing")
        want(nl.get("GT_OPENED") == 1 and nl.get("RESULTS") == 1 and nl.get("LABELS") == 1, f"log lines {nl}")
        rd = r.get("readings") or {}
        want(rd.get("P_V") == "VOLUME_ERROR_REDUCED", f"P_V {rd.get('P_V')}")
        want(r.get("excluded") == [DUP_VAL], f"excluded {r.get('excluded')}")
        want(rd.get("n_scans") == 19, f"n_scans {rd.get('n_scans')}")
        fam = rd.get("family", {})
        if s == "ac_combined_mode":
            want(set(fam) == {"H-D", "H-E", "P1-fresh", "S1-fresh"}, f"family {fam}")
        else:
            want(set(fam) == {"OS-D", "OS-E", "P1-fresh", "S1-fresh"}, f"family {fam}")
            want(fam.get("OS-D") == "OVERSEGMENTED", f"OS-D {fam.get('OS-D')}")
        want(len(rd.get("panels_rendered", [])) == 2, f"panels {rd.get('panels_rendered')}")
        want(rd.get("m_used") == 4, f"Holm m used {rd.get('m_used')}")
    if s == "not_run_binding":
        nrr = (r.get("readings") or {}).get("arms_not_run", [])
        want({"F1b_d3", "MERGE_N_d3"} <= set(nrr) and r.get("ignored_done") == ["F1b_d3", "MERGE_N_d3"],
             f"binding NOT RUN (F1b_d3; MERGE_N_d3 demoted because its input does not count): {r.get('ignored_done')}")
    if s == "smoke_failed":
        rd = r.get("readings") or {}
        want(rd.get("tier2") == "NOT RUN" and set(rd.get("arms_not_run", [])) >= {"V0", "VF5k_d1", "VF5k_d2", "VF5k_d3"},
             f"smoke failed: {rd.get('tier2')} {rd.get('arms_not_run')}")
    if s == "runner_dead_after_cutoff":
        want("VF5k_lrA_d1" in (r.get("readings") or {}).get("arms_not_run", []), "undeclared arm not NOT RUN")
    if s == "crash_scoring":
        want(nl.get("GT_CRASHED") == 1 and nl.get("GT_CONTINUED") == 1, f"crash/continue lines {nl}")
    if s == "crash_before_labels_json":
        want(nl.get("GT_CRASHED") == 1 and nl.get("GT_CONTINUED") == 1, f"lines {nl}")
    if s == "label_swapped_after_archive":
        want(nl.get("GT_NOTE") == 1, f"GT_NOTE {nl}")
    if s == "corrupt_label":
        want(nl.get("GT_ABORTED") == 1 and not r["results"], f"abort {nl}")
        want(str(r.get("second", "")).startswith("REFUSE"), f"continue after abort: {r.get('second')}")
    if s in ("archive_changed", "missing_arm"):
        want(not r["marker"] and str(r.get("first", "")).startswith("REFUSE"), f"{s}: {r.get('first')}")
    if s == "done_after_cutoff":
        want("VF5k_d3" in (r.get("readings") or {}).get("arms_not_run", []), "VF5k_d3 should be NOT RUN")
    if s == "second_look":
        want(str(r.get("second", "")).startswith("REFUSE"), f"second look: {r.get('second')}")
    return bad


def record_result(out):
    import wn_common as T
    rec = {"all_pass": all(not v["failed"] for v in out.values()), "scenarios": out, "written": T.now(),
           "code_sha256": T.code_hashes(sorted(p.name for p in HERE.glob("*.py"))),
           "note": "stand-in archive, images, labels, predictions, freeze and gates under /tmp; no real data read"}
    if T.FREEZE.exists():
        k = 1
        while (T.OUT / f"rehearsal_postfreeze_{k}.json").exists():
            k += 1
        T.write_json(T.OUT / f"rehearsal_postfreeze_{k}.json", rec)
    else:
        T.write_json(T.REHEARSAL_RECORD, rec)
    return rec


def parent(root_arg=None):
    base = Path(root_arg) if root_arg else Path(tempfile.mkdtemp(prefix="wn_rehearse_", dir="/tmp"))
    base = guard(base)
    out = {}
    for s in SCENARIOS:
        root = base / s
        root.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
        r = subprocess.run([sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--child", s, str(root)],
                           capture_output=True, text=True, env=env, timeout=3600)
        res = None
        for ln in r.stdout.splitlines():
            if ln.startswith("REHEARSAL_RESULT "):
                res = json.loads(ln[len("REHEARSAL_RESULT "):])
        if res is None:
            out[s] = {"failed": [f"child rc {r.returncode}: {r.stderr[-800:]}"]}
        else:
            out[s] = dict(res, failed=expect(res))
        print(f"{s}: {'PASS' if not out[s]['failed'] else 'FAIL ' + str(out[s]['failed'])}", flush=True)
    ok_r = (out.get("ok") or {}).get("readings")
    for s in ("crash_scoring", "crash_before_labels_json", "label_swapped_after_archive"):
        r = (out.get(s) or {}).get("readings")
        if ok_r and r and (r["P_V_M"] != ok_r["P_V_M"] or r["family"] != ok_r["family"]):
            out[s]["failed"].append("continued look's analysis differs from the uninterrupted look")
    rec = record_result(out)
    print(f"rehearsal all_pass {rec['all_pass']}")
    return 0 if rec["all_pass"] else 1


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["--child"]:
        child(a[1], a[2])
    elif a[:1] == ["--root"]:
        sys.exit(parent(a[1]))
    else:
        sys.exit(parent())
