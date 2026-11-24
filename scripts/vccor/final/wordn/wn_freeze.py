"""vccor/final WORD-N FREEZE and OSF DEPOSIT BUNDLE.

  --freeze    After the independent code review, the unit tests, the look rehearsal, the VISTA3D smoke test and the archive
              member pin, and before the author uploads the registration: hashes every WORD-N code file, every project
              module the code imports (found by importing the code; plus the lazily imported TS pipeline and VISTA3D
              engine), the records (plan, member pin, unit tests, rehearsal, smoke record and the smoke's V0 draw maps and
              composite VISTA labels, the registration skeleton), every tier-1 model file, TotalSegmentator's own weights
              (models['TS_WEIGHTS']), the VISTA3D bundle (models['VISTA_BUNDLE']), A1.2's LABELMAP rules
              (models['RULES_dX']) into I_wordn/code_freeze.json (read-only), with the package versions, a FREEZE line and
              one CODE line per code file. The VF tier-2 models (trained after the upload gate) are listed as pending.
              Refuses unless the plan fixes overshoot_mode, the unit tests pass on the current code, the rehearsal passed on
              the current wn_score.py and the member pin exists. The smoke test may have failed (then tier 2 is NOT RUN).
  --addendum  Hashes VF models trained after the freeze into I_wordn/freeze_addendum_<n>.json, before that arm's own
              prediction (the runner calls it); its time relative to the upload is disclosed. A model is hashed only if its
              export.json names the frozen labels manifest, the frozen wn_vista.py / wn_common.py and the pinned VISTA3D
              bundle, is not a smoke run, and its training finished after the upload gate's time and before the cutoff;
              otherwise it is listed as rejected (its arm then fails the freeze check: NOT RUN).
  --bundle    Builds the ONE outputs/results/vccor/osf_deposit_<date>_wordn/ and its zip (code, frozen imports, plan,
              records, freeze, skeleton, OSF guide, README) for the author to upload under osf.io/dekmv; DEPOSIT lines. No
              image, label or model is included. Refuses if a wordn bundle exists.

usage:  .venv/bin/python -I -B scripts/vccor/final/wordn/wn_freeze.py --freeze | --addendum | --bundle

Version of 2026-11-13.
"""
import json
import os
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import wn_common as T  # noqa: E402

R = T.REPO
LAZY_IMPORTS = [R / "scripts/vccor/tsft_full/ts_pipeline.py", R / "scripts/vccor/vista/vista_engine.py",
                R / "scripts/vccor/vista/vista_common.py", R / "scripts/vccor/paper_v2/v2_common.py"]
PENDING_RULE = ("a VF model is hashed by a freeze addendum before its own arm's prediction; the addendum's time relative to "
                "the upload is disclosed; an arm whose model fails the freeze check is NOT RUN")


def code_files():
    return sorted([p for p in HERE.iterdir() if p.suffix in (".py", ".service", ".md")])


def imported_project_modules():
    """Every project module (under scripts/) that the WORD-N code imports at module level, plus LAZY_IMPORTS."""
    for m in ("wn_analysis", "wn_score", "wn_cpu_arms", "wn_vista", "wn_intake", "wn_predict", "wn_run", "wn_plan",
              "word_a1_score"):          # word_a1_score: imported lazily by wn_analysis
        __import__(m)
    sys.path.insert(0, str(R / "scripts/vccor/vista"))
    __import__("vista_common")
    out = set()
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        p = Path(f).resolve()
        if str(p).startswith(str(R / "scripts")) and p.parent != HERE and p.suffix == ".py":
            out.add(p)
    return sorted(out | {p.resolve() for p in LAZY_IMPORTS})


def records():
    out = [T.PLAN, T.MEMBERS, T.UNIT_TESTS, T.REHEARSAL_RECORD, T.VSMOKE / "smoke.json"]
    if (T.VISTA / "labels" / "manifest.json").is_file():
        out += [T.VISTA / "labels" / "manifest.json", T.VISTA / "v0_draws" / "DONE.json"]
        out += sorted((T.VISTA / "labels").glob("WORD_*.nii.gz"))
    return out


def producer_differences():
    """Producer-code hashes recorded inside the pre-registration records vs the code being frozen (for disclosure)."""
    out = {}
    for name, p in (("archive_members", T.MEMBERS), ("vsmoke", T.VSMOKE / "smoke.json"), ("rehearsal", T.REHEARSAL_RECORD),
                    ("vista_smoke_model", T.VSMOKE / "model" / "export.json")):
        if not Path(p).is_file():
            continue
        for f, h in (json.load(open(p)).get("code_sha256") or {}).items():
            now_h = T.sha256(HERE / f) if (HERE / f).is_file() else None
            if now_h != h:
                out.setdefault(name, {})[f] = {"recorded": h, "frozen": now_h}
    return out


def freeze():
    if T.FREEZE.exists():
        raise T.Refuse("code_freeze.json exists (read-only)")
    T.overshoot_mode()
    ut = json.load(open(T.UNIT_TESTS))
    cur = {p.name: T.sha256(p) for p in sorted(HERE.glob("*.py"))}
    if not ut.get("all_pass") or ut.get("code_sha256") != cur:
        raise T.Refuse("unit_tests.json did not pass on the current code: rerun test_wordn.py --record")
    reh = json.load(open(T.REHEARSAL_RECORD))
    if not reh.get("all_pass") or reh.get("code_sha256") != cur:
        raise T.Refuse("the look rehearsal did not pass on the current code (every *.py must match): rerun wn_rehearse.py")
    if not T.MEMBERS.is_file():
        raise T.Refuse("the archive member pin is missing")
    if not (T.VSMOKE / "smoke.json").is_file():
        raise T.Refuse("the VISTA3D smoke test has not run")
    import wn_vista
    wn_vista.log_smoke_line()
    versions = T.check_versions()
    files = {}
    for p in code_files() + imported_project_modules() + records():
        files[T.rel(p)] = T.sha256(p)
    models, pending = {}, []
    for a in T.RUN_ORDER:
        k = T.arms()[a]["kind"]
        if a in ("F0", "V0") or k == "cpu":
            continue
        fs = T.model_files(a)
        if fs and all(p.is_file() for p in fs):
            T.resolve_model(a)
            models[a] = {T.rel(p): T.sha256(p) for p in fs}
        elif k == "vista":
            pending.append(a)
        else:
            raise T.Refuse(f"tier-1 arm {a}: model files missing")
    models["TS_WEIGHTS"] = {T.rel(p): T.sha256(p) for p in T.ts_weight_files()}
    models["VISTA_BUNDLE"] = {T.rel(p): T.sha256(p) for p in T.vista_bundle_files()}
    for d in T.DRAWS:
        p = T.arms()[f"LABELMAP_{d}"]["rules"]
        models[f"RULES_{d}"] = {T.rel(p): T.sha256(p)}
    cfg = Path(T.TS_ROOT) / "config.json"
    rec = {"what": "WORD-N code freeze (before the OSF upload)", "written": T.now(), "files": files, "models": models,
           "pending_models": pending, "pending_rule": PENDING_RULE, "versions": versions,
           "overshoot_mode": T.overshoot_mode(), "smoke_decision": json.load(open(T.VSMOKE / "smoke.json"))["decision"],
           "ts_config_sha256_informational": T.sha256(cfg) if cfg.is_file() else None,
           "producer_code_differences": producer_differences()}
    T.write_json(T.FREEZE, rec, mode=0o444)
    for p in (T.UNIT_TESTS, T.REHEARSAL_RECORD, T.PLAN, T.MEMBERS):
        os.chmod(p, 0o444)
    for p in code_files():
        T.prereg_line("CODE", p, "wordn code freeze")
    T.prereg_line("FREEZE", T.FREEZE, f"wordn code freeze: {len(files)} files, {len(models)} model sets incl. TS_WEIGHTS, "
                                      f"VISTA_BUNDLE and the LABELMAP rules; pending VF models {pending}; overshoot_mode "
                                      f"{rec['overshoot_mode']}; smoke {rec['smoke_decision']}")


def addendum_checks(arm, fz):
    """'' if the trained model may be hashed, else the reason (N9)."""
    sys.path.insert(0, str(R / "scripts/vccor/vista"))
    import vista_common as V
    w = Path(T.arms()[arm]["weights"])
    ex = json.load(open(w.parent / "export.json"))
    dn = json.load(open(w.parent / "DONE.json"))
    files = fz["files"]
    man = T.rel(T.VISTA / "labels" / "manifest.json")
    if ex.get("labels_manifest_sha256") != files.get(man):
        return "export.json names another labels manifest than the frozen one"
    for f in ("wn_vista.py", "wn_common.py"):
        if (ex.get("code_sha256") or {}).get(f) != files.get(T.rel(HERE / f)):
            return f"export.json's {f} hash differs from the frozen code"
    if ex.get("bundle_sha256") != V.BUNDLE_SHA:
        return "export.json's VISTA3D bundle hashes differ from the pinned bundle"
    if ex.get("smoke") or dn.get("smoke"):
        return "a smoke run"
    if not T.GATE_UPLOADED.is_file():
        return "no upload gate"
    up = datetime.strptime(json.load(open(T.GATE_UPLOADED))["uploaded_kst"], "%Y-%m-%d %H:%M:%S")
    fin = datetime.strptime(dn["finished"], "%Y-%m-%d %H:%M:%S")
    if not (up <= fin < T.CUTOFF):
        return f"training finished at {dn['finished']}, not between the upload ({up}) and the cutoff"
    return ""


def addendum():
    fz = json.load(open(T.FREEZE))
    done_a = sorted(p.name for p in T.OUT.glob("freeze_addendum_*.json"))
    hashed = set()
    for n in done_a:
        hashed |= set(json.load(open(T.OUT / n)).get("models", {}))
    pend = [a for a in fz["pending_models"] if a not in hashed and T.model_files(a)
            and all(p.is_file() for p in T.model_files(a))]
    if not pend:
        raise T.Refuse("no pending model has been trained")
    n = len(done_a) + 1
    out = {"addendum": n, "written": T.now(), "models": {}, "rejected": {},
           "upload_gate_existed": T.GATE_UPLOADED.is_file(),
           "note": "written before the arm's own prediction; disclosed relative to the upload"}
    for a in pend:
        why = addendum_checks(a, fz)
        if why:
            out["rejected"][a] = why
            continue
        T.resolve_model(a)
        out["models"][a] = {T.rel(p): T.sha256(p) for p in T.model_files(a)}
    p = T.OUT / f"freeze_addendum_{n}.json"
    T.write_json(p, out, mode=0o444)
    T.prereg_line("FREEZE", p, f"wordn freeze addendum {n}: VF models {sorted(out['models'])}; rejected "
                               f"{sorted(out['rejected']) or 'none'}; upload gate existed {out['upload_gate_existed']}")


README = """# WORD-N OSF deposit ({date})

This bundle belongs to the registration "WORD-N". The owner writes the registration text in their own words.

What it contains:
- `scripts/vccor/final/wordn/`: the frozen code (intake, predict, CPU arms, VISTA3D tier 2, runner, look, analysis, plan,
  freeze, unit tests, rehearsal, systemd units, the OSF guide and the registration skeleton).
- `frozen_imports/`: the project modules the code imports (frozen with it).
- `plan_wordn.json`: the machine-readable rules (overshoot_mode {mode}).
- `archive_members.json`: the pre-registration listing of the 40 whitelisted archive members (no content read).
- `unit_tests.json`, `rehearsal.json`, `vsmoke/smoke.json`, `vista/labels/manifest.json`, `code_freeze.json`: tests, the look
  rehearsal on stand-in data, the VISTA3D smoke test on the 12 draw scans, the composite VISTA training-label record, hashes.

No image, label or model file is included (their sha256 are in code_freeze.json). The code was written with AI assistance
(Claude), and the design came from an AI-simulated pre-submission review, not a human one.
"""


def bundle():
    if not T.FREEZE.is_file():
        raise T.Refuse("freeze first")
    if list(T.VC.glob("osf_deposit_*_wordn")) or list(T.VC.glob("osf_deposit_*_wordn.zip")):
        raise T.Refuse("a wordn deposit bundle exists: only one is built")
    date = datetime.now().strftime("%Y-%m-%d")
    d = T.VC / f"osf_deposit_{date}_wordn"
    z = d.with_suffix(".zip")
    for p in code_files():
        q = d / T.rel(p)
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, q)
    for p in imported_project_modules():
        q = d / "frozen_imports" / T.rel(p)
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, q)
    recs = [r for r in records() if not str(r).endswith(".nii.gz")] + [T.FREEZE] + sorted(T.OUT.glob("freeze_addendum_*"))
    for p in recs:
        q = d / Path(p).relative_to(T.OUT)
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, q)
    (d / "README.md").write_text(README.format(date=date, mode=T.overshoot_mode()))
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(d.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(d))
    for p in sorted(d.rglob("*")):
        if p.is_file():
            T.prereg_line("DEPOSIT", p, "wordn OSF deposit bundle")
    T.prereg_line("DEPOSIT", z, "wordn OSF deposit zip for the owner's upload under osf.io/dekmv")
    return z


def main():
    a = sys.argv[1:]
    try:
        T.kst_gate()
        if a == ["--freeze"]:
            freeze()
        elif a == ["--addendum"]:
            addendum()
        elif a == ["--bundle"]:
            bundle()
        else:
            raise SystemExit(__doc__)
    except T.Refuse as e:
        print(f"[REFUSE] {e.msg}", file=sys.stderr)
        return e.code
    return 0


if __name__ == "__main__":
    sys.exit(main())
