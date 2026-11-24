"""vccor/final FLARE-T50 FREEZE and OSF DEPOSIT BUNDLE.

  --freeze  After the independent code review, K1b, K1c and the look rehearsal, and before the author uploads the
            registration: hashes every FLARE-T50 code file, every frozen import (and the frozen fair-fight files), the
            records (Hub manifest, images record, K1b, exclusions, K1c, plan, unit tests, look rehearsal, smoke record,
            tier-2 code record and the reconstructed tier-2 t50_common, the registration skeleton), every model file of
            every arm, and TotalSegmentator's own weights (models['TS_WEIGHTS']; ~/.totalsegmentator/config.json, which
            TotalSegmentator's API rewrites with a usage counter, is recorded but not enforced) into
            I_flaret50/code_freeze.json (read-only), with the package versions, a FREEZE line and one CODE line per code
            file. Tier-2 models not trained yet are listed as pending (tier-1 models must all exist). The registration
            skeleton, unit_tests.json, rehearsal.json and smoke_predict.json become read-only (the author writes the
            registration text in a new file; the rehearsal and the smoke recorder refuse to rewrite theirs). The
            producer-code hashes recorded inside images_record / k1b / k1c / rehearsal are compared with the frozen code;
            differences are listed for disclosure. Refuses unless the unit tests pass on the current code, K1b and K1c
            passed, the rehearsal passed and every image matched its pinned pointer oid.
  --addendum  Hashes tier-2 models that were pending at the freeze into I_flaret50/freeze_addendum_<n>.json, before that
            arm's own prediction (the tier-1 runner calls it); the addendum's time relative to the upload is disclosed.
  --bundle  Builds the ONE outputs/results/vccor/osf_deposit_<date>_flaret50/ and its zip (code, frozen imports, frozen
            fair-fight files, plan, records, freeze, skeleton, README) for the author to upload under osf.io/dekmv; DEPOSIT
            lines. No image, label or model is included. Refuses if a flaret50 bundle exists.

usage:  .venv/bin/python -I -B scripts/vccor/final/flaret50/t50_freeze.py --freeze | --addendum | --bundle

Version of 2026-11-09; revised after the code review (M1b).
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
import t50_common as T  # noqa: E402

R = T.REPO
FROZEN_IMPORTS = [R / "scripts/vccor/final/final_common.py", R / "scripts/vccor/final/fin_look.py",
                  R / "scripts/vccor/m2/m2_cluster.py", R / "scripts/vccor/tsft_full/ts_pipeline.py",
                  R / "scripts/vccor/e0/overlap.py", R / "scripts/vccor/e0/e0_common.py",
                  R / "scripts/vccor/xdata/xdata_common.py", R / "scripts/vccor/final/word_common.py",
                  R / "scripts/vccor/amos240/a240_common.py",
                  R / "scripts/vccor/moonshot/ff_launch.py", R / "scripts/vccor/moonshot/ms_common.py"]
PENDING_RULE = ("a pending tier-2 model is hashed by a freeze addendum before its own arm's prediction; the addendum's time "
                "relative to the upload is disclosed; a K = 1 arm whose model fails the freeze check is NOT RUN")


def records():
    return [T.MANIFEST, T.IMAGES_RECORD, T.K1B / "k1b.json", T.EXCLUSIONS, T.K1C / "k1c.json", T.PLAN,
            T.OUT / "unit_tests.json", T.REHEARSAL_RECORD, T.OUT / "smoke_predict.json", T.TIER2 / "code_tier2.json",
            T.TIER2 / "t50_common_a75b24a5.py", T.OUT / "REGISTRATION_SKELETON.md"]


def code_files():
    return sorted([p for p in HERE.iterdir() if p.suffix in (".py", ".service")])


def fair_fight_files():
    sys.path.insert(0, str(R / "scripts/vccor/moonshot"))
    import ms_common as M
    return sorted(set(M.FF_FILES))


def producer_differences():
    """Producer-code hashes recorded inside the pre-registration records vs the code being frozen."""
    out = {}
    for name, p in (("images_record", T.IMAGES_RECORD), ("k1b", T.K1B / "k1b.json"), ("k1c", T.K1C / "k1c.json"),
                    ("rehearsal", T.REHEARSAL_RECORD)):
        rec = json.load(open(p))
        for f, h in (rec.get("code_sha256") or {}).items():
            cur = HERE / f if (HERE / f).is_file() else R / "scripts/vccor" / f
            now_h = T.sha256(cur) if cur.is_file() else None
            if now_h != h:
                out.setdefault(name, {})[f] = {"recorded": h, "frozen": now_h}
    return out


def freeze():
    if T.FREEZE.exists():
        raise T.Refuse("code_freeze.json exists (read-only)")
    ut = json.load(open(T.OUT / "unit_tests.json"))
    if not ut.get("all_pass"):
        raise T.Refuse("unit tests did not all pass")
    cur = {p.name: T.sha256(p) for p in sorted(HERE.glob("*.py"))}
    if ut["code_sha256"] != cur:
        raise T.Refuse("unit_tests.json was recorded on other code: rerun test_t50.py --record")
    reh = json.load(open(T.REHEARSAL_RECORD))
    if not reh.get("all_pass") or reh.get("code_sha256", {}).get("t50_score.py") != cur["t50_score.py"]:
        raise T.Refuse("the look rehearsal did not pass on the current t50_score.py")
    if json.load(open(T.K1B / "k1b.json")).get("decision") != "K1B_PASS":
        raise T.Refuse("K1b did not pass")
    if json.load(open(T.K1C / "k1c.json")).get("decision") != "K1C_PASS":
        raise T.Refuse("K1c did not pass")
    if not json.load(open(T.IMAGES_RECORD)).get("all_pointer_oids_match"):
        raise T.Refuse("not every image matched its pinned pointer oid")
    versions = T.check_versions()
    files = {}
    for p in code_files() + FROZEN_IMPORTS + fair_fight_files() + records():
        files[T.rel(p)] = T.sha256(p)
    models, pending = {}, []
    for a in T.RUN_ORDER:
        if a == "F0":
            continue
        fs = T.model_files(a)
        if all(p.is_file() for p in fs):
            T.resolve_model(a)
            models[a] = {T.rel(p): T.sha256(p) for p in fs}
        elif T.ARMS[a]["tier"] == 2:
            pending.append(a)
        else:
            raise T.Refuse(f"tier-1 arm {a}: model files missing")
    models["TS_WEIGHTS"] = {T.rel(p): T.sha256(p) for p in T.ts_weight_files()}
    cfg = Path(T.TS_ROOT) / "config.json"
    rec = {"what": "FLARE-T50 code freeze (before the OSF upload)", "written": T.now(), "files": files,
           "models": models, "pending_models": pending, "pending_rule": PENDING_RULE, "versions": versions,
           "ts_config_sha256_informational": T.sha256(cfg) if cfg.is_file() else None,
           "producer_code_differences": producer_differences()}
    T.write_json(T.FREEZE, rec, mode=0o444)
    for p in (T.OUT / "REGISTRATION_SKELETON.md", T.OUT / "unit_tests.json", T.REHEARSAL_RECORD,
              T.OUT / "smoke_predict.json"):
        os.chmod(p, 0o444)
    for p in code_files():
        T.prereg_line("CODE", p, "flaret50 code freeze")
    T.prereg_line("FREEZE", T.FREEZE, f"flaret50 code freeze: {len(files)} files, {len(models)} model sets incl. "
                                      f"TS_WEIGHTS; pending tier-2 models {pending}; producer-code differences "
                                      f"{sorted(rec['producer_code_differences']) or 'none'}")


def addendum():
    fz = json.load(open(T.FREEZE))
    done_a = {p.name for p in T.OUT.glob("freeze_addendum_*.json")}
    hashed = set()
    for n in sorted(done_a):
        hashed |= set(json.load(open(T.OUT / n)).get("models", {}))
    pend = [a for a in fz["pending_models"] if a not in hashed and all(p.is_file() for p in T.model_files(a))]
    if not pend:
        raise T.Refuse("no pending model has been trained")
    n = len(done_a) + 1
    up = T.GATE_UPLOADED
    out = {"addendum": n, "written": T.now(), "models": {},
           "upload_gate_existed": up.is_file(),
           "note": "written before the arm's own prediction; disclosed relative to the upload"}
    for a in pend:
        T.resolve_model(a)
        out["models"][a] = {T.rel(p): T.sha256(p) for p in T.model_files(a)}
    p = T.OUT / f"freeze_addendum_{n}.json"
    T.write_json(p, out, mode=0o444)
    T.prereg_line("FREEZE", p, f"flaret50 freeze addendum {n}: tier-2 models {sorted(out['models'])}; upload gate existed "
                               f"{out['upload_gate_existed']}")


README = """# FLARE-T50 OSF deposit ({date})

This bundle belongs to the registration "FLARE-T50". The owner writes the registration text in their own words.

What it contains:
- `scripts/vccor/final/flaret50/`: the frozen code (predict, run, look, analysis, K1b, K1c, tier 2, tests, rehearsal,
  smoke record).
- `frozen_imports/`: the project modules the code imports (frozen with it) and the frozen fair-fight training files.
- `plan_flaret50.json`: the machine-readable rules.
- `REGISTRATION_SKELETON.md`: the structured technical content; every outcome sentence is the owner's.
- `hf_manifest.json`: the public metadata that pins every image and label file (sizes, LFS-pointer git oids) at the
  pinned revision.
- `images_record.json`, `k1b/k1b.json`, `exclusions.json`, `k1c/k1c.json`: the image-only steps made before registration.
- `unit_tests.json`, `rehearsal.json`, `smoke_predict.json`, `code_freeze.json`, `tier2/`: tests, the look rehearsal on
  stand-in labels, hashes, and the tier-2 records.

No image, label or model file is included (their sha256 are in code_freeze.json). The code was written with AI assistance
(Claude), and the design came from an AI-simulated pre-submission review, not a human one.
"""


def bundle():
    if not T.FREEZE.is_file():
        raise T.Refuse("freeze first")
    if list(T.VC.glob("osf_deposit_*_flaret50")) or list(T.VC.glob("osf_deposit_*_flaret50.zip")):
        raise T.Refuse("a flaret50 deposit bundle exists: only one is built")
    date = datetime.now().strftime("%Y-%m-%d")
    d = T.VC / f"osf_deposit_{date}_flaret50"
    z = d.with_suffix(".zip")
    for p in code_files():
        q = d / T.rel(p)
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, q)
    for p in FROZEN_IMPORTS + fair_fight_files():
        q = d / "frozen_imports" / T.rel(p)
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, q)
    for p in records() + [T.FREEZE] + sorted(T.OUT.glob("freeze_addendum_*.json")):
        q = d / Path(p).relative_to(T.OUT)
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, q)
    (d / "README.md").write_text(README.format(date=date))
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(d.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(d))
    for p in sorted(d.rglob("*")):
        if p.is_file():
            T.prereg_line("DEPOSIT", p, "flaret50 OSF deposit bundle")
    T.prereg_line("DEPOSIT", z, "flaret50 OSF deposit zip for the owner's upload under osf.io/dekmv")


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
