"""vccor/final LCTSC-T23 FREEZE (CPU; once, before the deposit and before the plan-freeze deadline Wed 2026-11-18 18:00 KST):
K_lctsc_t23/code_freeze.json (read-only) with the sha256 of every study file, then the deposit zip
K_lctsc_t23/osf_deposit_lctsc_t23_<date>.zip and a 'FREEZE' and a 'DEPOSIT' line in final_prereg.log.
Revised after the independent code review of 2026-11-17 (B1, S1, S3, S5, N9, N10, N11).

Refuses if: a T23 freeze exists (unless --deposit-only finishes a freeze whose DEPOSIT line is missing; review N10); the
plan-freeze deadline has passed (registered go/no-go gate 1: then nothing runs and the 23 stay sealed); unit_tests.json,
rehearsal.json or smoke.json is missing, not all_pass / ok, or recorded on other code than the current files; t23_ct.json is
not all_ok or does not cover exactly the 4 stand-ins and the 23 patients; ids_t23.json differs from the derivation rule; a
derived script differs from derived.json; any of the 23 stored RTSTRUCT zips is missing or has another size than its
K_lctsc/download.json record, or an extracted copy (data_lctsc/dicom/<pid>_RTSTRUCT or .tmp_<pid>_RTSTRUCT) exists -- file
metadata only, no RTSTRUCT byte is read (review S1); the LCTSC freeze (K_lctsc/code_freeze.json) no longer verifies (the
frozen lctsc_gates.freeze_ok re-hashes its code, records, models and TotalSegmentator weights); the frozen d6ayj model
checkpoints differ from the LCTSC freeze.

Frozen: every scripts/vccor/final/lctsc_t23/*.py and unit file and the as-run copy of t23_ct_fetch.py (as_run/); the frozen
LCTSC modules that run or were derived from (lctsc_common, lctsc_analysis, lctsc_raster, lctsc_convert, lctsc_infra,
lctsc_runs, lctsc_gates, lctsc_log, lctsc_freeze, lctsc_predict, lctsc_predict_queue, lctsc_look_labels, lctsc_score) and the
frozen modules those import (final_common, fin_nn, fin_look, bsl_common, flaret50/t50_analysis, m2/m2_cluster,
tsft_full/tsft_full_common, tsft_full/ts_pipeline, peft/lora); the records (K_lctsc: tcia_manifest, download, images,
labels_train, labels_train_v2, draws, draws_site, code_freeze; K_lctsc_t23: ids_t23, t23_ct, plan_lctsc_t23, unit_tests,
rehearsal, smoke; derived.json); the registration notes and OSF guide; the d6ayj models (checkpoint + export.json +
DONE.json); TotalSegmentator's checkpoints.

The deposit zip (deposit_files()) holds the code, the records, the plan, the registration notes and the OSF guide, and the
T23 freeze as its ONLY member named code_freeze.json (review B1: the d6ayj freeze stays hash-pinned in the T23 freeze and is
public in the d6ayj deposit); no image, label or model file (models are listed by hash).

usage:  .venv/bin/python -I -B scripts/vccor/final/lctsc_t23/t23_freeze.py [--check | --deposit-only]

Version of 2026-11-17.
"""
import argparse
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
LCT = HERE.parent / "lctsc"
for _p in (str(HERE.parent.parent / "tsft_full"), str(HERE.parent.parent / "m2"), str(HERE.parent / "flaret50"),
           str(HERE.parent), str(LCT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import t23_common as T  # noqa: E402
import t23_gates as G  # noqa: E402
import t23_log as L  # noqa: E402

FINAL = HERE.parent
LCTSC_MODULES = [LCT / f for f in ("lctsc_common.py", "lctsc_analysis.py", "lctsc_raster.py", "lctsc_convert.py",
                                   "lctsc_infra.py", "lctsc_runs.py", "lctsc_gates.py", "lctsc_log.py", "lctsc_freeze.py",
                                   "lctsc_predict.py", "lctsc_predict_queue.py", "lctsc_look_labels.py", "lctsc_score.py")]
FROZEN_IMPORTS = [FINAL / "final_common.py", FINAL / "fin_nn.py", FINAL / "fin_look.py", FINAL / "bsl_common.py",
                  FINAL / "flaret50/t50_analysis.py", FINAL.parent / "m2/m2_cluster.py", FINAL.parent / "tsft_full/tsft_full_common.py",
                  FINAL.parent / "tsft_full/ts_pipeline.py", FINAL.parent / "peft/lora.py"]
LCTSC_FREEZE = T.SRC / "code_freeze.json"
RECORDS = [T.SRC / f for f in ("tcia_manifest.json", "download.json", "images.json", "labels_train.json", "labels_train_v2.json",
                               "draws.json", "draws_site.json")] + [LCTSC_FREEZE] + \
          [T.IDS_FILE, T.CT_RECORD, T.PLAN, T.OUT / "unit_tests.json", T.OUT / "rehearsal.json", T.OUT / "smoke.json",
           HERE / "derived.json"]
DOCS = [T.OUT / "REGISTRATION_NOTES_T23.md", T.OUT / "OSF_GUIDE_T23.md"]
DATA = T.DATA


def code_files():
    return sorted(HERE.glob("*.py")) + sorted(HERE.glob("*.service")) + sorted((HERE / "as_run").glob("*.py"))


def frozen_files():
    return code_files() + LCTSC_MODULES + FROZEN_IMPORTS + RECORDS + DOCS


def deposit_files(files):
    """review B1: the deposit's only member named code_freeze.json is the T23 freeze (the d6ayj freeze is left out of the zip;
    it stays hash-pinned in the T23 freeze as a file and as lctsc_freeze_sha256)."""
    keep = [Path(f) for f in files if Path(f).resolve() != LCTSC_FREEZE.resolve()]
    return keep + [G.FREEZE]


def deposit_member(f):
    return f"lctsc_t23_deposit/{L.rel(f)}"


def zips_untouched():
    """review S1: file metadata only (no byte of any RTSTRUCT is read)."""
    dl = json.load(open(T.SRC / "download.json"))["files"]
    bad = []
    for pid in T.ids():
        rec, z = dl.get(f"{pid}_RTSTRUCT.zip"), DATA / "tcia_zips" / f"{pid}_RTSTRUCT.zip"
        if rec is None or not z.is_file() or z.stat().st_size != rec["bytes"]:
            bad.append(f"{pid}: zip missing or size differs")
        if any((DATA / "dicom" / n).exists() for n in (f"{pid}_RTSTRUCT", f".tmp_{pid}_RTSTRUCT")):
            bad.append(f"{pid}: an extracted RTSTRUCT copy exists")
    return bad


def _lctsc_gates():
    """The FROZEN lctsc_gates module under its own name (its freeze_ok re-hashes the whole LCTSC freeze)."""
    spec = importlib.util.spec_from_file_location("frozen_lctsc_gates", LCT / "lctsc_gates.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def deposit(fz):
    files = [Path(p) if p.startswith("/") else G.REPO / p for p in fz["deposit_members_source"]]
    zp = T.OUT / f"osf_deposit_lctsc_t23_{fz['frozen_kst'][:10]}.zip"
    assert not zp.exists(), zp
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, deposit_member(f))
    with zipfile.ZipFile(zp) as z:
        hits = [n for n in z.namelist() if n.endswith("code_freeze.json")]
        assert hits == [deposit_member(G.FREEZE)], hits
    L.prereg_line("DEPOSIT", zp, "lctsc_t23 OSF deposit zip (code, records, plan, registration notes, guide; no image, label "
                                 "or model file; the T23 freeze is its only code_freeze.json)")
    print(zp, G.sha256(zp))


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true")
    g.add_argument("--deposit-only", action="store_true")
    a = ap.parse_args()
    G.kst_gate()
    if a.deposit_only:                                   # review N10: finish a freeze whose DEPOSIT line is missing
        if not G.FREEZE.exists() or G.logged("DEPOSIT", G.DEPOSIT_RX):
            raise G.Refuse("--deposit-only needs a T23 freeze and no LCTSC-T23 DEPOSIT line")
        G.freeze_ok()
        deposit(json.load(open(G.FREEZE)))
        return 0
    if G.FREEZE.exists():
        raise G.Refuse("an LCTSC-T23 freeze exists")
    if not G.ts(G.now()) < G.ts(T.PLAN_FREEZE_DEADLINE):
        raise G.Refuse(f"the plan-freeze deadline {T.PLAN_FREEZE_DEADLINE} KST has passed (go/no-go gate 1): nothing runs")
    cur = {f.name: G.sha256(f) for f in sorted(HERE.glob("*.py")) + sorted(HERE.glob("*.service"))}
    for rec, key in (("unit_tests.json", "all_pass"), ("rehearsal.json", "all_pass"), ("smoke.json", "ok")):   # review S3
        p = T.OUT / rec
        if not p.is_file() or not json.load(open(p)).get(key):
            raise G.Refuse(f"{rec} missing or not {key}")
        if json.load(open(p)).get("code_sha256") != cur:
            raise G.Refuse(f"{rec} was recorded on other code than the current files: re-run it")
    ct = json.load(open(T.CT_RECORD))
    if not ct.get("all_ok") or sorted(ct["patients"]) != sorted(T.STANDINS + T.ids()):            # review N9
        raise G.Refuse("t23_ct.json is not all_ok for exactly the 4 stand-ins and the 23 patients")
    if T.ids() != T.derive_ids():
        raise G.Refuse("ids_t23.json differs from the derivation rule")
    der = json.load(open(HERE / "derived.json"))
    bad = [f for f, h in der["outputs"].items() if G.sha256(HERE / f) != h]
    if bad:
        raise G.Refuse(f"derived scripts edited after the derivation: {bad}")
    bad = zips_untouched()
    if bad:
        raise G.Refuse(f"stored RTSTRUCT zips: {bad}")
    lfz = _lctsc_gates().freeze_ok()                   # re-hashes LCTSC's code, records, models and TS weights
    models = lfz["models"]
    for arm, parts in models.items():
        d = json.load(open(G.MODELS / G.ARMS[arm]["run"] / "DONE.json"))
        got = {str(t): G.sha256(Path(md) / "fold_0" / "checkpoint_final.pth") for t, md in d["resolved"]["parts"].items()}
        if got != parts:
            raise G.Refuse(f"{arm}: checkpoints differ from the LCTSC freeze")
    files = frozen_files()
    missing = [str(f) for f in files if not Path(f).is_file()]
    if missing:
        raise G.Refuse(f"missing files: {missing}")
    model_files = []
    for arm in sorted(models):
        d = json.load(open(G.MODELS / G.ARMS[arm]["run"] / "DONE.json"))
        model_files.append(G.MODELS / G.ARMS[arm]["run"] / "DONE.json")
        for md in d["resolved"]["parts"].values():
            model_files += [Path(md) / "fold_0" / "checkpoint_final.pth", Path(md) / "export.json"]
    dep = deposit_files(files)
    if sum(deposit_member(f).endswith("code_freeze.json") for f in dep) != 1:
        raise G.Refuse("the deposit would hold more than one code_freeze.json")
    if a.check:
        print(f"freeze check OK: {len(files)} files, {len(model_files)} model files, TS weights {len(lfz['ts_weights'])}, "
              f"deposit members {len(dep)}")
        return 0
    import lctsc_freeze as LF                           # frozen: versions() and cv2_record() of the LCTSC freeze
    fz = {"frozen_kst": G.now(), "files": {L.rel(f): G.sha256(f) for f in files + model_files},
          "ts_weights": dict(lfz["ts_weights"]), "models": models, "draws_status": lfz["draws_status"],
          "lctsc_freeze_sha256": G.sha256(LCTSC_FREEZE), "versions": LF.versions(), "opencv": LF.cv2_record(),
          "ids": T.ids(), "plan_freeze_deadline_kst": T.PLAN_FREEZE_DEADLINE, "public_deadline_kst": T.PUBLIC_DEADLINE,
          "cutoff_kst": G.CUTOFF, "stop_kst": G.STOP, "deposit_members_source": [L.rel(f) for f in dep]}
    for p, h in lfz["files"].items():                   # TotalSegmentator's own checkpoints (absolute paths)
        if p.startswith("/"):
            fz["files"][p] = h
    G.write_json(G.FREEZE, fz, mode=0o444)
    L.prereg_line("FREEZE", G.FREEZE, f"lctsc_t23 code / records / frozen d6ayj models frozen: {len(fz['files'])} files; "
                                       f"23 patients; no training")
    deposit(fz)
    return 0


if __name__ == "__main__":
    sys.exit(main())
