"""vccor/amos240 (pre-submission block B2, 2026-10-20): a NEW SEALED external look on the 240 AMOS22 CT scans whose labels
this project has never opened.

AMOS22 CT with labels = 200 official-train (imagesTr / labelsTr) + 100 official-validation (imagesVa / labelsVa) scans.
The project's AMOS work uses ONLY MagicNet/data/amos.txt: the first 40 train + first 20 validation ids (sorted), built by
scripts/prep_amos_ct.py; every AMOS label reader of the repository goes through that list (E0 all 60, the VC_COR AMOS look
56, E1 56 + the 4 labelled, VISTA3D 56 + 4, m2 4 (+ its pending look on the 56), the E2 forgetting arms 56, B5's labelled
draws). The other 240 (160 train + 80 validation) were never opened: AMOS240 = all labelled CT ids minus amos.txt, derived
from the directory listing (file names only; no label file is opened before the look) and audited by opened_record().
They include a SITE AND SCANNER the 60 never had (site 'people', 100 SIEMENS SOMATOM Force; the 60 are all site 'center').

Arms (PRIMARY, frozen E1 AMOS models, seed 1337, K = 4 = 0017 0001 0008 0048): F0 = TS-full's own weights; FULL_K4 = parts
291 + 293 fine-tuned with F1b (xdata models amos_s1337_K4_FULL_F1b); NORM_K4 = F3 recipe (amos_s1337_K4_FULL_F3). All through
the frozen tsft_full/ts_pipeline.py (TotalSegmentator 2.13.0's own pipeline; fine-tuned parts redirected, frozen parts
replayed from F0's per-scan cache). SECONDARY (HELD jobs; scored iff their predictions are DONE when the look is released):
FULL_K4 of AMOS draws 2025 / 3407 (block amos_draws, B5) and G17 (FULL_K4 + TS pseudo-labels on the AMOS pool36, block
pullback, B4). Predictions stay on the RAW grid (native); the scorer maps TS ids -> FLARE ids (xdata total_lut('amos')).

This module has NO import-time side effects beyond sys.path (safe before nnunetv2 / torch).
"""
import csv
import glob
import json
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path("/home/user/Documents/Medical Imagining SSL")
HERE = Path(__file__).resolve().parent
TSFT_DIR = REPO / "scripts/vccor/tsft_full"
XDATA_DIR = REPO / "scripts/vccor/xdata"
AFT_DIR = REPO / "scripts/vccor/audit_ft"
M2_DIR = REPO / "scripts/vccor/m2"
for _p in (TSFT_DIR, XDATA_DIR):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))
import xdata_common as X  # noqa: E402  (frozen: AMOS paths, LUTs, total_lut, nsd_hd95)

Q = X.Q
OUT = REPO / "outputs/results/vccor/amos240"
LOGD = REPO / "outputs/logs/vccor"
PREREG = OUT / "prereg_amos240.json"
PREREG_LOG = LOGD / "amos240_prereg.log"
META_CSV = OUT / "meta/labeled_data_meta_0000_0599.csv"
META_CSV_SHA = "cdb3ba1abe26940e71288fcae1c8f5839ff6428d169c2d7261453e8cc413e143"
META_SOURCE = "https://zenodo.org/records/7262581 (AMOS, Ji et al. NeurIPS 2022), file labeled_data_meta_0000_0599.csv"
OPENED = OUT / "opened_record.json"                     # the audit of every AMOS label the project opened before this block
PRED = OUT / "pred"                                       # pred/<arm>/{raw,rec}/<cid>, pred/<arm>/DONE.json; pred/F0/cache_frozen
RES = OUT / "results"
SMOKE = OUT / "smoke"
VENV = REPO / ".venv/bin/python"
AMOS_RAW = X.AMOS_RAW
AMOS_TXT = X.DATA / "amos.txt"

ARMS = {   # arm: (fine-tuned model run dir (xdata / B5 / B4 export) or None, primary?)
    "F0": (None, True),
    "FULL_K4": (X.MODELS / "amos_s1337_K4_FULL_F1b", True),
    "NORM_K4": (X.MODELS / "amos_s1337_K4_FULL_F3", True),
    "FULL_K4_s2025": (X.MODELS / "amos_s2025_K4_FULL_F1b", False),
    "FULL_K4_s3407": (X.MODELS / "amos_s3407_K4_FULL_F1b", False),
    "G17": (REPO / "outputs/results/vccor/pullback/models/G17", False),
}
PRIMARY_ARMS = [a for a, (_, p) in ARMS.items() if p]
SECONDARY_ARMS = [a for a, (_, p) in ARMS.items() if not p]
PARTS_FT = X.parts_ft("amos")                             # (291, 293)
PARTS_FROZEN = X.parts_frozen("amos")                     # (292, 294, 295)
RULE = {"min_mean": 0.30, "min_fraction_positive": 2.0 / 3.0,
        "statement": "PASS iff M(FULL_K4 - F0) >= +0.30 Dice points AND the 95% t-CI lower bound > 0 AND positive paired "
                     "differences on at least ceil(2/3 x 240) = 160 of the 240 scans (native-grid present-only macro Dice)"}
FP_MIN_ML = 1.0                                           # FP-penalised macro: an absent-in-GT organ predicted >= 1 mL scores 0


# ---------------------------------------------------------------------------------------------------- helpers
sha256, rel, write_json, Log, be_nice, cap_threads = Q.sha256, Q.rel, Q.write_json, Q.Log, Q.be_nice, Q.cap_threads


def now():
    return time.strftime("%F %T")


def amos_txt_ids():
    return X.amos_ids()


def all_labelled_ct():
    """{cid: (image path, label path, split)} for every AMOS22 CT image whose label FILE exists (os.path only)."""
    out = {}
    for sub, d, sp in (("train", "imagesTr", "train"), ("valid", "imagesVa", "validation")):
        for p in sorted(glob.glob(str(AMOS_RAW / sub / d / "amos_*.nii.gz"))):
            cid = Path(p).name[5:9]
            lp = p.replace("imagesTr", "labelsTr").replace("imagesVa", "labelsVa")
            if int(cid) < 500 and os.path.isfile(lp):
                out[cid] = (Path(p), Path(lp), sp)
    return out


def ids240():
    used = set(amos_txt_ids())
    allc = all_labelled_ct()
    ids = sorted(c for c in allc if c not in used)
    assert len(allc) == 300 and len(used) == 60 and used <= set(allc) and len(ids) == 240, (len(allc), len(used), len(ids))
    return ids


def raw_img(cid):
    return all_labelled_ct()[cid][0]


def raw_lbl(cid):
    return all_labelled_ct()[cid][1]


def meta():
    """AMOS's own metadata per id (sex, age, scanner model, manufacturer, acquisition date, site)."""
    assert sha256(META_CSV) == META_CSV_SHA, "AMOS metadata CSV changed"
    return {f"{int(r['amos_id']):04d}": r for r in csv.DictReader(open(META_CSV))}


def strata(ids, img_header=True):
    """Pre-registered strata from metadata + image headers (no label read): site, manufacturer, scanner model, sex, age
    group, official split, slice spacing group."""
    import nibabel as nib
    m, allc = meta(), all_labelled_ct()
    out = {}
    for c in ids:
        r = m[c]
        age = r["Patient's Age"].rstrip("Y").strip()
        age = int(age) if age.isdigit() else None
        z = float(nib.load(str(allc[c][0])).header.get_zooms()[2]) if img_header else None
        out[c] = {"site": r["Site"], "manufacturer": r["Manufacturer"], "scanner": r["Manufacturer's Model Name"],
                  "sex": r["Patient's Sex"], "age_group": None if age is None else ("<50" if age < 50 else ("50-64" if age < 65 else ">=65")),
                  "split": allc[c][2], "slice": None if z is None else ("5mm" if abs(z - 5.0) < 1e-3 else "<5mm"), "z_mm": z}
    return out


def pred_dir(arm, root=None):
    return Path(root or PRED) / arm


def pred_done(arm, root=None):
    return pred_dir(arm, root) / "DONE.json"


def cache_dir(cid, root=None):
    return pred_dir("F0", root) / "cache_frozen" / cid


def opened_record():
    """Every AMOS label this project opened before this block, by source, and the checks that none of the 240 is among
    them (directory listings and GT_OPENED markers only; no label file is read)."""
    used = set(amos_txt_ids())
    ids = set(ids240())
    rec = {"amos_txt": {"n": len(used), "ids": sorted(used), "built_by": "scripts/prep_amos_ct.py (first 40 imagesTr + "
                                                                       "first 20 imagesVa, sorted)"}}
    marks = {}
    for p in sorted((REPO / "outputs").rglob("GT_OPENED*")):
        try:
            d = json.loads(open(p).read().strip().splitlines()[0])
        except (ValueError, IndexError):
            continue
        ds = str(d.get("ds") or d.get("dataset") or "")
        if "amos" in str(p).lower() or ds == "amos":
            marks[rel(p)] = sorted(d.get("ids", []))
    rec["gt_opened_markers_amos"] = {k: {"n": len(v), "subset_of_amos_txt": set(v) <= used, "overlap_240": sorted(set(v) & ids)}
                                     for k, v in marks.items()}
    derived = {}
    for d in (X.DATA / "amos_ct_h5", X.DATA / "amos_pseudo_fullres", X.DATA / "amos_importance", X.DATA / "amos_importance_fm",
              X.COR_FEAS_TS_RAW / "amos", X.COR_FEAS_BUILD, X.OUT / "frames"):
        if not Path(d).exists():
            continue
        names = [f.name for f in Path(d).iterdir() if f.is_file()]
        hits = sorted({c for c in ids for n in names if n in (f"{c}.h5", f"{c}.nii.gz", f"amos_{c}.json", f"amos_{c}.nii.gz")})
        derived[rel(d)] = {"n_files": len(names), "files_of_240": hits}
    fr = json.load(open(X.OUT / "frames" / "amos.json"))["frames"]
    derived["xdata frames/amos.json keys"] = {"n": len(fr), "files_of_240": sorted(set(fr) & ids)}
    rec["derived_per_case_data"] = derived
    readers = {"e0 (all 60, amos_ct_h5)": 60, "VC_COR AMOS look (56)": 56, "E1 xdata (56 + 4 labelled)": 60,
               "VISTA3D (56 + 4 training)": 60, "m2 (4 training; look pending on the 56)": 60,
               "E2 forgetting (56)": 56, "B5 labelled draws 2025 / 3407 (8 of amos.txt[:40])": 60}
    rec["readers_all_via_amos_txt"] = readers
    rec["code_scan"] = ("every script under scripts/ and MagicNet/code that reads AMOS labels takes its ids from amos.txt / "
                        "amos_ct_h5 (60 files) / xdata_common.amos_ids(); scripts/amos_prep/make_split.py read the GT of the 60 "
                        "only; scripts/extras/external_rescore/check_orientation.py read 3 IMAGE headers (affine), no label")
    rec["all_clean"] = (all(v["subset_of_amos_txt"] and not v["overlap_240"] for v in rec["gt_opened_markers_amos"].values())
                        and all(not v["files_of_240"] for v in derived.values()))
    rec["n_240"] = len(ids)
    return rec


# ---------------------------------------------------------------------------------------------------- integrity
def _log_lines():
    return [ln.split() for ln in open(PREREG_LOG)] if PREREG_LOG.is_file() else []


def prereg_sha_logged():
    h = None
    for p in _log_lines():
        if len(p) >= 6 and p[2] == "PREREG" and p[3] == "sha256" and p[5] == rel(PREREG):
            h = p[4]
    return h


def frozen_code_hashes():
    out = {}
    for p in _log_lines():
        if len(p) >= 5 and p[2] in ("CODE", "IMPORT") and len(p[3]) == 64:
            out[" ".join(p[4:])] = p[3]
    return out


def check_code_frozen(paths, log=print, strict=True):
    fz = frozen_code_hashes()
    bad = {rel(p): (sha256(REPO / rel(p))[:16], (fz.get(rel(p)) or "NOT-FROZEN")[:16]) for p in paths
           if fz.get(rel(p)) != sha256(REPO / rel(p))}
    if bad:
        msg = f"code differs from the CODEFREEZE hashes in {rel(PREREG_LOG)}: {bad}"
        if strict:
            raise SystemExit("[REFUSE] " + msg)
        log("[warn] " + msg)
    return not bad


def check_prereg():
    h = sha256(PREREG)
    if h != prereg_sha_logged():
        raise SystemExit(f"[REFUSE] {rel(PREREG)} sha256 {h[:16]} != logged {str(prereg_sha_logged())[:16]}")
    return json.load(open(PREREG))


FROZEN_IMPORTS = [XDATA_DIR / "xdata_common.py", TSFT_DIR / "tsft_full_common.py", TSFT_DIR / "ts_pipeline.py",
                  AFT_DIR / "aft_stats.py", M2_DIR / "m2_cluster.py"]


def code_files():
    return sorted(HERE.glob("*.py"))


def preflight_real(script, log=print, extra=()):
    check_prereg()
    check_code_frozen([HERE / "a240_common.py", Path(script)] + [Path(e) for e in extra] + FROZEN_IMPORTS, log)
