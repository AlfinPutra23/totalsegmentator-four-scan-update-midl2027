"""vccor/xdata/a3: implementation of the MACT single-look AMENDMENT A3 (outputs/results/vccor/xdata/
prereg_e1_mact_amendment_A3.json; 'AMENDMENT sha256 ...' line in outputs/logs/vccor/xdata_prereg.log). The amendment fixes
every rule; this code may not change any of them. PREPARED, NOT LAUNCHED (scheduler patch queue_v8_patch_a3.py written,
not applied). Every rule came from the internal review of FLARE / AMOS results (review panel 2026-10-20), not from MACT data.

Pieces (MACT h5 ids 1 spleen 2 l.kid 3 gall 4 esoph 5 liver 6 stom 7 panc 8 duod; 81 primary / 53 never-used S2):
  A3.6 arms   P2 = FULL K=4 (F1b recipe) on the disjoint pool draw 0070 0087 0011 0080; K4t2025 / K4t3407 = FULL_K4 on
              0075 0049 0037 0041 with training seeds 2025 / 3407 (prep_a3.py, train_a3.py; predictions through the frozen
              xdata/predict_xdata.py, i.e. the same inference path as F0 and FULL_K4)
  A3.9        Gibson's Zenodo release (record 1169361) in MagicNet/data/mact_zenodo_1169361 (sha256-pinned tarballs, read in
              memory, member by member). Pool scans may be read at any time (training scans); an evaluation scan only after
              the A3 look's GT_OPENED marker exists (zenodo_member_bytes refuses otherwise).
  frames      H5          the existing frame (predictions = xdata_common.mact_raw_to_h5 of the raw map; reference = MagicNet
                          MACT_h5 'label' or, under the provenance rule, the Zenodo label mapped crop-first)
              H5_MATCHED  the same 1 mm frame with the predictions mapped CROP-FIRST, the convention of the reference itself
                          (crop to Gibson's box, then nearest neighbour onto the 1 mm grid, cval 0 beyond the box's last voxel
                          centre; = xdata_common.mact_image_check's image path with order 0). Bit-exact with MACT_h5 on 8/8
                          pool scans; the full-array label path differs only in the 1-2 superior h5 slices of z > 1 mm scans
              NATIVE_BOX  the raw CT grid (LPS order) inside Gibson's box: raw predictions vs the Zenodo label (native)
              NATIVE_FULL the whole raw CT grid (the reference "may be incomplete outside the box": descriptive)
  metrics     H5 frames: xdata_common.score_scan (== every MACT scorer). Native frames: Dice from a joint histogram
              (== xdata_common.dice_vec), NSD 2 mm / HD95 from surface voxel centres with a KD-tree (== xdata_common.
              nsd_hd95, asserted in smoke_a3.py) with the native spacing. Penalised variants (A3.3): an organ absent from the
              reference but predicted scores Dice 0 / NSD 0 / HD95 HD95_EMPTY_MM; GT present + empty prediction HD95 = 900 mm.

This module has NO import-time side effects beyond sys.path (safe before nnunetv2 / torch; GPU jobs import it).
"""
import gzip
import hashlib
import io
import json
import math
import os
import re
import sys
import tarfile
import time
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path("/home/user/Documents/Medical Imagining SSL")
HERE = Path(__file__).resolve().parent
XDATA_DIR = HERE.parent
G1_DIR = XDATA_DIR / "g1"
M2_DIR = REPO / "scripts/vccor/m2"
if str(XDATA_DIR) not in sys.path:
    sys.path.append(str(XDATA_DIR))
import xdata_common as X  # noqa: E402  (frozen)
Q = X.Q

# ---------------------------------------------------------------------------------------------------- layout
AMENDMENT = X.OUT / "prereg_e1_mact_amendment_A3.json"
OUT = X.OUT / "a3"                     # every A3 result lives here
LOOK = OUT / "look"                    # the A3 scorer: maps.json, GT_OPENED, provenance.json, pervol/, results.json, tables.md
AUDIT = OUT / "audit"                  # the post-look audit (results files only, no GT)
POWER = OUT / "power"                  # power_curve.json / .md (A3.5, FLARE / AMOS data only)
ZEN_OUT = OUT / "zenodo"               # pool_check.json (A3.9, labelled-pool scans only)
SMOKE_RES = OUT / "smoke"
TESTS_RES = OUT / "tests"
NN_ROOT = OUT / "nnunet"               # nnUNet_raw / nnUNet_preprocessed of the A3.6 datasets (transient) + results_<run>
LABELS = OUT / "labels"                # labels/<dataset name>/part291/MACT_<cid>.nii.gz (bit-exact copies) + manifest.json
PREP = OUT / "prep"                    # prep/datasets.json, prep/DONE.json
PREREG_LOG = X.PREREG_LOG              # outputs/logs/vccor/xdata_prereg.log
LOGD = X.LOGD
DEADLINE = "2026-10-27 23:59"          # KST (local clock) = A2's latest look time: an A3.6 arm not DONE by then is not run

# the MACT pre-registration chain this amendment builds on: (log kind, file, sha256)
CHAIN = [("PREREG", X.OUT / "prereg_e1_mact.json", "d3021f49e77c164c68828c9ca8623a3536d7f581fdcb4b380c38181f82b59986"),
         ("AMENDMENT", X.OUT / "prereg_e1_mact_amendment_A1.json", "190d728e2a6cceb578852b47dd946bf1bae716e1a7574adb9736f35200e7d972"),
         ("ADDENDUM", X.OUT / "prereg_e1_mact_addendum_G1.json", "9c9c28686490e1ff5cd38b0fdf67e1385500856886d8eb46a35e3f448220d0fe"),
         ("AMENDMENT", X.OUT / "prereg_e1_mact_amendment_A2.json", "125cacac7454d59817cb127f2ee94a6c852a951b52553916f15e3d317c3005c7")]
A1_PREP_SHA = "7f6d96de9f1d9e6faf32d6334328a55247f4416206d4ee3eae0d9c07fdc7694b"     # A1 'code' new_sha256 of prep_xdata.py
MACT_PREP_DONE = X.PREP / "mact" / "DONE_s1337_K4+d1_0075+d1_0049+d1_0037+d1_0041_FULL.json"
M2_PREREG = REPO / "outputs/results/vccor/m2/prereg_m2.json"
M2_PREREG_SHA = "50fa245d122991204e3e63da4d2ee37ad3a0659928fe022e6fd0a55b95c301f1"   # A2 (PREREG line of m2_prereg.log)
M2_PREREG_LOG = LOGD / "m2_prereg.log"
E2_PREREG_MACT = REPO / "outputs/results/vccor/e2/prereg_e2_budget_mact.json"
E2_PREREG_MACT_SHA = "6497ed90b3e23f78592533805f91c57c5a14f2098f28231cc32f767716a326fd"
E2_PREREG_LOG = LOGD / "e2_prereg.log"

NAMES = X.names("mact")                # spleen l.kid gall esoph liver stom panc duod
K = len(NAMES)
SHARED7 = ["spleen", "l.kid", "gall", "esoph", "liver", "stom", "panc"]       # MACT organs BTCV also annotates
POOL = list(X.mact_pool())             # 0075 0049 0037 0041 0070 0087 0011 0080
K4 = POOL[:4]
P2 = POOL[4:]
HD95_EMPTY_MM = 900.0                  # >= the largest full-FOV diagonal of the 89 MACT scans (880.9 mm, image headers only)
NSD_TOL_MM = 2.0
E0_MIN_MM3 = 67.5                      # e0/offsets.py: an organ is absent when |G_o| x voxel volume < 67.5 mm^3


def eval_ids():
    ids = list(X.eval_ids("mact"))
    assert len(ids) == 81
    return ids


def secondary_ids():
    s = list(json.load(open(CHAIN[0][1]))["evaluation"]["secondary"]["ids"])
    assert len(s) == 53 and set(s) <= set(eval_ids())
    return s


def strata():
    a = json.load(open(CHAIN[2][1]))["strata_descriptive"]
    st = {k: list(v) for k, v in a.items() if k != "note"}
    allv = [c for v in st.values() for c in v]
    assert sorted(allv) == sorted(eval_ids()) and len(allv) == len(set(allv)) == 81, "strata must partition the 81"
    return st


def btcv_training_ids():
    """{MACT id: BTCV Training id} of the 28 BTCV-Training images inside the 81 (== g1 BTCV_IDS)."""
    ev = set(eval_ids())
    out = {m: s["original_id"] for m, s in X.mact_sources().items()
           if s["source"] == "btcv" and s["btcv_subset"] == "Training" and m in ev}
    assert len(out) == 28
    return out


# ---------------------------------------------------------------------------------------------------- A3.6 runs
RUNS = {   # run (xdata naming: parse_run -> mact, 1337, draw, FULL, F1b) -> spec
    "mact_s1337_P2_FULL_F1b": dict(arm="P2", draw="P2", labelled=P2, dataset=761, train_seed=1337, worker_seeds=False,
                                   what="FULL K=4 (F1b recipe) on the disjoint MACT pool draw 0070 0087 0011 0080 (training "
                                        "seed 1337, data-augmentation workers seeded from OS entropy = E1's code path)",
                                   train_h=1.25, pred_h=0.6),
    "mact_s1337_K4t2025_FULL_F1b": dict(arm="K4t2025", draw="K4t2025", labelled=K4, dataset=762, train_seed=2025, worker_seeds=True,
                                        what="FULL_K4 replicate on 0075 0049 0037 0041, training seed 2025 (main-process seeds "
                                             "+ explicit nnU-Net data-augmentation worker seeds)", train_h=1.25, pred_h=0.6),
    "mact_s1337_K4t3407_FULL_F1b": dict(arm="K4t3407", draw="K4t3407", labelled=K4, dataset=762, train_seed=3407, worker_seeds=True,
                                        what="FULL_K4 replicate on 0075 0049 0037 0041, training seed 3407 (main-process seeds "
                                             "+ explicit nnU-Net data-augmentation worker seeds)", train_h=1.25, pred_h=0.6),
}
DATASETS = {   # A3 nnU-Net datasets (own nn root; ids not used by any block: TS 291-297, xdata 400-699, e2 7xx<719, m2 80x, 99x)
    761: dict(name="Dataset761_A3MACT291_P2", labelled=P2, consumers=["mact_s1337_P2_FULL_F1b"],
              labels="e2 label set mact_ts (outputs/results/vccor/e2/labels/mact_ts; == xdata composition, bit-identical on K4)"),
    762: dict(name="Dataset762_A3MACT291_K4", labelled=K4, consumers=["mact_s1337_K4t2025_FULL_F1b", "mact_s1337_K4t3407_FULL_F1b"],
              labels="xdata label set s1337_K4_FULL (the labels of E1's FULL_K4)"),
}
E2_LABELS = REPO / "outputs/results/vccor/e2/labels/mact_ts"
XD_K4_LABELS = X.LABELS_ROOT / "mact" / "s1337_K4_FULL"
WORKER_SEED_RULE = "train augmenter worker i: train_seed * 100 + i; validation augmenter worker i: train_seed * 100 + 50 + i"


def worker_seeds(train_seed, n, val=False):
    return [int(train_seed) * 100 + (50 if val else 0) + i for i in range(int(n))]


def run_spec(run):
    s = dict(RUNS[run])
    s["dataset_name"] = DATASETS[s["dataset"]]["name"]
    return s


def pred_dir(run):
    return X.PRED / run                 # predict_xdata.py's own output (frozen code, non-smoke)


# ---------------------------------------------------------------------------------------------------- arms of the A3 look
def arms():
    """Arm -> source. kind ts: raw = TS 'total' ids (lut total_lut('mact')); mrseg: MRSegmentator ids (m2 mrseg_lut)."""
    P = X.PRED
    a = {"F0": dict(dir=P / "mact_F0", lut="ts", group="E1", mandatory=True),
         "FULL_K4": dict(dir=P / "mact_s1337_K4_FULL_F1b", lut="ts", group="E1", mandatory=True),
         "NORM_K4": dict(dir=P / "mact_s1337_K4_FULL_F3", lut="ts", group="E1", mandatory=True)}
    for c in K4:
        a[f"FULL_K1@{c}"] = dict(dir=P / f"mact_s1337_d1_{c}_FULL_F1b", lut="ts", group="E1", mandatory=True)
    a["X1"] = dict(dir=X.OUT / "g1" / "pred" / "X1", lut="ts", group="G1", mandatory=True)
    a["M0"] = dict(dir=REPO / "outputs/results/vccor/m2/pred/M0_mact", lut="mrseg", group="A2", mandatory=True)
    a["MF"] = dict(dir=REPO / "outputs/results/vccor/m2/pred/mact_s1337", lut="mrseg", group="A2", mandatory=True)
    for run, s in RUNS.items():
        a[s["arm"]] = dict(dir=pred_dir(run), lut="ts", group="A3.6", mandatory=False, run=run)
    return a


K1_ARMS = [f"FULL_K1@{c}" for c in K4]
A36_ARMS = [s["arm"] for s in RUNS.values()]


MRSEG_DATASET_JSON = REPO / "outputs/results/vccor/m2/weights/mrseg_v1.1.0/dataset.json"
MRSEG_DATASET_JSON_SHA = "328d28d4787e99e1d4a3e0446d76e0c85ad56983b5aa1c8317de8defc384b2ec"
MRSEG_MACT_NAMES = ["spleen", "left_kidney", "gallbladder", "esophagus", "liver", "stomach", "pancreas", "duodenum"]


def mrseg_lut():
    """uint8 LUT MRSegmentator v1.1.0 id -> MACT id by class name (the weights' dataset.json; == m2_common.mrseg_lut('mact'),
    asserted in smoke_a3.py S0; at the look the raw -> h5 identity check re-proves it on every m2 map)."""
    import numpy as np
    assert X.sha256(MRSEG_DATASET_JSON) == MRSEG_DATASET_JSON_SHA, "MRSegmentator dataset.json changed"
    lab = {k: int(v) for k, v in json.load(open(MRSEG_DATASET_JSON))["labels"].items()}
    lut = np.zeros(256, np.uint8)
    for o, n in enumerate(MRSEG_MACT_NAMES, start=1):
        lut[lab[n]] = o
    return lut


def luts():
    """{'ts': TS 'total' id -> MACT id (xdata_common.total_lut), 'mrseg': MRSegmentator id -> MACT id (by name)}."""
    return {"ts": X.total_lut("mact"), "mrseg": mrseg_lut()}


# ---------------------------------------------------------------------------------------------------- chain / hashes
def logged(kind, path, log=None):
    """sha256 values logged as '<kind> sha256 <hex> <repo-relative path>' (in order; the last one wins)."""
    log = Path(log or PREREG_LOG)
    want, out = X.rel(path), []
    if log.is_file():
        for line in open(log):
            p = line.split()
            if len(p) >= 6 and p[2] == kind and p[3] == "sha256" and " ".join(p[5:]) == want:
                out.append(p[4])
    return out


def check_chain(include_a3=True):
    """A3.8: every file of the MACT chain hashes to its pinned value AND to its last logged line; with include_a3 also A3
    (AMENDMENT line) and A3's own 'amends' block names the same hashes. Returns {path: sha256}. Refuses otherwise."""
    out = {}
    for kind, path, pin in CHAIN:
        h, lg = X.sha256(path), logged(kind, path)
        if h != pin or not lg or lg[-1] != pin:
            raise SystemExit(f"[REFUSE] {X.rel(path)}: sha256 {h[:16]} pinned {pin[:16]} logged {[x[:16] for x in lg]}")
        out[X.rel(path)] = h
    if include_a3:
        h, lg = X.sha256(AMENDMENT), logged("AMENDMENT", AMENDMENT)
        if not lg or lg[-1] != h:
            raise SystemExit(f"[REFUSE] {X.rel(AMENDMENT)} sha256 {h[:16]} != its last AMENDMENT line {[x[:16] for x in lg]}")
        A = json.load(open(AMENDMENT))
        named = {k: v for k, v in A["amends"].items() if k.endswith("_sha256")}
        want = {"prereg_e1_mact_sha256": CHAIN[0][2], "amendment_A1_sha256": CHAIN[1][2], "addendum_G1_sha256": CHAIN[2][2],
                "amendment_A2_sha256": CHAIN[3][2]}
        if named != want:
            raise SystemExit(f"[REFUSE] A3 'amends' block {named} != the chain {want}")
        out[X.rel(AMENDMENT)] = h
    return out


def amendment():
    check_chain(include_a3=True)
    return json.load(open(AMENDMENT))


_RULE_TEXT = {   # what the G1 addendum / m2 prereg say, in words, for the rules A3 re-evaluates
    "P1": ("P1_external_criterion", "pass_rule"), "S1": ("secondary_criteria", "S1_single_scan", "pass_rule"),
    "S2": ("secondary_criteria", "S2_never_used", "pass_rule")}


def _dig(d, path):
    for k in path:
        d = d[k]
    return d


def check_rules_vs_sources(A):
    """A3.8: the structured thresholds in A3 'rules' == the numbers written in the G1 addendum (P1 / S1 / S2 / X1) and the
    m2 prereg (H1-MACT). The A3 scorer reads every threshold from A3; this proves they are G1's / m2's. Returns the checks."""
    G1 = json.load(open(CHAIN[2][1]))
    R = A["rules"]
    chk = {}
    for k in ("P1", "S2", "S1"):
        txt = _dig(G1, _RULE_TEXT[k])
        m = re.search(r"M >= \+([0-9.]+)", txt)
        chk[f"{k}_min_mean"] = m is not None and abs(float(m.group(1)) - R[k]["min_mean"]) < 1e-12
        chk[f"{k}_t_lower"] = "t-CI lower bound > 0" in txt and R[k]["t_ci_lower_gt"] == 0.0
        if k in ("P1", "S2"):
            m = re.search(r"at least (\d+) of (\d+)", txt)
            chk[f"{k}_min_positive"] = m is not None and (int(m.group(1)), int(m.group(2))) == (R[k]["min_positive"], R[k]["n"])
        else:
            m = re.search(r"at least (\d+) of the (\d+) single-scan draws", txt)
            chk["S1_draws"] = m is not None and (int(m.group(1)), int(m.group(2))) == (R["S1"]["min_draws_positive"], 4)
    x1 = G1["added_analyses"]["X1_transfer_prediction"]
    m = re.search(r"SUPPORTED if >= (\d+) of (\d+) signs match; PARTLY if (\d+)-(\d+); NOT SUPPORTED if <= (\d+)", x1["verdict_rule"])
    chk["X1_rule"] = m is not None and [int(x) for x in m.groups()] == [R["X1"]["supported_min"], len(R["X1"]["organs_tested"]),
                                                                       R["X1"]["partly_min"], R["X1"]["supported_min"] - 1,
                                                                       R["X1"]["partly_min"] - 1]
    pred = {n: v["predicted_sign"] for n, v in x1["per_organ"].items()}
    chk["X1_signs"] = all(pred[n] == s for n, s in R["X1"]["predicted"].items()) and \
        sorted(R["X1"]["organs_tested"]) == sorted(n for n, s in pred.items() if s in ("+", "-"))
    chk["X1R_signs"] = all(pred[n] == s for n, s in R["X1R"]["predicted"].items())
    m2 = json.load(open(M2_PREREG))["looks"]["mact"]["hypothesis"]["rule"]
    chk["H1_MACT"] = (abs(m2["min_mean"] - R["H1_MACT"]["min_mean"]) < 1e-12 and
                      abs(m2["min_fraction_positive"] - R["H1_MACT"]["min_fraction_positive"]) < 1e-12)
    chk["EXT_is_P1_and_S2"] = R["EXT"]["requires"] == ["P1", "S2"]
    return chk


def check_applied(A):
    """A3.8 'read AND applied': A1 = the MACT prep that made E1's training labels ran the A1 code (prep_xdata.py sha256 =
    A1's new hash); G1 / m2 = A3's thresholds equal theirs (check_rules_vs_sources); A2 = MRSegmentator's MACT arms are
    in this look and its prereg hash is the one A2 names (= m2_prereg.log PREREG line); A3 = its rules are what the scorer
    uses (the scorer passes A['rules'] to every verdict). Returns {check: bool}."""
    A1 = json.load(open(CHAIN[1][1]))
    out = {"A1_new_prep_sha_named": A1["code"]["scripts/vccor/xdata/prep_xdata.py"]["new_sha256"] == A1_PREP_SHA}
    d = json.load(open(MACT_PREP_DONE))
    out["A1_applied_in_mact_prep"] = d["code_sha256"].get("scripts/vccor/xdata/prep_xdata.py") == A1_PREP_SHA
    A2 = json.load(open(CHAIN[3][1]))
    out["A2_names_m2_score_mact"] = "m2_score_mact" in A2["release_rule"]
    out["A2_m2_prereg_sha"] = (X.sha256(M2_PREREG) == M2_PREREG_SHA and M2_PREREG_SHA.startswith(A2["change"].split("sha256 ")[1][:8])
                               and (logged("PREREG", M2_PREREG, M2_PREREG_LOG) or [None])[-1] == M2_PREREG_SHA)
    out["A2_m2_arms_in_look"] = all(a in arms() for a in ("M0", "MF"))
    out["A3_release_set"] = set(A["release"]["jobs"]) >= {"xd_mact_score_e1", "e2_score_budget_mact", "g1_score", "m2_score_mact",
                                                         "a3_score", "a3_audit"}
    out.update({f"rule_{k}": v for k, v in check_rules_vs_sources(A).items()})
    return out


# ---------------------------------------------------------------------------------------------------- Zenodo (A3.9)
ZEN_DIR = X.DATA / "mact_zenodo_1169361"
ZEN_TAR = {"tcia": ZEN_DIR / "label_tciapancreasct_multiorgan.tar.gz", "btcv": ZEN_DIR / "label_btcv_multiorgan.tar.gz"}
ZEN_SUB = {"tcia": "label_tcia_multiorgan", "btcv": "label_btcv_multiorgan"}
ZEN_SHA = {"label_tciapancreasct_multiorgan.tar.gz": "1790e252ba0732cc06ec727f00245c8d9c1d5f0bb9829fd3e9915863c5100c61",
           "label_btcv_multiorgan.tar.gz": "bb080c7de1094cc0ee46a5e1bef5b66e635075f8e9a3080ca681086cd2723998",
           "cropping.csv": "d6503bda3d776c10698523aae14446409bc9d3722fb405b0025b4a5f59094f8f",
           "index.htm": "ee147386daa1e40aa4b8c4cf9760024a6bd99ca3c370fbdcbe59316ad831f784"}
ZEN_MD5 = {"label_tciapancreasct_multiorgan.tar.gz": "f27695d2f91c1642cfa077d1e768e6af",       # the record's own checksums
           "label_btcv_multiorgan.tar.gz": "90d6f2bb4fd912d179cd8d9fdd2b621a",
           "cropping.csv": "69acd775d4c91d64ebc813216e924797", "index.htm": "0d79e751a06fe8eb3cd07212ac5ba5b6"}
# Gibson's 14 label values (index.htm 'File format', in order): 1 spleen 2 r.kid* 3 l.kid 4 gall 5 esoph 6 liver 7 stom
# 8 aorta* 9 ivc* 10 portal+splenic vein* 11 panc 12 r.adr* 13 l.adr* 14 duod (* BTCV images only, not MACT classes -> 0)
ZEN_TO_MACT = {1: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 11: 7, 14: 8}


class Unverified(Exception):
    """A Zenodo label that cannot be read or does not sit on the raw CT grid (A3.9: the scan is UNVERIFIED)."""


def zen_lut():
    import numpy as np
    lut = np.zeros(256, np.uint8)
    for z, m in ZEN_TO_MACT.items():
        lut[z] = m
    return lut


def zenodo_files_ok():
    """sha256 + the Zenodo record's md5 of the downloaded files. Returns {file: bool}."""
    out = {}
    for f, h in ZEN_SHA.items():
        p = ZEN_DIR / f
        ok = p.is_file() and X.sha256(p) == h
        if ok:
            md = hashlib.md5()
            with open(p, "rb") as fh:
                for b in iter(lambda: fh.read(1 << 22), b""):
                    md.update(b)
            ok = md.hexdigest() == ZEN_MD5[f]
        out[f] = bool(ok)
    return out


def zenodo_member(mid):
    s = X.mact_sources()[mid]
    return ZEN_TAR[s["source"]], f"{ZEN_SUB[s['source']]}/label{s['original_id']}.nii.gz"


def zenodo_member_bytes(mid, marker=None):
    """The ONLY reader of the Zenodo label files. A labelled-pool scan: always (training scan). An evaluation scan: only when
    `marker` (the A3 look's GT_OPENED file) exists. Reads the one member into memory (nothing is extracted to disk)."""
    if mid not in POOL:
        if mid not in set(eval_ids()):
            raise SystemExit(f"[REFUSE] Zenodo label of MACT {mid}: neither a pool scan nor an evaluation scan")
        if marker is None or not Path(marker).is_file():
            raise SystemExit(f"[REFUSE] Zenodo label of evaluation scan {mid} requested outside the A3 look (no GT_OPENED)")
    tar, name = zenodo_member(mid)
    with tarfile.open(tar, "r:gz") as t:
        try:
            m = t.getmember(name)
        except KeyError:
            raise Unverified(f"member {name} missing")
        if not m.isfile() or m.name != name or m.name.startswith("/") or ".." in m.name.split("/"):
            raise Unverified(f"member {name} is not a plain file")
        return t.extractfile(m).read()


def zenodo_image(mid, marker=None):
    import nibabel as nib
    data = zenodo_member_bytes(mid, marker)
    fh = nib.FileHolder(fileobj=io.BytesIO(gzip.decompress(data)))
    return nib.Nifti1Image.from_file_map({"header": fh, "image": fh})


def zenodo_raw14(mid, img, marker=None):
    """Gibson's label (14 values) on OUR raw CT grid (nibabel order of the raw CT file): the label is reoriented with its own
    affine onto the raw CT's orientation (TCIA labels are stored LAI with origin 0, our TCIA conversion LAS), then shape
    and affine must agree (1e-3). Validated: 8/8 pool scans map bit-exactly onto MACT_h5 (crop-first). Raises Unverified."""
    import nibabel as nib
    import numpy as np
    z = zenodo_image(mid, marker)
    a = np.asanyarray(z.dataobj)
    ai = np.rint(a)
    if not np.array_equal(ai, a) or ai.min() < 0 or ai.max() > 14:
        raise Unverified(f"label values outside the integers 0..14 ({float(a.min())}, {float(a.max())})")
    tr = nib.orientations.ornt_transform(nib.orientations.io_orientation(z.affine), nib.orientations.io_orientation(img.affine))
    A = nib.orientations.apply_orientation(ai.astype(np.uint8), tr)
    aff = z.affine @ nib.orientations.inv_ornt_aff(tr, z.shape[:3])
    if tuple(A.shape) != tuple(img.shape[:3]) or not np.allclose(aff, img.affine, atol=1e-3):
        raise Unverified(f"label grid {A.shape} / affine differ from the raw CT {img.shape[:3]}")
    return np.ascontiguousarray(A).astype(np.uint8)


def zenodo_raw_mact(mid, img, marker=None):
    return zen_lut()[zenodo_raw14(mid, img, marker)]


# ---------------------------------------------------------------------------------------------------- frames
def frame(mid, img):
    return X.mact_frame(mid, img)


def to_lps(arr_raw, fr):
    import nibabel as nib
    import numpy as np
    assert tuple(arr_raw.shape) == tuple(fr["raw_shape"]), (arr_raw.shape, fr["raw_shape"])
    return np.ascontiguousarray(nib.orientations.apply_orientation(np.asarray(arr_raw), fr["ornt"]))


def box_slices(fr):
    st, sz = fr["start"].astype(int), fr["size"].astype(int)
    return tuple(slice(int(a), int(a + b)) for a, b in zip(st, sz))


def crop_first_h5(seg_raw, mid, img, fr=None):
    """The reference's own convention (MagicNet MACT_h5 labels; xdata_common.mact_image_check's image path with order 0):
    raw -> LPS -> crop to Gibson's box -> nearest neighbour onto the 1 mm grid (h5 voxel j <-> crop index j / spacing),
    cval 0 beyond the crop's last voxel centre. Bit-exact with MACT_h5 'label' on the 8 pool scans (Zenodo labels)."""
    import numpy as np
    from scipy import ndimage as ndi
    fr = fr or frame(mid, img)
    C = to_lps(np.asarray(seg_raw).astype(np.uint8), fr)[box_slices(fr)]
    out = ndi.affine_transform(C, np.diag(1.0 / fr["spacing"]), offset=0.0, output_shape=fr["shape"], order=0, mode="constant",
                               cval=0, prefilter=False)
    return out.astype(np.uint8)


def full_array_h5(seg_raw, mid, img):
    """The predictions' path of every MACT scorer (xdata_common.mact_raw_to_h5, nearest neighbour on the full raw array)."""
    return X.mact_raw_to_h5(seg_raw, mid, img)


# ---------------------------------------------------------------------------------------------------- metrics
def joint_counts(Y, G, k=K):
    """(k+1) x (k+1) joint histogram [G, Y] of two label maps (values 0..k)."""
    import numpy as np
    n = k + 1
    return np.bincount((G.astype(np.int64).ravel() * n + Y.astype(np.int64).ravel()), minlength=n * n).reshape(n, n)


def dice_from_counts(H):
    """Per organ Dice x 100 from a joint histogram; nan when the organ is absent from G (== xdata_common.dice_vec)."""
    import numpy as np
    k = H.shape[0] - 1
    g, y = H.sum(1), H.sum(0)
    out = []
    for o in range(1, k + 1):
        out.append(float("nan") if g[o] == 0 else float(2 * H[o, o] / (g[o] + y[o])) * 100.0)
    return out


def _surface_points(m):
    """Surface voxels (mask minus its 6-connected erosion, as xdata_common.nsd_hd95) as integer index coordinates."""
    import numpy as np
    from scipy import ndimage
    idx = np.argwhere(m)
    lo = np.maximum(idx.min(0) - 2, 0)
    hi = np.minimum(idx.max(0) + 3, m.shape)
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    c = m[sl]
    s = c & ~ndimage.binary_erosion(c)
    return np.argwhere(s) + lo


def nsd_hd95_kd(Y, G, o, sp, tol=NSD_TOL_MM):
    """== xdata_common.nsd_hd95 (NSD at tol mm x 100, HD95 in mm; GT absent -> (nan, nan), empty prediction -> (0, nan)) via
    nearest surface-voxel-centre distances (cKDTree) instead of an EDT over the bounding box (memory-safe on full fields)."""
    import numpy as np
    from scipy.spatial import cKDTree
    a, b = Y == o, G == o
    if not b.any():
        return float("nan"), float("nan")
    if not a.any():
        return 0.0, float("nan")
    sp = np.asarray(sp, float)
    pa, pb = _surface_points(a) * sp, _surface_points(b) * sp
    d_ab = cKDTree(pb).query(pa, k=1)[0]
    d_ba = cKDTree(pa).query(pb, k=1)[0]
    nsd = float(((d_ab <= tol).sum() + (d_ba <= tol).sum()) / (len(d_ab) + len(d_ba)))
    hd = float(max(np.percentile(d_ab, 95), np.percentile(d_ba, 95)))
    return nsd * 100.0, hd


def score_native(maps, G, sp, surface=True):
    """{arm: map} vs G on a native (anisotropic) grid: the structure of xdata_common.score_scan with the native spacing."""
    import numpy as np
    from threadpoolctl import threadpool_limits
    rec = {"gt_vol": None, "present": None, "dice": {}, "vol": {}, "nsd": {}, "hd95": {}}
    with threadpool_limits(1):
        for a, Y in maps.items():
            assert Y.shape == G.shape, (a, Y.shape, G.shape)
            H = joint_counts(Y, G)
            if rec["gt_vol"] is None:
                rec["gt_vol"] = H.sum(1)[1:].astype(int).tolist()
                rec["present"] = [v > 0 for v in rec["gt_vol"]]
            rec["dice"][a] = dice_from_counts(H)
            rec["vol"][a] = H.sum(0)[1:].astype(int).tolist()
            if surface:
                sh = [nsd_hd95_kd(Y, G, o, sp) for o in range(1, K + 1)]
                rec["nsd"][a] = [s[0] for s in sh]
                rec["hd95"][a] = [s[1] for s in sh]
    return rec


def score_native_full(maps, G, rec_box, box, sp):
    """NATIVE_FULL record from the full LPS arrays: Dice / volumes from the full joint histograms; NSD / HD95 recomputed only
    for the (arm, organ) pairs with predicted or reference voxels OUTSIDE the box -- for every other pair both masks lie inside
    the box, so the surfaces (6-connected erosion; nothing of either mask outside) and their distances are those of
    NATIVE_BOX exactly (asserted on pool scans in smoke_a3.py S1)."""
    import numpy as np
    from threadpoolctl import threadpool_limits
    rec = {"gt_vol": None, "present": None, "dice": {}, "vol": {}, "nsd": {}, "hd95": {}, "recomputed": {}}
    gin = np.array(rec_box["gt_vol"], int)
    with threadpool_limits(1):
        for a, Y in maps.items():
            H = joint_counts(Y, G)
            if rec["gt_vol"] is None:
                rec["gt_vol"] = H.sum(1)[1:].astype(int).tolist()
                rec["present"] = [v > 0 for v in rec["gt_vol"]]
            rec["dice"][a] = dice_from_counts(H)
            rec["vol"][a] = H.sum(0)[1:].astype(int).tolist()
            yin = np.array(rec_box["vol"][a], int)
            out_y, out_g = np.array(rec["vol"][a]) - yin, np.array(rec["gt_vol"]) - gin
            nsd, hd, redo = [], [], []
            for o in range(1, K + 1):
                if out_y[o - 1] == 0 and out_g[o - 1] == 0:
                    nsd.append(rec_box["nsd"][a][o - 1])
                    hd.append(rec_box["hd95"][a][o - 1])
                else:
                    s = nsd_hd95_kd(Y, G, o, sp)
                    nsd.append(s[0])
                    hd.append(s[1])
                    redo.append(NAMES[o - 1])
            rec["nsd"][a], rec["hd95"][a], rec["recomputed"][a] = nsd, hd, redo
    return rec


def penalised(rec):
    """A3.3 penalised per-organ values from a score record: Dice / NSD 0 and HD95 = HD95_EMPTY_MM for a predicted organ the
    reference lacks; HD95 = HD95_EMPTY_MM for a reference organ with an empty prediction; both absent -> nan (excluded).
    Returns {'dice_pen', 'nsd_pen', 'hd95_pen', 'fp_organ' (bool), 'pred_empty' (bool)} per arm."""
    out = {k: {} for k in ("dice_pen", "nsd_pen", "hd95_pen", "fp_organ", "pred_empty")}
    for a in rec["dice"]:
        dp, npn, hp, fp, pe = [], [], [], [], []
        for j in range(K):
            g, y = rec["gt_vol"][j], rec["vol"][a][j]
            is_fp, is_empty = (g == 0 and y > 0), (g > 0 and y == 0)
            fp.append(bool(is_fp))
            pe.append(bool(is_empty))
            if is_fp:
                dp.append(0.0)
                npn.append(0.0)
                hp.append(HD95_EMPTY_MM)
            elif g == 0:
                dp.append(float("nan"))
                npn.append(float("nan"))
                hp.append(float("nan"))
            else:
                dp.append(rec["dice"][a][j])
                npn.append(rec["nsd"][a][j] if rec["nsd"] else float("nan"))
                hv = rec["hd95"][a][j] if rec["hd95"] else float("nan")
                hp.append(HD95_EMPTY_MM if is_empty else hv)
        out["dice_pen"][a], out["nsd_pen"][a], out["hd95_pen"][a], out["fp_organ"][a], out["pred_empty"][a] = dp, npn, hp, fp, pe
    return out


def offsets_vox(Gm, Y, vox_mm3):
    """== g1_common.offsets for any voxel volume: per organ s = (fn - fp) / (fn + fp) of the reference Gm vs the map Y inside
    the bbox of Gm | Y padded by 2 (s = 0 if fn + fp = 0); None when |Gm_o| x vox_mm3 < 67.5 mm^3 (E0's absence rule)."""
    import numpy as np
    out = {}
    for o in range(1, K + 1):
        Go, To = Gm == o, Y == o
        if int(Go.sum()) * vox_mm3 < E0_MIN_MM3:
            out[NAMES[o - 1]] = None
            continue
        U = Go | To
        idx = np.argwhere(U)
        lo, hi = np.maximum(idx.min(0) - 2, 0), np.minimum(idx.max(0) + 3, U.shape)
        sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        fn, fp = int((Go[sl] & ~To[sl]).sum()), int((To[sl] & ~Go[sl]).sum())
        out[NAMES[o - 1]] = float((fn - fp) / (fn + fp)) if fn + fp else 0.0
    return out


# ---------------------------------------------------------------------------------------------------- rules (A3 'rules')
def t_ci95(d):
    import numpy as np
    from scipy import stats
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    if len(d) < 2:
        return [None, None]
    se = d.std(ddof=1) / np.sqrt(len(d))
    q = stats.t.ppf(0.975, len(d) - 1)
    return [float(d.mean() - q * se), float(d.mean() + q * se)]


def rule_verdict(d, rule):
    """The P1-type rule on paired per-scan differences d (rule from A3 'rules'): PASS iff mean >= min_mean AND the 95% t-CI
    lower bound > t_ci_lower_gt AND (#d > 0) >= the required count. The count is min_positive when given, else
    ceil(min_fraction_positive x n - 1e-9) (m2's formula). Strict '>' for positivity and for the CI bound."""
    import numpy as np
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    n = int(len(d))
    M = float(d.mean()) if n else None
    ci = t_ci95(d)
    up = int((d > 0).sum())
    need = int(rule["min_positive"]) if "min_positive" in rule else int(math.ceil(rule["min_fraction_positive"] * n - 1e-9))
    crit = {"M_ge_min": bool(M is not None and M >= rule["min_mean"]),
            "t_ci_lower_gt": bool(ci[0] is not None and ci[0] > rule["t_ci_lower_gt"]),
            "positive_ge_needed": bool(up >= need)}
    if "n" in rule and n != int(rule["n"]):
        crit["n_as_registered"] = False
    return {"PASS": bool(all(crit.values())), "M": M, "t_ci95": ci, "positive": up, "n": n, "positive_needed": need,
            "criteria": crit}


def sign_match(obs, predicted):
    """{organ: observed mean} vs {organ: '+'/'-'}: matches (0 counts as a miss)."""
    return {n: bool(obs.get(n) is not None and ((obs[n] > 0 and s == "+") or (obs[n] < 0 and s == "-"))) for n, s in predicted.items()}


def x1_verdict(obs, rule):
    m = sign_match(obs, rule["predicted"])
    k = sum(m.values())
    v = "SUPPORTED" if k >= rule["supported_min"] else ("PARTLY" if k >= rule["partly_min"] else "NOT SUPPORTED")
    return {"verdict": v, "n_match": k, "n_tested": len(m), "match": m}


def x1r_verdict(obs, rule):
    m = sign_match(obs, rule["predicted"])
    k = sum(m.values())
    return {"verdict": "SUPPORTED" if k == len(m) else "NOT SUPPORTED", "n_match": k, "n_tested": len(m), "match": m}


# ---------------------------------------------------------------------------------------------------- helpers
sha256, rel, write_json, Log, be_nice, cap_threads = X.sha256, X.rel, X.write_json, X.Log, X.be_nice, X.cap_threads


def now():
    return time.strftime("%F %T")


def arr_sha(a):
    return X.h5_sha(a)


def deadline_passed():
    return time.strftime("%Y-%m-%d %H:%M") > DEADLINE


def frozen_check(paths, log=print):
    """Real (non-smoke) jobs: strict CODE / IMPORT check against xdata_prereg.log once the a3 CODEFREEZE exists."""
    if any("CODEFREEZE a3" in line for line in open(PREREG_LOG)):
        return X.check_code_frozen(paths, log, strict=True)
    log(f"[warn] no 'CODEFREEZE a3' in {rel(PREREG_LOG)} yet: running unfrozen code")
    return False


def require_frozen(paths, log=print):
    """The look (score_a3 / audit_a3): refuse unless the a3 CODEFREEZE exists and the code matches it."""
    if not any("CODEFREEZE a3" in line for line in open(PREREG_LOG)):
        raise SystemExit(f"[REFUSE] no 'CODEFREEZE a3' line in {rel(PREREG_LOG)}")
    return X.check_code_frozen(paths, log, strict=True)


COMMON_CODE = [HERE / "a3_common.py", XDATA_DIR / "xdata_common.py", X.TSFT_DIR / "tsft_full_common.py"]
