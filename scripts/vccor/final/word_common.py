"""vccor/final plan C (WORD, fix item 8): the one cohort whose labels this project has never opened.

Data: WORD-V0.1.0 (Luo et al., Medical Image Analysis 82:102642, 2022), downloaded 2026-10-22 into MagicNet/data/word_v010/
(archive/WORD-V0.1.0.zip, AES-encrypted, password published in the WORD README; archive_labelsTs/WORD_V0.1.0_labelsTs.zip =
the 30 test labels, the only place they exist; sha256 of both in outputs/results/vccor/final/C_word/download.json, pinned in
the plan). Official split (archive folders == the VISTA repo's WORD.json): 100 train / 20 val / 30 test. NO label member of
either archive has been read before the plan's registration: training labels (labelsTr) are read only by the prep for the
K = 4 draw scans; test labels only by the look, in memory, after its GT_OPENED marker (reference()).

Analysis label space (15 organs; WORD raw id -> analysis id):
   1 liver 2 spleen 3 kidney_left 4 kidney_right 5 stomach 6 gallbladder 7 esophagus 8 pancreas 9 duodenum
   10 colon_rectum (WORD colon 10 + rectum 13: TotalSegmentator / MRSegmentator / VISTA3D have one colon class, no rectum)
   11 intestine (WORD 11 = small bowel)  12 adrenal (WORD has ONE adrenal class: the models' left + right glands merged)
   13 bladder  14 femur_head_left  15 femur_head_right
PRIMARY organs (adapted and scored): 1-9, 11, 12, 13 (12 organs, one model class each). SECONDARY (adapted, scored separately):
10 colon_rectum. DESCRIPTIVE only (not adapted, protocol-mismatched: the models label the whole femur, WORD the femoral head):
14, 15. Training targets: WORD GT for organs 1-13 (adrenal voxels assigned to the model's left / right gland by
split_adrenal()), each model's own prediction for every other class (== the project's composite rule).
No import-time side effects beyond sys.path.
"""
import io
import json
import gzip
import os
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import final_common as F  # noqa: E402

PLAN = "word"
DIR = F.PLAN_DIR[PLAN]
WORD_DIR = F.DATA / "word_v010"
ARCHIVE = WORD_DIR / "archive/WORD-V0.1.0.zip"
LABELS_TS_ZIP = WORD_DIR / "archive_labelsTs/WORD_V0.1.0_labelsTs.zip"
IMAGES = WORD_DIR / "images"                     # extracted images of the test set + draw scans (word_prep.py)
META = WORD_DIR / "meta"
DOWNLOAD = DIR / "download.json"
PASSWORD = "word@uestc"                          # published in the WORD repository README (2024-07-25)
SEVEN_Z = "/usr/bin/7z"
NN_ROOT = DIR / "nnunet"
PREP = DIR / "prep"
MODELS = DIR / "models"
LOOK = DIR / "look"
F.allow_write_root(IMAGES)

ORGANS = ["liver", "spleen", "kidney_left", "kidney_right", "stomach", "gallbladder", "esophagus", "pancreas", "duodenum",
          "colon_rectum", "intestine", "adrenal", "bladder", "femur_head_left", "femur_head_right"]
N_ORG = len(ORGANS)
PRIMARY = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13]
SECONDARY = [10]
DESCRIPTIVE = [14, 15]
ADAPTED = list(range(1, 14))
WORD_TO_ANALYSIS = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 13: 10, 11: 11, 12: 12, 14: 13, 15: 14, 16: 15}
ADRENAL, ADRENAL_R_TMP = 12, 17                  # internal training-label ids: 12 = left gland, 17 = right gland
TS_NAMES = {1: ["liver"], 2: ["spleen"], 3: ["kidney_left", "kidney_cyst_left"], 4: ["kidney_right", "kidney_cyst_right"],
            5: ["stomach"], 6: ["gallbladder"], 7: ["esophagus"], 8: ["pancreas"], 9: ["duodenum"], 10: ["colon"],
            11: ["small_bowel"], 12: ["adrenal_gland_left", "adrenal_gland_right"], 13: ["urinary_bladder"],
            14: ["femur_left"], 15: ["femur_right"]}
TS_CANON = {1: "liver", 2: "spleen", 3: "kidney_left", 4: "kidney_right", 5: "stomach", 6: "gallbladder", 7: "esophagus",
            8: "pancreas", 9: "duodenum", 10: "colon", 11: "small_bowel", 12: "adrenal_gland_left", 13: "urinary_bladder",
            ADRENAL_R_TMP: "adrenal_gland_right"}
VISTA_NAMES = {1: ["liver"], 2: ["spleen"], 3: ["left kidney", "left kidney cyst"], 4: ["right kidney", "right kidney cyst"],
               5: ["stomach"], 6: ["gallbladder"], 7: ["esophagus"], 8: ["pancreas"], 9: ["duodenum"], 10: ["colon"],
               11: ["small bowel"], 12: ["left adrenal gland", "right adrenal gland"], 13: ["bladder"], 14: ["left femur"],
               15: ["right femur"]}
VISTA_CANON = {1: "liver", 2: "spleen", 3: "left kidney", 4: "right kidney", 5: "stomach", 6: "gallbladder", 7: "esophagus",
               8: "pancreas", 9: "duodenum", 10: "colon", 11: "small bowel", 12: "left adrenal gland", 13: "bladder",
               ADRENAL_R_TMP: "right adrenal gland"}
MRSEG_NAMES = {1: ["liver"], 2: ["spleen"], 3: ["left_kidney"], 4: ["right_kidney"], 5: ["stomach"], 6: ["gallbladder"],
               7: ["esophagus"], 8: ["pancreas"], 9: ["duodenum"], 10: ["colon"], 11: ["small_bowel"],
               12: ["left_adrenal_gland", "right_adrenal_gland"], 13: ["urinary_bladder"], 14: ["left_femur"], 15: ["right_femur"]}
MRSEG_CANON = {1: "liver", 2: "spleen", 3: "left_kidney", 4: "right_kidney", 5: "stomach", 6: "gallbladder", 7: "esophagus",
               8: "pancreas", 9: "duodenum", 10: "colon", 11: "small_bowel", 12: "left_adrenal_gland", 13: "urinary_bladder",
               ADRENAL_R_TMP: "right_adrenal_gland"}
DRAW_SEEDS = {"d1": 1337, "d2": 2025, "d3": 3407}
K = 4


# ---------------------------------------------------------------------------------------------------- split / draws
def download_record():
    return json.load(open(DOWNLOAD))


def split():
    s = download_record()["official_split"]
    return {"train": list(s["train"]), "val": list(s["val"]), "test": list(s["test"])}


def draw(d):
    """K = 4 training scans of draw d: numpy RandomState(seed).shuffle(sorted train ids)[:4] (== the project's trainer draw)."""
    import numpy as np
    tr = sorted(split()["train"])
    np.random.RandomState(DRAW_SEEDS[d]).shuffle(tr)
    return tr[:K]


def draw_scans():
    return sorted({c for d in DRAW_SEEDS for c in draw(d)})


def image_path(cid):
    return IMAGES / f"word_{cid}.nii.gz"


def member(kind, cid):
    sub = {"imagesTr": "imagesTr", "imagesTs": "imagesTs", "imagesVal": "imagesVal", "labelsTr": "labelsTr"}[kind]
    return f"WORD-V0.1.0/{sub}/word_{cid}.nii.gz"


def image_folder(cid):
    s = split()
    return "imagesTr" if cid in s["train"] else ("imagesTs" if cid in s["test"] else "imagesVal")


def extract_images(ids, log=print):
    """Extract the IMAGES of `ids` from the encrypted archive with the system 7z (no label member is touched)."""
    IMAGES.mkdir(parents=True, exist_ok=True)
    todo = [c for c in ids if not image_path(c).is_file()]
    for c in todo:
        m = member(image_folder(c), c)
        assert "/images" in m and "/labels" not in m, m
        tmp = IMAGES / f".tmp_{c}"
        tmp.mkdir(exist_ok=True)
        r = subprocess.run([SEVEN_Z, "e", "-y", f"-p{PASSWORD}", f"-o{tmp}", str(ARCHIVE), m], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-500:]
        os.replace(tmp / Path(m).name, image_path(c))
        tmp.rmdir()
        log(f"[images] extracted {m}")
    return {c: F.sha256(image_path(c)) for c in ids}


def nib_from_gz_bytes(b):
    import nibabel as nib
    fh = nib.FileHolder(fileobj=io.BytesIO(gzip.decompress(b)))
    return nib.Nifti1Image.from_file_map({"header": fh, "image": fh})


def train_label(cid, allowed, log=print):
    """WORD labelsTr member of a TRAINING scan of a draw, read into memory only (7z -so); WORD raw ids (0..16)."""
    import numpy as np
    assert cid in allowed and cid in split()["train"], f"WORD {cid}: not a draw (training) scan"
    m = member("labelsTr", cid)
    r = subprocess.run([SEVEN_Z, "e", "-so", f"-p{PASSWORD}", str(ARCHIVE), m], capture_output=True)
    assert r.returncode == 0 and r.stdout, r.stderr[-500:]
    im = nib_from_gz_bytes(r.stdout)
    a = np.asanyarray(im.dataobj)
    ai = np.rint(a)
    assert np.array_equal(ai, a) and ai.min() >= 0 and ai.max() <= 16, (cid, float(a.min()), float(a.max()))
    log(f"[gt] WORD labelsTr {cid} read (training scan)")
    return ai.astype(np.uint8), im


def reference(cid, marker):
    """The TEST label of scan cid (labelsTs zip member, read into memory) in the analysis label space, + spacing. Only
    inside the look (GT_OPENED marker)."""
    import zipfile
    import numpy as np
    F.require_marker(marker)
    assert cid in split()["test"], cid
    with zipfile.ZipFile(LABELS_TS_ZIP) as z:
        b = z.read(f"labelsTs/word_{cid}.nii.gz")
    im = nib_from_gz_bytes(b)
    a = np.rint(np.asanyarray(im.dataobj))
    assert a.min() >= 0 and a.max() <= 16, cid
    G = analysis_lut()[a.astype(np.uint8)]
    img = __import__("nibabel").load(str(image_path(cid)))
    assert G.shape == img.shape[:3] and np.allclose(im.affine, img.affine, atol=1e-3), f"WORD {cid}: label grid != image grid"
    return G, tuple(float(s) for s in im.header.get_zooms()[:3])


# ---------------------------------------------------------------------------------------------------- cohort metadata
COHORT_METADATA = {   # plan C record (request of 2026-10-22: WORD's scanner / site metadata if available)
    "source": "Luo et al., 'WORD: A large scale dataset, benchmark and clinical applicable study for abdominal organ segmentation "
              "from CT image', Medical Image Analysis 82:102642, 2022 (arXiv 2111.02403; PDF read 2026-10-22, section 3.1 and "
              "the introduction)",
    "quotes": ["The 150 CT scans in the WORD dataset were collected from 150 patients before the radiation therapy in a single "
               "center.",
               "All of them are scanned by a SIEMENS CT scanner without appearance enhancement.",
               "Each CT volume consists of 159 to 330 slices of 512 x 512 pixels, with an in-plane resolution of 0.976 mm x 0.976 mm "
               "and slice spacing of 2.5 mm to 3.0 mm",
               "WORD dataset was collected from a radiation therapy center and annotated by one senior oncologist (with 7 years of "
               "experience) and then checked, discussed and refined by an experts (more than 20 years of experience). All of "
               "images were scanned before the radiotherapy treatment, without any appearance enhancement, with a similar scan "
               "location and with a similar image spacing"],
    "site": "one radiotherapy centre (not named in the paper's dataset section; the authors' affiliations include the radiation "
            "oncology departments of Sichuan Cancer Hospital and West China Hospital, Chengdu)",
    "scanner": "one manufacturer (SIEMENS); the model is not stated",
    "contrast": "non-contrast ('without appearance enhancement') radiotherapy-planning CT",
    "annotation": "radiation-oncology contouring (one senior oncologist, checked by an expert), from scratch",
    "per_scan_metadata_in_release": "none: NIfTI only (no DICOM tags); readme.txt / dataset.json carry no scanner or site field; "
                                    "word_prep 'headers' records each image's geometry and its NIfTI text fields",
    "consequence": "no within-protocol scanner or site split exists in WORD: an alignment gain on WORD mixes the convention part "
                   "(radiotherapy contouring) and the domain part (non-contrast SIEMENS images of one centre); plan E (E4) splits "
                   "the two on AMOS22 / MACT only. The test scans' slice spacing (2.5 vs 3.0 mm) is the only per-scan acquisition "
                   "variable: a descriptive stratum of the look",
}


# ---------------------------------------------------------------------------------------------------- LUTs
def analysis_lut():
    import numpy as np
    lut = np.zeros(256, np.uint8)
    for w, a in WORD_TO_ANALYSIS.items():
        lut[w] = a
    return lut


def ts_lut():
    """TS 'total' id -> analysis id (scoring: adrenal glands merged, cysts -> kidney, colon -> colon_rectum)."""
    import numpy as np
    from totalsegmentator.map_to_binary import class_map
    inv = {v: k for k, v in class_map["total"].items()}
    lut = np.zeros(256, np.uint8)
    for o, names in TS_NAMES.items():
        for n in names:
            lut[inv[n]] = o
    return lut


def vista_lut():
    import numpy as np
    d = json.load(open(F.REPO / "models/vista3d/docs/labels.json"))
    lut = np.zeros(256, np.uint8)
    for o, names in VISTA_NAMES.items():
        for n in names:
            lut[int(d[n])] = o
    return lut


def mrseg_names_list():
    d = json.load(open(F.VC / "m2/weights/mrseg_v1.1.0/dataset.json"))
    return {k: int(v) for k, v in d["labels"].items()}


def mrseg_lut():
    import numpy as np
    d = mrseg_names_list()
    lut = np.zeros(256, np.uint8)
    for o, names in MRSEG_NAMES.items():
        for n in names:
            lut[d[n]] = o
    return lut


def train_lut(model):
    """model class id -> training id (analysis ids 1..13, right adrenal = 17): used to see whether the model already calls a
    GT organ ('same' rule of the composite) and to identify the model's own dataset-organ predictions."""
    import numpy as np
    lut = np.zeros(256, np.uint8)
    if model == "ts291":
        import tsft_full_common as Q
        pc = {n: j for j, n in Q.part_classes(291).items()}
        names = {o: n for o, n in TS_NAMES.items() if o in ADAPTED}
        for o, ns in names.items():
            for n in ns:
                if n in pc:
                    lut[pc[n]] = ADRENAL_R_TMP if n == "adrenal_gland_right" else o
        return lut
    if model == "vista":
        d = {k: int(v) for k, v in json.load(open(F.REPO / "models/vista3d/docs/labels.json")).items()}
        names = VISTA_NAMES
        right = "right adrenal gland"
    else:
        d = mrseg_names_list()
        names = MRSEG_NAMES
        right = "right_adrenal_gland"
    for o, ns in names.items():
        if o not in ADAPTED:
            continue
        for n in ns:
            lut[d[n]] = ADRENAL_R_TMP if n == right else o
    return lut


def canon(model):
    """training id (1..13, 17) -> the model's canonical class id."""
    import numpy as np
    c = np.zeros(ADRENAL_R_TMP + 1, np.uint8)
    if model == "ts291":
        import tsft_full_common as Q
        pc = {n: j for j, n in Q.part_classes(291).items()}
        for o, n in TS_CANON.items():
            c[o] = pc[n]
    elif model == "vista":
        d = {k: int(v) for k, v in json.load(open(F.REPO / "models/vista3d/docs/labels.json")).items()}
        for o, n in VISTA_CANON.items():
            c[o] = d[n]
    else:
        d = mrseg_names_list()
        for o, n in MRSEG_CANON.items():
            c[o] = d[n]
    return c


# ---------------------------------------------------------------------------------------------------- training labels
def adapted_gt(G_analysis):
    """Analysis-id GT -> the adapted organs only (femur heads -> 0: not adapted, the model keeps its own femur)."""
    import numpy as np
    G = np.asarray(G_analysis).astype(np.uint8).copy()
    G[(G == 14) | (G == 15)] = 0
    return G


def split_adrenal(G, affine, P_left, P_right, body=None):
    """WORD's single adrenal class -> left (12) / right (17) gland, per 26-connected component: the side of the model's own
    adrenal prediction it overlaps most; without overlap, the side of its centroid (world RAS x: larger = patient's right)
    relative to the midline = the midpoint of the model's two adrenal centroids, else the median world x of `body` (a mask,
    e.g. CT > -300 HU), else the volume centre. Returns (G', record)."""
    import numpy as np
    from scipy import ndimage
    G = np.asarray(G).astype(np.uint8).copy()
    m = G == ADRENAL
    rec = {"components": []}
    if not m.any():
        return G, rec

    def wx(mask):
        idx = np.argwhere(mask)
        if not len(idx):
            return None
        c = idx.mean(0)
        return float((affine @ np.r_[c, 1.0])[0])

    xl, xr = wx(P_left), wx(P_right)
    if xl is not None and xr is not None:
        mid, how = (xl + xr) / 2, "model adrenal centroids"
    elif body is not None and body.any():
        idx = np.argwhere(body)
        xs = (affine[0, :3] @ idx.T) + affine[0, 3]
        mid, how = float(np.median(xs)), "body median"
    else:
        mid, how = float((affine @ np.r_[(np.array(G.shape) - 1) / 2.0, 1.0])[0]), "volume centre"
    lab, n = ndimage.label(m, structure=np.ones((3, 3, 3), bool))
    for k in range(1, n + 1):
        cm = lab == k
        ol, orr = int((cm & P_left).sum()), int((cm & P_right).sum())
        if ol or orr:
            side, rule = ("left" if ol >= orr else "right"), "model overlap"
        else:
            side, rule = ("right" if wx(cm) > mid else "left"), f"centroid vs midline ({how})"
        if side == "right":
            G[cm] = ADRENAL_R_TMP
        rec["components"].append({"vox": int(cm.sum()), "side": side, "rule": rule, "overlap_left": ol, "overlap_right": orr})
    rec["midline_rule"] = how
    return G, rec


def composite(model, G, P, keep=None):
    """Training label in the model's class ids (== xdata_common.composite_target / vista_common.composite / m2_common.composite
    with the WORD organ set). G: training ids (adapted organs 1..13 with the adrenal split, 0 elsewhere); P: the model's own
    zero-shot prediction (model ids) on the same grid.
      GT organ o, the model already calls o (train_lut[P] == o)  -> P   (e.g. a kidney cyst class)
      GT organ o, otherwise                                      -> canonical model class of o
      GT background, P a class outside the adapted organs        -> P   (the model's own label of every other class)
      GT background, P an adapted-organ class                    -> 0   (a false positive under WORD's protocol)"""
    import numpy as np
    tl, cn = train_lut(model), canon(model)
    G = np.asarray(G).astype(np.uint8)
    P = np.asarray(P).astype(np.uint8)
    assert G.shape == P.shape, (G.shape, P.shape)
    flp = tl[P]
    Y = np.zeros(P.shape, np.uint8)
    bg = G == 0
    keep_non = bg & (P > 0) & (flp == 0)
    Y[keep_non] = P[keep_non]
    org = G > 0
    same = org & (flp == G)
    Y[same] = P[same]
    relab = org & ~same
    Y[relab] = cn[G[relab]]
    cnt = {"n_vox": int(G.size), "gt_organ_vox": int(org.sum()), "gt_organ_relabelled": int(relab.sum()),
           "model_ds_fp_removed": int((bg & (flp > 0)).sum()), "nonds_kept": int(keep_non.sum()),
           "classes": sorted(int(c) for c in np.unique(Y) if c > 0)}
    return Y, cnt


def model_adrenal_masks(model, P):
    """(left gland mask, right gland mask) of a model's own prediction P (model ids)."""
    import numpy as np
    T = train_lut(model)[np.asarray(P).astype(np.uint8)]
    return T == ADRENAL, T == ADRENAL_R_TMP


# ---------------------------------------------------------------------------------------------------- runs (train_final)
RUNS = {f"F1b_{d}": dict(model=f"word_F1b_{d}", recipe="F1b", ts_datasets={291: 820 + i}, stock_dataset=None, ds="word",
                         n_classes=None, labelled=d) for i, d in enumerate(DRAW_SEEDS, start=1)}
DATASETS = {820 + i: dict(name=f"Dataset{820 + i}_FINWORD291_{d}", part=291, draw=d) for i, d in enumerate(DRAW_SEEDS, start=1)}


def recipe_of(run, selection=None):
    return RUNS[run]["recipe"]


def labelled_of(run):
    return draw(RUNS[run]["labelled"])


def dataset_name(did):
    return DATASETS[did]["name"]


def verify_datasets(run, recipe, prep=None):
    p = Path(prep or PREP) / "DONE_datasets.json"
    if not p.is_file():
        raise SystemExit(f"[REFUSE] {F.rel(p)} missing: run word_prep.py first")
    d = json.load(open(p))
    bad = [f for f, h in d["files_sha256"].items() if F.sha256(f) != h]
    if bad:
        raise SystemExit(f"[REFUSE] WORD training dataset files changed since the prep: {bad[:3]}")
    need = dataset_name(RUNS[run]["ts_datasets"][291])
    if need not in d["datasets"]:
        raise SystemExit(f"[REFUSE] {need} missing from the prep record")
    return True


# ---------------------------------------------------------------------------------------------------- VISTA3D / MRSegmentator runs
VISTA_LR = 5e-6                 # the vista block's dev18 selection (outputs/results/vccor/vista/selection/lr_selection.json)
VISTA_RECIPE = "R1"             # the vista block's probe decision (probe/probe.json)
VISTA_ITERS = {"25k": 25000, "5k": 5000}   # 25k = F1b's patch count (50 x 250 it x batch 2 = 25,000 patches of 128^3) = the
                                           # F1b-EQUIVALENT alignment of VISTA3D (primary); 5k = the vista block's registered
                                           # recipe (5,000 patches), a sensitivity arm on draw d1
VF_RUNS = {f"VF_{d}": dict(draw=d, n_iter=VISTA_ITERS["25k"]) for d in DRAW_SEEDS}
VF_RUNS["VF5k_d1"] = dict(draw="d1", n_iter=VISTA_ITERS["5k"])
MODELS3 = {"TS": {"zero": "F0", "aligned": "F1b_{d}", "lut": "ts"}, "VISTA3D": {"zero": "V0", "aligned": "VF_{d}", "lut": "vista"},
           "MRSeg": {"zero": "M0", "aligned": "MF_{d}", "lut": "mrseg"}}
ALIGNMENT = {"TS": "F1b: parts 291 (all WORD organs), all weights, SGD lr 1e-4, 50 x 250 it x batch 2 (25,000 patches)",
             "VISTA3D": "R1 automatic path, AdamW lr 5e-6 (vista block dev18 selection), 25,000 it x batch 1 (25,000 patches)",
             "MRSeg": "all weights, SGD lr 1e-4, 50 x 250 it x batch 2 (25,000 patches; m2 recipe == F1b)"}
MF_RUNS = {f"MF_{d}": dict(draw=d) for d in DRAW_SEEDS}
MF_DATASETS = {f"MF_{d}": (830 + i, f"Dataset{830 + i}_FINWORDMRSEG_{d}") for i, d in enumerate(DRAW_SEEDS, start=1)}

GPU_H = {   # measured on these A5000s (queue_v8.log): TS-full F0 MACT-81 1.25 h; a3 part-291 trainings 1.0-1.05 h; a3 MACT-81
            # predictions 0.73-0.77 h; VISTA V0 50-60 scans 0.1-0.13 h, VF 0.9 h / 5,000 it, VF predict 4-8 min;
            # MRSegmentator M0 MACT-81 2.2 h, MF 1.0-1.4 h, MF predict MACT-81 2.6 h
    "F0__word_test30": 0.5, "word_F1b_train": 1.1, "word_F1b__word_test30": 0.35,
    "vista_V0": 0.15, "vista_VF5k_train": 0.95, "vista_VF25k_train": 4.3, "vista_VF_predict": 0.08,   # 25k = 5 x the measured 5k run
    "mrseg_M0": 1.2, "mrseg_MF_train": 1.1, "mrseg_MF_predict": 1.0,
}
