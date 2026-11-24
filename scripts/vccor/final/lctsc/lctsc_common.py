"""vccor/final LCTSC (decision of 2026-11-13): shared constants and rules.

Data: LCTSC, the AAPM 2017 Lung CT Segmentation Challenge collection on TCIA (Yang et al., Med Phys 45:4568, 2018;
DOI 10.7937/K9/TCIA.2017.3R3FVZ08; CC BY 3.0). 60 thoracic radiotherapy-planning CT patients from 3 institutions (S1-S3),
36 training (LCTSC-Train-S<s>-0nn) and 24 test (LCTSC-Test-S<s>-1nn / -2nn, the challenge's offline and live test sets).
Structures: esophagus, heart, left lung, right lung, spinal cord. Never used by this project before 2026-11-13.

Label rules (fixed before any LCTSC label is read):
  analysis ids  1 esophagus  2 heart  3 lung_left  4 lung_right  5 spinal_cord
  TS 'total' classes -> analysis id (F0 and every TS-based arm):
      esophagus -> 1; heart + atrial_appendage_left -> 2 (the left atrial appendage is part of the heart; RTOG-style heart
      contours include it); lung_upper_lobe_left + lung_lower_lobe_left -> 3; lung_upper_lobe_right +
      lung_middle_lobe_right + lung_lower_lobe_right -> 4; spinal_cord -> 5. Every other TS class -> 0.
  ADAPTED (fine-tuned, primary): 1-4 (TS parts 291 = lobes + esophagus and 293 = heart).
  DESCRIPTIVE (not adapted; the fine-tuned arms keep TS's own part 294): 5 spinal_cord.
  RTSTRUCT ROI names -> analysis id by the normalised name (lower case, letters and digits only), see ROI_NAMES; a patient
  whose RTSTRUCT lacks a structure or has an unknown extra ROI is recorded, never guessed.
  Overlapping contours: the smaller structure wins (priority spinal_cord > esophagus > heart > lungs), counts recorded.

Draws (SITE-MATCHED, 2026-11-13): draws S1, S2, S3, each K = 4 training patients of
ONE institution, fixed in K_lctsc/draws_site.json (seeds S1 1337, S2 2025, S3 3407; rule in that file: RandomState(seed)
order of the site's sorted training ids, already-read patients first) before any new label read. draw(d) reads that file.
SUPERSEDED (kept for the record): the pooled draws d1-d3 of K_lctsc/draws.json (09:52:44), via draw_pooled(d).
No import-time side effects beyond constants; numpy is imported inside functions.
"""
import json
import re
from pathlib import Path

REPO = Path("/home/user/Documents/Medical Imagining SSL")
OUT = REPO / "outputs/results/vccor/final/K_lctsc"
DATA = REPO / "data_lctsc"
ZIPS = DATA / "tcia_zips"
DICOM = DATA / "dicom"
IMAGES = DATA / "images"                 # <pid>.nii.gz, CT (HU), written by lctsc_convert.py
LABELS_TR = DATA / "labels_train_v2"     # <pid>.nii.gz, analysis ids, TRAINING draw patients only (lctsc_raster, v2)
LABELS_TR_V1 = DATA / "labels_train"     # SUPERSEDED v1 (rt-utils even-odd fill; pooled draws), record only
MANIFEST = OUT / "tcia_manifest.json"
DOWNLOAD = OUT / "download.json"
DRAWS = OUT / "draws.json"              # SUPERSEDED pooled draws (record only)
DRAWS_SITE = OUT / "draws_site.json"    # the draws in use (site-matched)
LOGD = REPO / "outputs/logs/vccor"
PREREG_LOG = LOGD / "final_prereg.log"
TAG = "lctsc"

ORGANS = ["esophagus", "heart", "lung_left", "lung_right", "spinal_cord"]
N_ORG = len(ORGANS)
ADAPTED = [1, 2, 3, 4]
PRIMARY = [1, 2, 3, 4]
DESCRIPTIVE = [5]
TS_NAMES = {1: ["esophagus"], 2: ["heart", "atrial_appendage_left"], 3: ["lung_upper_lobe_left", "lung_lower_lobe_left"],
            4: ["lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"], 5: ["spinal_cord"]}
ROI_NAMES = {"esophagus": 1, "oesophagus": 1, "heart": 2, "lungl": 3, "leftlung": 3, "lunglt": 3, "lungr": 4, "rightlung": 4,
             "lungrt": 4, "spinalcord": 5, "cord": 5}
PRIORITY = [5, 1, 2, 3, 4]               # painting order reversed: the first listed wins an overlap
DRAW_SEEDS_POOLED = {"d1": 1337, "d2": 2025, "d3": 3407}   # SUPERSEDED
DRAW_SEEDS = {"S1": 1337, "S2": 2025, "S3": 3407}           # site-matched draws (keys = institutions)
K = 4
PARTS = (291, 293)                       # TS parts fine-tuned (291: lobes + esophagus; 293: heart)


def norm_roi(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


def manifest():
    return json.load(open(MANIFEST))


def patients(kind):
    pre = {"train": "LCTSC-Train-", "test": "LCTSC-Test-"}[kind]
    return sorted({r["PatientID"] for r in manifest()["series"] if r["PatientID"].startswith(pre)})


def is_test(pid):
    return pid.startswith("LCTSC-Test-")


def site(pid):
    return pid.split("-")[2]


def draw_pooled(d):
    """SUPERSEDED pooled draw (draws.json): RandomState(seed).shuffle(sorted training ids)[:4]."""
    import numpy as np
    tr = list(patients("train"))
    np.random.RandomState(DRAW_SEEDS_POOLED[d]).shuffle(tr)
    return tr[:K]


def site_draw_rule(s, already_read):
    """The site-matched rule (draws_site.json): RandomState(seed_s) order of institution s's sorted training ids; the draw is
    the first K of [already-read patients of s, in that order] + [the others, in that order]."""
    import numpy as np
    order = [p for p in patients("train") if site(p) == s]
    np.random.RandomState(DRAW_SEEDS[s]).shuffle(order)
    prior = [p for p in order if p in set(already_read)]
    return (prior + [p for p in order if p not in set(already_read)])[:K]


def draw(d):
    """The site-matched draw d in {S1, S2, S3}, as fixed in draws_site.json."""
    return list(json.load(open(DRAWS_SITE))["draws"][d])


def draw_patients():
    return sorted({p for d in DRAW_SEEDS for p in draw(d)})


def image_path(pid):
    return IMAGES / f"{pid}.nii.gz"


def train_label_path(pid):
    assert not is_test(pid), f"{pid}: test labels are not converted before the look"
    return LABELS_TR / f"{pid}.nii.gz"


def ts_total_lut():
    """TS 'total' class id -> analysis id (uint8 LUT of 256)."""
    import numpy as np
    from totalsegmentator.map_to_binary import class_map
    inv = {v: k for k, v in class_map["total"].items()}
    lut = np.zeros(256, np.uint8)
    for o, names in TS_NAMES.items():
        for n in names:
            lut[inv[n]] = o
    return lut


def part_lut(tid):
    """TS part tid class id -> analysis id."""
    import numpy as np
    from totalsegmentator.map_to_binary import class_map_5_parts, map_taskid_to_partname_ct
    lut = np.zeros(256, np.uint8)
    for j, n in class_map_5_parts[map_taskid_to_partname_ct[tid]].items():
        for o, names in TS_NAMES.items():
            if n in names:
                lut[j] = o
    return lut


def part_class(tid, name):
    from totalsegmentator.map_to_binary import class_map_5_parts, map_taskid_to_partname_ct
    inv = {n: j for j, n in class_map_5_parts[map_taskid_to_partname_ct[tid]].items()}
    return inv[name]


def split_lungs(G, P291):
    """Whole-lung GT (3 left, 4 right) -> TS part-291 lobe ids: a GT lung voxel keeps TS's own lobe of that side where TS
    calls one there; otherwise it takes the lobe of the nearest TS lobe voxel of the same side (Euclidean, voxel grid); a
    side without any TS lobe falls back to its upper lobe. Returns (lobe map in part-291 ids, 0 elsewhere; record)."""
    import numpy as np
    from scipy import ndimage
    side_lobes = {3: [part_class(291, "lung_upper_lobe_left"), part_class(291, "lung_lower_lobe_left")],
                  4: [part_class(291, "lung_upper_lobe_right"), part_class(291, "lung_middle_lobe_right"),
                      part_class(291, "lung_lower_lobe_right")]}
    out = np.zeros(G.shape, np.uint8)
    rec = {}
    for side, lobes in side_lobes.items():
        g = G == side
        if not g.any():
            rec[ORGANS[side - 1]] = {"gt_vox": 0}
            continue
        ts_side = np.isin(P291, lobes)
        own = g & ts_side
        out[own] = P291[own]
        rest = g & ~ts_side
        if rest.any():
            if ts_side.any():
                ind = ndimage.distance_transform_edt(~ts_side, return_distances=False, return_indices=True)
                near = P291[tuple(ind[:, rest])]
                out[rest] = near
                rule = "nearest TS lobe"
            else:
                out[rest] = lobes[0]
                rule = "no TS lobe on this side: upper lobe"
        else:
            rule = "all voxels inside TS lobes"
        rec[ORGANS[side - 1]] = {"gt_vox": int(g.sum()), "own_lobe_vox": int(own.sum()), "reassigned_vox": int(rest.sum()),
                                 "rule": rule}
    return out, rec


def composite(tid, G, P):
    """Training target of TS part tid (part ids) for the adapted structures (== the project's composite rule, WORD
    word_common.composite, with LCTSC's structures):
      GT adapted structure o of this part, TS calls o there    -> P (TS's own class: lobe, atrial appendage)
      GT adapted structure o of this part, otherwise           -> TS's canonical class of o (lungs: split_lungs lobe)
      GT background (for this part's structures), P a class of another structure -> P
      GT background, P a class of this part's adapted structures -> 0 (a false positive under LCTSC's protocol)."""
    import numpy as np
    G = np.asarray(G).astype(np.uint8)
    P = np.asarray(P).astype(np.uint8)
    assert G.shape == P.shape, (G.shape, P.shape)
    pl = part_lut(tid)
    mine = [o for o in ADAPTED if any(pl[j] == o for j in range(256))]
    flp = pl[P]
    flp[~np.isin(flp, mine)] = 0
    Y = np.zeros(P.shape, np.uint8)
    org = np.isin(G, mine)
    bg = ~org
    keep_non = bg & (P > 0) & (flp == 0)
    Y[keep_non] = P[keep_non]
    same = org & (flp == G)
    Y[same] = P[same]
    relab = org & ~same
    rec = {"structures": [ORGANS[o - 1] for o in mine]}
    if tid == 291:
        lob, lrec = split_lungs(np.where(relab, G, 0), P)
        lungs = relab & np.isin(G, [3, 4])
        Y[lungs] = lob[lungs]
        rec["lung_split"] = lrec
        canon = {1: part_class(291, "esophagus")}
    else:
        canon = {2: part_class(293, "heart")}
    for o, c in canon.items():
        m = relab & (G == o)
        Y[m] = c
    rec.update({"gt_vox": int(org.sum()), "gt_relabelled": int(relab.sum()), "ts_fp_removed": int((bg & (flp > 0)).sum()),
                "other_kept": int(keep_non.sum()), "classes": sorted(int(c) for c in np.unique(Y) if c > 0)})
    assert not (org & (Y == 0)).any(), "an adapted GT voxel got background"
    return Y, rec
