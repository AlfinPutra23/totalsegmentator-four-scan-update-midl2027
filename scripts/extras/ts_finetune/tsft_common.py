"""Shared paths and conventions for the PRACTITIONER BASELINE "fine-tune TotalSegmentator on the 4 labels" (component ts_finetune).

What this baseline is
---------------------
Our teacher is TotalSegmentator v2 'total' run with --ml --fast, i.e. the 3 mm nnU-Net Dataset297
(nnUNetTrainer_4000epochs_NoMirroring, PlainConvUNet, patch 112x112x128, 118 classes). The baseline fine-tunes THAT
network on the SAME 4 labelled FLARE volumes each 4-label student saw (per seed), then runs it DIRECTLY on the 10 FLARE
test volumes through TotalSegmentator's own inference pipeline (canonical -> 3 mm -> nnU-Net -> back), maps the result to
the flare_h5 frame exactly like the TS-direct number (84.08) and scores it with the students' metric code.

Frames
------
* nnU-Net frame: closest-canonical orientation, 3 mm isotropic (what TotalSegmentator feeds Dataset297 at inference).
* raw frame:     the FLARE22 NIfTI (data/FLARE22Train), what TotalSegmentator returns.
* h5 frame:      scripts/preprocess_flare.py = scipy zoom to 1.5 mm (order 0 for labels) + GT-foreground bbox crop
                 (margin 10). h5_frame() re-derives it and asserts the stored flare_h5 label is reproduced bit-exactly.

Labelled subset per seed
------------------------
train_main_flare_octree.py seeds numpy at import (np.random.seed(args.seed), --deterministic 1 default) and the first
numpy call afterwards is np.random.shuffle(train_list) on the first 40 entries of flare.txt; labelled = shuffled[:labelnum].
train_order() replays that with a private RandomState (identical stream).

Import this module BEFORE nnunetv2/totalsegmentator and call set_nnunet_env() first: nnunetv2.paths reads the env at import.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True   # importing repo modules (build_flare_pseudo, add_pseudo_confidence, test_util) must not write caches there

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "scripts"
RAW_IMG = ROOT / "data/FLARE22Train/images/FLARE22_Tr_{cid}_0000.nii.gz"
RAW_LBL = ROOT / "data/FLARE22Train/labels/FLARE22_Tr_{cid}.nii.gz"
H5_DIR = ROOT / "MagicNet/data/flare_h5"
DATA_DIR = ROOT / "MagicNet/data"
SPLIT = ROOT / "MagicNet/data/flare.txt"
MAGICNET_CODE = ROOT / "MagicNet/code"
MODEL_DIR = ROOT / "MagicNet/model"

NN_ROOT = ROOT / "nnunet_extras"                       # nnU-Net raw + preprocessed roots (new, component-owned)
NN_RAW = NN_ROOT / "nnUNet_raw"
NN_PRE = NN_ROOT / "nnUNet_preprocessed"
PRETRAINED = NN_ROOT / "pretrained" / "Dataset297_to_FLARE14_headmapped.pth"
OUT = ROOT / "outputs/results/extras/ts_finetune"      # every job output (models, predictions, scores, logs)
NN_RES = OUT / "nnUNet_results"
FRAME_CACHE = OUT / "h5_frames"
PSEUDO_ROOT = ROOT / "MagicNet/data/extras_ts_finetune"  # adapted-teacher pseudo-label dirs (h5 frame)

TS297_NAME = "Dataset297_TotalSegmentator_total_3mm_1559subj"
TS297_MODEL = Path.home() / ".totalsegmentator/nnunet/results" / TS297_NAME / "nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres"
TS297_FOLDS = [0]

SEEDS = (1337, 2025, 3407)
B = 4                                                   # labelled volumes per student
DATASET_ID = {1337: 971, 2025: 972, 3407: 973}
PLANS_ID = "TS297Plans"                                 # Dataset297 plans moved onto our datasets (provenance in the name)
CONFIG = "3d_fullres"
FOLD = "all"                                            # train on all 4 labelled volumes (no internal hold-out)
TRAINER = "nnUNetTrainer_250epochs_NoMirroring"         # STOCK nnU-Net trainer: 250 x 250 it, SGD lr 1e-2 poly, no mirroring
TS_SPACING = 3.0
H5_SPACING = 1.5
H5_MARGIN = 10

# FLARE22 native ids (liver-first). Names are TotalSegmentator's class names so the head mapping is by name.
FLARE_NAMES = ["background", "liver", "kidney_right", "spleen", "pancreas", "aorta", "inferior_vena_cava",
               "adrenal_gland_right", "adrenal_gland_left", "gallbladder", "esophagus", "stomach", "duodenum", "kidney_left"]
ORG = ["liver", "r.kid", "spleen", "panc", "aorta", "ivc", "r.adr", "l.adr", "gall", "esoph", "stom", "duod", "l.kid"]
# Tail organs as FLARE ids, identical to scripts/v3_common.py (review fix: the earlier [4,7,8,9,10,12] matched neither):
TAIL_T2 = [4, 11, 9, 8, 12, 7, 10]                      # TAIL_T2 = panc,stom,gall,l.adr,duod,r.adr,esoph (main Table 2)
TAIL_T5 = [4, 11, 9, 8, 12, 7]                          # TAIL_T5 = same without esophagus (Table 5 / sec 4.7)
TAIL = TAIL_T2
# How the 4 training label maps were brought to the 3 mm grid (recorded in tsft_manifest.json; a different tag => rebuild).
# Review fix: nnU-Net's seg resampler (skimage resize, centre-aligned grid) put the labels up to ~0.75 input voxels per
# edge off the image grid that TotalSegmentator's change_spacing (scipy.ndimage.zoom, corner-aligned) produces and maps back
# from at inference. Now: one-hot -> the SAME change_spacing call as the image (order 1) -> argmax.
LABEL_RESAMPLER = "ts_change_spacing_onehot_linear_argmax"


def dataset_name(seed):
    return f"Dataset{DATASET_ID[seed]}_FLARE22b{B}s{seed}"


def model_folder(seed, results_root=None):
    return Path(results_root or NN_RES) / dataset_name(seed) / f"{TRAINER}__{PLANS_ID}__{CONFIG}"


def pseudo_dir(seed):
    return PSEUDO_ROOT / f"flare_tsft_s{seed}"


def set_nnunet_env(results_root=None):
    """Point nnU-Net at the component-owned roots. Must run before nnunetv2 is imported anywhere."""
    if "nnunetv2" in sys.modules:
        raise RuntimeError("set_nnunet_env() must be called before importing nnunetv2/totalsegmentator")
    os.environ["nnUNet_raw"] = str(NN_RAW)
    os.environ["nnUNet_preprocessed"] = str(NN_PRE)
    os.environ["nnUNet_results"] = str(results_root or NN_RES)


# ----------------------------------------------------------------------------------------------------- splits / seeds
def split_ids():
    """Exactly the trainer's parsing of flare.txt -> (40 train-pool ids, 10 test ids)."""
    with open(SPLIT) as f:
        total = ["{}".format(item.strip().split('.')[0][-4:]) for item in f.readlines()]
    assert len(total) == 50, len(total)
    return total[:40], total[40:]


def train_order(seed):
    tr, _ = split_ids()
    np.random.RandomState(seed).shuffle(tr)             # == np.random.seed(seed); np.random.shuffle(tr) in the trainer
    return tr


def labeled_ids(seed, b=B):
    return train_order(seed)[:b]


def unlabeled_ids(seed, b=B):
    return train_order(seed)[b:]


def test_ids():
    return split_ids()[1]


# ----------------------------------------------------------------------------------------------- label maps / head map
def ts_label_dict():
    return json.load(open(TS297_MODEL / "dataset.json"))["labels"]


def flare_to_ts_channels():
    """FLARE channel c (0..13) -> Dataset297 output channel, by class name; cross-checked against build_flare_pseudo."""
    ts = ts_label_dict()
    chans = [ts[n] for n in FLARE_NAMES]
    sys.path.insert(0, str(SCRIPTS))
    import build_flare_pseudo as v1                      # single source of truth for the teacher's TS->FLARE mapping
    assert {t: f for f, t in enumerate(chans) if f > 0} == v1.TS_TO_FLARE, "head map disagrees with build_flare_pseudo.TS_TO_FLARE"
    return chans


def ts_to_flare_lut(version="v1"):
    """uint8 LUT mapping Dataset297 ids -> FLARE ids. v1 = flare_pseudo (cysts dropped), v2 = flare_pseudo_v2 (23->13, 24->2)."""
    sys.path.insert(0, str(SCRIPTS))
    import build_flare_pseudo as v1
    m = dict(v1.TS_TO_FLARE)
    if version == "v2":
        m.update({23: 13, 24: 2})
    lut = np.zeros(256, np.uint8)
    for t, f in m.items():
        lut[t] = f
    return lut


# ------------------------------------------------------------------------------------------------------- h5 frame
def voxel_spacing(aff):
    return np.sqrt((aff[:3, :3] ** 2).sum(0))


def _zoom_label(arr, spacing):
    from scipy.ndimage import zoom
    return zoom(arr, np.asarray(spacing, float) / H5_SPACING, order=0, mode="nearest")


def h5_frame(cid):
    """Recover the flare_h5 frame of case cid (spacing, crop slices) and verify it reproduces the stored h5 label exactly.
    Cached per case in FRAME_CACHE/<cid>.json (atomic write; safe for concurrent jobs)."""
    cache = FRAME_CACHE / f"{cid}.json"
    if cache.exists():
        d = json.load(open(cache))
        return {"spacing": np.array(d["spacing"]), "slices": tuple(slice(a, b) for a, b in d["slices"]),
                "shape": tuple(d["shape"]), "raw_shape": tuple(d["raw_shape"])}
    import h5py
    import nibabel as nib
    img = nib.load(str(RAW_IMG).format(cid=cid))
    lbl = nib.load(str(RAW_LBL).format(cid=cid))
    assert img.shape == lbl.shape, (cid, img.shape, lbl.shape)
    sp = voxel_spacing(img.affine)                        # preprocess_flare.py uses the IMAGE affine for both
    lab = _zoom_label(np.asarray(lbl.dataobj).astype(np.uint8), sp)
    fg = np.argwhere(lab > 0)
    mn = np.maximum(fg.min(0) - H5_MARGIN, 0)
    mx = np.minimum(fg.max(0) + H5_MARGIN + 1, lab.shape)
    sl = tuple(slice(int(a), int(b)) for a, b in zip(mn, mx))
    with h5py.File(H5_DIR / f"{cid}.h5", "r") as f:
        gt = f["label"][:]
    assert gt.shape == lab[sl].shape and np.array_equal(gt, lab[sl]), f"{cid}: h5 frame not reproduced"
    FRAME_CACHE.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(f".tmp{os.getpid()}")
    json.dump({"spacing": sp.tolist(), "slices": [[s.start, s.stop] for s in sl], "shape": list(gt.shape),
               "raw_shape": list(img.shape)}, open(tmp, "w"))
    os.replace(tmp, cache)
    return {"spacing": sp, "slices": sl, "shape": gt.shape, "raw_shape": img.shape}


def raw_to_h5(seg_raw, cid):
    """Raw-frame label map (FLARE ids) -> flare_h5 frame, same zoom/crop as preprocess_flare.py / build_flare_pseudo.py."""
    fr = h5_frame(cid)
    assert tuple(seg_raw.shape) == tuple(fr["raw_shape"]), (cid, seg_raw.shape, fr["raw_shape"])
    out = _zoom_label(seg_raw.astype(np.uint8), fr["spacing"])[fr["slices"]]
    assert out.shape == tuple(fr["shape"]), (cid, out.shape, fr["shape"])
    return out


# ------------------------------------------------------------------------------- TotalSegmentator-identical inference
def ts_predict_raw(img_path, model_dir, folds, device="cuda", cpu_threads=None):
    """Run TotalSegmentator's OWN single-model pipeline (totalsegmentator.nnunet.nnUNet_predict_image with the exact
    arguments python_api uses for `--ml --fast`: canonical -> change_spacing 3 mm (order 3) -> nnU-Net (step 0.5,
    gaussian, no TTA, checkpoint_final.pth) -> change_spacing back (order 0) -> undo canonical), but with the nnU-Net
    model folder swapped for `model_dir`. Returns the raw-frame uint8 label map in the model's own label ids."""
    import multiprocessing
    import totalsegmentator.nnunet as tsn
    tsn.get_output_folder = lambda *a, **k: str(model_dir)   # the ONLY change vs TotalSegmentator: which weights
    if device == "cpu" and cpu_threads:                        # TS sets torch threads = cpu_count() on CPU; cap it
        multiprocessing.cpu_count = lambda: int(cpu_threads)
    seg, _, _ = tsn.nnUNet_predict_image(
        Path(img_path), None, -1, model=CONFIG, folds=folds, trainer="(model_dir)", tta=False,
        multilabel_image=True, resample=TS_SPACING, crop=None, crop_path=None, task_name="total",
        nora_tag="None", preview=False, nr_threads_resampling=1, nr_threads_saving=6, force_split=False,
        crop_addon=[3, 3, 3], roi_subset=None, output_type="nifti", statistics=False, quiet=True,
        verbose=False, test=0, skip_saving=False, device=device)
    return np.asarray(seg.dataobj).astype(np.uint8)


def write_pseudo_h5(path, pseudo, attrs=None):
    """Same layout as flare_pseudo / flare_pseudo_v2: 'pseudo' uint8, 'conf' float32 (boundary confidence),
    'protocol_mask' uint8 zeros. Atomic (tmp + rename) so a killed job never leaves a half-written file."""
    import h5py
    sys.path.insert(0, str(SCRIPTS))
    from add_pseudo_confidence import confidence
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp{os.getpid()}.h5")
    with h5py.File(tmp, "w") as f:
        f.create_dataset("pseudo", data=pseudo.astype(np.uint8), compression="gzip")
        f.create_dataset("conf", data=confidence(pseudo.astype(np.uint8)), compression="gzip")
        f.create_dataset("protocol_mask", data=np.zeros(pseudo.shape, np.uint8), compression="gzip")
        for k, v in (attrs or {}).items():
            f.attrs[k] = v
    os.replace(tmp, path)


def pseudo_ok(path, cid, ckpt_sha=None):
    """True if path is a complete pseudo h5 in the h5 frame of cid (used to make prediction jobs idempotent).
    With ckpt_sha, the file must also have been written from that checkpoint (a retrained model never reuses stale files)."""
    import h5py
    try:
        with h5py.File(path, "r") as f:
            ok = all(k in f for k in ("pseudo", "conf", "protocol_mask"))
            shp = f["pseudo"].shape if ok else None
            sha = f.attrs.get("checkpoint_sha256_16")
        if ckpt_sha is not None and (sha.decode() if isinstance(sha, bytes) else sha) != ckpt_sha:
            return False
        return ok and tuple(shp) == tuple(h5_frame(cid)["shape"])
    except (OSError, KeyError):
        return False
