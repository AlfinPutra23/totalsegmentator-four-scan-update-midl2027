"""vccor/tsft_full: the "why not just fine-tune TotalSegmentator on your 4 scans?" baseline, on TS-FULL (1.5 mm).

TS-full = TotalSegmentator 2.13.0 (.venv), task 'total', --ml, no --fast = five nnU-Net 3d_fullres models at 1.5 mm
(Dataset291 organs, 292 vertebrae, 293 cardiac/vessels, 294 muscles, 295 ribs) merged in that order (a later model's
non-background label overwrites an earlier one; totalsegmentator/nnunet.py nnUNet_predict_image).
FLARE's 13 organs live in two of them: 291 (liver, kidneys, spleen, pancreas, adrenals, gallbladder, oesophagus,
stomach, duodenum; + kidney_cyst_left/right, mapped to the kidneys by the v2 LUT) and 293 (aorta, IVC). Only these two
are fine-tuned; 292/294/295 stay TotalSegmentator's own weights.

Variants (all: seed-1337 labelled scans 0014/0001/0006/0038, nnU-Net 2.7.0 nnUNetTrainerNoMirroring = TS's own trainer,
TS init incl. heads, TS plans, 50 epochs x 250 it, poly lr, fold 'all', checkpoint_final):
  F1a  all weights,           lr 1e-3   (F1 primary lr)
  F1b  all weights,           lr 1e-4   (F1 lr sensitivity; F1 = the dev18-better of F1a/F1b)
  F2   decoder only (encoder frozen), lr 1e-3
  F3   normalisation affines only (InstanceNorm weight+bias, heads frozen), lr 1e-2
  F0   no fine-tuning: TS-full's own weights through the same (GPU) inference path = numerics control
Training labels (per sub-model m, on TS's own 1.5 mm canonical grid): FLARE GT for the FLARE organs m owns; 0 where GT
is a FLARE organ owned by the other fine-tuned model; TS-m's own prediction for m's non-FLARE classes where GT is
background; 0 where TS-m predicted a FLARE class on GT background (see composite_target()).

This module has NO import-time side effects beyond sys.path (safe before nnunetv2/torch; GPU jobs import it).
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path("/home/user/Documents/Medical Imagining SSL")
HERE = Path(__file__).resolve().parent
OUT = REPO / "outputs/results/vccor/tsft_full"
LOGD = REPO / "outputs/logs/vccor"
PREREG = OUT / "prereg.json"
PREREG_LOG = LOGD / "tsft_full_prereg.log"

# ---------------------------------------------------------------------------------------------------- layout
NN_ROOT = OUT / "nnunet"
NN_RAW = NN_ROOT / "nnUNet_raw"
NN_PRE = NN_ROOT / "nnUNet_preprocessed"
PREP = OUT / "prep"                          # labelled-4 TS capture, GT on the 1.5 mm grid, targets, manifest
PREP_DONE = PREP / "DONE.json"
MODELS = OUT / "models"                      # TS-compatible inference folders per variant / part
PRED = OUT / "pred"                          # pred/<variant>/{raw,h5}/<cid>, pred/<variant>/DONE.json
CACHE = PRED / "cache_frozen"                # frozen-part (292/294/295) outputs captured by the F0 run, per scan
MAPS = OUT / "maps"                          # label maps for scoring (no GT): maps/<set>/<cid>.npz + manifest
RES = OUT / "results"                        # pervol/, results.json, tables.md, GT_OPENED

TS_RESULTS = Path.home() / ".totalsegmentator/nnunet/results"
TS_TRAINER = "nnUNetTrainerNoMirroring"
CONFIG = "3d_fullres"
TS_DS = {291: "Dataset291_TotalSegmentator_part1_organs_1559subj", 292: "Dataset292_TotalSegmentator_part2_vertebrae_1532subj",
         293: "Dataset293_TotalSegmentator_part3_cardiac_1559subj", 294: "Dataset294_TotalSegmentator_part4_muscles_1559subj",
         295: "Dataset295_TotalSegmentator_part5_ribs_1559subj"}
TS_MODEL = {t: TS_RESULTS / d / f"{TS_TRAINER}__nnUNetPlans__{CONFIG}" for t, d in TS_DS.items()}
PARTS_ALL = (291, 292, 293, 294, 295)        # TS merge order
PARTS_FT = (291, 293)
PARTS_FROZEN = (292, 294, 295)
DATASET_ID = {291: 991, 293: 993}
DATASET_NAME = {291: "Dataset991_TSFT291b4s1337", 293: "Dataset993_TSFT293b4s1337"}
PLANS_ID = {291: "TS291Plans", 293: "TS293Plans"}
FOLD = "all"

# ---------------------------------------------------------------------------------------------------- the design
SEED = 1337
LAB = ["0014", "0001", "0006", "0038"]       # seed-1337 labelled 4 (trainer split replay; == VC_COR fit scans)
DEV18 = ["0002", "0003", "0004", "0005", "0007", "0008", "0009", "0010", "0011", "0012", "0013", "0015", "0016", "0017",
         "0018", "0019", "0020", "0021"]
HELD18 = ["0022", "0023", "0024", "0025", "0026", "0027", "0028", "0029", "0030", "0031", "0032", "0033", "0034", "0035",
          "0036", "0037", "0039", "0040"]
TEST10 = ["0041", "0042", "0043", "0044", "0045", "0046", "0047", "0048", "0049", "0050"]
EVAL = DEV18 + HELD18 + TEST10
SETS = {"dev18": DEV18, "held18": HELD18, "test10": TEST10}
NAMES = ['liver', 'r.kid', 'spleen', 'panc', 'aorta', 'ivc', 'r.adr', 'l.adr', 'gall', 'esoph', 'stom', 'duod', 'l.kid']
# FLARE id -> TotalSegmentator class name (== tsft_common.FLARE_NAMES; liver-first FLARE22 order)
FLARE_TS_NAMES = ["background", "liver", "kidney_right", "spleen", "pancreas", "aorta", "inferior_vena_cava",
                  "adrenal_gland_right", "adrenal_gland_left", "gallbladder", "esophagus", "stomach", "duodenum",
                  "kidney_left"]

EPOCHS = 50
ITERS_PER_EPOCH = 250
VAL_ITERS_PER_EPOCH = 10                     # monitoring only (fold 'all': validation patches come from the 4 train scans)
SAVE_EVERY = 5                               # checkpoint_latest every 5 epochs (resume granularity; training unchanged)
N_PROC_DA = 8
VARIANTS = {
    "F1a": dict(trainable="all", lr=1e-3, label="F1 full fine-tune, lr 1e-3 (primary F1 lr)"),
    "F1b": dict(trainable="all", lr=1e-4, label="F1 full fine-tune, lr 1e-4 (F1 lr sensitivity)"),
    "F2": dict(trainable="decoder", lr=1e-3, label="F2 decoder-only fine-tune (encoder frozen), lr 1e-3"),
    "F3": dict(trainable="norm", lr=1e-2, label="F3 normalisation-only fine-tune (InstanceNorm affines), lr 1e-2"),
}
FT_VARIANTS = list(VARIANTS)
HOSTS = ["F0"] + FT_VARIANTS                 # every host that VC_COR / COR_AND_TRC are applied on top of
F1_PAIR = ("F1a", "F1b")                     # F1 lr chosen on dev18: argmax dev18 macro Dice, tie -> F1a
BEST_CANDIDATES_AFTER_F1 = ("F2", "F3")      # "best" = argmax held18 macro over {F1(selected), F2, F3} (oracle)
STEP_SIZE_TS = 0.8                           # TS uses tile_step_size 0.8 for 'total' at 1.5 mm (nnunet.py)


def h5_spacing_frame_note():
    return "h5 frame = scripts/extras/ts_finetune/tsft_common.raw_to_h5 (1.5 mm zoom order 0 + flare_h5 GT-bbox crop)"


# ---------------------------------------------------------------------------------------------------- helpers
def sha256(path, n=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(n), b""):
            h.update(b)
    return h.hexdigest()


def rel(p):
    p = Path(p)
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    json.dump(obj, open(tmp, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(tmp, path)


class Log:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(self.path, "a") as f:
            f.write(line + "\n")


def be_nice(target=15):
    cur = os.nice(0)
    if cur < target:
        os.nice(target - cur)


def cap_threads(n):
    """CPU thread cap for this process and its children (call before numpy/torch import)."""
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = str(n)


def set_nnunet_env(results_root):
    """Component-owned nnU-Net roots. Must run before nnunetv2 is imported anywhere in the process."""
    if "nnunetv2" in sys.modules:
        raise RuntimeError("set_nnunet_env() must be called before importing nnunetv2")
    os.environ["nnUNet_raw"] = str(NN_RAW)
    os.environ["nnUNet_preprocessed"] = str(NN_PRE)
    os.environ["nnUNet_results"] = str(results_root)


def results_root(variant):
    return NN_ROOT / f"results_{variant}"


def train_fold_dir(variant, tid):
    return results_root(variant) / DATASET_NAME[tid] / f"{TS_TRAINER}__{PLANS_ID[tid]}__{CONFIG}" / f"fold_{FOLD}"


def model_dir(variant, tid):
    """TS-compatible inference folder: TS's own plans.json + dataset.json, fold_0/checkpoint_final.pth = fine-tuned."""
    return MODELS / variant / f"part{tid}"


def model_done(variant):
    return MODELS / variant / "DONE.json"


def pred_dir(variant):
    return PRED / variant


def pred_done(variant):
    return PRED / variant / "DONE.json"


def ts_dataset_json(tid):
    return json.load(open(TS_MODEL[tid] / "dataset.json"))


def part_classes(tid):
    """part class id -> TotalSegmentator class name (== totalsegmentator.map_to_binary.class_map_5_parts)."""
    return {int(v): k for k, v in ts_dataset_json(tid)["labels"].items() if int(v) > 0}


def part_flare_lut(tid):
    """uint8 LUT part class id -> FLARE id, through TS's 'total' ids and the repo's v2 LUT (cysts 23->13, 24->2),
    i.e. exactly what TS-full's merged output undergoes in teacher_common.map_seg. Cross-checked by name."""
    import numpy as np
    sys.path.insert(0, str(REPO / "scripts/extras/ts_finetune"))
    import tsft_common as C
    from totalsegmentator.map_to_binary import class_map, class_map_5_parts, map_taskid_to_partname_ct
    inv = {v: k for k, v in class_map["total"].items()}
    pmap = class_map_5_parts[map_taskid_to_partname_ct[tid]]
    assert pmap == part_classes(tid), f"TS class_map_5_parts != dataset.json labels for {tid}"
    v2 = C.ts_to_flare_lut("v2")
    lut = np.zeros(256, np.uint8)
    for j, name in pmap.items():
        lut[j] = v2[inv[name]]
    # name cross-check: canonical organ classes map to their own FLARE id; cysts to the same-side kidney
    for o in range(1, 14):
        for j, name in pmap.items():
            if name == FLARE_TS_NAMES[o]:
                assert lut[j] == o, (tid, name, lut[j], o)
    if tid == 291:
        names = {n: j for j, n in pmap.items()}
        assert lut[names["kidney_cyst_left"]] == 13 and lut[names["kidney_cyst_right"]] == 2
    return lut


def owned_organs(tid):
    """FLARE ids whose canonical TS class lives in part tid, and the canonical part class of each."""
    names = {n: j for j, n in part_classes(tid).items()}
    return {o: names[FLARE_TS_NAMES[o]] for o in range(1, 14) if FLARE_TS_NAMES[o] in names}


def composite_target(G, P, tid, lut=None):
    """Training label for sub-model tid on TS's 1.5 mm canonical grid.
    G: FLARE GT (0..13) on that grid; P: TS-tid's own prediction (part class ids) on that grid.
      GT organ o owned by tid      -> P if P already maps to o (e.g. a kidney_cyst of the same side), else canon_tid(o)
      GT organ owned by other part -> 0   (so tid neither competes with nor, in TS's merge order, overwrites it)
      GT background, P non-FLARE   -> P   (keeps TS's knowledge of the classes FLARE does not annotate)
      GT background, P FLARE class -> 0   (a TS false positive under FLARE's protocol)
      GT background, P == 0        -> 0
    Returns (Y uint8, counts dict)."""
    import numpy as np
    lut = part_flare_lut(tid) if lut is None else lut
    own = owned_organs(tid)
    canon = np.zeros(14, np.uint8)
    owned = np.zeros(14, bool)
    for o, j in own.items():
        canon[o], owned[o] = j, True
    G = G.astype(np.uint8)
    P = P.astype(np.uint8)
    flp = lut[P]
    Y = np.zeros(P.shape, np.uint8)
    bg = G == 0
    keep_nonflare = bg & (P > 0) & (flp == 0)
    Y[keep_nonflare] = P[keep_nonflare]
    ow = owned[G]
    same = ow & (flp == G)
    Y[same] = P[same]
    relab = ow & ~same
    Y[relab] = canon[G[relab]]
    other = (G > 0) & ~ow
    cnt = {"n_vox": int(G.size), "unchanged": int((Y == P).sum()),
           "gt_owned_relabelled": int(relab.sum()),
           "gt_owned_relabelled_from_nonflare": int((relab & (P > 0) & (flp == 0)).sum()),
           "gt_other_part_zeroed": int((other & (P > 0)).sum()),
           "ts_flare_fp_removed": int((bg & (flp > 0)).sum()),
           "nonflare_kept": int(keep_nonflare.sum()),
           "cyst_label_kept": int((same & (P != canon[G])).sum())}
    return Y, cnt


def gt_to_ts_grid(lab_c, img_in, img_rsp, spacing=1.5, n_labels=14):
    """Canonical native-grid GT -> TS's 1.5 mm grid with TotalSegmentator's own change_spacing geometry (identical zoom
    factors / shape / affine as the image call): one-hot channel -> order-1 zoom -> argmax (strict '>' = lowest id wins
    ties, as np.argmax). Same resampler as scripts/extras/ts_finetune/prepare_dataset.label_to_ts_grid (LABEL_RESAMPLER
    'ts_change_spacing_onehot_linear_argmax'), at 1.5 mm instead of 3 mm."""
    import nibabel as nib
    import numpy as np
    from totalsegmentator.resampling import change_spacing
    assert np.allclose(lab_c.affine, img_in.affine, atol=1e-4) and lab_c.shape == img_in.shape
    data = np.asarray(lab_c.dataobj)
    best = arg = None
    for k in range(n_labels):
        r = change_spacing(nib.Nifti1Image((data == k).astype(np.float32), img_in.affine), [spacing] * 3,
                           order=1, dtype=np.float32, nr_cpus=1, use_gpu=False)
        assert r.shape == img_rsp.shape and np.allclose(r.affine, img_rsp.affine), (r.shape, img_rsp.shape)
        v = np.asarray(r.dataobj)
        if best is None:
            best, arg = v, np.zeros(v.shape, np.uint8)
        else:
            m = v > best
            best = np.where(m, v, best)
            arg[m] = k
    return arg


def prereg_sha_logged():
    """sha256 of prereg.json as logged in tsft_full_prereg.log ('PREREG sha256 <hex>'), last entry wins."""
    h = None
    if PREREG_LOG.is_file():
        for line in open(PREREG_LOG):
            p = line.split()
            if len(p) >= 5 and p[2] == "PREREG" and p[3] == "sha256":
                h = p[4]
    return h


def frozen_code_hashes():
    """{repo-relative path: sha256} of the code files of the last 'CODEFREEZE' entry in tsft_full_prereg.log."""
    out = {}
    if PREREG_LOG.is_file():
        for line in open(PREREG_LOG):
            p = line.split()
            if len(p) >= 5 and p[2] == "CODE" and len(p[3]) == 64:
                out[" ".join(p[4:])] = p[3]
    return out


def check_code_frozen(paths, log=print, strict=True):
    """Refuse (strict) / warn unless every given code file hashes to its CODEFREEZE value in the prereg log."""
    fz = frozen_code_hashes()
    bad = {}
    for p in paths:
        r = rel(p)
        got = sha256(REPO / r)
        if fz.get(r) != got:
            bad[r] = (got[:16], (fz.get(r) or "NOT-FROZEN")[:16])
    if bad:
        msg = f"code differs from the CODEFREEZE hashes in {rel(PREREG_LOG)}: {bad}"
        if strict:
            raise SystemExit("[REFUSE] " + msg)
        log("[warn] " + msg)
    return not bad
