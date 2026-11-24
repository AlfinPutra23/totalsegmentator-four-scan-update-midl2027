"""vccor/audit_ft: AUDIT-GATED PARTIAL-LABEL FINE-TUNING of TotalSegmentator-full (1.5 mm) on FLARE22.
Pre-registered (outputs/results/vccor/audit_ft/prereg.json; sha256 + CODEFREEZE in outputs/logs/vccor/audit_ft_prereg.log).

Question. F1b (tsft_full: all weights, lr 1e-4, 50 x 250 it, TS init incl. heads, 4 labelled scans with GT for all 13
FLARE organs) lifts TS-full from 90.07 to 91.95 macro Dice on held18. The frozen VC audit (vv2_common.vc_fit on the
per-organ relative volume error of TS-full vs GT on the 4 labelled scans) marks the organs with a systematic protocol
offset, and the fine-tune's gain sits almost entirely inside those organs (screen 3). HYPOTHESIS: annotating only the
audit-flagged organs (TS-full's own labels fill the rest) gives the same fine-tuning gain as annotating all 13 organs,
and more than annotating (a) a random subset and (b) the worst-zero-shot-Dice subset of the same size. The novel part is
the selection rule (systematic, sign-consistent volume error), not the training machinery.

Flagged set (fixed before launch; judge's addendum): organ o is flagged iff its signed relative volume error
r_o = (|G_o| - |T_o|) / |T_o| has the same sign on >= 3 of the 4 labelled scans AND the frozen vc_fit rule gives it a
non-zero budget in that direction (grow: mean(r) - SE > 2 %; trim: -mean(r) - SE > 2 %).

Arms (per labelled draw; the F1b recipe and compute are identical, only the LABEL COMPOSITION on the 4 labelled scans
changes; non-FLARE classes always come from TS's own part prediction through tsft_full_common.composite_target):
  FULL    GT for all 13 FLARE organs (seed 1337 = the existing F1b run, reused; seeds 2025 / 3407 new runs)
  AUDIT   GT only for the flagged organs; TS-full's own (merged, FLARE-mapped) prediction for the other FLARE organs
  DICE    GT only for the |AUDIT| organs with the worst zero-shot TS-full Dice on the draw's 4 labelled scans (control)
  RANDOM  GT only for a seeded random subset of |AUDIT| organs (control)
  ANTI    GT only for the non-flagged organs (seed 1337 only; mechanism control)
  MIX2    two scans with GT for all 13 organs + two scans with GT for the flagged organs only (seed 1337 only;
          secondary: the practical pipeline where some full annotations exist)
Labelled draws: seeds 1337 / 2025 / 3407 = the trainer split (scripts/cor_build/build_cor.py recover_split).

This module has NO import-time side effects beyond sys.path (safe before nnunetv2 / torch; GPU jobs import it).
Everything under scripts/vccor/tsft_full, scripts/cor_salvage, scripts/cor_confirm, outputs/results/vccor/tsft_full and
outputs/results/cor_* is imported / read only.
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
TSFT_DIR = REPO / "scripts/vccor/tsft_full"            # FROZEN, read-only (label functions, ts_pipeline, layout)
sys.path.insert(0, str(TSFT_DIR))
import tsft_full_common as Q  # noqa: E402

OUT = REPO / "outputs/results/vccor/audit_ft"
LOGD = REPO / "outputs/logs/vccor"
PREREG = OUT / "prereg.json"
PREREG_LOG = LOGD / "audit_ft_prereg.log"
AUDIT_DIR = OUT / "audit"                               # audit_s<seed>.json + summary.json (labelled GT only)
AUDIT_SUMMARY = AUDIT_DIR / "summary.json"
LABELS_ROOT = REPO / "MagicNet/data/vccor_audit_ft"     # s<seed>/<arm>/{flare,part291,part293}/FLARE_<cid>.nii.gz
LABELS_MANIFEST = LABELS_ROOT / "manifest.json"

# ---------------------------------------------------------------------------------------------------- layout (ours)
NN_ROOT = OUT / "nnunet"
NN_RAW = NN_ROOT / "nnUNet_raw"
NN_PRE = NN_ROOT / "nnUNet_preprocessed"
PREP = OUT / "prep"                                     # prep/ts_cpu/<cid>/ (new labelled captures), DONE.json, verify.json
PREP_DONE = PREP / "DONE.json"
VERIFY_JSON = PREP / "verify.json"
MODELS = OUT / "models"                                 # models/<run>/part<tid>/, models/<run>/DONE.json
PRED = OUT / "pred"                                     # pred/<run>/{raw,h5}/<cid>, pred/<run>/DONE.json
MAPS = OUT / "maps"                                     # maps/<set>/<cid>.npz + label_manifest.json + DONE.json (no GT)
RES = OUT / "results"                                   # pervol/, results.json, tables.md, GT_OPENED

# ---------------------------------------------------------------------------------------------------- frozen inputs
TSFT_OUT, TSFT_PREP, TSFT_MAPS, TSFT_PRED = Q.OUT, Q.PREP, Q.MAPS, Q.PRED       # read-only
TSFT_RAW_LABELS = {t: Q.NN_RAW / Q.DATASET_NAME[t] / "labelsTr" for t in Q.PARTS_FT}  # F1b's prepared labels (seed 1337)
F0_CACHE = Q.CACHE                                      # frozen-part (292/294/295) outputs of the F0 GPU run + image key
F1B_MODELS = Q.MODELS / "F1b"
RAW_IMG = REPO / "data/FLARE22Train/images/FLARE22_Tr_{cid}_0000.nii.gz"
RAW_LBL = REPO / "data/FLARE22Train/labels/FLARE22_Tr_{cid}.nii.gz"    # read ONLY for a draw's 4 labelled scans
TS_FULL_RAW = REPO / "MagicNet/data/ts_fullres/raw_full/{cid}.nii.gz"  # stored TS-full raw output (capture check)
FLARE_H5 = REPO / "MagicNet/data/flare_h5"                             # 'label' (GT): labelled scans + scorer only
PSEUDO_FULLRES = REPO / "MagicNet/data/flare_pseudo_fullres"           # TS-full 'pseudo' (h5 frame)
VC_FIT_V2 = REPO / "outputs/results/cor_salvage/v2/vc_fit.json"        # labelled-4 (seed 1337) volumes of the frozen fit
PREREG_MERGED = REPO / "outputs/results/cor_confirm/prereg_merged.json"  # pinned seed-1337 VC budget (delta, eps)
TSFT_RESULTS = Q.RES / "results.json"                   # F1b per-organ gains (design data for the MDE / forecast)

# ---------------------------------------------------------------------------------------------------- the design
SEEDS = (1337, 2025, 3407)
FT_SEED = Q.SEED                                        # the fine-tuning RNG seed is 1337 for every arm (as F1b)
# the trainer split replayed by labelled_ids(); cross-checked against scripts/method_candidates/cor36/common36.LAB
LAB_EXPECTED = {1337: ["0014", "0001", "0006", "0038"], 2025: ["0007", "0026", "0032", "0029"],
                3407: ["0018", "0030", "0033", "0004"]}
DEV18, HELD18, TEST10 = list(Q.DEV18), list(Q.HELD18), list(Q.TEST10)
EVAL28 = HELD18 + TEST10                                # predicted + scored (dev18 is not used by this block)
SETS = {"held18": HELD18, "test10": TEST10}
NAMES = list(Q.NAMES)
PARTS_FT, PARTS_FROZEN, PARTS_ALL = Q.PARTS_FT, Q.PARTS_FROZEN, Q.PARTS_ALL
MIN_SIGN = 3                                            # same-sign on >= 3 of the 4 labelled scans
ARMS = {
    "FULL": "GT for all 13 FLARE organs (seed 1337: the existing F1b run, reused)",
    "AUDIT": "GT only for the audit-flagged organs; TS-full's own labels for the other FLARE organs",
    "DICE": "GT only for the |AUDIT| organs with the worst zero-shot TS-full Dice on the draw's 4 labelled scans",
    "RANDOM": "GT only for a seeded random subset of |AUDIT| organs",
    "ANTI": "GT only for the non-flagged organs (seed 1337 only; mechanism control)",
    "MIX2": "2 scans with GT for all 13 organs + 2 scans with GT for the flagged organs only (seed 1337 only; secondary)",
}
ARMS_BY_SEED = {1337: ["FULL", "AUDIT", "DICE", "RANDOM", "ANTI", "MIX2"],
                2025: ["FULL", "AUDIT", "DICE", "RANDOM"], 3407: ["FULL", "AUDIT", "DICE", "RANDOM"]}
PRIMARY_ARMS = ["FULL", "AUDIT", "DICE", "RANDOM"]      # every draw
REUSED = {(1337, "FULL")}                               # = tsft_full F1b (models / predictions), NOT retrained
RUN_ORDER = [(s, a) for s in SEEDS for a in ARMS_BY_SEED[s]]
TRAIN_RUNS = [(s, a) for s, a in RUN_ORDER if (s, a) not in REUSED]      # 13 trainings
# GPU launch order: seed-1337 primary arms + ANTI, then seeds 2025 / 3407 FULL + arms, MIX2 (secondary, droppable) last
TRAIN_RUNS_ORDERED = [(s, a) for s, a in TRAIN_RUNS if a != "MIX2"] + [(s, a) for s, a in TRAIN_RUNS if a == "MIX2"]
assert len(TRAIN_RUNS_ORDERED) == len(TRAIN_RUNS) == 13
LR, TRAINABLE, EPOCHS, ITERS_PER_EPOCH = 1e-4, "all", Q.EPOCHS, Q.ITERS_PER_EPOCH   # == F1b
VAL_ITERS_PER_EPOCH, SAVE_EVERY, N_PROC_DA = Q.VAL_ITERS_PER_EPOCH, Q.SAVE_EVERY, Q.N_PROC_DA
PLANS_ID, FOLD, CONFIG, TS_TRAINER = Q.PLANS_ID, Q.FOLD, Q.CONFIG, Q.TS_TRAINER
RULE = dict(ni_margin=-0.30, ni_ci_lower=-0.60, min_audit_minus_random=0.50, min_draws_positive=2, alpha=0.05,
            organ_floor=-1.00)
NBOOT, BOOT_SEED = 5000, 0


def run_name(seed, arm):
    return f"s{seed}_{arm}"


def dataset_id(seed, arm, tid):
    """3-digit nnU-Net ids 700-751 (own root; never collides with TS 291-295 / tsft_full 991-993 / ssft 911-934)."""
    return 700 + 20 * SEEDS.index(seed) + 2 * list(ARMS).index(arm) + (0 if tid == 291 else 1)


def dataset_name(seed, arm, tid):
    return f"Dataset{dataset_id(seed, arm, tid)}_AFT{tid}_s{seed}_{arm}"


def set_of(c):
    for s, ids in SETS.items():
        if c in ids:
            return s
    raise KeyError(c)


def eval_ids(seed, s):
    """Evaluation scans of draw `seed` in set s: the draw's own labelled scans are excluded (they are training data)."""
    return [c for c in SETS[s] if c not in LAB_EXPECTED[seed]]


# ---------------------------------------------------------------------------------------------------- helpers
sha256, rel, write_json, Log, be_nice, cap_threads = Q.sha256, Q.rel, Q.write_json, Q.Log, Q.be_nice, Q.cap_threads


def now():
    return time.strftime("%F %T")


def set_nnunet_env(results_root):
    """Component-owned nnU-Net roots (ours, not tsft_full's). Must run before nnunetv2 is imported."""
    if "nnunetv2" in sys.modules:
        raise RuntimeError("set_nnunet_env() must be called before importing nnunetv2")
    os.environ["nnUNet_raw"] = str(NN_RAW)
    os.environ["nnUNet_preprocessed"] = str(NN_PRE)
    os.environ["nnUNet_results"] = str(results_root)


def results_root(seed, arm):
    return NN_ROOT / f"results_{run_name(seed, arm)}"


def model_dir(seed, arm, tid):
    return MODELS / run_name(seed, arm) / f"part{tid}"


def model_done(seed, arm):
    return MODELS / run_name(seed, arm) / "DONE.json"


def pred_dir(seed, arm):
    return PRED / run_name(seed, arm)


def pred_done(seed, arm):
    return PRED / run_name(seed, arm) / "DONE.json"


def label_path(seed, arm, kind, cid):
    """kind: 'flare' (composed FLARE-id label on TS's grid) | 'part291' | 'part293' (composite training targets)."""
    return LABELS_ROOT / run_name(seed, arm) / kind / f"FLARE_{cid}.nii.gz"


def cap_dir(cid):
    """1.5 mm TS input image + the five per-part TS predictions on that grid: seed-1337 labelled scans from the frozen
    tsft_full prep (read-only); the other draws' labelled scans from our own CPU capture (prep_audit_ft.py)."""
    return (TSFT_PREP / "ts_cpu" / cid) if cid in LAB_EXPECTED[1337] else (PREP / "ts_cpu" / cid)


# ---------------------------------------------------------------------------------------------------- split replay
def labelled_ids(seed, labelnum=4, n_train=40):
    """== scripts/cor_build/build_cor.py recover_split: ids parsed as the trainer does, train = total_list[:40],
    RandomState(seed).shuffle(train), labelled = train[:labelnum]. Asserted against the known label sets."""
    import numpy as np
    with open(REPO / "MagicNet/data/flare.txt") as f:
        ids = ["{}".format(x.strip().split('.')[0][-4:]) for x in f.readlines()]
    train, test = list(ids[:n_train]), list(ids[n_train:])
    np.random.RandomState(seed).shuffle(train)
    lab = train[:labelnum]
    assert lab == LAB_EXPECTED[seed], (seed, lab, LAB_EXPECTED[seed])
    assert test == TEST10 and not set(lab) & set(TEST10)
    return lab


# ---------------------------------------------------------------------------------------------------- the audit
def vv2():
    """The frozen vc_fit (scripts/cor_salvage/verify_v2/vv2_common.py; read-only). Imported lazily: that module sets
    CUDA_VISIBLE_DEVICES="" and 1 BLAS thread at import, which is what every CPU script here wants anyway."""
    sys.path.insert(0, str(REPO / "scripts/cor_salvage/verify_v2"))
    import vv2_common as W
    return W


def audit_flags(vols, min_sign=MIN_SIGN):
    """vols: (4, 13, 2) = (|G_o|, |T_o|) on the labelled scans. Returns the per-organ audit:
    r (4 x 13 signed relative volume errors), the frozen VC budget (delta, eps), the sign counts, and
    flagged = budget on AND same sign on >= min_sign scans (grow: r > 0; trim: r < 0)."""
    import numpy as np
    W = vv2()
    v = np.asarray(vols, np.float64)
    assert v.shape[0] >= 2 and v.shape[1:] == (13, 2), v.shape
    F = W.vc_fit(v)
    r, delta, eps = F["r"], F["delta"], F["eps"]
    n_pos, n_neg = (r > 0).sum(0), (r < 0).sum(0)
    grow = (delta > 0) & (n_pos >= min_sign)
    trim = (eps > 0) & (n_neg >= min_sign)
    assert not (grow & trim).any()
    direction = ["grow" if grow[k] else ("trim" if trim[k] else "off") for k in range(13)]
    budget_only = [k + 1 for k in range(13) if delta[k] > 0 or eps[k] > 0]
    return dict(r=r.tolist(), mean_r=r.mean(0).tolist(), se_r=(r.std(0, ddof=1) / np.sqrt(r.shape[0])).tolist(),
                delta=delta.tolist(), eps=eps.tolist(), n_pos=n_pos.tolist(), n_neg=n_neg.tolist(), min_sign=min_sign,
                direction=direction, flagged=[k + 1 for k in range(13) if grow[k] or trim[k]],
                budget_only_flagged=budget_only,
                dropped_by_sign_rule=[k + 1 for k in budget_only_idx(delta, eps) if not (grow[k] or trim[k])])


def budget_only_idx(delta, eps):
    return [k for k in range(13) if delta[k] > 0 or eps[k] > 0]


def organ_names(ids):
    return [NAMES[o - 1] for o in ids]


# ---------------------------------------------------------------------------------------------------- label composition
def partial_gt(G, Tf, annotated):
    """Partial annotation on TS's 1.5 mm grid, FLARE ids. G: full GT (0..13); Tf: TS-full's merged prediction mapped to
    FLARE ids on the same grid; annotated: iterable of FLARE organ ids the annotator labelled.
      G == o, o annotated          -> o          (the annotated organs are authoritative, inside and outside)
      otherwise, Tf == u, u NOT annotated -> u   (TS-full's own label fills every unannotated organ)
      otherwise                    -> 0          (incl. TS's prediction of an annotated organ outside its annotation:
                                                  the annotator says it is not that organ; what it is, is unknown)
    With all 13 organs annotated the result is G itself (FULL = F1b's labels, bit-exact)."""
    import numpy as np
    ann = np.zeros(14, bool)
    for o in annotated:
        ann[int(o)] = True
    assert not ann[0]
    G = G.astype(np.uint8)
    Tf = Tf.astype(np.uint8)
    Gp = np.where(ann[G], G, 0).astype(np.uint8)
    fill = (Gp == 0) & (Tf > 0) & ~ann[Tf]
    Gp[fill] = Tf[fill]
    cnt = {"n_vox": int(G.size), "annotated_organs": sorted(int(o) for o in annotated),
           "gt_kept": int(ann[G].sum()), "ts_filled": int(fill.sum()),
           "gt_unannotated_vox": int(((G > 0) & ~ann[G]).sum()),
           "gt_unannotated_vox_ts_agrees": int(((G > 0) & ~ann[G] & (Gp == G)).sum()),
           "ts_annotated_organ_outside_gt_zeroed": int(((Tf > 0) & ann[Tf] & (Gp == 0)).sum()),
           "differs_from_gt": int((Gp != G).sum())}
    return Gp, cnt


def ts_full_flare_map(parts, luts):
    """TS-full's merged output on the 1.5 mm grid (parts 291..295 in TS's order; a later part's non-background class
    overwrites, totalsegmentator/nnunet.py), mapped to FLARE ids through the part LUTs (v2: cysts -> kidneys)."""
    import numpy as np
    Tf = None
    for tid in PARTS_ALL:
        P = parts[tid].astype(np.uint8)
        if Tf is None:
            Tf = np.zeros(P.shape, np.uint8)
        nz = P > 0
        Tf[nz] = luts[tid][P[nz]]
    return Tf


def arm_annotation(seed, arm, cid, audit):
    """The annotated organ set of (seed, arm) on scan cid, from the draw's audit record."""
    if arm == "FULL":
        return list(range(1, 14))
    if arm == "MIX2":
        return list(range(1, 14)) if cid in audit["mix2"]["full_scans"] else list(audit["sets"]["AUDIT"])
    return list(audit["sets"][arm])


def annotation_cost(G_raw, annotated, slice_axis=2):
    """Annotation cost on the native GT grid: organs, GT-foreground voxel fraction, and the number of slices (along
    slice_axis) that contain an annotated organ, vs the number containing any FLARE organ."""
    import numpy as np
    ann = np.zeros(14, bool)
    for o in annotated:
        ann[int(o)] = True
    fg = G_raw > 0
    a = ann[G_raw]
    axes = tuple(i for i in range(G_raw.ndim) if i != slice_axis)
    return {"n_organs": int(ann.sum()), "fg_vox": int(fg.sum()), "annotated_vox": int(a.sum()),
            "vox_fraction": float(a.sum() / max(1, fg.sum())),
            "slices_any_organ": int(fg.any(axis=axes).sum()), "slices_annotated": int(a.any(axis=axes).sum()),
            "n_slices_total": int(G_raw.shape[slice_axis])}


# ---------------------------------------------------------------------------------------------------- integrity
def prereg_sha_logged():
    h = None
    if PREREG_LOG.is_file():
        for line in open(PREREG_LOG):
            p = line.split()
            if len(p) >= 5 and p[2] == "PREREG" and p[3] == "sha256":
                h = p[4]
    return h


def frozen_code_hashes():
    out = {}
    if PREREG_LOG.is_file():
        for line in open(PREREG_LOG):
            p = line.split()
            if len(p) >= 5 and p[2] in ("CODE", "IMPORT") and len(p[3]) == 64:
                out[" ".join(p[4:])] = p[3]
    return out


def check_code_frozen(paths, log=print, strict=True):
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


def check_prereg():
    h = sha256(PREREG)
    if h != prereg_sha_logged():
        raise SystemExit(f"[REFUSE] prereg.json sha256 {h} != logged {prereg_sha_logged()} ({rel(PREREG_LOG)})")
    return json.load(open(PREREG))


def load_audit(seed, check_prereg_pin=True):
    """The draw's audit record (flagged / DICE / RANDOM / ANTI sets, MIX2 spec); pinned by sha256 in prereg.json."""
    p = AUDIT_DIR / f"audit_s{seed}.json"
    a = json.load(open(p))
    if check_prereg_pin and PREREG.is_file():
        pr = json.load(open(PREREG))
        want = pr["design"]["audit_files_sha256"].get(rel(p))
        if want is not None and want != sha256(p):
            raise SystemExit(f"[REFUSE] {p} changed after the pre-registration")
    assert a["seed"] == seed and a["labelled"] == LAB_EXPECTED[seed]
    return a


def verify_done(log=None):
    """prep/DONE.json present and every file it hashed unchanged (labels, datasets, captures)."""
    if not PREP_DONE.is_file():
        return False
    d = json.load(open(PREP_DONE))
    bad = [f for f, h in d["files_sha256"].items() if not (REPO / f).is_file() or sha256(REPO / f) != h]
    if bad and log:
        log(f"[verify] {len(bad)} prep files changed or missing: {bad[:5]}")
    return not bad
