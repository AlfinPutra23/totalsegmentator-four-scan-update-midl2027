"""vccor/final FLARE-T50 POST HOC (J_t50_posthoc): constants and helpers. EXPLORATORY, NOT REGISTERED.

Approved job of 2026-11-13: a post hoc second reading of the FLARE-T50 labels (first read by the registered look,
GT_OPENED 2026-11-13 09:01:23) to score three stored PEFT-study models that the registered look did not include:
  LR1E2   e2 flare_K4_LR1E2: the F1b recipe (all weights) at nnU-Net's default lr 1e-2   (the learning-rate factor)
  LORA_B  peft LORA_B: LoRA r 8 / alpha 16 on every Conv3d, lr 1e-3 (the dev18-selected LoRA arm)
  HEAD    peft HEAD: segmentation heads only, lr 1e-2                   (lr 1e-2 without training the whole network)
All three: TS-full parts 291 + 293 fine-tuned on FLARE K = 4, draw s1337 (0014 0001 0006 0038), 50 x 250 it, the same
draw as STOCK and FULL_s1337 on T50. Question: does the T50 STOCK collapse (STOCK - F0 = -13.5) come from the lr, from
training the whole network, or from neither?

Frozen modules (t50_common, final_common, fin_look, m2_cluster, peft_common, ts_pipeline) are imported read-only and are
never edited. Nothing is written under I_flaret50/ (the F0 frozen-part cache and the T50 images are only read).
No import-time side effects beyond sys.path.
Version of 2026-11-13.
"""
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
FINAL = HERE.parent
for _p in (str(FINAL), str(FINAL / "flaret50"), str(FINAL.parent / "m2"), str(FINAL.parent / "peft")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import t50_common as T  # noqa: E402  (frozen; read-only)

REPO = T.REPO
VC = T.VC
OUT = VC / "final" / "J_t50_posthoc"
PRED = OUT / "pred"
TMP_LABELS = OUT / "tmp_labels_verified"           # the verified label copy; deleted at the end of scoring
LOGP = OUT / "posthoc.log"
I_DIR = T.OUT                                      # I_flaret50 (read only)
F0_CACHE = T.PRED / "F0" / "cache_frozen"         # F0's frozen-part cache on T50 (read only)
FLOOR_GIB = 25.0
GPU1_UUID = "GPU-1a7b7614-499a-6540-55fd-5920dcee8340"   # bus 00000000:05:00.0 (nvidia-smi -L, 2026-11-13)
TAG = "t50_posthoc"

E2M, PEFTM = VC / "e2" / "models", VC / "peft" / "models"
# model files and sha256 as recorded in the PEFT records (prereg_peft.json look arms -> their pred DONE.json model_sha256;
# models/<v>/<part>/export.json checkpoint_sha256)
ARMS = {
    "LR1E2": dict(dir=E2M / "flare_K4_LR1E2", trainable="all", lr=1e-2, family="e2",
                  record=VC / "e2" / "pred" / "flare_K4_LR1E2" / "DONE.json",
                  shas={291: "0ce916f206a4515fcf334401af3a88c4c017504e8352c583236fb90b85822edd",
                        293: "dbd6656ef66a6bed150f45af98bc5e8d908c89217e6358c77ac352dadcedfdff"},
                  label="e2 flare_K4_LR1E2: F1b recipe (all weights) at lr 1e-2, K = 4 s1337"),
    "LORA_B": dict(dir=PEFTM / "LORA_B", trainable="lora", lr=1e-3, family="peft",
                   record=VC / "peft" / "pred" / "LORA_B" / "DONE.json",
                   shas={291: "aa8c65c84ac067d89d63d178443beac73ec38dbaabdd966c67fe7140105fa7ae",
                         293: "ce44a70653808cfb3927feb0cde9d06d7bddddaa9fcb52f4bd3a0bb3ff689d6d"},
                   label="peft LORA_B: LoRA r 8 / alpha 16 on every Conv3d (merged), lr 1e-3, K = 4 s1337"),
    "HEAD": dict(dir=PEFTM / "HEAD", trainable="heads", lr=1e-2, family="peft",
                 record=VC / "peft" / "pred" / "HEAD" / "DONE.json",
                 shas={291: "c8bbfac15696e06c54eeb36d552d22e2a9c549025e42a4903ff65c5da5783dca",
                       293: "73d40cda4056f04003696d82dd7a59c21ae85183ed63ed977107585f89006b26"},
                 label="peft HEAD: segmentation heads only (decoder.seg_layers), lr 1e-2, K = 4 s1337"),
}
ORDER = ["LR1E2", "LORA_B", "HEAD"]
PARTS = (291, 293)
S1337 = ["0014", "0001", "0006", "0038"]


def log(msg):
    OUT.mkdir(parents=True, exist_ok=True)
    T.log_event(msg, LOGP)


def write_json(path, obj):
    path = Path(path)
    assert str(path.resolve()).startswith(str(OUT.resolve())), f"post hoc writes only under {OUT}: {path}"
    T.write_json(path, obj)


def disk_ok(extra_gib=0.0):
    free = T.disk_free_gib()
    if free - extra_gib < FLOOR_GIB:
        raise T.Refuse(f"/ has {free:.2f} GiB free; floor {FLOOR_GIB} GiB", code=3)
    return free


def part_files(arm, tid):
    md = Path(ARMS[arm]["dir"]) / f"part{tid}"
    return [md / "fold_0" / "checkpoint_final.pth", md / "export.json", md / "plans.json", md / "dataset.json"]


def resolve(arm):
    """Asserts the arm's model == its records (== predict_peft.model_shas / predict_e2 checks): checkpoint sha256 equals
    export.json, the PEFT-record sha256 and the source pred DONE.json; full 50-epoch non-smoke run; recipe fields.
    -> (redirect {tid: dir}, shas {tid: sha256}, file_hashes {rel path: sha256})."""
    a = ARMS[arm]
    done = json.load(open(a["record"]))
    red, shas, fh = {}, {}, {}
    for tid in PARTS:
        md = Path(a["dir"]) / f"part{tid}"
        man = json.load(open(md / "export.json"))
        h = T.sha256_cached(md / "fold_0" / "checkpoint_final.pth")
        if not (h == man["checkpoint_sha256"] == a["shas"][tid] == done["model_sha256"][str(tid)]):
            raise T.Refuse(f"{arm} part {tid}: checkpoint sha256 {h[:16]} differs from export.json / the PEFT records")
        if man.get("num_epochs") != 50 or man.get("smoke"):
            raise T.Refuse(f"{arm} part {tid}: not a full 50-epoch non-smoke run")
        if abs(float(man["initial_lr"]) - a["lr"]) > 1e-12:
            raise T.Refuse(f"{arm} part {tid}: initial_lr {man['initial_lr']} != {a['lr']}")
        if list(man.get("labelled_ids") or []) != S1337:
            raise T.Refuse(f"{arm} part {tid}: labelled_ids {man.get('labelled_ids')} != s1337 {S1337}")
        if a["family"] == "peft":
            if man.get("variant") != arm:
                raise T.Refuse(f"{arm} part {tid}: export variant {man.get('variant')}")
            ok = man["trainable"] == a["trainable"] or (a["trainable"] == "lora" and isinstance(man["trainable"], int)
                                                        and man["trainable"] > 0 and man.get("lora_merged") is True)
            if not ok:
                raise T.Refuse(f"{arm} part {tid}: trainable {man['trainable']!r}")
        else:
            if man.get("run") != "flare_K4_LR1E2" or man.get("n_trainable") != man.get("n_total"):
                raise T.Refuse(f"{arm} part {tid}: not the all-weights e2 LR1E2 run")
        if int(man.get("part", tid)) != tid:
            raise T.Refuse(f"{arm} part {tid}: export part {man.get('part')}")
        red[tid], shas[tid] = str(md), h
        for f in part_files(arm, tid):
            fh[T.rel(f)] = T.sha256_cached(f)
    return red, shas, fh
