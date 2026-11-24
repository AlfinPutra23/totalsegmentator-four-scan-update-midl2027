"""vccor/final LCTSC TRAIN (GPU job; idempotent: an exported part is skipped; resumable from checkpoint_latest every 5 epochs):
one F1b run of lctsc_runs.RUNS (F1b_S1..S3, site-matched draws), parts 291 and 293.

== scripts/vccor/final/train_final.py (the WORD / plan-B trainer, frozen in WORD code v5), copied with these changes only:
the plan module is lctsc_runs (no --plan argument); no STOCK path is used (LCTSC has no STOCK arm); the frozen-plan
preflight (final_common.preflight, plan registry A-E) is replaced by lctsc_runs.verify_datasets and the code hashes in each
export; nnU-Net results stay under K_lctsc/nnunet. Recipe F1b (bsl_common.RECIPES): stock nnU-Net 2.7.0
nnUNetTrainerNoMirroring, init = EVERY tensor of TS's checkpoint_final of the part (strict), all weights trainable, SGD
Nesterov 0.99, wd 3e-5, PolyLR from lr 1e-4 over 50 x 250 it, 10 val it, batch 2, patch 128^3, seeds 1337.

usage:  .venv/bin/python -I -B scripts/vccor/final/lctsc/lctsc_train.py --run F1b_S1 --device cuda --n_proc_da 8
CPU smoke:  CUDA_VISIBLE_DEVICES="" .venv/bin/python -I -B scripts/vccor/final/lctsc/lctsc_train.py --run F1b_S1 --device cpu \
                --smoke --scratch <dir> --nn_root <K_lctsc/nnunet> --parts 291

Version of 2026-11-13.
"""
import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
FINAL = HERE.parent
for _p in (str(HERE), str(FINAL), str(FINAL.parent / "tsft_full")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import final_common as F  # noqa: E402  (frozen; read-only import)

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--run", required=True)
ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
ap.add_argument("--n_proc_da", type=int, default=8)
ap.add_argument("--cpu_threads", type=int, default=4)
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--smoke_iters", type=int, default=1)
ap.add_argument("--recipe", default=None, help="smoke only: recipe override")
ap.add_argument("--parts", default=None, help="smoke only: subset of parts")
ap.add_argument("--scratch", default=None)
ap.add_argument("--nn_root", default=None, help="smoke only")
args = ap.parse_args()
import lctsc_runs as C  # noqa: E402
args.plan = "lctsc"
RUN = C.RUNS[args.run]
if args.smoke:
    assert args.scratch and args.nn_root, "--smoke needs --scratch and --nn_root"
    RECIPE = args.recipe or C.recipe_of(args.run)
    NN_ROOT, EXPORT = Path(args.nn_root), Path(args.scratch) / "models" / args.run
    RES_ROOT = Path(args.scratch) / f"results_{args.run}"
else:
    assert not (args.recipe or args.parts or args.scratch or args.nn_root), "smoke-only options"
    RECIPE = C.recipe_of(args.run)
    NN_ROOT, EXPORT = C.NN_ROOT, C.MODELS / args.run
    RES_ROOT = C.NN_ROOT / f"results_{args.run}"
import bsl_common as B  # noqa: E402  (the recipe table is plan B's for both plans)
V = B.RECIPES[RECIPE]
STOCK = V["kind"] == "stock"
if STOCK:
    assert RUN.get("stock_dataset") is not None, f"{args.run}: no STOCK dataset defined"
PARTS = sorted(RUN["ts_datasets"]) if not STOCK else []
if args.smoke and args.parts:
    PARTS = [int(x) for x in args.parts.split(",")]
LOG = F.Log(F.LOGD / ("final_lctsc_smoke.log" if args.smoke else f"final_lctsc_train_{args.run}.log"))
os.environ["nnUNet_raw"] = str(NN_ROOT / "nnUNet_raw")
os.environ["nnUNet_preprocessed"] = str(NN_ROOT / "nnUNet_preprocessed")
os.environ["nnUNet_results"] = str(RES_ROOT)
os.environ["nnUNet_n_proc_DA"] = str(0 if args.smoke else args.n_proc_da)
os.environ["nnUNet_compile"] = "false"
if args.device == "cpu":
    F.cap_threads(args.cpu_threads)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from batchgenerators.utilities.file_and_folder_operations import load_json  # noqa: E402
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler  # noqa: E402
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerNoMirroring import nnUNetTrainerNoMirroring  # noqa: E402
from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs_NoMirroring import (  # noqa: E402
    nnUNetTrainer_250epochs_NoMirroring)
sys.path.insert(0, str(F.REPO / "scripts/vccor/peft"))
import lora as LR  # noqa: E402  (frozen peft LoRA)
import tsft_full_common as Q  # noqa: E402

TCLS = nnUNetTrainer_250epochs_NoMirroring if STOCK else nnUNetTrainerNoMirroring
TRAINER = TCLS.__name__
EPOCHS = V.get("epochs", Q.EPOCHS)
SEED = 1337
CONFIG = Q.CONFIG
FOLD_TS, FOLD_STOCK = "all", "all"


def plans_id(tid):
    return Q.PLANS_ID[tid] if tid else "TS291StockPlans"


def ts_weights(tid):
    ck = torch.load(Q.TS_MODEL[tid] / "fold_0" / "checkpoint_final.pth", map_location="cpu", weights_only=False)
    assert ck["trainer_name"] == Q.TS_TRAINER and ck["init_args"]["configuration"] == CONFIG
    return ck["network_weights"]


def unwrap(net):
    return net._orig_mod if hasattr(net, "_orig_mod") else net


def load_ts_init(net, tid):
    sd, mod = ts_weights(tid), unwrap(net)
    own = mod.state_dict()
    assert set(own) == set(sd) and all(own[k].shape == sd[k].shape for k in own), "TS checkpoint / network mismatch"
    mod.load_state_dict(sd, strict=True)
    return len(sd)


def load_pretrained_cpu(net, tid=291):
    sd, mod = ts_weights(tid), unwrap(net)
    own = mod.state_dict()
    keep = {}
    for k, v in own.items():
        if ".seg_layers." in k:
            continue
        assert k in sd and sd[k].shape == v.shape, f"pretrained mismatch at {k}"
        keep[k] = sd[k]
    own.update(keep)
    mod.load_state_dict(own)
    return {"transferred": len(keep), "reinitialised": sorted(k for k in own if ".seg_layers." in k)}


def setup_trainable(tr):
    mod, kind = unwrap(tr.network), V["trainable"]
    info = {}
    if kind == "all":
        params = list(mod.parameters())
    elif kind == "heads":
        for n, p in mod.named_parameters():
            p.requires_grad_(n.startswith("decoder.seg_layers."))
        params = [p for p in mod.parameters() if p.requires_grad]
    elif kind == "lora":
        info["wrapped_convs"] = LR.inject(mod, V["r"], V["alpha"])
        params = LR.lora_parameters(mod)
        info |= LR.count(mod)
    else:
        raise ValueError(kind)
    assert params and sum(p.numel() for p in params) == sum(p.numel() for p in mod.parameters() if p.requires_grad)
    opt = torch.optim.SGD(params, V["lr"], weight_decay=tr.weight_decay, momentum=0.99, nesterov=True)
    tr.optimizer, tr.lr_scheduler = opt, PolyLRScheduler(opt, V["lr"], tr.num_epochs)
    info["n_trainable"] = int(sum(p.numel() for p in params))
    return info


def merged_state(sd):
    out, s = {}, V["alpha"] / V["r"]
    for k, v in sd.items():
        if k.endswith(".lora_A") or k.endswith(".lora_B"):
            continue
        if k.endswith(".base.weight"):
            stem = k[: -len(".base.weight")]
            A, Bm = sd[stem + ".lora_A"], sd[stem + ".lora_B"]
            out[stem + ".weight"] = (v.float() + (Bm.float() @ A.float()).view(v.shape) * s).to(v.dtype)
        elif k.endswith(".base.bias"):
            out[k[: -len(".base.bias")] + ".bias"] = v
        else:
            out[k] = v
    return out


def verify_vs_ts(w, tid):
    sd = ts_weights(tid)
    assert set(w) == set(sd) and all(w[k].shape == sd[k].shape for k in w), "exported keys / shapes != TS's"
    kind = V["trainable"]
    if kind == "all":
        return {"frozen_checked": 0}
    if kind == "heads":
        frozen = [k for k in w if not k.startswith("decoder.seg_layers.")]
    else:
        conv_w = {k for k in w if k.endswith(".weight") and w[k].dim() == 5 and ".transpconvs." not in k}
        frozen = [k for k in w if k not in conv_w]
    moved = [k for k in frozen if not torch.equal(w[k].cpu(), sd[k].cpu())]
    assert not moved, f"frozen tensors changed: {moved[:5]}"
    return {"frozen_checked": len(frozen)}


def export_ts(tid, fold_dir, info):
    final = fold_dir / "checkpoint_final.pth"
    ck = torch.load(final, map_location="cpu", weights_only=False)
    assert ck["trainer_name"] == TRAINER and ck["init_args"]["configuration"] == CONFIG
    assert ck.get("inference_allowed_mirroring_axes") is None
    w = merged_state(ck["network_weights"]) if V["trainable"] == "lora" else ck["network_weights"]
    assert all(torch.isfinite(v).all() for v in w.values() if torch.is_floating_point(v)), "non-finite weights"
    info = dict(info, **verify_vs_ts(w, tid))
    ed = EXPORT / f"part{tid}"
    if ed.exists():
        shutil.rmtree(ed)
    (ed / "fold_0").mkdir(parents=True)
    shutil.copy(Q.TS_MODEL[tid] / "plans.json", ed / "plans.json")
    shutil.copy(Q.TS_MODEL[tid] / "dataset.json", ed / "dataset.json")
    out = {k: v for k, v in ck.items() if k != "optimizer_state"} | {"network_weights": w, "optimizer_state": None}
    torch.save(out, ed / "fold_0" / "checkpoint_final.pth")
    man = dict(info, part=tid, checkpoint_sha256=F.sha256(ed / "fold_0" / "checkpoint_final.pth"), source_checkpoint=str(final),
               lora_merged=V["trainable"] == "lora", plans_sha256=F.sha256(ed / "plans.json"), exported=F.now())
    _wj(ed / "export.json", man)
    return man


def export_stock(fold_dir, info):
    final = fold_dir / "checkpoint_final.pth"
    ck = torch.load(final, map_location="cpu", weights_only=False)
    assert ck["trainer_name"] == TRAINER and ck.get("inference_allowed_mirroring_axes") is None
    assert all(torch.isfinite(v).all() for v in ck["network_weights"].values() if torch.is_floating_point(v))
    ed = EXPORT / "stock"
    if ed.exists():
        shutil.rmtree(ed)
    (ed / f"fold_{FOLD_STOCK}").mkdir(parents=True)
    for f in ("plans.json", "dataset.json", "dataset_fingerprint.json"):
        shutil.copy(fold_dir.parent / f, ed / f)
    shutil.copy(final, ed / f"fold_{FOLD_STOCK}" / "checkpoint_final.pth")
    man = dict(info, checkpoint_sha256=F.sha256(ed / f"fold_{FOLD_STOCK}" / "checkpoint_final.pth"), exported=F.now())
    _wj(ed / "export.json", man)
    return man


def exported_ok(ed):
    try:
        man = json.load(open(ed / "export.json"))
        ck = ed / ("fold_0" if not STOCK else f"fold_{FOLD_STOCK}") / "checkpoint_final.pth"
        return man["checkpoint_sha256"] == F.sha256(ck) and man.get("num_epochs") == EPOCHS and not man["smoke"] \
            and man["variant"] == RECIPE and man
    except (OSError, KeyError, ValueError):
        return False


def cleanup(fold_dir):
    if args.smoke:
        return
    freed = 0
    for f in ("checkpoint_final.pth", "checkpoint_best.pth", "checkpoint_latest.pth"):
        q = fold_dir / f
        if q.is_file():
            freed += q.stat().st_size
            q.unlink()
    LOG(f"[disk] {args.run}: freed {freed / 1e9:.2f} GB of results-dir checkpoints (export kept)")


def train_one(tid):
    did = RUN["stock_dataset"] if STOCK else RUN["ts_datasets"][tid]
    dsn = C.dataset_name(did)
    pid = plans_id(None if STOCK else tid)
    fold = FOLD_STOCK if STOCK else FOLD_TS
    pre = NN_ROOT / "nnUNet_preprocessed" / dsn
    fold_dir = RES_ROOT / dsn / f"{TRAINER}__{pid}__{CONFIG}" / f"fold_{fold}"
    ed = EXPORT / ("stock" if STOCK else f"part{tid}")
    if not args.smoke and exported_ok(ed):
        LOG(f"[skip] {args.run} {ed.name} already exported")
        return json.load(open(ed / "export.json"))
    final, latest = fold_dir / "checkpoint_final.pth", fold_dir / "checkpoint_latest.pth"
    info = {"run": args.run, "plan": args.plan, "variant": RECIPE, "label": V["label"], "trainable": V["trainable"],
            "initial_lr": V["lr"], "trainer": TRAINER, "smoke": args.smoke, "device": args.device, "dataset": dsn, "plans": pid,
            "labelled_ids": C.labelled_of(args.run),
            "code_sha256": {F.rel(f): F.sha256(f) for f in (HERE / "lctsc_train.py", HERE / "lctsc_runs.py", HERE / "lctsc_common.py",
                                                            FINAL / "final_common.py", FINAL / "bsl_common.py",
                                                            F.REPO / "scripts/vccor/peft/lora.py")}}
    if final.is_file() and (fold_dir / "final_train.json").is_file():
        LOG(f"[export-only] {args.run} {ed.name}")
        man = export_stock(fold_dir, json.load(open(fold_dir / "final_train.json"))) if STOCK else \
            export_ts(tid, fold_dir, json.load(open(fold_dir / "final_train.json")))
        cleanup(fold_dir)
        return man
    resume = latest.is_file() and not args.smoke
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if args.device == "cpu":
        torch.set_num_threads(args.cpu_threads)
    plans = load_json(pre / f"{pid}.json")
    plans["continue_training"] = resume
    tr = TCLS(plans=plans, configuration=CONFIG, fold=fold, dataset_json=load_json(pre / "dataset.json"), device=torch.device(args.device))
    want = (250, 250, 1e-2, 3e-5, 0.33) if STOCK else (1000, 250, 1e-2, 3e-5, 0.33)
    assert (tr.num_epochs, tr.num_iterations_per_epoch, tr.initial_lr, tr.weight_decay, tr.oversample_foreground_percent) == want
    tr.initial_lr = V["lr"]
    if args.smoke:
        tr.num_epochs, tr.num_iterations_per_epoch, tr.num_val_iterations_per_epoch = 1, args.smoke_iters, 1
    elif STOCK:
        tr.save_every = Q.SAVE_EVERY
    else:
        tr.num_epochs, tr.num_iterations_per_epoch = Q.EPOCHS, Q.ITERS_PER_EPOCH
        tr.num_val_iterations_per_epoch, tr.save_every = Q.VAL_ITERS_PER_EPOCH, Q.SAVE_EVERY
    tr.initialize()
    if resume:
        if not STOCK:
            info |= setup_trainable(tr)
        tr.load_checkpoint(str(latest))
        LOG(f"[resume] {args.run} {ed.name} from epoch {tr.current_epoch}")
    else:
        tr.current_epoch = 0
        if STOCK:
            info["pretrained"] = load_pretrained_cpu(tr.network, 291)
        else:
            n = load_ts_init(tr.network, tid)
            info |= setup_trainable(tr)
            LOG(f"[init] {args.run} part {tid}: all {n} tensors from TS {Q.TS_DS[tid]} (strict); trainable {info['n_trainable']:,}")
    rot, d2, ips, mirror_axes = tr.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
    assert mirror_axes is None and tr.inference_allowed_mirroring_axes is None, "mirroring must be off"
    mod = unwrap(tr.network)
    info.update(n_total=int(sum(p.numel() for p in mod.parameters())), num_epochs=tr.num_epochs if not args.smoke else EPOCHS,
                epochs_run=tr.num_epochs, iters_per_epoch=tr.num_iterations_per_epoch, val_iters_per_epoch=tr.num_val_iterations_per_epoch,
                batch_size=tr.configuration_manager.batch_size, patch_size=tr.configuration_manager.patch_size, seed=SEED,
                resumed=resume, torch=torch.__version__, output_folder=str(fold_dir),
                has_ignore_label=bool(tr.label_manager.has_ignore_label))
    LOG(f"[train] {args.run} {ed.name}: {V['label']}; {tr.num_epochs} x {tr.num_iterations_per_epoch} it, lr {V['lr']}, "
        f"trainable {info.get('n_trainable', info['n_total']):,} / {info['n_total']:,}; ignore label {info['has_ignore_label']}")
    fold_dir.mkdir(parents=True, exist_ok=True)
    _wj(fold_dir / "final_train.json", info)
    t0 = time.time()
    tr.run_training()
    info["wall_hours"] = (time.time() - t0) / 3600
    _wj(fold_dir / "final_train.json", info)
    man = export_stock(fold_dir, info) if STOCK else export_ts(tid, fold_dir, info)
    cleanup(fold_dir)
    LOG(f"[done] {args.run} {ed.name}: {info['wall_hours']:.3f} h")
    return man


def _wj(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    json.dump(F.clean_nan(obj), open(tmp, "w"), indent=1, default=F._default)
    os.replace(tmp, path)


def main():
    if not args.smoke:
        F.be_nice(5)
        C.verify_datasets(args.run, RECIPE)
        if (EXPORT / "DONE.json").is_file():
            LOG(f"[skip] {args.run}: DONE.json present")
            return
        # review B4: the training job takes GPU 1 itself (identity gate, idle wait WITHOUT the lock, then lock + re-check;
        # every wait bounded by the training deadline) and holds the shared flock until it exits (lctsc_infra, copied
        # from WORD-N's wn_common primitives)
        import lctsc_infra as I
        assert args.device == "cuda", "real training runs on GPU 1 only"
        main._gpu_fd = I.gpu_session(I.TRAIN_DEADLINE, log=LOG)
        LOG(f"[gpu] {args.run}: GPU-1 lock taken (GPU 1 idle at the re-check)")
    LOG(f"train start run={args.run} recipe={RECIPE} ({V['label']}) device={args.device} smoke={args.smoke} pid={os.getpid()}")
    out = {"stock": train_one(None)} if STOCK else {str(t): train_one(t) for t in PARTS}
    if STOCK:
        resolved = {"kind": "stock", "dir": str(EXPORT / "stock"), "fold": FOLD_STOCK, "n_classes": RUN["n_classes"], "epochs": EPOCHS}
    else:
        resolved = {"kind": "ts", "parts": {str(t): str(EXPORT / f"part{t}") for t in PARTS},
                    "check": {"variant": RECIPE, "trainable": V["trainable"], "initial_lr": V["lr"]}}
    done = {"model": RUN["model"], "run": args.run, "plan": args.plan, "variant": RECIPE, "spec": V, "resolved": resolved,
            "parts": out, "labelled": C.labelled_of(args.run), "smoke": args.smoke, "finished": F.now()}
    if args.smoke or STOCK or sorted(PARTS) == sorted(RUN["ts_datasets"]):
        _wj(EXPORT / "DONE.json", done)
        LOG(f"[DONE] {args.run} -> {EXPORT / 'DONE.json'}")


if __name__ == "__main__":
    main()
