"""vccor/vista ENGINE: VISTA3D (MONAI bundle 0.5.8) automatic-branch inference and fine-tuning, built from the bundle's
OWN config files (parsed with monai.bundle.ConfigParser) and code (the bundle's scripts/inferer.py Vista3dInferer, loaded
by file path; monai.apps.vista3d transforms + prompt sampler). The bundle's ignite-based Vista3dTrainer/Evaluator are not
used (pytorch-ignite is not installed in .venv); their per-iteration logic is replicated exactly (see train_iteration and
Inference.predict), with two documented compatibility changes:
  * DiceCELoss targets are cast to float ((labels == c).float()): MONAI 1.6 rejects the bool target the bundle passes
    (identical values);
  * the bundle's per-iteration torch.cuda.empty_cache() is not called (memory housekeeping only).

Inference (== configs/inference.json with user_prompt, no point prompt -> 'everything_labels'):
  LoadImaged -> EnsureChannelFirstd -> EnsureTyped(device) -> Spacingd 1.5 mm bilinear -> CropForegroundd (margin 10) ->
  VistaPreTransformd -> ScaleIntensityRanged [-963.8, 1053.7] -> [0, 1] -> Orientationd RAS -> CastToTyped float32;
  Vista3dInferer (128^3 sliding window, sw batch 1; amp autocast on CUDA as the bundle evaluator) with the 117 class
  prompts; VistaPostTransformd (logit < 0 -> 0; argmax + background where every channel <= 0; channel -> VISTA id);
  Invertd (nearest) to the raw CT grid; NaN -> 255.
Training (configs/train.json + configs/train_continual.json; overrides listed in TRAIN_OVERRIDES): CacheDataset of the
  bundle's deterministic transforms + ClassesToIndicesd / RandCropByLabelClassesd (ratios 1 for every class) / Relabeld
  (identity: labels are VISTA global ids already); DataLoader batch 1 shuffled; per iteration sample_prompt_pairs
  (label_set = [0] + everything_labels, max_foreprompt 32, max_backprompt 4, drop_label_prob 0, drop_point_prob 1 =
  automatic branch only), forward with class prompts only, mean over prompts of DiceCELoss(sigmoid), AdamW (bundle
  continual optimizer: torch defaults incl. weight_decay 0.01), constant lr, amp + GradScaler.
Recipes: R1 = every weight on the automatic path trainable (image encoder + automatic decoder + class head; the point
  decoder / point head get no gradient and stay bit-identical); R2 (fallback, probe rule) = VISTA's own automatic-branch
  setting net.set_auto_grad(point_freeze=True): encoder + point branch frozen, automatic decoder + class head trained.
"""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import vista_common as V  # noqa: E402

TRAIN_OVERRIDES = {"drop_label_prob": 0.0, "drop_point_prob": 1.0, "max_foreprompt": V.MAX_FOREPROMPT,
                   "max_backprompt": V.MAX_BACKPROMPT, "max_prompt": None, "num_patches_per_image": 1,
                   "patch_size": [128, 128, 128], "amp": True, "exclude_background": True}


def _bundle_inferer_cls():
    spec = importlib.util.spec_from_file_location("vista_bundle_inferer", str(V.BUNDLE / "scripts/inferer.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Vista3dInferer


def load_network(device, weights=None, recipe_check=None):
    """vista3d132 with the bundle's model.pt (strict) and, if given, a fine-tuned trainable-tensor export on top.
    Returns (net, info)."""
    import torch
    from monai.networks.nets import vista3d132
    net = vista3d132(in_channels=1)
    sd = torch.load(str(V.MODEL_PT), map_location="cpu", weights_only=True)
    net.load_state_dict(sd, strict=True)
    info = {"base_sha256": V.BUNDLE_SHA["models/model.pt"], "override": None}
    if weights is not None:
        ov = torch.load(str(weights), map_location="cpu", weights_only=True)
        own = net.state_dict()
        assert set(ov) <= set(own), f"unknown tensors in {weights}: {sorted(set(ov) - set(own))[:5]}"
        for k, v in ov.items():
            assert own[k].shape == v.shape and own[k].dtype == v.dtype, (k, own[k].shape, v.shape)
        own.update(ov)
        net.load_state_dict(own, strict=True)
        info["override"] = {"path": V.rel(weights), "sha256": V.sha256(weights), "n_tensors": len(ov)}
    return net.to(device), info


class Inference:
    """The bundle's automatic 'segment everything' inference for one model (V0 = no weights; VF = an export)."""

    def __init__(self, device="cuda", weights=None, threads=None):
        import torch
        import monai
        from monai.bundle import ConfigParser
        from monai.transforms import Invertd
        from monai.apps.vista3d.transforms import VistaPostTransformd
        if threads:
            torch.set_num_threads(int(threads))
        self.dev = torch.device("cuda:0" if device in ("cuda", "gpu") else "cpu")
        if self.dev.type == "cuda":
            assert torch.cuda.is_available(), "device cuda requested but CUDA is not available"
        p = ConfigParser()
        p.read_config(str(V.BUNDLE / "configs/inference.json"))
        p["bundle_root"] = str(V.BUNDLE)
        p["device"] = f"$torch.device('{self.dev}')"
        ev = sorted(int(x) for x in p.get_parsed_content("everything_labels"))
        assert ev == V.EVERYTHING, "bundle everything_labels != vista_common.EVERYTHING"
        assert p.get_parsed_content("patch_size") == [128, 128, 128]
        assert p.get_parsed_content("resample_spacing") == [1.5, 1.5, 1.5]
        icfg = p.get("inferer")
        assert icfg["_target_"] == "scripts.inferer.Vista3dInferer", icfg
        monai.utils.set_determinism(seed=V.INFER_SEED)      # inference.json 'initialize'
        self.pre = p.get_parsed_content("preprocessing")
        self.net, self.info = load_network(self.dev, weights)
        self.net.eval()
        self.inferer = _bundle_inferer_cls()(roi_size=[128, 128, 128], overlap=float(p.get_parsed_content("inferer#overlap")),
                                             sw_batch_size=int(p.get_parsed_content("sw_batch_size")),
                                             use_point_window=bool(p.get_parsed_content("use_point_window")))
        self.post = VistaPostTransformd(keys="pred")
        # inference.json passes copy.deepcopy(@preprocessing); MONAI 1.6 checks that the inverted transform objects are the
        # ones that ran (TraceKeys.ID; fork start method) and refuses a deep copy, so the preprocessing object itself is
        # used (the inverse only replays the recorded operations: identical output)
        self.invert = Invertd(keys="pred", transform=self.pre, orig_keys="image", nearest_interp=True, to_tensor=True)
        self.amp = self.dev.type == "cuda"                  # bundle evaluator amp: true (autocast('cuda') is a no-op on CPU)
        self.prompts = list(V.EVERYTHING)

    def predict(self, img_path):
        """Raw-grid VISTA-id uint8 map of one CT (+ timings). Mirrors scripts/evaluator.py user_prompt branch."""
        import numpy as np
        import torch
        t0 = time.time()
        d = self.pre({"image": str(img_path), "label_prompt": list(self.prompts)})
        x = d["image"]
        x = x if x.ndim == 4 else x.unsqueeze(0)
        lp = torch.as_tensor([self.prompts], device=self.dev)[0].unsqueeze(-1)       # [N, 1] (evaluator)
        t1 = time.time()
        with torch.no_grad():
            if self.amp:
                with torch.amp.autocast("cuda"):
                    logits = self.inferer(inputs=x.unsqueeze(0).to(self.dev), network=self.net, point_coords=None,
                                          point_labels=None, class_vector=lp, labels=None, label_set=None)
            else:
                logits = self.inferer(inputs=x.unsqueeze(0).to(self.dev), network=self.net, point_coords=None,
                                      point_labels=None, class_vector=lp, labels=None, label_set=None)
        t2 = time.time()
        d["pred"] = logits[0]
        d["label_prompt"] = lp
        d = self.post(d)
        d = self.invert(d)
        pred = torch.nan_to_num(torch.as_tensor(d["pred"]), nan=float(V.NAN_VALUE))
        seg = np.rint(pred[0].detach().cpu().numpy()).astype(np.int32)
        assert seg.min() >= 0 and seg.max() <= 255, (seg.min(), seg.max())
        seg = seg.astype(np.uint8)
        aff = np.asarray(d["pred"].affine if hasattr(d["pred"], "affine") else np.eye(4), float)
        if self.dev.type == "cuda":
            torch.cuda.empty_cache()
        return seg, {"pre_s": round(t1 - t0, 2), "infer_s": round(t2 - t1, 2), "post_s": round(time.time() - t2, 2),
                     "shape_1p5": list(x.shape[1:]), "n_nan": int((seg == V.NAN_VALUE).sum()), "affine_out": aff.tolist()}


# ---------------------------------------------------------------------------------------------------- training
def train_parser(lr, n_scans, n_iter=V.N_ITER, device="cuda"):
    """Merged bundle training config (train.json + train_continual.json) with the pre-registered overrides."""
    from monai.bundle import ConfigParser
    p = ConfigParser()
    p.read_config([str(V.BUNDLE / "configs/train.json"), str(V.BUNDLE / "configs/train_continual.json")])
    p["bundle_root"] = str(V.BUNDLE)
    p["device"] = f"$torch.device('{device}')"
    assert abs(float(p.get_parsed_content("learning_rate")) - V.LR_A) < 1e-15, "bundle continual lr != 5e-5"
    assert p.get("lr_schedule#activate") is False and p.get("finetune") is True
    p["learning_rate"] = float(lr)
    p["label_mappings"] = {"default": [[c, c] for c in V.EVERYTHING]}
    p["epochs"] = V.epochs_equiv(n_scans, n_iter)
    for k, v in TRAIN_OVERRIDES.items():
        p[k] = v
    return p


def build_training(datalist, lr, recipe, n_iter=V.N_ITER, device="cuda", workers=4, cache_workers=4, net=None):
    """(net, loader, optimizer, loss_fn, scaler, label_set, meta). datalist = [{'image': raw CT, 'label': VISTA-id map}]."""
    import torch
    import monai
    from monai.data import CacheDataset, DataLoader
    p = train_parser(lr, len(datalist), n_iter, device)
    monai.utils.set_determinism(seed=V.TRAIN_SEED)            # train.json 'initialize'
    tfm = p.get_parsed_content("train#preprocessing")
    loss_fn = p.get_parsed_content("loss")
    label_set = [int(x) for x in p.get_parsed_content("label_set")]
    assert label_set == [0] + V.EVERYTHING, label_set[:5]
    oc = p.get("optimizer")
    assert oc == {"_target_": "torch.optim.AdamW", "lr": "@learning_rate", "params": "$@network.parameters()"}, oc
    lc = p.get("loss")
    assert lc["_target_"] == "DiceCELoss" and lc["sigmoid"] is True and lc["squared_pred"] is True, lc
    if net is None:
        net, _ = load_network(device)
    set_recipe(net, recipe)
    opt = torch.optim.AdamW(net.parameters(), lr=float(lr))   # == bundle continual optimizer (no weight_decay key)
    ds = CacheDataset(data=datalist, transform=tfm, cache_rate=1.0, hash_as_key=True, num_workers=cache_workers,
                      progress=False)
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=workers, pin_memory=str(device).startswith("cuda"),
                        persistent_workers=workers > 0)
    scaler = torch.amp.GradScaler("cuda") if str(device).startswith("cuda") else None
    meta = {"lr": float(lr), "recipe": recipe, "n_iter": int(n_iter), "epochs_equiv": int(p["epochs"]),
            "label_set_n": len(label_set), "loss": {k: v for k, v in lc.items()},
            "optimizer": {"type": "torch.optim.AdamW", "lr": float(lr), "defaults": {k: v for k, v in opt.defaults.items()
                                                                                       if k in ("betas", "eps", "weight_decay", "amsgrad")}},
            "overrides": TRAIN_OVERRIDES, "n_scans": len(datalist), "seed": V.TRAIN_SEED,
            "trainable_params": int(sum(x.numel() for x in net.parameters() if x.requires_grad))}
    return net, loader, opt, loss_fn, scaler, label_set, meta


def set_recipe(net, recipe):
    """R1: automatic path trainable (point decoder + point head requires_grad False; they get no gradient anyway).
    R2: VISTA's automatic-branch setting (encoder + point branch frozen)."""
    for prm in net.parameters():
        prm.requires_grad = True
    if recipe == "R1":
        for m in (net.image_encoder.up_layers, net.point_head):
            for prm in m.parameters():
                prm.requires_grad = False
    elif recipe == "R2":
        net.set_auto_grad(auto_freeze=False, point_freeze=True)
        assert not any(x.requires_grad for x in net.image_encoder.encoder.parameters())
        assert all(x.requires_grad for x in net.image_encoder.up_layers_auto.parameters())
        assert all(x.requires_grad for x in net.class_head.parameters())
    else:
        raise ValueError(recipe)
    return net


def train_iteration(net, batch, opt, loss_fn, scaler, label_set, device):
    """One Vista3dTrainer._iteration (bundle scripts/trainer.py) with class prompts only. Returns (loss, n_prompts)."""
    import torch
    from monai.apps.vista3d.sampler import sample_prompt_pairs
    inputs = batch["image"].to(device)
    labels = batch["label"].to(device)
    label_prompt, point, point_label, prompt_class = sample_prompt_pairs(
        labels, label_set, max_point=5, max_prompt=TRAIN_OVERRIDES["max_prompt"],
        max_backprompt=TRAIN_OVERRIDES["max_backprompt"], max_foreprompt=TRAIN_OVERRIDES["max_foreprompt"],
        drop_label_prob=TRAIN_OVERRIDES["drop_label_prob"], drop_point_prob=TRAIN_OVERRIDES["drop_point_prob"],
        include_background=not TRAIN_OVERRIDES["exclude_background"])
    if label_prompt is None:
        return None, 0
    point, point_label = None, None                           # drop_point_prob = 1: automatic branch only

    def _loss():
        out = net(input_images=inputs, point_coords=point, point_labels=point_label, class_vector=label_prompt)
        loss = torch.tensor(0.0, device=device)
        for i in range(len(prompt_class)):
            loss = loss + loss_fn(out[[i]].float(), (labels == prompt_class[i]).float())
        return loss / max(len(prompt_class), 1)

    net.train()
    opt.zero_grad(set_to_none=False)
    if scaler is not None:
        with torch.amp.autocast("cuda"):
            loss = _loss()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    else:
        loss = _loss()
        loss.backward()
        opt.step()
    return float(loss.detach().item()), int(len(prompt_class))


def infinite(loader):
    while True:
        for b in loader:
            yield b


def export_trainable(net, recipe, path):
    """Save the trainable tensors (fp32, cpu) and verify every frozen tensor + buffer is bit-identical to model.pt."""
    import torch
    base = torch.load(str(V.MODEL_PT), map_location="cpu", weights_only=True)
    sd = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    trainable = {n for n, prm in net.named_parameters() if prm.requires_grad}
    out = {k: sd[k].clone().contiguous() for k in sorted(trainable)}
    frozen_same = all(torch.equal(sd[k], base[k]) for k in sd if k not in trainable)
    changed = sum(1 for k in trainable if not torch.equal(sd[k], base[k]))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    torch.save(out, str(tmp))
    os.replace(tmp, path)
    return {"n_trainable_tensors": len(out), "n_changed_tensors": int(changed), "frozen_and_buffers_identical": bool(frozen_same),
            "n_params": int(sum(v.numel() for v in out.values())), "bytes": int(path.stat().st_size),
            "sha256": V.sha256(path), "recipe": recipe}


def gpu_mem_gb():
    import torch
    if not torch.cuda.is_available():
        return None
    return {"max_allocated_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 3),
            "max_reserved_gb": round(torch.cuda.max_memory_reserved() / 2 ** 30, 3)}


def datalist_for(run):
    """Training datalist of a fine-tuned run: composite labels for the labelled scans, V0's own raw predictions for pool
    scans (VS; ssft B rule). Paths only (no GT is read here)."""
    r = V.RUNS[run]
    ds = r["ds"]
    lab, pool = V.train_scans(run)
    dl = [{"image": str(V.raw_img(ds, c)), "label": str(V.label_path(ds, c)), "cid": c, "kind": "gt_composite"} for c in lab]
    dl += [{"image": str(V.raw_img(ds, c)), "label": str(V.v0_raw(ds, c)), "cid": c, "kind": "v0_pseudo"} for c in pool]
    return dl


def resolve_recipe(log=print):
    """(recipe, n_iter, probe sha256) from the probe file written by probe_vista.py (pre-registered rule)."""
    if not V.PROBE.is_file():
        raise SystemExit(f"[REFUSE] {V.rel(V.PROBE)} missing: run the probe job first")
    pr = json.load(open(V.PROBE))
    rec, n = pr["decision"]["recipe"], int(pr["decision"]["n_iter"])
    assert rec in ("R1", "R2") and 100 <= n <= V.N_ITER
    log(f"recipe {rec}, n_iter {n} (probe {V.sha256(V.PROBE)[:16]})")
    return rec, n, V.sha256(V.PROBE)


def resolve_lr(run, log=print):
    """lr of a run: A / B fixed; 'sel' = the dev18 selection (selection/lr_selection.json, pre-registered rule)."""
    lr = V.RUNS[run]["lr"]
    if lr == "A":
        return V.LR_A, None
    if lr == "B":
        return V.LR_B, None
    f = V.SELECT / "lr_selection.json"
    if not f.is_file():
        raise SystemExit(f"[REFUSE] {V.rel(f)} missing: the dev18 lr selection has not been made")
    s = json.load(open(f))
    vals = s["dev18_mean_macro_dice"]
    a, b = vals["flare_s1337_K4_lrA"], vals["flare_s1337_K4_lrB"]
    want = "flare_s1337_K4_lrB" if b > a else "flare_s1337_K4_lrA"   # tie -> bundle default (lrA)
    assert s["selected_run"] == want, (s["selected_run"], want)
    lr = V.LR_B if want.endswith("lrB") else V.LR_A
    log(f"lr {lr:g} = dev18 selection ({want}; lrA {a:.3f} vs lrB {b:.3f}; selection {V.sha256(f)[:16]})")
    return lr, V.sha256(f)
