"""TotalSegmentator-full inference with swappable part weights (vccor/tsft_full).

run_scan() calls TotalSegmentator 2.13.0's OWN pipeline function totalsegmentator.nnunet.nnUNet_predict_image with
exactly the arguments totalsegmentator.python_api.totalsegmentator(ml=True, fast=False, task='total') passes for the
'total' task (task ids [291..295], resample 1.5, trainer nnUNetTrainerNoMirroring, folds [0], tta False, crop None,
nr_thr_resamp 1). TS itself therefore does: closest-canonical -> change_spacing 1.5 mm (order 3, int32) -> per part
nnUNetv2_predict (tile step 0.8, gaussian, no mirroring, checkpoint_final) -> merge in order 291..295 (later
non-background overwrites) -> change_spacing back (order 0, force_affine) -> undo canonical.

The ONLY replaced function is the per-part call totalsegmentator.nnunet.nnUNetv2_predict. The replacement has the same
signature, device handling and nnUNetPredictor arguments as TS's own, and adds three things:
  * REDIRECT {tid: model folder}: a fine-tuned part is loaded from our folder (TS's own plans.json + dataset.json, the
    fine-tuned checkpoint as fold_0/checkpoint_final.pth); every other part from ~/.totalsegmentator (TS's weights).
  * predictor reuse across scans (same constructor arguments; initialisation is deterministic -> identical outputs).
  * CAPTURE / REPLAY of per-part outputs keyed by a sha256 of the 1.5 mm input image array (+ affine): the F0 run
    captures the frozen parts 292/294/295 once per scan; variant runs replay them byte-for-byte instead of recomputing
    (a key mismatch falls back to running the model and is recorded as a miss).
Python-API side effects that do not touch the segmentation (usage-statistics upload, prediction counter in
~/.totalsegmentator/config.json) are skipped on purpose. Equivalence with the stored TS-full output is tested by
smoke_test.py (CPU, original weights, all five parts).
"""
import hashlib
import multiprocessing
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.dont_write_bytecode = True

TASK_IDS = [291, 292, 293, 294, 295]
_STATE = {"redirect": {}, "replay": {}, "capture_dir": None, "capture_parts": (), "threads": 4, "calls": [],
          "predictors": {}, "image_key": None}


def _setup_ts_env():
    """== python_api.totalsegmentator(): setup_nnunet() before totalsegmentator.nnunet / nnunetv2 are imported."""
    from totalsegmentator.config import setup_nnunet
    if "nnunetv2" in sys.modules and os.environ.get("nnUNet_results") != str(Path.home() / ".totalsegmentator/nnunet/results"):
        raise RuntimeError("nnunetv2 was imported before the TotalSegmentator nnU-Net env was set")
    setup_nnunet()


def image_key(path):
    import nibabel as nib
    import numpy as np
    im = nib.load(str(path))
    a = np.ascontiguousarray(np.asarray(im.dataobj))
    h = hashlib.sha256()
    h.update(f"{a.dtype}|{a.shape}|".encode())
    h.update(a.tobytes())
    h.update(np.round(np.asarray(im.affine, float), 6).tobytes())
    return h.hexdigest()


def _patched_nnUNetv2_predict(dir_in, dir_out, task_id, model="3d_fullres", folds=None, trainer="nnUNetTrainer",
                              tta=False, num_threads_preprocessing=3, num_threads_nifti_save=2, plans="nnUNetPlans",
                              device="cuda", quiet=False, step_size=0.5, save_probabilities_path=None):
    """Drop-in for totalsegmentator.nnunet.nnUNetv2_predict (same signature and predictor settings)."""
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.utilities.file_path_utilities import get_output_folder
    assert save_probabilities_path is None and tta is False and model == "3d_fullres" and list(folds) == [0], \
        (save_probabilities_path, tta, model, folds)
    t0 = time.time()
    img = Path(dir_in) / "s01_0000.nii.gz"
    out = Path(dir_out) / "s01.nii.gz"
    key = image_key(img)
    if _STATE["image_key"] is None:
        _STATE["image_key"] = key
    assert key == _STATE["image_key"], "the 1.5 mm input image changed between parts of one scan"
    rec = {"tid": int(task_id), "step_size": float(step_size), "image_key16": key[:16]}
    rp = _STATE["replay"].get(int(task_id))
    if rp is not None:
        if rp["key"] == key and Path(rp["path"]).is_file():
            shutil.copyfile(rp["path"], out)
            rec.update(source="replay", path=str(rp["path"]), sec=round(time.time() - t0, 2))
            _STATE["calls"].append(rec)
            _capture(int(task_id), out, img, key)
            return
        rec["replay_miss"] = True
    model_folder = _STATE["redirect"].get(int(task_id)) or get_output_folder(task_id, trainer, plans, model)
    # device handling identical to totalsegmentator.nnunet.nnUNetv2_predict
    if device == "cpu":
        torch.set_num_threads(multiprocessing.cpu_count())          # cpu_count is capped in init()
        dev = torch.device("cpu")
    elif device == "cuda":
        torch.set_num_threads(1)
        dev = torch.device("cuda")
    elif isinstance(device, torch.device):
        torch.set_num_threads(1)
        dev = device
    else:
        dev = torch.device("mps")
    pk = (str(model_folder), str(dev), float(step_size))
    if pk not in _STATE["predictors"]:
        p = nnUNetPredictor(tile_step_size=step_size, use_gaussian=True, use_mirroring=bool(tta),  # TS: not disable_tta
                            perform_everything_on_device=True, device=dev, verbose=False, verbose_preprocessing=False,
                            allow_tqdm=not quiet)
        p.initialize_from_trained_model_folder(str(model_folder), use_folds=folds, checkpoint_name="checkpoint_final.pth")
        _STATE["predictors"][pk] = p
    _STATE["predictors"][pk].predict_from_files(
        str(dir_in), str(dir_out), save_probabilities=False, overwrite=True,
        num_processes_preprocessing=num_threads_preprocessing, num_processes_segmentation_export=num_threads_nifti_save,
        folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)
    rec.update(source="redirect" if int(task_id) in _STATE["redirect"] else "ts", model_folder=str(model_folder),
               sec=round(time.time() - t0, 2))
    _STATE["calls"].append(rec)
    _capture(int(task_id), out, img, key)


def _capture(tid, out, img, key):
    cd = _STATE["capture_dir"]
    if cd is None:
        return
    cd = Path(cd)
    cd.mkdir(parents=True, exist_ok=True)
    if tid in _STATE["capture_parts"]:
        tmp = cd / f".part_{tid}.tmp{os.getpid()}.nii.gz"
        shutil.copyfile(out, tmp)
        os.replace(tmp, cd / f"part_{tid}.nii.gz")
    if _STATE.get("capture_image") and not (cd / "image_1p5.nii.gz").is_file():
        tmp = cd / f".image.tmp{os.getpid()}.nii.gz"
        shutil.copyfile(img, tmp)
        os.replace(tmp, cd / "image_1p5.nii.gz")
    (cd / "image_key.txt").write_text(key + "\n")


def init(device="cpu", threads=4, redirect=None):
    """device: 'cpu' or 'gpu' (python_api semantics). threads caps torch on CPU (TS uses cpu_count())."""
    _STATE["threads"] = int(threads)
    multiprocessing.cpu_count = lambda: int(threads)               # TS: torch.set_num_threads(cpu_count()) on CPU
    _setup_ts_env()
    import torch
    torch.set_num_threads(int(threads))
    import totalsegmentator.nnunet as tsn
    tsn.nnUNetv2_predict = _patched_nnUNetv2_predict               # the only replaced function
    _STATE["redirect"] = {int(k): str(v) for k, v in (redirect or {}).items()}
    if device == "gpu":
        assert torch.cuda.is_available(), "device gpu requested but CUDA is not available"
        dev = torch.device("cuda:0")                               # == python_api.select_device('gpu')
    else:
        dev = "cpu"                                                # == python_api.select_device('cpu')
    _STATE["device"] = dev
    return dev


def run_scan(img_path, out_path, replay=None, capture_dir=None, capture_parts=(), capture_image=False,
             nr_thr_resamp=1, nr_thr_saving=1):
    """One scan through TS's nnUNet_predict_image ('total', full resolution). Writes the raw-frame multilabel NIfTI
    (TS 'total' ids) to out_path atomically and returns a record of what ran for each part."""
    import torch
    import totalsegmentator.nnunet as tsn
    assert tsn.nnUNetv2_predict is _patched_nnUNetv2_predict, "call init() first"
    _STATE["replay"] = {int(k): v for k, v in (replay or {}).items()}
    _STATE["capture_dir"] = capture_dir
    _STATE["capture_parts"] = tuple(int(t) for t in capture_parts)
    _STATE["capture_image"] = capture_image
    _STATE["calls"] = []
    _STATE["image_key"] = None
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name("." + out_path.name.replace(".nii.gz", "") + f"_tmp{os.getpid()}.nii.gz")
    init_cudnn, init_threads = torch.backends.cudnn.benchmark, torch.get_num_threads()
    t0 = time.time()
    tsn.nnUNet_predict_image(
        Path(img_path), tmp, TASK_IDS, model="3d_fullres", folds=[0], trainer="nnUNetTrainerNoMirroring", tta=False,
        multilabel_image=True, resample=1.5, crop=None, crop_path=tmp.parent, task_name="total", nora_tag="None",
        preview=False, nr_threads_resampling=nr_thr_resamp, nr_threads_saving=nr_thr_saving, force_split=False,
        crop_addon=[3, 3, 3], roi_subset=None, output_type="nifti", statistics=False, quiet=True, verbose=False,
        test=0, skip_saving=False, device=_STATE["device"], exclude_masks_at_border=True, no_derived_masks=False,
        v1_order=False, stats_aggregation="mean", remove_small_blobs=False, normalized_intensities=False,
        nnunet_resampling=False, save_probabilities=None, cascade=None, remove_outside_mask=None,
        remove_outside_dilation=None)
    torch.backends.cudnn.benchmark = init_cudnn                     # == python_api: restore torch settings
    torch.set_num_threads(init_threads)
    if not tmp.is_file():
        raise RuntimeError(f"TotalSegmentator wrote no output for {img_path}")
    os.replace(tmp, out_path)
    calls = list(_STATE["calls"])
    assert [c["tid"] for c in calls] == TASK_IDS, f"parts ran out of order / missing: {[c['tid'] for c in calls]}"
    return {"wall_s": round(time.time() - t0, 2), "parts": calls, "image_key": _STATE["image_key"]}


def tmpdir_note():
    return tempfile.gettempdir()
