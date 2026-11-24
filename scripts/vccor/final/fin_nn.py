"""vccor/final nnU-Net DATASET helpers shared by bsl_prep.py (plan B) and word_prep.py (plan C): == the frozen
xdata prep_xdata.nnunet_dataset / peft prep_peft.build machinery (TS's plans moved onto the dataset with nnU-Net's own
move_plans_between_datasets, the network / patch / spacing / normalisation asserted identical to TS's, 3d_fullres
preprocessing). nnU-Net is imported inside the functions, after set_env(). No import-time side effects beyond sys.path.
"""
import json
import os
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import final_common as F  # noqa: E402

PLAN_KEYS = ("patch_size", "spacing", "UNet_class_name", "UNet_base_num_features", "n_conv_per_stage_encoder",
             "n_conv_per_stage_decoder", "pool_op_kernel_sizes", "conv_kernel_sizes", "unet_max_num_features",
             "normalization_schemes", "use_mask_for_norm", "batch_size", "resampling_fn_data_kwargs", "resampling_fn_seg_kwargs",
             "batch_dice")


def set_env(nn_root):
    if "nnunetv2" in sys.modules and os.environ.get("nnUNet_preprocessed") != str(Path(nn_root) / "nnUNet_preprocessed"):
        raise RuntimeError("nnunetv2 imported before set_env")
    os.environ["nnUNet_raw"] = str(Path(nn_root) / "nnUNet_raw")
    os.environ["nnUNet_preprocessed"] = str(Path(nn_root) / "nnUNet_preprocessed")
    os.environ["nnUNet_results"] = str(Path(nn_root) / "results_prep")


def wj(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    json.dump(F.clean_nan(obj), open(tmp, "w"), indent=1, default=F._default)
    os.replace(tmp, path)


def save_nii(arr, affine, path):
    import nibabel as nib
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name.replace('.nii.gz', '')}.tmp{os.getpid()}.nii.gz")
    nib.save(nib.Nifti1Image(arr, affine), str(tmp))
    os.replace(tmp, path)
    return F.sha256(path)


def expose_ts_source(nn_root, tid):
    """TS's plans.json / dataset.json exposed as nnU-Net's 'source' dataset (== xdata / peft prep)."""
    import tsft_full_common as Q
    src = Path(nn_root) / "nnUNet_preprocessed" / Q.TS_DS[tid]
    src.mkdir(parents=True, exist_ok=True)
    for s, t in (("plans.json", "nnUNetPlans.json"), ("dataset.json", "dataset.json")):
        if not (src / t).is_file():
            shutil.copy(Q.TS_MODEL[tid] / s, src / t)


def plans_and_preprocess(nn_root, did, dsn, tid, plans_id, n_cases):
    import tsft_full_common as Q
    from batchgenerators.utilities.file_and_folder_operations import load_json
    from nnunetv2.experiment_planning.plan_and_preprocess_api import extract_fingerprints, preprocess_dataset
    from nnunetv2.experiment_planning.plans_for_pretraining.move_plans_between_datasets import move_plans_between_datasets
    nn_root = Path(nn_root)
    pre = nn_root / "nnUNet_preprocessed" / dsn
    pre.mkdir(parents=True, exist_ok=True)
    if not (pre / "dataset_fingerprint.json").is_file():
        extract_fingerprints([did], num_processes=2, check_dataset_integrity=True, clean=True, verbose=False)
    if not (pre / "dataset.json").is_file():
        shutil.copy(nn_root / "nnUNet_raw" / dsn / "dataset.json", pre / "dataset.json")
    expose_ts_source(nn_root, tid)
    pf = pre / f"{plans_id}.json"
    if not pf.is_file():
        move_plans_between_datasets(tid, did, "nnUNetPlans", plans_id)
    p, s = load_json(pf), load_json(Q.TS_MODEL[tid] / "plans.json")
    for k in PLAN_KEYS:
        assert p["configurations"][Q.CONFIG].get(k) == s["configurations"][Q.CONFIG].get(k), (dsn, k)
    assert p["foreground_intensity_properties_per_channel"] == s["foreground_intensity_properties_per_channel"]
    data_dir = pre / p["configurations"][Q.CONFIG]["data_identifier"]
    n = len(list(data_dir.glob("*.pkl"))) if data_dir.exists() else 0
    if n < n_cases:
        preprocess_dataset(did, plans_id, [Q.CONFIG], [2], verbose=False)
        n = len(list(data_dir.glob("*.pkl")))
    assert n == n_cases, f"{dsn}: expected {n_cases} preprocessed cases, found {n}"
    files = sorted(x for x in data_dir.iterdir() if x.is_file())
    rd = nn_root / "nnUNet_raw" / dsn
    return {"dataset": dsn, "id": did, "plans": str(pf), "plans_sha256": F.sha256(pf), "data_dir": str(data_dir), "n_cases": n,
            "data_files": {x.name: F.sha256(x) for x in files},
            "raw_files": {str(x): F.sha256(x) for x in sorted((rd / "imagesTr").glob("*.nii.gz")) + sorted((rd / "labelsTr").glob("*.nii.gz"))}}


def ts_part_dataset(nn_root, did, dsn, tid, cases, description):
    """cases = [(case name, image path, image sha, label path, label sha)]: a TS-part fine-tuning dataset (== xdata
    prep_xdata.nnunet_dataset): images = the captured 1.5 mm TS inputs, labels = the part's composite targets, TS's own
    label space and plans."""
    import tsft_full_common as Q
    nn_root = Path(nn_root)
    rd = nn_root / "nnUNet_raw" / dsn
    (rd / "imagesTr").mkdir(parents=True, exist_ok=True)
    (rd / "labelsTr").mkdir(parents=True, exist_ok=True)
    for cn, si, hi, sl, hl in cases:
        assert F.sha256(si) == hi, f"{si}: image changed since its manifest"
        assert F.sha256(sl) == hl, f"{sl}: label changed since its manifest"
        fi, fl = rd / "imagesTr" / f"{cn}_0000.nii.gz", rd / "labelsTr" / f"{cn}.nii.gz"
        if not (fi.is_file() and F.sha256(fi) == hi):
            shutil.copyfile(si, fi)
        if not (fl.is_file() and F.sha256(fl) == hl):
            shutil.copyfile(sl, fl)
    tsd = Q.ts_dataset_json(tid)
    dj = {"name": dsn, "description": description, "channel_names": {"0": "CT"}, "labels": tsd["labels"],
          "numTraining": len(cases), "file_ending": ".nii.gz", "overwrite_image_reader_writer": "NibabelIOWithReorient"}
    json.dump(dj, open(rd / "dataset.json", "w"), indent=1)
    return plans_and_preprocess(nn_root, did, dsn, tid, Q.PLANS_ID[tid], len(cases))


def files_of(info):
    """{path: sha256} of every produced file of a datasets record (symlinked raw images excluded)."""
    files = {}
    for k, r in info.items():
        if k.startswith("_"):
            continue
        files.update({p: h for p, h in r["raw_files"].items() if not Path(p).is_symlink()})
        files[r["plans"]] = r["plans_sha256"]
        for f, h in r["data_files"].items():
            files[f"{r['data_dir']}/{f}"] = h
    return files


def ts_capture(img, d, threads=6):
    """TS-full's own CPU pipeline (frozen ts_pipeline.run_scan, ORIGINAL weights): the exact 1.5 mm TS input image + all five
    part predictions + the merged raw output -> d (== xdata prep capture). Returns the run record."""
    sys.path.append(str(F.REPO / "scripts/vccor/tsft_full"))
    import ts_pipeline as TP
    if not getattr(ts_capture, "_init", False):
        TP.init("cpu", threads)
        ts_capture._init = True
    r = TP.run_scan(img, Path(d) / "raw_total.nii.gz", capture_dir=d, capture_parts=(291, 292, 293, 294, 295),
                    capture_image=True, nr_thr_resamp=1, nr_thr_saving=1)
    wj(Path(d) / "capture.json", {"run": r, "torch_threads": threads, "device": "cpu", "finished": F.now()})
    return r


def capture_ok(d):
    d = Path(d)
    return (d / "capture.json").is_file() and (d / "image_1p5.nii.gz").is_file() and \
        all((d / f"part_{t}.nii.gz").is_file() for t in (291, 292, 293, 294, 295))


def ts_grid_check(img_path, cap_dir):
    """(cap nifti, img_orig, img_in, img_rsp): the recomputed 1.5 mm image must equal the captured one (== xdata ts_grid)."""
    import nibabel as nib
    import numpy as np
    from totalsegmentator.alignment import as_closest_canonical
    from totalsegmentator.resampling import change_spacing
    img_orig = nib.load(str(img_path))
    img_in = as_closest_canonical(nib.Nifti1Image(img_orig.get_fdata(), img_orig.affine))
    img_rsp = change_spacing(img_in, [1.5] * 3, order=3, dtype=np.int32, nr_cpus=1, use_gpu=False)
    cap = nib.load(str(Path(cap_dir) / "image_1p5.nii.gz"))
    assert cap.shape == img_rsp.shape and np.array_equal(np.asarray(cap.dataobj), np.asarray(img_rsp.dataobj)), \
        f"{img_path}: recomputed 1.5 mm image != the image TS fed nnU-Net"
    assert np.allclose(cap.affine, img_rsp.affine, atol=1e-4)
    return cap, img_orig, img_in, img_rsp
