"""vccor/final WORD-N PREDICT (GPU job; idempotent, resumable per scan): one arm on the included WORD validation scans.
NO label is read or listed. Mechanics == scripts/vccor/final/predict_final.py / flaret50/t50_predict.py (frozen):

  'ts' arms      TotalSegmentator 2.13.0's own 'total' pipeline through the FROZEN scripts/vccor/tsft_full/ts_pipeline.py
                 (1.5 mm, no TTA, TS's merge order 291..295); F1b_dX redirects part 291 to its A1 export (checkpoint
                 sha256 == export.json, registered checks, full 50-epoch non-smoke run); every other part is replayed
                 byte-for-byte from F0's frozen-part cache (pred/F0/cache_frozen/<cid>/) when the 1.5 mm image key matches,
                 else run with TS's own weights (recorded as a replay miss). F0 CAPTURES parts 292-295; F1b needs F0 DONE.
  'nnunet' arms  STOCK_dX: nnU-Net's nnUNetPredictor (tile 0.5, gaussian, mirroring OFF) -> WORD's 15 analysis ids, fold 'all'.
  'vista' arms   VISTA3D 0.5.8 automatic 'segment everything' branch through the FROZEN scripts/vccor/vista/vista_engine.py
                 (V0 = the bundle's weights; VF runs = their WORD export, weights sha256 == DONE.json); the bundle is checked.
Per scan: pred/<arm>/raw/word_<cid>.nii.gz (affine / shape == the image, asserted) and rec/<cid>.json; DONE.json last
(per-scan raw sha256, model sha256). Every image is re-hashed against intake/images_record.json before use.

Modes:
  real   (no --smoke; RUNNER ONLY, launched by wn_run.py): GPU 1 only (CUDA_VISIBLE_DEVICES must be GPU 1's UUID; the job takes
         the shared flock /home/user/Documents/Handpose_project/runs/method_pilot3/gpu1.lock itself and waits, cutoff-aware,
         for the lock and for GPU 1 to be idle); refuses an arm the runner declared NOT RUN and a tier-2 arm without
         VSMOKE_PASS; only after the freeze (code, records, the arm's model, TotalSegmentator's weights / the
         VISTA3D bundle unchanged; package versions == the freeze's) and the gate_osf_uploaded.json record; never
         after the look; on wn_common.included_ids() with / kept >= 25 GiB free; started only before the cutoff
         (Thu 2026-11-19 18:00 KST), and DONE.json is never written at or after the cutoff (the arm is then NOT RUN).
  smoke  (--smoke --images DIR --root DIR --ids ...): unit tests / CPU smoke on stand-in images, writes only under --root;
         refuses any WORD validation id and any --images folder that is the WORD image folder.

usage:  runner only (wn_run.py launches '... wn_predict.py --arm <arm> --device gpu' with the GPU-1 environment); unit tests
        use --smoke.

Version of 2026-11-13.
"""
import argparse
import json
import os
import resource
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import wn_common as T  # noqa: E402


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=[a for a in T.RUN_ORDER if T.arms()[a]["kind"] != "cpu"])
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--nr_thr_saving", type=int, default=2)
    ap.add_argument("--ids", default=None, help="smoke only: comma list of scans")
    ap.add_argument("--smoke", action="store_true", help="unit tests: stand-in images, writes only under --root")
    ap.add_argument("--root", default=None, help="smoke only: prediction root")
    ap.add_argument("--images", default=None, help="smoke only: dir with word_<cid>.nii.gz stand-in images")
    a = ap.parse_args(argv)
    assert a.threads <= 6
    ids = [c.strip() for c in (a.ids or "").split(",") if c.strip()]
    if a.smoke:
        assert ids and a.root and a.images, "smoke needs --ids, --root and --images"
        assert not any(c in T.VAL_IDS for c in ids), "smoke never runs on a WORD validation id"
        img = Path(a.images).resolve()
        real_img = Path(T.W.IMAGES).resolve()
        assert img != real_img and real_img not in img.parents, "smoke never reads the WORD image folder"
    else:
        assert not (a.ids or a.root or a.images), "--ids / --root / --images are smoke options"
        assert a.device == "gpu", "real predictions run on GPU 1 only"
    return a


def check_images(ids, smoke):
    if smoke:
        return {c: None for c in ids}
    rec = json.load(open(T.IMAGES_RECORD))["images"]
    out = {}
    for c in ids:
        h = T.sha256(T.image_path(c))
        if h != rec[c]["sha256"]:
            raise T.Refuse(f"{c}: image sha256 {h[:16]} differs from images_record.json")
        out[c] = h
    return out


def replay_for(cache_root, cid, frozen):
    d = Path(cache_root) / cid
    kf = d / "image_key.txt"
    if not kf.is_file():
        return {}
    key = kf.read_text().strip()
    return {t: {"key": key, "path": str(d / f"part_{t}.nii.gz")} for t in frozen if (d / f"part_{t}.nii.gz").is_file()}


def done_ok(pd, cid, shas):
    try:
        rec = json.load(open(pd / "rec" / f"{cid}.json"))
        return (rec["model_sha256"] == {str(k): v for k, v in shas.items()}
                and rec["raw_sha256"] == T.sha256(pd / "raw" / f"word_{cid}.nii.gz"))
    except (OSError, KeyError, ValueError):
        return False


def main(argv=None):
    a = parse(argv)
    arm = a.arm
    real = not a.smoke
    vers = T.check_versions()
    if real:
        T.kst_gate()
        if T.look_made():
            raise T.Refuse("a look exists: no prediction after the look")
        fz = T.freeze_ok()
        vers = T.check_versions(fz.get("versions"))
        T.check_frozen_model(arm)
        T.gate_uploaded()
        T.arm_allowed(arm)
        if not T.before_cutoff():
            raise T.Refuse("cutoff passed (Thu 2026-11-19 18:00 KST): no prediction is started", code=T.CUTOFF_CODE)
        T.wait_disk(0.3, f"prediction {arm}")
        T.gpu_identity_gate()
        root, ids, img_root = T.PRED, T.included_ids(), None
    else:
        root, ids, img_root = Path(a.root), [c.strip() for c in a.ids.split(",") if c.strip()], Path(a.images)
    image = (lambda c: Path(img_root) / f"word_{c}.nii.gz") if img_root else T.image_path
    kind, R = T.resolve_model(arm, strict=real)
    pd = T.pred_dir(arm, root)
    if (pd / "DONE.json").is_file() and real:
        print(f"{arm}: DONE exists ({pd / 'DONE.json'}); nothing to do")
        return 0
    capture = T.CAPTURE if arm == "F0" else ()
    f0_cache = T.pred_dir("F0", root) / "cache_frozen"
    if kind == "ts" and arm != "F0" and not (T.pred_dir("F0", root) / "DONE.json").is_file():
        raise T.Refuse(f"{arm}: F0 must be DONE first (its frozen-part cache feeds the replay)")
    img_sha = check_images(ids, a.smoke)
    lock_fd = None
    if real:
        lock_fd = T.gpu_session(log=print)
        if T.declared_not_run(arm):
            T.release_gpu_lock(lock_fd)
            raise T.Refuse(f"{arm} was declared NOT RUN while waiting for the GPU")
    try:
        return _predict(a, arm, real, vers, root, ids, image, kind, R, pd, capture, f0_cache, img_sha)
    finally:
        T.release_gpu_lock(lock_fd)


def _predict(a, arm, real, vers, root, ids, image, kind, R, pd, capture, f0_cache, img_sha):
    for sub in ("raw", "rec"):
        (pd / sub).mkdir(parents=True, exist_ok=True)
    logp = pd / "predict.log"
    log = (lambda m: T.log_event(m, logp))
    import nibabel as nib
    import numpy as np
    import torch
    if a.device == "cpu":
        torch.set_num_threads(a.threads)
    if kind == "ts":
        sys.path.append(str(T.REPO / "scripts/vccor/tsft_full"))
        import ts_pipeline as TP
        dev = TP.init(a.device, a.threads, redirect=R["redirect"])
        frozen = [t for t in (291, 292, 293, 294, 295) if t not in R["redirect"]]
    elif kind == "nnunet":
        os.environ.setdefault("nnUNet_raw", "unused")
        os.environ.setdefault("nnUNet_preprocessed", "unused")
        os.environ.setdefault("nnUNet_results", "unused")
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        dev = torch.device("cuda:0") if a.device == "gpu" else torch.device("cpu")
        pr = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False, perform_everything_on_device=True,
                             device=dev, verbose=False, verbose_preprocessing=False, allow_tqdm=False)
        pr.initialize_from_trained_model_folder(R["dir"], use_folds=(R["fold"],), checkpoint_name="checkpoint_final.pth")
        if not (pr.allowed_mirroring_axes is None and pr.use_mirroring is False):
            raise T.Refuse(f"{arm}: the checkpoint must predict without mirroring")
    else:
        sys.path.insert(0, str(T.REPO / "scripts/vccor/vista"))
        import vista_common as V
        import vista_engine as E
        if real:
            V.check_bundle(log)
        eng = E.Inference("cuda" if a.device == "gpu" else "cpu", weights=R["weights"],
                          threads=a.threads if a.device == "cpu" else None)
        dev = eng.dev
    cuda = a.device == "gpu" and torch.cuda.is_available()
    log(f"predict start arm={arm} kind={kind} n={len(ids)} device={dev} capture={capture} "
        f"mode={'smoke' if a.smoke else 'real'} -> {pd} pid={os.getpid()}")
    t_all, n_new, misses, peak_all = time.time(), 0, 0, 0
    for i, cid in enumerate(ids):
        if done_ok(pd, cid, R["shas"]):
            log(f"  [{i + 1}/{len(ids)}] {cid} [skip] done")
            continue
        img = image(cid)
        rawp, t0 = pd / "raw" / f"word_{cid}.nii.gz", time.time()
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        extra = {}
        if kind == "ts":
            cap = (f0_cache / cid) if capture else None
            r = TP.run_scan(img, rawp, replay=({} if capture else replay_for(f0_cache, cid, frozen)), capture_dir=cap,
                            capture_parts=capture, capture_image=False, nr_thr_resamp=1, nr_thr_saving=a.nr_thr_saving)
            src = {c["tid"]: c["source"] for c in r["parts"]}
            assert all(src[t] == "redirect" for t in R["redirect"]), src
            assert all(src[t] in ("replay", "ts") for t in frozen), src
            miss = [c["tid"] for c in r["parts"] if c.get("replay_miss")]
            misses += len(miss)
            extra = {"wall_s": r["wall_s"], "parts": r["parts"], "replay_miss": miss,
                     "image_key16": (r["image_key"] or "")[:16], "sources": src}
        elif kind == "nnunet":
            with tempfile.TemporaryDirectory(dir=str(pd)) as td:
                ob = Path(td) / cid
                pr.predict_from_files([[str(img)]], [str(ob)], save_probabilities=False, overwrite=True,
                                      num_processes_preprocessing=1, num_processes_segmentation_export=1,
                                      folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)
                shutil.move(str(ob) + ".nii.gz", rawp)
            extra = {"wall_s": round(time.time() - t0, 1), "tta": "none"}
        else:
            im0 = nib.load(str(img))
            seg, tm = eng.predict(img)
            assert seg.shape == tuple(im0.shape[:3]), f"{cid}: prediction {seg.shape} != image {im0.shape}"
            assert np.allclose(np.asarray(tm["affine_out"]), im0.affine, atol=1e-3), f"{cid}: Invertd affine != image"
            if tm["n_nan"]:
                seg = np.where(seg == V.NAN_VALUE, 0, seg).astype(np.uint8)
            tmp = rawp.with_name(f".{rawp.name.replace('.nii.gz', '')}.tmp{os.getpid()}.nii.gz")
            out = nib.Nifti1Image(seg.astype(np.uint8), im0.affine)
            out.set_data_dtype(np.uint8)
            nib.save(out, str(tmp))
            os.replace(tmp, rawp)
            extra = {"wall_s": round(time.time() - t0, 1), "n_nan_marker": tm["n_nan"],
                     "timing": {k: tm[k] for k in ("pre_s", "infer_s", "post_s")}}
        peak = int(torch.cuda.max_memory_reserved() / 2 ** 20) if cuda else None
        peak_all = max(peak_all, peak or 0)
        im, sn = nib.load(str(img)), nib.load(str(rawp))
        assert tuple(sn.shape) == tuple(im.shape[:3]) and np.allclose(sn.affine, im.affine, atol=1e-3), f"{cid}: affine/shape"
        seg = np.asarray(sn.dataobj)
        if kind == "nnunet":
            assert int(seg.max()) <= R["n_classes"], (cid, int(seg.max()))
        rec = {"cid": cid, "arm": arm, "kind": kind, "model_sha256": {str(k): v for k, v in R["shas"].items()},
               "image_sha256": img_sha.get(cid), "raw_sha256": T.sha256(rawp), "fg_vox": int((seg > 0).sum()),
               "peak_gpu_reserved_mib": peak, "maxrss_mib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
               "written": T.now()} | extra
        T.write_json(pd / "rec" / f"{cid}.json", rec)
        n_new += 1
        log(f"  [{i + 1}/{len(ids)}] {cid} {time.time() - t0:.0f}s peak {peak} MiB"
            + (f" REPLAY-MISS {miss}" if kind == "ts" and miss else ""))
    per = {c: json.load(open(pd / "rec" / f"{c}.json"))["raw_sha256"] for c in ids}
    done = {"arm": arm, "kind": kind, "ids": ids, "n": len(ids), "model_sha256": {str(k): v for k, v in R["shas"].items()},
            "per_scan_raw_sha256": per, "device": str(dev), "replay_misses_this_run": misses,
            "captured_parts": list(capture), "minutes_this_run": (time.time() - t_all) / 60,
            "peak_gpu_reserved_mib": peak_all, "finished": T.now(), "versions": vers,
            "code_sha256": T.code_hashes(["wn_predict.py", "wn_common.py"]), "mode": "smoke" if a.smoke else "real"}
    if real and not T.before_cutoff():
        log(f"cutoff passed before {arm} finished: DONE.json is not written (the arm is NOT RUN)")
        raise T.Refuse("cutoff passed (Thu 2026-11-19 18:00 KST) before the arm finished: no DONE.json",
                       code=T.CUTOFF_CODE)
    T.write_json(pd / "DONE.json", done)
    log(f"predict done arm={arm}: {n_new} new, misses {misses}, {(time.time() - t_all) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except T.Refuse as e:
        print(f"[REFUSE] {e.msg}", file=sys.stderr)
        sys.exit(e.code)
