"""vccor/final FLARE-T50 PREDICT (GPU job; idempotent, resumable per scan): one arm on the included FLARE-T50 scans.
NO ground truth is read or listed. Mechanics == scripts/vccor/final/predict_final.py (frozen; not imported because it
parses its arguments at import):

  'ts' arms      TotalSegmentator 2.13.0's own 'total' pipeline through the FROZEN scripts/vccor/tsft_full/ts_pipeline.py
                 (1.5 mm, no TTA, TS's merge order 291..295); the arm's parts are redirected to its export (checkpoint
                 sha256 == export.json, registered checks, full 50-epoch non-smoke run asserted by t50_common.resolve_model);
                 every other part is replayed byte-for-byte from F0's frozen-part cache when the 1.5 mm image key matches,
                 else run with TS's own weights (recorded as a replay miss). F0 CAPTURES parts 292-295 per scan into
                 pred/F0/cache_frozen/<cid>/; every other 'ts' arm needs F0 DONE first.
  'nnunet' arms  nnU-Net's nnUNetPredictor (tile 0.5, gaussian, mirroring OFF; the checkpoint must allow no mirroring) on the
                 raw image -> FLARE's 13 ids on the raw grid (fold 'all').
Per scan: pred/<arm>/raw/<cid>.nii.gz (affine / shape == the image, asserted) and rec/<cid>.json (wall time, peak GPU
memory, sources); DONE.json last (per-scan raw sha256). Every image is re-hashed against images_record.json before use.

Modes:
  real    (no --timing / --smoke): GPU only; only after the freeze (code, records, the arm's model and TotalSegmentator's
          weights unchanged; package versions == the freeze's) and the gate_osf_uploaded.json record; never after
          the look; on t50_common.included_ids() (exclusions fixed by K1b), with / kept >= 35 GiB free; started only before
          the cutoff (Fri 2026-11-13 18:00 KST), and DONE.json is never written at or after the cutoff (the arm is then
          NOT RUN).
  timing  (--timing, K1c, before registration, disclosed): exactly the registered pair FLARETs_0030 and FLARETs_0050,
          arms F0 / FULL_s1337 only, once (refuses if k1c.json or timing_<arm>.json exists); outputs under
          I_flaret50/k1c/run/ (F0 captures there); after the run every map is hashed into k1c/timing_<arm>.json and DELETED.
  smoke   (--smoke --images DIR --root DIR --ids ...): unit tests / CPU smoke on stand-in images, writes only under --root;
          refuses any FLARETs_ id and any --images folder under data_flaret50/.

usage:  .venv/bin/python -I -B scripts/vccor/final/flaret50/t50_predict.py --arm F0 --device gpu
        .venv/bin/python -I -B scripts/vccor/final/flaret50/t50_predict.py --arm F0 --device gpu --timing --ids FLARETs_0030,FLARETs_0050

Version of 2026-11-09.
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
import t50_common as T  # noqa: E402

TIMING_ARMS = ("F0", "FULL_s1337")


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=list(T.ARMS))
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--nr_thr_saving", type=int, default=2)
    ap.add_argument("--ids", default=None, help="timing / smoke only: comma list of scans")
    ap.add_argument("--timing", action="store_true", help="K1c: <= 2 scans, F0 / FULL_s1337, outputs deleted after hashing")
    ap.add_argument("--smoke", action="store_true", help="unit tests: stand-in images, writes only under --root")
    ap.add_argument("--root", default=None, help="smoke only: prediction root")
    ap.add_argument("--images", default=None, help="smoke only: dir with <cid>_0000.nii.gz stand-in images")
    a = ap.parse_args(argv)
    assert a.threads <= 6
    ids = [c.strip() for c in (a.ids or "").split(",") if c.strip()]
    if a.timing:
        assert a.arm in TIMING_ARMS, "K1c times F0 and FULL_s1337 only"
        assert sorted(ids) == sorted(T.K1C_IDS), f"K1c runs exactly the registered pair {T.K1C_IDS}"
        assert not (a.smoke or a.root or a.images)
        assert a.device == "gpu", "K1c times the GPU path"
    elif a.smoke:
        assert ids and a.root and a.images, "smoke needs --ids, --root and --images"
        assert not any(c.startswith("FLARETs_") for c in ids), "smoke never runs on a FLARE-T50 id"
        img = Path(a.images).resolve()
        data = Path(T.DATA).resolve()
        assert img != data and data not in img.parents, "smoke never reads the FLARE-T50 data folder"
    else:
        assert not (a.ids or a.root or a.images), "--ids / --root / --images are timing / smoke options"
        assert a.device == "gpu", "real predictions run on GPU 1 only"
    return a


def mode_paths(a):
    if a.timing:
        root = T.K1C / "run"
        return root, [c.strip() for c in a.ids.split(",") if c.strip()], None
    if a.smoke:
        return Path(a.root), [c.strip() for c in a.ids.split(",") if c.strip()], Path(a.images)
    return T.PRED, T.included_ids(), None


def check_images(ids, img_root, smoke):
    if smoke:
        return {c: None for c in ids}
    rec = json.load(open(T.IMAGES_RECORD))["images"]
    out = {}
    for c in ids:
        p = T.image_path(c)
        h = T.sha256(p)
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
        return rec["model_sha256"] == {str(k): v for k, v in shas.items()} and rec["raw_sha256"] == T.sha256(pd / "raw" / f"{cid}.nii.gz")
    except (OSError, KeyError, ValueError):
        return False


def main(argv=None):
    a = parse(argv)
    root, ids, img_root = mode_paths(a)
    arm = a.arm
    real = not (a.timing or a.smoke)
    vers = T.check_versions()
    if real:
        T.kst_gate()
        if T.look_made():
            raise T.Refuse("a look exists: no prediction after the look")
        fz = T.freeze_ok()
        vers = T.check_versions(fz.get("versions"))
        T.check_frozen_model(arm)
        T.gate_uploaded()
        if not T.before_cutoff():
            raise T.Refuse("cutoff passed (Fri 2026-11-13 18:00 KST): no prediction is started")
        T.disk_gate(0.3, f"prediction {arm}")
    if a.timing:
        T.kst_gate()
        if (T.K1C / "k1c.json").exists() or (T.K1C / f"timing_{arm}.json").exists():
            raise T.Refuse("K1c already ran (k1c.json or timing record exists): it runs once")
        T.disk_gate(1.0, f"K1c {arm}")
    kind, R = T.resolve_model(arm, strict=not a.smoke)
    pd = T.pred_dir(arm, root)
    if (pd / "DONE.json").is_file() and not a.smoke:
        print(f"{arm}: DONE exists ({pd / 'DONE.json'}); nothing to do")
        return 0
    capture = T.CAPTURE if arm == "F0" else ()
    f0_cache = T.pred_dir("F0", root) / "cache_frozen"
    if kind == "ts" and arm != "F0" and not a.timing and not (T.pred_dir("F0", root) / "DONE.json").is_file():
        raise T.Refuse(f"{arm}: F0 must be DONE first (its frozen-part cache feeds the replay)")
    img_sha = check_images(ids, img_root, a.smoke)
    for sub in ("raw", "rec"):
        (pd / sub).mkdir(parents=True, exist_ok=True)
    logp = (pd / "predict.log")
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
    else:
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
    cuda = a.device == "gpu" and torch.cuda.is_available()
    log(f"predict start arm={arm} kind={kind} n={len(ids)} device={dev} capture={capture} mode="
        f"{'timing' if a.timing else ('smoke' if a.smoke else 'real')} -> {pd} pid={os.getpid()}")
    t_all, n_new, misses, peak_all = time.time(), 0, 0, 0
    for i, cid in enumerate(ids):
        if done_ok(pd, cid, R["shas"]):
            log(f"  [{i + 1}/{len(ids)}] {cid} [skip] done")
            continue
        img = T.image_path(cid, img_root)
        rawp, t0 = pd / "raw" / f"{cid}.nii.gz", time.time()
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
        else:
            with tempfile.TemporaryDirectory(dir=str(pd)) as td:
                ob = Path(td) / cid
                pr.predict_from_files([[str(img)]], [str(ob)], save_probabilities=False, overwrite=True,
                                      num_processes_preprocessing=1, num_processes_segmentation_export=1,
                                      folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)
                shutil.move(str(ob) + ".nii.gz", rawp)
            extra = {"wall_s": round(time.time() - t0, 1), "tta": "none"}
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
            "code_sha256": T.code_hashes(["t50_predict.py", "t50_common.py"]),
            "mode": "timing" if a.timing else ("smoke" if a.smoke else "real")}
    if a.timing:
        def vox15(c):
            h = nib.load(str(T.image_path(c))).header
            return int(np.prod([n * float(z) / 1.5 for n, z in zip(h.get_data_shape()[:3], h.get_zooms()[:3])]))
        timing = {"arm": arm, "ids": ids, "per_scan": {c: json.load(open(pd / "rec" / f"{c}.json")) for c in ids},
                  "image_sizes_mb": {c: round(T.image_path(c).stat().st_size / 1e6, 1) for c in ids},
                  "voxels_at_1p5mm": {c: vox15(c) for c in ids}, "versions": vers,
                  "code_sha256": T.code_hashes(["t50_predict.py", "t50_common.py"]),
                  "deleted_maps": {c: per[c] for c in ids}, "written": T.now(),
                  "note": "K1c pipeline/timing test before registration (images only, disclosed); every map listed here "
                          "was hashed and then deleted"}
        T.write_json(T.K1C / f"timing_{arm}.json", timing, mode=0o444)
        shutil.rmtree(pd / "raw")
        if arm != "F0":
            shutil.rmtree(T.K1C / "run", ignore_errors=True)     # F0's timing cache is kept until FULL_s1337's timing ran
        log(f"K1c {arm}: maps hashed into k1c/timing_{arm}.json and deleted")
        return 0
    if real and not T.before_cutoff():
        log(f"cutoff passed before {arm} finished: DONE.json is not written (the arm is NOT RUN)")
        raise T.Refuse("cutoff passed (Fri 2026-11-13 18:00 KST) before the arm finished: no DONE.json")
    T.write_json(pd / "DONE.json", done)
    log(f"predict done arm={arm}: {n_new} new, misses {misses}, {(time.time() - t_all) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except T.Refuse as e:
        print(f"[REFUSE] {e.msg}", file=sys.stderr)
        sys.exit(e.code)
