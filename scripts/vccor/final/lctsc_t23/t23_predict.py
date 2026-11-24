"""vccor/final LCTSC-T23 PREDICT (DERIVED from the frozen lctsc_predict.py by derive_t23.py; GPU 1; idempotent,
resumable per patient): one arm (F0, F1b_S1..S3 = the frozen d6ayj models) on the CTs of the 23 LCTSC-T23 patients.
Their RTSTRUCT zips are on disk but are never extracted before the look. Mechanics == FLARE-T50's t50_predict.py / the frozen predict_final.py:
TotalSegmentator 2.13.0's own 'total' pipeline through the FROZEN scripts/vccor/tsft_full/ts_pipeline.py (1.5 mm, no TTA,
TS's merge order 291..295); an F1b arm's parts 291 and 293 are redirected to its export (checkpoint sha256 == export.json,
recipe F1b, full 50-epoch non-smoke run asserted); parts 292, 294, 295 are replayed byte-for-byte from F0's frozen-part
cache (pred/F0/cache_frozen/<pid>/) when the 1.5 mm image key matches, else run with TS's own weights (recorded as a
replay miss). F0 captures 292 / 294 / 295; every F1b arm needs F0 DONE first. Per patient: pred/<arm>/raw/<pid>.nii.gz
(TS total ids; shape / affine == the image, asserted) and rec/<pid>.json; DONE.json last. Every image is re-hashed against
K_lctsc/images.json before use.

Modes:
  real   (default): GPU only; only after the freeze (t23_gates.freeze_ok) and the upload gate; never after the
         look; started only before the cutoff (Fri 2026-11-20 10:00 KST); / kept >= 25 GiB free.
  smoke  (--smoke --root DIR --images DIR --ids ...): stand-in images (e.g. training patients) only, writes only under --root;
         refuses any LCTSC-Test id, any LCTSC-T23 id and any --images folder inside data_lctsc/images.

usage:  run only through t23_predict_queue.py (unit ssl-lctsc-t23-predict), which sets CUDA_VISIBLE_DEVICES = GPU 1's UUID;
        real mode itself refuses any other device and takes the shared GPU-1 flock (review B3)

Version of 2026-11-13; LCTSC-T23 derivation 2026-11-17.
"""
import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent / "lctsc"), str(HERE.parent), str(HERE.parent.parent / "tsft_full")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import lctsc_common as C  # noqa: E402
import t23_gates as G  # noqa: E402
import t23_common as T  # noqa: E402
import lctsc_infra as I  # noqa: E402

_LOCK = {"fd": None}


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=list(G.ARMS))
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--nr_thr_saving", type=int, default=2)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--root", default=None)
    ap.add_argument("--images", default=None)
    ap.add_argument("--ids", default=None)
    ap.add_argument("--model_dir", default=None, help="smoke only: a smoke-trained model dir (DONE.json) for an F1b arm")
    a = ap.parse_args(argv)
    if a.smoke:
        assert a.root and a.images and a.ids, "smoke needs --root, --images and --ids"
        ids = [c.strip() for c in a.ids.split(",") if c.strip()]
        assert not any(C.is_test(c) or c in T.ids() for c in ids), "smoke never runs on an LCTSC test or LCTSC-T23 patient"
        img = Path(a.images).resolve()
        assert img != C.IMAGES.resolve(), "smoke reads stand-in copies, never data_lctsc/images"
    else:
        assert not (a.root or a.images or a.ids or a.model_dir), "smoke-only options"
        assert a.device == "gpu", "real predictions run on GPU 1 only"
    return a


def resolve_model(arm, strict=True, model_dir=None):
    """-> {'redirect': {tid: dir}, 'shas': {tid: sha}} ('ts' kind only)."""
    run = G.ARMS[arm]["run"]
    if run is None:
        return {"redirect": {}, "shas": {"ts": "TotalSegmentator 2.13.0 original weights"}}
    d = Path(model_dir) if model_dir else G.MODELS / run
    done = json.load(open(d / "DONE.json"))
    red, shas = {}, {}
    for tid, md in sorted(done["resolved"]["parts"].items()):
        md = Path(md)
        man = json.load(open(md / "export.json"))
        h = G.sha256(md / "fold_0" / "checkpoint_final.pth")
        if h != man["checkpoint_sha256"]:
            raise G.Refuse(f"{md}: checkpoint changed after export")
        if strict:
            if not (man["num_epochs"] == G.TS_EPOCHS and not man["smoke"] and man["variant"] == "F1b"
                    and abs(man["initial_lr"] - 1e-4) < 1e-12 and man["trainable"] == "all"):
                raise G.Refuse(f"{md}: not a full F1b training run")
        red[int(tid)], shas[int(tid)] = str(md), h
    if sorted(red) != sorted(C.PARTS):
        raise G.Refuse(f"{arm}: parts {sorted(red)} != {list(C.PARTS)}")
    return {"redirect": red, "shas": shas}


def frozen_model_check(arm, fz):
    want = fz.get("models", {}).get(arm)
    if G.ARMS[arm]["run"] is None:
        return
    got = {str(k): v for k, v in resolve_model(arm)["shas"].items()}
    if want != got:
        raise G.Refuse(f"{arm}: model hashes {got} differ from the freeze {want}")


def replay_for(cache_root, pid, frozen):
    d = Path(cache_root) / pid
    kf = d / "image_key.txt"
    if not kf.is_file():
        return {}
    key = kf.read_text().strip()
    return {t: {"key": key, "path": str(d / f"part_{t}.nii.gz")} for t in frozen if (d / f"part_{t}.nii.gz").is_file()}


def done_ok(pd, pid, shas):
    try:
        rec = json.load(open(pd / "rec" / f"{pid}.json"))
        return rec["model_sha256"] == {str(k): v for k, v in shas.items()} and rec["raw_sha256"] == G.sha256(pd / "raw" / f"{pid}.nii.gz")
    except (OSError, KeyError, ValueError):
        return False


def main(argv=None):
    a = parse(argv)
    arm = a.arm
    if a.smoke:
        root, ids = Path(a.root), [c.strip() for c in a.ids.split(",") if c.strip()]
        image = (lambda c: Path(a.images) / f"{c}.nii.gz")
        img_sha = {c: None for c in ids}
    else:
        G.kst_gate()
        if G.look_made():
            raise G.Refuse("a look exists: no prediction after the look")
        fz = G.freeze_ok()
        if G.notrun_declared(arm):                       # review NS1: a binding NOT RUN declaration cannot be undone
            raise G.Refuse(f"{arm} is declared NOT RUN (NOTRUN line): no prediction")
        if G.frozen_draw_status(arm) != "DONE":          # review NB1: a draw NOT RUN in the freeze has no prediction
            raise G.Refuse(f"{arm}: its training draw is {G.frozen_draw_status(arm)} in the freeze")
        frozen_model_check(arm, fz)
        G.gate_uploaded()
        if not G.before_cutoff():
            raise I.Refuse("cutoff passed (Fri 2026-11-20 10:00 KST): no prediction is started", code=I.DEADLINE_CODE)
        I.wait_disk(0.5, f"prediction {arm}", T.CUTOFF_DT)                        # review B3: wait, never refuse
        # review B3: GPU 1 only (UUID identity gate), idle wait WITHOUT the lock, then the shared flock + re-check; the
        # lock is held until this process exits (lctsc_infra, copied from WORD-N's wn_common primitives)
        _LOCK["fd"] = I.gpu_session(T.CUTOFF_DT)
        root, ids, image = G.PRED, G.test_ids(), C.image_path
        rec_img = json.load(open(C.OUT / "images.json"))["patients"]
        img_sha = {}
        for c in ids:
            h = G.sha256(image(c))
            if h != rec_img[c]["sha256"]:
                raise G.Refuse(f"{c}: image sha256 differs from images.json")
            img_sha[c] = h
    R = resolve_model(arm, strict=not a.smoke, model_dir=a.model_dir)
    pd = G.pred_dir(arm, root)
    if (pd / "DONE.json").is_file() and not a.smoke:
        print(f"{arm}: DONE exists; nothing to do")
        return 0
    capture = G.CAPTURE if arm == "F0" else ()
    f0_cache = G.pred_dir("F0", root) / "cache_frozen"
    if arm != "F0" and not (G.pred_dir("F0", root) / "DONE.json").is_file():
        raise G.Refuse(f"{arm}: F0 must be DONE first (its frozen-part cache feeds the replay)")
    for sub in ("raw", "rec"):
        (pd / sub).mkdir(parents=True, exist_ok=True)
    logp = pd / "predict.log"

    def log(m):
        line = f"[{G.now()}] {m}"
        print(line, flush=True)
        with open(logp, "a") as f:
            f.write(line + "\n")

    import nibabel as nib
    import numpy as np
    import torch
    import ts_pipeline as TP
    if a.device == "cpu":
        torch.set_num_threads(a.threads)
    dev = TP.init(a.device, a.threads, redirect=R["redirect"])
    frozen = [t for t in (291, 292, 293, 294, 295) if t not in R["redirect"]]
    cuda = a.device == "gpu" and torch.cuda.is_available()
    log(f"predict start arm={arm} n={len(ids)} device={dev} capture={capture} mode={'smoke' if a.smoke else 'real'} -> {pd} pid={os.getpid()}")
    t_all, misses, peak_all = time.time(), 0, 0
    for i, pid in enumerate(ids):
        if done_ok(pd, pid, R["shas"]):
            log(f"  [{i + 1}/{len(ids)}] {pid} [skip] done")
            continue
        img, rawp, t0 = image(pid), pd / "raw" / f"{pid}.nii.gz", time.time()
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        cap = (f0_cache / pid) if capture else None
        r = TP.run_scan(img, rawp, replay=({} if capture else replay_for(f0_cache, pid, frozen)), capture_dir=cap,
                        capture_parts=capture, capture_image=False, nr_thr_resamp=1, nr_thr_saving=a.nr_thr_saving)
        src = {c["tid"]: c["source"] for c in r["parts"]}
        assert all(src[t] == "redirect" for t in R["redirect"]), src
        assert all(src[t] in ("replay", "ts") for t in frozen), src
        miss = [c["tid"] for c in r["parts"] if c.get("replay_miss")]
        misses += len(miss)
        peak = int(torch.cuda.max_memory_reserved() / 2 ** 20) if cuda else None
        peak_all = max(peak_all, peak or 0)
        im, sn = nib.load(str(img)), nib.load(str(rawp))
        assert tuple(sn.shape) == tuple(im.shape[:3]) and np.allclose(sn.affine, im.affine, atol=1e-3), f"{pid}: affine/shape"
        seg = np.asarray(sn.dataobj)
        rec = {"pid": pid, "arm": arm, "model_sha256": {str(k): v for k, v in R["shas"].items()}, "image_sha256": img_sha.get(pid),
               "raw_sha256": G.sha256(rawp), "fg_vox": int((seg > 0).sum()), "peak_gpu_reserved_mib": peak,
               "maxrss_mib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024), "wall_s": r["wall_s"],
               "parts": r["parts"], "replay_miss": miss, "sources": src, "written": G.now()}
        G.write_json(pd / "rec" / f"{pid}.json", rec)
        log(f"  [{i + 1}/{len(ids)}] {pid} {time.time() - t0:.0f}s peak {peak} MiB" + (f" REPLAY-MISS {miss}" if miss else ""))
    if not a.smoke and not G.before_cutoff():
        log("cutoff passed before DONE: the arm is NOT RUN (no DONE.json written)")
        return 4
    per = {c: json.load(open(pd / "rec" / f"{c}.json"))["raw_sha256"] for c in ids}
    done = {"arm": arm, "ids": ids, "n": len(ids), "model_sha256": {str(k): v for k, v in R["shas"].items()},
            "per_scan_raw_sha256": per, "device": str(dev), "replay_misses_this_run": misses, "captured_parts": list(capture),
            "minutes_this_run": (time.time() - t_all) / 60, "peak_gpu_reserved_mib": peak_all, "finished": G.now(),
            "code_sha256": {f: G.sha256(HERE / f) for f in ("t23_predict.py", "t23_gates.py", "t23_common.py")},
            "mode": "smoke" if a.smoke else "real"}
    G.write_json(pd / "DONE.json", done)
    log(f"predict done arm={arm}: {(time.time() - t_all) / 60:.1f} min, replay misses {misses}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
