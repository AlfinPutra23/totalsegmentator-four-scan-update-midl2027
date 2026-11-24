"""vccor/final FLARE-T50 POST HOC PREDICT (GPU 1; EXPLORATORY, NOT REGISTERED; no label is read or listed).

Predicts the post hoc arms (ph_common.ORDER: LR1E2, LORA_B, HEAD) on the 50 FLARE-T50 images that I_flaret50 already holds,
with each model's own registered predict path: TotalSegmentator 2.13.0's 'total' pipeline through the FROZEN
tsft_full/ts_pipeline.py (TS's own settings, no TTA, merge 291..295), parts 291 + 293 redirected to the model's export
(== predict_peft.py / predict_e2.py: ts_pipeline.init(redirect) + run_scan(replay, nr_thr_resamp 1, nr_thr_saving 2)), parts
292 / 294 / 295 replayed byte-for-byte from F0's frozen-part cache ON T50 (I_flaret50/pred/F0/cache_frozen, read only; the
same replay FULL_s1337 used there; a key mismatch re-runs TS's own weights and is counted as a replay miss). The PEFT
records' second step (teacher_common.map_seg to the FLARE22 h5 frame) does not apply to T50: the raw map in TS 'total' ids
is kept and scored on the native grid through final_common.ts_lut('flare'), exactly like every 'ts' arm of the T50 look.

Outputs: J_t50_posthoc/pred/<arm>/raw/<cid>.nii.gz, rec/<cid>.json, DONE.json (per-scan raw sha256), predict.log.
Checks: model == PEFT records (ph_common.resolve); images == images_record.json; package versions == the T50 freeze's;
TS's own weights == the T50 freeze; the F0 cache files used are hashed before and after (unchanged); / >= 25 GiB free.

usage (wrapped by the caller in flock on the shared GPU-1 lock):
  CUDA_VISIBLE_DEVICES=<GPU 1 UUID> PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      .venv/bin/python -I -B scripts/vccor/final/t50_posthoc/ph_predict.py [--arms LR1E2,LORA_B,HEAD]
Version of 2026-11-13.
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
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import ph_common as PH  # noqa: E402

T = PH.T


def replay_for(cid, frozen):
    d = PH.F0_CACHE / cid
    kf = d / "image_key.txt"
    if not kf.is_file():
        return {}
    key = kf.read_text().strip()
    return {t: {"key": key, "path": str(d / f"part_{t}.nii.gz")} for t in frozen if (d / f"part_{t}.nii.gz").is_file()}


def cache_hashes(ids, frozen):
    out = {}
    for c in ids:
        d = PH.F0_CACHE / c
        for f in [d / "image_key.txt"] + [d / f"part_{t}.nii.gz" for t in frozen]:
            if f.is_file():
                out[T.rel(f)] = T.sha256(f)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default=",".join(PH.ORDER))
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    assert all(x in PH.ARMS for x in arms), arms
    T.kst_gate()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != PH.GPU1_UUID:
        raise T.Refuse(f"CUDA_VISIBLE_DEVICES must be GPU 1's UUID {PH.GPU1_UUID}")
    PH.disk_ok(1.0)
    fz = json.load(open(T.FREEZE))
    vers = T.check_versions(fz.get("versions"))
    T._check_hashes(fz["models"]["TS_WEIGHTS"], "TS_WEIGHTS")          # read-only hash check
    ids = T.included_ids()
    assert len(ids) == 50, len(ids)
    rec_img = json.load(open(T.IMAGES_RECORD))["images"]
    img_sha = {}
    for c in ids:
        h = T.sha256(T.image_path(c))
        if h != rec_img[c]["sha256"]:
            raise T.Refuse(f"{c}: image differs from images_record.json")
        img_sha[c] = h
    frozen = [t for t in (291, 292, 293, 294, 295) if t not in PH.PARTS]
    cache_before = cache_hashes(ids, frozen)
    PH.log(f"post hoc predict: arms {arms}, {len(ids)} images verified, F0 cache files hashed ({len(cache_before)}), "
           f"versions == freeze, TS weights == freeze, CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    u = T.wait_gpu1_idle(wait_s=2 * 3600, log=PH.log)
    PH.log(f"GPU 1 memory.used {u} MiB at start")
    import nibabel as nib
    import numpy as np
    import torch
    sys.path.append(str(T.REPO / "scripts/vccor/tsft_full"))
    import ts_pipeline as TP
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1, "exactly GPU 1 must be visible"
    PH.log(f"torch device 0 = {torch.cuda.get_device_name(0)}")
    gpu_t0 = time.time()
    summary = {}
    for arm in arms:
        red, shas, fh = PH.resolve(arm)
        pd = PH.PRED / arm
        if (pd / "DONE.json").is_file():
            PH.log(f"{arm}: DONE exists; skipped")
            continue
        for sub in ("raw", "rec"):
            (pd / sub).mkdir(parents=True, exist_ok=True)
        plog = (lambda m, _p=pd / "predict.log": T.log_event(m, _p))
        dev = TP.init("gpu", a.threads, redirect=red)
        plog(f"predict start arm={arm} n={len(ids)} device={dev} redirect={ {k: T.rel(v) for k, v in red.items()} } "
             f"pid={os.getpid()}")
        t_arm, misses, peak_all = time.time(), 0, 0
        for i, cid in enumerate(ids):
            rawp, recp = pd / "raw" / f"{cid}.nii.gz", pd / "rec" / f"{cid}.json"
            if recp.is_file() and rawp.is_file():
                r0 = json.load(open(recp))
                if r0.get("model_sha256") == {str(k): v for k, v in shas.items()} and r0.get("raw_sha256") == T.sha256(rawp):
                    plog(f"  [{i + 1}/{len(ids)}] {cid} [skip] done")
                    continue
            PH.disk_ok(0.2)
            t0 = time.time()
            torch.cuda.reset_peak_memory_stats()
            r = TP.run_scan(T.image_path(cid), rawp, replay=replay_for(cid, frozen), nr_thr_resamp=1, nr_thr_saving=2)
            src = {c["tid"]: c["source"] for c in r["parts"]}
            assert all(src[t] == "redirect" for t in PH.PARTS), src
            assert all(src[t] in ("replay", "ts") for t in frozen), src
            miss = [c["tid"] for c in r["parts"] if c.get("replay_miss")]
            misses += len(miss)
            peak = int(torch.cuda.max_memory_reserved() / 2 ** 20)
            peak_all = max(peak_all, peak)
            im, sn = nib.load(str(T.image_path(cid))), nib.load(str(rawp))
            assert tuple(sn.shape) == tuple(im.shape[:3]) and np.allclose(sn.affine, im.affine, atol=1e-3), f"{cid}: affine"
            seg = np.asarray(sn.dataobj)
            rec = {"cid": cid, "arm": arm, "kind": "ts", "post_hoc": True,
                   "model_sha256": {str(k): v for k, v in shas.items()}, "image_sha256": img_sha[cid],
                   "raw_sha256": T.sha256(rawp), "fg_vox": int((seg > 0).sum()), "peak_gpu_reserved_mib": peak,
                   "maxrss_mib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024), "written": T.now(),
                   "wall_s": r["wall_s"], "parts": r["parts"], "replay_miss": miss,
                   "image_key16": (r["image_key"] or "")[:16], "sources": src}
            PH.write_json(recp, rec)
            plog(f"  [{i + 1}/{len(ids)}] {cid} {time.time() - t0:.0f}s peak {peak} MiB" + (f" REPLAY-MISS {miss}" if miss else ""))
        per = {c: json.load(open(pd / "rec" / f"{c}.json"))["raw_sha256"] for c in ids}
        done = {"arm": arm, "post_hoc": True, "exploratory": True, "kind": "ts", "label": PH.ARMS[arm]["label"],
                "ids": ids, "n": len(ids), "model_sha256": {str(k): v for k, v in shas.items()}, "model_files_sha256": fh,
                "per_scan_raw_sha256": per, "device": str(dev), "gpu_uuid": PH.GPU1_UUID,
                "replay_misses_this_run": misses, "minutes_this_run": (time.time() - t_arm) / 60,
                "peak_gpu_reserved_mib": peak_all, "finished": T.now(), "versions": vers,
                "code_sha256": {n: T.sha256(HERE / n) for n in ("ph_predict.py", "ph_common.py")}}
        PH.write_json(pd / "DONE.json", done)
        summary[arm] = done["minutes_this_run"]
        plog(f"predict done arm={arm}: misses {misses}, {done['minutes_this_run']:.1f} min, peak {peak_all} MiB")
        PH.log(f"{arm} DONE: {done['minutes_this_run']:.1f} min, replay misses {misses}")
    cache_after = cache_hashes(ids, frozen)
    if cache_after != cache_before:
        raise T.Refuse("the F0 frozen-part cache changed during the run")
    gpu_min = (time.time() - gpu_t0) / 60
    PH.write_json(PH.OUT / "predict_run.json", {"arms": arms, "minutes_per_arm_this_run": summary,
                                                "gpu_wall_minutes_total": gpu_min, "f0_cache_unchanged": True,
                                                "f0_cache_files_checked": len(cache_after), "finished": T.now(),
                                                "gpu_uuid": PH.GPU1_UUID})
    PH.log(f"post hoc predict finished: {gpu_min:.1f} GPU-wall minutes; F0 cache unchanged ({len(cache_after)} files)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except T.Refuse as e:
        PH.log(f"[REFUSE] {e.msg}")
        sys.exit(e.code)
