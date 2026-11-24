"""vccor/final FLARE-T50 K1c: pipeline and timing test on at most two images, before registration (images only; disclosed).

Runs t50_predict.py --timing for F0 (with capture) and then FULL_s1337 (replaying F0's capture) on the smallest FLARE-T50
image (FLARETs_0030, 18.2 MB) and the largest (FLARETs_0050, 485.1 MB), GPU 1 only, under the shared GPU-1 lock with an
idle check. Every predicted map is hashed into k1c/timing_<arm>.json and deleted; no label is involved.

Kill test (AC.md GO-1 K1c; the expectations are an operational reading of "more than twice the expected time
or memory", fixed here before any run):
  expected minutes per scan = rate x max(1, image MB / 29.5), rate = 1.44 (F0 with capture) or 0.65 (FULL_s1337), the
  measured upper rates of AC.md section 5d, scaled by image size against the FLARE training images' 29.5 MB mean;
  STOP iff any (arm, image) takes more than 2 x its expected minutes, or a run fails (e.g. CUDA out of memory or a killed
  process), or its peak reserved GPU memory exceeds 22 GiB (the A5000 has 24 GiB).
Output: I_flaret50/k1c/k1c.json (read-only; K1C_PASS / K1C_STOP) and a K1C line in final_prereg.log.

usage:  .venv/bin/python -I -B scripts/vccor/final/flaret50/t50_k1c.py          (normally via the ssl-flaret50-k1c unit)

Version of 2026-11-09.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import t50_common as T  # noqa: E402

IDS = ["FLARETs_0030", "FLARETs_0050"]
RATE = {"F0": 1.44, "FULL_s1337": 0.65}
REF_MB = 29.5
FACTOR = 2.0
MAX_GPU_MIB = 22 * 1024
K1C_JSON = T.K1C / "k1c.json"


def expected_min(arm, mb):
    return RATE[arm] * max(1.0, mb / REF_MB)


def main():
    if K1C_JSON.exists():
        raise T.Refuse("k1c.json exists (read-only)")
    T.kst_gate()
    if not T.IMAGES_RECORD.is_file():
        raise T.Refuse("images_record.json missing")
    T.K1C.mkdir(parents=True, exist_ok=True)
    out = {"ids": IDS, "rule": {"rate_min_per_scan": RATE, "ref_mb": REF_MB, "factor": FACTOR, "max_gpu_mib": MAX_GPU_MIB},
           "runs": {}, "started": T.now(), "versions": T.versions(),
           "code_sha256": T.code_hashes(["t50_k1c.py", "t50_predict.py", "t50_common.py"])}
    stop_reasons = []
    for arm in ("F0", "FULL_s1337"):
        fd = T.acquire_gpu_lock(log=lambda m: T.log_event(m, T.K1C / "k1c.log"))
        try:
            T.wait_gpu1_idle(log=lambda m: T.log_event(m, T.K1C / "k1c.log"))
            t0 = time.time()
            r = subprocess.run([str(T.VENV), "-I", "-B", str(HERE / "t50_predict.py"), "--arm", arm, "--device", "gpu",
                                "--timing", "--ids", ",".join(IDS)], cwd=str(T.REPO),
                               env=dict(os.environ, **T.GPU_ENV, PYTHONDONTWRITEBYTECODE="1"),
                               capture_output=True, text=True)
            wall = time.time() - t0
        finally:
            T.release_gpu_lock(fd)
        run = {"rc": r.returncode, "wall_s": round(wall, 1), "stderr_tail": r.stderr[-1500:]}
        tf = T.K1C / f"timing_{arm}.json"
        if r.returncode != 0 or not tf.is_file():
            stop_reasons.append(f"{arm}: run failed (rc {r.returncode})")
            out["runs"][arm] = run
            break
        tj = json.load(open(tf))
        per = {}
        for c in IDS:
            rec = tj["per_scan"][c]
            mb = tj["image_sizes_mb"][c]
            minutes = float(rec["wall_s"]) / 60.0
            exp = expected_min(arm, mb)
            peak = rec.get("peak_gpu_reserved_mib")
            ok_t = minutes <= FACTOR * exp
            ok_m = peak is not None and peak <= MAX_GPU_MIB
            per[c] = {"image_mb": mb, "voxels_at_1p5mm": tj.get("voxels_at_1p5mm", {}).get(c),
                      "minutes": round(minutes, 2), "expected_minutes": round(exp, 2),
                      "within_time": ok_t, "peak_gpu_reserved_mib": peak, "within_memory": ok_m,
                      "maxrss_mib": rec.get("maxrss_mib")}
            if not ok_t:
                stop_reasons.append(f"{arm} on {c}: {minutes:.1f} min > {FACTOR} x {exp:.1f}")
            if not ok_m:
                stop_reasons.append(f"{arm} on {c}: peak GPU memory {peak} MiB")
        run.update(per_scan=per, timing_record=T.rel(tf), timing_sha256=T.sha256(tf))
        out["runs"][arm] = run
    out["stop_reasons"] = stop_reasons
    out["decision"] = "K1C_STOP" if stop_reasons else "K1C_PASS"
    out["finished"] = T.now()
    T.write_json(K1C_JSON, out, mode=0o444)
    T.prereg_line("K1C", K1C_JSON, f"flaret50 K1c pipeline/timing test on FLARETs_0030 and FLARETs_0050 (images only, "
                                   f"pre-registration; maps deleted): {out['decision']}")
    return 0 if out["decision"] == "K1C_PASS" else 4


if __name__ == "__main__":
    try:
        sys.exit(main())
    except T.Refuse as e:
        print(f"[REFUSE] {e.msg}", file=sys.stderr)
        sys.exit(e.code)
