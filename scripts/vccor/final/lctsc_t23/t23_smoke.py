"""vccor/final LCTSC-T23 GPU SMOKE (before the freeze; review S3): t23_predict.py in smoke mode on a stand-in copy of ONE
site-draw training CT (t23_common.STANDINS[0], label read on 2026-11-13) -- F0, then F1b_S1 with the frozen d6ayj model --
under a new root K_lctsc_t23/smoke_<n>/. Checks: both arms DONE in smoke mode, no replay miss, the F1b_S1 checkpoint hashes
equal the LCTSC freeze, parts 291/293 redirected and 292/294/295 replayed, the spinal cord (part 294) unchanged, and the
adapted structures changed. Writes K_lctsc_t23/smoke.json with the code_sha256 of every lctsc_t23 *.py and *.service (the
freeze compares them with the current files). Refuses after the T23 freeze (review S4). No LCTSC-T23 or test file is read.

usage:  .venv/bin/python -I -B scripts/vccor/final/lctsc_t23/t23_smoke.py

Version of 2026-11-17.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
for _p in (str(HERE.parent / "lctsc"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import lctsc_common as C  # noqa: E402
import lctsc_infra as I  # noqa: E402
import t23_common as T  # noqa: E402
import t23_log as L  # noqa: E402

PY = "/home/user/Documents/Medical Imagining SSL/.venv/bin/python"


def main():
    if (T.OUT / "code_freeze.json").exists():
        raise SystemExit("an LCTSC-T23 freeze exists: no new smoke record")
    used = I.gpu1_used_mib()
    if used >= I.GPU_IDLE_MIB:
        raise SystemExit(f"GPU 1 busy ({used} MiB): try later")
    n = 1
    while (T.OUT / f"smoke_{n}").exists():
        n += 1
    root = T.OUT / f"smoke_{n}"
    pid = T.STANDINS[0]
    assert pid not in T.ids() and not C.is_test(pid)
    (root / "images").mkdir(parents=True)
    shutil.copyfile(C.image_path(pid), root / "images" / f"{pid}.nii.gz")
    env = dict(os.environ, **I.GPU_ENV, PYTHONDONTWRITEBYTECODE="1")
    runs = {}
    for arm in ("F0", "F1b_S1"):
        r = subprocess.run([PY, "-I", "-B", str(HERE / "t23_predict.py"), "--arm", arm, "--device", "gpu", "--smoke",
                            "--root", str(root), "--images", str(root / "images"), "--ids", pid],
                           env=env, capture_output=True, text=True, timeout=1800)
        runs[arm] = {"rc": r.returncode, "tail": (r.stdout + r.stderr)[-400:]}
        if r.returncode != 0:
            break
    import nibabel as nib
    import numpy as np
    ok, rec = all(v["rc"] == 0 for v in runs.values()) and len(runs) == 2, {}
    if ok:
        d0, d1 = (json.load(open(root / a / "DONE.json")) for a in ("F0", "F1b_S1"))
        r0, r1 = (json.load(open(root / a / "rec" / f"{pid}.json")) for a in ("F0", "F1b_S1"))
        fz = json.load(open(T.SRC / "code_freeze.json"))
        lut = C.ts_total_lut()
        a0 = lut[np.asarray(nib.load(str(root / "F0" / "raw" / f"{pid}.nii.gz")).dataobj).astype(np.uint8)]
        a1 = lut[np.asarray(nib.load(str(root / "F1b_S1" / "raw" / f"{pid}.nii.gz")).dataobj).astype(np.uint8)]
        diff = {C.ORGANS[o - 1]: int(((a0 == o) != (a1 == o)).sum()) for o in range(1, 6)}
        ok = (d0["mode"] == d1["mode"] == "smoke" and d1["model_sha256"] == fz["models"]["F1b_S1"]
              and d0["replay_misses_this_run"] == 0 and d1["replay_misses_this_run"] == 0 and diff["spinal_cord"] == 0
              and sum(diff[k] for k in ("esophagus", "heart", "lung_left", "lung_right")) > 0
              and r1["sources"] == {"291": "redirect", "292": "replay", "293": "redirect", "294": "replay", "295": "replay"})
        rec = {"F0": {"wall_s": r0["wall_s"], "peak_gpu_reserved_mib": r0["peak_gpu_reserved_mib"], "minutes": d0["minutes_this_run"]},
               "F1b_S1": {"wall_s": r1["wall_s"], "peak_gpu_reserved_mib": r1["peak_gpu_reserved_mib"],
                          "minutes": d1["minutes_this_run"], "sources": r1["sources"],
                          "model_sha256_equals_lctsc_freeze": d1["model_sha256"] == fz["models"]["F1b_S1"]},
               "voxels_changed_F1b_vs_F0": diff,
               "estimated_real_run_minutes": round(23 * (d0["minutes_this_run"] + 3 * d1["minutes_this_run"]), 1)}
    out = dict(rec, ok=bool(ok), root=L.rel(root), standin=pid, runs=runs,
               note="GPU smoke of t23_predict.py in smoke mode on a stand-in copy of a site-draw training CT (label read on "
                    "2026-11-13); no LCTSC-T23 or test file",
               code_sha256={f.name: L.sha256(f) for f in sorted(HERE.glob("*.py")) + sorted(HERE.glob("*.service"))},
               written=L.now())
    L.write_json(T.OUT / "smoke.json", out)
    print(json.dumps({k: out[k] for k in ("ok", "root")} | {"est_min": out.get("estimated_real_run_minutes")}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
