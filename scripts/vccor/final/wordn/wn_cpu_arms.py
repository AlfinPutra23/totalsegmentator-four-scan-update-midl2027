"""vccor/final WORD-N CPU ARMS: MERGE_N_dX and LABELMAP_dX on the included WORD validation scans, with A1.2's frozen
operators unchanged (scripts/vccor/final/word_a1/word_a1_cpu_arms.py, imported read-only):

  MERGE_N_dX   paper_v2/v2_common.merge_maps on the raw TS-id maps of F0 and F1b_dX, the site classes T = word_a1_cpu_arms.
               site_lut() (non-target classes equal F0); scored with lut 'ts'.
  LABELMAP_dX  F0's raw map relabelled with draw dX's FROZEN rules: the rules.json that A1.2 fixed from the draw's four
               training scans (C_word/pred/LABELMAP_dX__word_test30/rules.json, hashed in the freeze; nothing is re-derived
               and no training label is read); scored with lut 'id'.
Output: pred/<ARM>/{raw/word_<cid>.nii.gz, rec/<cid>.json, DONE.json} (DONE last; the sha256 of every input DONE record).
A real run needs the freeze, the upload gate, no look, the cutoff not passed and its inputs DONE.

usage:  runner only (wn_run.py launches '... wn_cpu_arms.py --arm MERGE_N_d1'); unit tests use --smoke-root.

Version of 2026-11-13.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import wn_common as T  # noqa: E402
import word_a1_cpu_arms as CA  # noqa: E402


def _done(path):
    p = Path(path)
    if not p.is_file():
        raise T.Refuse(f"input not DONE: {T.rel(p)}")
    return json.load(open(p)), T.sha256(p)


def frozen_rules(draw):
    """A1.2's rules of draw `draw` (the file hashed in the freeze), without its bookkeeping fields."""
    p = T.arms()[f"LABELMAP_{draw}"]["rules"]
    r = json.load(open(p))
    return {k: v for k, v in r.items() if k not in ("written", "draw")}, T.sha256(p)


def build_merge_n(root, ids, draw, log=print):
    import numpy as np
    out = T.pred_dir(f"MERGE_N_{draw}", root)
    if (out / "DONE.json").is_file():
        log(f"[skip] MERGE_N_{draw} already DONE")
        return json.load(open(out / "DONE.json"))
    V2 = CA.v2_common()
    isT = CA.site_lut()
    d0, h0 = _done(T.done_file("F0", root))
    d1, h1 = _done(T.done_file(f"F1b_{draw}", root))
    per, recs, t0 = {}, {}, time.time()
    for c in ids:
        Z, zi = CA._load_raw(T.pred_dir("F0", root) / "raw" / f"word_{c}.nii.gz", d0["per_scan_raw_sha256"][c])
        Fm, fi = CA._load_raw(T.pred_dir(f"F1b_{draw}", root) / "raw" / f"word_{c}.nii.gz", d1["per_scan_raw_sha256"][c])
        assert Z.shape == Fm.shape and np.allclose(zi.affine, fi.affine, atol=1e-3), f"{c}: F0 / F1b grids differ"
        _mt, mn = V2.merge_maps(Z, Fm, isT)
        assert mn.dtype == np.uint8 and mn.shape == Z.shape
        h = CA._save_like(mn, zi, out / "raw" / f"word_{c}.nii.gz")
        vox_ml = float(np.prod(zi.header.get_zooms()[:3])) / 1000.0
        st = CA.merge_stats(Z, Fm, mn, isT, vox_ml)
        T.write_json(out / "rec" / f"{c}.json", {"cid": c, "arm": f"MERGE_N_{draw}", "raw_sha256": h,
                                                 "f0_raw_sha256": d0["per_scan_raw_sha256"][c],
                                                 "f1b_raw_sha256": d1["per_scan_raw_sha256"][c], "written": T.now()} | st)
        per[c], recs[c] = h, st
        log(f"MERGE_N_{draw} {c}: {st['n_changed_vs_f1b']} voxels differ from F1b_{draw}")
    done = {"arm": f"MERGE_N_{draw}", "draw": draw, "kind": "cpu", "operator": CA.OPERATOR_MERGE,
            "v2_common_sha256": T.sha256(CA.A.V2_COMMON), "site_classes_T": [int(t) for t in np.flatnonzero(isT)],
            "lut_kind": "ts", "ids": list(ids), "n": len(ids),
            "inputs": {"F0_done_sha256": h0, f"F1b_{draw}_done_sha256": h1},
            "per_scan_raw_sha256": per, "per_scan_merge": recs, "minutes": round((time.time() - t0) / 60, 2),
            "finished": T.now(), "code_sha256": T.code_hashes(["wn_cpu_arms.py", "wn_common.py"])}
    return done


def build_labelmap(root, ids, draw, log=print):
    import numpy as np
    out = T.pred_dir(f"LABELMAP_{draw}", root)
    if (out / "DONE.json").is_file():
        log(f"[skip] LABELMAP_{draw} already DONE")
        return json.load(open(out / "DONE.json"))
    rules, hr = frozen_rules(draw)
    d0, h0 = _done(T.done_file("F0", root))
    lut = np.asarray(rules["lut"], np.uint8)
    merged = {int(c) for c in rules["merged"]}
    per, recs, t0 = {}, {}, time.time()
    for c in ids:
        Z, zi = CA._load_raw(T.pred_dir("F0", root) / "raw" / f"word_{c}.nii.gz", d0["per_scan_raw_sha256"][c])
        h = CA._save_like(lut[Z], zi, out / "raw" / f"word_{c}.nii.gz")
        cnt = np.bincount(Z.ravel(), minlength=256)
        st = {"relabelled_vox": {str(k): int(cnt[k]) for k in sorted(merged) if cnt[k]},
              "vox_ml": float(np.prod(zi.header.get_zooms()[:3])) / 1000.0}
        T.write_json(out / "rec" / f"{c}.json", {"cid": c, "arm": f"LABELMAP_{draw}", "raw_sha256": h,
                                                 "f0_raw_sha256": d0["per_scan_raw_sha256"][c], "rules_sha256": hr,
                                                 "written": T.now()} | st)
        per[c], recs[c] = h, st
    done = {"arm": f"LABELMAP_{draw}", "draw": draw, "kind": "cpu", "procedure": CA.PROCEDURE_LABELMAP,
            "rules_file": T.rel(T.arms()[f"LABELMAP_{draw}"]["rules"]), "rules_sha256": hr, "merged": rules["merged"],
            "train_ids": rules.get("train_ids"), "lut_kind": "id", "ids": list(ids), "n": len(ids),
            "inputs": {"F0_done_sha256": h0}, "per_scan_raw_sha256": per, "per_scan_relabel": recs,
            "minutes": round((time.time() - t0) / 60, 2), "finished": T.now(),
            "code_sha256": T.code_hashes(["wn_cpu_arms.py", "wn_common.py"])}
    return done


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=[a for a in T.TIER1 if T.arms()[a]["kind"] == "cpu"])
    ap.add_argument("--smoke-root", default=None, help="unit tests: a /tmp root with stand-in F0 / F1b predictions")
    ap.add_argument("--ids", default=None, help="smoke only")
    a = ap.parse_args(argv)
    arm = a.arm
    draw = T.arms()[arm]["draw"]
    real = a.smoke_root is None
    if real:
        T.kst_gate()
        if T.look_made():
            raise T.Refuse("a look exists")
        T.freeze_ok()
        T.check_frozen_model(arm)
        T.gate_uploaded()
        T.arm_allowed(arm)
        if not T.before_cutoff():
            raise T.Refuse("cutoff passed: no arm is started", code=T.CUTOFF_CODE)
        T.wait_disk(0.1, arm)
        root, ids = T.PRED, T.included_ids()
    else:
        assert str(Path(a.smoke_root).resolve()).startswith("/tmp/") and a.ids
        root, ids = Path(a.smoke_root), [c for c in a.ids.split(",") if c]
        assert not any(c in T.VAL_IDS for c in ids)
    log = (lambda m: T.log_event(m, T.pred_dir(arm, root).parent / f"{arm}.log"))
    done = (build_merge_n if arm.startswith("MERGE_N") else build_labelmap)(root, ids, draw, log)
    if real and not T.before_cutoff():
        raise T.Refuse("cutoff passed before the arm finished: no DONE.json", code=T.CUTOFF_CODE)
    if not T.done_file(arm, root).is_file():
        T.write_json(T.done_file(arm, root), dict(done, mode="real" if real else "smoke"))
    log(f"{arm} DONE")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except T.Refuse as e:
        print(f"[REFUSE] {e.msg}", file=sys.stderr)
        sys.exit(e.code)
