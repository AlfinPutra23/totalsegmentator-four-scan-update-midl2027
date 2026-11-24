"""vccor/final LCTSC PREP (CPU only, nice 15; runs in the SSL .venv; idempotent). Reads LCTSC TRAINING labels of the 12 draw
patients only (data_lctsc/labels_train_v2, written by lctsc_convert.py --stage labels). No test label exists on disk.

Stages (each in its own process with --stage all):
  capture   TS-full's own CPU pipeline on the 12 draw patients (fin_nn.ts_capture == the WORD / xdata capture, ORIGINAL
            weights): 1.5 mm input image + all five part predictions + merged raw output -> prep/ts_cpu/<pid>/
  labels    per draw patient: GT (analysis ids) on TS's 1.5 mm grid (tsft_full_common.gt_to_ts_grid), the merged 1.5 mm
            parts must round-trip exactly to TS's merged raw output, the recomputed 1.5 mm image must equal the captured
            one; composite targets for parts 291 and 293 (lctsc_common.composite) -> prep/labels/{gt_ts,part291,part293}/
            + prep/labels/manifest.json. Also a training-set F0 agreement record (TS's merged raw output vs the GT on the
            native grid, Dice per structure): an orientation / mapping check on training patients only, descriptive.
  datasets  Dataset861-863 (part 291) and 871-873 (part 293), draws d1-d3: captured images + composite targets, TS's
            label space and plans moved on (fin_nn == WORD / xdata), 3d_fullres preprocessing, in K_lctsc/nnunet
  done      prep/DONE_datasets.json (sha256 of every produced file; lctsc_train.py verifies it)

usage:  CUDA_VISIBLE_DEVICES="" nice -n 15 .venv/bin/python -I -B scripts/vccor/final/lctsc/lctsc_prep.py --stage all

Version of 2026-11-13.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
FINAL = HERE.parent
for p in (str(HERE), str(FINAL), str(FINAL.parent / "tsft_full")):
    if p not in sys.path:
        sys.path.insert(0, p)
import final_common as F  # noqa: E402  (frozen; read-only import)
import fin_nn as N  # noqa: E402  (frozen; read-only import)
import lctsc_common as C  # noqa: E402
import lctsc_log as L  # noqa: E402
import lctsc_runs as R  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--stage", required=True, choices=["all", "capture", "labels", "datasets", "done"])
ap.add_argument("--ts_threads", type=int, default=6)
ap.add_argument("--shard", default=None, help="capture only: i/n, patients [i::n] of the sorted draw patients (parallel CPU workers)")
args = ap.parse_args()
PREP = R.PREP


def cap_dir(pid):
    return PREP / "ts_cpu" / pid


def stage_capture():
    missing = []
    todo = C.draw_patients()
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        todo = todo[i::n]
    for pid in todo:
        if N.capture_ok(cap_dir(pid)):
            continue
        if not C.image_path(pid).is_file():
            missing.append(pid)
            L.event(f"[capture] {pid}: image not converted yet, skipped")
            continue
        r = N.ts_capture(C.image_path(pid), cap_dir(pid), args.ts_threads)
        L.event(f"[capture] {pid} wall {r['wall_s']:.0f}s")
    if missing:
        raise SystemExit(f"capture incomplete: {missing}")
    if not all(N.capture_ok(cap_dir(c)) for c in C.draw_patients()):
        L.event(f"[capture] shard {args.shard} done; other patients still pending")
        return
    N.wj(PREP / "ts_cpu" / "CAPTURE_DONE.json", {"ids": C.draw_patients(), "finished": L.now(),
                                                 "sha256": {c: {f.name: L.sha256(f) for f in sorted(cap_dir(c).iterdir()) if f.is_file()}
                                                            for c in C.draw_patients()}})


def part_to_total_luts():
    import numpy as np
    from totalsegmentator.map_to_binary import class_map, class_map_5_parts, map_taskid_to_partname_ct
    inv = {v: k for k, v in class_map["total"].items()}
    out = {}
    for t in (291, 292, 293, 294, 295):
        lut = np.zeros(256, np.uint8)
        for j, n in class_map_5_parts[map_taskid_to_partname_ct[t]].items():
            lut[j] = inv[n]
        out[t] = lut
    return out


def dice(a, b):
    s = int(a.sum()) + int(b.sum())
    return None if s == 0 else 200.0 * int((a & b).sum()) / s


def stage_labels():
    import nibabel as nib
    import numpy as np
    import tsft_full_common as Q
    from totalsegmentator.alignment import as_closest_canonical, undo_canonical
    from totalsegmentator.resampling import change_spacing
    p2t = part_to_total_luts()
    tl = C.ts_total_lut()
    ld = PREP / "labels"
    man_p = ld / "manifest.json"
    man = json.load(open(man_p)) if man_p.is_file() else {"scans": {}, "gt_opened_training": []}
    for pid in C.draw_patients():
        if pid in man["scans"] and all((ld / k / f"{pid}.nii.gz").is_file() for k in ("gt_ts", "part291", "part293")):
            continue
        if not N.capture_ok(cap_dir(pid)):
            L.event(f"[labels] {pid}: capture not finished, skipped")
            continue
        d = cap_dir(pid)
        cap, img_orig, img_in, img_rsp = N.ts_grid_check(C.image_path(pid), d)
        lbl = nib.load(str(C.train_label_path(pid)))
        man["gt_opened_training"] = sorted(set(man["gt_opened_training"]) | {pid})
        raw = np.rint(np.asanyarray(lbl.dataobj)).astype(np.uint8)
        assert raw.max() <= C.N_ORG, (pid, int(raw.max()))
        assert img_orig.shape[:3] == raw.shape and np.allclose(img_orig.affine, lbl.affine, atol=1e-3), f"{pid}: GT grid != image"
        ts_raw = np.asarray(nib.load(str(d / "raw_total.nii.gz")).dataobj).astype(np.uint8)
        f0 = tl[ts_raw]
        f0_dice = {C.ORGANS[o - 1]: dice(f0 == o, raw == o) for o in range(1, C.N_ORG + 1)}
        A = raw.copy()
        A[~np.isin(A, C.ADAPTED)] = 0
        lab_c = as_closest_canonical(nib.Nifti1Image(A, img_orig.affine))
        G = Q.gt_to_ts_grid(lab_c, img_in, img_rsp, n_labels=C.N_ORG)
        parts = {t: np.asarray(nib.load(str(d / f"part_{t}.nii.gz")).dataobj).astype(np.uint8) for t in (291, 292, 293, 294, 295)}
        Tf = np.zeros(parts[291].shape, np.uint8)
        for t in (291, 292, 293, 294, 295):
            nz = parts[t] > 0
            Tf[nz] = p2t[t][parts[t][nz]]
        back = change_spacing(nib.Nifti1Image(Tf, cap.affine), [1.5] * 3, img_in.shape, order=0, dtype=np.uint8, nr_cpus=1,
                              force_affine=img_in.affine, use_gpu=False)
        back = np.asarray(undo_canonical(back, img_orig).dataobj).astype(np.uint8)
        rt = int((back != ts_raw).sum())
        assert rt == 0, f"{pid}: merged 1.5 mm map round trip differs from TS's merged output on {rt} voxels"
        Y1, c1 = C.composite(291, G, parts[291])
        Y3, c3 = C.composite(293, G, parts[293])
        rec = {"gt_vox_1p5": np.bincount(G.ravel(), minlength=C.N_ORG + 1)[1:].tolist(), "composite_291": c1, "composite_293": c3,
               "merged_roundtrip_n_differ": rt, "shape_1p5": list(G.shape), "f0_dice_native_training": f0_dice,
               "capture_sha256": {f"part_{t}": L.sha256(d / f"part_{t}.nii.gz") for t in (291, 292, 293, 294, 295)}
               | {"image_1p5": L.sha256(d / "image_1p5.nii.gz")},
               "sha256": {"gt_ts": N.save_nii(G, cap.affine, ld / "gt_ts" / f"{pid}.nii.gz"),
                          "part291": N.save_nii(Y1, cap.affine, ld / "part291" / f"{pid}.nii.gz"),
                          "part293": N.save_nii(Y3, cap.affine, ld / "part293" / f"{pid}.nii.gz")}}
        man["scans"][pid] = rec
        N.wj(man_p, man)
        L.event(f"[labels] {pid}: grid {list(G.shape)}, round trip exact, F0 training Dice "
                f"{ {k: (round(v, 1) if v is not None else None) for k, v in f0_dice.items()} }")


def stage_datasets():
    N.set_env(R.NN_ROOT)
    ld = PREP / "labels"
    man = json.load(open(ld / "manifest.json"))
    info_p = PREP / "datasets.json"
    info = json.load(open(info_p)) if info_p.is_file() else {}
    for did, spec in R.DATASETS.items():
        if spec["name"] in info:
            continue
        ids, t = C.draw(spec["draw"]), spec["part"]
        cases = [(f"LCTSC_{c}", cap_dir(c) / "image_1p5.nii.gz", man["scans"][c]["capture_sha256"]["image_1p5"],
                  ld / f"part{t}" / f"{c}.nii.gz", man["scans"][c]["sha256"][f"part{t}"]) for c in ids]
        info[spec["name"]] = N.ts_part_dataset(R.NN_ROOT, did, spec["name"], t, cases,
                                               f"vccor/final LCTSC: draw {spec['draw']} ({', '.join(ids)}), part {t}; composite "
                                               f"labels (LCTSC GT for the adapted structures, TS-{t} elsewhere)")
        N.wj(info_p, info)
        L.event(f"[datasets] {spec['name']}: {info[spec['name']]['n_cases']} cases")


def stage_done():
    info = json.load(open(PREP / "datasets.json"))
    files = N.files_of(info)
    bad = [f for f, h in files.items() if L.sha256(f) != h]
    assert not bad, f"files changed since they were hashed: {bad[:5]}"
    gt = json.load(open(PREP / "labels" / "manifest.json")).get("gt_opened_training", [])
    N.wj(PREP / "DONE_datasets.json", {"files_sha256": files, "datasets": sorted(info), "gt_opened_training_labels": gt,
                                       "finished": L.now()})
    L.event(f"[done] LCTSC prep: {len(files)} files hashed")


def main():
    if args.stage == "all":
        for st in ("capture", "labels", "datasets", "done"):
            subprocess.run([sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--stage", st,
                            "--ts_threads", str(args.ts_threads)], check=True)
        return
    L.event(f"LCTSC prep stage={args.stage} pid={os.getpid()}")
    {"capture": stage_capture, "labels": stage_labels, "datasets": stage_datasets, "done": stage_done}[args.stage]()


if __name__ == "__main__":
    main()
