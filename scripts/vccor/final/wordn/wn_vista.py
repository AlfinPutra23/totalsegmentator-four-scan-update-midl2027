"""vccor/final WORD-N TIER 2: VISTA3D (MONAI bundle 0.5.8 through the FROZEN scripts/vccor/vista/vista_engine.py; WORD is
absent from VISTA3D's training lists: C_word/exposure_check.json). Plan C's WORD recipe (word_common.VISTA_RECIPE 'R1',
the vista block's automatic path) as reinstated in the design (GO-2 tier 2): VF5k_dX = 5,000 iterations x batch 1
at lr 5e-6 on WORD draw dX's four labelled scans (d1, d2, d3); VF5k_lrA_d1 = the same at lr 5e-5 (descriptive sensitivity).
Training labels = word_common.composite('vista') of the 12 draw scans (WORD GT of the adapted organs, adrenal split by V0's
own glands, V0 elsewhere), exactly word_vista.py's never-run 'labels' stage.

  --smoke            BEFORE the registration, on the 12 draw scans only (no validation image, no other label; GPU 1 under
                     the shared flock). The registered smoke test; tier 2 is NOT RUN unless it passes before the look:
                       S1 the bundle verifies (vista_common.check_bundle);
                       S2 V0 predicts the 12 draw scans (raw grid, affine == image)  -> vista/v0_draws/ (kept: the
                          composite labels are built from these maps);
                       S3 the composite labels of the 12 draw scans: the label-identity check PASSES for every adapted organ
                          present (word_vista.identity rule)  -> vista/labels/ (kept, hashed by the freeze; the 12 draw
                          labels are read again through word_common.train_label, which refuses any non-draw scan and logs
                          a [gt] line per read);
                       S4 a 20-iteration R1 training on draw d1 at lr 5e-6: finite losses; frozen tensors and buffers
                          unchanged (export check)  -> vsmoke/model/ (smoke only);
                       S5 that smoke export predicts one draw scan (0004): shape / affine == the image.
                     -> vsmoke/smoke.json {decision: VSMOKE_PASS | VSMOKE_FAIL, steps}; a VSMOKE line. A rerun is
                     possible only before the freeze: the earlier attempt (record, smoke model, V0 draw maps, labels) is
                     moved to vsmoke/attempt_<n>/ (kept, never deleted; a NOTE line; disclosed).
  --train --run R    AFTER the upload gate (the runner): R in VF5k_d1 / VF5k_d2 / VF5k_d3 / VF5k_lrA_d1 -> vista/models/R/
                     {weights_trainable.pt, export.json, train_log.jsonl, DONE.json}; needs VSMOKE_PASS and the frozen
                     labels; refused after the cutoff or after the look.

usage:  --smoke: through the ssl-wordn-vsmoke unit only (it sets CUDA_VISIBLE_DEVICES to GPU 1's UUID);
        --train: runner only (wn_run.py launches '... wn_vista.py --train --run <run>' with the GPU-1 environment).
Both refuse unless CUDA_VISIBLE_DEVICES is GPU 1's UUID, and both take the shared GPU-1 flock themselves.

Version of 2026-11-13.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import wn_common as T  # noqa: E402

W = T.W
SMOKE_ITERS = 20
SMOKE_PREDICT = "0004"
WORKERS = 4


def _vista():
    sys.path.insert(0, str(T.REPO / "scripts/vccor/vista"))
    import vista_common as V
    import vista_engine as E
    return V, E


def save_raw(seg, img, path):
    import nibabel as nib
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name.replace('.nii.gz', '')}.tmp{os.getpid()}.nii.gz")
    out = nib.Nifti1Image(np.asarray(seg).astype(np.uint8), img.affine)
    out.set_data_dtype(np.uint8)
    nib.save(out, str(tmp))
    os.replace(tmp, path)
    return T.sha256(path)


def identity(G, P, lut):
    """== word_vista.identity (frozen, not importable: it parses arguments at import)."""
    import numpy as np
    D = lut[P]
    n = W.N_ORG
    vol = np.bincount(D.ravel(), minlength=n + 1)[1:n + 1]
    out = {}
    for k in range(1, 14):
        m = G == k
        if not m.any():
            continue
        inter = np.bincount(D[m], minlength=n + 1)[1:n + 1]
        dice = 2 * inter / (m.sum() + vol)
        out[W.ORGANS[k - 1]] = bool(dice.argmax() + 1 == k and dice.max() > 0)
    return out


def infer_draws(eng, V, out_dir, log):
    import nibabel as nib
    import numpy as np
    per = {}
    for cid in W.draw_scans():
        img = nib.load(str(W.image_path(cid)))
        seg, tm = eng.predict(W.image_path(cid))
        assert seg.shape == tuple(img.shape[:3]), cid
        assert np.allclose(np.asarray(tm["affine_out"]), img.affine, atol=1e-3), cid
        if tm["n_nan"]:
            seg = np.where(seg == V.NAN_VALUE, 0, seg).astype(np.uint8)
        per[cid] = save_raw(seg, img, out_dir / "raw" / f"word_{cid}.nii.gz")
        log(f"V0 draw {cid}: {tm['infer_s']} s")
    T.write_json(out_dir / "DONE.json", {"run": "V0", "ids": W.draw_scans(), "per_scan_raw_sha256": per,
                                         "bundle_sha256": V.BUNDLE_SHA, "finished": T.now()})
    return per


def build_labels(log):
    """S3: composite VISTA training labels of the 12 draw scans (== word_vista.stage_labels)."""
    import nibabel as nib
    import numpy as np
    v0 = T.VISTA / "v0_draws"
    done = json.load(open(v0 / "DONE.json"))
    man = {"rule": "word_common.composite('vista'): WORD GT (adapted organs, adrenal split by V0's own glands) + V0 elsewhere",
           "labels": {}, "written": T.now()}
    bad = {}
    for cid in W.draw_scans():
        pth = v0 / "raw" / f"word_{cid}.nii.gz"
        assert T.sha256(pth) == done["per_scan_raw_sha256"][cid], f"V0 map of {cid} changed"
        img = nib.load(str(W.image_path(cid)))
        P = np.asarray(nib.load(str(pth)).dataobj).astype(np.uint8)
        raw, _ = W.train_label(cid, set(W.draw_scans()), lambda m: T.log_event(m))
        A = W.adapted_gt(W.analysis_lut()[raw])
        assert A.shape == P.shape == tuple(img.shape[:3]), cid
        idc = identity(A, P, W.vista_lut())
        nok = [o for o, ok in idc.items() if not ok]
        if nok:
            bad[cid] = nok
            continue
        pl, pr = W.model_adrenal_masks("vista", P)
        body = np.asarray(img.dataobj) > -300
        Gs, srec = W.split_adrenal(A, img.affine, pl, pr, body)
        Y, cnt = W.composite("vista", Gs, P)
        h = save_raw(Y, img, T.VISTA / "labels" / f"WORD_{cid}.nii.gz")
        man["labels"][cid] = cnt | {"label_sha256": h, "v0_raw_sha256": T.sha256(pth), "identity": "PASS",
                                    "adrenal_split": srec}
        log(f"[labels] VISTA WORD {cid}: identity PASS")
    man["identity_failures"] = bad
    T.write_json(T.VISTA / "labels" / "manifest.json", man)
    return man, bad


def train(run, n_iter, lr, draw, out_dir, log, smoke=False):
    import numpy as np
    import torch
    V, E = _vista()
    man = json.load(open(T.VISTA / "labels" / "manifest.json"))
    lab = W.draw(draw)
    dl = []
    for cid in lab:
        lp = T.VISTA / "labels" / f"WORD_{cid}.nii.gz"
        assert T.sha256(lp) == man["labels"][cid]["label_sha256"], f"{lp} changed after the labels stage"
        dl.append({"image": str(W.image_path(cid)), "label": str(lp)})
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"train start {run}: draw {draw} {lab}, recipe {T.VISTA_RECIPE}, lr {lr:g}, n_iter {n_iter}, device {dev}")
    t0 = time.time()
    net, loader, opt, loss_fn, scaler, label_set, meta = E.build_training(dl, lr, T.VISTA_RECIPE, n_iter=n_iter, device=dev,
                                                                           workers=WORKERS, cache_workers=WORKERS)
    it = E.infinite(loader)
    losses = []
    with open(out_dir / "train_log.jsonl", "a") as tlog:
        tlog.write(json.dumps({"start": T.now(), "run": run}) + "\n")
        for i in range(1, n_iter + 1):
            loss, npr = E.train_iteration(net, next(it), opt, loss_fn, scaler, label_set, dev)
            if loss is None or not np.isfinite(loss):
                raise RuntimeError(f"iteration {i}: loss {loss}")
            losses.append(loss)
            if i % 50 == 0 or i <= 2 or i == n_iter:
                tlog.write(json.dumps({"it": i, "loss_mean_last50": float(np.mean(losses[-50:])), "prompts": npr,
                                       "elapsed_min": (time.time() - t0) / 60, "gpu": E.gpu_mem_gb()}) + "\n")
                tlog.flush()
    ex = E.export_trainable(net, T.VISTA_RECIPE, out_dir / "weights_trainable.pt")
    if not ex["frozen_and_buffers_identical"]:
        raise RuntimeError("a frozen tensor or buffer changed during training")
    rec = {"run": run, "draw": draw, "labelled": lab, "recipe": T.VISTA_RECIPE, "lr": lr, "n_iter": n_iter,
           "train_meta": meta, "export": ex, "loss_first50": float(np.mean(losses[:50])),
           "loss_last50": float(np.mean(losses[-50:])), "minutes": (time.time() - t0) / 60, "gpu": E.gpu_mem_gb(),
           "bundle_sha256": V.BUNDLE_SHA, "labels_manifest_sha256": T.sha256(T.VISTA / "labels" / "manifest.json"),
           "smoke": smoke, "finished": T.now(), "code_sha256": T.code_hashes(["wn_vista.py", "wn_common.py"])}
    T.write_json(out_dir / "export.json", rec)
    T.write_json(out_dir / "DONE.json", {"run": run, "draw": draw, "lr": lr, "n_iter": n_iter, "smoke": smoke,
                                         "export_sha256": T.sha256(out_dir / "export.json"), "weights_sha256": ex["sha256"],
                                         "finished": T.now()})
    log(f"train done {run}: {rec['minutes']:.1f} min, loss {rec['loss_first50']:.4f} -> {rec['loss_last50']:.4f}")
    return rec


def smoke():
    T.kst_gate()
    if T.look_made():
        raise T.Refuse("a look exists")
    if T.FREEZE.exists():
        raise T.Refuse("the freeze exists: the smoke test is not rerun")
    sm = T.VSMOKE / "smoke.json"
    prev = [p for p in (sm, T.VSMOKE / "model", T.VISTA / "labels", T.VISTA / "v0_draws") if p.exists()]
    if prev:            # an earlier attempt (before the freeze) is kept, moved aside, never deleted; disclosed
        keep = T.VSMOKE / f"attempt_{len(list(T.VSMOKE.glob('attempt_*'))) + 1}"
        keep.mkdir(parents=True)
        for p in prev:
            os.replace(p, keep / (p.name if p.parent == T.VSMOKE else f"vista_{p.name}"))
        T.prereg_note("NOTE", f"VISTA3D smoke test rerun before the freeze; the earlier attempt is kept in {T.rel(keep)}")
    attempt = len(list(T.VSMOKE.glob("attempt_*"))) + 1 if T.VSMOKE.exists() else 1
    T.disk_gate(1.0, "VISTA3D smoke test")
    miss = [c for c in W.draw_scans() if not W.image_path(c).is_file()]
    if miss:
        raise T.Refuse(f"draw images missing: {miss}")
    log = (lambda m: T.log_event(m, T.VSMOKE / "smoke.log"))
    steps = {}
    fd = T.gpu_session(log=log, cutoff_aware=False)
    try:
        V, E = _vista()
        try:
            V.check_bundle(log)
            steps["S1_bundle"] = "PASS"
        except SystemExit as e:
            steps["S1_bundle"] = f"FAIL: {e}"
        if steps["S1_bundle"] == "PASS":
            try:
                eng = E.Inference("cuda")
                infer_draws(eng, V, T.VISTA / "v0_draws", log)
                del eng
                steps["S2_v0_draws"] = "PASS"
            except Exception as e:                       # noqa: BLE001  (a smoke failure is a registered outcome)
                steps["S2_v0_draws"] = f"FAIL: {type(e).__name__}: {str(e)[:200]}"
        if steps.get("S2_v0_draws") == "PASS":
            try:
                _man, bad = build_labels(log)
                steps["S3_labels_identity"] = "PASS" if not bad else f"FAIL: identity {bad}"
            except Exception as e:                       # noqa: BLE001
                steps["S3_labels_identity"] = f"FAIL: {type(e).__name__}: {str(e)[:200]}"
        if steps.get("S3_labels_identity") == "PASS":
            try:
                train("SMOKE_d1", SMOKE_ITERS, 5e-6, "d1", T.VSMOKE / "model", log, smoke=True)
                steps["S4_train_20it"] = "PASS"
            except Exception as e:                       # noqa: BLE001
                steps["S4_train_20it"] = f"FAIL: {type(e).__name__}: {str(e)[:200]}"
        if steps.get("S4_train_20it") == "PASS":
            try:
                import nibabel as nib
                import numpy as np
                eng = E.Inference("cuda", weights=T.VSMOKE / "model" / "weights_trainable.pt")
                img = nib.load(str(W.image_path(SMOKE_PREDICT)))
                seg, tm = eng.predict(W.image_path(SMOKE_PREDICT))
                ok = seg.shape == tuple(img.shape[:3]) and np.allclose(np.asarray(tm["affine_out"]), img.affine, atol=1e-3)
                steps["S5_predict"] = "PASS" if ok else "FAIL: shape / affine"
                steps["S5_detail"] = {"cid": SMOKE_PREDICT, "fg_vox": int((seg > 0).sum()), "timing": tm["infer_s"]}
            except Exception as e:                       # noqa: BLE001
                steps["S5_predict"] = f"FAIL: {type(e).__name__}: {str(e)[:200]}"
    finally:
        T.release_gpu_lock(fd)
    need = ["S1_bundle", "S2_v0_draws", "S3_labels_identity", "S4_train_20it", "S5_predict"]
    dec = "VSMOKE_PASS" if all(steps.get(k) == "PASS" for k in need) else "VSMOKE_FAIL"
    rec = {"decision": dec, "attempt": attempt, "steps": steps, "draw_scans_only": W.draw_scans(),
           "validation_images_used": False, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "written": T.now(), "code_sha256": T.code_hashes(["wn_vista.py", "wn_common.py"])}
    T.write_json(sm, rec, mode=0o444)
    log_smoke_line()
    return 0 if dec == "VSMOKE_PASS" else 6


def log_smoke_line():
    """The VSMOKE line of smoke.json (written once; also written by the freeze if a crash left it missing)."""
    sm = T.VSMOKE / "smoke.json"
    h = T.sha256(sm)
    if not any(f"VSMOKE sha256 {h} " in ln for ln in T.prereg_lines()):
        T.prereg_line("VSMOKE", sm, f"wordn VISTA3D smoke test on the 12 draw scans (before registration): "
                                    f"{json.load(open(sm))['decision']}")


def train_real(run):
    T.kst_gate()
    if T.look_made():
        raise T.Refuse("a look exists: no training after the look")
    fz = T.freeze_ok()
    T.check_versions(fz.get("versions"))
    T.gate_uploaded()
    T.arm_allowed(run)
    if T.declared_not_run("V0"):
        raise T.Refuse(f"{run}: V0 was declared NOT RUN, so V-P1 cannot be read: no training (N7)")
    if not T.before_cutoff():
        raise T.Refuse("cutoff passed: no training is started", code=T.CUTOFF_CODE)
    md = T.VISTA / "models" / run
    if (md / "DONE.json").is_file():
        print(f"{run}: DONE")
        return 0
    T.wait_disk(1.0, f"training {run}")
    T.gpu_identity_gate()
    V, _E = _vista()
    V.check_bundle()
    s = T.VF_RUNS[run]
    log = (lambda m: T.log_event(m, md.parent / f"{run}.log"))
    fd = T.gpu_session(log=log)
    try:
        if T.declared_not_run(run):
            raise T.Refuse(f"{run} was declared NOT RUN while waiting for the GPU")
        train(run, s["n_iter"], s["lr"], s["draw"], md, log)
    finally:
        T.release_gpu_lock(fd)
    if not T.before_cutoff():
        log(f"cutoff passed during {run}'s training: its arm is NOT RUN")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--run", default=None, choices=list(T.VF_RUNS))
    a = ap.parse_args()
    try:
        if a.smoke and not a.train and a.run is None:
            return smoke()
        if a.train and a.run and not a.smoke:
            return train_real(a.run)
        raise SystemExit(__doc__)
    except T.Refuse as e:
        print(f"[REFUSE] {e.msg}", file=sys.stderr)
        return e.code


if __name__ == "__main__":
    sys.exit(main())
