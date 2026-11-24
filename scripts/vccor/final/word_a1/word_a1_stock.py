"""vccor/final plan C, amendment word A1.2: the STOCK arm on WORD, draws d1 / d2 / d3 = nnU-Net's stock pretrained fine-tuning
recipe of TS part 291, as registered for arm STOCK in outputs/results/vccor/peft/prereg_peft.json (variants.STOCK,
stock_recipe) and plan B (bsl_common.RECIPES['STOCK']): one network on the dataset's own classes, TS part 291's plans moved on
(TS291StockPlans), init = TS part 291 minus '.seg_layers.' (-pretrained_weights semantics: the head is re-initialised),
nnUNetTrainer_250epochs_NoMirroring, SGD lr 1e-2 (nnU-Net's default), 250 epochs x 250 iterations, fold 'all', seeds 1337;
prediction with nnUNetPredictor (tile 0.5, gaussian, mirroring OFF).

The frozen plan-C scripts run UNCHANGED (the E3b pattern, scripts/vccor/gaps_2026_11/e3b/e3b_common.install()): install()
adds to the already-imported plan-C modules, in this process only,
  word_common.RUNS['STOCK_dX']         recipe 'STOCK', stock_dataset 824 / 825 / 826, ds 'word', n_classes 15 (WORD's analysis
                                       label space), labelled = draw dX
  word_common.DATASETS[824 / 825 / 826] Dataset824_FINWORDSTOCK_d1 / ..._d2 / ..._d3
  word_common.NN_ROOT                  C_word/nnunet_stock (the F1b datasets 821-823 in C_word/nnunet are not touched)
  word_common.verify_datasets          STOCK_dX is checked against C_word/prep_stock/DONE_stock_dX.json (F1b: unchanged)
  fin_registry.MODELS['word_STOCK_dX'] kind 'export', dir C_word/models/STOCK_dX (written by train_final.py)
and asserts that every F1b entry is unchanged.

Stages (one runner job each):
  prep     (CPU) the draw's STOCK dataset (== bsl_prep.stock_dataset and peft prep_peft.py --stage stock): imagesTr =
           symlinks to the draw's 4 raw WORD images (sha256 == prep/images.json); labelsTr = the draw's 4 WORD training
           labels, read in memory from the main archive by the frozen word_common.train_label (it refuses any non-draw scan;
           these are 4 of the 12 labels already read on 2026-10-23; each read is logged), mapped to WORD's analysis ids by
           word_common.analysis_lut (WORD's 16 classes with colon + rectum merged: the label space the test references are
           scored in); dataset.json = background + the 15 analysis organs, NibabelIOWithReorient; fingerprint; TS part 291's
           plans moved on and 3d_fullres preprocessing by the frozen fin_nn.plans_and_preprocess -> prep_stock/
           DONE_stock_dX.json (sha256 of every produced file)
  train    (GPU) train_final.py --plan word --run STOCK_dX, unchanged (its STOCK branch is the registered recipe)
  predict  (GPU) predict_final.py --plan word --job word_STOCK_dX__word_test30, unchanged -> C_word/pred/
           word_STOCK_dX__word_test30/ (the dataset's own ids = analysis ids on the raw grid; scored with lut 'id')
A real run first checks the registration (word_a1_common.real_job_guard: no look, OSF gate, frozen code), so a hand start
before the registration refuses. Module level holds no side effect (nnU-Net's worker processes may re-import the main module).

usage (the runner):  .venv/bin/python -B scripts/vccor/final/word_a1/word_a1_stock.py --stage prep --draw d1
                     .venv/bin/python -B scripts/vccor/final/word_a1/word_a1_stock.py --stage train --draw d1 --device cuda \\
                         --n_proc_da 8
                     .venv/bin/python -B scripts/vccor/final/word_a1/word_a1_stock.py --stage predict --draw d1 --device gpu
unit checks:         ... --stage train|predict --draw d1 --load_only (executes the frozen script's module code only, on CPU:
                     no training, no inference, no file written)
smoke (tests):       ... --stage prep --draw d1 --smoke <scratch under /tmp> --smoke-images <dir> --smoke-labels <dir>

Version of 2026-11-06.
"""
import argparse
import copy
import json
import os
import runpy
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

DRAWS = ["d1", "d2", "d3"]
RUN = {d: f"STOCK_{d}" for d in DRAWS}
MODEL = {d: f"word_STOCK_{d}" for d in DRAWS}
JOB = {d: f"word_STOCK_{d}__word_test30" for d in DRAWS}
PLANS_ID = "TS291StockPlans"
REGISTERED = {"recipe": "STOCK", "trainer": "nnUNetTrainer_250epochs_NoMirroring", "lr": 1e-2, "epochs": 250,
              "iters_per_epoch": 250, "fold": "all", "init": "TS part 291 minus '.seg_layers.'", "seed": 1337,
              "plans": PLANS_ID, "mirroring": "off (trainer and predictor)"}


def modules():
    import word_a1_common as A
    import fin_registry as RG
    return A, A.F, A.W, RG


def ds_name(d):
    A, _, _, _ = modules()
    return f"Dataset{A.STOCK_DATASETS[d]}_FINWORDSTOCK_{d}"


def default_dirs():
    _, _, W, _ = modules()
    return W.DIR / "nnunet_stock", W.DIR / "prep_stock"


def _snapshot(W, RG):
    return copy.deepcopy({"RUNS": {k: v for k, v in W.RUNS.items() if k.startswith("F1b_")},
                          "DATASETS": {k: v for k, v in W.DATASETS.items() if k in (821, 822, 823)},
                          "MODELS": {k: {kk: str(vv) for kk, vv in v.items()} for k, v in RG.MODELS.items()
                                     if k.startswith("word_F1b_")}})


def install(nn_root=None, prep_dir=None):
    """Add the WORD STOCK runs to word_common / fin_registry in THIS process (see the module docstring). Idempotent."""
    A, F, W, RG = modules()
    if getattr(W, "_A12_STOCK_INSTALLED", None):
        return W._A12_STOCK_INSTALLED
    dn, dp = default_dirs()
    nn_root = Path(nn_root) if nn_root else dn
    prep_dir = Path(prep_dir) if prep_dir else dp
    before = _snapshot(W, RG)
    for d in DRAWS:
        did = A.STOCK_DATASETS[d]
        assert did not in W.DATASETS and RUN[d] not in W.RUNS and MODEL[d] not in RG.MODELS, (did, RUN[d], MODEL[d])
        W.DATASETS[did] = dict(name=ds_name(d), part=None, draw=d)
        W.RUNS[RUN[d]] = dict(model=MODEL[d], recipe="STOCK", ts_datasets={}, stock_dataset=did, ds="word",
                              n_classes=A.STOCK_N_CLASSES, labelled=d)
        RG.MODELS[MODEL[d]] = dict(kind="export", site="word", dir=RG.C_DIR / f"models/{RUN[d]}",
                                   label=f"A1.2 STOCK: nnU-Net's pretrained fine-tuning recipe (TS-291 init without the "
                                         f"segmentation layers, lr 1e-2, 250 epochs) on WORD K = 4 training draw {d}")
    W.NN_ROOT = nn_root
    orig = W.verify_datasets

    def verify_datasets(run, recipe, prep=None):
        if run not in RUN.values():
            return orig(run, recipe, prep)
        import bsl_common as B
        if B.RECIPES[recipe]["kind"] != "stock":
            raise SystemExit(f"[REFUSE] {run} uses the STOCK recipe, not {recipe}")
        d = W.RUNS[run]["labelled"]
        p = Path(prep or prep_dir) / f"DONE_stock_{d}.json"
        if not p.is_file():
            raise SystemExit(f"[REFUSE] {F.rel(p)} missing: run word_a1_stock.py --stage prep --draw {d} first")
        rec = json.load(open(p))
        bad = [f for f, h in rec["files_sha256"].items() if F.sha256(f) != h]
        if bad:
            raise SystemExit(f"[REFUSE] STOCK training dataset files changed since the prep: {bad[:3]}")
        if W.dataset_name(W.RUNS[run]["stock_dataset"]) not in rec["datasets"] or rec["ids"] != W.draw(d):
            raise SystemExit(f"[REFUSE] the STOCK prep record of {d} does not hold {W.dataset_name(W.RUNS[run]['stock_dataset'])} "
                             f"for the draw {W.draw(d)}")
        return True

    W.verify_datasets = verify_datasets
    assert _snapshot(W, RG) == before, "install() changed an F1b entry"
    W._A12_STOCK_INSTALLED = {"nn_root": str(nn_root), "prep_dir": str(prep_dir)}
    return W._A12_STOCK_INSTALLED


# ---------------------------------------------------------------------------------------------------- prep
def build_dataset(d, nn_root, prep_dir, image_fn, label_fn, image_shas=None, log=print):
    """The STOCK dataset of draw d (see the module docstring); returns the DONE record (written last)."""
    import nibabel as nib
    import numpy as np
    A, F, W, _ = modules()
    import fin_nn as N
    nn_root, prep_dir = Path(nn_root), Path(prep_dir)
    done_p = prep_dir / f"DONE_stock_{d}.json"
    if done_p.is_file():
        log(f"[skip] STOCK prep {d} already DONE")
        return json.load(open(done_p))
    N.set_env(nn_root)
    did, dsn = A.STOCK_DATASETS[d], ds_name(d)
    ids = W.draw(d)
    rd = nn_root / "nnUNet_raw" / dsn
    (rd / "imagesTr").mkdir(parents=True, exist_ok=True)
    (rd / "labelsTr").mkdir(parents=True, exist_ok=True)
    alut = W.analysis_lut()
    rec = {}
    for cid in ids:
        src = Path(image_fn(cid))
        hi = F.sha256(src)
        if image_shas is not None and image_shas.get(cid) != hi:
            raise A.Refuse(f"STOCK prep {d}: image {src} sha256 {hi[:16]} != prep/images.json")
        fi = rd / "imagesTr" / f"WORD_{cid}_0000.nii.gz"
        if fi.is_symlink() or fi.exists():
            fi.unlink()
        os.symlink(src, fi)
        img = nib.load(str(src))
        raw, lbl = label_fn(cid)
        raw = np.asarray(raw)
        assert raw.shape == img.shape[:3] and np.allclose(lbl.affine, img.affine, atol=1e-3), f"{cid}: label grid != image"
        assert raw.min() >= 0 and raw.max() <= 16, (cid, int(raw.min()), int(raw.max()))
        lab = alut[raw.astype(np.uint8)]
        h = N.save_nii(lab.astype(np.uint8), img.affine, rd / "labelsTr" / f"WORD_{cid}.nii.gz")
        rec[cid] = {"label_sha256": h, "image": str(src), "image_sha256": hi,
                    "analysis_ids_present": sorted(int(x) for x in np.unique(lab) if x > 0)}
        log(f"[stock prep] {d} {cid}: label written in WORD's analysis ids ({len(rec[cid]['analysis_ids_present'])} organs)")
    labels = {"background": 0} | {W.ORGANS[o - 1]: o for o in range(1, W.N_ORG + 1)}
    dj = {"name": dsn, "description": f"vccor/final plan C amendment word A1.2 STOCK: WORD draw {d} ({', '.join(ids)}), raw "
                                      f"images + raw-grid WORD training labels in the 15 analysis ids; nnU-Net's pretrained "
                                      f"fine-tuning recipe from TS part 291", "channel_names": {"0": "CT"},
          "labels": labels, "numTraining": len(ids), "file_ending": ".nii.gz",
          "overwrite_image_reader_writer": "NibabelIOWithReorient"}
    json.dump(dj, open(rd / "dataset.json", "w"), indent=1)
    r = N.plans_and_preprocess(nn_root, did, dsn, 291, PLANS_ID, len(ids))
    r["labels"] = rec
    info = {dsn: r}
    N.wj(prep_dir / f"datasets_stock_{d}.json", info)
    files = N.files_of(info)
    bad = [f for f, h in files.items() if F.sha256(f) != h]
    assert not bad, f"files changed since they were hashed: {bad[:5]}"
    done = {"draw": d, "ids": ids, "dataset": dsn, "dataset_id": did, "datasets": [dsn], "files_sha256": files,
            "labels": rec, "label_space": "word_common.analysis_lut (15 analysis organs; colon + rectum -> colon_rectum)",
            "plans": PLANS_ID, "gt_opened_training_labels": ids, "registered": REGISTERED, "finished": F.now()}
    N.wj(done_p, done)
    log(f"[stock prep] {d} DONE: {len(files)} files hashed")
    return done


# ---------------------------------------------------------------------------------------------------- train / predict
def load_train(d, device, n_proc_da):
    """Execute train_final.py's module code (unchanged) for STOCK_<d> after install(); returns its globals."""
    A, F, W, _ = modules()
    install()
    src = A.FINAL / "train_final.py"
    argv = sys.argv
    sys.argv = [str(src), "--plan", "word", "--run", RUN[d], "--device", device, "--n_proc_da", str(n_proc_da)]
    try:
        g = runpy.run_path(str(src), run_name="word_a12_train_final")
    finally:
        sys.argv = argv
    assert g["C"] is W and g["RUN"] is W.RUNS[RUN[d]] and g["RECIPE"] == "STOCK" and g["STOCK"] and g["PARTS"] == []
    assert g["TRAINER"] == REGISTERED["trainer"] and g["EPOCHS"] == REGISTERED["epochs"] and g["FOLD_STOCK"] == "all"
    assert g["V"]["lr"] == REGISTERED["lr"] and g["V"]["trainable"] == "all" and g["SEED"] == REGISTERED["seed"]
    assert g["NN_ROOT"] == W.NN_ROOT and g["EXPORT"] == W.MODELS / RUN[d] and g["plans_id"](None) == PLANS_ID
    return g


def load_predict(d, device):
    """Execute predict_final.py's module code (unchanged) for word_STOCK_<d>__word_test30 after install()."""
    A, F, W, RG = modules()
    install()
    src = A.FINAL / "predict_final.py"
    argv = sys.argv
    sys.argv = [str(src), "--plan", "word", "--job", JOB[d], "--device", device]
    try:
        g = runpy.run_path(str(src), run_name="word_a12_predict_final")
    finally:
        sys.argv = argv
    assert g["RG"] is RG and g["MODEL"] == MODEL[d] and g["TARGET"] == "word_test30" and not g["SMOKE"]
    assert g["PD"] == RG.C_DIR / "pred" / JOB[d] and g["M"]["dir"] == RG.C_DIR / f"models/{RUN[d]}"
    assert g["IDS"] == list(W.split()["test"])
    return g


def describe(g, d, stage):
    _, F, W, _ = modules()
    if stage == "train":
        return {"run": RUN[d], "recipe": g["RECIPE"], "trainer": g["TRAINER"], "epochs": g["EPOCHS"], "lr": g["V"]["lr"],
                "fold": g["FOLD_STOCK"], "seed": g["SEED"], "plans": g["plans_id"](None), "nn_root": str(g["NN_ROOT"]),
                "export": str(g["EXPORT"]), "stock_dataset": W.dataset_name(g["RUN"]["stock_dataset"]),
                "n_classes": g["RUN"]["n_classes"], "labelled": W.labelled_of(RUN[d]),
                "train_final_sha256": F.sha256(F.REPO / "scripts/vccor/final/train_final.py")}
    return {"job": JOB[d], "model": g["MODEL"], "target": g["TARGET"], "pred_dir": str(g["PD"]), "n": len(g["IDS"]),
            "model_dir": str(g["M"]["dir"]), "predict_final_sha256": F.sha256(F.REPO / "scripts/vccor/final/predict_final.py")}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["prep", "train", "predict"])
    ap.add_argument("--draw", required=True, choices=DRAWS)
    ap.add_argument("--device", default=None)
    ap.add_argument("--n_proc_da", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--load_only", action="store_true", help="unit checks: execute the frozen script's module code only")
    ap.add_argument("--smoke", default=None, help="tests: scratch root under /tmp (prep only)")
    ap.add_argument("--smoke-images", default=None, help="tests: dir with word_<cid>.nii.gz images")
    ap.add_argument("--smoke-labels", default=None, help="tests: dir with word_<cid>_label.nii.gz WORD raw-id labels")
    a = ap.parse_args(argv)
    A, F, W, _ = modules()
    if a.load_only:
        assert a.stage in ("train", "predict") and a.smoke is None
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        g = load_train(a.draw, "cpu", 0) if a.stage == "train" else load_predict(a.draw, "cpu")
        print(json.dumps(describe(g, a.draw, a.stage), default=str))
        return 0
    if a.smoke is not None:
        assert a.stage == "prep" and a.smoke_images and a.smoke_labels
        root = Path(a.smoke).resolve()
        if not str(root).startswith("/tmp/") or A.touches_real_labels_or_look(root):
            print(f"[REFUSE] --smoke must be under /tmp: {root}", file=sys.stderr)
            return 2
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        F.cap_threads(a.threads)
        import nibabel as nib
        import numpy as np

        def lab(cid):
            im = nib.load(str(Path(a.smoke_labels) / f"word_{cid}_label.nii.gz"))
            return np.rint(np.asarray(im.dataobj)).astype(np.uint8), im

        rec = build_dataset(a.draw, root / "nnunet_stock", root / "prep_stock",
                            lambda cid: Path(a.smoke_images) / f"word_{cid}.nii.gz", lab, None,
                            log=lambda m: print(m, file=sys.stderr))
        print(json.dumps({"dataset": rec["dataset"], "n_files": len(rec["files_sha256"]), "ids": rec["ids"]}))
        return 0
    log = F.Log(F.LOGD / f"final_word_a12_stock_{a.stage}_{a.draw}.log")
    try:
        A.real_job_guard(Path(__file__).resolve(), log)
        F.check_code_frozen([HERE / "word_a1_common.py", HERE / "word_a1_stock.py"], log)
    except A.Refuse as e:
        log(f"[REFUSE] {e.msg}")
        return e.code
    except SystemExit as e:
        log(f"[REFUSE] {e}")
        return 2
    if a.stage == "prep":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        F.cap_threads(a.threads)
        F.be_nice(15)
        shas = json.load(open(W.PREP / "images.json"))["sha256"]
        allowed = set(W.draw(a.draw))
        build_dataset(a.draw, *default_dirs(), W.image_path, lambda cid: W.train_label(cid, allowed, log), shas, log)
        return 0
    if a.stage == "train":
        g = load_train(a.draw, a.device or "cuda", a.n_proc_da)
        g["LOG"](f"A1.2 STOCK train {RUN[a.draw]} (draw {' '.join(W.draw(a.draw))}) via train_final.py sha "
                 f"{F.sha256(A.FINAL / 'train_final.py')[:16]}")
        g["main"]()
        done = W.MODELS / RUN[a.draw] / "DONE.json"
        if done.is_file():
            dd = json.load(open(done))
            assert dd["model"] == MODEL[a.draw] and dd["variant"] == "STOCK" and dd["resolved"]["kind"] == "stock", dd["model"]
            assert dd["resolved"]["n_classes"] == A.STOCK_N_CLASSES and dd["resolved"]["epochs"] == REGISTERED["epochs"]
            assert dd["labelled"] == W.draw(a.draw)
        return 0
    g = load_predict(a.draw, a.device or "gpu")
    g["LOG"](f"A1.2 STOCK predict {JOB[a.draw]} via predict_final.py sha {F.sha256(A.FINAL / 'predict_final.py')[:16]}")
    g["main"]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
