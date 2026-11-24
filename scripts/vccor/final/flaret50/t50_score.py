"""vccor/final FLARE-T50 LOOK (CPU, <= 6 workers, nice 15; ONE look).

Gates, in this order, before any FLARE-T50 label byte is downloaded or read:
  1. no second look: I_flaret50/look/GT_OPENED and look/results.json must not exist;
  2. the public gate (t50_common.gate_public): the gate_osf_uploaded.json record (naming the deposit zip and this
     code freeze) and gate_osf_public.json (HTTP 200 / public = true from the OSF API, between the upload and the stop rule
     Tue 2026-11-17 12:00 KST);
  3. integrity: every file of code_freeze.json (code, frozen imports, records) unchanged; package versions == the freeze's;
     every scored arm's model files and TotalSegmentator's own weights unchanged; every DONE.json's model_sha256 equal to
     the frozen checkpoint hashes;
  4. arms: an arm counts as DONE only if its DONE.json was written before the cutoff (Fri 2026-11-13 18:00 KST); a DONE.json
     written at or after the cutoff makes the arm NOT RUN (code review SF2); F0 and FULL_s1337 DONE; every other arm DONE or
     declared NOT RUN (no interim look); the runner's RUN_FINISHED.json exists, or the cutoff has passed;
  5. every included arm map is checked against its DONE record and hashed into look/maps.json;
  6. the author's Hugging Face login exists and the pinned revision is still listed (no label byte involved).
Then look/GT_OPENED is created O_EXCL and a 'GT_OPENED sha256 ...' line is appended to final_prereg.log BEFORE any label is
downloaded. Only then are the 50 label files downloaded at the pinned revision (exact paths), each checked against its
pinned size and pointer oid (hf_manifest.json, fixed before registration; a mismatch logs GT_ABORTED and stops for good).
labels.json (per-file sha256 + the archive's sha256) is written before the deterministic tar look/labels_archive.tar is moved
into place ('LABELS sha256 ...' line). The label sanity gate follows: every included label must load, hold only exact
integers in 0..13 and have the image's shape; otherwise GT_ABORTED. Then every included scan is scored (final_common.Scorer,
conventions v1, native grid, FLARE's 13 organs; 'ts' maps through final_common.ts_lut('flare'), nnU-Net maps through the
identity LUT), per-scan records go to look/pervol/<cid>.json, the analysis (t50_analysis.analyse) to look/results.json,
and one 'RESULTS sha256 ...' line. Two qualitative panels (the fixed selection rule of t50_analysis, drawn with F0 and
FULL_s1337) are rendered to look/panels/.
After the marker, any exception, SIGTERM, SIGHUP or SIGINT logs 'GT_CRASHED ...' and exits non-zero. --continue resumes
the SAME look (only if GT_OPENED exists, results.json does not, and there is no GT_ABORTED line): maps.json must match the
marker; every label file must match labels.json (a changed or missing file is restored from the verified archive and noted;
without a verified archive the look is aborted); a crash before labels.json is repaired by re-checking the files against the
pinned pointer oids and rebuilding the deterministic archive; finished per-scan records are reused.

usage:  CUDA_VISIBLE_DEVICES="" nice -n 15 taskset -c 16-23 .venv/bin/python -I -B scripts/vccor/final/flaret50/t50_score.py
        .venv/bin/python -I -B scripts/vccor/final/flaret50/t50_score.py --check      (gates 1-6 only; writes nothing
            except one GATE line in final_prereg.log per gate file on its first use: the public gate, and the upload
            gate if no earlier step used it)
        ... t50_score.py --continue
The look rehearsal (t50_rehearse.py) runs this module under t50_common.use_root with stand-in labels.

Version of 2026-11-09; revised after the code review (M1b).
"""
import argparse
import json
import os
import shutil
import signal
import sys
import tarfile
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import t50_common as T  # noqa: E402

os.environ.update(T.HF_ENV)          # before huggingface_hub is imported (lazily, inside the look)
WORKERS = 6


def P():
    """Every path of the look, read from t50_common at call time (use_root redirects them for the rehearsal)."""
    lk = T.LOOK
    return SimpleNamespace(look=lk, marker=lk / "GT_OPENED", results=lk / "results.json", maps=lk / "maps.json",
                           archive=lk / "labels_archive.tar", archive_tmp=lk / ".labels_archive.tar.tmp",
                           labels_json=lk / "labels.json", pervol=lk / "pervol", panels=lk / "panels", run=T.RUN,
                           loglook=lk / "look.log")


class Terminated(BaseException):
    pass


def _on_term(signum, frame):
    raise Terminated(f"signal {signum}")


def _kst(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def arm_status():
    """(have, not_run): DONE before the cutoff, or NOT RUN (declared by the runner, or DONE written at / after the cutoff)."""
    have, nr = [], {}
    for a in T.RUN_ORDER:
        df = T.done_file(a)
        if df.is_file():
            fin = json.load(open(df)).get("finished")
            if fin and _kst(fin) >= T.CUTOFF:
                nr[a] = f"DONE.json written at {fin}, at or after the cutoff (Fri 2026-11-13 18:00 KST)"
            else:
                have.append(a)
        else:
            sp = P().run / f"{a}.json"
            r = json.load(open(sp)).get("not_run") if sp.is_file() else None
            if r:
                nr[a] = r
    return have, nr


def hub_ready():
    """Pre-marker check (no label byte): a Hugging Face login exists and the pinned revision is still listed."""
    if T.REHEARSAL:
        return {"rehearsal": True}
    from huggingface_hub import HfApi, get_token
    if get_token() is None:
        raise T.Refuse("no Hugging Face login on this machine: the owner logs in first")
    info = HfApi().dataset_info(T.HF_REPO, revision=T.HF_REVISION)
    if info.sha != T.HF_REVISION:
        raise T.Refuse(f"pinned revision {T.HF_REVISION} not available (got {info.sha})")
    return {"revision_ok": True}


def gates(log=print):
    T.kst_gate()
    if T.look_made():
        raise T.Refuse("a look exists (GT_OPENED or results.json): no second look")
    g = T.gate_public()
    fz = T.freeze_ok()
    T.check_versions(fz.get("versions"))
    ex = json.load(open(T.EXCLUSIONS))
    if ex.get("decision") != "K1B_PASS":
        raise T.Refuse("K1b did not pass")
    have, nr = arm_status()
    for a in ("F0", "FULL_s1337"):
        if a not in have:
            raise T.Refuse(f"required arm {a} is not DONE before the cutoff")
    missing = [a for a in T.RUN_ORDER if a not in have and a not in nr]
    if missing:
        raise T.Refuse(f"arms neither DONE nor declared NOT RUN: {missing} (no interim look)")
    if not (P().run / "RUN_FINISHED.json").is_file() and T.before_cutoff():
        raise T.Refuse("the runner has not finished and the cutoff has not passed")
    ids = T.included_ids()
    maps = {}
    for a in have:
        T.check_frozen_model(a)
        done = json.load(open(T.done_file(a)))
        if sorted(done.get("model_sha256", {}).values()) != T.frozen_checkpoint_hashes(a):
            raise T.Refuse(f"{a}: DONE.json model_sha256 differs from the frozen checkpoint hashes")
        maps[a] = {}
        for c in ids:
            p = T.pred_dir(a) / "raw" / f"{c}.nii.gz"
            h = T.sha256(p)
            if h != done["per_scan_raw_sha256"].get(c):
                raise T.Refuse(f"{a}/{c}: map differs from its DONE record")
            maps[a][c] = h
    hub = hub_ready()
    return {"gate_public": g, "freeze_sha256": T.sha256(T.FREEZE), "have": have, "not_run": nr, "ids": ids, "maps": maps,
            "hub": hub}


def score_one(args):
    cid, arms_maps = args
    import nibabel as nib
    import numpy as np
    import final_common as F
    import fin_look as L
    lbl = nib.load(str(T.label_path(cid)))
    raw = np.asarray(lbl.dataobj)
    G = np.rint(raw)
    assert np.array_equal(G, raw) and G.min() >= 0 and G.max() <= T.LABEL_MAX, cid
    G = G.astype(np.uint8)
    sp = tuple(float(s) for s in lbl.header.get_zooms()[:3])
    sc = F.Scorer(G, T.N_ORG, sp)
    rec = sc.frame_record()
    for a, (p, h, kind) in arms_maps.items():
        if T.sha256(p) != h:
            raise RuntimeError(f"{a}/{cid}: map changed after maps.json")
        seg, _ = L.read_raw(p)
        lut = F.ts_lut("flare") if kind == "ts" else F.identity_lut(T.N_ORG)
        Y = lut[seg]
        assert Y.shape == G.shape, (a, cid)
        rec["arms"][a] = sc.score(Y, surface=True)
    return cid, {"NATIVE": rec}


def download_labels(log):
    """The 50 label files at the pinned revision (exact paths), each matched to its pinned size and pointer oid.
    Rehearsal only (t50_common.REHEARSAL with LABEL_SOURCE): copied from the stand-in folder instead of the Hub."""
    m = T.load_manifest()
    T.LBL_DIR.mkdir(parents=True, exist_ok=True)
    got = {}
    for cid in T.IDS:
        e = m["labels"][cid]
        assert e["path"].startswith(T.HF_LBL_PREFIX), e["path"]
        if T.REHEARSAL and T.LABEL_SOURCE:
            shutil.copyfile(Path(T.LABEL_SOURCE) / f"{cid}.nii.gz", T.label_path(cid))
            p = T.label_path(cid)
        else:
            from huggingface_hub import hf_hub_download
            p = Path(hf_hub_download(repo_id=T.HF_REPO, repo_type="dataset", filename=e["path"], revision=T.HF_REVISION,
                                     local_dir=str(T.HF_LOCAL)))
        if p.resolve() != T.label_path(cid).resolve():
            raise RuntimeError(f"{cid}: label downloaded to {p}")
        size = p.stat().st_size
        h = T.sha256(p)
        if size != e["size"] or T.pointer_oid(h, size) != e["pointer_oid"]:
            return None, f"{cid}: label does not match its pinned size / pointer oid"
        got[cid] = {"sha256": h, "size": size}
    log("labels downloaded and matched to their pinned pointer oids")
    return got, None


def build_archive(got, dest):
    """Deterministic tar of the label files (sorted names; mtime, owner and mode fixed) -> sha256."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        os.chmod(dest, 0o644)
        dest.unlink()
    with tarfile.open(dest, "w", format=tarfile.GNU_FORMAT) as tf:
        for cid in sorted(got):
            p = T.label_path(cid)
            ti = tf.gettarinfo(str(p), arcname=f"{cid}.nii.gz")
            ti.mtime, ti.uid, ti.gid, ti.uname, ti.gname, ti.mode = 0, 0, 0, "", "", 0o644
            with open(p, "rb") as f:
                tf.addfile(ti, f)
    return T.sha256(dest)


def restore_from_archive(cid, archive):
    with tarfile.open(archive, "r") as tf:
        m = tf.getmember(f"{cid}.nii.gz")
        data = tf.extractfile(m).read()
    p = T.label_path(cid)
    if p.exists():
        os.chmod(p, 0o644)
    p.write_bytes(data)


def labels_logged(ah):
    return any(f"LABELS sha256 {ah} " in ln for ln in T.prereg_lines())


def finish_labels(got, log):
    """labels.json first (atomic), then the archive moved into place, then the LABELS line (SF1 b)."""
    q = P()
    ah = build_archive(got, q.archive_tmp)
    T.write_json(q.labels_json, {"archive_sha256": ah, "files": got, "revision": T.HF_REVISION, "written": T.now()})
    os.replace(q.archive_tmp, q.archive)
    os.chmod(q.archive, 0o444)
    if not labels_logged(ah):
        T.prereg_line("LABELS", q.archive, "flaret50 FLARE-T50 label archive (50 files, each matching its pinned pointer "
                                           "oid), packed deterministically after GT_OPENED")
    return ah


def ensure_labels(log):
    """-> (ok, message). First look: download + verify + labels.json + archive. --continue: verify every file against
    labels.json (restore a changed or missing file from the verified archive), or repair a crash before labels.json."""
    q = P()
    if not q.labels_json.is_file():
        if q.archive.is_file() or q.archive_tmp.is_file() or T.LBL_DIR.is_dir():
            log("labels.json missing (stopped before it was written): re-checking the label files against the pinned "
                "pointer oids")
        m = T.load_manifest()
        got = {}
        need_dl = False
        for cid in T.IDS:
            p = T.label_path(cid)
            e = m["labels"][cid]
            if not p.is_file():
                need_dl = True
                break
            h, s = T.sha256(p), p.stat().st_size
            if s != e["size"] or T.pointer_oid(h, s) != e["pointer_oid"]:
                need_dl = True
                break
            got[cid] = {"sha256": h, "size": s}
        if need_dl:
            got, err = download_labels(log)
            if err:
                return False, err
        finish_labels(got, log)
        return True, "labels verified and archived"
    lj = json.load(open(q.labels_json))
    ah = lj["archive_sha256"]
    if not q.archive.is_file():
        if q.archive_tmp.is_file() and T.sha256(q.archive_tmp) == ah:
            os.replace(q.archive_tmp, q.archive)
            os.chmod(q.archive, 0o444)
        else:
            files_ok = all(T.label_path(c).is_file() and T.sha256(T.label_path(c)) == v["sha256"] for c, v in lj["files"].items())
            if not files_ok or build_archive(lj["files"], q.archive) != ah:
                return False, "label archive missing and the label files do not rebuild it"
            os.chmod(q.archive, 0o444)
    if T.sha256(q.archive) != ah:
        return False, "label archive differs from labels.json"
    restored = []
    for c, v in lj["files"].items():
        p = T.label_path(c)
        if not p.is_file() or T.sha256(p) != v["sha256"]:
            restore_from_archive(c, q.archive)
            if T.sha256(p) != v["sha256"]:
                return False, f"{c}: label cannot be restored from the archive"
            restored.append(c)
    if restored:
        T.prereg_note("GT_NOTE", f"label files {restored} differed from labels.json and were restored from the hashed "
                                 f"archive before scoring")
    if not labels_logged(ah):
        T.prereg_line("LABELS", q.archive, "flaret50 FLARE-T50 label archive (logged on continue)")
    return True, f"labels verified ({len(restored)} restored)"


def sanity_gate(ids):
    """-> (bad, diagnostics). bad: unreadable label, a value outside the exact integers 0..13, or a shape other than the
    image's (registered: GT_ABORTED). diagnostics (descriptive only): affine max |difference| to the image, whether id 1
    (liver) is the largest labelled organ."""
    import nibabel as nib
    import numpy as np
    bad, diag = {}, {}
    for c in ids:
        try:
            lbl = nib.load(str(T.label_path(c)))
            im = nib.load(str(T.image_path(c)))
            a = np.asarray(lbl.dataobj)
            u = np.unique(a)
            if not np.array_equal(u, np.rint(u)) or u.min() < 0 or u.max() > T.LABEL_MAX:
                bad[c] = f"label values {u[:20].tolist()}"
                continue
            if tuple(lbl.shape[:3]) != tuple(im.shape[:3]):
                bad[c] = f"shape {lbl.shape} != image {im.shape}"
                continue
            cnt = np.bincount(np.rint(a).astype(np.int64).ravel(), minlength=T.LABEL_MAX + 1)[1:]
            diag[c] = {"affine_max_abs_diff": float(np.abs(lbl.affine - im.affine).max()),
                       "liver_largest": bool(cnt.argmax() == 0) if cnt.sum() else None}
        except Exception as e:                       # noqa: BLE001  (an unreadable label is a registered abort)
            bad[c] = f"unreadable: {type(e).__name__}: {str(e)[:120]}"
    return bad, diag


def render_panels(sel, maps, log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nibabel as nib
    import numpy as np
    import final_common as F
    out = {}
    pdir = P().panels
    pdir.mkdir(parents=True, exist_ok=True)
    lut = F.ts_lut("flare")
    for kind, cid in (("median_gain", sel.get("median_gain_scan")), ("largest_loss", sel.get("largest_loss_scan"))):
        if not cid:
            continue
        img = np.asarray(nib.load(str(T.image_path(cid))).dataobj, dtype=np.float32)
        G = np.rint(np.asarray(nib.load(str(T.label_path(cid))).dataobj)).astype(np.uint8)
        A = lut[np.asarray(nib.load(maps["F0"][cid]).dataobj).astype(np.uint8)]
        B = lut[np.asarray(nib.load(maps["FULL_s1337"][cid]).dataobj).astype(np.uint8)]
        z = int(np.argmax(((A != B) & ((A > 0) | (B > 0))).sum((0, 1))))
        fig, ax = plt.subplots(1, 3, figsize=(12, 4.4))
        ct = np.clip((img[:, :, z].T - (-125.0)) / 400.0, 0, 1)
        for i, (t, Y) in enumerate((("reference", G), ("F0 (zero-shot)", A), ("FULL_s1337 (K = 4)", B))):
            ax[i].imshow(ct, cmap="gray", origin="lower")
            m = np.ma.masked_where(Y[:, :, z].T == 0, Y[:, :, z].T)
            ax[i].imshow(m, cmap="tab20", vmin=0, vmax=20, alpha=0.45, origin="lower", interpolation="nearest")
            ax[i].set_title(t, fontsize=10)
            ax[i].axis("off")
        fig.suptitle(f"{cid}, axial slice {z} ({kind.replace('_', ' ')} scan by the three-draw average; drawn with "
                     f"FULL_s1337)", fontsize=10)
        p = pdir / f"{kind}_{cid}.png"
        fig.savefig(p, dpi=110, bbox_inches="tight")
        plt.close(fig)
        out[kind] = {"cid": cid, "slice": z, "png": T.rel(p), "sha256": T.sha256(p)}
    log(f"panels rendered: {[v['png'] for v in out.values()]}")
    return out


def look(cont=False):
    q = P()
    log = (lambda m: T.log_event(m, q.loglook))
    if cont:
        if not q.marker.exists() or q.results.exists():
            raise T.Refuse("--continue needs GT_OPENED and no results.json")
        if any(f"GT_ABORTED {T.TAG}" in ln for ln in T.prereg_lines()):
            raise T.Refuse("the look was aborted (GT_ABORTED): it cannot continue")
        ctx = json.load(open(q.marker))
        if T.sha256(q.maps) != ctx["maps_sha256"]:
            raise T.Refuse("maps.json changed after the marker")
        ctx["maps"] = json.load(open(q.maps))["maps"]
        T.freeze_ok()
        T.prereg_note("GT_CONTINUED", f"look continued at {T.now()}")
    else:
        ctx = gates(log)
        q.look.mkdir(parents=True, exist_ok=True)
        T.write_json(q.maps, {"maps": ctx["maps"], "ids": ctx["ids"], "have": ctx["have"], "not_run": ctx["not_run"]})
        ctx["maps_sha256"] = T.sha256(q.maps)
        fd = os.open(str(q.marker), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        with os.fdopen(fd, "w") as f:
            json.dump({"opened": T.now(), "maps_sha256": ctx["maps_sha256"], "freeze_sha256": ctx["freeze_sha256"],
                       "gate_public": ctx["gate_public"], "ids": ctx["ids"], "have": ctx["have"],
                       "not_run": ctx["not_run"]}, f, indent=1)
        T.prereg_line("GT_OPENED", q.marker, f"flaret50 look: marker written before any FLARE-T50 label is downloaded; "
                                             f"maps.json sha256 {ctx['maps_sha256']}")
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, _on_term)
    try:
        ok, msg = ensure_labels(log)
        if not ok:
            T.prereg_note("GT_ABORTED", msg)
            log(f"GT_ABORTED: {msg}")
            return 5
        ids = ctx["ids"]
        bad, diag = sanity_gate(ids)
        if bad:
            T.prereg_note("GT_ABORTED", f"label sanity gate: {bad}")
            log(f"GT_ABORTED label sanity: {bad}")
            return 5
        have = ctx["have"]
        arms_maps = {a: {c: (str(T.pred_dir(a) / "raw" / f"{c}.nii.gz"), ctx["maps"][a][c], T.LUT_KIND[a]) for c in ids}
                     for a in have}
        R, todo = {}, []
        for c in ids:
            pv = q.pervol / f"{c}.json"
            if pv.is_file():
                R[c] = json.load(open(pv))
            else:
                todo.append((c, {a: arms_maps[a][c] for a in have}))
        from multiprocessing import Pool
        t0 = time.time()
        with Pool(WORKERS) as pool:
            for c, rec in pool.imap_unordered(score_one, todo, chunksize=1):
                T.write_json(q.pervol / f"{c}.json", rec)
                R[c] = rec
                log(f"scored {c} ({len(R)}/{len(ids)}, {time.time() - t0:.0f} s)")
        import t50_analysis as AN
        res = AN.analyse(R, ids, have, ctx["not_run"])
        res["label_diagnostics_descriptive"] = diag
        try:
            res["panels_rendered"] = render_panels(res.get("panels", {}),
                                                   {a: {c: arms_maps[a][c][0] for c in ids} for a in ("F0", "FULL_s1337")},
                                                   log)
        except Exception as e:                       # noqa: BLE001  (a figure error never blocks the registered results)
            res["panels_rendered"] = {"error": f"{type(e).__name__}: {e}"}
        res["look"] = {"marker_sha256": T.sha256(q.marker), "maps_sha256": T.sha256(q.maps),
                       "labels_archive_sha256": T.sha256(q.archive), "finished": T.now(),
                       "code_sha256": T.code_hashes(["t50_score.py", "t50_analysis.py", "t50_common.py"])}
        T.write_json(q.results, res, mode=0o444)
        T.prereg_line("RESULTS", q.results, f"flaret50: {AN.summary(res)}")
        return 0
    except (Exception, Terminated, KeyboardInterrupt) as e:
        T.prereg_note("GT_CRASHED", f"{type(e).__name__}: {str(e)[:300]}")
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--continue", dest="cont", action="store_true")
    a = ap.parse_args()
    try:
        if a.check:
            c = gates()
            print(f"gates OK: {len(c['ids'])} scans, arms {c['have']}, not run {c['not_run']}")
            return 0
        return look(cont=a.cont)
    except T.Refuse as e:
        print(f"[REFUSE] {e.msg}", file=sys.stderr)
        return e.code


if __name__ == "__main__":
    sys.exit(main())
