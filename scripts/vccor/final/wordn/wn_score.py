"""vccor/final WORD-N LOOK (CPU, <= 6 workers, nice 15; ONE look).

Gates, in this order, before any WORD validation label byte is extracted or read:
  1. no second look: I_wordn/look/GT_OPENED and look/results.json must not exist;
  2. the public gate (wn_common.gate_public): the gate_osf_uploaded.json record (naming the deposit zip and this code
     freeze) and gate_osf_public.json (HTTP 200 / public = true from the OSF API, between the upload and the stop rule
     Sat 2026-11-21 12:00 KST);
  3. integrity: every file of code_freeze.json unchanged; package versions == the freeze's; the intake passed; the WORD
     archive's size and sha256 equal its download record and the pinned member listing; every scored arm's model files
     (TotalSegmentator's weights / the VISTA3D bundle / the LABELMAP rules) unchanged; every DONE.json's model_sha256 equal
     to the frozen weight hashes; every CPU arm's DONE names the current DONE records of its inputs;
  4. arms (wn_score.arm_status): a runner NOT RUN declaration is binding (a later DONE for that arm is ignored and logged);
     a tier-2 arm counts only if vsmoke/smoke.json says VSMOKE_PASS; an arm counts as DONE only if its DONE.json (mode
     'real') was written before the cutoff (Thu 2026-11-19 18:00 KST); a DONE.json written at or after the cutoff, or no
     DONE once the cutoff has passed, makes the arm NOT RUN; F0 and F1b_d1 DONE; before the cutoff every other arm DONE or
     declared NOT RUN (no interim look) and the runner's RUN_FINISHED.json must exist; a tier-2 arm that fails its
     integrity check (VISTA3D bundle, frozen weights) is NOT RUN (tier 2 only; a tier-1 integrity failure refuses the look);
     a CPU arm counts only if its inputs count;
  5. every included arm map is checked against its DONE record and hashed into look/maps.json (a tier-2 mismatch makes
     that arm NOT RUN; a tier-1 mismatch refuses the look).
Then look/GT_OPENED is created O_EXCL and a 'GT_OPENED sha256 ...' line is appended to final_prereg.log BEFORE any label
member is extracted. Only then are the 20 whitelisted labelsVal members extracted (7z, AES authentication checked by 7z)
into look/labels/, each checked against its pinned size (archive_members.json; a mismatch logs GT_ABORTED and stops for
good); labels.json (per-file sha256 + the archive's sha256) is written before the deterministic tar look/labels_archive.tar
is moved into place ('LABELS sha256 ...' line). The label sanity gate follows: every included label must load, hold only
exact integers 0..16 and have the image's shape and affine (max |diff| <= 1e-3); otherwise GT_ABORTED. Then every
included scan is scored (final_common.Scorer, conventions v1, native grid, WORD's 15 analysis organs via
word_common.analysis_lut; arm maps through fin_look.lut('word', ts / id / vista)); per-scan records go to
look/pervol/<cid>.json, the analysis (wn_analysis.analyse, the plan's overshoot_mode) to look/results.json, and one
'RESULTS sha256 ...' line. Two qualitative panels (wn_analysis.panels' fixed rules, drawn with F0 and F1b_d1 and the
reference) are rendered to look/panels/.
After the marker, any exception, SIGTERM, SIGHUP or SIGINT logs 'GT_CRASHED ...' and exits non-zero. --continue resumes
the SAME look (only if GT_OPENED exists, results.json does not, and there is no GT_ABORTED line): maps.json must match the
marker; every label file must match labels.json (a changed or missing file is restored from the verified tar and noted);
a crash before labels.json is repaired by re-extracting the members and re-checking their pinned sizes; finished per-scan
records are reused.

usage:  CUDA_VISIBLE_DEVICES="" nice -n 15 taskset -c 16-23 .venv/bin/python -I -B scripts/vccor/final/wordn/wn_score.py
        .venv/bin/python -I -B scripts/vccor/final/wordn/wn_score.py --check     (gates 1-5 without writing maps.json;
            writes nothing except one GATE line per gate file on its first use)
        ... wn_score.py --continue
The look rehearsal (wn_rehearse.py) runs this module under wn_common.use_root with stand-in labels in a stand-in archive.

Version of 2026-11-13.
"""
import argparse
import json
import os
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
import wn_common as T  # noqa: E402

W = T.W
WORKERS = 6


def P():
    lk = T.LOOK
    return SimpleNamespace(look=lk, marker=lk / "GT_OPENED", results=lk / "results.json", maps=lk / "maps.json",
                           archive=lk / "labels_archive.tar", archive_tmp=lk / ".labels_archive.tar.tmp",
                           labels_json=lk / "labels.json", pervol=lk / "pervol", panels=lk / "panels", run=T.RUN,
                           loglook=lk / "look.log")


class Terminated(BaseException):
    pass


def _on_term(signum, frame):
    raise Terminated(f"signal {signum}")


def _worker_init():
    """Scoring workers: default SIGTERM / SIGHUP (the pool's terminate() must end them), SIGINT ignored (== word_a1_score)."""
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGHUP, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _kst(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def arm_status():
    """(have, not_run, ignored): the registered arm accounting at the look (see gate 4). ignored = {arm: why a DONE.json
    was not counted} (a binding NOT RUN declaration, no smoke pass, a non-real DONE)."""
    have, nr, ign = [], {}, {}
    smoke = T.smoke_passed()
    past = not T.before_cutoff()
    for a in T.RUN_ORDER:
        df = T.done_file(a)
        decl = T.declared_not_run(a)
        if decl:
            nr[a] = decl
            if df.is_file():
                ign[a] = "DONE.json exists but the runner had declared the arm NOT RUN (binding)"
            continue
        if a in T.TIER2 and not smoke:
            nr[a] = "the VISTA3D smoke test did not pass (tier 2 NOT RUN)"
            if df.is_file():
                ign[a] = "DONE.json exists but the VISTA3D smoke test did not pass"
            continue
        if df.is_file():
            d = json.load(open(df))
            fin = d.get("finished")
            if d.get("mode") != "real":
                nr[a] = f"DONE.json mode {d.get('mode')!r}, not 'real'"
                ign[a] = nr[a]
            elif fin and _kst(fin) >= T.CUTOFF:
                nr[a] = f"DONE.json written at {fin}, at or after the cutoff (Thu 2026-11-19 18:00 KST)"
            else:
                have.append(a)
        elif past:
            nr[a] = "not DONE by the cutoff (Thu 2026-11-19 18:00 KST)"
    for a in list(have):                       # N-D1: a CPU arm counts only if every input counts
        miss = [i for i in T.arms()[a].get("inputs", []) if i not in have] if T.arms()[a]["kind"] == "cpu" else []
        if miss:
            have.remove(a)
            nr[a] = f"its inputs {miss} do not count at the look"
            ign[a] = nr[a]
    return have, nr, ign


def cpu_inputs_ok(a):
    arm = T.arms()[a]
    done = json.load(open(T.done_file(a)))
    for i in arm["inputs"]:
        key = f"{i}_done_sha256"
        if done["inputs"].get(key) != T.sha256(T.done_file(i)):
            raise T.Refuse(f"{a}: its DONE names another {i} DONE record than the current one")


def gates(log=print, write=True):
    T.kst_gate()
    if T.look_made():
        raise T.Refuse("a look exists (GT_OPENED or results.json): no second look")
    g = T.gate_public()
    fz = T.freeze_ok()
    T.check_versions(fz.get("versions"))
    mode = T.overshoot_mode()
    ex = json.load(open(T.EXCLUSIONS)) if T.EXCLUSIONS.is_file() else {}
    if ex.get("decision") != "INTAKE_PASS":
        raise T.Refuse("the intake did not pass")
    T.archive_ok()
    pinned = json.load(open(T.MEMBERS))
    if pinned["archive_sha256"] != T.sha256_cached(T.ARCHIVE):
        raise T.Refuse("the archive differs from the pinned member listing")
    have, nr, ign = arm_status()
    for a in T.REQUIRED:
        if a not in have:
            raise T.Refuse(f"required arm {a} is not DONE before the cutoff")
    missing = [a for a in T.RUN_ORDER if a not in have and a not in nr]
    if missing:
        raise T.Refuse(f"arms neither DONE nor declared NOT RUN: {missing} (no interim look)")
    if not (P().run / "RUN_FINISHED.json").is_file() and T.before_cutoff():
        raise T.Refuse("the runner has not finished and the cutoff has not passed")
    ids = T.included_ids()
    maps = {}
    for a in list(have):
        done = json.load(open(T.done_file(a)))
        kind = T.arms()[a]["kind"]
        try:
            T.check_frozen_model(a)
            if kind == "cpu":
                cpu_inputs_ok(a)
            elif a != "V0" and sorted(done.get("model_sha256", {}).values()) != T.frozen_checkpoint_hashes(a):
                raise T.Refuse(f"{a}: DONE.json model_sha256 differs from the frozen weight hashes")
        except T.Refuse as e:
            if a not in T.TIER2:
                raise
            have.remove(a)
            nr[a] = f"tier-2 integrity check failed at the look: {e.msg}"
            continue
        m = {}
        bad = None
        for c in ids:
            p = T.pred_dir(a) / "raw" / f"word_{c}.nii.gz"
            h = T.sha256(p) if p.is_file() else None
            if h is None or h != done["per_scan_raw_sha256"].get(c):
                bad = f"{a}/{c}: map differs from its DONE record"
                break
            m[c] = h
        if bad:
            if a not in T.TIER2:
                raise T.Refuse(bad)
            have.remove(a)
            nr[a] = f"tier-2 integrity check failed at the look: {bad}"
            continue
        maps[a] = m
    return {"gate_public": g, "freeze_sha256": T.sha256(T.FREEZE), "have": have, "not_run": nr, "ids": ids, "maps": maps,
            "mode": mode, "members_sha256": T.sha256(T.MEMBERS), "ignored_done": ign}


def score_one(args):
    cid, arms_maps = args
    import nibabel as nib
    import numpy as np
    import final_common as F
    import fin_look as L
    lbl = nib.load(str(T.label_path(cid)))
    raw = np.asarray(lbl.dataobj)
    G = np.rint(raw)
    assert np.array_equal(G, raw) and G.min() >= 0 and G.max() <= T.LABEL_MAX_RAW, cid
    G = W.analysis_lut()[G.astype(np.uint8)]
    sp = tuple(float(s) for s in lbl.header.get_zooms()[:3])
    sc = F.Scorer(G, T.N_ORG, sp)
    rec = sc.frame_record()
    for a, (p, h, kind) in arms_maps.items():
        if T.sha256(p) != h:
            raise RuntimeError(f"{a}/{cid}: map changed after maps.json")
        seg, _ = L.read_raw(p)
        Y = L.lut("word", kind)[seg]
        assert Y.shape == G.shape, (a, cid)
        rec["arms"][a] = sc.score(Y, surface=True)
    return cid, {"NATIVE": rec}


def extract_labels(log):
    """The 20 whitelisted labelsVal members -> look/labels/, each matched to its pinned size."""
    pinned = json.load(open(T.MEMBERS))
    T.LBL_DIR.mkdir(parents=True, exist_ok=True)
    got = {}
    for cid in T.VAL_IDS:
        m = T.member("labelsVal", cid)
        b = T.extract_member_bytes(m)
        if len(b) != pinned["sizes"][m]:
            return None, f"{m}: {len(b)} bytes, pinned {pinned['sizes'][m]}"
        p = T.label_path(cid)
        if p.exists():
            os.chmod(p, 0o644)
        tmp = p.with_name(f".{p.name}.tmp{os.getpid()}")
        tmp.write_bytes(b)
        os.replace(tmp, p)
        got[cid] = {"member": m, "sha256": T.sha256(p), "size": len(b)}
        T.log_event(f"[gt] WORD labelsVal {cid} extracted after GT_OPENED ({len(b)} bytes)", P().loglook)
    log("labels extracted and matched to their pinned sizes")
    return got, None


def build_archive(got, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        os.chmod(dest, 0o644)
        dest.unlink()
    with tarfile.open(dest, "w", format=tarfile.GNU_FORMAT) as tf:
        for cid in sorted(got):
            p = T.label_path(cid)
            ti = tf.gettarinfo(str(p), arcname=f"word_{cid}.nii.gz")
            ti.mtime, ti.uid, ti.gid, ti.uname, ti.gname, ti.mode = 0, 0, 0, "", "", 0o644
            with open(p, "rb") as f:
                tf.addfile(ti, f)
    return T.sha256(dest)


def restore_from_archive(cid, archive):
    with tarfile.open(archive, "r") as tf:
        data = tf.extractfile(tf.getmember(f"word_{cid}.nii.gz")).read()
    p = T.label_path(cid)
    if p.exists():
        os.chmod(p, 0o644)
    p.write_bytes(data)


def labels_logged(ah):
    return any(f"LABELS sha256 {ah} " in ln for ln in T.prereg_lines())


def finish_labels(got):
    q = P()
    ah = build_archive(got, q.archive_tmp)
    T.write_json(q.labels_json, {"archive_sha256": ah, "files": got, "word_archive_sha256": T.sha256_cached(T.ARCHIVE),
                                 "written": T.now()})
    os.replace(q.archive_tmp, q.archive)
    os.chmod(q.archive, 0o444)
    if not labels_logged(ah):
        T.prereg_line("LABELS", q.archive, "wordn WORD validation label archive (20 labelsVal members, each matching its "
                                           "pinned size), packed deterministically after GT_OPENED")
    return ah


def ensure_labels(log):
    q = P()
    if not q.labels_json.is_file():
        if q.archive.is_file() or q.archive_tmp.is_file() or T.LBL_DIR.is_dir():
            log("labels.json missing (stopped before it was written): re-extracting the members")
        got, err = extract_labels(log)
        if err:
            return False, err
        finish_labels(got)
        return True, "labels extracted, verified and archived"
    lj = json.load(open(q.labels_json))
    ah = lj["archive_sha256"]
    if not q.archive.is_file():
        if q.archive_tmp.is_file() and T.sha256(q.archive_tmp) == ah:
            os.replace(q.archive_tmp, q.archive)
            os.chmod(q.archive, 0o444)
        else:
            ok = all(T.label_path(c).is_file() and T.sha256(T.label_path(c)) == v["sha256"] for c, v in lj["files"].items())
            if not ok or build_archive(lj["files"], q.archive) != ah:
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
        T.prereg_line("LABELS", q.archive, "wordn WORD validation label archive (logged on continue)")
    return True, f"labels verified ({len(restored)} restored)"


def sanity_gate(ids):
    import nibabel as nib
    import numpy as np
    bad, diag = {}, {}
    for c in ids:
        try:
            lbl = nib.load(str(T.label_path(c)))
            im = nib.load(str(T.image_path(c)))
            a = np.asarray(lbl.dataobj)
            u = np.unique(a)
            if not np.array_equal(u, np.rint(u)) or u.min() < 0 or u.max() > T.LABEL_MAX_RAW:
                bad[c] = f"label values {u[:20].tolist()}"
                continue
            if tuple(lbl.shape[:3]) != tuple(im.shape[:3]):
                bad[c] = f"shape {lbl.shape} != image {im.shape}"
                continue
            d = float(np.abs(lbl.affine - im.affine).max())
            if d > 1e-3:
                bad[c] = f"affine differs from the image by {d}"
                continue
            cnt = np.bincount(np.rint(a).astype(np.int64).ravel(), minlength=T.LABEL_MAX_RAW + 1)[1:]
            diag[c] = {"affine_max_abs_diff": d, "raw_ids_present": [int(i + 1) for i in np.flatnonzero(cnt)]}
        except Exception as e:                       # noqa: BLE001  (an unreadable label is a registered abort)
            bad[c] = f"unreadable: {type(e).__name__}: {str(e)[:120]}"
    return bad, diag


def render_panels(sel, maps, log):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nibabel as nib
    import numpy as np
    import fin_look as L
    out = {}
    pdir = P().panels
    pdir.mkdir(parents=True, exist_ok=True)
    lut = L.lut("word", "ts")
    for kind, cid in (("median_gain", sel.get("median_gain_scan")),
                      ("largest_duodenum_overshoot", sel.get("largest_duodenum_overshoot_scan"))):
        if not cid:
            continue
        img = np.asarray(nib.load(str(T.image_path(cid))).dataobj, dtype=np.float32)
        G = W.analysis_lut()[np.rint(np.asarray(nib.load(str(T.label_path(cid))).dataobj)).astype(np.uint8)]
        A = lut[np.asarray(nib.load(maps["F0"][cid]).dataobj).astype(np.uint8)]
        B = lut[np.asarray(nib.load(maps["F1b_d1"][cid]).dataobj).astype(np.uint8)]
        if kind == "median_gain":
            z = int(np.argmax(((A != B) & ((A > 0) | (B > 0))).sum((0, 1))))
        else:
            z = int(np.argmax(((B == T.DUODENUM) & (G != T.DUODENUM)).sum((0, 1))))
        fig, ax = plt.subplots(1, 3, figsize=(12, 4.4))
        ct = np.clip((img[:, :, z].T - (-125.0)) / 400.0, 0, 1)
        for i, (t, Y) in enumerate((("reference", G), ("F0 (zero-shot)", A), ("F1b_d1 (K = 4)", B))):
            ax[i].imshow(ct, cmap="gray", origin="lower")
            m = np.ma.masked_where(Y[:, :, z].T == 0, Y[:, :, z].T)
            ax[i].imshow(m, cmap="tab20", vmin=0, vmax=20, alpha=0.45, origin="lower", interpolation="nearest")
            ax[i].set_title(t, fontsize=10)
            ax[i].axis("off")
        fig.suptitle(f"WORD validation {cid}, axial slice {z} ({kind.replace('_', ' ')} scan by the registered rule)",
                     fontsize=10)
        p = pdir / f"{kind}_{cid}.png"
        fig.savefig(p, dpi=110, bbox_inches="tight")
        plt.close(fig)
        out[kind] = {"cid": cid, "slice": z, "png": T.rel(p), "sha256": T.sha256(p)}
    log(f"panels rendered: {[v['png'] for v in out.values()]}")
    return out


def extras(have):
    ex = {"merge": {}, "labelmap": {}}
    for d in T.DRAWS:
        if f"MERGE_N_{d}" in have:
            ex["merge"][d] = json.load(open(T.done_file(f"MERGE_N_{d}")))
        if f"LABELMAP_{d}" in have:
            ex["labelmap"][d] = json.load(open(T.done_file(f"LABELMAP_{d}")))
    return ex


def look(cont=False):
    q = P()
    log = (lambda m: T.log_event(m, q.loglook))
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, _on_term)
    if cont and q.results.exists() and q.marker.exists():
        h = T.sha256(q.results)
        if not any(f"RESULTS sha256 {h} " in ln for ln in T.prereg_lines()):
            import wn_analysis as AN
            T.prereg_line("RESULTS", q.results, f"wordn: {AN.summary(json.load(open(q.results)))} (logged on continue)")
            return 0
    if cont:
        if not q.marker.exists() or q.results.exists():
            raise T.Refuse("--continue needs GT_OPENED and no results.json")
        if any(f"GT_ABORTED {T.TAG}" in ln for ln in T.prereg_lines()):
            raise T.Refuse("the look was aborted (GT_ABORTED): it cannot continue")
        ctx = json.load(open(q.marker))
        if T.sha256(q.maps) != ctx["maps_sha256"]:
            raise T.Refuse("maps.json changed after the marker")
        mj = json.load(open(q.maps))
        ctx["maps"] = mj["maps"]
        T.freeze_ok()
        mh = T.sha256(q.marker)
        if not any(f"GT_OPENED sha256 {mh} " in ln for ln in T.prereg_lines()):
            T.prereg_line("GT_OPENED", q.marker, "wordn look: marker line written on continue (the first run stopped "
                                                 "between the marker and its line; no label had been extracted)")
        T.prereg_note("GT_CONTINUED", f"look continued at {T.now()}")
    else:
        ctx = gates(log)
        q.look.mkdir(parents=True, exist_ok=True)
        T.write_json(q.maps, {"maps": ctx["maps"], "ids": ctx["ids"], "have": ctx["have"], "not_run": ctx["not_run"],
                              "mode": ctx["mode"], "ignored_done": ctx["ignored_done"]})
        for a, why in ctx["ignored_done"].items():
            log(f"ignored DONE of {a}: {why}")
        ctx["maps_sha256"] = T.sha256(q.maps)
        fd = os.open(str(q.marker), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        with os.fdopen(fd, "w") as f:
            json.dump({"opened": T.now(), "maps_sha256": ctx["maps_sha256"], "freeze_sha256": ctx["freeze_sha256"],
                       "gate_public": ctx["gate_public"], "ids": ctx["ids"], "have": ctx["have"],
                       "not_run": ctx["not_run"], "mode": ctx["mode"], "members_sha256": ctx["members_sha256"],
                       "ignored_done": ctx["ignored_done"],
                       "declared": "FIRST and only look at the WORD validation labels (registered on OSF)"}, f, indent=1)
        T.prereg_line("GT_OPENED", q.marker, f"wordn look: marker written before any WORD validation label is extracted; "
                                             f"maps.json sha256 {ctx['maps_sha256']}; ignored DONE records "
                                             f"{sorted(ctx['ignored_done']) or 'none'}")
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
        arms_maps = {a: {c: (str(T.pred_dir(a) / "raw" / f"word_{c}.nii.gz"), ctx["maps"][a][c], T.lut_kind(a))
                         for c in ids} for a in have}
        R, todo = {}, []
        for c in ids:
            pv = q.pervol / f"{c}.json"
            if pv.is_file():
                R[c] = json.load(open(pv))
            else:
                todo.append((c, {a: arms_maps[a][c] for a in have}))
        from multiprocessing import Pool
        t0 = time.time()
        with Pool(WORKERS, initializer=_worker_init) as pool:
            for c, rec in pool.imap_unordered(score_one, todo, chunksize=1):
                T.write_json(q.pervol / f"{c}.json", rec)
                R[c] = rec
                log(f"scored {c} ({len(R)}/{len(ids)}, {time.time() - t0:.0f} s)")
        import wn_analysis as AN
        res = AN.analyse(R, ids, have, ctx["not_run"], mode=ctx.get("mode") or T.overshoot_mode(), extras=extras(have))
        res["label_diagnostics_descriptive"] = diag
        try:
            res["panels_rendered"] = render_panels(res.get("panels", {}),
                                                   {a: {c: arms_maps[a][c][0] for c in ids} for a in ("F0", "F1b_d1")},
                                                   log)
        except Exception as e:                       # noqa: BLE001  (a figure error never blocks the registered results)
            res["panels_rendered"] = {"error": f"{type(e).__name__}: {e}"}
        res["look"] = {"marker_sha256": T.sha256(q.marker), "maps_sha256": T.sha256(q.maps),
                       "labels_archive_sha256": T.sha256(q.archive), "finished": T.now(),
                       "code_sha256": T.code_hashes(["wn_score.py", "wn_analysis.py", "wn_common.py"])}
        summ = AN.summary(res)
        T.write_json(q.results, res, mode=0o444)
        T.prereg_line("RESULTS", q.results, f"wordn: {summ}")
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
            print(f"gates OK: {len(c['ids'])} scans, arms {c['have']}, not run {c['not_run']}, mode {c['mode']}")
            return 0
        return look(cont=a.cont)
    except T.Refuse as e:
        print(f"[REFUSE] {e.msg}", file=sys.stderr)
        return e.code


if __name__ == "__main__":
    sys.exit(main())
