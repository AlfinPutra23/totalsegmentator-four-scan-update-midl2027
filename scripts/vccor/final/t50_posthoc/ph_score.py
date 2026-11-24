"""vccor/final FLARE-T50 POST HOC SCORE (CPU, <= 6 workers, nice 15). EXPLORATORY, NOT REGISTERED: a post hoc SECOND
reading of the FLARE-T50 labels (first read by the registered look, GT_OPENED 2026-11-13 09:01:23), approved
2026-11-13.

Order:
  1. gates without any label byte: the registered look is finished (look/results.json); every post hoc arm map in
     J_t50_posthoc/pred/<arm> equals its DONE.json; the F0 / FULL_s1337 maps equal look/maps.json (for the re-score check);
  2. ONE 'NOTE flaret50 ...' line in final_prereg.log through t50_common.prereg_note (post hoc second reading; the three
     models and their sha256; verified copies only; exploratory) - BEFORE any label byte is read (skipped on a re-run if
     the same NOTE already exists);
  3. look/labels_archive.tar is hashed against look/labels.json (archive_sha256) and the registration log's LABELS line;
     each member is read from the archive, its sha256 checked against look/labels.json, and only then written to
     J_t50_posthoc/tmp_labels_verified/ (re-hashed after writing); the live label folder of the look is never touched;
  4. scoring == t50_score.score_one: final_common.Scorer, conventions v1, native grid, FLARE's 13 organs, surface metrics;
     the post hoc arms (TS 'total' ids) through final_common.ts_lut('flare') == the 'ts' LUT of the look; F0 and FULL_s1337
     are re-scored from their look maps ONLY as an equality check against the stored look/pervol records;
  5. the comparisons use the STORED look/pervol records of F0, FULL_s1337 (and STOCK, context) - not the re-scores;
  6. the temp label folder is deleted.
Writes J_t50_posthoc/pervol/<cid>.json, J_t50_posthoc/results_posthoc.json. Nothing is written under I_flaret50/.

usage:  CUDA_VISIBLE_DEVICES="" nice -n 15 .venv/bin/python -I -B scripts/vccor/final/t50_posthoc/ph_score.py
Version of 2026-11-13.
"""
import hashlib
import json
import os
import shutil
import sys
import tarfile
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import ph_common as PH  # noqa: E402

T = PH.T
WORKERS = 6
LOOK = T.LOOK
PERVOL = PH.OUT / "pervol"
NEW = list(PH.ORDER)
CHECK = ["F0", "FULL_s1337"]               # re-scored only to check the scoring path against the stored look records
NOTE_KEY = "post hoc second reading (J_t50_posthoc)"


def note_text():
    parts = []
    for a in NEW:
        s = PH.ARMS[a]["shas"]
        parts.append(f"{a} ({PH.ARMS[a]['label']}; part291 checkpoint sha256 {s[291]}, part293 checkpoint sha256 {s[293]})")
    return (f"{NOTE_KEY}: owner-approved 2026-11-13; a POST HOC SECOND READING of the 50 FLARE-T50 labels, first read by the "
            f"registered look (GT_OPENED 2026-11-13 09:01:23); EXPLORATORY, NOT REGISTERED; scores three stored K = 4 s1337 "
            f"models that the look did not include: " + "; ".join(parts) + ". Labels are used ONLY through the verified copy: "
            f"look/labels_archive.tar checked against look/labels.json archive_sha256, each label file checked against "
            f"look/labels.json before use, extracted to J_t50_posthoc/tmp_labels_verified (deleted at the end); scoring == "
            f"t50_score (final_common.Scorer, conventions v1, native grid, FLARE 13 organs, ts_lut flare); compared with the "
            f"stored look/pervol records of F0 and FULL_s1337. Every reading is exploratory (post hoc); no registered claim "
            f"changes.")


def gates():
    if not (LOOK / "results.json").is_file() or not (LOOK / "GT_OPENED").is_file():
        raise T.Refuse("the registered look has not finished: no post hoc reading")
    ids = T.included_ids()
    assert len(ids) == 50
    maps = {}
    for a in NEW:
        d = json.load(open(PH.PRED / a / "DONE.json"))
        red, shas, _ = PH.resolve(a)
        if d["model_sha256"] != {str(k): v for k, v in shas.items()} or d["ids"] != ids:
            raise T.Refuse(f"{a}: DONE.json differs from the model / ids")
        maps[a] = {}
        for c in ids:
            p = PH.PRED / a / "raw" / f"{c}.nii.gz"
            if T.sha256(p) != d["per_scan_raw_sha256"][c]:
                raise T.Refuse(f"{a}/{c}: map differs from DONE.json")
            maps[a][c] = (str(p), d["per_scan_raw_sha256"][c])
    lm = json.load(open(LOOK / "maps.json"))["maps"]
    for a in CHECK:
        maps[a] = {}
        for c in ids:
            p = T.pred_dir(a) / "raw" / f"{c}.nii.gz"
            if T.sha256(p) != lm[a][c]:
                raise T.Refuse(f"{a}/{c}: look map differs from look/maps.json")
            maps[a][c] = (str(p), lm[a][c])
    return ids, maps


def note_once():
    if any(NOTE_KEY in ln for ln in T.prereg_lines()):
        PH.log("NOTE already in final_prereg.log (re-run); not repeated")
        return False
    T.prereg_note("NOTE", note_text())
    return True


def verified_copy(log):
    """look/labels_archive.tar -> TMP_LABELS, every file checked against look/labels.json. -> {cid: sha256}."""
    lj = json.load(open(LOOK / "labels.json"))
    ah = T.sha256(LOOK / "labels_archive.tar")
    if ah != lj["archive_sha256"]:
        raise T.Refuse("labels_archive.tar differs from look/labels.json")
    if not any(f"LABELS sha256 {ah} " in ln for ln in T.prereg_lines()):
        raise T.Refuse("no LABELS line in final_prereg.log for this archive")
    if PH.TMP_LABELS.exists():
        shutil.rmtree(PH.TMP_LABELS)
    PH.TMP_LABELS.mkdir(parents=True)
    got = {}
    with tarfile.open(LOOK / "labels_archive.tar", "r") as tf:
        names = sorted(m.name for m in tf.getmembers())
        if names != sorted(f"{c}.nii.gz" for c in lj["files"]):
            raise T.Refuse("archive members differ from look/labels.json")
        for c, v in sorted(lj["files"].items()):
            m = tf.getmember(f"{c}.nii.gz")
            assert m.isfile()
            data = tf.extractfile(m).read()
            h = hashlib.sha256(data).hexdigest()
            if h != v["sha256"] or len(data) != v["size"]:
                raise T.Refuse(f"{c}: archive member differs from look/labels.json")
            p = PH.TMP_LABELS / f"{c}.nii.gz"
            p.write_bytes(data)
            if T.sha256(p) != v["sha256"]:
                raise T.Refuse(f"{c}: written copy differs from look/labels.json")
            got[c] = v["sha256"]
    log(f"verified label copy: archive {ah[:16]} == labels.json; {len(got)} files checked against labels.json and "
        f"written to {T.rel(PH.TMP_LABELS)}")
    return got, ah


def score_one(job):
    cid, arms_maps, lsha = job
    import nibabel as nib
    import numpy as np
    import final_common as F
    import fin_look as L
    lp = PH.TMP_LABELS / f"{cid}.nii.gz"
    if T.sha256(lp) != lsha:
        raise RuntimeError(f"{cid}: verified label copy changed")
    lbl = nib.load(str(lp))
    raw = np.asarray(lbl.dataobj)
    G = np.rint(raw)
    assert np.array_equal(G, raw) and G.min() >= 0 and G.max() <= T.LABEL_MAX, cid
    G = G.astype(np.uint8)
    sp = tuple(float(s) for s in lbl.header.get_zooms()[:3])
    sc = F.Scorer(G, T.N_ORG, sp)
    rec = sc.frame_record()
    lut = F.ts_lut("flare")
    for a, (p, h) in arms_maps.items():
        if T.sha256(p) != h:
            raise RuntimeError(f"{a}/{cid}: map changed")
        seg, _ = L.read_raw(p)
        Y = lut[seg]
        assert Y.shape == G.shape, (a, cid)
        rec["arms"][a] = sc.score(Y, surface=True)
    return cid, {"NATIVE": rec}


def same_record(a, b):
    import numpy as np
    for k in ("vox", "inter", "present", "status"):
        if list(a[k]) != list(b[k]):
            return False
    for k in ("ml", "dice", "hd95", "nsd"):
        x, y = np.array(a[k], float), np.array(b[k], float)
        if not np.allclose(x, y, rtol=0, atol=1e-9, equal_nan=True):
            return False
    return True


def main():
    import final_common as F
    T.kst_gate()
    log = PH.log
    ids, maps = gates()
    log(f"post hoc score gates OK: {len(ids)} scans; maps of {NEW} == DONE; maps of {CHECK} == look/maps.json")
    wrote_note = note_once()
    t0 = time.time()
    try:
        lsha, ah = verified_copy(log)
        todo = [(c, {a: maps[a][c] for a in NEW + CHECK}, lsha[c]) for c in ids
                if not (PERVOL / f"{c}.json").is_file()]
        from multiprocessing import Pool
        with Pool(WORKERS) as pool:
            for c, rec in pool.imap_unordered(score_one, todo, chunksize=1):
                PH.write_json(PERVOL / f"{c}.json", rec)
                log(f"scored {c} ({time.time() - t0:.0f} s)")
    finally:
        if PH.TMP_LABELS.exists():
            shutil.rmtree(PH.TMP_LABELS)
            log(f"temp label folder {T.rel(PH.TMP_LABELS)} deleted")
    # equality check of the scoring path + assemble R with the STORED look records of F0, FULL_s1337, STOCK
    R, mism = {}, []
    for c in ids:
        mine = json.load(open(PERVOL / f"{c}.json"))["NATIVE"]
        st = json.load(open(LOOK / "pervol" / f"{c}.json"))["NATIVE"]
        if mine["ref"] != st["ref"] or mine["shape"] != st["shape"]:
            mism.append([c, "ref"])
        for a in CHECK:
            if not same_record(mine["arms"][a], st["arms"][a]):
                mism.append([c, a])
        rec = dict(st)
        rec["arms"] = {a: st["arms"][a] for a in ("F0", "FULL_s1337", "STOCK")}
        rec["arms"].update({a: mine["arms"][a] for a in NEW})
        R[c] = {"NATIVE": rec}
    log(f"re-score check of F0 / FULL_s1337 vs stored look/pervol: {len(mism)} mismatches")
    res = analyse(R, ids)
    res["post_hoc"] = True
    res["status"] = "EXPLORATORY, NOT REGISTERED: post hoc second reading of the FLARE-T50 labels (owner-approved 2026-11-13)"
    res["rescore_check"] = {"arms": CHECK, "n_scans": len(ids), "mismatches": mism, "identical": not mism,
                            "what": "F0 / FULL_s1337 re-scored from their look maps with this code vs the stored look/pervol "
                                    "records (vox, inter, presence, status exact; ml, Dice, HD95, NSD within 1e-9); also "
                                    "the reference record of every scan"}
    res["labels"] = {"archive_sha256": ah, "verified_against": "look/labels.json", "temp_folder_deleted": not PH.TMP_LABELS.exists(),
                     "note_written_this_run": wrote_note}
    res["models"] = {a: {"label": PH.ARMS[a]["label"], "checkpoint_sha256": {str(k): v for k, v in PH.ARMS[a]["shas"].items()},
                         "record": T.rel(PH.ARMS[a]["record"])} for a in NEW}
    res["code_sha256"] = {n: T.sha256(HERE / n) for n in ("ph_score.py", "ph_common.py")}
    res["finished"] = T.now()
    PH.write_json(PH.OUT / "results_posthoc.json", res)
    log(f"post hoc results written ({time.time() - t0:.0f} s)")
    return 0


# ---------------------------------------------------------------------------------------------------- post hoc analysis
def analyse(R, ids):
    import numpy as np
    import final_common as F
    import fin_look as L
    O = list(range(1, 14))
    names = F.FLARE13
    arms_all = ["F0", "FULL_s1337", "STOCK"] + NEW
    out = {"frame": "NATIVE", "n_scans": len(ids), "organs": names,
           "bootstrap": f"scan bootstrap final_common.cluster_boot ({F.NBOOT}, seed {F.SEED_CLUSTER})"}
    absm = {}
    for a in arms_all:
        v = L.macro_values(R, "NATIVE", a, O, ids)
        cb = F.cluster_boot(v)
        absm[a] = {"post_hoc": a in NEW, "mean": cb["mean"], "ci": cb["ci"], "n": cb["n"]}
    out["absolute_macro_dice"] = absm
    con = {}
    for a in NEW + ["STOCK", "FULL_s1337"]:
        for b in ("F0", "FULL_s1337"):
            if a == b:
                continue
            d = L.macro_delta(R, "NATIVE", a, b, O, ids)
            cb = F.cluster_boot(d)
            tc = F.t_ci(list(d.values()))
            po = {}
            for o in O:
                od = L.organ_delta(R, "NATIVE", a, b, o, ids)
                po[names[o - 1]] = {"mean_delta": F.nanmean(list(od.values())) if od else None, "n": len(od),
                                    "up": int(sum(x > 0 for x in od.values())), "down": int(sum(x < 0 for x in od.values()))}
            con[f"{a} - {b}"] = {"post_hoc": a in NEW, "mean": cb["mean"], "ci": cb["ci"], "t_ci": tc["ci"], "n": cb["n"],
                                 "scans_up": cb["up"], "scans_down": cb["down"], "per_organ": po,
                                 "per_scan": d}
    out["contrasts"] = con
    org, vol, fnfp = {}, {}, {}
    for a in arms_all:
        org[a], vol[a] = {}, {}
        fn_tot, fp_tot, fn_o, fp_o = 0, 0, {}, {}
        for o in O:
            j = o - 1
            dv, rv = [], []
            nfn = nfp = 0
            for c in ids:
                f = R[c]["NATIVE"]
                arm = f["arms"][a]
                if f["ref"]["present"][j]:
                    dv.append(F.organ_dice(arm, f["ref"], o))
                    rv.append(float(arm["ml"][j]) / float(f["ref"]["ml"][j]))
                nfn += arm["status"][j] == "FN"
                nfp += arm["status"][j] == "FP"
            org[a][names[j]] = {"mean_dice": F.nanmean(dv) if dv else None, "n": len(dv)}
            vol[a][names[j]] = {"median_ratio": float(np.median(rv)) if rv else None, "n": len(rv),
                                "q25": float(np.percentile(rv, 25)) if rv else None,
                                "q75": float(np.percentile(rv, 75)) if rv else None}
            fn_o[names[j]], fp_o[names[j]] = int(nfn), int(nfp)
            fn_tot += nfn
            fp_tot += nfp
        fnfp[a] = {"FN": int(fn_tot), "FP": int(fp_tot), "FN_per_organ": fn_o, "FP_per_organ": fp_o,
                   "FN_scans": sorted({c for c in ids for j in range(13) if R[c]["NATIVE"]["arms"][a]["status"][j] == "FN"}),
                   "FP_scans": sorted({c for c in ids for j in range(13) if R[c]["NATIVE"]["arms"][a]["status"][j] == "FP"})}
    out["per_organ_mean_dice"] = org
    out["volume_ratio"] = {"definition": "V_pred / V_ref per scan and organ (V_pred = all predicted voxels, mL; organs present "
                                         "in the reference, >= 1 mL); median over the scans", "arms": vol}
    out["fn_fp"] = {"definition": "conventions v1 status counts over 50 scans x 13 organs (FN: reference >= 1 mL, prediction "
                                  "< 1 mL; FP: the reverse)", "arms": fnfp}
    return out


if __name__ == "__main__":
    try:
        sys.exit(main())
    except T.Refuse as e:
        PH.log(f"[REFUSE] {e.msg}")
        sys.exit(e.code)
