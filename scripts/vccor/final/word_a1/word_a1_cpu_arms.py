"""vccor/final plan C, amendment word A1.2: the two CPU arms built from stored maps (no network, no GPU, no label archive).

  MERGE_N_dX   the class-wise merge of F0 and F1b_dX in which the NON-TARGET classes win: conventions v1 item6
               class_wise_merge 'MERGE_N (sensitivity; non-target wins)' (outputs/results/vccor/paper_tables_v2/
               analysis_conventions_v1.json, sha256 309b9fe3..., pinned by A1 as conventions v1), computed by the frozen
               operator scripts/vccor/paper_v2/v2_common.merge_maps (sha256 079e529e..., CODE line in outputs/logs/vccor/
               paper_v2.log; IMPORT line in final_prereg.log) on the raw TS-id maps of F0 and F1b_dX. T (the site classes) = the
               TS 'total' classes that word_common.ts_lut maps to one of the 13 adapted WORD organs (16 classes: the 12 primary
               organs' classes incl. both adrenal glands and both kidney-cyst classes, and colon). Every other class (incl. the
               femurs: WORD's femoral heads are scored against them but are not adapted) is non-target and equals F0 exactly.
               Output: TS 'total' ids on the raw grid (scored with lut 'ts', as F0 and F1b). Per scan it also records the
               label-free agreement of F1b_dX with F0 on every non-target class (voxel counts) and the voxels MERGE_N changes
               relative to F1b_dX (the 'merged classes' of A1.2 S2).
  LABELMAP_dX  F0's raw map relabelled into WORD's analysis ids with draw dX's class-merge rules, fixed from the draw's four
               training scans only by the rule-construction procedure registered for the closed cohorts on 2026-11-06:
               outputs/results/vccor/ontology_oracle/plan_ontology_oracle.json 'rule_from_labelled4' (sha256 44f7d2b7...,
               PREREG 2026-11-06 09:21:28; code scripts/vccor/review_fix_2026_11_06/ontology_oracle.py sha256 72717692...):
                 per adapted organ o (1-13) and training scan c: the reference-only voxels of o (WORD GT == o and the
                 name-mapped TS label != o; o present iff >= 20 voxels, as the protocol atlas), their composition by raw TS
                 class, shares below 1 % dropped (the atlas frac_dict rule), mean over the 4 scans (an absent share counts 0);
                 a TS class outside WORD's name map (not background; word_common.ts_lut == 0) with mean share >= 0.10 is
                 merged into o.
               WORD additions (an actual map, not an oracle bound; fixed in A1.2): a class that qualifies for two organs goes
               to the organ with the larger mean share (ties: the lower organ id); every other class keeps the name map
               (word_common.ts_lut, the map F0 is scored with). Inputs: prep/labels/gt_ts/WORD_<cid>.nii.gz (the WORD GT of the
               draw scans on TS's 1.5 mm grid, derived on 2026-10-23 from the 12 already-read training labels; right adrenal
               17 -> 12, WORD's single adrenal class) and prep/ts_cpu/<cid>/part_<t>.nii.gz (TS's own zero-shot parts of the
               same scans on the same grid, merged in TS's order 291..295 == word_prep.stage_labels), each checked against
               prep/labels/manifest.json. Output: analysis ids on the raw grid (lut 'id'); the rules in
               <pred>/LABELMAP_dX__word_test30/rules.json.
Both write <pred>/<ARM>_dX__word_test30/{raw/<cid>.nii.gz, rec/<cid>.json, DONE.json} for the 30 WORD test scans (DONE last;
per_scan_raw_sha256 as predict_final.py writes it; the sha256 of every input DONE record). Idempotent: a DONE arm is left
alone. A real run (no --test-* option) first checks the registration (word_a1_common.real_job_guard: no look, OSF gate,
frozen code), so a hand start before the registration refuses.

usage (the runner):  .venv/bin/python -B scripts/vccor/final/word_a1/word_a1_cpu_arms.py --arm MERGE_N --draw d1
                     .venv/bin/python -B scripts/vccor/final/word_a1/word_a1_cpu_arms.py --arm LABELMAP --draw d1

Version of 2026-11-06.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import word_a1_common as A

F, W = A.F, A.W
ADAPTED = list(range(1, 14))
MIN_MEAN_SHARE = 0.10
FRAC_MIN = 0.01
MIN_VOX = 20
TS_ORDER = (291, 292, 293, 294, 295)
OPERATOR_MERGE = ("conventions v1 item6.class_wise_merge MERGE_N (non-target wins) == scripts/vccor/paper_v2/"
                  "v2_common.merge_maps")
PROCEDURE_LABELMAP = ("plan_ontology_oracle.json rule_from_labelled4 (non-ontology TS classes with mean share >= 0.10 of the "
                      "organ's reference-only voxels on the four labelled scans; shares < 1 % dropped; organ present iff "
                      ">= 20 voxels) + A1.2: conflicts to the larger mean share (ties: lower organ id); other classes keep "
                      "word_common.ts_lut")


def v2_common():
    """The frozen MERGE operator module (scripts/vccor/paper_v2/v2_common.py), hash-checked."""
    h = F.sha256(A.V2_COMMON)
    if h != A.V2_COMMON_SHA256:
        raise A.Refuse(f"{F.rel(A.V2_COMMON)} sha256 {h[:16]} != the pinned {A.V2_COMMON_SHA256[:16]}")
    p = str(A.V2_COMMON.parent)
    if p not in sys.path:
        sys.path.insert(0, p)
    import v2_common as V2
    return V2


def site_lut():
    """bool[256]: the TS 'total' classes that word_common.ts_lut maps to an adapted WORD organ (1-13): T of conventions v1
    item6 for the WORD-tuned model."""
    import numpy as np
    lut = W.ts_lut()
    return (lut >= 1) & (lut <= 13) & (np.arange(256) > 0)


def part_to_total_luts():
    """part id -> TS 'total' id per part (== word_prep.part_to_total_luts)."""
    import numpy as np
    from totalsegmentator.map_to_binary import class_map, class_map_5_parts, map_taskid_to_partname_ct
    inv = {v: k for k, v in class_map["total"].items()}
    out = {}
    for t in TS_ORDER:
        lut = np.zeros(256, np.uint8)
        for j, n in class_map_5_parts[map_taskid_to_partname_ct[t]].items():
            lut[j] = inv[n]
        out[t] = lut
    return out


def merged_parts(parts, p2t):
    """TS's merged 'total' map from its five part maps (TS's merge order: a later part's foreground overwrites)."""
    import numpy as np
    shp = parts[TS_ORDER[0]].shape
    Tf = np.zeros(shp, np.uint8)
    for t in TS_ORDER:
        P = parts[t]
        assert P.shape == shp, (t, P.shape, shp)
        nz = P > 0
        Tf[nz] = p2t[t][P[nz]]
    return Tf


def _save_like(arr, ref_img, path):
    """Save a uint8 map with the reference image's affine / header (atomic); returns its sha256."""
    import nibabel as nib
    import numpy as np
    path = Path(path)
    rp = str(path.resolve())
    if not (rp.startswith(str(F.OUT) + os.sep) or rp.startswith("/tmp/")):
        raise A.Refuse(f"the A1.2 CPU arms write only under {F.rel(F.OUT)} or /tmp: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    hdr = ref_img.header.copy()
    hdr.set_data_dtype(np.uint8)
    hdr.set_slope_inter(None, None)
    im = nib.Nifti1Image(np.asarray(arr, np.uint8), ref_img.affine, header=hdr)
    im.set_qform(ref_img.affine, code=int(ref_img.header["qform_code"]) or 1)
    im.set_sform(ref_img.affine, code=int(ref_img.header["sform_code"]) or 1)
    tmp = path.with_name(f".{path.name.replace('.nii.gz', '')}.tmp{os.getpid()}.nii.gz")
    nib.save(im, str(tmp))
    back = np.asarray(nib.load(str(tmp)).dataobj)
    assert back.shape == arr.shape and np.array_equal(back, arr), f"{path}: written map does not read back"
    os.replace(tmp, path)
    return F.sha256(path)


def _wj(path, obj):
    F.write_json(path, obj)
    return F.sha256(path)


def _load_raw(path, sha=None):
    import nibabel as nib
    import numpy as np
    if sha is not None and F.sha256(path) != sha:
        raise A.Refuse(f"{path}: sha256 differs from its DONE record")
    im = nib.load(str(path))
    a = np.asarray(im.dataobj)
    assert a.min() >= 0 and a.max() <= 255, path
    return a.astype(np.uint8), im


def _done(path):
    p = Path(path)
    if not p.is_file():
        raise A.Refuse(f"input not DONE: {p}")
    return json.load(open(p)), F.sha256(p)


# ---------------------------------------------------------------------------------------------------- MERGE_N
def merge_stats(Z, Fm, MN, isT, vox_ml):
    """Per non-target TS class c (outside T) present in F0 or F1b: [voxels in F0, voxels in F1b, voxels in both]; and the
    voxels where MERGE_N differs from F1b, as 'F1b class -> MERGE_N class' counts."""
    import numpy as np
    J = np.bincount((Z.astype(np.int64) * 256 + Fm).ravel(), minlength=65536).reshape(256, 256)
    n0, n1 = J.sum(1), J.sum(0)
    nt = {}
    for c in range(1, 256):
        if isT[c] or (n0[c] == 0 and n1[c] == 0):
            continue
        nt[str(c)] = [int(n0[c]), int(n1[c]), int(J[c, c])]
    d = MN != Fm
    ch = {}
    if d.any():
        K = np.bincount((Fm[d].astype(np.int64) * 256 + MN[d]), minlength=65536)
        for k in np.flatnonzero(K):
            ch[f"{k // 256}->{k % 256}"] = int(K[k])
    return {"vox_ml": float(vox_ml), "nontarget_f0_f1b_both": nt, "changed_vs_f1b": ch, "n_changed_vs_f1b": int(d.sum())}


def build_merge_n(f0_dir, f1b_dir, out_dir, ids, draw, log=print):
    """MERGE_N_<draw> for the scans `ids` from the DONE maps of F0 and F1b_<draw>; returns the DONE record (written last)."""
    import numpy as np
    out_dir = Path(out_dir)
    f0_dir, f1b_dir = Path(f0_dir), Path(f1b_dir)
    if (out_dir / "DONE.json").is_file():
        log(f"[skip] MERGE_N_{draw} already DONE")
        return json.load(open(out_dir / "DONE.json"))
    V2 = v2_common()
    isT = site_lut()
    d0, h0 = _done(f0_dir / "DONE.json")
    d1, h1 = _done(f1b_dir / "DONE.json")
    per, recs, t0 = {}, {}, time.time()
    for c in ids:
        Z, zi = _load_raw(f0_dir / "raw" / f"{c}.nii.gz", d0["per_scan_raw_sha256"][c])
        Fm, fi = _load_raw(f1b_dir / "raw" / f"{c}.nii.gz", d1["per_scan_raw_sha256"][c])
        assert Z.shape == Fm.shape and np.allclose(zi.affine, fi.affine, atol=1e-3), f"{c}: F0 / F1b grids differ"
        _mt, mn = V2.merge_maps(Z, Fm, isT)
        assert mn.dtype == np.uint8 and mn.shape == Z.shape
        h = _save_like(mn, zi, out_dir / "raw" / f"{c}.nii.gz")
        vox_ml = float(np.prod(zi.header.get_zooms()[:3])) / 1000.0
        st = merge_stats(Z, Fm, mn, isT, vox_ml)
        rec = {"cid": c, "arm": f"MERGE_N_{draw}", "raw_sha256": h, "f0_raw_sha256": d0["per_scan_raw_sha256"][c],
               "f1b_raw_sha256": d1["per_scan_raw_sha256"][c], "written": F.now()} | st
        _wj(out_dir / "rec" / f"{c}.json", rec)
        per[c], recs[c] = h, st
        log(f"MERGE_N_{draw} {c}: {st['n_changed_vs_f1b']} voxels differ from F1b_{draw}")
    done = {"job": f"MERGE_N_{draw}__word_test30", "arm": f"MERGE_N_{draw}", "draw": draw, "kind": "cpu_arm",
            "operator": OPERATOR_MERGE, "v2_common_sha256": F.sha256(A.V2_COMMON),
            "site_classes_T": [int(t) for t in np.flatnonzero(isT)], "lut_kind": "ts", "ids": list(ids), "n": len(ids),
            "inputs": {"F0_done_sha256": h0, f"F1b_{draw}_done_sha256": h1, "F0_dir": F.rel(f0_dir),
                       "F1b_dir": F.rel(f1b_dir)},
            "per_scan_raw_sha256": per, "per_scan_merge": recs, "minutes": round((time.time() - t0) / 60, 2),
            "finished": F.now(), "code_sha256": F.sha256(Path(__file__).resolve())}
    _wj(out_dir / "DONE.json", done)
    return done


# ---------------------------------------------------------------------------------------------------- LABELMAP
def ontology_classes(name_lut):
    """The TS classes inside WORD's name map (any analysis id > 0): never merged elsewhere."""
    return {int(t) for t in range(1, 256) if name_lut[t] > 0}


def scan_shares(G, Tf, name_lut, organs=ADAPTED, min_vox=MIN_VOX, frac_min=FRAC_MIN):
    """{organ: record}: the composition by raw TS class of organ o's reference-only voxels (G == o and the name-mapped TS
    label != o), shares >= frac_min kept (== the protocol atlas: organ_record fn_raw with frac_dict). o is present iff it
    has >= min_vox reference voxels."""
    import numpy as np
    assert G.shape == Tf.shape, (G.shape, Tf.shape)
    Tn = name_lut[Tf]
    out = {}
    for o in organs:
        Go = G == o
        nG = int(Go.sum())
        if nG < min_vox:
            out[o] = {"present": False, "nG": nG, "n_ref_only": 0, "shares": {}}
            continue
        FN = Go & (Tn != o)
        nfn = int(FN.sum())
        sh = {}
        if nfn:
            cnt = np.bincount(Tf[FN].ravel(), minlength=256)
            for c in np.flatnonzero(cnt):
                f = float(cnt[c]) / nfn
                if f >= frac_min:
                    sh[int(c)] = f
        out[o] = {"present": True, "nG": nG, "n_ref_only": nfn, "shares": sh}
    return out


def build_rules(per_scan, name_lut, n_scans, min_share=MIN_MEAN_SHARE, organs=ADAPTED):
    """The draw's rules from {cid: scan_shares record}: mean share over the n_scans scans of every non-ontology TS class in
    each organ's reference-only voxels; a class with mean share >= min_share qualifies for the organ; a class that qualifies
    for several organs goes to the organ with the larger mean share (ties: the lower organ id). Returns the rules record
    (with 'lut': the 256-entry TS id -> analysis id map = the name map + the merges)."""
    import numpy as np
    onto = ontology_classes(name_lut)
    mean = {o: {} for o in organs}
    for cid in sorted(per_scan):
        for o in organs:
            for c, f in per_scan[cid][o]["shares"].items():
                c = int(c)
                if c == 0 or c in onto:
                    continue
                mean[o][c] = mean[o].get(c, 0.0) + f / n_scans
    qual = {}
    for o in organs:
        for c, s in mean[o].items():
            if s >= min_share:
                qual.setdefault(c, []).append((o, s))
    merged, conflicts = {}, []
    for c in sorted(qual):
        cand = sorted(qual[c], key=lambda x: (-x[1], x[0]))
        merged[c] = cand[0][0]
        if len(cand) > 1:
            conflicts.append({"ts_class": c, "candidates": [[o, s] for o, s in cand], "assigned": cand[0][0]})
    lut = np.asarray(name_lut, np.uint8).copy()
    for c, o in merged.items():
        lut[c] = o
    return {"merged": {str(c): int(o) for c, o in merged.items()}, "conflicts": conflicts,
            "mean_shares": {str(o): {str(c): s for c, s in sorted(mean[o].items())} for o in organs},
            "n_scans": n_scans, "min_mean_share": min_share, "frac_min": FRAC_MIN, "min_vox": MIN_VOX,
            "lut": [int(x) for x in lut], "procedure": PROCEDURE_LABELMAP}


def training_cases(draw, prep_dir, log=print):
    """[(cid, G, Tf)] of the draw's four training scans from the prep's derived files (each checked against the labels
    manifest): G = the WORD GT on TS's 1.5 mm grid (right adrenal 17 -> 12), Tf = TS's merged zero-shot map on that grid."""
    import nibabel as nib
    import numpy as np
    prep_dir = Path(prep_dir)
    man = json.load(open(prep_dir / "labels" / "manifest.json"))
    ids = W.draw(draw)
    seen = set(man.get("gt_opened_training", []))
    if not set(ids) <= seen:
        raise A.Refuse(f"LABELMAP_{draw}: draw scans {sorted(set(ids) - seen)} are not among the prep's read training labels")
    p2t = part_to_total_luts()
    out = []
    for c in ids:
        rec = man["scans"][c]
        gp = prep_dir / "labels" / "gt_ts" / f"WORD_{c}.nii.gz"
        if F.sha256(gp) != rec["sha256"]["gt_ts"]:
            raise A.Refuse(f"{gp}: sha256 differs from the labels manifest")
        G = np.rint(np.asarray(nib.load(str(gp)).dataobj)).astype(np.uint8)
        assert set(np.unique(G).tolist()) <= set(range(0, 14)) | {W.ADRENAL_R_TMP}, (c, np.unique(G))
        G[G == W.ADRENAL_R_TMP] = W.ADRENAL
        parts = {}
        for t in TS_ORDER:
            pp = prep_dir / "ts_cpu" / c / f"part_{t}.nii.gz"
            if F.sha256(pp) != rec["capture_sha256"][f"part_{t}"]:
                raise A.Refuse(f"{pp}: sha256 differs from the labels manifest")
            parts[t] = np.asarray(nib.load(str(pp)).dataobj).astype(np.uint8)
        out.append((c, G, merged_parts(parts, p2t)))
        log(f"LABELMAP_{draw}: training scan {c} (derived 1.5 mm files, manifest-checked)")
    return out


def rules_for_draw(cases, name_lut):
    per = {c: scan_shares(G, Tf, name_lut) for c, G, Tf in cases}
    r = build_rules(per, name_lut, len(cases))
    r["per_scan"] = {c: {str(o): v for o, v in rec.items()} for c, rec in per.items()}
    r["train_ids"] = [c for c, _, _ in cases]
    return r


def build_labelmap(f0_dir, out_dir, ids, draw, rules, log=print):
    """LABELMAP_<draw> for the scans `ids`: lut[F0 raw]; returns the DONE record (written last)."""
    import numpy as np
    out_dir = Path(out_dir)
    f0_dir = Path(f0_dir)
    if (out_dir / "DONE.json").is_file():
        log(f"[skip] LABELMAP_{draw} already DONE")
        return json.load(open(out_dir / "DONE.json"))
    d0, h0 = _done(f0_dir / "DONE.json")
    rules = dict(rules, draw=draw, written=F.now())
    hr = _wj(out_dir / "rules.json", rules)
    lut = np.asarray(rules["lut"], np.uint8)
    merged = {int(c) for c in rules["merged"]}
    per, recs, t0 = {}, {}, time.time()
    for c in ids:
        Z, zi = _load_raw(f0_dir / "raw" / f"{c}.nii.gz", d0["per_scan_raw_sha256"][c])
        LM = lut[Z]
        h = _save_like(LM, zi, out_dir / "raw" / f"{c}.nii.gz")
        cnt = np.bincount(Z.ravel(), minlength=256)
        st = {"relabelled_vox": {str(k): int(cnt[k]) for k in sorted(merged) if cnt[k]},
              "vox_ml": float(np.prod(zi.header.get_zooms()[:3])) / 1000.0}
        _wj(out_dir / "rec" / f"{c}.json", {"cid": c, "arm": f"LABELMAP_{draw}", "raw_sha256": h,
                                             "f0_raw_sha256": d0["per_scan_raw_sha256"][c], "rules_sha256": hr,
                                             "written": F.now()} | st)
        per[c], recs[c] = h, st
    done = {"job": f"LABELMAP_{draw}__word_test30", "arm": f"LABELMAP_{draw}", "draw": draw, "kind": "cpu_arm",
            "procedure": PROCEDURE_LABELMAP, "ontology_plan": F.rel(A.ONTOLOGY_PLAN),
            "ontology_plan_sha256": A.ONTOLOGY_PLAN_SHA256, "rules_file": F.rel(out_dir / "rules.json"),
            "rules_sha256": hr, "merged": rules["merged"], "train_ids": rules.get("train_ids"), "lut_kind": "id",
            "ids": list(ids), "n": len(ids), "inputs": {"F0_done_sha256": h0, "F0_dir": F.rel(f0_dir)},
            "per_scan_raw_sha256": per, "per_scan_relabel": recs, "minutes": round((time.time() - t0) / 60, 2),
            "finished": F.now(), "code_sha256": F.sha256(Path(__file__).resolve())}
    _wj(out_dir / "DONE.json", done)
    return done


# ---------------------------------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=["MERGE_N", "LABELMAP"])
    ap.add_argument("--draw", required=True, choices=A.DRAWS)
    a = ap.parse_args(argv)
    F.be_nice(15)
    log = F.Log(F.LOGD / f"final_word_a12_{a.arm}_{a.draw}.log")
    try:
        A.real_job_guard(Path(__file__).resolve(), log)
        dirs = A.arm_dirs(W.DIR / "pred")
        ids = list(W.split()["test"])
        if a.arm == "MERGE_N":
            build_merge_n(dirs["F0"], dirs[f"F1b_{a.draw}"], dirs[f"MERGE_N_{a.draw}"], ids, a.draw, log)
        else:
            cases = training_cases(a.draw, W.PREP, log)
            rules = rules_for_draw(cases, W.ts_lut())
            build_labelmap(dirs["F0"], dirs[f"LABELMAP_{a.draw}"], ids, a.draw, rules, log)
    except A.Refuse as e:
        log(f"[REFUSE] {e.msg}")
        return e.code
    log(f"{a.arm}_{a.draw} DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
