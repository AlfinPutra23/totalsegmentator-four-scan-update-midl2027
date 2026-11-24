"""vccor/final plan C LOOK (CPU, <= 6 workers, nice 15; runs ONCE): "is it the model, or the labels?" on WORD, the cohort
whose labels were never opened: TotalSegmentator, VISTA3D and MRSegmentator zero-shot vs after a four-scan alignment
(F1b-equivalent per model, 3 shared draws), WORD test set (30), native raw grid, full field, conventions v1 (plan_word.json).

Steps: preflight (registered plan, frozen code, conventions + labelsTs zip sha pinned in the plan); every arm's maps checked
against their DONE records and hashed into look/maps.json; the GT_OPENED marker (O_EXCL); the 30 test labels read from the
labelsTs zip IN MEMORY; per scan: v1 records (fin_look) + the plan-E decomposition records (dec_common); analyse().
Arms not run are reported NOT RUN; the rules use the draws every compared model ran (pre-specified).

usage:  CUDA_VISIBLE_DEVICES="" nice -n 15 .venv/bin/python -B scripts/vccor/final/word_score.py --workers 6
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import final_common as F  # noqa: E402
import fin_look as L  # noqa: E402
import word_common as W  # noqa: E402

DRAWS = list(W.DRAW_SEEDS)
PAIRS = [("TS", "VISTA3D"), ("TS", "MRSeg"), ("VISTA3D", "MRSeg")]
REQUIRED = ["F0", "V0", "M0", "F1b_d1", "VF_d1", "MF_d1"]


def arms():
    """arm -> (raw dir, lut kind, DONE file)."""
    P = W.DIR / "pred"
    a = {"F0": (P / "F0__word_test30", "ts"), "V0": (P / "V0__word_test30", "vista"), "M0": (P / "M0__word_test30", "mrseg")}
    for d in DRAWS:
        a[f"F1b_{d}"] = (P / f"word_F1b_{d}__word_test30", "ts")
        a[f"VF_{d}"] = (P / f"VF_{d}__word_test30", "vista")
        a[f"MF_{d}"] = (P / f"MF_{d}__word_test30", "mrseg")
    a["VF5k_d1"] = (P / "VF5k_d1__word_test30", "vista")
    return {k: (v[0] / "raw", v[1], v[0] / "DONE.json") for k, v in a.items()}


def zero(m):
    return W.MODELS3[m]["zero"]


def aligned(m, d):
    return W.MODELS3[m]["aligned"].format(d=d)


def score_of(R, arm, organs, ids, penalised=True):
    return L.macro_values(R, "NATIVE", arm, organs, ids, penalised)


def boot_ratio(e0, e1d, nboot=F.NBOOT, seed=F.SEED_TWO_LEVEL):
    """S = (E0 - E1) / E0 with E0 = mean_i e0_i and E1 = mean over draws of mean_i e1_{d,i}; crossed two-level bootstrap
    (scans and draws resampled independently, == m2_cluster's scheme; one draw -> the scan bootstrap). e0 {cid: v};
    e1d {draw: {cid: v}}. Returns (S, ci)."""
    import numpy as np
    draws = sorted(e1d)
    cids = sorted(set(e0) & set.intersection(*[set(v) for v in e1d.values()])) if draws else []
    if not cids:
        return None, [None, None]
    a0 = np.array([e0[c] for c in cids])
    A1 = np.array([[e1d[d][c] for c in cids] for d in draws])
    S = float((a0.mean() - A1.mean(1).mean()) / a0.mean()) if a0.mean() > 0 else None
    rng = np.random.default_rng(seed)
    n, k = len(cids), len(draws)
    reps = np.empty(nboot)
    for b in range(nboot):
        w = rng.multinomial(n, np.full(n, 1.0 / n)).astype(float)
        u = rng.multinomial(k, np.full(k, 1.0 / k)).astype(float) if k > 1 else np.ones(1)
        E0 = (w @ a0) / w.sum()
        E1 = float(u @ ((A1 @ w) / w.sum()) / u.sum())
        reps[b] = (E0 - E1) / E0 if E0 > 0 else np.nan
    reps = reps[np.isfinite(reps)]
    return S, [float(np.quantile(reps, 0.025)), float(np.quantile(reps, 0.975))]


def rank_class(g0, g1, dg):
    """Pre-specified ranking-change classes of a model pair from the CIs of the zero-shot gap g0, the aligned gap g1 and the
    gap change dg = g1 - g0 (each a dict with 'ci')."""
    def sig(ci):
        lo, hi = ci
        if lo is None or hi is None:
            return None
        return 1 if lo > 0 else (-1 if hi < 0 else 0)

    s0, s1, sd = sig(g0["ci"]), sig(g1["ci"]), sig(dg["ci"])
    if None in (s0, s1, sd):
        return "NOT EVALUABLE"
    if s0 != 0 and s1 != 0 and s0 != s1:
        return "REVERSED"
    if s0 != 0 and s1 == 0 and sd == -s0:
        return "CLOSED"
    if s0 == 0 and s1 != 0 and sd == s1:
        return "OPENED"
    if s0 != 0 and s1 == s0 and sd != 0:
        return "SHIFTED"
    return "STABLE"


CHANGE = {"REVERSED", "CLOSED", "OPENED"}


def pair_test(R, ids, m, n, draws, organs):
    """Zero-shot gap g0 (scan bootstrap), aligned gap g1 (two-level over the draws both models ran), gap change dg."""
    z_m, z_n = score_of(R, zero(m), organs, ids), score_of(R, zero(n), organs, ids)
    g0v = {c: z_m[c] - z_n[c] for c in ids if c in z_m and c in z_n}
    per1, perd = {}, {}
    for d in draws:
        a_m, a_n = score_of(R, aligned(m, d), organs, ids), score_of(R, aligned(n, d), organs, ids)
        per1[d] = {c: a_m[c] - a_n[c] for c in ids if c in a_m and c in a_n}
        perd[d] = {c: per1[d][c] - g0v[c] for c in per1[d] if c in g0v}
    g0 = F.cluster_boot(g0v)
    g1, dg = F.two_level(per1), F.two_level(perd)
    g1["mean"], dg["mean"] = g1["M"], dg["M"]
    return {"pair": f"{m} - {n}", "draws": draws, "zero_shot_gap": {k: g0[k] for k in ("mean", "ci", "n")},
            "aligned_gap": {k: g1.get(k) for k in ("mean", "ci", "n_scans", "draw_means")},
            "gap_change": {k: dg.get(k) for k in ("mean", "ci", "n_scans")}, "class": rank_class(g0, g1, dg)}


def organ_vals(R, arm, o, ids):
    return {c: R[c]["NATIVE"]["arms"][arm]["dice"][o - 1] for c in ids if R[c]["NATIVE"]["ref"]["present"][o - 1]}


def systematic_share(R, ids, m, draws, organs):
    """C2: S_m = the share of the zero-shot penalised-macro error (100 - Dice) removed by the alignment (two-level CI); per
    organ the same on the organ's Dice (scans with the organ in the reference)."""
    e0 = {c: 100.0 - v for c, v in score_of(R, zero(m), organs, ids).items()}
    e1 = {d: {c: 100.0 - v for c, v in score_of(R, aligned(m, d), organs, ids).items()} for d in draws}
    S, ci = boot_ratio(e0, e1)
    po = {}
    for o in organs:
        z = organ_vals(R, zero(m), o, ids)
        a = {d: organ_vals(R, aligned(m, d), o, ids) for d in draws}
        so, cio = boot_ratio({c: 100 - v for c, v in z.items()}, {d: {c: 100 - v for c, v in a[d].items()} for d in draws})
        po[W.ORGANS[o - 1]] = {"S": so, "ci": cio}
    return {"model": m, "draws": draws, "S": S, "ci": ci, "E0_zero_shot_error": float(F.nanmean(list(e0.values()))),
            "claim_systematic_share_present": bool(ci[0] is not None and ci[0] > 0), "per_organ": po}


def analyse(R, ids, have, dec=None):
    """R {cid: frames}; have = arms run; dec {cid: {arm: dec records}} (optional). Pure function (smoke-tested)."""
    out = {"arms_run": sorted(have), "arms_not_run": sorted(set(arms()) - set(have)), "organs_primary": [W.ORGANS[o - 1] for o in W.PRIMARY]}
    dr = {m: [d for d in DRAWS if aligned(m, d) in have] for m in W.MODELS3}
    ok = {m: zero(m) in have and bool(dr[m]) for m in W.MODELS3}
    rank = {}
    for m, n in PAIRS:
        if not (ok[m] and ok[n]):
            rank[f"{m} - {n}"] = "NOT RUN"
            continue
        both = [d for d in DRAWS if d in dr[m] and d in dr[n]]
        t = pair_test(R, ids, m, n, both, W.PRIMARY)
        t["per_organ"] = {W.ORGANS[o - 1]: pair_test(R, ids, m, n, both, [o])["class"] for o in W.PRIMARY}
        rank[f"{m} - {n}"] = t
    tested = [v for v in rank.values() if isinstance(v, dict)]
    out["C1_ranking"] = {"pairs": rank, "any_ranking_change": any(v["class"] in CHANGE for v in tested),
                         "reading": ("ZERO-SHOT RANKING IS CONVENTION-CONFOUNDED (at least one pair REVERSED / CLOSED / OPENED)"
                                     if any(v["class"] in CHANGE for v in tested) else "RANKING UNCHANGED BY ALIGNMENT"),
                         "zero_shot_order": sorted([m for m in W.MODELS3 if ok[m]], key=lambda m: -F.nanmean(list(score_of(R, zero(m), W.PRIMARY, ids).values()))),
                         "aligned_order": sorted([m for m in W.MODELS3 if ok[m]], key=lambda m: -F.nanmean(
                             [F.nanmean(list(score_of(R, aligned(m, d), W.PRIMARY, ids).values())) for d in dr[m]]))}
    out["C2_systematic_share"] = {m: systematic_share(R, ids, m, dr[m], W.PRIMARY) for m in W.MODELS3 if ok[m]}
    cmp_ = {}
    for m, n in PAIRS:
        if ok[m] and ok[n]:
            both = [d for d in DRAWS if d in dr[m] and d in dr[n]]
            sm, sn = systematic_share(R, ids, m, both, W.PRIMARY), systematic_share(R, ids, n, both, W.PRIMARY)
            cmp_[f"{m} - {n}"] = {"S_m": sm["S"], "S_n": sn["S"], "diff": (sm["S"] - sn["S"]) if None not in (sm["S"], sn["S"]) else None}
    out["C2_share_differences_descriptive"] = cmp_
    gains = {}
    for m in W.MODELS3:
        if not ok[m]:
            continue
        per = {d: L.macro_delta(R, "NATIVE", aligned(m, d), zero(m), W.PRIMARY, ids) for d in dr[m]}
        tl = F.two_level(per)
        rule = F.p1_rule(list(tl["scan_values"].values()), boot=tl["ci"])
        po = {}
        for o in W.PRIMARY:
            t = F.two_level({d: L.organ_delta(R, "NATIVE", aligned(m, d), zero(m), o, ids) for d in dr[m]})
            po[W.ORGANS[o - 1]] = {"M": t["M"], "ci": t["ci"], "ni": F.ni_class(t["ci"]), "harm": F.harm_class(t["ci"])}
        gains[m] = {"draws": dr[m], "macro": {k: tl.get(k) for k in ("M", "ci", "up", "down", "n_scans")}, "P1_rule": rule,
                    "per_organ": po, "all_organs_non_inferior": all(v["ni"] == "NON-INFERIOR" for v in po.values())}
    out["secondary_alignment_gains"] = gains
    if "VF5k_d1" in have and "VF_d1" in have:
        c = L.contrast(R, "NATIVE", "VF_d1", "VF5k_d1", W.PRIMARY, ids)
        c["reading"] = ("VISTA3D alignment RECIPE-SENSITIVE (|VF25k - VF5k| > 0.5)" if abs(c["macro_penalised"]["mean"] or 0) > 0.5
                        else "VISTA3D alignment insensitive to the budget (|difference| <= 0.5)")
        out["secondary_vista_budget"] = c
    out["secondary_colon_rectum"] = {m: F.two_level({d: L.organ_delta(R, "NATIVE", aligned(m, d), zero(m), 10, ids)
                                                     for d in dr[m]}) for m in W.MODELS3 if ok[m]}
    out["descriptive_femur_heads_zero_shot"] = {zero(m): {W.ORGANS[o - 1]: F.nanmean(list(organ_vals(R, zero(m), o, ids).values()))
                                                          for o in W.DESCRIPTIVE} for m in W.MODELS3 if zero(m) in have}
    sz = {c: round(max(float(x) for x in R[c]["NATIVE"]["spacing"]), 1) for c in ids if R.get(c, {}).get("NATIVE", {}).get("spacing")}
    strata = {}
    for z in sorted(set(sz.values())):
        ii = [c for c in ids if sz.get(c) == z]
        st = {"n": len(ii)}
        for m in W.MODELS3:
            if ok[m]:
                st[f"{zero(m)} macro"] = F.nanmean(list(score_of(R, zero(m), W.PRIMARY, ii).values()))
                st[f"{m} gain (mean over draws)"] = F.nanmean([F.nanmean(list(L.macro_delta(R, "NATIVE", aligned(m, d), zero(m),
                                                                                              W.PRIMARY, ii).values())) for d in dr[m]])
        strata[f"{z} mm"] = st
    out["descriptive_slice_spacing_strata"] = {"strata": strata, "note": "descriptive only: WORD's one per-scan acquisition variable "
                                               "(one centre, one SIEMENS scanner, non-contrast; word_common.COHORT_METADATA)"}
    if dec:
        import dec_common as E
        dd = {}
        for m in W.MODELS3:
            if not ok[m]:
                continue
            for d in dr[m]:
                recs = {"F0": {c: dec[c][zero(m)] for c in dec if zero(m) in dec[c]},
                        "FT": {c: dec[c][aligned(m, d)] for c in dec if aligned(m, d) in dec[c]}}
                cc = sorted(set(recs["F0"]) & set(recs["FT"]))
                means, _ = E.organ_means({k: {c: v[c] for c in cc} for k, v in recs.items()}, W.PRIMARY)
                dd[f"{m} {d}"] = means
        out["secondary_decomposition_plan_E_definitions"] = dd
    ep = F.PLAN_DIR["decomposition"] / "results/results.json"
    if ep.is_file() and ok["VISTA3D"] and ok["MRSeg"]:
        e3 = json.load(open(ep)).get("E3_exposure", {})
        w_v0 = F.cluster_boot(L.macro_delta(R, "NATIVE", "V0", "F0", W.PRIMARY, ids))
        w_m0 = F.cluster_boot(L.macro_delta(R, "NATIVE", "M0", "F0", W.PRIMARY, ids))
        out["secondary_exposure_contrast"] = {
            "WORD (no model saw the labels)": {"V0 - F0": w_v0, "M0 - F0": w_m0},
            "exposed benchmarks (plan E, E3; VISTA3D trained on these labels)": {k: {a: v[a]["macro_penalised"] for a in ("V0 - F0", "M0 - F0")}
                                                                                 for k, v in e3.items()},
            "note": "descriptive: organ sets differ (WORD 12 primary organs vs FLARE / AMOS 13)"}
    return out


def duplicates(ids, log=print):
    """Within-WORD header-identical groups of the test set (word_prep 'headers' stage, prep/overlap.json) are checked from the
    IMAGES before any label is read: Pearson r of the intensities on a 4-voxel stride grid (== the dedup convention); r >= 0.999
    -> the same image: the lower id is kept, the others are dropped from every analysis (reported)."""
    import nibabel as nib
    import numpy as np
    ov = json.load(open(W.PREP / "overlap.json"))
    out, drop = [], set()
    for grp in ov.get("within_word_header_groups", []):
        g = sorted(c for c in grp if c in ids)
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                a = np.asarray(nib.load(str(W.image_path(g[i]))).dataobj)[::4, ::4, ::4].astype(np.float64).ravel()
                b = np.asarray(nib.load(str(W.image_path(g[j]))).dataobj)[::4, ::4, ::4].astype(np.float64).ravel()
                r = float(np.corrcoef(a, b)[0, 1]) if a.shape == b.shape else None
                same = r is not None and r >= 0.999
                out.append({"pair": [g[i], g[j]], "r_stride4": r, "same_image": same})
                if same:
                    drop.add(g[j])
    log(f"WORD test duplicates check: {out}; dropped {sorted(drop)}")
    return out, sorted(drop)


def job(a):
    cid, arm_paths, shas, marker = a
    import numpy as np
    import dec_common as E
    rec = L.score_scan("word", cid, arm_paths, marker, shas=shas)
    G, sp = W.reference(cid, marker)
    vox = float(np.prod(sp)) / 1000.0
    dec = {}
    for arm, (p, lk) in arm_paths.items():
        seg, _ = L.read_raw(p)
        dec[arm] = E.scan_record(L.lut("word", lk)[seg], G, W.N_ORG, sp, vox)
    F.write_json(W.LOOK / "pervol" / f"{cid}.json", {"cid": cid, "frames": rec, "dec": dec})
    return cid, rec, dec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    assert a.workers <= F.MAX_WORKERS
    F.be_nice(15)
    log = F.job_log("word_score")
    plan = F.preflight(W.PLAN, HERE / "word_score.py", log, extra=[HERE / "fin_look.py", HERE / "word_common.py", HERE / "dec_common.py"])
    F.check_conventions(plan["implementation"]["conventions_sha256"])
    assert F.sha256(W.LABELS_TS_ZIP) == plan["implementation"]["labels_ts_zip_sha256"], "labelsTs zip changed since registration"
    if (W.LOOK / "results.json").exists():
        raise SystemExit("[REFUSE] the WORD look has already been made")
    ids = list(W.split()["test"])
    dup, dropped = duplicates(ids, log)
    ids = [c for c in ids if c not in dropped]
    have, files = [], {}
    for arm, (d, lk, done) in arms().items():
        if not done.is_file():
            if arm in REQUIRED:
                raise SystemExit(f"[REFUSE] required arm {arm}: predictions not DONE")
            continue
        if any(L.done_raw_sha(done, c) != F.sha256(d / f"{c}.nii.gz") for c in ids):
            raise SystemExit(f"[REFUSE] arm {arm}: maps do not match their DONE record")
        have.append(arm)
        files[arm] = ({c: str(d / f"{c}.nii.gz") for c in ids}, lk)
    shas = L.maps_manifest({a_: f for a_, (f, _) in files.items()})
    F.write_json(W.LOOK / "maps.json", {"sha256": shas, "arms": have, "written": F.now()})
    mk = F.open_marker(W.LOOK / "GT_OPENED", {"set": "WORD test (30)", "ids": ids, "maps_sha256": F.sha256(W.LOOK / "maps.json"),
                                             "declared": "FIRST look at the WORD test labels (pre-registered on OSF)"})
    log(f"GT_OPENED {F.rel(mk)}; scoring {len(have)} arms")
    jobs = [(c, {a_: (files[a_][0][c], files[a_][1]) for a_ in have}, {a_: shas[a_][c] for a_ in have}, str(mk)) for c in ids]
    R, D = {}, {}
    with Pool(a.workers) as pool:
        for c, rec, dec in pool.imap_unordered(job, jobs):
            R[c], D[c] = rec, dec
    res = analyse(R, ids, set(have), D)
    res.update(plan_sha256=F.sha256(F.plan_path(W.PLAN)), written=F.now(), duplicates_check=dup, dropped_as_duplicates=dropped)
    F.write_json(W.LOOK / "results.json", res)
    F.PLOG(f"RESULTS sha256 {F.sha256(W.LOOK / 'results.json')} {F.rel(W.LOOK / 'results.json')}")
    log(f"WORD look done: C1 {res['C1_ranking']['reading']}")


if __name__ == "__main__":
    main()
