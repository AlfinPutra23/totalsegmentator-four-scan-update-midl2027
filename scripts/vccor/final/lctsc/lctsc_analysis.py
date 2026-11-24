"""vccor/final LCTSC ANALYSIS (pure functions; no file access; unit-tested on synthetic records by test_lctsc.py).

Input R {pid: {'NATIVE': final_common.Scorer frame record}} (arms scored against LCTSC's test reference in LCTSC's 5 ids,
native grid, conventions v1: presence >= 1 mL on both sides, FP structures scored 0, penalised macro).

P1 (primary; alpha 0.05, not adjusted) = WORD A1's / FLARE-T50's P1 rule on the three K = 4 F1b draws vs F0, over the 4
ADAPTED structures (esophagus, heart, left lung, right lung): M = mean over the draws of mean_i D_{d,i},
D = macro(F1b_d) - macro(F0); PASS iff M >= +0.30 AND scan t-CI lower > 0 AND crossed two-level bootstrap CI lower > 0
(m2_cluster.boot2 via final_common.two_level, 10,000, seed 20261020) AND scans with a positive draw-averaged D >= ceil(2n/3).
Per draw: the single-draw form (t50_analysis.p1_single: scan bootstrap seed 20261022) -> k of n_run -> the combination
(GAIN_EVERY_DRAW ...). The pooled and single-draw rule implementations are imported read-only from the frozen
flaret50/t50_analysis.py (p1_pooled, p1_single, combination, organ_reading).
Organ costs (registered): per adapted structure, the two-level CI of its Dice change (F1b_d - F0), NON-INFERIOR /
INFERIOR / INCONCLUSIVE at margin -1.0 Dice (final_common.ni_class), harm class (final_common.harm_class).
SITE-MATCHED DRAWS (2026-11-13): draw S_s = 4 training patients of institution s, arm F1b_S_s. P1 is unchanged (all
24 test patients, M = mean of the three draw means). Draws NOT RUN (training failed twice or not DONE by the training
deadline, or the arm NOT RUN by the prediction runner) are left out; P1 runs on the draws that remain (two-level over >= 2
draws, the single-draw form for one; the combination over the draws run; t50_analysis.combination).
OWN-SITE vs OTHER-SITES (registered DESCRIPTIVE, not a test; review power simulation): per draw s, O_s = mean D over the test
patients of institution s - mean D over the other institutions' test patients (D = macro(F1b_S_s) - macro(F0)); O = mean
of O_s over the draws run (an institution main effect cancels in O when all three draws ran). 95% percentile CI from a
patient bootstrap stratified by institution (10,000, seed 20261022). Reported as an estimate with its CI only.
MISSING STRUCTURES (registered scoring rule, review SF2): a structure missing from a patient's reference (no ROI, or MISSING
by an lctsc_raster outcome) is NOT EVALUABLE for that patient: it is removed from every arm's macro and from every
per-structure contrast for that patient (mask_missing), never scored as a false positive.
Descriptive (registered): every patient (k of n); per-site P1 values (8 patients per site, scan bootstrap); spinal cord
Dice of F0 (F1b = F0 by construction: part 294 is TS's own in every arm); absolute macro per arm; per-draw
fin_look.contrast (HD95 / NSD / failures / FP events); volume ratios per structure (median V_pred / V_ref).
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent), str(HERE.parent / "flaret50"), str(HERE.parent.parent / "m2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import final_common as F  # noqa: E402  (frozen; read-only)
import fin_look as L  # noqa: E402  (frozen; read-only)
import t50_analysis as TA  # noqa: E402  (frozen FLARE-T50 analysis; read-only: p1_pooled, p1_single, combination, organ_reading)
import lctsc_common as C  # noqa: E402

FRAME = "NATIVE"
ADAPTED = list(C.ADAPTED)
DRAWS = {"S1": "F1b_S1", "S2": "F1b_S2", "S3": "F1b_S3"}       # site-matched draws (draw key = institution)


def md(R, arm, base, ids, organs=ADAPTED):
    return L.macro_delta(R, FRAME, arm, base, organs, ids)


def mask_missing(R, missing):
    """Registered missing-structure rule: missing {pid: [structure names]} -> a copy of R in which those structures are
    not present for the reference and for every arm (Dice nan), so F.macro and fin_look.organ_delta skip them."""
    import copy
    import math
    out = copy.deepcopy(R)
    for pid, names in (missing or {}).items():
        f = out.get(pid, {}).get(FRAME)
        if f is None:
            continue
        for nm in names:
            j = C.ORGANS.index(nm)
            f["ref"]["present"][j] = False
            for a in f["arms"].values():
                a["present"][j] = False
                a["dice"][j] = math.nan
                a["status"][j] = "NOT_EVALUABLE"
    return out


def own_vs_other(per, nboot=F.NBOOT, seed=F.SEED_CLUSTER):
    """per {draw S_s: {pid: D}} -> O = mean_s [mean_{site s} D - mean_{other sites} D]; stratified patient bootstrap CI."""
    import numpy as np
    draws = sorted(per)
    ids = sorted(set().union(*[set(v) for v in per.values()])) if per else []
    if not draws or not ids:
        return {"estimate": None, "ci": [None, None], "draws": draws}

    def stat(sample):
        vals = []
        for d in draws:
            own = [per[d][c] for c in sample if c in per[d] and C.site(c) == d]
            oth = [per[d][c] for c in sample if c in per[d] and C.site(c) != d]
            if own and oth:
                vals.append(float(np.mean(own)) - float(np.mean(oth)))
        return (float(np.mean(vals)) if vals else float("nan")), vals
    est, per_draw = stat(ids)
    strata = {s: [c for c in ids if C.site(c) == s] for s in sorted({C.site(c) for c in ids})}
    rng = np.random.RandomState(seed)
    reps = []
    for _ in range(nboot):
        smp = [x for s in sorted(strata) for x in rng.choice(strata[s], len(strata[s]), replace=True)]
        reps.append(stat(smp)[0])
    reps = np.array(reps)
    reps = reps[np.isfinite(reps)]
    lo, hi = (float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))) if len(reps) else (None, None)
    return {"estimate": est, "per_draw": dict(zip(draws, per_draw)), "ci": [lo, hi], "draws": draws,
            "n_per_site": {s: len(v) for s, v in strata.items()}, "bootstrap": f"patients stratified by institution ({nboot}, seed {seed})",
            "status": "registered descriptive (not a test)"}


def vol_ratio(R, arm, ids, o):
    import numpy as np
    v = []
    for c in ids:
        f = R.get(c, {}).get(FRAME)
        if f is None or not f["ref"]["present"][o - 1]:
            continue
        v.append(float(f["arms"][arm]["ml"][o - 1]) / float(f["ref"]["ml"][o - 1]))
    return float(np.median(v)) if v else None


def analyse(R, ids, have, not_run=None, missing=None):
    R = mask_missing(R, missing)
    have = set(have)
    assert "F0" in have, "P1 requires F0"
    draws = [d for d in sorted(DRAWS) if DRAWS[d] in have]
    assert draws, "P1 requires at least one F1b draw"
    out = {"scheme": "LCTSC (owner's yes 2026-11-13), WORD A1 / FLARE-T50 P1 rule", "frame": FRAME, "n_patients": len(ids),
           "ids": list(ids), "arms_scored": sorted(have), "arms_not_run": dict(not_run or {}), "draws_run": draws,
           "adapted": [C.ORGANS[o - 1] for o in ADAPTED], "descriptive": [C.ORGANS[o - 1] for o in C.DESCRIPTIVE]}
    per = {d: md(R, DRAWS[d], "F0", ids) for d in draws}
    p1 = TA.p1_pooled(per)
    out["P1"] = dict({k: p1[k] for k in ("M", "ci", "draw_means", "n_scans", "scans_up", "need_up", "t_ci", "scan_level_mean",
                                         "M_ge_0_30", "rule", "pass", "bootstrap")},
                     reading="GAIN" if p1["pass"] else "NO_GAIN")
    pd_, k = {}, 0
    for d in draws:
        pd_[d] = TA.p1_single(per[d])
        k += int(pd_[d]["pass"])
    comb = TA.combination(p1["pass"], k, len(draws))
    out["P1_per_draw"] = pd_
    out["P1_combination"] = {"combination": comb, "k_of_n_run": f"{k} of {len(draws)}"}
    sv = p1["scan_values"]
    up = sum(1 for v in sv.values() if v > 0)
    out["every_patient"] = {"k_improved": up, "n": len(sv), "every_patient": up == len(sv) and len(sv) > 0}
    po, cls = {}, {}
    for o in ADAPTED:
        t = F.two_level({d: L.organ_delta(R, FRAME, DRAWS[d], "F0", o, ids) for d in draws}) if len(draws) > 1 else None
        if t is None:
            cb = F.cluster_boot(L.organ_delta(R, FRAME, DRAWS[draws[0]], "F0", o, ids))
            t = {"M": cb["mean"], "ci": cb["ci"], "n_scans": cb["n"]}
        name = C.ORGANS[o - 1]
        po[name] = {"M": t["M"], "ci": t["ci"], "n_scans": t["n_scans"], "ni": F.ni_class(t["ci"]), "harm": F.harm_class(t["ci"]),
                    "median_volume_ratio": {"F0": vol_ratio(R, "F0", ids, o),
                                            **{DRAWS[d]: vol_ratio(R, DRAWS[d], ids, o) for d in draws}}}
        cls[name] = po[name]["ni"]
    out["organ_costs"] = dict(TA.organ_reading(cls), per_organ=po, margin=-F.NI_MARGIN)
    sites = {}
    for s in ("S1", "S2", "S3"):
        sid = [c for c in sv if C.site(c) == s]
        if sid:
            cb = F.cluster_boot({c: sv[c] for c in sid})
            sites[s] = {"n": len(sid), "mean": cb["mean"], "ci": cb["ci"], "up": int(sum(sv[c] > 0 for c in sid))}
    out["per_site_descriptive"] = sites
    out["own_site_vs_other_sites_descriptive"] = own_vs_other(per)
    out["not_evaluable_structures"] = {p: list(v) for p, v in (missing or {}).items() if v}
    cord = [o for o in C.DESCRIPTIVE]
    out["spinal_cord_descriptive"] = {a: F.nanmean([F.organ_dice(R[c][FRAME]["arms"][a], R[c][FRAME]["ref"], cord[0]) for c in ids
                                                    if FRAME in R.get(c, {})]) for a in sorted(have)}
    out["descriptive_per_draw_contrasts"] = {d: L.contrast(R, FRAME, DRAWS[d], "F0", ADAPTED, ids) for d in draws}
    out["absolute_macro"] = {a: F.nanmean(list(L.macro_values(R, FRAME, a, ADAPTED, ids).values())) for a in sorted(have)}
    out["claims_allowed"] = {
        "four labelled LCTSC scans improve TS on LCTSC's never-opened test patients (P1)": bool(p1["pass"]),
        "for each of the three draws": comb == "GAIN_EVERY_DRAW",
        "every patient improved": out["every_patient"]["every_patient"],
        "structures to name as a cost": out["organ_costs"]["inferior"],
        "structures inconclusive": out["organ_costs"]["inconclusive"],
        "site adaptation (a gain specific to the own institution)": False,   # O is descriptive only
        "each draw used four labelled scans of ONE institution (design fact)": True,
        "any spinal cord claim": False,
    }
    return out
