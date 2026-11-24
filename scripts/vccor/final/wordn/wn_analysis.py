"""vccor/final WORD-N ANALYSIS (pure functions; no file access; unit-tested on synthetic records by test_wordn.py).

Input R {cid: {'NATIVE': fin_look.score_scan-style frame record}}: every arm scored against the WORD validation reference in
WORD's 15 analysis ids (word_common), native grid, conventions v1 (presence >= 1 mL, FP organs scored 0, penalised macro
over the 12 PRIMARY organs). Volumes: e = 100 (V_pred - V_ref) / V_ref per organ present in the reference (>= 1 mL), V_pred
the arm's total predicted volume of the organ (mL), as flaret50/t50_analysis.vol_err_delta (frozen, imported).
Bootstraps (frozen helpers of flaret50/t50_analysis): crossed two-level (scan x draw) percentile bootstrap, 10,000
resamples, seed 20261020 (m2_cluster.boot2 stream), for readings pooled over draws; scan bootstrap, 10,000, seed 20261022
(final_common.cluster_boot stream), for single-model contrasts. Two-sided bootstrap p = min(1, 2 (min(#r*<=0, #r*>=0) + 1)
/ (B + 1)) from the replicates of the two-level contrast it belongs to. Exception (disclosed): with ONE draw run, P1-fresh's
CI comes from A1's one-draw scan bootstrap (word_a1_score.p1_pooled, seed 20261022) while its Holm p comes from the
two-level helper with one draw (scan resampling, seed 20261020). The intersection-union tests use two-sided p-values and
require both estimates > 0, i.e. each one-sided part is tested at 0.025 (conservative).

PRIMARY P-V (alpha 0.05, alone): per scan the mean over liver, spleen, kidney_left, kidney_right, pancreas (present) of
  |e_F1b_d| - |e_F0|, two-level over the draws that ran -> VOLUME_ERROR_REDUCED iff CI upper < 0; VOLUME_ERROR_INCREASED
  iff CI lower > 0; else NOT_SHOWN. Per-organ values descriptive.
SECONDARY Holm family (m = 4 always; Holm on two-sided p, family-wise alpha 0.05; a member that cannot be evaluated (an arm
NOT RUN, no p) enters Holm with p = 1.0 so that m stays 4 as registered, and its reading is NOT RUN / NOT EVALUABLE; the
evaluable members are listed):
  overshoot_mode 'SPLIT' (option A; the author's choice of 2026-11-13):
    OS-D / OS-E   duodenum / oesophagus OVER-SEGMENTATION: mean signed e_F1b > 0 AND mean (e_F1b - e_F0) > 0, both
                  two-level; intersection-union: p = the larger of the two p-values; OVERSEGMENTED iff Holm-adjusted p < 0.05
                  and both estimates > 0; else NOT_CONFIRMED.
    AC-D / AC-E   (registered DESCRIPTIVE) the volume-accuracy cost |e_F1b| - |e_F0|, two-level: COST_SHOWN iff CI lower > 0,
                  NO_COST_SHOWN otherwise (unadjusted, stated as descriptive).
  overshoot_mode 'AC_COMBINED' (the design's original rule, kept as the switch's alternative): H-D / H-E = mean signed
                  e_F1b > 0 AND mean (|e_F1b| - |e_F0|) > 0 (intersection-union) in the family; the shift is descriptive.
  P1-fresh        WORD A1's P1 rule on F1b_d - F0 (penalised macro, 12 primary organs; word_a1_score.p1_pooled), and its
                  two-level p (direction > 0) enters Holm: GAIN iff P1 PASS and Holm-adjusted p < 0.05.
  S1-fresh        F1b_d - STOCK_d paired by draw, two-level, direction > 0: F1B_BETTER iff Holm-adjusted p < 0.05 and the
                  estimate > 0; STOCK_BETTER if adjusted p < 0.05 and the estimate < 0; else NOT_DETECTED.
TIER 2 (its own family, alpha 0.05, one test): V-P1 = A1's P1 rule on VF5k_d - V0 over the draws that ran ->
  V_GAIN_EVERY_DRAW / V_GAIN_POOLED (P1 PASS, k of 3 draws pass the single-draw form) / V_NO_GAIN (+ 'and lost Dice' iff
  the CI upper < 0) / NOT RUN. Descriptive, no ranking: V0 - F0, VF5k_d - F1b_d, VF5k_lrA_d1 - V0, VF5k_lrA_d1 - VF5k_d1.
DESCRIPTIVE (registered): A1's per-organ non-inferiority of F1b vs F0 (margin -1.0 Dice; word_a1_score.analyse_a1 in
  full: P1, organ reading, P2 per draw) and A1.2's secondary contrasts (word_a1_score.analyse_a1_2, unchanged: S1, S1b,
  S2 MERGE_N, S3/S4 LABELMAP); the acceptance check (k of 6: each F1b_d and STOCK_d is ACCEPTED iff its penalised macro
  gain over F0 has a scan t-CI lower bound > 0 and no primary organ is INFERIOR at -1.0 Dice (scan bootstrap); correct =
  F1b accepted, STOCK rejected; on all included scans and on the first 10 of a seeded order, seed 20261022); volume
  limits of agreement (SD of e), the share of organ volumes within +-10 % (12 primary organs), per-organ signed e of every
  arm; failures fixed / created (Dice-50 rule); HD95 / NSD summaries; the two panels' scans: the median draw-averaged P1
  gain (lower median) and the largest duodenum e_F1b_d1.

Version of 2026-11-13.
"""
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
for _p in (str(HERE.parent), str(HERE), str(HERE.parent / "flaret50"), str(HERE.parent / "word_a1"),
           str(HERE.parent.parent / "m2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import final_common as F  # noqa: E402
import fin_look as L  # noqa: E402
import word_common as W  # noqa: E402
import t50_analysis as TA  # noqa: E402  (frozen FLARE-T50 helpers: bootstrap replicates, p-values, Holm)

FRAME = "NATIVE"
DRAWS = ["d1", "d2", "d3"]
PRIMARY = list(W.PRIMARY)
SOLID = [1, 2, 3, 4, 8]
DUODENUM, ESOPHAGUS = 9, 7
ALPHA = 0.05
P1_MIN_MEAN = 0.30
ACCEPT_FIRST_N = 10
ACCEPT_SEED = F.SEED_CLUSTER


def _f1b(d):
    return f"F1b_{d}"


def organ_name(o):
    return W.ORGANS[o - 1]


def signed_e(R, arm, ids, organ):
    """{cid: e} for the scans whose reference has the organ (>= 1 mL)."""
    out = {}
    for c in ids:
        f = R.get(c, {}).get(FRAME)
        if f is None or not f["ref"]["present"][organ - 1] or arm not in f["arms"]:
            continue
        vr = float(f["ref"]["ml"][organ - 1])
        out[c] = 100.0 * (float(f["arms"][arm]["ml"][organ - 1]) - vr) / vr
    return out


def per_draw(fn, draws):
    return {d: fn(d) for d in draws}


def vol_reading(ci):
    lo, hi = ci
    if lo is None or hi is None:
        return "NOT EVALUABLE"
    if hi < 0:
        return "VOLUME_ERROR_REDUCED"
    if lo > 0:
        return "VOLUME_ERROR_INCREASED"
    return "NOT_SHOWN"


def iut(parts):
    """Intersection-union of one-sided parts {name: contrast}: p = max p; all estimates must be > 0."""
    ps = [v["p_two_sided"] for v in parts.values()]
    if any(p is None for p in ps):
        return None, False
    return max(ps), all(v["estimate"] is not None and v["estimate"] > 0 for v in parts.values())


def organ_volume_block(R, ids, organ, draws):
    """All volume readings of one tubular organ: the signed e of F1b (two-level), the shift e_F1b - e_F0 and the accuracy
    cost |e_F1b| - |e_F0| (two-level), plus the per-draw / F0 descriptives."""
    e0 = signed_e(R, "F0", ids, organ)
    sig = {d: signed_e(R, _f1b(d), ids, organ) for d in draws}
    shift = {d: {c: v - e0[c] for c, v in sig[d].items() if c in e0} for d in draws}
    cost = {d: {c: abs(v) - abs(e0[c]) for c, v in sig[d].items() if c in e0} for d in draws}
    sig = {d: {c: v for c, v in sig[d].items() if c in e0} for d in draws}
    out = {"organ": organ_name(organ),
           "signed_e_F1b": TA.two_level_contrast(sig, ">0"),
           "shift_e_F1b_minus_e_F0": TA.two_level_contrast(shift, ">0"),
           "accuracy_cost_abs": TA.two_level_contrast(cost, ">0"),
           "F0_signed_e": TA.single_contrast(e0, "descriptive") if e0 else None,
           "per_draw_mean_signed_e": {d: (sum(v.values()) / len(v) if v else None) for d, v in sig.items()},
           "n_over_25pct": {"F0": int(sum(v > 25 for v in e0.values())),
                            **{_f1b(d): int(sum(v > 25 for v in sig[d].values())) for d in draws}},
           "n_scans": len(e0)}
    c = out["accuracy_cost_abs"]["ci"]
    out["accuracy_cost_reading_descriptive"] = ("NOT EVALUABLE" if c[0] is None else
                                                ("COST_SHOWN" if c[0] > 0 else "NO_COST_SHOWN"))
    return out


def acceptance(R, ids, arm, base="F0"):
    """ACCEPT iff the penalised macro gain over F0 has a scan t-CI lower bound > 0 and no primary organ is INFERIOR at
    -1.0 Dice (scan bootstrap per organ)."""
    dm = L.macro_delta(R, FRAME, arm, base, PRIMARY, ids)
    t = F.t_ci(list(dm.values()))
    inf = []
    for o in PRIMARY:
        od = L.organ_delta(R, FRAME, arm, base, o, ids)
        if od:
            cb = F.cluster_boot(od)
            if F.ni_class(cb["ci"]) == "INFERIOR":
                inf.append(organ_name(o))
    acc = t["ci"][0] is not None and t["ci"][0] > 0 and not inf
    return {"accept": bool(acc), "mean_gain": t["mean"], "t_ci": t["ci"], "inferior": inf, "n": len(dm)}


def acceptance_check(R, ids, have):
    import numpy as np
    arms_ = [a for a in [f"F1b_{d}" for d in DRAWS] + [f"STOCK_{d}" for d in DRAWS] if a in have]
    order = list(np.random.default_rng(ACCEPT_SEED).permutation(sorted(ids)))
    sets = {"all": list(ids), f"first_{ACCEPT_FIRST_N}_seeded": order[:ACCEPT_FIRST_N]}
    out = {"rule": "ACCEPT iff scan t-CI lower bound of the penalised macro gain over F0 > 0 AND no primary organ INFERIOR "
                   "at -1.0 Dice (scan bootstrap, seed 20261022); correct = F1b accepted, STOCK rejected",
           "order_seed": ACCEPT_SEED, "order": order}
    for k, s in sets.items():
        dec = {a: acceptance(R, s, a) for a in arms_}
        corr = {a: (v["accept"] if a.startswith("F1b") else not v["accept"]) for a, v in dec.items()}
        out[k] = {"decisions": dec, "correct": corr, "k_correct": int(sum(corr.values())), "n_decisions": len(corr)}
    return out


def share_within(R, ids, arm, organs=PRIMARY, tol=10.0):
    v = []
    for o in organs:
        v += [abs(x) <= tol for x in signed_e(R, arm, ids, o).values()]
    return (sum(v) / len(v)) if v else None


def descriptive_volumes(R, ids, have):
    import numpy as np
    out = {}
    for a in sorted(have):
        org = {}
        for o in PRIMARY + [10]:
            e = list(signed_e(R, a, ids, o).values())
            if e:
                org[organ_name(o)] = {"mean_signed": float(np.mean(e)), "sd_loa": float(np.std(e, ddof=1)) if len(e) > 1
                                      else None, "mean_abs": float(np.mean(np.abs(e))), "n": len(e)}
        out[a] = {"share_within_10pct_primary": share_within(R, ids, a), "organs": org}
    return out


def surface_summ(R, ids, have):
    out = {}
    for a in sorted(have):
        out[a] = {m: L.surface_summary(R, FRAME, a, PRIMARY, ids, m) for m in ("hd95", "nsd")}
    return out


def tier2(R, ids, have, not_run):
    import numpy as np
    vd = [d for d in DRAWS if f"VF5k_{d}" in have]
    out = {"family": "tier 2, alpha 0.05, one test (V-P1)", "draws_run": vd,
           "arms_not_run": {a: (not_run or {}).get(a, "not run") for a in ["V0", "VF5k_d1", "VF5k_d2", "VF5k_d3",
                                                                            "VF5k_lrA_d1"] if a not in have}}
    if "V0" not in have or not vd:
        out["V_P1"] = {"reading": "NOT RUN"}
        return out
    per = {d: L.macro_delta(R, FRAME, f"VF5k_{d}", "V0", PRIMARY, ids) for d in vd}
    import word_a1_score as WA
    p1 = WA.p1_pooled(per)
    single = {d: TA.p1_single(per[d]) for d in vd}
    k = int(sum(v["pass"] for v in single.values()))
    if p1["pass"]:
        r = "V_GAIN_EVERY_DRAW" if (k == 3 and len(vd) == 3) else "V_GAIN_POOLED"
    else:
        r = "V_NO_GAIN"
        if p1["ci"][1] is not None and p1["ci"][1] < 0:
            r += " and lost Dice"
    out["V_P1"] = {"reading": r, "p1": p1, "per_draw_single": single, "k_draws_pass": k, "n_draws": len(vd)}
    desc = {"V0_minus_F0": TA.single_contrast(L.macro_delta(R, FRAME, "V0", "F0", PRIMARY, ids), "descriptive")}
    both = [d for d in vd if _f1b(d) in have]
    if both:
        desc["VF5k_minus_F1b"] = TA.two_level_contrast({d: L.macro_delta(R, FRAME, f"VF5k_{d}", _f1b(d), PRIMARY, ids)
                                                        for d in both}, "descriptive")
    if "VF5k_lrA_d1" in have:
        desc["VF5k_lrA_d1_minus_V0"] = TA.single_contrast(L.macro_delta(R, FRAME, "VF5k_lrA_d1", "V0", PRIMARY, ids),
                                                          "descriptive")
        if "VF5k_d1" in have:
            desc["VF5k_lrA_d1_minus_VF5k_d1"] = TA.single_contrast(
                L.macro_delta(R, FRAME, "VF5k_lrA_d1", "VF5k_d1", PRIMARY, ids), "descriptive")
    desc["note"] = "descriptive only; no ranking of TotalSegmentator against VISTA3D"
    out["descriptive"] = desc
    _ = np
    return out


def panels(R, ids, have, p1_scan_values):
    sel = {}
    sv = {c: v for c, v in (p1_scan_values or {}).items() if v is not None and math.isfinite(v)}
    if sv:
        srt = sorted(sv, key=lambda c: (sv[c], c))
        sel["median_gain_scan"] = srt[(len(srt) - 1) // 2]
    if "F1b_d1" in have:
        e = signed_e(R, "F1b_d1", ids, DUODENUM)
        if e:
            sel["largest_duodenum_overshoot_scan"] = max(sorted(e), key=lambda c: e[c])
    return sel


def analyse(R, ids, have, not_run=None, mode="SPLIT", extras=None):
    """R {cid: {'NATIVE': frame}}; ids = included scans; have = arms DONE and scored; mode = the plan's overshoot_mode."""
    assert mode in ("SPLIT", "AC_COMBINED"), mode
    import word_a1_score as WA
    have = set(have)
    assert "F0" in have and "F1b_d1" in have, "WORD-N requires F0 and F1b_d1"
    draws = [d for d in DRAWS if _f1b(d) in have]
    res = {"study": "WORD-N", "n_scans": len(ids), "ids": list(ids), "arms_run": sorted(have),
           "arms_not_run": dict(not_run or {}), "draws_run": draws, "overshoot_mode": mode,
           "organs_primary": [organ_name(o) for o in PRIMARY], "solid_organs": [organ_name(o) for o in SOLID]}
    # primary P-V
    pv = {d: TA.vol_err_delta(R, _f1b(d), "F0", ids, organs=SOLID) for d in draws}
    pvc = TA.two_level_contrast(pv, "<0")
    pvc["reading"] = vol_reading(pvc["ci"])
    pvc["per_organ_descriptive"] = {organ_name(o): TA.two_level_contrast(
        {d: TA.vol_err_delta(R, _f1b(d), "F0", ids, organs=[o]) for d in draws}, "descriptive") for o in SOLID}
    pvc["scans_below_0"] = int(sum(1 for v in _scan_avg(pv).values() if v < 0))
    pvc["n_scans_with_value"] = len(_scan_avg(pv))
    res["P_V"] = pvc
    # tubular organs
    vb = {"duodenum": organ_volume_block(R, ids, DUODENUM, draws), "esophagus": organ_volume_block(R, ids, ESOPHAGUS, draws)}
    res["tubular_volumes"] = vb
    # P1-fresh (A1's rule) + S1-fresh
    p1per = {d: L.macro_delta(R, FRAME, _f1b(d), "F0", PRIMARY, ids) for d in draws}
    p1 = WA.p1_pooled(p1per)
    p1c = TA.two_level_contrast(p1per, ">0")
    sd = [d for d in draws if f"STOCK_{d}" in have]
    s1c = TA.two_level_contrast({d: L.macro_delta(R, FRAME, _f1b(d), f"STOCK_{d}", PRIMARY, ids) for d in sd}, ">0") \
        if sd else None
    fam = {}
    for key, org in (("D", "duodenum"), ("E", "esophagus")):
        b = vb[org]
        if mode == "SPLIT":
            p, pos = iut({"signed": b["signed_e_F1b"], "shift": b["shift_e_F1b_minus_e_F0"]})
            fam[f"OS-{key}"] = {"organ": org, "p_iut": p, "estimates_positive": pos,
                                "parts": ["signed_e_F1b > 0", "e_F1b - e_F0 > 0"]}
        else:
            p, pos = iut({"signed": b["signed_e_F1b"], "cost": b["accuracy_cost_abs"]})
            fam[f"H-{key}"] = {"organ": org, "p_iut": p, "estimates_positive": pos,
                               "parts": ["signed_e_F1b > 0", "|e_F1b| - |e_F0| > 0"]}
    fam["P1-fresh"] = {"p_iut": p1c["p_two_sided"], "estimates_positive": bool(p1c["estimate"] and p1c["estimate"] > 0)}
    fam["S1-fresh"] = {"p_iut": s1c["p_two_sided"] if s1c else None,
                       "estimates_positive": bool(s1c and s1c["estimate"] is not None and s1c["estimate"] > 0)}
    evaluable = [k for k, v in fam.items() if v["p_iut"] is not None]
    p_in = {k: (1.0 if v["p_iut"] is None else v["p_iut"]) for k, v in fam.items()}
    adj = TA.holm(p_in)
    for k, v in fam.items():
        v["p_holm"] = adj[k]
        v["evaluable"] = k in evaluable
        sig = v["evaluable"] and adj[k] is not None and adj[k] < ALPHA
        if k.startswith(("OS-", "H-")):
            v["reading"] = "NOT EVALUABLE" if not v["evaluable"] else ("OVERSEGMENTED" if (sig and v["estimates_positive"])
                                                                       else "NOT_CONFIRMED")
        elif k == "P1-fresh":
            v["reading"] = ("GAIN" if (sig and p1["pass"]) else "NO_GAIN") if v["evaluable"] else "NOT EVALUABLE"
            v["p1_rule_pass"] = p1["pass"]
        else:
            if s1c is None:
                v["reading"] = "NOT RUN"
            elif sig and s1c["estimate"] > 0:
                v["reading"] = "F1B_BETTER"
            elif sig and s1c["estimate"] < 0:
                v["reading"] = "STOCK_BETTER"
            else:
                v["reading"] = "NOT_DETECTED"
    res["holm_family"] = {"members": list(fam), "alpha": ALPHA, "results": fam, "m": len(fam),
                          "m_used_in_holm": sum(1 for v in p_in.values() if v is not None),
                          "evaluable_members": evaluable,
                          "missing_rule": "a member without a p-value enters Holm with p = 1.0 (m stays 4)"}
    res["P1_fresh"] = {"p1": p1, "two_level_contrast": p1c,
                       "per_draw_single": {d: TA.p1_single(p1per[d]) for d in draws}}
    res["S1_fresh"] = s1c
    # replication of A1 / A1.2's full readings on the fresh scans (descriptive, frozen code)
    ts_have = {a for a in have if a in ("F0", "F1b_d1", "F1b_d2", "F1b_d3")}
    try:
        res["A1_readings_fresh_descriptive"] = WA.analyse_a1(R, ids, ts_have, None, not_run)
    except Exception as e:                           # noqa: BLE001  (descriptive: never blocks the registered readings)
        res["A1_readings_fresh_descriptive"] = {"error": f"{type(e).__name__}: {e}"}
    try:
        res["A1_2_readings_fresh_descriptive"] = WA.analyse_a1_2(R, ids, have, extras or {}, not_run, p1_pass=p1["pass"])
    except Exception as e:                           # noqa: BLE001
        res["A1_2_readings_fresh_descriptive"] = {"error": f"{type(e).__name__}: {e}"}
    try:
        res["acceptance_check_descriptive"] = acceptance_check(R, ids, have)
    except Exception as e:                           # noqa: BLE001
        res["acceptance_check_descriptive"] = {"error": f"{type(e).__name__}: {e}"}
    res["volumes_descriptive"] = descriptive_volumes(R, ids, have)
    try:
        res["surface_descriptive"] = surface_summ(R, ids, have)
        res["failures_descriptive"] = {a: L.failures(R, FRAME, a, "F0", PRIMARY, ids) for a in sorted(have) if a != "F0"}
    except Exception as e:                           # noqa: BLE001
        res["surface_descriptive"] = {"error": f"{type(e).__name__}: {e}"}
    res["tier2"] = tier2(R, ids, have, not_run)
    res["panels"] = panels(R, ids, have, p1.get("scan_values"))
    return res


def _scan_avg(per):
    acc = {}
    for d, v in per.items():
        for c, x in v.items():
            acc.setdefault(c, []).append(x)
    return {c: sum(x) / len(x) for c, x in acc.items()}


def summary(res):
    f = res["holm_family"]["results"]
    t2 = res.get("tier2", {}).get("V_P1", {}).get("reading")
    pv = res["P_V"]
    return (f"P-V {pv['reading']} M {pv['estimate']:+.2f} pp; " +
            "; ".join(f"{k} {v['reading']}" for k, v in f.items()) + f"; tier 2 V-P1 {t2}")
