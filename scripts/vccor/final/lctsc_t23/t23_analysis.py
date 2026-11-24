"""vccor/final LCTSC-T23 ANALYSIS (pure functions; no file access; unit-tested on synthetic records by test_t23.py).

Input R {pid: {'NATIVE': final_common.Scorer frame record}} (arms scored against the LCTSC reference in LCTSC's 5 ids,
native grid, conventions v1: presence >= 1 mL on both sides, FP structures scored 0, penalised macro). Imports the FROZEN
LCTSC helpers read-only (lctsc_analysis: md, mask_missing, own_vs_other, vol_ratio, FRAME, DRAWS) and the frozen rule
implementations (flaret50/t50_analysis: p1_pooled, p1_single, combination, organ_reading; final_common: two_level,
cluster_boot, ni_class, harm_class, nanmean, organ_dice; fin_look: organ_delta, contrast, macro_values).

REGISTERED (plan_lctsc_t23.json):
  P1_HF  primary (alpha 0.05, not adjusted): the d6ayj P1 rule over the HEART-FREE macro (esophagus, left lung, right lung):
         M = mean over the draws run of mean_i D_{d,i}, D = macro_HF(F1b_S_d) - macro_HF(F0); PASS iff M >= +0.30 AND scan
         t-CI lower > 0 AND crossed two-level bootstrap CI lower > 0 (m2_cluster.boot2 via final_common.two_level, 10,000,
         seed 20261020) AND patients with a positive draw-averaged D >= ceil(2n/3) (16 of 23). Reading GAIN / NO_GAIN.
  S_4S   registered secondary (unadjusted): the same rule over the 4 adapted structures (heart included) = d6ayj's P1 rule
         on the 23 patients. Reading GAIN / NO_GAIN.
REGISTERED DESCRIPTIVE: the per-draw single-draw forms (scan bootstrap, seed 20261022) and their combination for P1_HF and
S_4S; organ costs of the 4 adapted structures (two-level CI; NON-INFERIOR / INFERIOR / INCONCLUSIVE at -1.0 Dice; harm
class); the heart's share of the S_4S gain; every patient (k of n); per-institution means; own institution vs other
institutions (O, d6ayj's definition, for the 4-structure and the heart-free D); spinal cord (F1b = F0 by construction);
absolute macros; per-draw HD95 / NSD / failures / FP events; median volume ratios.
MISSING STRUCTURES (registered rule, as d6ayj): a structure missing from a patient's reference is not evaluable for that
patient (removed from every arm's macro and every per-structure contrast).
Draws NOT RUN are left out; the readings use the draws that remain (two-level over >= 2 draws; single-draw form for one).

Version of 2026-11-17.
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
for _p in (str(HERE.parent.parent / "m2"), str(HERE.parent / "flaret50"), str(HERE.parent), str(HERE.parent / "lctsc"),
           str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import final_common as F  # noqa: E402  (frozen; read-only)
import fin_look as L  # noqa: E402  (frozen; read-only)
import t50_analysis as TA  # noqa: E402  (frozen; read-only)
import lctsc_common as C  # noqa: E402  (frozen; read-only)
import lctsc_analysis as A  # noqa: E402  (frozen; read-only)

FRAME = A.FRAME
ADAPTED = list(C.ADAPTED)                 # 1 esophagus, 2 heart, 3 lung_left, 4 lung_right
HEART = 2
HEART_FREE = [1, 3, 4]
DRAWS = dict(A.DRAWS)                     # {"S1": "F1b_S1", "S2": "F1b_S2", "S3": "F1b_S3"}
KEEP = ("M", "ci", "draw_means", "n_scans", "scans_up", "need_up", "t_ci", "scan_level_mean", "M_ge_0_30", "rule", "pass",
        "bootstrap")


def reading(per):
    """The d6ayj P1 rule on per {draw: {pid: D}} -> (record, full p1_pooled output)."""
    p1 = TA.p1_pooled(per)
    return dict({k: p1[k] for k in KEEP}, reading="GAIN" if p1["pass"] else "NO_GAIN"), p1


def per_draw(per, draws, pooled_pass):
    out, k = {}, 0
    for d in draws:
        out[d] = TA.p1_single(per[d])
        k += int(out[d]["pass"])
    return out, {"combination": TA.combination(pooled_pass, k, len(draws)), "k_of_n_run": f"{k} of {len(draws)}"}


def heart_share(R, draws, ids, s4_M):
    """Registered descriptive: the heart's share of the S_4S gain = mean over draws and patients of the heart's Dice change
    divided by the number of evaluable adapted structures of that patient, over S_4S's M (the macro is a plain mean)."""
    import numpy as np
    vals = []
    for d in draws:
        dl = L.organ_delta(R, FRAME, DRAWS[d], "F0", HEART, ids)
        for c, v in dl.items():
            if v is None or not np.isfinite(v):
                continue
            n_ev = sum(bool(R[c][FRAME]["ref"]["present"][o - 1]) for o in ADAPTED)
            vals.append(float(v) / max(n_ev, 1))
    if not vals or not s4_M:
        return {"share": None, "heart_contribution": None}
    contrib = float(np.mean(vals)) * len(vals) / (len(draws) * len(ids))
    return {"share": contrib / s4_M, "heart_contribution": contrib, "S_4S_M": s4_M,
            "rule": "mean over draws and patients of [heart Dice change / evaluable adapted structures], over S_4S M"}


def analyse(R, ids, have, not_run=None, missing=None):
    R = A.mask_missing(R, missing)
    have = set(have)
    assert "F0" in have, "the readings require F0"
    draws = [d for d in sorted(DRAWS) if DRAWS[d] in have]
    assert draws, "the readings require at least one F1b draw"
    out = {"scheme": "LCTSC-T23 (owner's yes 2026-11-17): d6ayj P1 rule, heart-free primary", "frame": FRAME,
           "n_patients": len(ids), "ids": list(ids), "arms_scored": sorted(have), "arms_not_run": dict(not_run or {}),
           "draws_run": draws, "primary_structures": [C.ORGANS[o - 1] for o in HEART_FREE],
           "adapted": [C.ORGANS[o - 1] for o in ADAPTED], "descriptive": [C.ORGANS[o - 1] for o in C.DESCRIPTIVE]}
    per_hf = {d: A.md(R, DRAWS[d], "F0", ids, organs=HEART_FREE) for d in draws}
    per_4s = {d: A.md(R, DRAWS[d], "F0", ids, organs=ADAPTED) for d in draws}
    out["P1_HF"], p1 = reading(per_hf)
    out["S_4S"], s4 = reading(per_4s)
    out["P1_HF_per_draw_descriptive"], out["P1_HF_combination_descriptive"] = per_draw(per_hf, draws, p1["pass"])
    out["S_4S_per_draw_descriptive"], out["S_4S_combination_descriptive"] = per_draw(per_4s, draws, s4["pass"])
    ev = {}
    for name, pp in (("P1_HF", p1), ("S_4S", s4)):
        sv = pp["scan_values"]
        up = sum(1 for v in sv.values() if v > 0)
        ev[name] = {"k_improved": up, "n": len(sv), "every_patient": up == len(sv) and len(sv) > 0}
    out["every_patient_descriptive"] = ev
    po, cls = {}, {}
    for o in ADAPTED:
        t = F.two_level({d: L.organ_delta(R, FRAME, DRAWS[d], "F0", o, ids) for d in draws}) if len(draws) > 1 else None
        if t is None:
            cb = F.cluster_boot(L.organ_delta(R, FRAME, DRAWS[draws[0]], "F0", o, ids))
            t = {"M": cb["mean"], "ci": cb["ci"], "n_scans": cb["n"]}
        name = C.ORGANS[o - 1]
        po[name] = {"M": t["M"], "ci": t["ci"], "n_scans": t["n_scans"], "ni": F.ni_class(t["ci"]), "harm": F.harm_class(t["ci"]),
                    "median_volume_ratio": {"F0": A.vol_ratio(R, "F0", ids, o),
                                            **{DRAWS[d]: A.vol_ratio(R, DRAWS[d], ids, o) for d in draws}}}
        cls[name] = po[name]["ni"]
    out["organ_costs"] = dict(TA.organ_reading(cls), per_organ=po, margin=-F.NI_MARGIN, status="registered descriptive")
    out["heart_share_descriptive"] = heart_share(R, draws, ids, s4["M"])
    sites = {}
    for name, pp in (("P1_HF", p1), ("S_4S", s4)):
        sv = pp["scan_values"]
        for s in ("S1", "S2", "S3"):
            sid = [c for c in sv if C.site(c) == s]
            if sid:
                cb = F.cluster_boot({c: sv[c] for c in sid})
                sites.setdefault(name, {})[s] = {"n": len(sid), "mean": cb["mean"], "ci": cb["ci"],
                                                 "up": int(sum(sv[c] > 0 for c in sid))}
    out["per_site_descriptive"] = sites
    out["own_site_vs_other_sites_descriptive"] = {"S_4S": A.own_vs_other(per_4s), "P1_HF": A.own_vs_other(per_hf)}
    out["not_evaluable_structures"] = {p: list(v) for p, v in (missing or {}).items() if v}
    cord = C.DESCRIPTIVE[0]
    out["spinal_cord_descriptive"] = {a: F.nanmean([F.organ_dice(R[c][FRAME]["arms"][a], R[c][FRAME]["ref"], cord) for c in ids
                                                    if FRAME in R.get(c, {})]) for a in sorted(have)}
    out["descriptive_per_draw_contrasts"] = {d: L.contrast(R, FRAME, DRAWS[d], "F0", ADAPTED, ids) for d in draws}
    out["absolute_macro"] = {"heart_free": {a: F.nanmean(list(L.macro_values(R, FRAME, a, HEART_FREE, ids).values()))
                                            for a in sorted(have)},
                             "four_structures": {a: F.nanmean(list(L.macro_values(R, FRAME, a, ADAPTED, ids).values()))
                                                 for a in sorted(have)}}
    out["claims_allowed"] = {
        "the frozen LCTSC models improve TS's heart-free macro (esophagus and lungs) on 23 new LCTSC patients (P1_HF)": bool(p1["pass"]),
        "the d6ayj four-structure gain replicates on 23 new patients of the same institutions (S_4S)": bool(s4["pass"]),
        "for each of the three draws (descriptive only)": out["P1_HF_combination_descriptive"]["combination"] == "GAIN_EVERY_DRAW",
        "every patient improved (heart-free macro)": ev["P1_HF"]["every_patient"],
        "structures to name as a cost": out["organ_costs"]["inferior"],
        "structures inconclusive": out["organ_costs"]["inconclusive"],
        "external or independent validation": False,          # same institutions, same models, training patients of LCTSC
        "site adaptation (a gain specific to the own institution)": False,
        "any spinal cord claim": False,
    }
    return out
