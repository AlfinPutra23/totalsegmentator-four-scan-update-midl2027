"""vccor/final plan C, amendments word A1, A1.1 and A1.2: the TS-only WORD LOOK (CPU, <= 6 workers, nice 15; ONE look).

Arms (amend_word_A1.json remaining_arms): F0 = TotalSegmentator zero-shot; F1b_d1 / F1b_d2 / F1b_d3 = the F1b fine-tune on
the three registered draws. REQUIRED: F0 and F1b_d1. A1.2 (amend_word_A1_2.json) adds, per draw, the secondary arms STOCK_dX
(nnU-Net's stock pretrained fine-tune; maps in WORD's analysis ids), MERGE_N_dX (class-wise merge of F0 and F1b_dX, non-target
wins; TS ids) and LABELMAP_dX (F0 relabelled by the draw's class-merge rules; analysis ids); all arms are scored in the same
single look, each map with its own label map (word_a1_common.ARM_LUT). An arm that is not DONE must be declared NOT RUN in
final_prereg.log before the look ('NOTRUN word_A1 <arm> ...', written by word_a1_run.py); otherwise this scorer refuses (no
interim look). P1, P2 and the outcome label are computed from F0 and F1b only, exactly as under A1 / A1.1.

Gates, in this order, before any label byte is read:
  1. no second look: C_word/look/GT_OPENED and C_word/look/results.json must not exist;
  2. the OSF gate (word_a1_common.osf_gate): C_word/A1_osf_registration.txt + its OSF-REGISTRATION line (v5 deposit);
  3. integrity: plan C, every code file and frozen import, A1, A1.1, A1.2 and the v5 code amendment hash to their logged
     lines; conventions v1 pinned; the labelsTs zip's size (os.stat only, its content is not read) equals the plan's record;
  4. arms: F0 and F1b_d1 DONE; every other arm DONE before the cutoff or declared NOT RUN; no EXPERIMENT NOT RUN line; an
     A1.2 CPU arm's DONE record names the current DONE records of its inputs (MERGE_N_dX: F0 and F1b_dX; LABELMAP_dX: F0 and
     its rules file);
  5. the registered duplicate rule on the IMAGES (word_score.duplicates, unchanged): test pair 0017 ~ 0131 with Pearson
     r >= 0.999 on a 4-voxel stride grid -> 0131 dropped, otherwise both stay;
  6. every included arm's maps checked against their DONE records and hashed into C_word/look/maps.json, with every arm's
     label map, the sha256 of every DONE record and of the LABELMAP rules files.
Then the marker C_word/look/GT_OPENED is created O_EXCL (the path word_score.py uses, so neither scorer can look twice),
and a 'GT_OPENED sha256 ...' line is appended to final_prereg.log BEFORE the labelsTs zip is opened. Only then is the
zip hashed (sha256 pinned in plan C; a mismatch logs GT_ABORTED and stops for good) and read, in memory, by
word_common.reference() through word_score.job (unchanged), one scan per worker process.

After the marker (A1.1):
  * every scan's result is kept as it arrives: word_score.job writes look/pervol/<cid>.json, and this scorer then writes
    look/pervol_done/<cid>.json (the exact record, NaN kept, bound to the marker, maps.json and the scan's map sha256);
  * any exception, SIGTERM or SIGHUP logs 'GT_CRASHED <type> <message>' to final_prereg.log and exits non-zero;
  * a worker that dies (OOM kill) breaks the pool (BrokenProcessPool) and a scan that gives no result within
    --result-timeout seconds (default 7200) fails the look: neither can hang it;
  * --continue (A1.1 continuation rule) resumes the SAME single look: allowed only if GT_OPENED exists, results.json does
    not, final_prereg.log has no RESULTS line for it and no GT_ABORTED line, there is exactly one GT_OPENED line for the
    marker and the marker and maps.json still match it, the OSF gate holds, and every frozen code file still matches its
    CODE line (a continuation with changed code is not allowed: a code change after the marker is a disclosed deviation
    outside this rule). It skips the marker, logs 'GT_CONTINUED ...', reuses every look/pervol/<cid>.json whose
    pervol_done record passes its integrity check and re-scores the other scans. The ids, arms and duplicate decision come
    from maps.json (nothing is re-decided). Both paths analyse the same exact per-scan records.

Analysis (analyse_a1, a pure function; unit-tested on synthetic records):
  P1 gain      M = mean over the draws that ran of mean_i D_{d,i} (the mean of the draw means, A1's estimand), D = macro(F1b_d)
               - macro(F0) (penalised macro Dice, 12 primary organs); PASS iff M >= +0.30 AND scan t-CI lower > 0 AND the
               bootstrap CI lower > 0 AND scans with a positive draw-averaged D >= ceil(2/3 n). Two or three draws: two-level
               (scan x draw) CI, final_common.two_level, seed 20261020. One draw: A1's scan bootstrap, final_common.cluster_boot,
               seed 20261022 (A1.1).
  P1 organs    per primary organ: two-level CI of the Dice change; NON-INFERIOR / INFERIOR / INCONCLUSIVE at -1.0 Dice;
               reading ALL_NI / SOME_INFERIOR / INCONCLUSIVE / NO_ORGAN_COST_AMONG_EVALUABLE (A1.1).
  P2           per draw: the same rule with the scan bootstrap (cluster_boot) -> k of n_run; pairwise draw differences with
               scan-bootstrap CIs -> DRAW-DEPENDENT iff any CI excludes 0; range and SD (ddof 1) of the draw gains.
  outcome      GAIN_EVERY_DRAW (k = 3 of 3) / GAIN_EVERY_RUN_DRAW (A1.1: fewer than 3 draws ran, all gain) /
               GAIN_NOT_EVERY_DRAW / GAIN_POOLED_ONLY / NO_GAIN_SOME_DRAWS / NO_GAIN, crossed with the organ reading and the
               draw-dependence reading; claim flags per A1's claim_rules and A1.1.
  secondary    S for TS (word_score.systematic_share / boot_ratio), colon_rectum, per-draw per-organ results.
  descriptive  femoral heads zero-shot, slice-spacing strata, plan-E decomposition, NSD / HD95, failures, FP organs.
  A1.2         (analyse_a1_2, a pure function; secondary, unadjusted; P1 / P2 / the outcome label unchanged) on the 12 primary
               organs' penalised macro, per draw (scan bootstrap, final_common.cluster_boot, seed 20261022) and pooled over
               the draws in which both arms ran (two-level, final_common.two_level, seed 20261020; one draw: the scan
               bootstrap):
               S1  F1b - STOCK: F1B_BETTER / STOCK_BETTER / NO_DIFFERENCE_DETECTED (CI sign);
               S1b STOCK - F0 with P1's rule form: STOCK_GAINS / STOCK_NO_GAIN; absolute Dice (VISTA3D link, descriptive);
               S2  MERGE_N - F1b: macro and per-organ non-inferiority at -1.0 Dice + harm class; colon_rectum; femoral
                   heads; the merged (non-target) classes: label-free rule B of conventions v1 on F1b vs F0 (descriptive);
               S3  LABELMAP - F0: RELABEL_GAINS / RELABEL_LOSES / NO_DIFFERENCE_DETECTED; S_LM = share of the zero-shot
                   error removed by relabelling alone (word_score.boot_ratio); the rules of each draw;
               S4  F1b - LABELMAP: FT_EXCEEDS_RELABEL / RELABEL_EXCEEDS_FT / NOT_DETECTED; Q = the share of F1b's gain that
                   relabelling recovers (two-level ratio bootstrap, seed 20261020; only under P1 PASS).
               A secondary-analysis error never blocks the primary results (A1.2): if analyse_a1_2 raises, the A1.2
               block of results.json is {"error": "<type>: <message>"}, a 'SECONDARY_ERROR word_A1 ...' line is logged,
               and results.json (P1, P2, the outcome label) and the RESULTS line are written as usual.

usage:  CUDA_VISIBLE_DEVICES="" nice -n 15 .venv/bin/python -B scripts/vccor/final/word_a1/word_a1_score.py --workers 6
        .venv/bin/python -B scripts/vccor/final/word_a1/word_a1_score.py --check
            (gates 1-4, without the zip size, which only the look stats; writes nothing)
        .venv/bin/python -B scripts/vccor/final/word_a1/word_a1_score.py --continue --workers 6
        .venv/bin/python -B scripts/vccor/final/word_a1/word_a1_score.py --continue --check
            (the continuation gates only; writes nothing)
NEVER run scripts/vccor/final/word_score.py (the old Plan C scorer): it hashes the label archive before any marker.

Version of 2026-11-05 (A1, A1.1) and 2026-11-06 (A1.2).
"""
import argparse
import contextlib
import hashlib
import json
import os
import re
import signal
import sys
import time
from datetime import datetime
from itertools import combinations
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import word_a1_common as A
import final_common as F
import fin_look as L
import word_common as W
import word_score as WS

DRAWS = A.DRAWS
FRAME = "NATIVE"
P1_MIN_MEAN = 0.30
RESULT_TIMEOUT_S = 7200
SETTLED_BY_A1_1 = [
    "GAIN_EVERY_DRAW needs all three draws run; fewer draws run and all gain -> GAIN_EVERY_RUN_DRAW (k of n_run)",
    "organ reading with NOT EVALUABLE organs and no INFERIOR / INCONCLUSIVE organ -> NO_ORGAN_COST_AMONG_EVALUABLE",
    "SD of the draw gains: ddof = 1",
    "one-draw P1 bootstrap: final_common.cluster_boot, seed 20261022 (A1's scan bootstrap)",
    "P1's +0.30 bar applies to M = the mean of the draw means",
]


class Terminated(BaseException):
    """SIGTERM / SIGHUP during the look (so the GT_CRASHED line is written)."""


def _on_signal(signum, frame):
    raise Terminated(f"signal {signum} ({signal.Signals(signum).name})")


def _worker_init():
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGHUP, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_IGN)


# ---------------------------------------------------------------------------------------------------- analysis
def arm_of(d):
    return f"F1b_{d}"


def organ_name(o):
    return W.ORGANS[o - 1]


def excludes_zero(ci):
    lo, hi = ci
    return lo is not None and hi is not None and (lo > 0 or hi < 0)


def organ_reading(classes):
    """A1 P1_organ_costs.reading from {organ: NI class}, with A1.1's NO_ORGAN_COST_AMONG_EVALUABLE (some organs NOT
    EVALUABLE, none INFERIOR or INCONCLUSIVE; the non-evaluable organs are listed)."""
    inf = [o for o, c in classes.items() if c == "INFERIOR"]
    inc = [o for o, c in classes.items() if c == "INCONCLUSIVE"]
    ne = [o for o, c in classes.items() if c not in ("NON-INFERIOR", "INFERIOR", "INCONCLUSIVE")]
    if inf:
        r = "SOME_INFERIOR"
    elif inc:
        r = "INCONCLUSIVE"
    elif not ne:
        r = "ALL_NI"
    else:
        r = "NO_ORGAN_COST_AMONG_EVALUABLE"
    return {"reading": r, "inferior": inf, "inconclusive": inc, "not_evaluable": ne}


def combination(p1_pass, k, n_run=3):
    """A1 outcome_combinations with A1.1: GAIN_EVERY_DRAW needs k = 3 of 3 draws run; fewer than three draws run and all
    of them gain -> GAIN_EVERY_RUN_DRAW."""
    if p1_pass:
        if k == 3 and n_run == 3:
            return "GAIN_EVERY_DRAW"
        if n_run < 3 and k == n_run:
            return "GAIN_EVERY_RUN_DRAW"
        return "GAIN_NOT_EVERY_DRAW" if k >= 1 else "GAIN_POOLED_ONLY"
    return "NO_GAIN_SOME_DRAWS" if k >= 1 else "NO_GAIN"


def p1_pooled(per):
    """A1 P1 on {draw: {cid: D}}: M = the mean of the draw means; the +0.30 bar on M; scan t-CI and the ceil(2/3 n) count
    on the per-scan values averaged over the draws in which the scan has a value; bootstrap CI lower > 0 (two-level with
    two or three draws, A1's scan bootstrap with one draw)."""
    import numpy as np
    draws = sorted(per)
    assert draws, "P1 needs at least one draw"
    if len(draws) > 1:
        tl = F.two_level(per)
        M, ci, sv, dm = tl["M"], tl["ci"], tl["scan_values"], tl.get("draw_means")
        boot = {"method": tl.get("method"), "seed": F.SEED_TWO_LEVEL, "nboot": F.NBOOT}
    else:
        cb = F.cluster_boot(per[draws[0]])
        M, ci = cb["mean"], cb["ci"]
        sv = {c: float(v) for c, v in per[draws[0]].items() if v is not None and np.isfinite(v)}
        dm = {draws[0]: cb["mean"]}
        boot = {"method": "scan bootstrap (final_common.cluster_boot, one cluster per scan; one draw ran)",
                "seed": cb.get("seed"), "nboot": cb.get("nboot")}
    r = F.p1_rule(list(sv.values()), min_mean=float("-inf"), boot=ci)
    m_ok = M is not None and M >= P1_MIN_MEAN
    return {"M": M, "ci": ci, "draw_means": dm, "scan_values": sv, "n_scans": len(sv), "scans_up": r["up"],
            "need_up": r["need_up"], "t_ci": r["t_ci"], "scan_level_mean": r["mean"], "M_ge_0_30": bool(m_ok),
            "pass": bool(r["pass"] and m_ok),
            "rule": "M (mean of the draw means) >= +0.30 AND scan t-CI lower > 0 AND bootstrap-CI lower > 0 AND up >= "
                    "ceil(0.667 n)", "bootstrap": boot}


def analyse_a1(R, ids, have, dec=None, not_run=None):
    """R {cid: frames (fin_look.score_scan records)}; ids = the test scans after the duplicate rule; have = arms run;
    dec {cid: {arm: dec_common records}} (optional); not_run {arm: reason}. Pure function."""
    import numpy as np
    have = set(have)
    assert "F0" in have and "F1b_d1" in have, "A1 requires F0 and F1b_d1"
    draws = [d for d in DRAWS if arm_of(d) in have]
    out = {"amendment": "word A1 (TS only), with A1.1", "arms_run": sorted(have), "draws_run": draws,
           "arms_not_run": {a: (not_run or {}).get(a, "not run") for a in A.ARMS if a not in have},
           "n_scans": len(ids), "ids": list(ids), "organs_primary": [organ_name(o) for o in W.PRIMARY], "frame": FRAME,
           "undecided_by_registration": [], "settled_by_A1_1": list(SETTLED_BY_A1_1)}
    per = {d: L.macro_delta(R, FRAME, arm_of(d), "F0", W.PRIMARY, ids) for d in draws}

    p1 = p1_pooled(per)
    out["P1_gain"] = {"M": p1["M"], "two_level_ci": p1["ci"] if len(draws) > 1 else None, "ci": p1["ci"],
                      "draw_means": p1["draw_means"], "n_scans": p1["n_scans"], "scans_up": p1["scans_up"],
                      "need_up": p1["need_up"], "t_ci": p1["t_ci"], "scan_level_mean": p1["scan_level_mean"],
                      "M_ge_0_30": p1["M_ge_0_30"], "rule": p1["rule"], "pass": p1["pass"],
                      "reading": "GAIN" if p1["pass"] else "NO GAIN", "bootstrap": p1["bootstrap"]["method"],
                      "seed": p1["bootstrap"]["seed"], "nboot": p1["bootstrap"]["nboot"]}

    po, cls = {}, {}
    for o in W.PRIMARY:
        t = F.two_level({d: L.organ_delta(R, FRAME, arm_of(d), "F0", o, ids) for d in draws})
        po[organ_name(o)] = {"M": t["M"], "ci": t["ci"], "n_scans": t["n_scans"], "ni": F.ni_class(t["ci"]),
                             "harm": F.harm_class(t["ci"])}
        cls[organ_name(o)] = po[organ_name(o)]["ni"]
    rd = organ_reading(cls)
    out["P1_organ_costs"] = dict(rd, per_organ=po, margin=-F.NI_MARGIN)

    pd_, k = {}, 0
    for d in draws:
        cb = F.cluster_boot(per[d])
        r = F.p1_rule(list(per[d].values()), boot=cb["ci"])
        pd_[d] = {"M": cb["mean"], "scan_boot_ci": cb["ci"], "t_ci": r["t_ci"], "scans_up": r["up"],
                  "need_up": r["need_up"], "n": r["n"], "pass": r["pass"], "seed": cb.get("seed"), "nboot": cb.get("nboot")}
        k += int(r["pass"])
    pairs = {}
    for d, e in combinations(draws, 2):
        v = {c: per[d][c] - per[e][c] for c in per[d] if c in per[e]}
        cb = F.cluster_boot(v)
        pairs[f"{d} - {e}"] = {"mean": cb["mean"], "ci": cb["ci"], "n": cb["n"], "excludes_0": excludes_zero(cb["ci"]),
                               "seed": cb.get("seed"), "nboot": cb.get("nboot")}
    Ms = [pd_[d]["M"] for d in draws if pd_[d]["M"] is not None]
    if len(draws) < 2:
        dep = "NOT EVALUABLE (one draw)"
    elif any(p["excludes_0"] for p in pairs.values()):
        dep = "DRAW-DEPENDENT"
    else:
        dep = "NO DRAW DEPENDENCE DETECTED"
    out["P2_draw_dependence"] = {"per_draw": pd_, "k": k, "k_of": len(draws), "n_run": len(draws), "pairwise": pairs,
                                 "reading": dep, "range": (max(Ms) - min(Ms)) if len(Ms) >= 2 else None,
                                 "sd_ddof1": float(np.std(Ms, ddof=1)) if len(Ms) >= 2 else None,
                                 "multiplicity": "three unadjusted pairwise CIs ('at least one pair, unadjusted')"}

    comb = combination(p1["pass"], k, len(draws))
    out["outcome"] = {"combination": comb, "organ_costs": rd["reading"], "draw_dependence": dep,
                      "k_of_n_run": f"{k} of {len(draws)}"}
    out["claims_allowed"] = {
        "four labelled WORD scans improve TS over zero-shot": bool(p1["pass"]),
        "for each of the three draws": comb == "GAIN_EVERY_DRAW",
        "for each draw that ran (report k of n_run)": comb in ("GAIN_EVERY_DRAW", "GAIN_EVERY_RUN_DRAW"),
        "no primary organ loses more than 1 Dice point": rd["reading"] == "ALL_NI",
        "no evaluable primary organ loses more than 1 Dice point (list the others)":
            rd["reading"] in ("ALL_NI", "NO_ORGAN_COST_AMONG_EVALUABLE"),
        "organs to name as a cost": rd["inferior"],
        "organs to list as not evaluable": rd["not_evaluable"],
        "the gain depends on which four scans are labelled": dep == "DRAW-DEPENDENT",
        "ranking of TS, VISTA3D and MRSegmentator on WORD": False,
    }

    S = WS.systematic_share(R, ids, "TS", draws, W.PRIMARY)
    out["secondary_S_TS"] = S
    cr = F.two_level({d: L.organ_delta(R, FRAME, arm_of(d), "F0", 10, ids) for d in draws})
    out["secondary_colon_rectum"] = {k_: cr.get(k_) for k_ in ("M", "ci", "n_scans", "draw_means")}
    pdo = {}
    for d in draws:
        pdo[d] = {}
        for o in W.PRIMARY + W.SECONDARY:
            cb = F.cluster_boot(L.organ_delta(R, FRAME, arm_of(d), "F0", o, ids))
            pdo[d][organ_name(o)] = {"mean": cb["mean"], "ci": cb["ci"], "n": cb["n"], "ni": F.ni_class(cb["ci"]),
                                     "harm": F.harm_class(cb["ci"])}
    out["secondary_per_draw_per_organ_descriptive"] = pdo

    out["descriptive_femur_heads_zero_shot"] = {organ_name(o): F.nanmean(list(WS.organ_vals(R, "F0", o, ids).values()))
                                                for o in W.DESCRIPTIVE}
    sz = {c: round(max(float(x) for x in R[c][FRAME]["spacing"]), 1) for c in ids
          if R.get(c, {}).get(FRAME, {}).get("spacing")}
    strata = {}
    for z in sorted(set(sz.values())):
        ii = [c for c in ids if sz.get(c) == z]
        strata[f"{z} mm"] = {"n": len(ii), "F0 macro": F.nanmean(list(WS.score_of(R, "F0", W.PRIMARY, ii).values())),
                             "gain (mean over draws)": F.nanmean([F.nanmean(list(L.macro_delta(
                                 R, FRAME, arm_of(d), "F0", W.PRIMARY, ii).values())) for d in draws])}
    out["descriptive_slice_spacing_strata"] = strata
    out["descriptive_contrasts"] = {d: L.contrast(R, FRAME, arm_of(d), "F0", W.PRIMARY, ids) for d in draws}
    if dec:
        import dec_common as E
        dd = {}
        for d in draws:
            recs = {"F0": {c: dec[c]["F0"] for c in dec if "F0" in dec[c]},
                    "FT": {c: dec[c][arm_of(d)] for c in dec if arm_of(d) in dec[c]}}
            cc = sorted(set(recs["F0"]) & set(recs["FT"]))
            means, _ = E.organ_means({k_: {c: v[c] for c in cc} for k_, v in recs.items()}, W.PRIMARY)
            dd[d] = means
        out["descriptive_decomposition_plan_E"] = dd
    return out


# ---------------------------------------------------------------------------------------------------- A1.2 analysis
A12_RULE = "unadjusted secondary contrasts of amendment word A1.2; none changes P1, P2 or the outcome label"
LN_110 = 0.09531017980432493


def per_draw_block(per):
    """{draw: scan bootstrap (cluster_boot, seed 20261022) + t-CI} of per {draw: {cid: value}}."""
    out = {}
    for d in sorted(per):
        cb = F.cluster_boot(per[d])
        t = F.t_ci(list(per[d].values()))
        out[d] = {"mean": cb["mean"], "ci": cb["ci"], "t_ci": t["ci"], "n": cb["n"], "up": cb["up"], "down": cb["down"],
                  "seed": cb.get("seed"), "nboot": cb.get("nboot")}
    return out


def pooled_ci(per):
    """M = mean of the draw means with A1's CI: two-level (final_common.two_level, seed 20261020) for >= 2 draws, the scan
    bootstrap (cluster_boot, seed 20261022) for one draw (A1.1's one-draw rule); NOT RUN without a draw."""
    draws = sorted(d for d in per if per[d])
    if not draws:
        return {"M": None, "ci": [None, None], "draws": [], "draw_means": {}, "n_scans": 0, "method": "NOT RUN"}
    if len(draws) > 1:
        tl = F.two_level({d: per[d] for d in draws})
        return {"M": tl["M"], "ci": tl["ci"], "draws": draws, "draw_means": tl["draw_means"], "n_scans": tl["n_scans"],
                "up": tl.get("up"), "down": tl.get("down"), "method": "crossed two-level bootstrap (final_common.two_level)",
                "seed": F.SEED_TWO_LEVEL, "nboot": F.NBOOT}
    cb = F.cluster_boot(per[draws[0]])
    return {"M": cb["mean"], "ci": cb["ci"], "draws": draws, "draw_means": {draws[0]: cb["mean"]}, "n_scans": cb["n"],
            "up": cb["up"], "down": cb["down"], "method": "scan bootstrap (final_common.cluster_boot; one draw)",
            "seed": cb.get("seed"), "nboot": cb.get("nboot")}


def sign_reading(ci, pos, neg, none):
    lo, hi = ci
    if lo is None or hi is None:
        return "NOT EVALUABLE"
    if lo > 0:
        return pos
    if hi < 0:
        return neg
    return none


def contrast_a12(R, ids, a_of, b_of, draws, organs=None):
    """{per_draw, pooled} of the penalised macro difference a_of(d) - b_of(d) over `organs` (default: the 12 primary)."""
    organs = organs or W.PRIMARY
    per = {d: L.macro_delta(R, FRAME, a_of(d), b_of(d), organs, ids) for d in draws}
    return per, {"per_draw": per_draw_block(per), "pooled": pooled_ci(per), "draws": list(draws)}


def organ_block(R, ids, a_of, b_of, draws, organs):
    """Per organ: M and CI of the Dice difference (pooled_ci), NI class at -1.0 Dice and harm class."""
    out = {}
    for o in organs:
        pc = pooled_ci({d: L.organ_delta(R, FRAME, a_of(d), b_of(d), o, ids) for d in draws})
        out[organ_name(o)] = {"M": pc["M"], "ci": pc["ci"], "n_scans": pc["n_scans"], "ni": F.ni_class(pc["ci"]),
                              "harm": F.harm_class(pc["ci"]), "method": pc["method"]}
    return out


def ratio_boot(num_per, den_per, nboot=F.NBOOT, seed=F.SEED_TWO_LEVEL):
    """Q = (mean over draws of mean_i num) / (mean over draws of mean_i den) on the scans every draw has in both; crossed
    two-level percentile bootstrap with word_score.boot_ratio's scheme (scans and draws resampled independently; one draw:
    scans only); replicates with a denominator <= 0 are dropped (counted)."""
    import numpy as np
    draws = sorted(d for d in num_per if d in den_per)
    if not draws:
        return None, [None, None], 0
    cids = sorted(set.intersection(*[set(num_per[d]) & set(den_per[d]) for d in draws]))
    if not cids:
        return None, [None, None], 0
    Nm = np.array([[num_per[d][c] for c in cids] for d in draws])
    Dn = np.array([[den_per[d][c] for c in cids] for d in draws])
    den0 = float(Dn.mean(1).mean())
    q = float(Nm.mean(1).mean() / den0) if den0 > 0 else None
    rng = np.random.default_rng(seed)
    n, k = len(cids), len(draws)
    reps = np.empty(nboot)
    for b in range(nboot):
        w = rng.multinomial(n, np.full(n, 1.0 / n)).astype(float)
        u = rng.multinomial(k, np.full(k, 1.0 / k)).astype(float) if k > 1 else np.ones(1)
        nb = float(u @ ((Nm @ w) / w.sum()) / u.sum())
        db = float(u @ ((Dn @ w) / w.sum()) / u.sum())
        reps[b] = nb / db if db > 0 else np.nan
    ok = reps[np.isfinite(reps)]
    ci = [float(np.quantile(ok, 0.025)), float(np.quantile(ok, 0.975))] if len(ok) else [None, None]
    return q, ci, int(nboot - len(ok))


def ts_class_name(c):
    from totalsegmentator.map_to_binary import class_map
    return class_map["total"].get(int(c), f"ts_{int(c)}")


def merged_classes_rule_b(merge_done, ids):
    """Label-free rule B of conventions v1 item6.acceptance_check.non_target_classes on F1b vs F0 (the classes MERGE_N
    restores): every TS class outside T with F0 >= 1 mL on >= 3 of the look's scans; PASS iff median Dice(F1b_c, F0_c) >= 90
    AND |median ln(V_F1b / V_F0)| <= ln 1.10, else FLAG. Descriptive (no WORD reference exists for these classes)."""
    import numpy as np
    per = merge_done.get("per_scan_merge") or {}
    classes = sorted({int(c) for cid in ids for c in (per.get(cid) or {}).get("nontarget_f0_f1b_both", {})})
    out = {}
    for c in classes:
        dice, lr = [], []
        for cid in ids:
            r = per.get(cid) or {}
            v = r.get("nontarget_f0_f1b_both", {}).get(str(c))
            if not v:
                continue
            n0, n1, both = v
            if n0 * float(r.get("vox_ml", 0.0)) < F.FLOOR_ML:
                continue
            dice.append(200.0 * both / (n0 + n1))
            lr.append(float(np.log(n1 / n0)) if n1 > 0 else float("-inf"))
        if len(dice) < 3:
            continue
        md, ml = float(np.median(dice)), float(np.median(lr))
        out[ts_class_name(c)] = {"ts_id": c, "n_scans": len(dice), "median_dice_f1b_vs_f0": md,
                                 "median_ln_volume_ratio": ml,
                                 "rule_B": "PASS" if (md >= 90.0 and abs(ml) <= LN_110) else "FLAG"}
    changed = [int((per.get(cid) or {}).get("n_changed_vs_f1b", 0)) for cid in ids if cid in per]
    pairs = {}
    for cid in ids:
        for k, v in ((per.get(cid) or {}).get("changed_vs_f1b") or {}).items():
            pairs[k] = pairs.get(k, 0) + int(v)
    named = {}
    for k, v in sorted(pairs.items(), key=lambda x: -x[1])[:20]:
        a, b = (int(x) for x in k.split("->"))
        named[f"{ts_class_name(a)} -> {ts_class_name(b) if b else 'background'}"] = v
    return {"rule": "conventions v1 item6.acceptance_check.non_target_classes (rule B), label-free, descriptive",
            "classes": out, "flagged": sorted(k for k, v in out.items() if v["rule_B"] == "FLAG"),
            "voxels_merge_n_differs_from_f1b": {"median": float(np.median(changed)) if changed else None,
                                                "max": int(max(changed)) if changed else None},
            "largest_changes_f1b_to_merge_n": named}


def p1_form(per):
    """A1's P1 rule form on {draw: {cid: diff}} (p1_pooled: the +0.30 bar on M, t-CI, bootstrap, ceil(2/3 n)) + per draw."""
    if not any(per.values()):
        return {"reading": "NOT RUN"}
    p = p1_pooled({d: v for d, v in per.items() if v})
    pdr = {}
    for d in sorted(per):
        if not per[d]:
            continue
        cb = F.cluster_boot(per[d])
        r = F.p1_rule(list(per[d].values()), boot=cb["ci"])
        pdr[d] = {"M": cb["mean"], "ci": cb["ci"], "t_ci": r["t_ci"], "up": r["up"], "need_up": r["need_up"],
                  "pass": r["pass"]}
    return {"M": p["M"], "ci": p["ci"], "t_ci": p["t_ci"], "scans_up": p["scans_up"], "need_up": p["need_up"],
            "M_ge_0_30": p["M_ge_0_30"], "pass": p["pass"], "bootstrap": p["bootstrap"], "per_draw": pdr,
            "k": sum(int(v["pass"]) for v in pdr.values()), "n_draws": len(pdr)}


def absolute_dice(R, ids, arms):
    """Mean over scans (and draws) of the per-scan macro Dice: penalised and reference-present over the 12 primary organs, and
    reference-present over all 15 analysis organs (descriptive; VISTA3D's Table 3 uses another organ set, metric and K)."""
    out = {}
    for name, members in arms.items():
        if not members:
            continue
        row = {}
        for key, organs, pen in (("macro_penalised_12", W.PRIMARY, True), ("macro_reference_present_12", W.PRIMARY, False),
                                 ("macro_reference_present_15", list(range(1, W.N_ORG + 1)), False)):
            vals = [F.nanmean(list(L.macro_values(R, FRAME, a, organs, ids, pen).values())) for a in members]
            row[key] = F.nanmean(vals)
        row["arms"] = list(members)
        out[name] = row
    return out


def analyse_a1_2(R, ids, have, extras=None, not_run=None, p1_pass=False):
    """The A1.2 secondary contrasts (see the module docstring). R {cid: frames}; have = arms run; extras = {'merge': {draw:
    MERGE_N DONE record}, 'labelmap': {draw: LABELMAP DONE record}, 'rules': {draw: rules}}; p1_pass = P1's reading (Q is
    computed only under P1 PASS). Pure function."""
    have = set(have)
    extras = extras or {}
    nr = not_run or {}
    f1b = lambda d: f"F1b_{d}"
    stock = lambda d: f"STOCK_{d}"
    mergen = lambda d: f"MERGE_N_{d}"
    lmap = lambda d: f"LABELMAP_{d}"
    f0 = lambda d: "F0"
    out = {"amendment": "word A1.2 (secondary, unadjusted)", "rule": A12_RULE,
           "arms_run": sorted(a for a in A.A12_ARMS if a in have),
           "arms_not_run": {a: nr.get(a, "not run") for a in A.A12_ARMS if a not in have}}

    d1 = [d for d in DRAWS if f1b(d) in have and stock(d) in have]
    _, s1 = contrast_a12(R, ids, f1b, stock, d1)
    s1["reading"] = sign_reading(s1["pooled"]["ci"], "F1B_BETTER", "STOCK_BETTER", "NO_DIFFERENCE_DETECTED") if d1 \
        else "NOT RUN"
    s1["per_draw_reading"] = {d: sign_reading(v["ci"], "F1B_BETTER", "STOCK_BETTER", "NO_DIFFERENCE_DETECTED")
                              for d, v in s1["per_draw"].items()}
    s1["per_organ"] = organ_block(R, ids, f1b, stock, d1, W.PRIMARY) if d1 else {}
    s1["estimand"] = "mean over the draws with F1b_d and STOCK_d of mean_i [macro(F1b_d, i) - macro(STOCK_d, i)]"
    out["S1_F1b_minus_STOCK"] = s1

    ds = [d for d in DRAWS if stock(d) in have]
    s1b = p1_form({d: L.macro_delta(R, FRAME, stock(d), "F0", W.PRIMARY, ids) for d in ds})
    if ds:
        s1b["reading"] = "STOCK_GAINS" if s1b["pass"] else "STOCK_NO_GAIN"
    s1b["estimand"] = "mean over the draws with STOCK_d of mean_i [macro(STOCK_d, i) - macro(F0, i)] (A1's P1 rule form)"
    out["S1b_STOCK_minus_F0"] = s1b

    dm = [d for d in DRAWS if f1b(d) in have and mergen(d) in have]
    _, s2 = contrast_a12(R, ids, mergen, f1b, dm)
    ci2 = s2["pooled"]["ci"]
    if dm:
        ni = F.ni_class(ci2)
        s2["reading_ni"] = {"NON-INFERIOR": "MERGE_N_NON_INFERIOR", "INFERIOR": "MERGE_N_INFERIOR",
                            "INCONCLUSIVE": "INCONCLUSIVE"}.get(ni, "NOT EVALUABLE")
        s2["reading_harm"] = {"HARMFUL": "MERGE_COSTS", "BENEFICIAL": "MERGE_GAINS",
                              "NEUTRAL": "NO_COST_DETECTED"}.get(F.harm_class(ci2), "NOT EVALUABLE")
    else:
        s2["reading_ni"] = s2["reading_harm"] = "NOT RUN"
    s2["per_organ"] = organ_block(R, ids, mergen, f1b, dm, W.PRIMARY) if dm else {}
    s2["organs_inferior"] = [o for o, v in s2["per_organ"].items() if v["ni"] == "INFERIOR"]
    s2["organs_inconclusive"] = [o for o, v in s2["per_organ"].items() if v["ni"] == "INCONCLUSIVE"]
    s2["colon_rectum"] = organ_block(R, ids, mergen, f1b, dm, [10]) if dm else {}
    s2["femoral_heads_merge_n_minus_f1b"] = organ_block(R, ids, mergen, f1b, dm, W.DESCRIPTIVE) if dm else {}
    s2["femoral_heads_f1b_minus_f0"] = organ_block(R, ids, f1b, f0, dm, W.DESCRIPTIVE) if dm else {}
    s2["merged_classes"] = {d: merged_classes_rule_b(extras.get("merge", {}).get(d) or {}, ids) for d in dm}
    s2["companion_MERGE_N_minus_F0"] = p1_form({d: L.macro_delta(R, FRAME, mergen(d), "F0", W.PRIMARY, ids) for d in dm})
    s2["estimand"] = "mean over the draws with MERGE_N_d and F1b_d of mean_i [macro(MERGE_N_d, i) - macro(F1b_d, i)]"
    out["S2_MERGE_N_minus_F1b"] = s2

    dl = [d for d in DRAWS if lmap(d) in have]
    _, s3 = contrast_a12(R, ids, lmap, f0, dl)
    s3["reading"] = sign_reading(s3["pooled"]["ci"], "RELABEL_GAINS", "RELABEL_LOSES", "NO_DIFFERENCE_DETECTED") if dl \
        else "NOT RUN"
    s3["per_draw_reading"] = {d: sign_reading(v["ci"], "RELABEL_GAINS", "RELABEL_LOSES", "NO_DIFFERENCE_DETECTED")
                              for d, v in s3["per_draw"].items()}
    if dl:
        e0 = {c: 100.0 - v for c, v in WS.score_of(R, "F0", W.PRIMARY, ids).items()}
        e1 = {d: {c: 100.0 - v for c, v in WS.score_of(R, lmap(d), W.PRIMARY, ids).items()} for d in dl}
        S, ci = WS.boot_ratio(e0, e1)
        s3["S_LM"] = {"S": S, "ci": ci, "present": bool(ci[0] is not None and ci[0] > 0),
                      "what": "(E0 - E_LABELMAP) / E0, E = 100 - penalised macro (word_score.boot_ratio, seed 20261020)"}
    else:
        s3["S_LM"] = {"S": None, "ci": [None, None], "present": False}
    s3["per_organ"] = organ_block(R, ids, lmap, f0, dl, W.PRIMARY) if dl else {}
    rules = {}
    for d in dl:
        m = (extras.get("labelmap", {}).get(d) or {}).get("merged") or {}
        rules[d] = {ts_class_name(c): organ_name(int(o)) for c, o in sorted(m.items(), key=lambda x: int(x[0]))}
    s3["rules"] = rules
    s3["rules_identical_across_draws"] = len({json.dumps(v, sort_keys=True) for v in rules.values()}) <= 1 if rules else None
    s3["estimand"] = "mean over the draws with LABELMAP_d of mean_i [macro(LABELMAP_d, i) - macro(F0, i)]"
    out["S3_LABELMAP_minus_F0"] = s3

    dfl = [d for d in DRAWS if f1b(d) in have and lmap(d) in have]
    _, s4 = contrast_a12(R, ids, f1b, lmap, dfl)
    s4["reading"] = sign_reading(s4["pooled"]["ci"], "FT_EXCEEDS_RELABEL", "RELABEL_EXCEEDS_FT", "NOT_DETECTED") if dfl \
        else "NOT RUN"
    s4["per_draw_reading"] = {d: sign_reading(v["ci"], "FT_EXCEEDS_RELABEL", "RELABEL_EXCEEDS_FT", "NOT_DETECTED")
                              for d, v in s4["per_draw"].items()}
    s4["per_organ"] = organ_block(R, ids, f1b, lmap, dfl, W.PRIMARY) if dfl else {}
    if dfl and p1_pass:
        num = {d: L.macro_delta(R, FRAME, lmap(d), "F0", W.PRIMARY, ids) for d in dfl}
        den = {d: L.macro_delta(R, FRAME, f1b(d), "F0", W.PRIMARY, ids) for d in dfl}
        q, qci, ndrop = ratio_boot(num, den)
        s4["Q_share_of_F1b_gain_recovered_by_relabelling"] = {"Q": q, "ci": qci, "replicates_dropped": ndrop,
                                                             "draws": dfl, "seed": F.SEED_TWO_LEVEL, "nboot": F.NBOOT}
    else:
        s4["Q_share_of_F1b_gain_recovered_by_relabelling"] = {"Q": None, "ci": [None, None],
                                                             "why": "NOT EVALUABLE: needs P1 PASS and both arms"}
    s4["estimand"] = "mean over the draws with F1b_d and LABELMAP_d of mean_i [macro(F1b_d, i) - macro(LABELMAP_d, i)]"
    out["S4_F1b_minus_LABELMAP"] = s4

    out["descriptive_absolute_dice_vista3d_link"] = absolute_dice(
        R, ids, {"F0": ["F0"], "F1b": [f1b(d) for d in DRAWS if f1b(d) in have], "STOCK": [stock(d) for d in ds],
                 "MERGE_N": [mergen(d) for d in dm], "LABELMAP": [lmap(d) for d in dl]})
    out["readings"] = {"S1": s1["reading"], "S1b": s1b.get("reading", "NOT RUN"), "S2_ni": s2["reading_ni"],
                       "S2_harm": s2["reading_harm"], "S3": s3["reading"], "S3_S_LM_present": s3["S_LM"]["present"],
                       "S4": s4["reading"]}
    out["claims_allowed"] = {
        "the gentle fine-tune (F1b) beats nnU-Net's stock fine-tune on WORD": s1["reading"] == "F1B_BETTER",
        "nnU-Net's stock fine-tune beats the gentle fine-tune on WORD": s1["reading"] == "STOCK_BETTER",
        "the stock fine-tune gains over zero-shot TS on WORD": s1b.get("reading") == "STOCK_GAINS",
        "merging the non-target classes back costs the 12 organs' macro at most 1 Dice point":
            s2["reading_ni"] == "MERGE_N_NON_INFERIOR",
        "organs to name as a cost of the merge": s2["organs_inferior"],
        "a class-level relabelling fixed from four scans improves zero-shot TS": s3["reading"] == "RELABEL_GAINS",
        "the fine-tune gain is more than a class-level relabelling": s4["reading"] == "FT_EXCEEDS_RELABEL",
        "a class-level relabelling does better than the fine-tune": s4["reading"] == "RELABEL_EXCEEDS_FT",
        "any A1.2 contrast changes P1, P2 or the outcome label": False,
    }
    return out


def analyse_all(R, ids, have, dec=None, not_run=None, extras=None):
    """A1 / A1.1 (analyse_a1, unchanged) + A1.2 (analyse_a1_2) on the same exact per-scan records. A1.2 rule: a
    secondary-analysis error never blocks the primary results. If analyse_a1_2 raises an Exception, or returns a block
    that results.json cannot hold (serialised here exactly as final_common.write_json does), the A1.2 block is
    {"error": "<type>: <message>"} and P1, P2 and the outcome label are returned unchanged (finish() then logs a
    SECONDARY_ERROR line). A BaseException that is not an Exception (a signal, SystemExit) still propagates."""
    res = analyse_a1(R, ids, have, dec, not_run=not_run)
    try:
        blk = analyse_a1_2(R, ids, have, extras, not_run=not_run, p1_pass=res["P1_gain"]["pass"])
        json.dumps(F.clean_nan(blk), default=F._default)
        res["A1_2"] = blk
    except Exception as e:
        res["A1_2"] = {"error": f"{type(e).__name__}: {e}"}
    return res


def secondary_error(res):
    """The A1.2 block's error text, or None when the secondary analysis ran."""
    blk = res.get("A1_2")
    if isinstance(blk, dict) and "error" in blk:
        return " ".join(str(blk["error"]).split()) or "-"
    return None


# ---------------------------------------------------------------------------------------------------- gates
def arms_gate(ctx):
    """(have, files, notes). F0 / F1b_d1 DONE; every other arm (A1's draws, A1.2's arms) DONE before the cutoff or declared
    NOT RUN; an A1.2 CPU arm must name the current DONE records of its inputs (a12_inputs_consistent)."""
    nr = A.notrun_declared(ctx)
    if "EXPERIMENT" in nr:
        raise A.Refuse(f"the experiment was declared NOT RUN ({nr['EXPERIMENT']}): no look")
    dirs = A.arm_dirs(ctx.pred_dir)
    have, notes = [], {}
    for arm in A.ALL_ARMS:
        done = dirs[arm] / "DONE.json"
        if arm in nr:
            if arm in A.REQUIRED:
                raise A.Refuse(f"required arm {arm} was declared NOT RUN ({nr[arm]}): no look")
            notes[arm] = f"declared NOT RUN: {nr[arm]}"
            continue
        if not done.is_file():
            if arm in A.REQUIRED:
                raise A.Refuse(f"required arm {arm}: predictions not DONE")
            raise A.Refuse(f"arm {arm}: neither DONE nor declared NOT RUN in {ctx.rel(ctx.prereg_log)} (no interim look)")
        if arm != "F0" and datetime.fromtimestamp(os.path.getmtime(done)) >= ctx.cutoff:
            raise A.Refuse(f"arm {arm}: DONE after the cutoff {ctx.cutoff}; declare it NOT RUN first")
        have.append(arm)
    a12_inputs_consistent(have, dirs)
    return have, dirs, notes


def a12_inputs_consistent(have, dirs):
    """MERGE_N_dX was built from the current F0 and F1b_dX DONE records (and F1b_dX is in the look); LABELMAP_dX from the
    current F0 DONE record, with its rules file unchanged. Returns {arm: record}."""
    out = {}
    for d in DRAWS:
        m = f"MERGE_N_{d}"
        if m in have:
            if f"F1b_{d}" not in have:
                raise A.Refuse(f"arm {m} is DONE but its input F1b_{d} is not in the look; declare {m} NOT RUN first")
            rec = json.load(open(dirs[m] / "DONE.json"))
            want = {"F0_done_sha256": F.sha256(dirs["F0"] / "DONE.json"),
                    f"F1b_{d}_done_sha256": F.sha256(dirs[f"F1b_{d}"] / "DONE.json")}
            got = {k: (rec.get("inputs") or {}).get(k) for k in want}
            if got != want:
                raise A.Refuse(f"arm {m}: built from other input DONE records than the current ones ({got} != {want})")
            out[m] = {"inputs": want}
        lm = f"LABELMAP_{d}"
        if lm in have:
            rec = json.load(open(dirs[lm] / "DONE.json"))
            h0 = F.sha256(dirs["F0"] / "DONE.json")
            if (rec.get("inputs") or {}).get("F0_done_sha256") != h0:
                raise A.Refuse(f"arm {lm}: built from another F0 DONE record than the current one")
            rf = dirs[lm] / "rules.json"
            if not rf.is_file() or F.sha256(rf) != rec.get("rules_sha256"):
                raise A.Refuse(f"arm {lm}: its rules file is missing or differs from its DONE record")
            out[lm] = {"inputs": {"F0_done_sha256": h0}, "rules_sha256": rec["rules_sha256"]}
    return out


def a12_extras(have, dirs):
    """The A1.2 records the analysis reads besides the per-scan records (hashed in maps.json before the marker)."""
    ex = {"merge": {}, "labelmap": {}, "rules": {}}
    for d in DRAWS:
        if f"MERGE_N_{d}" in have:
            ex["merge"][d] = json.load(open(dirs[f"MERGE_N_{d}"] / "DONE.json"))
        if f"LABELMAP_{d}" in have:
            ex["labelmap"][d] = json.load(open(dirs[f"LABELMAP_{d}"] / "DONE.json"))
            ex["rules"][d] = json.load(open(dirs[f"LABELMAP_{d}"] / "rules.json"))
    return ex


def gates(ctx, check=False, log=print, stat_zip=True):
    """Gates 1-4. Returns a record; raises A.Refuse on the first failure (check=True collects every result instead).
    stat_zip=False (--check) leaves the labelsTs zip alone entirely; the look itself stats it."""
    rec, fails = {}, []

    def step(name, fn):
        try:
            rec[name] = {"ok": True, "detail": fn()}
        except A.Refuse as e:
            rec[name] = {"ok": False, "detail": e.msg}
            if not check:
                raise
            fails.append(name)
        return rec[name]

    step("1_no_second_look", lambda: A.look_not_made(ctx))
    step("2_osf_gate", lambda: A.osf_gate(ctx)[1])
    plan_box = {}

    def integ():
        plan, r = A.integrity(ctx, HERE / "word_a1_score.py", log)
        plan_box["plan"] = plan
        if ctx.real:
            try:
                F.check_conventions(plan["implementation"]["conventions_sha256"])
            except SystemExit as e:
                raise A.Refuse(f"integrity: {e}")
            reg = plan["sampling_plan"]["download"]["archive_labelsTs/WORD_V0.1.0_labelsTs.zip"]["bytes"]
            if ctx.labels_zip_bytes != reg:
                raise A.Refuse(f"integrity: the scorer's labelsTs size {ctx.labels_zip_bytes} != plan C's {reg}")
        return r

    step("3_integrity", integ)

    def zip_size():
        if not ctx.labels_zip.is_file():
            raise A.Refuse(f"labelsTs zip missing: {ctx.rel(ctx.labels_zip)}")
        n = os.stat(ctx.labels_zip).st_size
        if ctx.labels_zip_bytes is not None and n != ctx.labels_zip_bytes:
            raise A.Refuse(f"labelsTs zip size {n} != registered {ctx.labels_zip_bytes}")
        return {"bytes": n, "content_read": False}

    if stat_zip:
        step("3b_labels_zip_size", zip_size)
    else:
        rec["3b_labels_zip_size"] = {"ok": None, "detail": "not checked by --check: the zip is not touched; the look "
                                                           "checks its size by os.stat before the marker"}

    def arms():
        have, _, notes = arms_gate(ctx)
        return {"have": have, "not_run": notes}

    step("4_arms", arms)
    rec["all_ok"] = not fails
    rec["failed"] = fails
    return rec, plan_box.get("plan")


# ---------------------------------------------------------------------------------------------------- per-scan records
def _exact_default(o):
    import numpy as np
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def _canon(obj):
    return json.dumps(obj, sort_keys=True, allow_nan=True, default=_exact_default)


def record_done(ctx, cid, rec, dec, have, shas, mk_sha, mj):
    """Write look/pervol_done/<cid>.json: the exact per-scan record (NaN / inf kept) bound to the marker, maps.json, the
    scan's map sha256 and the sha256 of word_score.job's look/pervol/<cid>.json. Returns the exact record as re-read."""
    payload = json.loads(_canon({"cid": cid, "frames": rec, "dec": dec}))
    pv = ctx.look_dir / "pervol" / f"{cid}.json"
    doc = {"cid": cid, "arms": sorted(have), "maps_sha256": {a: shas[a][cid] for a in sorted(have)},
           "marker_sha256": mk_sha, "maps_json_sha256": mj, "pervol_file": ctx.rel(pv),
           "pervol_sha256": F.sha256(pv) if pv.is_file() else None,
           "payload_sha256": hashlib.sha256(_canon(payload).encode()).hexdigest(), "written": F.now(),
           "payload": payload}
    F.write_text(ctx.look_dir / "pervol_done" / f"{cid}.json", json.dumps(doc, allow_nan=True, indent=None) + "\n")
    return payload


def check_pervol(ctx, cid, have, shas, mk_sha, mj):
    """(ok, why, payload): may look/pervol/<cid>.json be reused by a continuation? Its pervol_done record must exist,
    parse, name this cid, these arms, these map sha256, this marker and this maps.json, carry an intact payload, and the
    pervol file must still have the sha256 recorded there."""
    p = ctx.look_dir / "pervol_done" / f"{cid}.json"
    pv = ctx.look_dir / "pervol" / f"{cid}.json"
    if not p.is_file():
        return False, "no pervol_done record", None
    if not pv.is_file():
        return False, "no look/pervol file", None
    try:
        doc = json.loads(p.read_text())
        pdoc = json.loads(pv.read_text())
    except (OSError, ValueError) as e:
        return False, f"unreadable: {type(e).__name__}", None
    pay = doc.get("payload") or {}
    checks = [
        (doc.get("cid") == cid and pay.get("cid") == cid and pdoc.get("cid") == cid, "cid"),
        (doc.get("arms") == sorted(have), "arms"),
        (doc.get("maps_sha256") == {a: shas[a][cid] for a in sorted(have)}, "map sha256"),
        (doc.get("marker_sha256") == mk_sha, "marker"),
        (doc.get("maps_json_sha256") == mj, "maps.json"),
        (doc.get("pervol_sha256") == F.sha256(pv), "pervol file sha256"),
        (hashlib.sha256(_canon(pay).encode()).hexdigest() == doc.get("payload_sha256"), "payload sha256"),
        (sorted(((pay.get("frames") or {}).get(FRAME) or {}).get("arms", {})) == sorted(have), "payload arms"),
        (sorted(pay.get("dec") or {}) == sorted(have), "payload dec arms"),
        (sorted(((pdoc.get("frames") or {}).get(FRAME) or {}).get("arms", {})) == sorted(have), "pervol arms"),
    ]
    bad = [n for ok, n in checks if not ok]
    if bad:
        return False, "integrity: " + ", ".join(bad), None
    return True, "ok", pay


def run_jobs(jobs, workers, timeout_s, got, log, job_fn=None):
    """Score the scans. workers > 1: a fork ProcessPoolExecutor; a killed worker raises BrokenProcessPool, and no result
    within timeout_s raises TimeoutError; in both cases the remaining workers are killed. Never hangs."""
    job_fn = job_fn or WS.job
    if not jobs:
        return
    if workers <= 1:
        for j in jobs:
            c, rec, dec = job_fn(j)
            got(c, rec, dec)
        return
    import multiprocessing as mp
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
    ex = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork"), initializer=_worker_init)
    ok = False
    try:
        pending = {ex.submit(job_fn, j): j[0] for j in jobs}
        while pending:
            done, _ = wait(list(pending), timeout=timeout_s, return_when=FIRST_COMPLETED)
            if not done:
                raise TimeoutError(f"no scan finished within {timeout_s:.0f} s (pending {sorted(pending.values())[:6]}): "
                                   f"a worker is stuck")
            for fu in done:
                pending.pop(fu)
                c, rec, dec = fu.result()
                got(c, rec, dec)
        ok = True
    finally:
        if ok:
            ex.shutdown(wait=True)
        else:
            procs = list((getattr(ex, "_processes", None) or {}).values())
            ex.shutdown(wait=False, cancel_futures=True)
            for p in procs:
                try:
                    p.kill()
                    p.join(10)
                except (OSError, ValueError, AttributeError):
                    pass


def score_all(ctx, st, mk, workers, timeout_s, reuse, log, job_fn=None):
    """(R, D, record). Every result goes through record_done; analysis uses the exact records as re-read."""
    have, ids, shas, mj = st["have"], st["ids"], st["shas"], st["mj"]
    mk_sha = F.sha256(mk)
    R, D, reused, why = {}, {}, [], {}
    if reuse:
        for c in ids:
            ok, w, pay = check_pervol(ctx, c, have, shas, mk_sha, mj)
            if ok:
                R[c], D[c] = pay["frames"], pay["dec"]
                reused.append(c)
            else:
                why[c] = w
    todo = [c for c in ids if c not in R]
    lut = st["lut"]
    assert all(lut[a_] == A.ARM_LUT[a_] for a_ in have), lut
    jobs = [(c, {a_: (st["files"][a_][c], lut[a_]) for a_ in have}, {a_: shas[a_][c] for a_ in have}, str(mk))
            for c in todo]

    def got(c, rec, dec):
        pay = record_done(ctx, c, rec, dec, have, shas, mk_sha, mj)
        R[c], D[c] = pay["frames"], pay["dec"]

    t0 = time.time()
    log(f"scoring {len(todo)} scans ({len(reused)} reused) with {workers} workers")
    run_jobs(jobs, workers, timeout_s, got, log, job_fn)
    return R, D, {"reused": reused, "scored": todo, "not_reused_why": why, "workers": workers,
                  "result_timeout_s": timeout_s, "wall_s": round(time.time() - t0, 1)}


# ---------------------------------------------------------------------------------------------------- the look
@contextlib.contextmanager
def after_marker(ctx, log):
    """Everything after the GT_OPENED marker: SIGTERM / SIGHUP raise Terminated, and any exception except the zip-mismatch
    abort (which has its own GT_ABORTED line) logs 'GT_CRASHED <type> <message>' before it propagates."""
    old = {}
    for s in (signal.SIGTERM, signal.SIGHUP):
        try:
            old[s] = signal.signal(s, _on_signal)
        except ValueError:
            pass
    try:
        yield
    except A.Refuse as e:
        if e.code != 4:
            _crashed(ctx, log, e)
        raise
    except BaseException as e:
        _crashed(ctx, log, e)
        raise
    finally:
        for s, h in old.items():
            try:
                signal.signal(s, h)
            except ValueError:
                pass


def _crashed(ctx, log, e):
    msg = " ".join(str(getattr(e, "msg", e)).split())[:400] or "-"
    line = (f"GT_CRASHED {type(e).__name__} {msg} (no results; the single look is not over; A1.1 continuation: "
            f"word_a1_score.py --continue)")
    try:
        ctx.plog(line)
    except BaseException:
        pass
    try:
        log(line)
    except BaseException:
        pass


def pre_marker(ctx, log):
    """Gates 1-6 (nothing after the marker). Returns the look's state."""
    ctx.check_word_paths()
    grec, plan = gates(ctx, check=False, log=log)
    url = grec["2_osf_gate"]["detail"]["url"]
    pinned_sha = plan["implementation"]["labels_ts_zip_sha256"] if ctx.real else ctx.labels_zip_sha256
    assert pinned_sha and len(pinned_sha) == 64, "the labelsTs zip sha256 must be pinned"
    ids = list(W.split()["test"])
    dup, dropped = WS.duplicates(ids, log)
    ids = [c for c in ids if c not in dropped]
    have, dirs, notes = arms_gate(ctx)
    files = {}
    for arm in have:
        done = dirs[arm] / "DONE.json"
        bad = [c for c in ids if L.done_raw_sha(done, c) != F.sha256(dirs[arm] / "raw" / f"{c}.nii.gz")]
        if bad:
            raise A.Refuse(f"arm {arm}: maps do not match their DONE record ({bad[:3]})")
        files[arm] = {c: str(dirs[arm] / "raw" / f"{c}.nii.gz") for c in ids}
    shas = L.maps_manifest(files)
    lut = {a_: A.ARM_LUT[a_] for a_ in have}
    done_sha = {a_: F.sha256(dirs[a_] / "DONE.json") for a_ in have}
    rules_sha = {a_: F.sha256(dirs[a_] / "rules.json") for a_ in have if a_ in A.LABELMAP_ARMS}
    extras = a12_extras(have, dirs)
    ctx.look_dir.mkdir(parents=True, exist_ok=True)
    maps_json = ctx.look_dir / "maps.json"
    F.write_json(maps_json, {"sha256": shas, "arms": have, "lut": lut, "done_sha256": done_sha, "rules_sha256": rules_sha,
                             "not_run": notes, "ids": ids, "dropped_as_duplicates": dropped, "duplicates_check": dup,
                             "scorer": "word A1 TS-only (word_a1_score.py), A1.2 arms included", "written": F.now()})
    return {"grec": grec, "url": url, "pinned": pinned_sha, "ids": ids, "dup": dup, "dropped": dropped, "have": have,
            "notes": notes, "files": files, "shas": shas, "lut": lut, "done_sha": done_sha, "extras": extras,
            "mj": F.sha256(maps_json)}


def check_zip(ctx, st):
    zsha = F.sha256(ctx.labels_zip)
    if zsha != st["pinned"]:
        ctx.plog(f"GT_ABORTED sha256 {zsha} {ctx.rel(ctx.labels_zip)} (labelsTs zip differs from the registered "
                 f"{st['pinned'][:16]}...; no label read; the single look is used up)")
        raise A.Refuse("labelsTs zip changed since registration: look aborted after the marker", code=4)
    return zsha


def finish(ctx, st, R, D, zsha, scoring, continuation=None, log=print):
    res = analyse_all(R, st["ids"], set(st["have"]), D, not_run={a_: v for a_, v in A.notrun_declared(ctx).items()},
                      extras=st["extras"])
    res.update(written=F.now(), duplicates_check=st["dup"], dropped_as_duplicates=st["dropped"], gates=st["grec"],
               maps_sha256=st["mj"], labels_ts_zip_sha256=zsha, osf_registration=st["url"],
               plan_sha256=F.sha256(F.plan_path("word")) if ctx.real else None,
               a1_sha256=F.sha256(A.A1_JSON) if ctx.real else None,
               a1_1_sha256=F.sha256(A.A1_1_JSON) if ctx.real else None,
               a1_2_sha256=F.sha256(A.A1_2_JSON) if ctx.real else None,
               look={"single_look": True, "continuation": continuation, "scoring": scoring,
                     "rule": "one look (A1); a continuation under A1.1 is the same look; A1.2's arms are scored in it"})
    out = ctx.look_dir / "results.json"
    F.write_json(out, res)
    cont = f"; continued (continuation {continuation['n']})" if continuation else ""
    err = secondary_error(res)
    if err is not None:
        ctx.plog(f"SECONDARY_ERROR {A.NOTRUN_TAG} {err[:400]} (the A1.2 secondary analysis raised; A1.2 rule: a "
                 f"secondary-analysis error never blocks the primary results; P1, P2 and the outcome label are in "
                 f"{ctx.rel(out)}, whose A1_2 block holds this error; the results line follows)")
        log(f"SECONDARY_ERROR {err[:400]}")
        a12 = f"A1.2 secondary: ERROR ({err[:160]}; see the SECONDARY_ERROR line)"
        r12 = {"error": err}
    else:
        r12 = dict(res["A1_2"].get("readings") or {})
        a12 = (f"A1.2 secondary: F1b-STOCK {r12.get('S1')}; STOCK-F0 {r12.get('S1b')}; MERGE_N-F1b {r12.get('S2_ni')} / "
               f"{r12.get('S2_harm')}; LABELMAP-F0 {r12.get('S3')}; F1b-LABELMAP {r12.get('S4')}")
    ctx.plog(f"RESULTS sha256 {F.sha256(out)} {ctx.rel(out)} (word A1: {res['outcome']['combination']}; organs "
             f"{res['outcome']['organ_costs']}; {res['outcome']['draw_dependence']}{cont}; {a12})")
    log(f"WORD A1 look done: {res['outcome']}; A1.2 {r12}")
    return res


def look(ctx, workers=6, log=print, result_timeout_s=RESULT_TIMEOUT_S, job_fn=None):
    """Gates, then the single look. Returns the results dict. Raises A.Refuse on a refusal."""
    st = pre_marker(ctx, log)
    mk = F.open_marker(ctx.look_dir / "GT_OPENED",
                       {"set": f"WORD test ({len(st['ids'])})", "ids": st["ids"], "maps_sha256": st["mj"],
                        "arms": st["have"], "osf_registration": st["url"],
                        "scorer": "word A1 TS-only (word_a1_score.py)",
                        "declared": "FIRST and ONLY look at the WORD test labels (amendment word A1, registered on "
                                    "OSF; A1.1 continuation rule)"})
    with after_marker(ctx, log):
        ctx.plog(f"GT_OPENED sha256 {F.sha256(mk)} {ctx.rel(mk)} (maps.json sha256 {st['mj']}; word A1 TS-only scorer; "
                 f"arms {','.join(st['have'])}; {len(st['ids'])} test scans; dropped as duplicates "
                 f"{st['dropped'] or 'none'}; OSF {st['url']})")
        log(f"GT_OPENED {ctx.rel(mk)}; opening {ctx.rel(ctx.labels_zip)}")
        zsha = check_zip(ctx, st)
        R, D, sc = score_all(ctx, st, mk, workers, result_timeout_s, False, log, job_fn)
        return finish(ctx, st, R, D, zsha, sc, None, log)


def continue_gates(ctx, log=print):
    """A1.1 continuation rule: refuses unless every condition holds. Returns the look's state (from maps.json)."""
    ctx.check_word_paths()
    mk = ctx.look_dir / "GT_OPENED"
    res = ctx.look_dir / "results.json"
    if not mk.is_file():
        raise A.Refuse(f"continue: no {ctx.rel(mk)}: there is no look to continue")
    if res.exists():
        raise A.Refuse(f"continue: {ctx.rel(res)} exists: the look is complete")
    lines = ctx.log_text_lines()
    sp = [ln.split() for ln in lines]
    if any(len(p) >= 6 and p[2] == "RESULTS" and p[5] == ctx.rel(res) for p in sp):
        raise A.Refuse(f"continue: {ctx.rel(ctx.prereg_log)} already has a RESULTS line for {ctx.rel(res)}")
    if any(len(p) >= 3 and p[2] == "GT_ABORTED" for p in sp):
        raise A.Refuse("continue: a GT_ABORTED line exists (the labelsTs zip differed from its pin): the look is over",
                       code=4)
    gt = [ln for ln, p in zip(lines, sp) if len(p) >= 6 and p[2] == "GT_OPENED" and p[5] == ctx.rel(mk)]
    if len(gt) != 1:
        raise A.Refuse(f"continue: expected exactly one GT_OPENED line for {ctx.rel(mk)}, found {len(gt)}")
    mk_sha = F.sha256(mk)
    if gt[0].split()[4] != mk_sha:
        raise A.Refuse("continue: the marker file no longer matches its GT_OPENED line")
    m = re.search(r"\(maps\.json sha256 ([0-9a-f]{64});", gt[0])
    maps_json = ctx.look_dir / "maps.json"
    mj = F.sha256(maps_json) if maps_json.is_file() else None
    if m is None or mj != m.group(1):
        raise A.Refuse("continue: maps.json does not match the maps.json sha256 on the GT_OPENED line")
    mkrec = json.loads(mk.read_text())
    if mkrec.get("maps_sha256") != mj:
        raise A.Refuse("continue: the marker's maps_sha256 differs from maps.json")
    url, osf = A.osf_gate(ctx)
    if mkrec.get("osf_registration") != url:
        raise A.Refuse("continue: the OSF registration differs from the one recorded in the marker")
    note = ("a continuation with changed code is not allowed (A1.1); a code change after the marker is a disclosed "
            "deviation outside the continuation rule")
    if ctx.real:
        try:
            plan, integ = A.integrity(ctx, HERE / "word_a1_score.py", log)
            F.check_conventions(plan["implementation"]["conventions_sha256"])
        except A.Refuse as e:
            raise A.Refuse(f"continue: {e.msg}; {note}")
        except SystemExit as e:
            raise A.Refuse(f"continue: {e}; {note}")
        pinned = plan["implementation"]["labels_ts_zip_sha256"]
    else:
        bad = A.frozen_code_mismatch(ctx, ctx.code_files)
        if bad:
            raise A.Refuse(f"continue: code differs from its CODE lines {bad}; {note}")
        integ, pinned = {"code_files": [ctx.rel(p) for p in ctx.code_files]}, ctx.labels_zip_sha256
    mp = json.load(open(maps_json))
    have, ids, shas = list(mp["arms"]), list(mp["ids"]), mp["sha256"]
    dirs = A.arm_dirs(ctx.pred_dir)
    files = {a_: {c: str(dirs[a_] / "raw" / f"{c}.nii.gz") for c in ids} for a_ in have}
    changed = [(a_, c) for a_ in have for c in ids if F.sha256(files[a_][c]) != shas[a_][c]]
    if changed:
        raise A.Refuse(f"continue: prediction maps changed since maps.json was written: {changed[:3]}")
    lut = dict(mp.get("lut") or {})
    if set(lut) != set(have) or any(lut[a_] != A.ARM_LUT[a_] for a_ in have):
        raise A.Refuse("continue: maps.json's label maps differ from the registered ones (word_a1_common.ARM_LUT)")
    dch = [a_ for a_ in have if F.sha256(dirs[a_] / "DONE.json") != (mp.get("done_sha256") or {}).get(a_)]
    rch = [a_ for a_, h in (mp.get("rules_sha256") or {}).items() if F.sha256(dirs[a_] / "rules.json") != h]
    if dch or rch or set(mp.get("rules_sha256") or {}) != {a_ for a_ in have if a_ in A.LABELMAP_ARMS}:
        raise A.Refuse(f"continue: DONE records or LABELMAP rules changed since maps.json was written: {dch or rch}")
    extras = a12_extras(have, dirs)
    n_prev = sum(1 for p in sp if len(p) >= 6 and p[2] == "GT_CONTINUED" and p[5] == ctx.rel(mk))
    n_crash = sum(1 for p in sp if len(p) >= 3 and p[2] == "GT_CRASHED")
    grec = {"continue": {"ok": True, "detail": {"marker_sha256": mk_sha, "maps_json_sha256": mj, "osf": osf,
                                                "integrity": integ, "previous_continuations": n_prev,
                                                "gt_crashed_lines": n_crash}}}
    return {"grec": grec, "url": url, "pinned": pinned, "ids": ids, "dup": mp.get("duplicates_check"),
            "dropped": mp.get("dropped_as_duplicates", []), "have": have, "notes": mp.get("not_run", {}),
            "files": files, "shas": shas, "lut": lut, "done_sha": mp.get("done_sha256"), "extras": extras, "mj": mj,
            "mk": mk, "mk_sha": mk_sha, "n": n_prev + 1, "n_crash": n_crash}


def continue_look(ctx, workers=6, log=print, result_timeout_s=RESULT_TIMEOUT_S, job_fn=None):
    """The A1.1 continuation of the single look."""
    st = continue_gates(ctx, log)
    with after_marker(ctx, log):
        ctx.plog(f"GT_CONTINUED sha256 {st['mk_sha']} {ctx.rel(st['mk'])} (continuation {st['n']} of the single WORD "
                 f"look under amendment word A1.1; maps.json sha256 {st['mj']} as on the GT_OPENED line; code unchanged "
                 f"(CODE lines); GT_CRASHED lines so far {st['n_crash']}; completed per-scan records are reused after "
                 f"their integrity check)")
        zsha = check_zip(ctx, st)
        R, D, sc = score_all(ctx, st, st["mk"], workers, result_timeout_s, True, log, job_fn)
        cont = {"n": st["n"], "reused": sc["reused"], "rescored": sc["scored"], "not_reused_why": sc["not_reused_why"],
                "gt_crashed_lines_before": st["n_crash"]}
        return finish(ctx, st, R, D, zsha, sc, cont, log)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--check", action="store_true", help="evaluate the gates and print them; write nothing")
    ap.add_argument("--continue", dest="cont", action="store_true", help="A1.1 continuation of the single look")
    ap.add_argument("--result-timeout", type=float, default=RESULT_TIMEOUT_S)
    a = ap.parse_args(argv)
    assert 1 <= a.workers <= F.MAX_WORKERS
    ctx = A.Ctx.real_ctx()
    err = lambda m: print(m, file=sys.stderr)
    if a.check and not a.cont:
        rec, _ = gates(ctx, check=True, log=err, stat_zip=False)
        print(json.dumps(rec, indent=1, default=str))
        return 0 if rec["all_ok"] else 2
    if a.check and a.cont:
        try:
            st = continue_gates(ctx, log=err)
        except A.Refuse as e:
            print(json.dumps({"continue_allowed": False, "why": e.msg}, indent=1))
            return e.code
        print(json.dumps({"continue_allowed": True, "detail": st["grec"]}, indent=1, default=str))
        return 0
    F.be_nice(15)
    try:
        if a.cont:
            if not (ctx.look_dir / "GT_OPENED").is_file():
                raise A.Refuse("continue: no GT_OPENED marker: there is no look to continue")
        else:
            A.look_not_made(ctx)
        A.osf_gate(ctx)
    except A.Refuse as e:
        print(f"[REFUSE] {e.msg}", file=sys.stderr, flush=True)
        return e.code
    log = F.Log(ctx.score_log)
    try:
        if a.cont:
            continue_look(ctx, a.workers, log, a.result_timeout)
        else:
            look(ctx, a.workers, log, a.result_timeout)
    except A.Refuse as e:
        log(f"[REFUSE] {e.msg}")
        return e.code
    return 0


if __name__ == "__main__":
    sys.exit(main())
