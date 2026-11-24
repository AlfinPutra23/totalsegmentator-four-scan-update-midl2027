"""vccor/final FLARE-T50 ANALYSIS (pure functions; no file access; unit-tested on synthetic records by test_t50.py).

Input R {cid: {'NATIVE': fin_look.score_scan-style frame record}} (arms scored against FLARE-T50's reference in FLARE's 13
ids, native grid, conventions v1: presence >= 1 mL on both sides, FP organs scored 0, penalised macro over the 13 organs).

Primary R1 (alpha 0.05, not adjusted) = WORD A1's P1 rule on the three K = 4 FULL draws (s1337, s2025, s3407) vs F0:
  M = mean over the draws of mean_i D_{d,i}, D = macro(FULL_d) - macro(F0); PASS iff M >= +0.30 AND the scan t-CI lower
  bound > 0 AND the crossed two-level (scan x draw) bootstrap CI lower bound > 0 (m2_cluster.boot2 via final_common.two_level,
  10,000 resamples, seed 20261020) AND scans with a positive draw-averaged D >= ceil(2n/3). Per draw: the single-draw form
  (scan t-CI, scan bootstrap final_common.cluster_boot seed 20261022, the count) -> k of n_run -> WORD A1's outcome
  combination (GAIN_EVERY_DRAW ...).
Secondary Holm family (m = 6; Holm on two-sided bootstrap p-values, family-wise alpha 0.05; 'detected' iff the Holm-adjusted
p < 0.05 AND the estimate lies in the stated direction; the opposite direction with adjusted p < 0.05 is 'REVERSED'):
  S1 pull-back        SSFT_B - FULL_s1337, scan bootstrap (seed 20261022), direction < 0
  S2 label source     SSFT_D - SSFT_B, scan bootstrap, direction > 0
  S3 gentle vs stock  FULL_s1337 - STOCK, scan bootstrap, direction > 0   (STOCK - F0 read only as STOCK_GAINS /
                      STOCK_NO_GAIN by the P1 single-draw form)
  S4 more labels      K16_d - FULL_d paired by draw, two-level (seed 20261020), direction > 0
  S5 local vs borrowed mean of the 3 FLARE K = 4 draws - mean of the 3 borrowed models (each minus F0). The three borrowed
                      models are held FIXED (the inference is about these specific models, not a population of cohorts):
                      each replicate draws the scans with replacement (shared by both families) and the FLARE draws with
                      replacement; the borrowed family is averaged over its three models (seed 20261020), direction > 0
                      (code review SF7, option A); borrowed - F0 per cohort is descriptive (HELPS / HARMS / NEITHER by its
                      scan-bootstrap CI)
  S6 solid-organ volume per scan, the mean over liver, spleen, both kidneys and pancreas (organs present in the reference) of
                      |e_FULL_d| - |e_F0|, e = 100 (V_pred - V_ref) / V_ref; two-level over the 3 draws, direction < 0
Two-sided bootstrap p-value: p = min(1, 2 (min(#{r* <= 0}, #{r* >= 0}) + 1) / (B + 1)) over the B replicate estimates r*,
generated with exactly the resampling (and RNG stream) of the CI it accompanies.
Tier 2 (its own family, one test): K1 pooled - K4, crossed two-level with each family's draws resampled (seed 20261020),
  detected iff p < 0.05 and the estimate < 0; SINGLE_SCAN_RISK replicated iff at least one K = 1 draw has >= 25 % of its
  scans below F0.
Descriptive (registered): every scan (k of n with a positive draw-averaged D; 'every scan' only if k = n); organ costs (WORD
  A1's per-organ rule: two-level CI of each organ's Dice change, NON-INFERIOR / INFERIOR / INCONCLUSIVE at -1.0 Dice);
  SCRATCH_NM1000 - F0; per-draw values; the two qualitative panels (the scan with the median draw-averaged D, lower median for
  even n; the scan with the smallest, i.e. the largest loss).

Version of 2026-11-09.
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
for _p in (str(HERE.parent), str(HERE), str(HERE.parent.parent / "m2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import final_common as F  # noqa: E402
import fin_look as L  # noqa: E402
import m2_cluster as MC  # noqa: E402

FRAME = "NATIVE"
ORGANS = list(range(1, 14))
SOLID = [1, 3, 2, 13, 4]
NBOOT = F.NBOOT
SEED_TWO = F.SEED_TWO_LEVEL
SEED_SCAN = F.SEED_CLUSTER
ALPHA = 0.05
P1_MIN_MEAN = 0.30
RISK_FRAC = 0.25
K4 = {"s1337": "FULL_s1337", "s2025": "FULL_s2025", "s3407": "FULL_s3407"}
K16 = {"s1337": "K16_s1337", "s2025": "K16_s2025", "s3407": "K16_s3407"}
BORROWED = {"amos": "BORROW_amos", "mact": "BORROW_mact", "word": "BORROW_word"}
K1 = {"0014": "K1_0014", "0001": "K1_0001", "0006": "K1_0006", "0038": "K1_0038"}
FAMILY = ["S1", "S2", "S3", "S4", "S5", "S6"]


# ---------------------------------------------------------------------------------------------------- bootstrap replicates
def scan_boot_reps(values, nboot=NBOOT, seed=SEED_SCAN):
    """Replicates of the scan-level mean with exactly final_common.cluster_boot's resampling (one cluster per scan):
    -> (mean, reps, ids). The percentile CI of reps == cluster_boot(values)['ci'] (asserted in the unit tests)."""
    import numpy as np
    ids = [c for c in sorted(values) if values[c] is not None and np.isfinite(values[c])]
    if not ids:
        return None, np.zeros(0), ids
    v = np.array([values[c] for c in ids], float)
    G = len(ids)
    rng = np.random.default_rng(seed)
    reps = np.empty(nboot)
    for b0 in range(0, nboot, 1000):
        nb = min(1000, nboot - b0)
        w = rng.multinomial(G, np.full(G, 1.0 / G), size=nb).astype(float)
        reps[b0:b0 + nb] = (w @ v) / w.sum(1)
    return float(v.mean()), reps, ids


def two_level_reps(per_draw, nboot=NBOOT, seed=SEED_TWO):
    """Replicates of M (mean of the draw means) with exactly m2_cluster.boot2's crossed resampling and RNG stream:
    -> (M, reps). Percentile CI of reps == m2_cluster.boot2's CI (asserted in the unit tests)."""
    import numpy as np
    draws, scans, D = MC.matrix(per_draw)
    if not draws or not scans:
        return None, np.zeros(0)
    S, n = D.shape
    M = float(np.mean([np.nanmean(D[j]) for j in range(S)]))
    rng = np.random.default_rng(seed)
    pn, ps = np.full(n, 1.0 / n), np.full(S, 1.0 / S)
    cs = rng.multinomial(n, pn, size=nboot).astype(float)
    resample = S > 1
    cd = rng.multinomial(S, ps, size=nboot).astype(float) if resample else np.ones((nboot, S))
    val, bad = MC._stat(D, cs, cd)
    rounds = 0
    while bad.any():
        rounds += 1
        if rounds > MC.MAX_REDRAW_ROUNDS:
            raise RuntimeError("two-level bootstrap: replicates without an evaluation scan keep recurring")
        idx = np.flatnonzero(bad)
        cs[idx] = rng.multinomial(n, pn, size=len(idx))
        if resample:
            cd[idx] = rng.multinomial(S, ps, size=len(idx))
        v2, b2 = MC._stat(D, cs[idx], cd[idx])
        val[idx] = v2
        bad[:] = False
        bad[idx] = b2
    return M, val


def two_family_reps(per_a, per_b, nboot=NBOOT, seed=SEED_TWO, resample_b=True):
    """M_A - M_B for two families of draws evaluated on the same scans: each replicate draws the scans with replacement
    (shared by both families) and, independently, the draws of family A and (resample_b) the draws of family B with
    replacement (the crossed two-level bootstrap of m2_cluster applied to each family). resample_b=False holds family B's
    members fixed (S5: the three specific borrowed models). -> (estimate, reps, n_scans)."""
    import numpy as np
    da = sorted(per_a)
    db = sorted(per_b)
    scans = sorted(set().union(*[set(per_a[d]) for d in da], *[set(per_b[d]) for d in db]))

    def mat(per, ds):
        Z = np.full((len(ds), len(scans)), np.nan)
        for j, d in enumerate(ds):
            for i, c in enumerate(scans):
                v = per[d].get(c)
                if v is not None and np.isfinite(v):
                    Z[j, i] = float(v)
        return Z
    A, B = mat(per_a, da), mat(per_b, db)
    keep = np.isfinite(A).any(0) & np.isfinite(B).any(0)
    A, B = A[:, keep], B[:, keep]
    n = A.shape[1]
    if n == 0:
        return None, np.zeros(0), 0
    est = float(np.mean([np.nanmean(A[j]) for j in range(A.shape[0])]) -
                np.mean([np.nanmean(B[j]) for j in range(B.shape[0])]))
    rng = np.random.default_rng(seed)
    pn = np.full(n, 1.0 / n)
    sa, sb = A.shape[0], B.shape[0]
    cs = rng.multinomial(n, pn, size=nboot).astype(float)
    rb = resample_b and sb > 1
    ca = rng.multinomial(sa, np.full(sa, 1.0 / sa), size=nboot).astype(float) if sa > 1 else np.ones((nboot, sa))
    cb = rng.multinomial(sb, np.full(sb, 1.0 / sb), size=nboot).astype(float) if rb else np.ones((nboot, sb))
    va, bad_a = MC._stat(A, cs, ca)
    vb, bad_b = MC._stat(B, cs, cb)
    bad = bad_a | bad_b
    rounds = 0
    while bad.any():
        rounds += 1
        if rounds > MC.MAX_REDRAW_ROUNDS:
            raise RuntimeError("two-family bootstrap: replicates without an evaluation scan keep recurring")
        idx = np.flatnonzero(bad)
        cs[idx] = rng.multinomial(n, pn, size=len(idx))
        if sa > 1:
            ca[idx] = rng.multinomial(sa, np.full(sa, 1.0 / sa), size=len(idx))
        if rb:
            cb[idx] = rng.multinomial(sb, np.full(sb, 1.0 / sb), size=len(idx))
        a2, ba2 = MC._stat(A, cs[idx], ca[idx])
        b2, bb2 = MC._stat(B, cs[idx], cb[idx])
        va[idx], vb[idx] = a2, b2
        bad[:] = False
        bad[idx] = ba2 | bb2
    return est, va - vb, n


def pct_ci(reps, level=0.95):
    import numpy as np
    if len(reps) == 0:
        return [None, None]
    a = 100.0 * (1.0 - level) / 2.0
    lo, hi = np.percentile(reps, [a, 100.0 - a])
    return [float(lo), float(hi)]


def p_two_sided(reps):
    import numpy as np
    B = len(reps)
    if B == 0:
        return None
    le, ge = int((np.asarray(reps) <= 0).sum()), int((np.asarray(reps) >= 0).sum())
    return float(min(1.0, 2.0 * (min(le, ge) + 1) / (B + 1)))


def holm(p):
    """Holm step-down adjusted p-values for {name: p} (None entries are left out and reported as None)."""
    items = sorted([(v, k) for k, v in p.items() if v is not None])
    m = len(items)
    adj, run = {}, 0.0
    for i, (v, k) in enumerate(items):
        run = max(run, min(1.0, (m - i) * v))
        adj[k] = run
    return {k: adj.get(k) for k in p}


def direction_reading(est, p_adj, direction):
    if est is None or p_adj is None:
        return "NOT EVALUABLE"
    if p_adj < ALPHA:
        good = est < 0 if direction == "<0" else est > 0
        return "DETECTED" if good else "REVERSED"
    return "NOT DETECTED"


# ---------------------------------------------------------------------------------------------------- readings (WORD A1)
def organ_reading(classes):
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
    if p1_pass:
        if k == 3 and n_run == 3:
            return "GAIN_EVERY_DRAW"
        if n_run < 3 and k == n_run:
            return "GAIN_EVERY_RUN_DRAW"
        return "GAIN_NOT_EVERY_DRAW" if k >= 1 else "GAIN_POOLED_ONLY"
    return "NO_GAIN_SOME_DRAWS" if k >= 1 else "NO_GAIN"


def p1_pooled(per):
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
        boot = {"method": "scan bootstrap (final_common.cluster_boot; one draw ran)", "seed": cb.get("seed"),
                "nboot": cb.get("nboot")}
    r = F.p1_rule(list(sv.values()), min_mean=float("-inf"), boot=ci)
    m_ok = M is not None and M >= P1_MIN_MEAN
    return {"M": M, "ci": ci, "draw_means": dm, "scan_values": sv, "n_scans": len(sv), "scans_up": r["up"],
            "need_up": r["need_up"], "t_ci": r["t_ci"], "scan_level_mean": r["mean"], "M_ge_0_30": bool(m_ok),
            "pass": bool(r["pass"] and m_ok),
            "rule": "M (mean of the draw means) >= +0.30 AND scan t-CI lower > 0 AND bootstrap-CI lower > 0 AND up >= "
                    "ceil(2n/3)", "bootstrap": boot}


def p1_single(values):
    cb = F.cluster_boot(values)
    r = F.p1_rule(list(values.values()), boot=cb["ci"])
    return {"M": cb["mean"], "scan_boot_ci": cb["ci"], "t_ci": r["t_ci"], "scans_up": r["up"], "need_up": r["need_up"],
            "n": r["n"], "pass": r["pass"], "seed": cb.get("seed"), "nboot": cb.get("nboot")}


# ---------------------------------------------------------------------------------------------------- per-scan values
def vol_err_delta(R, arm, base, ids, organs=SOLID):
    """{cid: mean over the organs present in the reference of |e_arm| - |e_base|}, e = 100 (V_pred - V_ref) / V_ref with
    V_pred the arm's total predicted volume of the organ (mL, every voxel) and V_ref the reference volume."""
    import numpy as np
    out = {}
    for c in ids:
        f = R.get(c, {}).get(FRAME)
        if f is None:
            continue
        vals = []
        for o in organs:
            j = o - 1
            if not f["ref"]["present"][j]:
                continue
            vr = float(f["ref"]["ml"][j])
            ea = abs(100.0 * (float(f["arms"][arm]["ml"][j]) - vr) / vr)
            eb = abs(100.0 * (float(f["arms"][base]["ml"][j]) - vr) / vr)
            vals.append(ea - eb)
        if vals:
            out[c] = float(np.mean(vals))
    return out


def md(R, arm, base, ids):
    return L.macro_delta(R, FRAME, arm, base, ORGANS, ids)


def single_contrast(values, direction):
    est, reps, ids_ = scan_boot_reps(values)
    return {"estimate": est, "ci": pct_ci(reps), "p_two_sided": p_two_sided(reps), "n": len(ids_),
            "up": int(sum(values[c] > 0 for c in ids_)), "down": int(sum(values[c] < 0 for c in ids_)),
            "bootstrap": f"scan bootstrap ({NBOOT}, seed {SEED_SCAN})", "direction": direction}


def two_level_contrast(per, direction):
    est, reps = two_level_reps(per)
    return {"estimate": est, "ci": pct_ci(reps), "p_two_sided": p_two_sided(reps), "draws": sorted(per),
            "n_scans": len(set().union(*[set(v) for v in per.values()])) if per else 0,
            "bootstrap": f"crossed two-level ({NBOOT}, seed {SEED_TWO})", "direction": direction}


def two_family_contrast(per_a, per_b, direction, resample_b=True):
    est, reps, n = two_family_reps(per_a, per_b, resample_b=resample_b)
    how = ("draws of each family resampled independently" if resample_b else
           "scans and the first family's draws resampled; the second family's members held fixed")
    return {"estimate": est, "ci": pct_ci(reps), "p_two_sided": p_two_sided(reps), "families": [sorted(per_a), sorted(per_b)],
            "n_scans": n, "bootstrap": f"crossed two-level, {how} ({NBOOT}, seed {SEED_TWO})", "direction": direction,
            "second_family_fixed": not resample_b}


# ---------------------------------------------------------------------------------------------------- the analysis
def analyse(R, ids, have, not_run=None):
    """R {cid: {'NATIVE': frame record}}; ids = the included scans; have = arms DONE and scored; not_run {arm: reason}."""
    import numpy as np
    have = set(have)
    nr = dict(not_run or {})
    assert "F0" in have and "FULL_s1337" in have, "R1 requires F0 and FULL_s1337"
    draws = [d for d in sorted(K4) if K4[d] in have]
    out = {"scheme": "FLARE-T50 (AC.md GO-1), WORD A1 P1 rule", "frame": FRAME, "n_scans": len(ids), "ids": list(ids),
           "arms_scored": sorted(have), "arms_not_run": nr, "k4_draws_run": draws, "organs": F.FLARE13}
    per = {d: md(R, K4[d], "F0", ids) for d in draws}
    p1 = p1_pooled(per)
    out["R1"] = dict({k: p1[k] for k in ("M", "ci", "draw_means", "n_scans", "scans_up", "need_up", "t_ci",
                                         "scan_level_mean", "M_ge_0_30", "rule", "pass", "bootstrap")},
                     reading="REPLICATED" if p1["pass"] else "NOT REPLICATED")
    pd_, k = {}, 0
    for d in draws:
        pd_[d] = p1_single(per[d])
        k += int(pd_[d]["pass"])
    comb = combination(p1["pass"], k, len(draws))
    out["R1_per_draw"] = pd_
    out["R1_combination"] = {"combination": comb, "k_of_n_run": f"{k} of {len(draws)}"}
    sv = p1["scan_values"]
    up = sum(1 for v in sv.values() if v > 0)
    out["every_scan"] = {"k_improved": up, "n": len(sv), "every_scan": up == len(sv) and len(sv) > 0}
    po, cls = {}, {}
    for o in ORGANS:
        t = F.two_level({d: L.organ_delta(R, FRAME, K4[d], "F0", o, ids) for d in draws}) if len(draws) > 1 else None
        if t is None:
            cb = F.cluster_boot(L.organ_delta(R, FRAME, K4[draws[0]], "F0", o, ids))
            t = {"M": cb["mean"], "ci": cb["ci"], "n_scans": cb["n"]}
        name = F.FLARE13[o - 1]
        po[name] = {"M": t["M"], "ci": t["ci"], "n_scans": t["n_scans"], "ni": F.ni_class(t["ci"]), "harm": F.harm_class(t["ci"])}
        cls[name] = po[name]["ni"]
    out["organ_costs"] = dict(organ_reading(cls), per_organ=po, margin=-F.NI_MARGIN)

    S = {}
    if {"SSFT_B", "FULL_s1337"} <= have:
        S["S1"] = dict(single_contrast(md(R, "SSFT_B", "FULL_s1337", ids), "<0"), name="pull-back: SSFT_B - FULL_s1337")
    if {"SSFT_D", "SSFT_B"} <= have:
        S["S2"] = dict(single_contrast(md(R, "SSFT_D", "SSFT_B", ids), ">0"), name="label source: SSFT_D - SSFT_B")
    if {"STOCK", "FULL_s1337"} <= have:
        S["S3"] = dict(single_contrast(md(R, "FULL_s1337", "STOCK", ids), ">0"), name="gentle vs stock: FULL_s1337 - STOCK")
    d16 = [d for d in draws if K16[d] in have]
    if d16:
        S["S4"] = dict(two_level_contrast({d: md(R, K16[d], K4[d], ids) for d in d16}, ">0"),
                       name="more labels: K16_d - FULL_d (paired by draw)")
    bor = [c for c in sorted(BORROWED) if BORROWED[c] in have]
    if bor and draws:
        S["S5"] = dict(two_family_contrast({d: per[d] for d in draws}, {c: md(R, BORROWED[c], "F0", ids) for c in bor}, ">0",
                                           resample_b=False),
                       name="local vs borrowed: mean FLARE K=4 draws - mean of the three fixed borrowed models (each minus F0)")
    S["S6"] = dict(two_level_contrast({d: vol_err_delta(R, K4[d], "F0", ids) for d in draws}, "<0"),
                   name="solid-organ |volume error| change, FULL_d vs F0 (percentage points)")
    pv = {s: (S[s]["p_two_sided"] if s in S else None) for s in FAMILY}
    adj = holm(pv)
    for s in FAMILY:
        if s in S:
            S[s]["p_holm"] = adj[s]
            S[s]["reading"] = direction_reading(S[s]["estimate"], adj[s], S[s]["direction"])
        else:
            S[s] = {"reading": "NOT RUN", "reason": "an arm of this contrast was not run"}
    out["secondary_holm_family"] = {"m": sum(1 for s in FAMILY if pv[s] is not None), "alpha": ALPHA, "contrasts": S,
                                    "note": "Holm over the contrasts that ran; NOT RUN contrasts are listed and not "
                                            "counted in m"}
    if "STOCK" in have:
        st = p1_single(md(R, "STOCK", "F0", ids))
        out["stock_vs_F0"] = dict(st, reading="STOCK_GAINS" if st["pass"] else "STOCK_NO_GAIN")
    out["borrowed_vs_F0"] = {}
    for c in bor:
        cb = F.cluster_boot(md(R, BORROWED[c], "F0", ids))
        lo, hi = cb["ci"]
        out["borrowed_vs_F0"][c] = {"mean": cb["mean"], "ci": cb["ci"], "n": cb["n"],
                                    "reading": "HELPS" if (lo is not None and lo > 0) else
                                    ("HARMS" if (hi is not None and hi < 0) else "NEITHER"),
                                    "per_organ_mean_delta": {F.FLARE13[o - 1]: F.nanmean(list(
                                        L.organ_delta(R, FRAME, BORROWED[c], "F0", o, ids).values())) for o in ORGANS}}
    if "SCRATCH_NM1000" in have:
        cb = F.cluster_boot(md(R, "SCRATCH_NM1000", "F0", ids))
        out["scratch_vs_F0"] = {"mean": cb["mean"], "ci": cb["ci"], "n": cb["n"], "up": cb["up"], "down": cb["down"]}

    k1 = [x for x in sorted(K1) if K1[x] in have]
    if k1 and draws:
        perk1 = {x: md(R, K1[x], "F0", ids) for x in k1}
        t2 = two_family_contrast(perk1, {d: per[d] for d in draws}, "<0")
        t2["reading"] = direction_reading(t2["estimate"], t2["p_two_sided"], "<0")
        risk = {x: {"frac_below_F0": float(np.mean([v < 0 for v in perk1[x].values()])) if perk1[x] else None,
                    "n": len(perk1[x])} for x in k1}
        rep = any(v["frac_below_F0"] is not None and v["frac_below_F0"] >= RISK_FRAC for v in risk.values())
        out["tier2"] = {"K1_minus_K4": t2, "single_scan_risk": {"per_draw": risk, "threshold": RISK_FRAC,
                                                                "reading": "REPLICATED" if rep else "NOT REPLICATED"},
                        "k1_draws_run": k1}
    else:
        out["tier2"] = {"reading": "NOT RUN", "k1_draws_run": k1}

    order = sorted(sv.items(), key=lambda kv: (kv[1], kv[0]))
    if order:
        med = order[(len(order) - 1) // 2][0]
        loss = order[0][0]
        out["panels"] = {"median_gain_scan": med, "median_gain": sv[med], "largest_loss_scan": loss,
                         "largest_loss": sv[loss], "rule": "draw-averaged D: lower median; minimum"}
    out["descriptive_per_draw_contrasts"] = {d: L.contrast(R, FRAME, K4[d], "F0", ORGANS, ids) for d in draws}
    out["absolute_macro"] = {a: F.nanmean(list(L.macro_values(R, FRAME, a, ORGANS, ids).values())) for a in sorted(have)}
    out["claims_allowed"] = {
        "four labelled scans improve TS on never-opened FLARE scans (R1)": bool(p1["pass"]),
        "for each of the three draws": comb == "GAIN_EVERY_DRAW",
        "every scan improved": out["every_scan"]["every_scan"],
        "organs to name as a cost": out["organ_costs"]["inferior"],
        "organs inconclusive": out["organ_costs"]["inconclusive"],
        "external validation": False,
    }
    return out


def summary(res):
    S = res["secondary_holm_family"]["contrasts"]
    s = "; ".join(f"{k} {S[k].get('reading')}" for k in FAMILY)
    t2 = res.get("tier2", {})
    t2r = t2.get("reading") or t2.get("K1_minus_K4", {}).get("reading")
    m = res["R1"]["M"]
    return (f"R1 {res['R1']['reading']} M {'NA' if m is None else f'{m:+.2f}'}; {res['R1_combination']['combination']}; "
            f"every scan {res['every_scan']['k_improved']}/{res['every_scan']['n']}; organs "
            f"{res['organ_costs']['reading']}; {s}; tier 2 {t2r}")
