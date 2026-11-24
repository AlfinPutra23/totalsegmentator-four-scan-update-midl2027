"""vccor/m2 AMENDMENT A1 statistics (outputs/results/vccor/m2/prereg_m2_amendment_A1.json): the FLARE look's draw-pooled
contrasts counted per SCAN and given a two-level (scan x draw) bootstrap CI. Pure numpy / scipy; the only file access is
the amendment / prereg-log helpers at the end. Imported by score_m2.py and smoke_m2_A1.py; m2_common.py is NOT changed
(every running m2 job checks its hash).

Why: the three FLARE draws (seeds 1337 / 2025 / 3407) are evaluated on the same held18 scans (each draw's own labelled
scans excluded: 18 / 15 / 16 scans), so a held18 scan enters the draw-pooled (draw, scan) pairs up to three times and the
per-scan differences of the draws are correlated (TS-full audit_ft FULL - TS: r = 0.66 - 0.78). The v1 rule counted the
49 pairs and used the Welch t-CI of aft_stats.pooled, which treats the draws as independent strata.

Definitions (contrast values D[s, i] = per-scan value of draw s on scan i; nan when scan i is not an evaluation scan of
draw s, or the value is undefined):
  M            = mean over the draws s of the draw means m_s = mean_i D[s, i]      (UNCHANGED: == aft_stats.pooled 'mean')
  d_i          = mean over the draws in which scan i appears of D[s, i]           (each scan counted ONCE)
  scans_up     = #{i: d_i > 0} of the n_scans scans with at least one value
  boot2_ci95   = crossed two-level percentile bootstrap of M: each replicate draws n_scans scans with replacement AND
                 S draws with replacement, independently (the 'pigeonhole' bootstrap for a crossed scan x draw design,
                 Owen 2007, Ann. Appl. Stat. 1:246); M* = mean over the drawn draws (with multiplicity) of the mean of
                 D[s, i] over the drawn scans (with multiplicity) that are evaluation scans of draw s. A replicate in which
                 a drawn draw has no drawn evaluation scan is redrawn. S = 1: the scan bootstrap.
  descriptive  : scan_cluster_boot_ci95 (scans resampled, all draws kept), draw_level_t_ci95 (t_{S-1} on the S draw
                 means), scan_level_mean = mean_i d_i.
Rule A1 (binding for the FLARE look; M, sets, arms and metrics unchanged):
  PASS iff M >= min_mean AND boot2_ci95 lower bound > 0 AND scans_up >= ceil(min_fraction_positive x n_scans).
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

METHOD = "crossed_two_level_percentile_bootstrap"
MAX_REDRAW_ROUNDS = 1000


def matrix(per_draw, use=None):
    """per_draw {draw: {cid: value}} -> (draws, scans, D) with D[s, i] = value (nan where missing / non-finite).
    use: the draws to keep (default: all, sorted). Scans without any finite value are dropped."""
    draws = sorted(per_draw) if use is None else [d for d in use]
    allc = sorted(set().union(*[set(per_draw[s]) for s in draws])) if draws else []
    D = np.full((len(draws), len(allc)), np.nan)
    for j, s in enumerate(draws):
        for i, c in enumerate(allc):
            v = per_draw[s].get(c)
            if v is not None and np.isfinite(v):
                D[j, i] = float(v)
    keep = np.isfinite(D).any(0) if D.size else np.zeros(0, bool)
    return draws, [c for c, k in zip(allc, keep) if k], D[:, keep] if D.size else D


def scan_values(D):
    """d_i = mean of the finite D[s, i] over s; and the number of draws per scan."""
    ok = np.isfinite(D)
    n_d = ok.sum(0)
    tot = np.where(ok, D, 0.0).sum(0)
    return tot / np.maximum(n_d, 1), n_d


def _stat(D, cs, cd):
    """M* for replicate count matrices cs (B x n scans) and cd (B x S draws); bad[b] = a drawn draw has no drawn scan."""
    E = np.isfinite(D)
    Z = np.where(E, D, 0.0)
    num = cs @ Z.T
    den = cs @ E.T.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = num / den
    bad = ((den == 0) & (cd > 0)).any(1)
    m = np.where(cd > 0, np.nan_to_num(m, nan=0.0), 0.0)
    return (cd * m).sum(1) / cd.sum(1), bad


def boot2(D, n_boot, seed, level=0.95, resample_draws=True):
    """Crossed two-level percentile bootstrap CI of M (resample_draws=False: the scan-cluster bootstrap)."""
    S, n = D.shape
    if S == 0 or n == 0:
        return [None, None], 0
    rng = np.random.default_rng(seed)
    pn, ps = np.full(n, 1.0 / n), np.full(S, 1.0 / S)
    cs = rng.multinomial(n, pn, size=n_boot).astype(float)
    cd = rng.multinomial(S, ps, size=n_boot).astype(float) if resample_draws else np.ones((n_boot, S))
    val, bad = _stat(D, cs, cd)
    redrawn, rounds = 0, 0
    while bad.any():
        rounds += 1
        if rounds > MAX_REDRAW_ROUNDS:
            raise RuntimeError("two-level bootstrap: replicates without an evaluation scan keep recurring")
        idx = np.flatnonzero(bad)
        redrawn += len(idx)
        cs[idx] = rng.multinomial(n, pn, size=len(idx))
        if resample_draws:
            cd[idx] = rng.multinomial(S, ps, size=len(idx))
        v2, b2 = _stat(D, cs[idx], cd[idx])
        val[idx] = v2
        bad[:] = False
        bad[idx] = b2
    a = 100.0 * (1.0 - level) / 2.0
    lo, hi = np.percentile(val, [a, 100.0 - a])
    return [float(lo), float(hi)], int(redrawn)


def draw_level_t(D, level=0.95):
    S = D.shape[0]
    dm = np.array([np.nanmean(D[j]) for j in range(S)]) if S else np.zeros(0)
    if S < 2:
        return [None, None]
    q = stats.t.ppf(0.5 + level / 2.0, S - 1)
    se = dm.std(ddof=1) / math.sqrt(S)
    return [float(dm.mean() - q * se), float(dm.mean() + q * se)]


def clustered(per_draw, use, n_boot, seed, level=0.95, keep_per_scan=True):
    """The A1 summary of one draw-pooled contrast. per_draw {draw: {cid: value}}; use = the draws aft_stats.pooled used."""
    draws, scans, D = matrix(per_draw, use)
    if not draws or not scans:
        return {"method": METHOD, "draws": draws, "n_scans": 0, "n_pairs": 0, "M_mean_of_draw_means": None,
                "scan_level_mean": None, "scans_up": 0, "scans_down": 0, "boot2_ci95": [None, None],
                "scan_cluster_boot_ci95": [None, None], "draw_level_t_ci95": [None, None], "n_boot": n_boot, "seed": seed}
    v, n_d = scan_values(D)
    dm = np.array([np.nanmean(D[j]) if np.isfinite(D[j]).any() else np.nan for j in range(len(draws))])
    ci2, red2 = boot2(D, n_boot, seed, level, resample_draws=True)
    ci1, _ = boot2(D, n_boot, seed, level, resample_draws=False)
    out = {"method": METHOD, "draws": draws, "n_scans": len(scans), "n_pairs": int(np.isfinite(D).sum()),
           "M_mean_of_draw_means": float(np.nanmean(dm)), "scan_level_mean": float(v.mean()),
           "scans_up": int((v > 0).sum()), "scans_down": int((v < 0).sum()), "scans_zero": int((v == 0).sum()),
           "boot2_ci95": ci2, "boot2_redrawn_replicates": red2, "scan_cluster_boot_ci95": ci1,
           "draw_level_t_ci95": draw_level_t(D, level), "n_boot": int(n_boot), "seed": int(seed), "level": level}
    if keep_per_scan:
        out["per_scan_mean_over_draws"] = dict(zip(scans, [float(x) for x in v]))
        out["draws_per_scan"] = dict(zip(scans, [int(x) for x in n_d]))
    return out


def needed(frac, n):
    """== score_m2.rule_check's count threshold: ceil(frac x n) with the same 1e-9 guard."""
    return int(math.ceil(frac * n - 1e-9))


def rule_a1(M, cl, rule):
    """Amendment A1 rule on the pre-registered M and the clustered summary cl of the H1 Dice contrast."""
    n, up = cl["n_scans"], cl["scans_up"]
    need = needed(rule["min_fraction_positive"], n)
    ci = cl["boot2_ci95"]
    crit = {"M_ge_min": bool(M is not None and np.isfinite(M) and M >= rule["min_mean"]),
            "boot2_ci_lower_gt_0": bool(ci and ci[0] is not None and ci[0] > 0),
            "positive_scans_ge_two_thirds": bool(n > 0 and up >= need)}
    return {"rule": "amendment A1 (scan-level count, two-level bootstrap CI)", "M": M, "boot2_ci95": ci,
            "positive_scans": up, "n_scans": n, "positive_needed": need, "n_pairs": cl["n_pairs"], "criteria": crit,
            "verdict": "PASS" if all(crit.values()) else "FAIL"}


# --------------------------------------------------------------------------------------------- amendment + log helpers
def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def amendment_sha_logged(log_path, amend_rel):
    """The last 'AMENDMENT sha256 <hex> <repo-relative path> ...' line of the block's prereg log for this file (or None)."""
    h = None
    if Path(log_path).is_file():
        for line in open(log_path):
            p = line.split()
            if len(p) >= 6 and p[2] == "AMENDMENT" and p[3] == "sha256" and len(p[4]) == 64 and p[5] == amend_rel:
                h = p[4]
    return h


def load_rule(amendment):
    """The machine-readable A1 block of the amendment, validated (the scorer reads its numbers from here)."""
    r = amendment["rule_A1"]
    assert r["method"] == METHOD, r["method"]
    assert r["unit"] == "scan" and r["per_scan_value"] == "mean over the draws in which the scan is an evaluation scan"
    assert r["applies_to"] == {"look": "flare", "contrasts": "every draw-pooled contrast (binding: H1 'MF - M0')"}
    assert 0 < float(r["min_mean"]) and 0 < float(r["min_fraction_positive"]) <= 1 and float(r["level"]) == 0.95
    assert int(r["n_boot"]) >= 1000 and isinstance(r["seed"], int)
    return {"min_mean": float(r["min_mean"]), "min_fraction_positive": float(r["min_fraction_positive"]),
            "n_boot": int(r["n_boot"]), "seed": int(r["seed"]), "level": float(r["level"]), "method": r["method"]}


def read_amendment(path):
    return json.load(open(path))
