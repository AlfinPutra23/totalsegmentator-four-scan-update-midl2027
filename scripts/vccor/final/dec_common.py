"""vccor/final plan E (ERROR DECOMPOSITION): the FIXED definitions of systematic vs random error per organ and cohort,
volume-based and boundary-based (signed surface distance), for TotalSegmentator zero-shot (F0) vs four-scan-aligned (F1b
recipe, FULL_K4) maps. POST HOC but DEFINITION-FIXED: every set below was opened by earlier looks; this module and
plan_decomposition.json are hashed (PREREG line) before any number under these definitions is computed.

Per (scan i, organ o, arm a), conventions v1 frames and presence (>= 1 mL on both sides):
  volume     defined iff the reference is present (R): e = V_pred - V_ref (mL; V_pred = 0 when the prediction is absent) and
             r = 100 e / V_ref (%)
  boundary   defined iff R and P (both present). Surfaces = mask minus its 6-connected erosion, surface-voxel centres in mm
             (== the v1 HD95 convention). Signed distances, pooled from both surfaces:
               prediction surface point p: |d| = distance to the nearest reference surface point; sign + iff p lies OUTSIDE the
                 reference mask
               reference surface point q: |d| = distance to the nearest prediction surface point; sign + iff q lies INSIDE the
                 prediction mask
             (+ = the prediction extends beyond the reference = over-segmentation). Per scan: n_i points, m_i = mean signed
             distance, q_i = mean squared signed distance, s2_i = q_i - m_i^2.
Cohort-level decomposition per (cohort, organ, arm) over the scans that define it, every scan weighted equally:
  volume     bias = mean_i e_i; var = mean_i (e_i - bias)^2; MSE = mean_i e_i^2 = bias^2 + var; SYSTEMATIC SHARE
             sigma_V = bias^2 / MSE, RANDOM SHARE = var / MSE (absolute mL and relative %, separately)
  boundary   M = mean_i m_i (cohort offset, mm); between = mean_i (m_i - M)^2; within = mean_i s2_i;
             total = M^2 + between + within (= mean_i q_i); SYSTEMATIC SHARE sigma_B = M^2 / total; scan-offset share
             (M^2 + between) / total; within-scan share within / total
Alignment (F0 -> FT) per (cohort, organ): removed share of F0's error = (err0 - err1) / err0, split into its systematic part
(volume: (bias0^2 - bias1^2) / MSE0; boundary: (M0^2 - M1^2) / total0) and its random part (the rest).
Summaries: per cohort the MEAN and the MEDIAN over organs of every share (F0, FT, removed); 95% patient-cluster bootstrap CIs
(10,000, seed 20261022; AMOS patients from the metadata key; FLARE / MACT one cluster per scan) of the organ-mean and organ-median
shares and of their FT - F0 differences.
PRIMARY summary (the one number that replaces the unscripted claim X1 'systematic share of squared volume error, TS 0.48 -> F1b
0.11 on held18, median over organs'): the MEDIAN over organs of sigma_V (absolute mL) for F0 and FT on FLARE held18 (draw 1337),
with its CI; every other cohort, summary and the boundary version are secondary. No import-time side effects beyond sys.path.
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import final_common as F  # noqa: E402

PLAN = "decomposition"
DIR = F.PLAN_DIR[PLAN]
VC = F.VC


def sets():
    """cohort -> (label space, ids, arms {arm: raw-map dir}, excluded {arm: [ids]})."""
    import tsft_full_common as Q
    import xdata_common as X
    import a240_common as A240
    return {
        "FLARE-held18": ("flare", list(Q.HELD18), {"F0": VC / "tsft_full/pred/F0/raw", "FT": VC / "tsft_full/pred/F1b/raw",
                                                  "FT@2025": VC / "audit_ft/pred/s2025_FULL/raw", "FT@3407": VC / "audit_ft/pred/s3407_FULL/raw"},
                         {"FT@2025": ["0007", "0026", "0032", "0029"], "FT@3407": ["0018", "0030", "0033", "0004"]}),
        "AMOS-56": ("amos", list(X.eval_ids("amos", 1337)), {"F0": VC / "xdata/pred/amos_F0/raw",
                                                             "FT": VC / "xdata/pred/amos_s1337_K4_FULL_F1b/raw"}, {}),
        "AMOS-240": ("amos", list(A240.ids240()), {"F0": VC / "amos240/pred/F0/raw", "FT": VC / "amos240/pred/FULL_K4/raw"}, {}),
        "MACT-81": ("mact", list(X.eval_ids("mact")), {"F0": VC / "xdata/pred/mact_F0/raw",
                                                       "FT": VC / "xdata/pred/mact_s1337_K4_FULL_F1b/raw",
                                                       **{f"K1@{c}": VC / f"xdata/pred/mact_s1337_d1_{c}_FULL_F1b/raw" for c in MACT_K1},
                                                       "FLARE_F1b": VC / "xdata/g1/pred/X1/raw"}, {}),
    }


MACT_K1 = {"0037": "tcia", "0041": "tcia", "0075": "btcv", "0049": "btcv"}     # the four single-scan (K = 1) draws and their source


EXPOSURE = {   # E3 (descriptive): zero-shot model comparison on benchmarks whose labels VISTA3D was trained on
    "FLARE-held18": {"V0": (VC / "vista/pred/V0_flare/raw", "vista"), "M0": (VC / "m2/pred/M0_flare/raw", "mrseg")},
    "AMOS-56": {"V0": (VC / "vista/pred/V0_amos/raw", "vista"), "M0": (VC / "m2/pred/M0_amos/raw", "mrseg")},
}


def signed_surface(Y, G, o, sp, sl_y, sl_g):
    """(n, mean, mean square) of the pooled signed surface distances of organ o (both masks non-empty); sl_y / sl_g = the
    organ's find_objects slices in Y / G."""
    import numpy as np
    from scipy.spatial import cKDTree
    sp = np.asarray(sp, float)
    py = F._surface_points_lab(Y, o, sl_y)
    pg = F._surface_points_lab(G, o, sl_g)
    ty, tg = cKDTree(py * sp), cKDTree(pg * sp)
    d_p = tg.query(py * sp, k=1)[0]
    d_q = ty.query(pg * sp, k=1)[0]
    s_p = np.where(G[tuple(py.T)] == o, -1.0, 1.0)            # prediction point outside the reference -> +
    s_q = np.where(Y[tuple(pg.T)] == o, 1.0, -1.0)            # reference point inside the prediction -> +
    d = np.r_[d_p * s_p, d_q * s_q]
    return int(len(d)), float(d.mean()), float((d * d).mean())


def scan_record(Y, G, k, sp, vox_ml):
    """Per organ: volumes, presence, signed-surface summary (TP only)."""
    import numpy as np
    from scipy import ndimage
    H = F.joint_counts_chunked(Y, G, k)
    g, y = H.sum(1)[1:].astype(np.int64), H.sum(0)[1:].astype(np.int64)
    sy, sg = ndimage.find_objects(Y, max_label=k), ndimage.find_objects(G, max_label=k)
    out = []
    for j in range(k):
        o = j + 1
        vr, vp = g[j] * vox_ml, y[j] * vox_ml
        R, P = vr >= F.FLOOR_ML, vp >= F.FLOOR_ML
        rec = {"R": bool(R), "P": bool(P), "v_ref": float(vr) if R else 0.0, "v_pred": float(vp) if P else 0.0}
        if R and P:
            n, m, q = signed_surface(Y, G, o, sp, sy[j], sg[j])
            rec.update(n=n, m=m, q=q)
        out.append(rec)
    return out


def decompose(recs, organ, weights=None):
    """recs {cid: [per-organ records]} -> the volume / boundary decomposition of one organ (weights: bootstrap multiplicities)."""
    import numpy as np
    j = organ - 1
    cids = sorted(recs)
    w = np.array([1.0 if weights is None else weights.get(c, 0.0) for c in cids])
    vr = np.array([recs[c][j]["v_ref"] for c in cids])
    vp = np.array([recs[c][j]["v_pred"] for c in cids])
    R = np.array([recs[c][j]["R"] for c in cids])
    out = {}
    for name, e in (("abs", vp - vr), ("rel", np.where(vr > 0, 100.0 * (vp - vr) / np.where(vr > 0, vr, 1.0), 0.0))):
        ww = w * R
        if ww.sum() <= 0:
            out[f"vol_{name}"] = None
            continue
        bias = float((ww * e).sum() / ww.sum())
        mse = float((ww * e * e).sum() / ww.sum())
        var = mse - bias * bias
        out[f"vol_{name}"] = {"bias": bias, "var": max(var, 0.0), "mse": mse, "sys_share": (bias * bias / mse) if mse > 0 else None,
                              "n": int(R.sum())}
    tp = np.array([("m" in recs[c][j]) for c in cids])
    ww = w * tp
    if ww.sum() > 0:
        m = np.array([recs[c][j].get("m", 0.0) for c in cids])
        q = np.array([recs[c][j].get("q", 0.0) for c in cids])
        M = float((ww * m).sum() / ww.sum())
        between = float((ww * (m - M) ** 2).sum() / ww.sum())
        within = float((ww * np.maximum(q - m * m, 0.0)).sum() / ww.sum())
        total = M * M + between + within
        out["bnd"] = {"offset_mm": M, "between": between, "within": within, "total": total,
                      "sys_share": (M * M / total) if total > 0 else None,
                      "scan_offset_share": ((M * M + between) / total) if total > 0 else None,
                      "within_share": (within / total) if total > 0 else None, "n": int(tp.sum())}
    else:
        out["bnd"] = None
    return out


def removed(d0, d1):
    """F0 -> FT: removed share of F0's error and its systematic / random split (volume rel + abs, boundary)."""
    out = {}
    for key in ("vol_abs", "vol_rel"):
        a, b = d0.get(key), d1.get(key)
        if a and b and a["mse"] > 0:
            out[key] = {"removed": (a["mse"] - b["mse"]) / a["mse"], "removed_systematic": (a["bias"] ** 2 - b["bias"] ** 2) / a["mse"],
                        "removed_random": (a["var"] - b["var"]) / a["mse"]}
    a, b = d0.get("bnd"), d1.get("bnd")
    if a and b and a["total"] > 0:
        out["bnd"] = {"removed": (a["total"] - b["total"]) / a["total"],
                      "removed_systematic": (a["offset_mm"] ** 2 - b["offset_mm"] ** 2) / a["total"],
                      "removed_between": (a["between"] - b["between"]) / a["total"],
                      "removed_within": (a["within"] - b["within"]) / a["total"]}
    return out


SUMMARY_KEYS = [("vol_rel", "sys_share"), ("vol_abs", "sys_share"), ("bnd", "sys_share"), ("bnd", "scan_offset_share")]


def organ_means(recs_by_arm, organs, weights=None):
    """{arm: {key: mean over organs of the share}} + removed shares (FT vs F0)."""
    import numpy as np
    res = {}
    D = {a: {o: decompose(r, o, weights) for o in organs} for a, r in recs_by_arm.items()}
    for a in D:
        res[a] = {}
        for k1, k2 in SUMMARY_KEYS:
            v = [D[a][o][k1][k2] for o in organs if D[a][o].get(k1) and D[a][o][k1].get(k2) is not None]
            res[a][f"{k1}.{k2}"] = float(np.mean(v)) if v else None
    if "F0" in D and "FT" in D:
        rm = {o: removed(D["F0"][o], D["FT"][o]) for o in organs}
        res["removed"] = {}
        for k1 in ("vol_rel", "vol_abs", "bnd"):
            for k2 in ("removed", "removed_systematic"):
                v = [rm[o][k1][k2] for o in organs if k1 in rm[o]]
                res["removed"][f"{k1}.{k2}"] = float(np.mean(v)) if v else None
    return res, D


# ---------------------------------------------------------------------------------------------------- E4: convention vs domain
DOMAIN_DEF = """E4 CONVENTION vs DOMAIN (definition fixed with plan E; added 2026-10-22 after the AMOS-240 cell finding of paper_tables_v2
item2 section 7: FULL_K4 - F0 = +1.89 center/Philips, +1.45 center/Toshiba, +0.39 people/Siemens; site interaction -1.28).
Within ONE annotation protocol with several scanner / site cells c (AMOS22: center/Philips/Brilliance16, center/TOSHIBA/Aquilion
ONE, people/SIEMENS/SOMATOM Force; MACT: the TCIA Pancreas-CT and BTCV sources, both re-annotated by Gibson under one protocol):
  local model L_c  = aligned on labelled scans of the SAME site as c        (AMOS: FULL_K4@1337 for center cells, XS for people;
                                                                             MACT K = 1: the d1 models of the same source)
  cross model X_c  = aligned on labelled scans of the OTHER site, same protocol (AMOS: XS for center cells, FULL_K4@1337 for
                                                                             people; MACT K = 1: the d1 models of the other source)
  generic model G  = aligned on ANOTHER protocol (FLARE F1b)
  per cell, with gains g(M) = mean over the cell's scans of [penalised macro(M) - macro(F0)] (paired per scan):
    total local gain   T_c = g(L_c)
    DOMAIN part        D_c = g(L_c) - g(X_c)        does not transfer across scanners within the protocol
    CONVENTION part    P_c = g(X_c) - g(G)          transfers across scanners within the protocol, not across protocols
    GENERIC part       Gm_c = g(G)                  transfers across protocols (a TS bias every reference standard rejects)
    PROTOCOL-SPECIFIC  PS_c = g(L_c) - g(G) = D_c + P_c   (the only split available where no within-protocol cross model exists:
                                                    the transfer-matrix targets FLARE / MACT-81 with the other protocols' models)
    T_c = D_c + P_c + Gm_c exactly (every component of a cell is computed on the SAME scans: those with every arm of the cell;
    per organ: those of them with the organ in the reference); shares D / T, P / T, Gm / T when |T| >= 0.25 Dice (else not
    reported)
    HEADROOM h_c = mean F0 macro of the cell; T_c / (100 - h_c) reported next to T_c
  the site x training-site interaction (plan B's B-H2) = D_people + D_center.
Several draws of one model type (MACT K = 1: two per source) are averaged per scan before the differences. CIs: patient-cluster
bootstrap within the cell (10,000, seed 20261022); per organ the same on the organ's Dice (scans with the organ in the reference).
MACT caveat: L / X are K = 1 models and G is a K = 4 model: D is budget-matched, P and Gm are reported but their sum with D mixes
budgets (descriptive). WORD (single centre, single SIEMENS scanner per its paper) has no within-protocol scanner split."""


def _mean_macro(R, frame, arms, organs, ids):
    import numpy as np
    out = {}
    for c in ids:
        f = R.get(c, {}).get(frame)
        if f is None:
            continue
        vals = [F.macro(f["arms"][a], f["ref"], organs, True) for a in arms if a in f["arms"]]
        if len(vals) == len(arms) and all(np.isfinite(vals)):
            out[c] = float(np.mean(vals))
    return out


def _mean_organ(R, frame, arms, o, ids):
    import numpy as np
    out = {}
    for c in ids:
        f = R.get(c, {}).get(frame)
        if f is None or not f["ref"]["present"][o - 1]:
            continue
        vals = [F.organ_dice(f["arms"][a], f["ref"], o) for a in arms if a in f["arms"]]
        if len(vals) == len(arms) and all(np.isfinite(vals)):
            out[c] = float(np.mean(vals))
    return out


def domain_cells(R, frame, organs, cells, local_of, cross_of, generic, clusters=None, base="F0", per_organ=True):
    """E4 per cell: R {cid: frames}; cells {cell: [cids]}; local_of / cross_of: cell -> [arms] (averaged per scan) or None;
    generic: [arms] or None. Returns {cell: components with CIs}. Pure function (smoke-tested)."""
    import numpy as np

    def comp(vals_a, vals_b, ids):
        d = {c: vals_a[c] - vals_b[c] for c in ids if c in vals_a and c in vals_b}
        return F.cluster_boot(d, clusters)

    def common(vals):
        ks = None
        for v in vals:
            if v is not None:
                ks = set(v) if ks is None else ks & set(v)
        return sorted(ks or [])

    def mean_of(d, ids):
        return float(np.mean([d[c] for c in ids])) if ids else None

    out = {}
    for cell, ids in cells.items():
        L_, X_ = local_of(cell), cross_of(cell)
        m0 = _mean_macro(R, frame, [base], organs, ids)
        mL = _mean_macro(R, frame, L_, organs, ids) if L_ else None
        mX = _mean_macro(R, frame, X_, organs, ids) if X_ else None
        mG = _mean_macro(R, frame, generic, organs, ids) if generic else None
        use = common([m0, mL, mX, mG])
        rec = {"n": len(use), "n_dropped (an arm missing)": len(ids) - len(use), "F0_level": mean_of(m0, use),
               "local": L_, "cross": X_, "generic": generic}
        for key, a_, b_ in (("T_local_gain", mL, m0), ("cross_gain", mX, m0), ("D_domain", mL, mX), ("P_convention", mX, mG),
                            ("Gm_generic", mG, m0), ("PS_protocol_specific", mL, mG)):
            if a_ is not None and b_ is not None:
                rec[key] = comp(a_, b_, use)
        T = rec.get("T_local_gain", {}).get("mean")
        if T is not None and rec["F0_level"] is not None:
            rec["T_over_headroom"] = T / (100.0 - rec["F0_level"])
            if abs(T) >= 0.25:
                rec["shares"] = {k: rec[k]["mean"] / T for k in ("D_domain", "P_convention", "Gm_generic", "PS_protocol_specific")
                                 if k in rec}
        if per_organ:
            po = {}
            for o in organs:
                o0 = _mean_organ(R, frame, [base], o, use)
                oL = _mean_organ(R, frame, L_, o, use) if L_ else None
                oX = _mean_organ(R, frame, X_, o, use) if X_ else None
                oG = _mean_organ(R, frame, generic, o, use) if generic else None
                uo = common([o0, oL, oX, oG])
                r = {"n": len(uo), "F0_dice": mean_of(o0, uo)}
                for key, a_, b_ in (("T", oL, o0), ("cross", oX, o0), ("D", oL, oX), ("P", oX, oG), ("Gm", oG, o0), ("PS", oL, oG)):
                    if a_ is not None and b_ is not None:
                        if key in ("D", "P"):
                            dd = comp(a_, b_, uo)
                            r[key], r[key + "_ci"] = dd["mean"], dd["ci"]
                        else:
                            r[key] = mean_of({c: a_[c] - b_[c] for c in uo}, uo)
                po[str(o)] = r
            rec["per_organ"] = po
        rec["reading"] = reading(rec)
        out[cell] = rec
    return out


def site_cells(cells):
    """The scanner cells pooled by site (the training-site level of the roles): {'center': [...], 'people': [...]}."""
    out = {}
    for cell, ids in cells.items():
        out.setdefault(cell.split(" / ")[0], []).extend(ids)
    return {k: sorted(v) for k, v in out.items()}


def reading(rec):
    """Pre-specified reading of one cell's components (95% CIs): PRESENT iff the CI lower bound > 0, NEGATIVE iff the upper
    bound < 0, NOT DETECTED otherwise; the cell is CONVENTION-DOMINATED iff P is PRESENT and P / T >= 0.5, DOMAIN-DOMINATED iff
    D is PRESENT and D / T >= 0.5, GENERIC-DOMINATED iff Gm is PRESENT and Gm / T >= 0.5 (checked in this order; the shares sum
    to 1 so at most one can reach 0.5 unless another is negative), otherwise MIXED; T < 0.25 Dice: NO GAIN (no reading)."""
    def cls(k):
        ci = rec.get(k, {}).get("ci", [None, None])
        if ci[0] is None:
            return None
        return "PRESENT" if ci[0] > 0 else ("NEGATIVE" if ci[1] < 0 else "NOT DETECTED")
    out = {k: cls(k) for k in ("T_local_gain", "D_domain", "P_convention", "Gm_generic", "PS_protocol_specific") if k in rec}
    T = rec.get("T_local_gain", {}).get("mean")
    if T is None:
        out["cell"] = None
        return out
    if T < 0.25:
        out["cell"] = "NO GAIN (T < 0.25)"
        return out
    sh = rec["shares"]
    for k, lab in (("P_convention", "CONVENTION-DOMINATED"), ("D_domain", "DOMAIN-DOMINATED"), ("Gm_generic", "GENERIC-DOMINATED")):
        if out.get(k) == "PRESENT" and sh.get(k, 0) >= 0.5:
            out["cell"] = lab
            break
    else:
        out["cell"] = "MIXED"
    return out


def amos_cells(ids):
    """AMOS-240 scans by (Site, Manufacturer, Manufacturer's Model Name) from AMOS's metadata (== conventions amos_strata.cells)."""
    out = {}
    for c in ids:
        s, m, mm = F.amos_site(c)
        out.setdefault(f"{s} / {m} / {mm}", []).append(c)
    return out


def amos_roles(cell, local_center="FULL_K4", local_people="XS"):
    """(local arms, cross arms) of an AMOS cell: center cells are local to the center-trained model, the people cell to XS."""
    if cell.startswith("people"):
        return [local_people], [local_center]
    return [local_center], [local_people]
