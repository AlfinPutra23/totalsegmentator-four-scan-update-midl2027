"""vccor/final plan C, amendments word A1, A1.1 and A1.2: UNIT TESTS of the TS-only scorer, the gated runner, the rehearsal and
the A1.2 STOCK wrapper and CPU arms, on SYNTHETIC data only.

Every fixture (records, images, labels, the labelsTs zip, prediction maps, DONE records, final_prereg.log, the OSF gate
file, decisions.json, q8 markers, the GPU-1 lock file, code files, training labels and TS parts of the label-map rules,
the STOCK smoke images and labels) is fabricated under the scratch folder (--scratch, which must be under /tmp). No test
opens, lists or stats the real WORD archives or any real label, no test creates, opens or lists the real look directory,
and no test trains, predicts or uses a GPU (the runner's jobs are replaced by tiny CPU stand-ins; the GPU-1 memory query is
replaced by a function; the GPU-1 lock is a scratch file; the STOCK wrapper is run with --load_only, which executes the
frozen train_final.py / predict_final.py module code only, or with --smoke, which builds a STOCK dataset from synthetic
volumes under /tmp: nnU-Net fingerprint, plans move and preprocessing, no training). The only real files a test reads are
the frozen code, scripts/queue_v8.py (imported in dry mode, which writes nothing), outputs/logs/q8/decisions.json, WORD's
split record C_word/download.json (scan ids) and TotalSegmentator's part-291 plans. T11 checks that the real OSF gate file
and the real look directory do NOT exist (and refuses to run at all if either exists, FIX-1), then runs the two scripts in
--check mode only (never bare), and checks that nothing real changed; T27 applies the same precondition.

  T1  analyse_a1 scenario A (30 scans, 3 draws): P1, P1 organ costs, P2, S, colon_rectum vs hand-computed values;
      P1's two-level CI and a P2 scan-bootstrap CI re-computed by an independent implementation of the registered schemes
  T2  analyse_a1 scenario B (29 scans, one draw): the ceil(2/3 n) = 20 of 29 count; P2 NOT EVALUABLE; GAIN_EVERY_RUN_DRAW
  T3  analyse_a1 scenario C (two draws ran): one pairwise difference; draw dependence detected / not detected
  T4  outcome combinations and the organ reading (A1's tables with A1.1's labels)
  T5  end-to-end look on a synthetic WORD (6 test scans, duplicate pair 0017 ~ 0131, synthetic labelsTs zip; A1.2 arms
      declared NOT RUN)
  T6  look aborted after the marker when the zip differs from the pinned sha256 (GT_ABORTED; no GT_CRASHED; no continue)
  T7  scorer gates: OSF file / URL / log line / v1 and v2 zips / order, second look, missing draw, required arm, NOT RUN,
      cutoff, zip size, an A1.2 arm neither DONE nor declared
  T8  duplicate rule with different images: both kept
  T9  runner on stand-in jobs: A1.2 order (22 jobs: A1's seven, then the six CPU arm jobs, then the nine STOCK jobs),
      GPU-1 / CPU environments, rerun once, double failure, required failure, cutoff, refusals
  T10 queue_v8.py: the runner's seven A1 command lines == queue_v8's pfinC rows; the 28 A1 rows absent from the live table;
      no queue_v8 row (live or full table) runs an A1.2 job or names an A1.2 script
  T11 real mode today, --check only: both scripts refuse at the OSF gate and write nothing
  T12 code style: no code comment on the same line as code (every .py and .service file)
  T13 one-draw P1 bootstrap == A1's scan bootstrap (cluster_boot, seed 20261022), checked independently
  T14 M with unequal scan sets: the +0.30 bar uses the mean of the draw means
  T15 GT_CRASHED and --continue: crash after the marker, continuation reuses completed scans and gives the uninterrupted
      results; continuation refused without a marker, after results, with a RESULTS line, with maps.json or the marker
      changed, and with changed code
  T16 a killed worker and a stuck worker do not hang the look (BrokenProcessPool / per-result timeout -> GT_CRASHED)
  T17 runner: the cutoff re-checked after the idle and lock waits; a DONE job is never re-run; hand stop + reason
  T18 the shared GPU-1 flock: held during the job, waited for, timed out, missing file; file content unchanged
  T19 queue_v8_pids matches only python processes executing queue_v8.py
  T20 T11's refusal when the OSF gate file or the look directory exists
  T21 the rehearsal: refuses real paths, its audit guard blocks protected paths in workers, end-to-end on synthetic WORD
  T22 A1.2 secondary contrasts S1, S1b, S2, S3, S4 and Q vs hand-computed values (and independent bootstraps); P1 / P2 / the
      outcome label identical with and without the A1.2 arms; every reading; one draw; NOT RUN; Q only under P1 PASS
  T23 the label-map rules (plan_ontology_oracle rule_from_labelled4 + A1.2's conflict rule) on synthetic training labels:
      hand-computed mean shares, the 1 % truncation, the 20-voxel presence rule, ontology classes and background not
      eligible, conflicts, ties; the manifest-checked inputs (right adrenal 17 -> 12, TS's part merge order); the map
  T24 MERGE_N == the frozen v2_common.merge_maps; non-target classes equal F0; rule-B voxel counts; DONE inputs; the
      scorer's input-consistency check
  T25 the runner's A1.2 rules: the six CPU arm jobs before the nine STOCK jobs and STOCK after every A1 job, the STOCK chain
      guard before the cutoff, STOCK failure / cutoff -> NOT RUN without stopping the runner, MERGE_N NOT RUN when its F1b
      draw is NOT RUN, CPU jobs take no GPU-1 lock
  T26 the v5 gate refuses the v1, v2, v3 and v4 zips (name or sha256, gate file or DEPOSIT line) and accepts v5
  T27 the STOCK wrapper: install() adds the three runs without changing F1b; train_final.py / predict_final.py module code
      (--load_only, CPU) take the registered STOCK settings; verify_datasets checks the STOCK prep record; nothing written
  T28 the STOCK prep (--smoke, synthetic volumes): labels in WORD's 15 analysis ids, dataset.json, image symlinks, TS part
      291's plans moved (TS291StockPlans), 4 preprocessed cases, DONE record
  T29 end-to-end look with all 13 arms on synthetic WORD (MERGE_N / LABELMAP built by word_a1_cpu_arms): hand-computed A1.2
      contrasts; P1 / P2 identical to T5; crash + --continue with all arms; the rehearsal with all arms and its guard armed
  T30 a secondary-analysis error never blocks the primary results: with analyse_a1_2 made to raise (and, separately, to
      return a block results.json cannot hold), the look still writes results.json with P1, P2 and the outcome label
      (identical to the look without the error), its A1_2 block holds the error, final_prereg.log gets a SECONDARY_ERROR
      line and then the RESULTS line, no GT_CRASHED; a signal-type BaseException still propagates; the rehearsal reports
      the error and still passes its re-analysis check

usage:  taskset -c 8-15 .venv/bin/python -B scripts/vccor/final/word_a1/test_word_a1.py --scratch <dir under /tmp> --out <json>

Version of 2026-11-05 (A1, A1.1) and 2026-11-06 (A1.2 and its
v5 revision).
"""
import argparse
import contextlib
import fcntl
import gzip
import io
import json
import math
import multiprocessing
import os
import shutil
import signal
import subprocess
import sys
import time
import tokenize
import traceback
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import word_a1_common as A
import final_common as F
import word_common as W
import word_score as WS
import word_a1_score as S
import word_a1_run as RUN
import word_a1_rehearse as RH
import word_a1_stock as ST
import word_a1_cpu_arms as CA

SCR = None
FAKE_V5 = "ab" * 32
V5_REL = "outputs/results/vccor/osf_deposit_2026-11-06_word_v5.zip"
URL = "https://osf.io/abc12"
TOL = 1e-6
K = W.N_ORG
PARTIAL = 200.0 * 270 / (300 + 270)
REAL_W = {k: getattr(W, k) for k in ("DIR", "LOOK", "LABELS_TS_ZIP", "IMAGES", "PREP", "DOWNLOAD")}


def close(a, b, tol=TOL):
    return a is not None and b is not None and abs(float(a) - float(b)) <= tol


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def expect_refuse(fn, needle, code=None):
    try:
        fn()
    except A.Refuse as e:
        check(needle in e.msg, f"refusal text {e.msg!r} lacks {needle!r}")
        if code is not None:
            check(e.code == code, f"refusal code {e.code} != {code} ({e.msg})")
        return e.msg
    raise AssertionError(f"no refusal (expected {needle!r})")


def log_tags(ctx):
    return [p[2] for p in ctx.log_lines() if len(p) > 2]


# ---------------------------------------------------------------------------------------------------- synthetic records
def rec(dice):
    return {"present": [True] * K, "dice": list(dice), "status": ["TP"] * K, "hd95": [5.0] * K, "nsd": [90.0] * K,
            "ml": [10.0] * K, "dice_vox_old": list(dice)}


def frames(arms, spacing):
    return {"NATIVE": {"ref": {"present": [True] * K}, "arms": arms, "spacing": spacing}}


def scenario_a(n=30):
    """F0: 80 on every primary organ (colon 70, femoral heads 60). F1b_d: +g_d on ten primary organs, pancreas -3,
    adrenal +2 (even scans) / -4 (odd scans), colon +2; g = 3.0 / 1.8 / 0.6."""
    g = {"d1": 3.0, "d2": 1.8, "d3": 0.6}
    ids = [f"{i:04d}" for i in range(1, n + 1)]
    R, adr = {}, {}
    for i, c in enumerate(ids):
        base = [80.0] * K
        base[9] = 70.0
        base[13] = base[14] = 60.0
        arms = {"F0": rec(base)}
        adr[c] = 2.0 if i % 2 == 0 else -4.0
        for d, gd in g.items():
            v = list(base)
            for o in W.PRIMARY:
                v[o - 1] = 80.0 + gd
            v[7] = 77.0
            v[11] = 80.0 + adr[c]
            v[9] = 72.0
            arms[f"F1b_{d}"] = rec(v)
        R[c] = frames(arms, [0.977, 0.977, 2.5 if i % 3 else 3.0])
    return R, ids, g, adr


def gains_by_scan(R, ids, d, organs=None):
    """Hand computation of the per-scan penalised macro gain (every organ present: the plain mean over the organs)."""
    organs = organs or W.PRIMARY
    out = {}
    for c in ids:
        a = R[c]["NATIVE"]["arms"][f"F1b_{d}"]["dice"]
        b = R[c]["NATIVE"]["arms"]["F0"]["dice"]
        out[c] = sum(a[o - 1] for o in organs) / len(organs) - sum(b[o - 1] for o in organs) / len(organs)
    return out


def hand_t_ci(v):
    from scipy import stats
    import numpy as np
    v = np.asarray(v, float)
    n = len(v)
    m = float(v.mean())
    se = float(v.std(ddof=1)) / math.sqrt(n)
    q = float(stats.t.ppf(0.975, n - 1))
    return [m - q * se, m + q * se]


def indep_cluster_boot(vals, nboot=10000, seed=20261022):
    """Independent implementation of the registered scan bootstrap (one cluster per scan, percentile, 1000-row blocks)."""
    import numpy as np
    ids = sorted(vals)
    v = np.array([vals[c] for c in ids])
    G = len(ids)
    rng = np.random.default_rng(seed)
    reps = []
    for b0 in range(0, nboot, 1000):
        nb = min(1000, nboot - b0)
        w = rng.multinomial(G, np.full(G, 1.0 / G), size=nb).astype(float)
        reps.append((w @ v) / w.sum(1))
    reps = np.concatenate(reps)
    return [float(np.quantile(reps, 0.025)), float(np.quantile(reps, 0.975))]


def indep_two_level(per, nboot=10000, seed=20261020):
    """Independent implementation of the registered crossed scan x draw bootstrap for a complete design."""
    import numpy as np
    draws = sorted(per)
    ids = sorted(per[draws[0]])
    D = np.array([[per[d][c] for c in ids] for d in draws])
    S_, n = D.shape
    rng = np.random.default_rng(seed)
    cs = rng.multinomial(n, np.full(n, 1.0 / n), size=nboot).astype(float)
    cd = rng.multinomial(S_, np.full(S_, 1.0 / S_), size=nboot).astype(float)
    m = (cs @ D.T) / n
    val = (cd * m).sum(1) / cd.sum(1)
    lo, hi = np.percentile(val, [2.5, 97.5])
    return [float(lo), float(hi)]


# ---------------------------------------------------------------------------------------------------- T1 - T4, T13, T14
def t1_scenario_a():
    import numpy as np
    R, ids, g, adr = scenario_a()
    res = S.analyse_a1(R, ids, {"F0", "F1b_d1", "F1b_d2", "F1b_d3"})
    per = {d: gains_by_scan(R, ids, d) for d in g}
    for d in g:
        for c in ids:
            check(close(per[d][c], (10 * g[d] - 3 + adr[c]) / 12, 1e-9), "hand macro gain")
    M = float(np.mean([np.mean(list(per[d].values())) for d in g]))
    check(close(M, 7 / 6, 1e-9), "M = 7/6 by construction")
    p1 = res["P1_gain"]
    check(close(p1["M"], M), f"P1 M {p1['M']} != {M}")
    sv = [np.mean([per[d][c] for d in g]) for c in ids]
    tci = hand_t_ci(sv)
    check(close(p1["t_ci"][0], tci[0]) and close(p1["t_ci"][1], tci[1]), f"t-CI {p1['t_ci']} != {tci}")
    ci2 = indep_two_level(per)
    check(close(p1["two_level_ci"][0], ci2[0], 1e-9) and close(p1["two_level_ci"][1], ci2[1], 1e-9),
          f"two-level CI {p1['two_level_ci']} != independent {ci2}")
    check(p1["seed"] == 20261020, "two-level seed")
    check(p1["two_level_ci"][0] > 0 and p1["scans_up"] == 30 and p1["need_up"] == 20 and p1["pass"], "P1 PASS")
    check(p1["reading"] == "GAIN" and p1["M_ge_0_30"], "P1 reading")
    oc = res["P1_organ_costs"]
    for o in W.PRIMARY:
        nm = W.ORGANS[o - 1]
        r = oc["per_organ"][nm]
        if o == 8:
            check(close(r["M"], -3.0) and close(r["ci"][0], -3.0) and close(r["ci"][1], -3.0), f"pancreas {r}")
            check(r["ni"] == "INFERIOR" and r["harm"] == "HARMFUL", f"pancreas class {r}")
        elif o == 12:
            check(close(r["M"], -1.0) and r["ci"][0] < -1.0 < r["ci"][1] and r["ni"] == "INCONCLUSIVE", f"adrenal {r}")
        else:
            check(close(r["M"], 1.8) and close(r["ci"][0], 0.6) and close(r["ci"][1], 3.0), f"{nm} {r}")
            check(r["ni"] == "NON-INFERIOR", f"{nm} class")
    check(oc["reading"] == "SOME_INFERIOR" and oc["inferior"] == ["pancreas"] and oc["inconclusive"] == ["adrenal"],
          f"organ reading {oc['reading']} {oc['inferior']} {oc['inconclusive']}")
    p2 = res["P2_draw_dependence"]
    want = {"d1": 26 / 12, "d2": 14 / 12, "d3": 2 / 12}
    for d, m in want.items():
        check(close(p2["per_draw"][d]["M"], m), f"per-draw {d}")
    check(p2["per_draw"]["d1"]["pass"] and p2["per_draw"]["d2"]["pass"] and not p2["per_draw"]["d3"]["pass"], "per-draw")
    check(p2["per_draw"]["d3"]["scans_up"] == 15 and p2["per_draw"]["d3"]["need_up"] == 20, "d3 count")
    check(p2["k"] == 2 and p2["n_run"] == 3, "k = 2 of 3")
    ci1 = indep_cluster_boot(per["d1"])
    check(close(p2["per_draw"]["d1"]["scan_boot_ci"][0], ci1[0], 1e-9)
          and close(p2["per_draw"]["d1"]["scan_boot_ci"][1], ci1[1], 1e-9), "independent scan bootstrap of d1")
    for pair, m in {"d1 - d2": 1.0, "d1 - d3": 2.0, "d2 - d3": 1.0}.items():
        p = p2["pairwise"][pair]
        check(close(p["mean"], m) and close(p["ci"][0], m) and close(p["ci"][1], m) and p["excludes_0"], f"{pair} {p}")
    check(p2["reading"] == "DRAW-DEPENDENT" and close(p2["range"], 2.0) and close(p2["sd_ddof1"], 1.0), "P2 reading")
    check(res["outcome"] == {"combination": "GAIN_NOT_EVERY_DRAW", "organ_costs": "SOME_INFERIOR",
                             "draw_dependence": "DRAW-DEPENDENT", "k_of_n_run": "2 of 3"}, f"outcome {res['outcome']}")
    st = res["secondary_S_TS"]
    check(close(st["S"], M / 20.0) and st["ci"][0] > 0 and st["claim_systematic_share_present"], f"S {st['S']}")
    cr = res["secondary_colon_rectum"]
    check(close(cr["M"], 2.0) and close(cr["ci"][0], 2.0) and close(cr["ci"][1], 2.0), f"colon {cr}")
    ca = res["claims_allowed"]
    check(ca["four labelled WORD scans improve TS over zero-shot"] and not ca["for each of the three draws"]
          and not ca["no primary organ loses more than 1 Dice point"] and ca["organs to name as a cost"] == ["pancreas"]
          and ca["the gain depends on which four scans are labelled"], f"claims {ca}")
    check(close(res["descriptive_femur_heads_zero_shot"]["femur_head_left"], 60.0), "femur heads")
    check(set(res["descriptive_slice_spacing_strata"]) == {"2.5 mm", "3.0 mm"}, "strata")
    check(res["undecided_by_registration"] == [], "nothing undecided")
    return {"M": p1["M"], "t_ci": p1["t_ci"], "two_level_ci": p1["two_level_ci"], "k": p2["k"],
            "S": st["S"], "outcome": res["outcome"]}


def one_draw_records(deltas):
    ids = [f"{i:04d}" for i in range(1, len(deltas) + 1)]
    R = {}
    for c, dl in zip(ids, deltas):
        base = [80.0] * K
        v = list(base)
        for o in W.PRIMARY:
            v[o - 1] = 80.0 + dl
        R[c] = frames({"F0": rec(base), "F1b_d1": rec(v)}, [0.977, 0.977, 2.5])
    return R, ids


def t2_scenario_b():
    R, ids = one_draw_records([2.0] * 19 + [-0.5] * 10)
    res = S.analyse_a1(R, ids, {"F0", "F1b_d1"}, not_run={"F1b_d2": "test", "F1b_d3": "test"})
    p1 = res["P1_gain"]
    check(res["n_scans"] == 29 and p1["need_up"] == 20 and p1["scans_up"] == 19, f"count {p1}")
    check(close(p1["M"], (19 * 2.0 - 10 * 0.5) / 29), "M")
    check(not p1["pass"] and p1["reading"] == "NO GAIN", "19 of 29 up fails the 20 of 29 count")
    check(p1["bootstrap"].startswith("scan bootstrap") and p1["seed"] == 20261022, f"one draw -> scan bootstrap {p1}")
    check(res["P2_draw_dependence"]["reading"] == "NOT EVALUABLE (one draw)", "P2 with one draw")
    check(res["outcome"]["combination"] == "NO_GAIN", f"{res['outcome']}")
    check(res["arms_not_run"] == {"F1b_d2": "test", "F1b_d3": "test"}, f"not run {res['arms_not_run']}")
    R2, ids2 = one_draw_records([2.0] * 20 + [-0.5] * 9)
    r2 = S.analyse_a1(R2, ids2, {"F0", "F1b_d1"})
    check(r2["P1_gain"]["scans_up"] == 20 and r2["P1_gain"]["need_up"] == 20 and r2["P1_gain"]["pass"], "20 of 29 passes")
    check(r2["outcome"]["combination"] == "GAIN_EVERY_RUN_DRAW" and r2["outcome"]["k_of_n_run"] == "1 of 1",
          f"A1.1: one draw ran and gains -> GAIN_EVERY_RUN_DRAW {r2['outcome']}")
    check(r2["claims_allowed"]["for each draw that ran (report k of n_run)"]
          and not r2["claims_allowed"]["for each of the three draws"], "claims with one draw")
    check(r2["undecided_by_registration"] == [] and r2["settled_by_A1_1"], "settled by A1.1")
    return {"need_up_29": p1["need_up"], "19_up": p1["pass"], "20_up": r2["P1_gain"]["pass"],
            "combination_1_of_1": r2["outcome"]["combination"]}


def two_draw_records(g1, g3, n=12):
    ids = [f"{i:04d}" for i in range(1, n + 1)]
    R = {}
    for c in ids:
        base = [80.0] * K
        arms = {"F0": rec(base)}
        for d, gd in (("d1", g1), ("d3", g3)):
            v = list(base)
            for o in W.PRIMARY:
                v[o - 1] = 80.0 + gd
            arms[f"F1b_{d}"] = rec(v)
        R[c] = frames(arms, [0.977, 0.977, 3.0])
    return R, ids


def t3_scenario_c():
    R, ids = two_draw_records(1.0, 1.0)
    res = S.analyse_a1(R, ids, {"F0", "F1b_d1", "F1b_d3"}, not_run={"F1b_d2": "NOTRUN line"})
    p2 = res["P2_draw_dependence"]
    check(list(p2["pairwise"]) == ["d1 - d3"] and close(p2["pairwise"]["d1 - d3"]["ci"][0], 0.0)
          and not p2["pairwise"]["d1 - d3"]["excludes_0"], "equal draws")
    check(p2["reading"] == "NO DRAW DEPENDENCE DETECTED" and p2["k"] == 2 and p2["k_of"] == 2, f"{p2['reading']}")
    check(res["draws_run"] == ["d1", "d3"] and res["arms_not_run"] == {"F1b_d2": "NOTRUN line"}, "draws run")
    check(close(res["P1_gain"]["two_level_ci"][0], 1.0) and res["P1_gain"]["pass"], "P1 two draws")
    check(res["outcome"]["combination"] == "GAIN_EVERY_RUN_DRAW" and res["undecided_by_registration"] == [],
          "two of two draws gain: GAIN_EVERY_RUN_DRAW (A1.1)")
    R2, ids2 = two_draw_records(1.0, 0.5)
    r2 = S.analyse_a1(R2, ids2, {"F0", "F1b_d1", "F1b_d3"})
    check(r2["P2_draw_dependence"]["reading"] == "DRAW-DEPENDENT"
          and close(r2["P2_draw_dependence"]["pairwise"]["d1 - d3"]["mean"], 0.5), "different draws")
    R3, ids3 = two_draw_records(1.0, 0.1)
    r3 = S.analyse_a1(R3, ids3, {"F0", "F1b_d1", "F1b_d3"})
    check(r3["P2_draw_dependence"]["k"] == 1 and r3["outcome"]["combination"] == "GAIN_NOT_EVERY_DRAW",
          f"one of two draws gains -> GAIN_NOT_EVERY_DRAW {r3['outcome']}")
    return {"equal": p2["reading"], "different": r2["P2_draw_dependence"]["reading"],
            "two_of_two": res["outcome"]["combination"], "one_of_two": r3["outcome"]["combination"]}


def t4_tables():
    tab = {(True, 3, 3): "GAIN_EVERY_DRAW", (True, 2, 3): "GAIN_NOT_EVERY_DRAW", (True, 1, 3): "GAIN_NOT_EVERY_DRAW",
           (True, 0, 3): "GAIN_POOLED_ONLY", (True, 2, 2): "GAIN_EVERY_RUN_DRAW", (True, 1, 1): "GAIN_EVERY_RUN_DRAW",
           (True, 1, 2): "GAIN_NOT_EVERY_DRAW", (True, 0, 2): "GAIN_POOLED_ONLY", (True, 0, 1): "GAIN_POOLED_ONLY",
           (False, 3, 3): "NO_GAIN_SOME_DRAWS", (False, 1, 3): "NO_GAIN_SOME_DRAWS", (False, 2, 2): "NO_GAIN_SOME_DRAWS",
           (False, 0, 3): "NO_GAIN", (False, 0, 1): "NO_GAIN"}
    for (p, k, n), want in tab.items():
        check(S.combination(p, k, n) == want, f"combination {p} {k} {n}: {S.combination(p, k, n)} != {want}")
    ni = {o: "NON-INFERIOR" for o in ("a", "b")}
    check(S.organ_reading(ni)["reading"] == "ALL_NI", "ALL_NI")
    check(S.organ_reading(dict(ni, c="INCONCLUSIVE"))["reading"] == "INCONCLUSIVE", "INCONCLUSIVE")
    check(S.organ_reading(dict(ni, c="INCONCLUSIVE", d="INFERIOR"))["inferior"] == ["d"], "SOME_INFERIOR")
    r = S.organ_reading(dict(ni, c="NOT EVALUABLE"))
    check(r["reading"] == "NO_ORGAN_COST_AMONG_EVALUABLE" and r["not_evaluable"] == ["c"], f"not evaluable {r}")
    r = S.organ_reading(dict(ni, c="NOT EVALUABLE", d="INCONCLUSIVE"))
    check(r["reading"] == "INCONCLUSIVE" and r["not_evaluable"] == ["c"], "inconclusive wins over not evaluable")
    return {"combinations": len(tab)}


def t13_one_draw_bootstrap():
    import numpy as np
    import m2_cluster as MC
    rng = np.random.default_rng(5)
    R, ids = one_draw_records(list(rng.normal(0.8, 1.0, 30)))
    res = S.analyse_a1(R, ids, {"F0", "F1b_d1"})
    per = gains_by_scan(R, ids, "d1")
    want = indep_cluster_boot(per, seed=20261022)
    p1 = res["P1_gain"]
    check(close(p1["ci"][0], want[0], 1e-9) and close(p1["ci"][1], want[1], 1e-9),
          f"one-draw P1 CI {p1['ci']} != independent scan bootstrap {want}")
    cb = F.cluster_boot(per)
    check(p1["ci"] == cb["ci"] and p1["seed"] == 20261022 and p1["nboot"] == 10000, "== final_common.cluster_boot")
    check(p1["two_level_ci"] is None, "no two-level CI with one draw")
    old, _ = MC.boot2(MC.matrix({"d1": per})[2], 10000, 20261020, resample_draws=False)
    check(not (close(old[0], p1["ci"][0], 1e-12) and close(old[1], p1["ci"][1], 1e-12)),
          "the one-draw CI is not the v2 code's boot2(resample_draws=False, seed 20261020)")
    pd1 = res["P2_draw_dependence"]["per_draw"]["d1"]
    check(pd1["scan_boot_ci"] == p1["ci"], "P1 with one draw == that draw's P2 rule CI")
    return {"ci": p1["ci"], "independent": want, "v2_boot2_ci": old}


def t14_unequal_scan_sets():
    import numpy as np
    rng = np.random.default_rng(3)
    d1 = {f"{i:04d}": 0.5 + float(rng.normal(0, 0.01)) for i in range(1, 31)}
    d2 = {f"{i:04d}": 0.05 + float(rng.normal(0, 0.005)) for i in range(1, 7)}
    p = S.p1_pooled({"d1": d1, "d2": d2})
    m_draws = (np.mean(list(d1.values())) + np.mean(list(d2.values()))) / 2
    sv = [np.mean([v for v in (d1.get(c), d2.get(c)) if v is not None]) for c in sorted(d1)]
    check(close(p["M"], m_draws, 1e-12), f"M {p['M']} != mean of draw means {m_draws}")
    check(close(p["scan_level_mean"], float(np.mean(sv)), 1e-12), "scan-level mean of draw-averaged values")
    check(p["M"] < 0.30 <= p["scan_level_mean"], f"fixture: M {p['M']} < 0.30 <= scan mean {p['scan_level_mean']}")
    check(p["t_ci"][0] > 0 and p["scans_up"] == 30 and p["ci"][0] > 0, f"every other condition holds {p}")
    check(not p["M_ge_0_30"] and not p["pass"], "the +0.30 bar is applied to M (mean of the draw means): FAIL")
    q = S.p1_pooled({"d1": d1, "d2": {c: 0.2 for c in d1}})
    check(close(q["M"], q["scan_level_mean"], 1e-12) and q["pass"], "equal scan sets: the two means coincide")
    return {"M": p["M"], "scan_level_mean": p["scan_level_mean"], "pass": p["pass"]}


# ---------------------------------------------------------------------------------------------------- synthetic WORD
@contextlib.contextmanager
def patched_word(root):
    root = Path(root)
    keep = {k: getattr(W, k) for k in ("DIR", "LOOK", "LABELS_TS_ZIP", "IMAGES", "PREP", "DOWNLOAD", "reference")}
    try:
        W.DIR = root / "C_word"
        W.LOOK = root / "C_word/look"
        W.LABELS_TS_ZIP = root / "archive_labelsTs/WORD_V0.1.0_labelsTs.zip"
        W.IMAGES = root / "images"
        W.PREP = root / "C_word/prep"
        W.DOWNLOAD = root / "C_word/download.json"
        yield
    finally:
        for k, v in keep.items():
            setattr(W, k, v)


def plog_line(ctx, msg):
    ctx.prereg_log.parent.mkdir(parents=True, exist_ok=True)
    with open(ctx.prereg_log, "a") as f:
        f.write(f"[{F.now()}] {msg}\n")


def setup_osf(ctx, url=URL, deposit=True, name_sha=None, name_path=None, log_line=True, osf_before_deposit=False,
              gate_text=None):
    ctx.osf_file.parent.mkdir(parents=True, exist_ok=True)
    txt = gate_text if gate_text is not None else (
        f"{url}\nOSF registration (PUBLIC) of {name_path or V5_REL} sha256 {name_sha or FAKE_V5}\n")

    def dep():
        if deposit:
            plog_line(ctx, f"DEPOSIT sha256 {FAKE_V5} {ctx.deposit_rel} (synthetic)")

    def osf():
        ctx.osf_file.write_text(txt)
        if log_line:
            plog_line(ctx, f"OSF-REGISTRATION {url} sha256 {F.sha256(ctx.osf_file)} {ctx.rel(ctx.osf_file)}")

    if osf_before_deposit:
        osf()
        dep()
    else:
        dep()
        osf()


ANALYSIS_TO_WORD = {a: w for w, a in W.WORD_TO_ANALYSIS.items() if w != 13}
SHAPE = (45, 10, 10)
AFF = [[10.0, 0, 0, 0], [0, 10.0, 0, 0], [0, 0, 10.0, 0], [0, 0, 0, 1.0]]
TEST_IDS = ["0014", "0016", "0017", "0019", "0021", "0131"]


def ts_ids():
    from totalsegmentator.map_to_binary import class_map
    inv = {v: k for k, v in class_map["total"].items()}
    return {a: inv[W.TS_NAMES[a][0]] for a in range(1, K + 1)}


def build_word(root, same_dup=True):
    """Synthetic WORD under root: images, labels (zip), predictions + DONE records, split, overlap groups."""
    import nibabel as nib
    import numpy as np
    root = Path(root)
    assert A.under_tmp(root), root
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "C_word/prep").mkdir(parents=True, exist_ok=True)
    json.dump({"official_split": {"train": ["0004"], "val": ["0001"], "test": TEST_IDS}},
              open(root / "C_word/download.json", "w"))
    json.dump({"within_word_header_groups": [["0017", "0131"]]}, open(root / "C_word/prep/overlap.json", "w"))
    aff = np.array(AFF)
    rng = np.random.default_rng(11)
    imgs = {c: rng.normal(0, 100, SHAPE).astype(np.int16) for c in TEST_IDS}
    if same_dup:
        imgs["0131"] = imgs["0017"].copy()
    for c, a in imgs.items():
        nib.save(nib.Nifti1Image(a, aff), str(root / "images" / f"word_{c}.nii.gz"))
    lab = np.zeros(SHAPE, np.uint8)
    for a in range(1, K + 1):
        lab[3 * (a - 1):3 * a, :, :] = ANALYSIS_TO_WORD[a]
    z = root / "archive_labelsTs/WORD_V0.1.0_labelsTs.zip"
    z.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(z, "w") as zz:
        for c in TEST_IDS:
            bio = io.BytesIO()
            fh = nib.FileHolder(fileobj=bio)
            nib.Nifti1Image(lab, aff).to_file_map({"header": fh, "image": fh})
            zz.writestr(f"labelsTs/word_{c}.nii.gz", gzip.compress(bio.getvalue()))
    tid = ts_ids()

    def pred(full):
        P = np.zeros(SHAPE, np.uint8)
        for a in range(1, K + 1):
            P[3 * (a - 1):3 * a, :10 if a in full else 9, :] = tid[a]
        return P

    allo = set(range(1, K + 1))
    maps = {"F0": pred(set()), "F1b_d1": pred(allo), "F1b_d2": pred({1}), "F1b_d3": pred(set())}
    dirs = A.arm_dirs(root / "C_word/pred")
    for arm, P in maps.items():
        (dirs[arm] / "raw").mkdir(parents=True, exist_ok=True)
        shas = {}
        for c in TEST_IDS:
            p = dirs[arm] / "raw" / f"{c}.nii.gz"
            nib.save(nib.Nifti1Image(P, aff), str(p))
            shas[c] = F.sha256(p)
        json.dump({"per_scan_raw_sha256": shas}, open(dirs[arm] / "DONE.json", "w"))
    return z


def e2e_ctx(root, pinned=None, code_files=None):
    z = Path(root) / "archive_labelsTs/WORD_V0.1.0_labelsTs.zip"
    return A.Ctx.scratch_ctx(root, labels_zip=z, labels_zip_bytes=os.stat(z).st_size,
                             labels_zip_sha256=pinned or F.sha256(z), code_files=code_files)


def declare_a12(ctx, arms=None, reason="unit test: this fixture builds no A1.2 arm"):
    for a in (arms if arms is not None else A.A12_ARMS):
        plog_line(ctx, f"NOTRUN word_A1 {a} {reason}")


def word_fixture(name, pinned=None, with_code=False, a12="declared"):
    """a12: 'declared' = the nine A1.2 arms declared NOT RUN (the A1 / A1.1 TS-only look); 'build' = STOCK, MERGE_N and
    LABELMAP built for every draw (build_a12_arms)."""
    root = Path(SCR) / name
    shutil.rmtree(root, ignore_errors=True)
    build_word(root)
    code = None
    if with_code:
        code = [root / "code/word_a1_score_copy.py", root / "code/word_a1_common_copy.py"]
        code[0].parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(HERE / "word_a1_score.py", code[0])
        shutil.copyfile(HERE / "word_a1_common.py", code[1])
    ctx = e2e_ctx(root, pinned, code)
    setup_osf(ctx)
    for p in code or []:
        plog_line(ctx, f"CODE {F.sha256(p)} {ctx.rel(p)}")
    if a12 == "declared":
        declare_a12(ctx)
    elif a12 == "build":
        build_a12_arms(root)
    return root, ctx


EXTRA_KEYS = {"written", "duplicates_check", "dropped_as_duplicates", "gates", "maps_sha256", "labels_ts_zip_sha256",
              "osf_registration", "plan_sha256", "a1_sha256", "a1_1_sha256", "look"}


def analysis_part(res):
    return json.dumps(F.clean_nan({k: v for k, v in res.items() if k not in EXTRA_KEYS}), sort_keys=True,
                      default=F._default)


def t5_end_to_end():
    root, ctx = word_fixture("t5_e2e")
    seen = {"reads": 0, "checked_first": False}
    with patched_word(root):
        orig = W.reference

        def guarded(cid, marker):
            mk = Path(marker)
            check(mk.resolve() == (ctx.look_dir / "GT_OPENED").resolve() and mk.is_file(), "marker before reading")
            lines = [ln.split() for ln in open(ctx.prereg_log)]
            hit = [p for p in lines if len(p) >= 6 and p[2] == "GT_OPENED" and p[4] == F.sha256(mk)]
            check(len(hit) == 1, "exactly one GT_OPENED line with the marker's sha256 before the first label read")
            seen["reads"] += 1
            seen["checked_first"] = True
            return orig(cid, marker)

        W.reference = guarded
        res = S.look(ctx, workers=1, log=lambda m: None)
        second = expect_refuse(lambda: S.look(ctx, workers=1, log=lambda m: None), "already been made")
        expect_refuse(lambda: S.continue_gates(ctx), "the look is complete")
    check(res["dropped_as_duplicates"] == ["0131"] and res["n_scans"] == 5, f"duplicate rule {res['dropped_as_duplicates']}")
    check(res["duplicates_check"][0]["r_stride4"] is not None and res["duplicates_check"][0]["r_stride4"] >= 0.999,
          "0017 ~ 0131 identical images")
    check(seen["checked_first"] and seen["reads"] == 2 * 5, f"label reads {seen['reads']} (two per scan, 5 scans)")
    d1 = 100.0 - PARTIAL
    d2 = d1 / 12
    M = (d1 + d2 + 0.0) / 3
    p1 = res["P1_gain"]
    check(close(p1["M"], M, 1e-9) and close(p1["two_level_ci"][0], 0.0, 1e-9) and close(p1["two_level_ci"][1], d1, 1e-9),
          f"P1 {p1['M']} {p1['two_level_ci']}")
    check(p1["need_up"] == 4 and p1["scans_up"] == 5 and not p1["pass"], "P1 fails: two-level lower bound is 0, not > 0")
    p2 = res["P2_draw_dependence"]
    check(close(p2["per_draw"]["d1"]["M"], d1, 1e-9) and close(p2["per_draw"]["d2"]["M"], d2, 1e-9)
          and close(p2["per_draw"]["d3"]["M"], 0.0, 1e-9), "per-draw gains")
    check(p2["k"] == 2 and p2["reading"] == "DRAW-DEPENDENT", f"k {p2['k']} {p2['reading']}")
    check(res["outcome"] == {"combination": "NO_GAIN_SOME_DRAWS", "organ_costs": "ALL_NI",
                             "draw_dependence": "DRAW-DEPENDENT", "k_of_n_run": "2 of 3"}, f"{res['outcome']}")
    check(close(res["secondary_S_TS"]["S"], 13 / 36, 1e-9), f"S {res['secondary_S_TS']['S']}")
    check(close(res["secondary_colon_rectum"]["M"], d1 / 3, 1e-9), "colon_rectum")
    check(close(res["descriptive_femur_heads_zero_shot"]["femur_head_left"], PARTIAL, 1e-9), "femur heads")
    check(close(res["P1_organ_costs"]["per_organ"]["liver"]["ci"][1], d1, 1e-9), "liver organ CI")
    tags = log_tags(ctx)
    check(tags.count("GT_OPENED") == 1 and tags.index("GT_OPENED") < tags.index("RESULTS"), f"log order {tags}")
    check("GT_CRASHED" not in tags and "GT_CONTINUED" not in tags, "no crash, no continuation")
    gt = [p for p in ctx.log_lines() if p[2] == "GT_OPENED"][0]
    check(gt[5] == ctx.rel(ctx.look_dir / "GT_OPENED") and F.sha256(ctx.look_dir / "maps.json") in " ".join(gt),
          "GT_OPENED line names the marker path and the maps.json sha256")
    mk = json.loads((ctx.look_dir / "GT_OPENED").read_text())
    check(mk["ids"] == ["0014", "0016", "0017", "0019", "0021"] and mk["osf_registration"] == URL, "marker record")
    pv = sorted(p.name for p in (ctx.look_dir / "pervol_done").glob("*.json"))
    check(pv == [f"{c}.json" for c in mk["ids"]], f"per-scan records {pv}")
    check(res["look"]["continuation"] is None and res["look"]["single_look"], "look record")
    return {"n_scans": res["n_scans"], "dropped": res["dropped_as_duplicates"], "M": p1["M"], "P1": p1["pass"],
            "k": p2["k"], "S": res["secondary_S_TS"]["S"], "outcome": res["outcome"], "second_look": second[:60]}


def t6_abort():
    root, ctx = word_fixture("t6_abort", pinned="0" * 64)
    with patched_word(root):
        def never(cid, marker):
            raise AssertionError("label read after an aborted look")

        W.reference = never
        expect_refuse(lambda: S.look(ctx, workers=1, log=lambda m: None), "look aborted", code=4)
        expect_refuse(lambda: S.look(ctx, workers=1, log=lambda m: None), "already been made")
        expect_refuse(lambda: S.continue_gates(ctx), "GT_ABORTED", code=4)
    tags = log_tags(ctx)
    check("GT_OPENED" in tags and "GT_ABORTED" in tags and tags.index("GT_OPENED") < tags.index("GT_ABORTED"), "order")
    check("GT_CRASHED" not in tags, "an abort is not a crash")
    check(not (ctx.look_dir / "results.json").exists(), "no results after an abort")
    return {"log": [t for t in tags if t.startswith("GT_")]}


def gate_fixture(name, arms=("F0", "F1b_d1", "F1b_d2", "F1b_d3"), osf=True, a12_declared=True, **kw):
    root = Path(SCR) / "t7_gates" / name
    shutil.rmtree(root, ignore_errors=True)
    z = root / "archive_labelsTs/WORD_V0.1.0_labelsTs.zip"
    z.parent.mkdir(parents=True, exist_ok=True)
    z.write_bytes(b"synthetic zip stand-in")
    ctx = A.Ctx.scratch_ctx(root, labels_zip=z, labels_zip_bytes=z.stat().st_size, labels_zip_sha256="1" * 64, **kw)
    for a_ in arms:
        d = A.arm_dirs(ctx.pred_dir)[a_]
        d.mkdir(parents=True, exist_ok=True)
        (d / "DONE.json").write_text("{}")
    if osf:
        setup_osf(ctx)
    if a12_declared:
        declare_a12(ctx, [a_ for a_ in A.A12_ARMS if a_ not in arms])
    return root, ctx


def run_gate(root, ctx, needle, code=2):
    with patched_word(root):
        def never(cid, marker):
            raise AssertionError("label read in a refused look")

        W.reference = never
        msg = expect_refuse(lambda: S.look(ctx, workers=1, log=lambda m: None), needle, code)
    check(not (ctx.look_dir / "GT_OPENED").exists() and not (ctx.look_dir / "maps.json").exists(),
          "a refused look leaves no marker and no maps.json")
    check("GT_CRASHED" not in log_tags(ctx), "a refusal before the marker is not a crash")
    return msg


def t7_scorer_gates():
    out = {}
    r, c = gate_fixture("no_osf_file", osf=False)
    out["no_osf_file"] = run_gate(r, c, "does not exist")
    r, c = gate_fixture("no_url", osf=False)
    setup_osf(c, gate_text="registration pending\n")
    out["no_url"] = run_gate(r, c, "is not an https://osf.io/<id> URL")
    r, c = gate_fixture("no_log_line", osf=False)
    setup_osf(c, log_line=False)
    out["no_log_line"] = run_gate(r, c, "no 'OSF-REGISTRATION")
    r, c = gate_fixture("edited_after_logging")
    c.osf_file.write_text(c.osf_file.read_text() + "edited later\n")
    out["edited_after_logging"] = run_gate(r, c, "no 'OSF-REGISTRATION")
    r, c = gate_fixture("names_v1_sha", osf=False)
    setup_osf(c, name_sha=A.DEPOSIT_V1_SHA256)
    out["names_v1_sha"] = run_gate(r, c, "names the v1 zip")
    r, c = gate_fixture("names_v2_sha", osf=False)
    setup_osf(c, name_sha=A.DEPOSIT_V2_SHA256)
    out["names_v2_sha"] = run_gate(r, c, "names the v2 zip")
    r, c = gate_fixture("names_v2_path", osf=False)
    setup_osf(c, name_path="outputs/results/vccor/osf_deposit_2026-11-05_word_v2.zip")
    out["names_v2_path"] = run_gate(r, c, "names the v2 zip")
    r, c = gate_fixture("names_v1_path", osf=False)
    setup_osf(c, name_path="outputs/results/vccor/osf_deposit_2026-11-05_word.zip")
    out["names_v1_path"] = run_gate(r, c, "names the v1 zip")
    r, c = gate_fixture("v2_deposit_line_only", osf=False, deposit_rel=V5_REL)
    plog_line(c, f"DEPOSIT sha256 {A.DEPOSIT_V2_SHA256} outputs/results/vccor/osf_deposit_2026-11-05_word_v2.zip (v2)")
    setup_osf(c, deposit=False)
    out["v2_deposit_line_only"] = run_gate(r, c, f"no DEPOSIT line for {V5_REL}")
    r, c = gate_fixture("osf_before_deposit", osf=False)
    setup_osf(c, osf_before_deposit=True)
    out["osf_before_deposit"] = run_gate(r, c, "precedes the v5 DEPOSIT line")
    r, c = gate_fixture("no_v3_deposit", osf=False)
    setup_osf(c, deposit=False)
    out["no_v3_deposit"] = run_gate(r, c, "no DEPOSIT line")
    r, c = gate_fixture("marker_exists")
    c.look_dir.mkdir(parents=True, exist_ok=True)
    (c.look_dir / "GT_OPENED").write_text("{}")
    with patched_word(r):
        out["marker_exists"] = expect_refuse(lambda: S.look(c, workers=1, log=lambda m: None), "no second look")
    r, c = gate_fixture("results_exist")
    c.look_dir.mkdir(parents=True, exist_ok=True)
    (c.look_dir / "results.json").write_text("{}")
    out["results_exist"] = run_gate(r, c, "no second look")
    r, c = gate_fixture("missing_draw", arms=("F0", "F1b_d1", "F1b_d2"))
    out["missing_draw"] = run_gate(r, c, "neither DONE nor declared NOT RUN")
    r, c = gate_fixture("missing_required", arms=("F0", "F1b_d2", "F1b_d3"))
    out["missing_required"] = run_gate(r, c, "required arm F1b_d1")
    r, c = gate_fixture("missing_F0", arms=("F1b_d1", "F1b_d2", "F1b_d3"))
    out["missing_F0"] = run_gate(r, c, "required arm F0")
    r, c = gate_fixture("required_declared", arms=("F0", "F1b_d2", "F1b_d3"))
    plog_line(c, "NOTRUN word_A1 F1b_d1 train failed twice")
    out["required_declared"] = run_gate(r, c, "required arm F1b_d1 was declared NOT RUN")
    r, c = gate_fixture("experiment_declared")
    plog_line(c, "NOTRUN word_A1 EXPERIMENT required arm F0 NOT RUN")
    out["experiment_declared"] = run_gate(r, c, "experiment was declared NOT RUN")
    r, c = gate_fixture("done_after_cutoff", cutoff=datetime.now() - timedelta(days=1))
    out["done_after_cutoff"] = run_gate(r, c, "DONE after the cutoff")
    r, c = gate_fixture("zip_size")
    c.labels_zip_bytes = 7
    out["zip_size"] = run_gate(r, c, "labelsTs zip size")
    r, c = gate_fixture("a12_missing", a12_declared=False)
    out["a12_missing"] = run_gate(r, c, "arm STOCK_d1: neither DONE nor declared NOT RUN")
    r, c = gate_fixture("declared_d3_ok", arms=("F0", "F1b_d1", "F1b_d2"))
    plog_line(c, "NOTRUN word_A1 F1b_d3 not DONE by the cutoff")
    with patched_word(r):
        rec_, _ = S.gates(c, check=True, log=lambda m: None)
        rec2, _ = S.gates(c, check=True, log=lambda m: None, stat_zip=False)
    check(rec_["all_ok"] and rec_["4_arms"]["detail"]["have"] == ["F0", "F1b_d1", "F1b_d2"], f"declared d3 {rec_}")
    check(rec2["3b_labels_zip_size"]["ok"] is None and rec2["all_ok"], "--check leaves the zip alone")
    out["declared_d3_ok"] = "gates pass; F1b_d3 excluded as declared NOT RUN"
    return {k: v[:90] for k, v in out.items()}


def t8_duplicates_kept():
    root = Path(SCR) / "t8_dup"
    shutil.rmtree(root, ignore_errors=True)
    build_word(root, same_dup=False)
    with patched_word(root):
        out, drop = WS.duplicates(list(TEST_IDS), lambda m: None)
    check(drop == [] and not out[0]["same_image"] and out[0]["r_stride4"] < 0.999, f"different images kept {out}")
    return {"r": out[0]["r_stride4"], "dropped": drop}


# ---------------------------------------------------------------------------------------------------- T15, T16
def crash_after(orig, n_ok):
    cnt = {"n": 0}

    def ref(cid, marker):
        cnt["n"] += 1
        if cnt["n"] > n_ok:
            raise RuntimeError("injected crash after the marker")
        return orig(cid, marker)

    return ref, cnt


def crashed_fixture(name, with_code=True):
    """A synthetic look that crashes after two of five scans (workers = 1)."""
    root, ctx = word_fixture(name, with_code=with_code)
    with patched_word(root):
        ref, _ = crash_after(W.reference, 4)
        W.reference = ref
        try:
            S.look(ctx, workers=1, log=lambda m: None)
            raise AssertionError("the injected crash did not happen")
        except RuntimeError as e:
            check("injected crash" in str(e), str(e))
    return root, ctx


def t15_crash_continue():
    out = {}
    r0, c0 = word_fixture("t15_reference")
    with patched_word(r0):
        ref_res = S.look(c0, workers=1, log=lambda m: None)
    root, ctx = crashed_fixture("t15_crash")
    tags = log_tags(ctx)
    check(tags.count("GT_OPENED") == 1 and tags.count("GT_CRASHED") == 1 and "RESULTS" not in tags, f"log {tags}")
    cl = [ln for ln in ctx.log_text_lines() if " GT_CRASHED " in ln][0]
    check(" GT_CRASHED RuntimeError injected crash after the marker" in cl, f"GT_CRASHED line {cl}")
    check(not (ctx.look_dir / "results.json").exists(), "no results after a crash")
    done = sorted(p.stem for p in (ctx.look_dir / "pervol_done").glob("*.json"))
    check(done == ["0014", "0016"], f"completed per-scan records {done}")
    with patched_word(root):
        expect_refuse(lambda: S.look(ctx, workers=1, log=lambda m: None), "already been made")
        ok = S.continue_gates(ctx)
        check(ok["n"] == 1 and ok["n_crash"] == 1, f"continuation gates {ok['grec']}")
        orig = W.reference
        cnt = {"n": 0}

        def counting(cid, marker):
            cnt["n"] += 1
            return orig(cid, marker)

        W.reference = counting
        res = S.continue_look(ctx, workers=1, log=lambda m: None)
        expect_refuse(lambda: S.continue_gates(ctx), "the look is complete")
    check(cnt["n"] == 2 * 3, f"only the three unfinished scans are re-read ({cnt['n']} reads)")
    lk = res["look"]["continuation"]
    check(lk["n"] == 1 and lk["reused"] == ["0014", "0016"] and lk["rescored"] == ["0017", "0019", "0021"], f"{lk}")
    check(analysis_part(res) == analysis_part(ref_res), "continued results == uninterrupted results")
    tags = log_tags(ctx)
    check(tags.index("GT_CRASHED") < tags.index("GT_CONTINUED") < tags.index("RESULTS") and tags.count("GT_OPENED") == 1,
          f"log order {tags}")
    rl = [ln for ln in ctx.log_text_lines() if " RESULTS " in ln][0]
    check("continuation 1" in rl, rl)
    out["continued"] = {"reused": lk["reused"], "rescored": lk["rescored"], "equal_to_uninterrupted": True}

    root, ctx = word_fixture("t15_no_marker")
    with patched_word(root):
        out["no_marker"] = expect_refuse(lambda: S.continue_gates(ctx), "no look to continue")[:80]
    root, ctx = crashed_fixture("t15_results_line")
    plog_line(ctx, f"RESULTS sha256 {'c' * 64} {ctx.rel(ctx.look_dir / 'results.json')} (synthetic)")
    with patched_word(root):
        out["results_line"] = expect_refuse(lambda: S.continue_gates(ctx), "already has a RESULTS line")[:80]
    root, ctx = crashed_fixture("t15_results_file")
    (ctx.look_dir / "results.json").write_text("{}")
    with patched_word(root):
        out["results_file"] = expect_refuse(lambda: S.continue_gates(ctx), "the look is complete")[:80]
    root, ctx = crashed_fixture("t15_maps_changed")
    with open(ctx.look_dir / "maps.json", "a") as fh:
        fh.write(" ")
    with patched_word(root):
        out["maps_changed"] = expect_refuse(lambda: S.continue_gates(ctx), "maps.json does not match")[:80]
    root, ctx = crashed_fixture("t15_marker_changed")
    mkp = ctx.look_dir / "GT_OPENED"
    os.chmod(mkp, 0o644)
    with open(mkp, "a") as fh:
        fh.write(" ")
    with patched_word(root):
        out["marker_changed"] = expect_refuse(lambda: S.continue_gates(ctx), "marker file no longer matches")[:80]
    root, ctx = crashed_fixture("t15_code_changed")
    with open(ctx.code_files[0], "a") as fh:
        fh.write("\n")
    with patched_word(root):
        msg = expect_refuse(lambda: S.continue_gates(ctx), "code differs from its CODE lines")
        check("a continuation with changed code is not allowed" in msg and "disclosed deviation" in msg, msg)
        expect_refuse(lambda: S.continue_look(ctx, workers=1, log=lambda m: None), "code differs")
    check("GT_CONTINUED" not in log_tags(ctx), "a refused continuation logs nothing")
    out["code_changed"] = msg[:110]
    root, ctx = crashed_fixture("t15_tampered_record")
    p = ctx.look_dir / "pervol_done" / "0014.json"
    d = json.loads(p.read_text())
    d["payload"]["frames"]["NATIVE"]["arms"]["F0"]["dice"][0] = 1.0
    p.write_text(json.dumps(d))
    with patched_word(root):
        res2 = S.continue_look(ctx, workers=1, log=lambda m: None)
    lk2 = res2["look"]["continuation"]
    check(lk2["reused"] == ["0016"] and "0014" in lk2["rescored"] and "payload sha256" in lk2["not_reused_why"]["0014"],
          f"a tampered per-scan record is re-scored, not reused {lk2}")
    check(analysis_part(res2) == analysis_part(ref_res), "results unaffected by the tampered record")
    out["tampered_record"] = "re-scored (payload sha256 mismatch)"
    return out


def t16_killed_worker():
    out = {}
    root, ctx = word_fixture("t16_killed")
    with patched_word(root):
        orig = W.reference

        def killer(cid, marker):
            if cid == "0017":
                os.kill(os.getpid(), signal.SIGKILL)
            return orig(cid, marker)

        W.reference = killer
        t0 = time.time()
        try:
            S.look(ctx, workers=2, log=lambda m: None, result_timeout_s=60)
            raise AssertionError("a killed worker did not fail the look")
        except Exception as e:
            check(type(e).__name__ == "BrokenProcessPool", f"{type(e).__name__}: {e}")
        dt = time.time() - t0
        check(dt < 60, f"the look failed after {dt:.1f} s (must not hang)")
        cl = [ln for ln in ctx.log_text_lines() if " GT_CRASHED " in ln]
        check(len(cl) == 1 and " GT_CRASHED BrokenProcessPool " in cl[0], f"{cl}")
        out["killed"] = f"BrokenProcessPool after {dt:.1f} s; GT_CRASHED logged"
        W.reference = orig
        res = S.continue_look(ctx, workers=2, log=lambda m: None, result_timeout_s=60)
        check(res["n_scans"] == 5 and res["look"]["continuation"]["n"] == 1, "continued to results after the kill")
        out["continued_after_kill"] = {"reused": res["look"]["continuation"]["reused"],
                                       "rescored": res["look"]["continuation"]["rescored"]}
    root, ctx = word_fixture("t16_stuck")
    with patched_word(root):
        orig = W.reference

        def stuck(cid, marker):
            if cid == "0017":
                time.sleep(300)
            return orig(cid, marker)

        W.reference = stuck
        t0 = time.time()
        try:
            S.look(ctx, workers=2, log=lambda m: None, result_timeout_s=4)
            raise AssertionError("a stuck worker did not fail the look")
        except TimeoutError as e:
            check("a worker is stuck" in str(e), str(e))
        dt = time.time() - t0
        check(dt < 60, f"stuck worker: {dt:.1f} s")
        time.sleep(0.5)
        alive = [p for p in multiprocessing.active_children() if p.is_alive()]
        check(not alive, f"workers left alive: {alive}")
        cl = [ln for ln in ctx.log_text_lines() if " GT_CRASHED " in ln]
        check(len(cl) == 1 and " GT_CRASHED TimeoutError " in cl[0], f"{cl}")
        out["stuck"] = f"TimeoutError after {dt:.1f} s; workers killed; GT_CRASHED logged"
    return out


# ---------------------------------------------------------------------------------------------------- runner
STANDIN = r"""
import fcntl, json, os, sys, time
name, done, plan_file, work, lock = sys.argv[1:6]
plan = json.load(open(plan_file)).get(name, {})
cf = os.path.join(work, name + ".count")
n = int(open(cf).read()) if os.path.exists(cf) else 0
open(cf, "w").write(str(n + 1))
open(os.path.join(work, "order.txt"), "a").write(name + "\n")
keys = ["CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "PYTORCH_CUDA_ALLOC_CONF"]
json.dump({k: os.environ.get(k) for k in keys}, open(os.path.join(work, name + ".env.json"), "w"))
if plan.get("check_lock"):
    fd = os.open(lock, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = "free"
    except OSError:
        state = "held"
    os.close(fd)
    open(os.path.join(work, name + ".locked"), "w").write(state)
if n < int(plan.get("sleep_n", 99)):
    time.sleep(float(plan.get("sleep", 0)))
if n < int(plan.get("fail_times", 0)):
    sys.exit(1)
os.makedirs(os.path.dirname(done), exist_ok=True)
open(done, "w").write("{}")
"""
LOCK_BYTES = b"shared GPU-1 lock (scratch stand-in; content must never change)\n"


def runner_fixture(name, plan=None, osf=True, decisions=True, prep=True, lock=True, **kw):
    root = Path(SCR) / "t9_runner" / name
    shutil.rmtree(root, ignore_errors=True)
    ctx = A.Ctx.scratch_ctx(root, labels_zip_bytes=None, **kw)
    work = root / "work"
    work.mkdir(parents=True, exist_ok=True)
    (work / "plan.json").write_text(json.dumps(plan or {}))
    ctx.q8_dir.mkdir(parents=True, exist_ok=True)
    if decisions:
        ctx.decisions.write_text(json.dumps({"drop": ["x"] + A.SCHEDULER_NEUTRALISED}))
    else:
        ctx.decisions.write_text(json.dumps({"drop": A.DROPPED_JOBS}))
    if prep:
        for n in A.PREP_DONE_Q8:
            (ctx.q8_dir / f"{n}.done").write_text("0")
    if lock:
        ctx.gpu_lock.write_bytes(LOCK_BYTES)
    if osf:
        setup_osf(ctx)
    for rp in ("labels/manifest.json", "ts_cpu/CAPTURE_DONE.json", "images.json"):
        (ctx.prep_dir / rp).parent.mkdir(parents=True, exist_ok=True)
        (ctx.prep_dir / rp).write_text("{}")
    jobs = A.run_jobs(ctx.pred_dir, ctx.models_dir, root / "C_word/prep_stock")
    for j in jobs:
        j["interp"] = "/usr/bin/python3"
        j["args"] = ["-I", "-c", STANDIN, j["name"], j["done_file"], str(work / "plan.json"), str(work), str(ctx.gpu_lock)]
        j["cwd"] = str(work)
    return root, ctx, jobs, work


def order(work):
    p = work / "order.txt"
    return p.read_text().split() if p.exists() else []


def run_ok(ctx, jobs, **kw):
    kw.setdefault("pids_fn", lambda: [])
    kw.setdefault("used_fn", lambda: 0)
    kw.setdefault("wait_s", 1.0)
    kw.setdefault("poll_s", 0.1)
    kw.setdefault("lock_wait_s", 5.0)
    kw.setdefault("lock_poll_s", 0.1)
    return RUN.run(ctx, jobs, log=lambda m: None, **kw)


def notrun_lines(ctx):
    return [" ".join(p[2:]) for p in ctx.log_lines() if len(p) > 2 and p[2] == "NOTRUN"]


def mark_done(jobs, n):
    for j in jobs[:n]:
        Path(j["done_file"]).parent.mkdir(parents=True, exist_ok=True)
        Path(j["done_file"]).write_text("{}")


def t9_runner():
    out = {}
    r, c, jobs, work = runner_fixture("happy")
    check(run_ok(c, jobs) == 0 and order(work) == A.RUN_JOBS, f"A1.2 order {order(work)}")
    check(order(work)[:7] == A.TS_JOBS and order(work)[7:] == A.A12_JOBS, "A1's seven jobs first, then A1.2's fifteen")
    cpu6 = [f"a12_cpu_{k}_{d}" for k in ("MERGE_N", "LABELMAP") for d in A.DRAWS]
    stock9 = [n for d in A.DRAWS
              for n in (f"a12_prep_STOCK_{d}", f"a12_train_STOCK_{d}", f"a12_pred_STOCK_{d}__word_test30")]
    check(order(work)[7:13] == cpu6 and order(work)[13:] == stock9 and A.A12_CPU_JOBS == cpu6 and A.A12_STOCK_JOBS == stock9,
          f"A1.2 order: the six CPU arm jobs (MERGE_N, then LABELMAP) before the nine STOCK jobs: {order(work)[7:]}")
    by = {j["name"]: j for j in jobs}
    for n in A.RUN_JOBS:
        env = json.load(open(work / f"{n}.env.json"))
        if by[n]["gpu"]:
            check(env == A.GPU_ENV, f"GPU-1 environment of {n}: {env}")
        else:
            check(env["CUDA_VISIBLE_DEVICES"] == "" and env["PYTORCH_CUDA_ALLOC_CONF"] is None, f"CPU environment of {n}")
        check(json.load(open(c.state_dir / f"{n}.json"))["done_sha256"] == F.sha256(by[n]["done_file"]),
              "DONE sha256 recorded")
    check(sorted(n for n in A.RUN_JOBS if not by[n]["gpu"]) == sorted(
        [f"a12_prep_STOCK_{d}" for d in A.DRAWS] + [f"a12_cpu_{k}_{d}" for k in ("MERGE_N", "LABELMAP") for d in A.DRAWS]),
          "the CPU jobs")
    check(notrun_lines(c) == [], "no NOT RUN")
    check(run_ok(c, jobs) == 0 and order(work) == A.RUN_JOBS, "a second run skips DONE jobs")
    out["order_env"] = ("22 jobs in A1.2 order (A1's 7, the 6 CPU arms, the 9 STOCK jobs); GPU env " + json.dumps(A.GPU_ENV)
                        + "; CPU env CUDA_VISIBLE_DEVICES=''")

    r, c, jobs, work = runner_fixture("rerun_once", plan={"pfinC_train_F1b_d2": {"fail_times": 1}})
    check(run_ok(c, jobs) == 0 and order(work).count("pfinC_train_F1b_d2") == 2, "rerun once")
    st = json.load(open(c.state_dir / "pfinC_train_F1b_d2.json"))
    check(st["attempts"] == 2 and st["failures"] == 1 and notrun_lines(c) == [], f"state {st}")
    out["rerun_once"] = "one failure -> one unchanged rerun -> DONE"

    r, c, jobs, work = runner_fixture("double_failure", plan={"pfinC_pred_word_F1b_d3__word_test30": {"fail_times": 9}})
    check(run_ok(c, jobs) == 0 and order(work).count("pfinC_pred_word_F1b_d3__word_test30") == 2, "two attempts only")
    nl = notrun_lines(c)
    check(len(nl) == 2 and nl[0].startswith("NOTRUN word_A1 F1b_d3 pfinC_pred_word_F1b_d3__word_test30 failed twice")
          and nl[1].startswith("NOTRUN word_A1 MERGE_N_d3 its input arm F1b_d3 was declared NOT RUN"), f"{nl}")
    check("a12_cpu_MERGE_N_d3" not in order(work) and "a12_cpu_LABELMAP_d3" in order(work)
          and "a12_train_STOCK_d3" in order(work), "only MERGE_N_d3 depends on F1b_d3")
    check(run_ok(c, jobs) == 0 and order(work).count("pfinC_pred_word_F1b_d3__word_test30") == 2, "never a third run")
    out["double_failure"] = nl[0]

    r, c, jobs, work = runner_fixture("required_failure", plan={"pfinC_train_F1b_d1": {"fail_times": 9}})
    check(run_ok(c, jobs) == 1, "required failure -> exit 1")
    nl = notrun_lines(c)
    check(any(x.startswith("NOTRUN word_A1 F1b_d1") for x in nl) and any(x.startswith("NOTRUN word_A1 EXPERIMENT")
                                                                          for x in nl), f"{nl}")
    check(order(work) == ["pfinC_pred_F0__word_test30", "pfinC_train_F1b_d1", "pfinC_train_F1b_d1"], f"{order(work)}")
    expect_refuse(lambda: run_ok(c, jobs), "experiment was declared NOT RUN")
    out["required_failure"] = "; ".join(nl)

    r, c, jobs, work = runner_fixture("cutoff_passed", cutoff=datetime.now() - timedelta(hours=1))
    mark_done(jobs, 3)
    check(run_ok(c, jobs) == 0 and order(work) == [], "nothing launched after the cutoff")
    nl = notrun_lines(c)
    arms_nr = [x.split()[2] for x in nl]
    check(nl[0].startswith("NOTRUN word_A1 F1b_d2 not DONE by the cutoff") and nl[1].startswith("NOTRUN word_A1 F1b_d3"),
          f"{nl}")
    check(arms_nr == ["F1b_d2", "F1b_d3"] + A.MERGE_ARMS + A.LABELMAP_ARMS + A.STOCK_ARMS,
          f"every arm not DONE, in the A1.2 order (CPU arms before STOCK): {arms_nr}")
    check(all("EXPERIMENT" not in x for x in nl), "A1.2 arms NOT RUN never stop the experiment")
    out["cutoff_passed"] = nl[:3]

    r, c, jobs, work = runner_fixture("cutoff_nothing_done", cutoff=datetime.now() - timedelta(hours=1))
    check(run_ok(c, jobs) == 1 and order(work) == [], "F0 not done by the cutoff -> experiment NOT RUN")
    nl = notrun_lines(c)
    check(nl[0].startswith("NOTRUN word_A1 F0 not DONE by the cutoff") and nl[1].startswith("NOTRUN word_A1 EXPERIMENT"),
          f"{nl}")
    out["cutoff_F0"] = nl

    r, c, jobs, work = runner_fixture("cutoff_during_job", plan={"pfinC_train_F1b_d3": {"sleep": 60}},
                                      cutoff=datetime.now() + timedelta(seconds=8))
    mark_done(jobs, 5)
    t0 = time.time()
    check(run_ok(c, jobs) == 0 and time.time() - t0 < 50, "the running job is stopped at the cutoff")
    st = json.load(open(c.state_dir / "pfinC_train_F1b_d3.json"))
    check(st["history"][0]["stopped_at_cutoff"] and st["failures"] == 0, f"{st}")
    check(notrun_lines(c)[0].startswith("NOTRUN word_A1 F1b_d3 not DONE by the cutoff"), f"{notrun_lines(c)}")
    out["cutoff_during_job"] = notrun_lines(c)[0]

    r, c, jobs, work = runner_fixture("no_osf", osf=False)
    expect_refuse(lambda: run_ok(c, jobs), "does not exist")
    check(order(work) == [] and not c.state_dir.exists(), "nothing launched or written before the OSF gate")
    out["no_osf"] = "refused"

    r, c, jobs, work = runner_fixture("queue_v8_running")
    q8 = work / "queue_v8.py"
    q8.write_text("import time\ntime.sleep(60)\n")
    dummy = subprocess.Popen(["/usr/bin/python3", "-I", str(q8)])
    try:
        time.sleep(0.5)
        msg = expect_refuse(lambda: RUN.run(c, jobs, used_fn=lambda: 0, log=lambda m: None), "queue_v8.py is running")
        check(str(dummy.pid) in msg, "the queue_v8 pid is named")
    finally:
        dummy.kill()
        dummy.wait()
    check(order(work) == [], "nothing launched while queue_v8 runs")
    out["queue_v8_running"] = msg[:70]

    r, c, jobs, work = runner_fixture("decisions_missing", decisions=False)
    out["decisions_missing"] = expect_refuse(lambda: run_ok(c, jobs), "'drop' lacks the A1 rows")[:70]

    r, c, jobs, work = runner_fixture("disk", min_free_gb=1e9)
    expect_refuse(lambda: run_ok(c, jobs), "GB free", code=3)
    check(order(work) == [] and not (c.state_dir / "pfinC_pred_F0__word_test30.json").exists(), "no attempt counted")
    out["disk"] = "refused before launch (code 3)"

    r, c, jobs, work = runner_fixture("gpu_busy")
    expect_refuse(lambda: run_ok(c, jobs, used_fn=lambda: 9000, wait_s=0.3, poll_s=0.1), "GPU 1 not idle", code=3)
    check(order(work) == [], "nothing launched on a busy GPU")
    out["gpu_busy"] = "refused after the idle wait (code 3)"

    r, c, jobs, work = runner_fixture("look_exists")
    c.look_dir.mkdir(parents=True, exist_ok=True)
    (c.look_dir / "GT_OPENED").write_text("{}")
    expect_refuse(lambda: run_ok(c, jobs), "no second look")
    out["look_exists"] = "refused"

    r, c, jobs, work = runner_fixture("lock_held")
    c.state_dir.mkdir(parents=True, exist_ok=True)
    fd = os.open(c.state_dir / "runner.lock", os.O_CREAT | os.O_RDWR, 0o644)
    pid = os.fork()
    if pid == 0:
        fcntl.flock(fd, fcntl.LOCK_EX)
        time.sleep(5)
        os._exit(0)
    try:
        time.sleep(0.5)
        expect_refuse(lambda: run_ok(c, jobs), "holds the lock", code=3)
    finally:
        os.waitpid(pid, 0)
        os.close(fd)
    out["lock_held"] = "refused (code 3)"

    r, c, jobs, work = runner_fixture("prep_missing", prep=False)
    expect_refuse(lambda: run_ok(c, jobs), "prep jobs not done")
    out["prep_missing"] = "refused"
    return out


def hold_lock_in_child(path, seconds):
    """Fork a child that holds LOCK_EX on path for `seconds`; returns its pid once the lock is held."""
    r_, w_ = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r_)
        fd = os.open(path, os.O_RDONLY)
        fcntl.flock(fd, fcntl.LOCK_EX)
        os.write(w_, b"1")
        time.sleep(seconds)
        os._exit(0)
    os.close(w_)
    os.read(r_, 1)
    os.close(r_)
    return pid


def t17_runner_extras():
    out = {}
    r, c, jobs, work = runner_fixture("cutoff_after_idle_wait", cutoff=datetime.now() + timedelta(seconds=3))
    mark_done(jobs, 5)
    t_end = time.time() + 3.8
    check(run_ok(c, jobs, used_fn=lambda: 9000 if time.time() < t_end else 0, wait_s=30, poll_s=0.2) == 0,
          "cutoff after the idle wait: finished")
    nl = notrun_lines(c)
    check(order(work) == [] and "the cutoff passed during the GPU lock / idle wait" in nl[0]
          and nl[0].startswith("NOTRUN word_A1 F1b_d3"), f"{order(work)} {nl}")
    check([x.split()[2] for x in nl[1:]] == A.MERGE_ARMS + A.LABELMAP_ARMS + A.STOCK_ARMS,
          f"A1.2 arms after the cutoff, in the A1.2 order (CPU arms before STOCK) {nl}")
    check(not (c.state_dir / "pfinC_train_F1b_d3.json").exists(), "no attempt counted")
    out["cutoff_after_idle_wait"] = nl[0]

    r, c, jobs, work = runner_fixture("cutoff_after_lock_wait", cutoff=datetime.now() + timedelta(seconds=3))
    mark_done(jobs, 5)
    pid = hold_lock_in_child(c.gpu_lock, 5)
    try:
        check(run_ok(c, jobs, lock_wait_s=30, lock_poll_s=0.2) == 0, "cutoff after the lock wait")
    finally:
        os.waitpid(pid, 0)
    nl = notrun_lines(c)
    check(order(work) == [] and "during the GPU lock / idle wait" in nl[0] and nl[0].startswith("NOTRUN word_A1 F1b_d3")
          and len(nl) == 10, f"{nl}")
    out["cutoff_after_lock_wait"] = nl[0]

    r, c, jobs, work = runner_fixture("cutoff_F0_after_wait", cutoff=datetime.now() + timedelta(seconds=3))
    t_end = time.time() + 3.8
    check(run_ok(c, jobs, used_fn=lambda: 9000 if time.time() < t_end else 0, wait_s=30, poll_s=0.2) == 1,
          "F0 not launched before the cutoff -> experiment NOT RUN")
    check(any("EXPERIMENT" in x for x in notrun_lines(c)) and order(work) == [], f"{notrun_lines(c)}")

    r, c, jobs, work = runner_fixture("done_not_rerun")
    check(run_ok(c, jobs) == 0 and order(work) == A.RUN_JOBS, "first run")
    victim = [j for j in jobs if j["name"] == "pfinC_pred_word_F1b_d2__word_test30"][0]
    os.remove(victim["done_file"])
    msg = expect_refuse(lambda: run_ok(c, jobs), "the runner never re-runs a DONE job")
    check(order(work) == A.RUN_JOBS, "nothing re-launched")
    out["done_file_missing"] = msg[:100]
    Path(victim["done_file"]).write_text("{}")
    Path(jobs[0]["done_file"]).write_text('{"rewritten": true}')
    msg = expect_refuse(lambda: run_ok(c, jobs), "DONE record changed")
    check(order(work) == A.RUN_JOBS, "nothing re-launched")
    out["done_record_changed"] = msg[:100]
    msg = expect_refuse(lambda: RUN.guard_not_done(jobs[1], {"history": []}), "never re-runs a DONE job")
    out["guard"] = msg[:80]

    r, c, jobs, work = runner_fixture("hand_stop", plan={"pfinC_train_F1b_d1": {"sleep": 30, "sleep_n": 1}})
    pid = os.fork()
    if pid == 0:
        code = 99
        try:
            for s in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
                signal.signal(s, RUN._on_term)
            code = run_ok(c, jobs)
        except RUN.HandStop as e:
            code = 50 + e.signum
        except BaseException:
            code = 98
        os._exit(code)
    t0 = time.time()
    while "pfinC_train_F1b_d1" not in order(work) and time.time() - t0 < 30:
        time.sleep(0.1)
    time.sleep(0.5)
    os.kill(pid, signal.SIGTERM)
    _, status = os.waitpid(pid, 0)
    check(os.WEXITSTATUS(status) == 50 + signal.SIGTERM, f"runner exit {os.WEXITSTATUS(status)}")
    time.sleep(1.0)
    hs = [ln for ln in c.log_text_lines() if " HANDSTOP word_A1 " in ln]
    check(len(hs) == 1 and "pfinC_train_F1b_d1 attempt 1 stopped by SIGTERM; not counted as a failed attempt" in hs[0],
          f"{hs}")
    st = json.load(open(c.state_dir / "pfinC_train_F1b_d1.json"))
    check(st["failures"] == 0 and st["attempts"] == 1 and st["history"][0]["hand_stopped"], f"{st}")
    check(not Path(jobs[1]["done_file"]).exists(), "the stopped job did not finish")
    msg = expect_refuse(lambda: run_ok(c, jobs), "have no HANDSTOP_REASON line")
    expect_refuse(lambda: RUN.log_hand_stop_reason(c, "   "), "non-empty reason")
    check(RUN.log_hand_stop_reason(c, "unit test: owner stopped the runner") == "pfinC_train_F1b_d1", "reason logged")
    expect_refuse(lambda: RUN.log_hand_stop_reason(c, "again"), "no HANDSTOP line is waiting")
    check(run_ok(c, jobs) == 0, "runs again after the reason")
    st = json.load(open(c.state_dir / "pfinC_train_F1b_d1.json"))
    check(st["attempts"] == 2 and st["failures"] == 0 and notrun_lines(c) == [], f"{st}")
    out["hand_stop"] = hs[0].split("] ", 1)[1][:120]
    out["hand_stop_gate"] = msg[:90]
    return out


def t18_gpu_flock():
    out = {}
    r, c, jobs, work = runner_fixture("flock_held_during_job", plan={"pfinC_pred_F0__word_test30": {"check_lock": True},
                                                                     "pfinC_train_F1b_d2": {"check_lock": True}})
    check(run_ok(c, jobs) == 0, "run")
    check((work / "pfinC_pred_F0__word_test30.locked").read_text() == "held"
          and (work / "pfinC_train_F1b_d2.locked").read_text() == "held", "the lock is held while the job runs")
    fd = os.open(c.gpu_lock, os.O_RDONLY)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.close(fd)
    check(c.gpu_lock.read_bytes() == LOCK_BYTES, "lock file content unchanged")
    out["held_during_job"] = "held (checked by the job with LOCK_NB); released after the run"

    r, c, jobs, work = runner_fixture("flock_wait")
    pid = hold_lock_in_child(c.gpu_lock, 2)
    t0 = time.time()
    try:
        check(run_ok(c, jobs, lock_wait_s=30, lock_poll_s=0.1) == 0 and order(work) == A.RUN_JOBS, "ran after the wait")
    finally:
        os.waitpid(pid, 0)
    check(time.time() - t0 >= 1.5, "the runner waited for the other job's lock")
    out["waits"] = f"waited {time.time() - t0:.1f} s, then ran the 22 jobs"

    r, c, jobs, work = runner_fixture("flock_timeout")
    pid = hold_lock_in_child(c.gpu_lock, 6)
    try:
        msg = expect_refuse(lambda: run_ok(c, jobs, lock_wait_s=1.0, lock_poll_s=0.1), "held by another job", code=3)
    finally:
        os.waitpid(pid, 0)
    check(order(work) == [] and not (c.state_dir / "pfinC_pred_F0__word_test30.json").exists(), "nothing launched")
    out["timeout"] = msg[:90]

    r, c, jobs, work = runner_fixture("flock_missing", lock=False)
    msg = expect_refuse(lambda: run_ok(c, jobs), "does not exist (it is never created here)", code=3)
    check(not c.gpu_lock.exists() and order(work) == [], "the lock file is not created")
    out["missing"] = msg[:90]
    check(r.joinpath("gpu1.lock") == c.gpu_lock and A.Ctx.real_ctx().gpu_lock == A.GPU_LOCK
          and str(A.GPU_LOCK) == "/home/user/Documents/Handpose_project/runs/method_pilot3/gpu1.lock",
          "the real context uses the shared lock path")
    check(A.GPU_LOCK_WAIT_S == 12 * 3600, "12 h wait")
    return out


def t19_queue_v8_match():
    yes = [["/home/user/Documents/Medical Imagining SSL/.venv/bin/python", "-u", "scripts/queue_v8.py"],
           ["python3", "-B", "/x/queue_v8.py", "--arm"], ["python3.10", "-X", "dev", "queue_v8.py"],
           ["python", "-m", "queue_v8"]]
    no = [["less", "scripts/queue_v8.py"], ["vim", "queue_v8.py"], ["python3", "-c", "pass", "queue_v8.py"],
          ["python3", "scripts/other.py", "queue_v8.py"], ["tail", "-f", "/x/queue_v8.py"], ["bash", "queue_v8.py"],
          ["python3", "-Bc", "pass", "queue_v8.py"]]
    for a in yes:
        check(A.runs_queue_v8(a), f"should match {a}")
    for a in no:
        check(not A.runs_queue_v8(a), f"false positive {a}")
    work = Path(SCR) / "t19"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    q8 = work / "queue_v8.py"
    q8.write_text("import time\ntime.sleep(30)\n")
    procs = {"python_script": subprocess.Popen(["/usr/bin/python3", "-I", str(q8)]),
             "tail": subprocess.Popen(["/usr/bin/tail", "-f", str(q8)], stdout=subprocess.DEVNULL),
             "python_c": subprocess.Popen(["/usr/bin/python3", "-I", "-c", "import time; time.sleep(30)", str(q8)])}
    try:
        time.sleep(0.7)
        pids = A.queue_v8_pids()
        check(procs["python_script"].pid in pids, "a python process executing queue_v8.py is found")
        check(procs["tail"].pid not in pids and procs["python_c"].pid not in pids, f"false positives in {pids}")
    finally:
        for p in procs.values():
            p.kill()
            p.wait()
    return {"matched": len(yes), "not_matched": len(no), "live": "python executing it found; tail / python -c not"}


# ---------------------------------------------------------------------------------------------------- queue_v8 / real mode
def q8_probe(code):
    py = str(F.VENV)
    full = ("import json, sys\nsys.dont_write_bytecode = True\nsys.argv = ['queue_v8']\n"
            f"sys.path.insert(0, {str(A.REPO / 'scripts')!r})\nimport queue_v8 as Q\nQ.DRY = True\n" + code)
    r = subprocess.run(["taskset", "-c", "8-15", py, "-B", "-c", full], capture_output=True, text=True, cwd=SCR,
                       timeout=600)
    check(r.returncode == 0, f"queue_v8 probe failed: {r.stderr[-800:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def t10_queue_v8():
    rows = q8_probe("d = dict(Q.DEFAULTS); d['prio'] = {}; d['drop'] = []\n"
                    "J = {j['name']: j for j in Q.plan(d)}\n"
                    f"print(json.dumps({{n: J[n] for n in {A.TS_JOBS!r}}}, default=str))")
    for j in A.ts_jobs():
        q = rows[j["name"]]
        check(q["interp"] == j["interp"] and q["args"] == j["args"] and q["cwd"] == str(A.REPO) and q["gpu"] is True
              and q["done_file"] == j["done_file"], f"{j['name']}: runner command != queue_v8 row")
    live = q8_probe("d = Q.decisions()\nnames = [j['name'] for j in Q.plan(d)]\n"
                    "print(json.dumps({'names': names, 'gpus': d.get('gpus')}))")
    present = [n for n in A.SCHEDULER_NEUTRALISED if n in live["names"]]
    check(present == [], f"A1 rows still in queue_v8's live table: {present}")
    pf = [n for n in live["names"] if n.startswith("pfinC_")]
    check(sorted(pf) == sorted(A.PREP_DONE_Q8), f"the only pfinC rows left are the three DONE prep rows: {pf}")
    full = q8_probe("d = dict(Q.DEFAULTS); d['prio'] = {}; d['drop'] = []\n"
                    "rows = [{'name': j['name'], 'args': [str(x) for x in j['args']]} for j in Q.plan(d)]\n"
                    "print(json.dumps(rows))")
    a12_named = [r["name"] for r in full if r["name"] in A.A12_JOBS or r["name"].startswith("a12_")]
    a12_scripts = [r["name"] for r in full + [{"name": n, "args": []} for n in live["names"]]
                   if any(k in " ".join(r["args"]) for k in ("word_a1_stock.py", "word_a1_cpu_arms.py", "word_a1_run.py",
                                                              "word_a1_score.py", "STOCK_d", "MERGE_N", "LABELMAP"))]
    check(a12_named == [] and a12_scripts == [], f"queue_v8 rows that would run A1.2 work: {a12_named} {a12_scripts}")
    check(not set(A.A12_JOBS) & set(live["names"]), "no A1.2 job in the live table")
    gate = A.scheduler_gate(A.Ctx.real_ctx(), pids_fn=lambda: [])
    return {"commands_equal": len(rows), "a1_rows_in_live_table": present, "pfinC_rows_left": pf, "gate": gate,
            "a12_rows_in_queue_v8_full_table": a12_named + a12_scripts, "queue_v8_rows_full_table": len(full)}


def t11_preconditions(osf_file, look_dir):
    """FIX-1: the real-mode test refuses to run at all once the OSF gate file or the look directory exists."""
    bad = [str(p) for p in (osf_file, look_dir) if os.path.lexists(p)]
    if bad:
        raise AssertionError(f"T11 REFUSES TO RUN: {bad} exist(s). The OSF gate is open or a look exists, so the real-mode "
                             f"test could start real jobs or a real look; it is for use before registration only")
    return True


def real_snapshot():
    paths = [A.W.DIR / "a1_run", A.W.DIR / "pred", A.W.DIR / A.OSF_FILE_NAME, F.LOGD / "final_word_a1_run.log",
             F.LOGD / "final_word_a1_score.log", A.W.DIR / "models", A.W.DIR / "prep_stock", A.W.DIR / "nnunet_stock",
             A.W.DIR / "labelmap"] + [F.LOGD / f"final_train_STOCK_{d}.log" for d in A.DRAWS] + [
        F.LOGD / f"final_predict_word_STOCK_{d}__word_test30.log" for d in A.DRAWS] + [
        F.LOGD / f"final_word_a12_{k}_{d}.log" for k in ("MERGE_N", "LABELMAP") for d in A.DRAWS] + [
        F.LOGD / f"final_word_a12_stock_{st}_{d}.log" for st in ("prep", "train", "predict") for d in A.DRAWS]
    return {"exists": {str(p): p.exists() for p in paths}, "final_prereg_log": F.sha256(F.PREREG_LOG),
            "decisions": F.sha256(A.REPO / "outputs/logs/q8/decisions.json")}


def t11_real_mode():
    ctx = A.Ctx.real_ctx()
    t11_preconditions(ctx.osf_file, ctx.look_dir)
    before = real_snapshot()
    py = str(F.VENV)
    out = {}
    reps = {}
    for name in ("word_a1_score.py", "word_a1_run.py"):
        cmd = ["taskset", "-c", "8-15", py, "-B", str(HERE / name), "--check"]
        check(cmd[-1] == "--check", "T11 runs the scripts in --check mode only")
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=SCR, timeout=600)
        rep = json.loads(r.stdout)
        check(r.returncode == 2 and rep["2_osf_gate"]["ok"] is False and "does not exist" in rep["2_osf_gate"]["detail"],
              f"{name} --check: rc {r.returncode} {rep.get('2_osf_gate')}")
        reps[name] = rep
    sc, rn = reps["word_a1_score.py"], reps["word_a1_run.py"]
    check(sc["1_no_second_look"]["ok"] is True and sc["3b_labels_zip_size"]["ok"] is None, f"scorer --check {sc}")
    check(all(rn[k]["ok"] is True for k in ("1_no_look", "4_scheduler", "5_kst", "6_prep_done",
                                            "7_experiment_not_declared_not_run", "8_hand_stops_have_reasons",
                                            "9_a12_inputs")), f"runner --check {rn}")
    check(sc["4_arms"]["ok"] is False and "required arm F0" in sc["4_arms"]["detail"], f"scorer arms gate {sc['4_arms']}")
    check(set(rn["jobs"]) == set(A.RUN_JOBS) and all(not v["done"] for v in rn["jobs"].values()), "22 jobs, none DONE")
    check(rn["gpu1_lock_file"]["ok"] is True, "the shared GPU-1 lock file exists")
    for name, rep in reps.items():
        out[name] = {k: v["ok"] for k, v in rep.items() if isinstance(v, dict) and "ok" in v}
    out["note"] = ("3_integrity is True only after the freeze (CODE lines); the scorer's 4_arms fails until the runner "
                   "has run")
    after = real_snapshot()
    check(before == after, f"real files changed: {before} -> {after}")
    out["real_files_unchanged"] = True
    return out


def t20_t11_refusal():
    root = Path(SCR) / "t20"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    osf, look = root / "A1_osf_registration.txt", root / "look"
    check(t11_preconditions(osf, look), "neither exists: T11 may run")
    osf.write_text("https://osf.io/abc12\n")
    try:
        t11_preconditions(osf, look)
        raise RuntimeError("T11 did not refuse with the gate file present")
    except AssertionError as e:
        check("T11 REFUSES TO RUN" in str(e), str(e))
    osf.unlink()
    look.mkdir()
    try:
        t11_preconditions(osf, look)
        raise RuntimeError("T11 did not refuse with the look directory present")
    except AssertionError as e:
        check("T11 REFUSES TO RUN" in str(e), str(e))
    src = Path(__file__).read_text()
    body = src[src.index("def t11_real_mode"):src.index("def t20_t11_refusal")]
    check("t11_preconditions(ctx.osf_file, ctx.look_dir)" in body.split("before = real_snapshot()")[0],
          "T11 checks its preconditions before anything else")
    return {"gate_file": "refused", "look_dir": "refused"}


def t12_style():
    bad = []
    for p in sorted(HERE.glob("*.py")):
        with open(p, "rb") as fh:
            for tok in tokenize.tokenize(fh.readline):
                if tok.type == tokenize.COMMENT and tok.line[:tok.start[1]].strip():
                    bad.append(f"{p.name}:{tok.start[0]}")
    for p in sorted(HERE.glob("*.service")):
        for i, ln in enumerate(p.read_text().splitlines(), 1):
            if "#" in ln and not ln.lstrip().startswith("#"):
                bad.append(f"{p.name}:{i}")
            if ln.lstrip().startswith("#") and "install " in ln and "install -D " not in ln:
                bad.append(f"{p.name}:{i} (install without -D)")
    check(not bad, f"comments on code lines: {bad}")
    look = (HERE / "ssl-word-a1-look.service").read_text()
    check("MemoryMax=100G" in look and "MemoryMax=30G" not in look, "look unit memory limit")
    return {"files": sorted(p.name for p in list(HERE.glob("*.py")) + list(HERE.glob("*.service")))}


# ---------------------------------------------------------------------------------------------------- T21 rehearsal
GUARD_PROBE = r"""
import os, sys
sys.dont_write_bytecode = True
sys.path.insert(0, sys.argv[1])
import word_a1_rehearse as RH
prot, proj, okdir = sys.argv[2:5]
RH.install_guard(protected=[prot], no_write_root=proj)
res = {}
def tryit(name, fn):
    try:
        fn()
        res[name] = "allowed"
    except PermissionError:
        res[name] = "blocked"
tryit("read_protected", lambda: open(os.path.join(prot, "labels.zip"), "rb"))
tryit("list_protected", lambda: os.listdir(prot))
tryit("write_project", lambda: open(os.path.join(proj, "x.txt"), "w"))
tryit("mkdir_project", lambda: os.mkdir(os.path.join(proj, "d")))
tryit("write_scratch", lambda: open(os.path.join(okdir, "ok.txt"), "w").close())
tryit("read_project", lambda: open(os.path.join(proj, "readable.txt")).read())
pid = os.fork()
if pid == 0:
    try:
        open(os.path.join(prot, "labels.zip"), "rb")
        os._exit(1)
    except PermissionError:
        os._exit(0)
_, st = os.waitpid(pid, 0)
res["forked_worker_read_protected"] = "blocked" if os.WEXITSTATUS(st) == 0 else "allowed"
import json
print(json.dumps(res))
"""


REHEARSE_PROBE = r"""
import json, sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, sys.argv[1])
import word_a1_rehearse as RH
import word_common as W
root, scratch = Path(sys.argv[2]), Path(sys.argv[3])
W.DIR = root / "C_word"
W.IMAGES = root / "images"
W.PREP = root / "C_word/prep"
W.DOWNLOAD = root / "C_word/download.json"
W.LOOK = root / "C_word/look"
W.LABELS_TS_ZIP = root / "archive_labelsTs/WORD_V0.1.0_labelsTs.zip"
RH.install_guard(protected=[root / "archive_labelsTs", root / "C_word/look"], no_write_root=root)
rep = RH.rehearse(scratch, pred_dir=root / "C_word/pred", workers=2, result_timeout_s=60, log=lambda m: None)
print(json.dumps({"equal": rep["continuation_reanalysis_equal"], "n": rep["n_scans"], "workers": rep["workers"]}))
"""


def t21_rehearsal():
    out = {}
    check(RH.PROTECTED == [os.path.normpath(str(p)) for p in A.REAL_LABEL_PATHS + [A.REAL_LOOK]],
          "the guard protects the real label folders and the real look directory")
    check(str(A.REAL_LOOK).endswith("outputs/results/vccor/final/C_word/look")
          and str(A.REAL_LABELS_ZIP).endswith("MagicNet/data/word_v010/archive_labelsTs/WORD_V0.1.0_labelsTs.zip"),
          "real paths as registered")
    refused = {}
    for nm, p in (("real_look", A.REAL_LOOK), ("real_labels_zip", A.REAL_LABELS_ZIP), ("project", A.REPO / "x"),
                  ("home", Path("/home/user/word_a1_rehearsal"))):
        refused[nm] = expect_refuse(lambda: RH.check_scratch(p), "rehearsal: --scratch")[:70]
    full = Path(SCR) / "t21_nonempty"
    shutil.rmtree(full, ignore_errors=True)
    full.mkdir(parents=True)
    (full / "x").write_text("x")
    refused["non_empty"] = expect_refuse(lambda: RH.check_scratch(full), "is not empty")[:70]
    refused["pred_dir_protected"] = expect_refuse(lambda: RH.rehearsal_ctx(Path(SCR) / "t21_x", A.REAL_LABEL_PATHS[0]),
                                                  "protected path")[:70]
    check(A.touches_real_labels_or_look(A.REAL_LOOK / "results.json")
          and A.touches_real_labels_or_look(A.REAL_LABELS_ZIP) and not A.touches_real_labels_or_look(Path(SCR)),
          "touches_real_labels_or_look")
    for nm, p in (("real_look", A.REAL_LOOK), ("project", A.REPO / "outputs")):
        r = subprocess.run([str(F.VENV), "-B", str(HERE / "word_a1_rehearse.py"), "--scratch", str(p)],
                           capture_output=True, text=True, cwd=SCR, timeout=300)
        check(r.returncode == 2 and "[REFUSE] rehearsal: --scratch" in r.stderr, f"CLI {nm}: {r.returncode} {r.stderr[-300:]}")
        refused[f"cli_{nm}"] = "refused"
    r = subprocess.run([str(F.VENV), "-B", str(HERE / "word_a1_rehearse.py"), "--scratch", str(Path(SCR) / "t21_cli"),
                        "--pred-dir", str(A.W.DIR / "pred")], capture_output=True, text=True, cwd=SCR, timeout=300)
    check(r.returncode == 2 and "--pred-dir is for tests" in r.stderr, f"CLI pred-dir {r.stderr[-300:]}")
    refused["cli_real_pred_dir_override"] = "refused"
    out["refused"] = refused

    g = Path(SCR) / "t21_guard"
    shutil.rmtree(g, ignore_errors=True)
    for d in ("prot", "proj", "ok"):
        (g / d).mkdir(parents=True)
    (g / "prot/labels.zip").write_bytes(b"fake")
    (g / "proj/readable.txt").write_text("r")
    r = subprocess.run([str(F.VENV), "-B", "-c", GUARD_PROBE, str(HERE), str(g / "prot"), str(g / "proj"), str(g / "ok")],
                       capture_output=True, text=True, cwd=SCR, timeout=300)
    check(r.returncode == 0, f"guard probe failed: {r.stderr[-800:]}")
    gres = json.loads(r.stdout.strip().splitlines()[-1])
    want = {"read_protected": "blocked", "list_protected": "blocked", "write_project": "blocked",
            "mkdir_project": "blocked", "write_scratch": "allowed", "read_project": "allowed",
            "forked_worker_read_protected": "blocked"}
    check(gres == want, f"guard {gres}")
    out["guard"] = gres

    root = Path(SCR) / "t21_word"
    shutil.rmtree(root, ignore_errors=True)
    build_word(root)
    scratch = Path(SCR) / "t21_rehearsal_scratch"
    shutil.rmtree(scratch, ignore_errors=True)
    real_zip_cls = zipfile.ZipFile

    def no_zip(*a, **k):
        raise AssertionError("the rehearsal opened a zip")

    with patched_word(root):
        def never(cid, marker):
            raise AssertionError("the rehearsal called the real reference reader")

        W.reference = never
        zipfile.ZipFile = no_zip
        try:
            rep = RH.rehearse(scratch, pred_dir=root / "C_word/pred", workers=2, result_timeout_s=60, log=lambda m: None)
        finally:
            zipfile.ZipFile = real_zip_cls
        check(W.reference is never and W.LOOK == root / "C_word/look", "word_common restored after the rehearsal")
    check({k: getattr(W, k) for k in REAL_W} == REAL_W, "real word_common paths restored")
    check(rep["continuation_reanalysis_equal"] and rep["results_json_reparsed"] and rep["n_scans"] == 5
          and rep["arms"] == ["F0", "F1b_d1", "F1b_d2", "F1b_d3"] and rep["pervol_done_records"] == 5
          and rep["a12_secondary_error"] is None, f"report {rep}")
    check(not (root / "C_word/look").exists(), "the synthetic project's look directory was not created")
    res = json.loads((scratch / "C_word/look/results.json").read_text())
    check(res["secondary_S_TS"]["S"] is not None and res["secondary_S_TS"]["E0_zero_shot_error"] > 0,
          "stand-in reference = F0 shifted: zero-shot error > 0, S computable")
    check(close(res["descriptive_contrasts"]["d1"]["macro_penalised"]["mean"], res["P2_draw_dependence"]["per_draw"]["d1"]["M"]),
          "results consistent")
    check((scratch / "rehearsal_report.json").is_file(), "report written in the scratch folder")
    tags = [ln.split()[2] for ln in open(scratch / "logs/final_prereg.log")]
    check(tags == ["GT_OPENED", "RESULTS"], f"scratch log only: {tags}")
    out["rehearsal"] = {k: rep[k] for k in ("n_scans", "workers", "scoring_wall_s", "peak_child_rss_gb",
                                            "continuation_reanalysis_equal")}
    root2 = Path(SCR) / "t21_word_guarded"
    shutil.rmtree(root2, ignore_errors=True)
    build_word(root2)
    scratch2 = Path(SCR) / "t21_rehearsal_guarded_scratch"
    shutil.rmtree(scratch2, ignore_errors=True)
    r = subprocess.run([str(F.VENV), "-B", "-c", REHEARSE_PROBE, str(HERE), str(root2), str(scratch2)],
                       capture_output=True, text=True, cwd=SCR, timeout=600)
    check(r.returncode == 0, f"guarded rehearsal failed: {r.stderr[-1500:]}")
    g2 = json.loads(r.stdout.strip().splitlines()[-1])
    check(g2 == {"equal": True, "n": 5, "workers": 2}, f"guarded rehearsal {g2}")
    check(not (root2 / "C_word/look").exists(), "guarded rehearsal: the protected look was not created")
    out["rehearsal_with_guard_armed"] = g2
    return out


# ---------------------------------------------------------------------------------------------------- T22 A1.2 contrasts
def a12_records(n=30, g=(3.0, 1.8, 0.6), s=(1.0, 0.8, 0.4), mcost=None, lm=(2.4, 2.4, 2.4), lm_gb=(0.0, 0.0, -1.2),
                lm_all=None, s_noise=None, draws=("d1", "d2", "d3")):
    """F0 80 on every primary organ (colon 70, femoral heads 60). F1b_d: +g on the 12 primary organs, colon 72. STOCK_d: +s on
    the 12 primary organs (plus s_noise[i] per scan if given). MERGE_N_d: F1b_d with the per-organ costs mcost {organ index:
    cost} (default pancreas -1.5). LABELMAP_d: F0 with liver +lm and gallbladder +lm_gb (or +lm_all on every primary
    organ)."""
    mcost = {7: 1.5} if mcost is None else mcost
    ids = [f"{i:04d}" for i in range(1, n + 1)]
    R = {}
    for i, c in enumerate(ids):
        base = [80.0] * K
        base[9] = 70.0
        base[13] = base[14] = 60.0
        arms = {"F0": rec(base)}
        for j, d in enumerate(draws):
            jd = A.DRAWS.index(d)
            f1 = list(base)
            for o in W.PRIMARY:
                f1[o - 1] = 80.0 + g[jd]
            f1[9] = 72.0
            arms[f"F1b_{d}"] = rec(f1)
            st = list(base)
            for o in W.PRIMARY:
                st[o - 1] = 80.0 + s[jd] + (s_noise[i] if s_noise is not None else 0.0)
            arms[f"STOCK_{d}"] = rec(st)
            mn = list(f1)
            for k, v in mcost.items():
                mn[k] = f1[k] - (v[jd] if isinstance(v, (list, tuple)) else v)
            arms[f"MERGE_N_{d}"] = rec(mn)
            lmv = list(base)
            if lm_all is not None:
                for o in W.PRIMARY:
                    lmv[o - 1] = 80.0 + lm_all[jd]
            else:
                lmv[0] = 80.0 + lm[jd]
                lmv[5] = 80.0 + lm_gb[jd]
            arms[f"LABELMAP_{d}"] = rec(lmv)
        R[c] = frames(arms, [0.977, 0.977, 2.5])
    return R, ids


def ts_only(R):
    return {c: {"NATIVE": dict(fr["NATIVE"], arms={a: v for a, v in fr["NATIVE"]["arms"].items() if a in A.ARMS})}
            for c, fr in R.items()}


def merge_extras(ids, draws=("d1", "d2", "d3")):
    """Synthetic MERGE_N DONE records: prostate (22) 1.0 mL in F0, 0.8 in F1b, 0.8 shared (Dice 88.9, ln 0.8 -> FLAG);
    trachea (16) 5 / 5 / 4.9 mL (Dice 98, ln 1 -> PASS); thyroid (17) 0.5 mL (< 1 mL: not evaluated); portal vein (64) on
    two scans only (< 3: not evaluated); 100 voxels pancreas -> prostate changed per scan."""
    out = {}
    for d in draws:
        per = {}
        for i, c in enumerate(ids):
            nt = {"22": [1000, 800, 800], "16": [5000, 5000, 4900], "17": [500, 500, 500]}
            if i < 2:
                nt["64"] = [2000, 2000, 2000]
            per[c] = {"vox_ml": 0.001, "nontarget_f0_f1b_both": nt, "changed_vs_f1b": {"7->22": 100},
                      "n_changed_vs_f1b": 100}
        out[d] = {"per_scan_merge": per}
    return {"merge": out, "labelmap": {d: {"merged": {"64": 1}} for d in draws}, "rules": {}}


def indep_ratio_boot(num, den, nboot=10000, seed=20261020):
    """Independent implementation of the A1.2 Q bootstrap (boot_ratio's scheme: scans, then draws, per replicate)."""
    import numpy as np
    draws = sorted(num)
    cids = sorted(set.intersection(*[set(num[d]) for d in draws]))
    Nm = np.array([[num[d][c] for c in cids] for d in draws])
    Dn = np.array([[den[d][c] for c in cids] for d in draws])
    rng = np.random.default_rng(seed)
    n, k = len(cids), len(draws)
    reps = []
    for _ in range(nboot):
        w = rng.multinomial(n, np.full(n, 1.0 / n)).astype(float)
        u = rng.multinomial(k, np.full(k, 1.0 / k)).astype(float) if k > 1 else np.ones(1)
        a = float(u @ ((Nm @ w) / w.sum()) / u.sum())
        b = float(u @ ((Dn @ w) / w.sum()) / u.sum())
        if b > 0:
            reps.append(a / b)
    return [float(np.quantile(reps, 0.025)), float(np.quantile(reps, 0.975))]


def t22_a12_contrasts():
    import numpy as np
    out = {}
    R, ids = a12_records()
    have = set(A.ALL_ARMS)
    ex = merge_extras(ids)
    res = S.analyse_all(R, ids, have, None, {}, ex)
    a = res["A1_2"]
    ref = S.analyse_a1(ts_only(R), ids, set(A.ARMS))
    for k in ("P1_gain", "P1_organ_costs", "P2_draw_dependence", "outcome", "claims_allowed", "secondary_S_TS",
              "secondary_colon_rectum", "secondary_per_draw_per_organ_descriptive", "descriptive_contrasts"):
        check(json.dumps(F.clean_nan(res[k]), sort_keys=True, default=F._default)
              == json.dumps(F.clean_nan(ref[k]), sort_keys=True, default=F._default), f"{k} changed by the A1.2 arms")
    check(res["P1_gain"]["pass"] and close(res["P1_gain"]["M"], 1.8), "P1 PASS, M 1.8 (A1 arms only)")
    s1 = a["S1_F1b_minus_STOCK"]
    want = {"d1": 2.0, "d2": 1.0, "d3": 0.2}
    for d, v in want.items():
        pdv = s1["per_draw"][d]
        check(close(pdv["mean"], v) and close(pdv["ci"][0], v) and close(pdv["ci"][1], v) and pdv["seed"] == 20261022,
              f"S1 {d} {pdv}")
    per = {d: gains_by_scan_pair(R, ids, f"F1b_{d}", f"STOCK_{d}") for d in A.DRAWS}
    ci2 = indep_two_level(per)
    check(close(s1["pooled"]["M"], 3.2 / 3) and close(s1["pooled"]["ci"][0], ci2[0], 1e-9)
          and close(s1["pooled"]["ci"][1], ci2[1], 1e-9) and close(ci2[0], 0.2) and close(ci2[1], 2.0), f"S1 pooled {s1['pooled']}")
    check(s1["reading"] == "F1B_BETTER" and set(s1["per_draw_reading"].values()) == {"F1B_BETTER"}, "S1 reading")
    check(all(v["harm"] == "BENEFICIAL" for v in s1["per_organ"].values()), "S1 per organ")
    s1b = a["S1b_STOCK_minus_F0"]
    check(close(s1b["M"], 2.2 / 3) and s1b["pass"] and s1b["reading"] == "STOCK_GAINS" and s1b["k"] == 3
          and s1b["need_up"] == 20 and s1b["scans_up"] == 30 and close(s1b["ci"][0], 0.4), f"S1b {s1b}")
    s2 = a["S2_MERGE_N_minus_F1b"]
    check(close(s2["pooled"]["M"], -0.125) and close(s2["pooled"]["ci"][0], -0.125) and close(s2["pooled"]["ci"][1], -0.125),
          f"S2 macro {s2['pooled']}")
    check(s2["reading_ni"] == "MERGE_N_NON_INFERIOR" and s2["reading_harm"] == "MERGE_COSTS", "S2 readings")
    check(s2["organs_inferior"] == ["pancreas"] and s2["per_organ"]["pancreas"]["harm"] == "HARMFUL"
          and close(s2["per_organ"]["pancreas"]["M"], -1.5) and s2["per_organ"]["liver"]["ni"] == "NON-INFERIOR",
          f"S2 organs {s2['organs_inferior']}")
    check(close(s2["colon_rectum"]["colon_rectum"]["M"], 0.0) and close(s2["femoral_heads_merge_n_minus_f1b"]["femur_head_left"]["M"], 0.0),
          "S2 colon / femoral heads")
    mc = s2["merged_classes"]["d1"]
    check(set(mc["classes"]) == {"prostate", "trachea"}, f"rule B classes {mc['classes']}")
    pr, tr = mc["classes"]["prostate"], mc["classes"]["trachea"]
    check(close(pr["median_dice_f1b_vs_f0"], 200 * 800 / 1800) and close(pr["median_ln_volume_ratio"], math.log(0.8))
          and pr["rule_B"] == "FLAG" and pr["n_scans"] == 30, f"prostate {pr}")
    check(close(tr["median_dice_f1b_vs_f0"], 98.0) and close(tr["median_ln_volume_ratio"], 0.0) and tr["rule_B"] == "PASS",
          f"trachea {tr}")
    check(mc["flagged"] == ["prostate"] and mc["voxels_merge_n_differs_from_f1b"]["median"] == 100
          and mc["largest_changes_f1b_to_merge_n"] == {"pancreas -> prostate": 3000}, f"merged classes {mc}")
    s3 = a["S3_LABELMAP_minus_F0"]
    check(close(s3["pooled"]["M"], 0.5 / 3) and close(s3["pooled"]["ci"][0], 0.1) and close(s3["pooled"]["ci"][1], 0.2)
          and s3["reading"] == "RELABEL_GAINS", f"S3 {s3['pooled']}")
    check(close(s3["S_LM"]["S"], (20 - (19.8 + 19.8 + 19.9) / 3) / 20) and close(s3["S_LM"]["ci"][0], 0.005)
          and close(s3["S_LM"]["ci"][1], 0.01) and s3["S_LM"]["present"], f"S_LM {s3['S_LM']}")
    check(s3["per_organ"]["liver"]["harm"] == "BENEFICIAL" and close(s3["per_organ"]["gallbladder"]["M"], -0.4)
          and s3["per_organ"]["gallbladder"]["ni"] == "INCONCLUSIVE" and s3["per_organ"]["gallbladder"]["harm"] == "NEUTRAL",
          "S3 organs")
    check(s3["rules"]["d1"] == {"portal_vein_and_splenic_vein": "liver"} and s3["rules_identical_across_draws"], "S3 rules")
    s4 = a["S4_F1b_minus_LABELMAP"]
    check(close(s4["pooled"]["M"], 4.9 / 3) and close(s4["pooled"]["ci"][0], 0.5) and close(s4["pooled"]["ci"][1], 2.8)
          and s4["reading"] == "FT_EXCEEDS_RELABEL", f"S4 {s4['pooled']}")
    q = s4["Q_share_of_F1b_gain_recovered_by_relabelling"]
    num = {d: gains_by_scan_pair(R, ids, f"LABELMAP_{d}", "F0") for d in A.DRAWS}
    den = {d: gains_by_scan_pair(R, ids, f"F1b_{d}", "F0") for d in A.DRAWS}
    qi = indep_ratio_boot(num, den)
    check(close(q["Q"], (0.5 / 3) / 1.8) and close(q["ci"][0], qi[0], 1e-12) and close(q["ci"][1], qi[1], 1e-12)
          and 0.2 / 3.0 - 1e-9 <= q["ci"][0] <= q["ci"][1] <= 0.1 / 0.6 + 1e-9 and q["replicates_dropped"] == 0, f"Q {q}")
    ad = a["descriptive_absolute_dice_vista3d_link"]
    check(close(ad["F0"]["macro_penalised_12"], 80.0) and close(ad["STOCK"]["macro_penalised_12"], 80 + 2.2 / 3)
          and close(ad["F1b"]["macro_penalised_12"], 81.8) and close(ad["MERGE_N"]["macro_penalised_12"], 81.675)
          and close(ad["LABELMAP"]["macro_penalised_12"], 80 + 0.5 / 3)
          and close(ad["F0"]["macro_reference_present_15"], (12 * 80 + 70 + 60 + 60) / 15), f"absolute Dice {ad}")
    check(a["readings"] == {"S1": "F1B_BETTER", "S1b": "STOCK_GAINS", "S2_ni": "MERGE_N_NON_INFERIOR",
                            "S2_harm": "MERGE_COSTS", "S3": "RELABEL_GAINS", "S3_S_LM_present": True,
                            "S4": "FT_EXCEEDS_RELABEL"}, f"readings {a['readings']}")
    ca = a["claims_allowed"]
    check(ca["the gentle fine-tune (F1b) beats nnU-Net's stock fine-tune on WORD"]
          and not ca["nnU-Net's stock fine-tune beats the gentle fine-tune on WORD"]
          and ca["organs to name as a cost of the merge"] == ["pancreas"]
          and not ca["any A1.2 contrast changes P1, P2 or the outcome label"], f"claims {ca}")
    out["scenario"] = a["readings"]

    R2, ids2 = a12_records(s=(4.0, 2.5, 1.0), lm_all=(4.0, 2.5, 1.0), mcost={o - 1: 1.5 for o in W.PRIMARY})
    b = S.analyse_all(R2, ids2, have, None, {}, merge_extras(ids2))["A1_2"]
    check(b["readings"]["S1"] == "STOCK_BETTER" and b["readings"]["S4"] == "RELABEL_EXCEEDS_FT"
          and b["readings"]["S2_ni"] == "MERGE_N_INFERIOR" and b["readings"]["S2_harm"] == "MERGE_COSTS"
          and len(b["S2_MERGE_N_minus_F1b"]["organs_inferior"]) == 12, f"reversed readings {b['readings']}")
    R3, ids3 = a12_records(s=(2.0, 2.0, 0.6), lm=(-1.2, -1.2, -1.2), lm_gb=(0.0, 0.0, 0.0),
                           mcost={o - 1: (1.5, 0.5, 0.5) for o in W.PRIMARY})
    c3 = S.analyse_all(R3, ids3, have, None, {}, merge_extras(ids3))["A1_2"]
    check(c3["readings"]["S1"] == "NO_DIFFERENCE_DETECTED" and c3["readings"]["S3"] == "RELABEL_LOSES"
          and c3["readings"]["S2_ni"] == "INCONCLUSIVE" and not c3["readings"]["S3_S_LM_present"]
          and c3["S1_F1b_minus_STOCK"]["per_draw_reading"] == {"d1": "F1B_BETTER", "d2": "STOCK_BETTER",
                                                                "d3": "NO_DIFFERENCE_DETECTED"}, f"mixed readings {c3['readings']}")
    out["other_readings"] = {"reversed": b["readings"], "mixed": c3["readings"]}

    rng = np.random.default_rng(7)
    noise = list(rng.normal(0.0, 0.8, 30))
    R4, ids4 = a12_records(s_noise=noise, draws=("d1",))
    one = S.analyse_all(R4, ids4, {"F0", "F1b_d1", "STOCK_d1", "MERGE_N_d1", "LABELMAP_d1"}, None, {},
                        merge_extras(ids4, ("d1",)))["A1_2"]
    p1 = one["S1_F1b_minus_STOCK"]["pooled"]
    want_ci = indep_cluster_boot(gains_by_scan_pair(R4, ids4, "F1b_d1", "STOCK_d1"), seed=20261022)
    check(p1["method"].startswith("scan bootstrap") and p1["seed"] == 20261022 and close(p1["ci"][0], want_ci[0], 1e-9)
          and close(p1["ci"][1], want_ci[1], 1e-9) and p1["ci"] == one["S1_F1b_minus_STOCK"]["per_draw"]["d1"]["ci"],
          f"one draw -> A1's scan bootstrap {p1}")
    check(set(one["arms_not_run"]) == {"STOCK_d2", "STOCK_d3", "MERGE_N_d2", "MERGE_N_d3", "LABELMAP_d2", "LABELMAP_d3"},
          f"A1.2 arms not run {one['arms_not_run']}")
    out["one_draw"] = {"ci": p1["ci"], "independent": want_ci}

    none = S.analyse_all(ts_only(R), ids, set(A.ARMS), None, {}, {})["A1_2"]
    check(set(none["readings"].values()) <= {"NOT RUN", False} and none["readings"]["S1"] == "NOT RUN"
          and not any(v for k, v in none["claims_allowed"].items() if k != "organs to name as a cost of the merge")
          and set(none["arms_not_run"]) == set(A.A12_ARMS), f"no A1.2 arm {none['readings']}")
    R5, ids5 = a12_records(g=(0.1, 0.1, 0.1), s=(0.0, 0.0, 0.0), lm=(0.6, 0.6, 0.6), lm_gb=(0.0, 0.0, 0.0))
    r5 = S.analyse_all(R5, ids5, have, None, {}, merge_extras(ids5))
    check(not r5["P1_gain"]["pass"] and r5["A1_2"]["S4_F1b_minus_LABELMAP"]["Q_share_of_F1b_gain_recovered_by_relabelling"]["Q"] is None,
          "Q only under P1 PASS")
    out["not_run_and_q"] = "readings NOT RUN without A1.2 arms; Q NOT EVALUABLE without P1 PASS"
    return out


def gains_by_scan_pair(R, ids, a, b, organs=None):
    organs = organs or W.PRIMARY
    out = {}
    for c in ids:
        x = R[c]["NATIVE"]["arms"][a]["dice"]
        y = R[c]["NATIVE"]["arms"][b]["dice"]
        out[c] = sum(x[o - 1] for o in organs) / len(organs) - sum(y[o - 1] for o in organs) / len(organs)
    return out


# ---------------------------------------------------------------------------------------------------- T23 label-map rules
def ts_id(name):
    from totalsegmentator.map_to_binary import class_map
    return {v: k for k, v in class_map["total"].items()}[name]


def scan_arrays(triples):
    """(G, Tf) 1-D uint8 arrays from [(G label, TS total id, count)]."""
    import numpy as np
    g = np.concatenate([np.full(n, gl, np.uint8) for gl, _, n in triples])
    t = np.concatenate([np.full(n, tl, np.uint8) for _, tl, n in triples])
    return g, t


def t23_scans():
    """Four synthetic training scans (analysis ids for G, right adrenal as 17 in scan 1; TS 'total' ids for Tf). Designed
    mean shares of the reference-only voxels (divided by 4): liver <- portal / splenic vein 0.75, 0.40, 0, 0.05 -> 0.30;
    pancreas <- portal / splenic vein 0.20, 0.30 -> 0.125 (conflict: liver wins); pancreas <- IVC 0.80 -> 0.20; spleen and
    left kidney <- aorta 0.50 / 0.50 on scans 1 + 2 -> 0.25 each (tie: spleen, the lower id); stomach <- SVC 0.009 x 3
    (dropped, < 1 %) + 0.39 -> 0.0975 (not merged; 0.104 without the 1 % rule); duodenum <- small bowel 1.0 (an ontology class:
    never merged); gallbladder 10 voxels (absent: < 20); bladder <- femur 0.5 (ontology: never merged)."""
    pv, ivc, svc, ao = ts_id("portal_vein_and_splenic_vein"), ts_id("inferior_vena_cava"), ts_id("superior_vena_cava"), \
        ts_id("aorta")
    liv, pan, spl, kl = ts_id("liver"), ts_id("pancreas"), ts_id("spleen"), ts_id("kidney_left")
    sto, duo, sb, gb = ts_id("stomach"), ts_id("duodenum"), ts_id("small_bowel"), ts_id("gallbladder")
    adr_r, bl, fem = ts_id("adrenal_gland_right"), ts_id("urinary_bladder"), ts_id("femur_left")
    s1 = [(1, liv, 800), (1, pv, 150), (1, 0, 50), (8, pan, 400), (8, pv, 20), (8, ivc, 80), (2, spl, 100), (2, ao, 50),
          (2, 0, 50), (3, kl, 100), (3, ao, 50), (3, 0, 50), (5, sto, 1000), (5, svc, 9), (5, 0, 991), (9, duo, 200),
          (9, sb, 200), (6, gb, 10), (17, adr_r, 90), (17, 0, 10), (13, bl, 100), (13, fem, 50), (13, 0, 50), (0, pv, 300)]
    s2 = [(1, liv, 900), (1, pv, 40), (1, 0, 60), (8, pan, 400), (8, pv, 30), (8, 0, 70), (2, spl, 100), (2, ao, 50),
          (2, 0, 50), (3, kl, 100), (3, ao, 50), (3, 0, 50), (5, sto, 1000), (5, svc, 9), (5, 0, 991), (9, duo, 300)]
    s3 = [(1, liv, 1000), (8, pan, 500), (5, sto, 1000), (5, svc, 9), (5, 0, 991), (9, duo, 300)]
    s4 = [(1, liv, 900), (1, pv, 5), (1, 0, 95), (8, pan, 500), (5, sto, 1000), (5, svc, 390), (5, 0, 610), (9, duo, 300)]
    return [scan_arrays(x) for x in (s1, s2, s3, s4)]


def t23_labelmap_rules():
    import nibabel as nib
    import numpy as np
    out = {}
    name_lut = W.ts_lut()
    per = {}
    for i, (G, Tf) in enumerate(t23_scans()):
        G = G.copy()
        G[G == W.ADRENAL_R_TMP] = W.ADRENAL
        per[f"s{i + 1}"] = CA.scan_shares(G, Tf, name_lut)
    pv, ivc, svc, ao = (ts_id(n) for n in ("portal_vein_and_splenic_vein", "inferior_vena_cava", "superior_vena_cava",
                                           "aorta"))
    sb, fem = ts_id("small_bowel"), ts_id("femur_left")
    check(close(per["s1"][1]["shares"][pv], 0.75) and close(per["s1"][1]["shares"][0], 0.25) and per["s1"][1]["n_ref_only"] == 200,
          f"liver shares scan 1 {per['s1'][1]}")
    check(per["s1"][6] == {"present": False, "nG": 10, "n_ref_only": 0, "shares": {}}, "gallbladder absent (< 20 voxels)")
    check(svc not in per["s1"][5]["shares"] and close(per["s4"][5]["shares"][svc], 0.39), "1 % truncation")
    check(per["s1"][12]["present"] and per["s1"][12]["n_ref_only"] == 10, "right adrenal 17 -> 12 counted as adrenal")
    r = CA.build_rules(per, name_lut, 4)
    ms = r["mean_shares"]
    check(close(ms["1"][str(pv)], 0.30) and close(ms["8"][str(pv)], 0.125) and close(ms["8"][str(ivc)], 0.20)
          and close(ms["2"][str(ao)], 0.25) and close(ms["3"][str(ao)], 0.25) and close(ms["5"][str(svc)], 0.0975),
          f"mean shares {ms}")
    check(str(sb) not in ms["9"] and str(fem) not in ms["13"] and "0" not in ms["1"], "ontology classes and background excluded")
    check(r["merged"] == {str(ao): 2, str(ivc): 8, str(pv): 1}, f"merged {r['merged']}")
    conf = {c["ts_class"]: c for c in r["conflicts"]}
    check(set(conf) == {pv, ao} and conf[pv]["assigned"] == 1 and conf[ao]["assigned"] == 2
          and [x[0] for x in conf[ao]["candidates"]] == [2, 3], f"conflicts {r['conflicts']}")
    lut = np.asarray(r["lut"], np.uint8)
    check(lut[pv] == 1 and lut[ivc] == 8 and lut[ao] == 2 and lut[svc] == 0 and lut[sb] == 11 and lut[fem] == 14
          and all(lut[t] == name_lut[t] for t in range(256) if t not in (pv, ivc, ao)), "the map = name map + merges")
    out["rules"] = dict(r["merged"])
    per_nt = {}
    for i, (G, Tf) in enumerate(t23_scans()):
        G = G.copy()
        G[G == W.ADRENAL_R_TMP] = W.ADRENAL
        per_nt[f"s{i + 1}"] = CA.scan_shares(G, Tf, name_lut, frac_min=0.0)
    r_nt = CA.build_rules(per_nt, name_lut, 4)
    check(close(r_nt["mean_shares"]["5"][str(svc)], (3 * 0.009 + 0.39) / 4) and r_nt["merged"].get(str(svc)) == 5,
          "without the 1 % rule the SVC (mean 0.104) would merge into the stomach: the atlas rule keeps it out")

    root = Path(SCR) / "t23_files"
    shutil.rmtree(root, ignore_errors=True)
    prep = root / "prep"
    ids = W.draw("d1")
    from totalsegmentator.map_to_binary import class_map_5_parts, map_taskid_to_partname_ct
    local = {t: {v: k for k, v in class_map_5_parts[map_taskid_to_partname_ct[t]].items()} for t in CA.TS_ORDER}
    aff = np.diag([1.5, 1.5, 1.5, 1.0])
    man = {"scans": {}, "gt_opened_training": sorted(ids)}
    scans = t23_scans()
    for c, (G, Tf) in zip(ids, scans):
        n = len(G)
        G3 = G.reshape(n, 1, 1)
        parts = {t: np.zeros((n, 1, 1), np.uint8) for t in CA.TS_ORDER}
        from totalsegmentator.map_to_binary import class_map
        for i, tv in enumerate(Tf):
            if tv == 0:
                continue
            nm = class_map["total"][int(tv)]
            t = [t for t in CA.TS_ORDER if nm in local[t]][0]
            parts[t][i, 0, 0] = local[t][nm]
        rec_ = {"sha256": {}, "capture_sha256": {}}
        rec_["sha256"]["gt_ts"] = CA.F.sha256(save_nii_tmp(G3, aff, prep / "labels/gt_ts" / f"WORD_{c}.nii.gz"))
        for t in CA.TS_ORDER:
            rec_["capture_sha256"][f"part_{t}"] = CA.F.sha256(save_nii_tmp(parts[t], aff, prep / "ts_cpu" / c / f"part_{t}.nii.gz"))
        man["scans"][c] = rec_
    (prep / "labels").mkdir(parents=True, exist_ok=True)
    (prep / "labels/manifest.json").write_text(json.dumps(man))
    cases = CA.training_cases("d1", prep, log=lambda m: None)
    for (c, G, Tf), (G0, T0) in zip(cases, scans):
        G0 = G0.copy()
        G0[G0 == W.ADRENAL_R_TMP] = W.ADRENAL
        check(np.array_equal(G.ravel(), G0) and np.array_equal(Tf.ravel(), T0), f"{c}: files -> the same G (17 -> 12) and Tf")
    rr = CA.rules_for_draw(cases, name_lut)
    check(rr["merged"] == r["merged"] and rr["train_ids"] == ids, "rules from the manifest-checked files")
    pv_part = [t for t in CA.TS_ORDER if "portal_vein_and_splenic_vein" in local[t]][0]
    p291 = np.zeros((2, 1, 1), np.uint8)
    p291[:, 0, 0] = local[291]["liver"]
    pp = {t: np.zeros((2, 1, 1), np.uint8) for t in CA.TS_ORDER}
    pp[291] = p291
    pp[pv_part] = np.array([[[local[pv_part]["portal_vein_and_splenic_vein"]]], [[0]]], np.uint8)
    mg = CA.merged_parts(pp, CA.part_to_total_luts())
    check(mg.ravel().tolist() == [pv, ts_id("liver")], f"TS's merge order: part {pv_part} overwrites part 291 {mg.ravel()}")
    gpath = prep / "labels/gt_ts" / f"WORD_{ids[0]}.nii.gz"
    raw_g = bytearray(gpath.read_bytes())
    gpath.write_bytes(bytes(raw_g) + b"\0")
    out["tampered"] = expect_refuse(lambda: CA.training_cases("d1", prep, log=lambda m: None),
                                    "sha256 differs from the labels manifest")[:80]
    gpath.write_bytes(bytes(raw_g))
    man2 = dict(man, gt_opened_training=sorted(ids)[:3])
    (prep / "labels/manifest.json").write_text(json.dumps(man2))
    out["not_read_before"] = expect_refuse(lambda: CA.training_cases("d1", prep, log=lambda m: None),
                                           "not among the prep's read training labels")[:80]

    f0 = root / "pred/F0__word_test30"
    lm_dir = root / "pred/LABELMAP_d1__word_test30"
    Z = np.array([ts_id("liver"), pv, ivc, ao, svc, sb, fem, 0], np.uint8).reshape(8, 1, 1)
    h = CA.F.sha256(save_nii_tmp(Z, np.diag([1.0, 1.0, 2.5, 1.0]), f0 / "raw/0001.nii.gz"))
    (f0 / "DONE.json").write_text(json.dumps({"per_scan_raw_sha256": {"0001": h}}))
    done = CA.build_labelmap(f0, lm_dir, ["0001"], "d1", dict(rr), log=lambda m: None)
    LM = np.asarray(nib.load(str(lm_dir / "raw/0001.nii.gz")).dataobj).ravel().tolist()
    check(LM == [1, 1, 8, 2, 0, 11, 14, 0], f"LABELMAP map {LM}")
    check(done["rules_sha256"] == CA.F.sha256(lm_dir / "rules.json") and done["lut_kind"] == "id"
          and done["per_scan_raw_sha256"]["0001"] == CA.F.sha256(lm_dir / "raw/0001.nii.gz")
          and done["inputs"]["F0_done_sha256"] == CA.F.sha256(f0 / "DONE.json")
          and done["ontology_plan_sha256"] == A.ONTOLOGY_PLAN_SHA256, "LABELMAP DONE record")
    check(done["per_scan_relabel"]["0001"]["relabelled_vox"] == {str(ao): 1, str(ivc): 1, str(pv): 1}, "relabelled voxels")
    out["labelmap_map"] = LM
    return out


def save_nii_tmp(arr, aff, path):
    import nibabel as nib
    import numpy as np
    path = Path(path)
    assert A.under_tmp(path), path
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(arr, np.uint8), aff), str(path))
    return path


# ---------------------------------------------------------------------------------------------------- T24 MERGE_N
def t24_merge_n():
    import nibabel as nib
    import numpy as np
    V2 = CA.v2_common()
    check(CA.F.sha256(A.V2_COMMON) == A.V2_COMMON_SHA256, "the frozen v2_common.py")
    isT = CA.site_lut()
    names = {"liver", "spleen", "kidney_left", "kidney_cyst_left", "kidney_right", "kidney_cyst_right", "stomach",
             "gallbladder", "esophagus", "pancreas", "duodenum", "colon", "small_bowel", "adrenal_gland_left",
             "adrenal_gland_right", "urinary_bladder"}
    check(sorted(int(t) for t in np.flatnonzero(isT)) == sorted(ts_id(n) for n in names) and not isT[ts_id("femur_left")],
          "T = the 16 TS classes of the 13 adapted organs; femurs non-target")
    liv, pan, bl, spl = ts_id("liver"), ts_id("pancreas"), ts_id("urinary_bladder"), ts_id("spleen")
    pv, pr, fem = ts_id("portal_vein_and_splenic_vein"), ts_id("prostate"), ts_id("femur_left")
    Z = np.array([liv, liv, pv, pr, 0, pan, fem, 0], np.uint8).reshape(8, 1, 1)
    Fm = np.array([liv, liv, liv, bl, pan, 0, fem, spl], np.uint8).reshape(8, 1, 1)
    mt, mn = V2.merge_maps(Z, Fm, isT)
    check(mn.ravel().tolist() == [liv, liv, pv, pr, pan, 0, fem, spl], f"MERGE_N by hand {mn.ravel()}")
    check(mt.ravel().tolist() == [liv, liv, liv, bl, pan, 0, fem, spl], f"MERGE_T by hand {mt.ravel()}")
    nt = ~isT[Z] & (Z > 0)
    check(np.array_equal(mn[nt], Z[nt]), "every non-target class of F0 is kept exactly")
    root = Path(SCR) / "t24_merge"
    shutil.rmtree(root, ignore_errors=True)
    f0, f1, out_dir = root / "pred/F0__word_test30", root / "pred/word_F1b_d1__word_test30", root / "pred/MERGE_N_d1__word_test30"
    aff = np.diag([1.0, 1.0, 10.0, 1.0])
    h0 = CA.F.sha256(save_nii_tmp(Z, aff, f0 / "raw/0001.nii.gz"))
    h1 = CA.F.sha256(save_nii_tmp(Fm, aff, f1 / "raw/0001.nii.gz"))
    (f0 / "DONE.json").write_text(json.dumps({"per_scan_raw_sha256": {"0001": h0}}))
    (f1 / "DONE.json").write_text(json.dumps({"per_scan_raw_sha256": {"0001": h1}}))
    done = CA.build_merge_n(f0, f1, out_dir, ["0001"], "d1", log=lambda m: None)
    MN = np.asarray(nib.load(str(out_dir / "raw/0001.nii.gz")).dataobj)
    check(np.array_equal(MN, mn) and done["per_scan_raw_sha256"]["0001"] == CA.F.sha256(out_dir / "raw/0001.nii.gz"),
          "build_merge_n == v2_common.merge_maps")
    stt = done["per_scan_merge"]["0001"]
    check(stt["nontarget_f0_f1b_both"] == {str(pv): [1, 0, 0], str(pr): [1, 0, 0], str(fem): [1, 1, 1]}
          and stt["changed_vs_f1b"] == {f"{liv}->{pv}": 1, f"{bl}->{pr}": 1} and stt["n_changed_vs_f1b"] == 2
          and close(stt["vox_ml"], 0.01), f"merge stats {stt}")
    check(done["inputs"] == {"F0_done_sha256": CA.F.sha256(f0 / "DONE.json"),
                             "F1b_d1_done_sha256": CA.F.sha256(f1 / "DONE.json"),
                             "F0_dir": CA.F.rel(f0), "F1b_dir": CA.F.rel(f1)}
          and done["lut_kind"] == "ts" and done["v2_common_sha256"] == A.V2_COMMON_SHA256, "MERGE_N DONE record")
    again = CA.build_merge_n(f0, f1, out_dir, ["0001"], "d1", log=lambda m: None)
    check(again == done, "a DONE arm is left alone")
    dirs = A.arm_dirs(root / "pred")
    ok = S.a12_inputs_consistent(["F0", "F1b_d1", "MERGE_N_d1"], dirs)
    check(set(ok) == {"MERGE_N_d1"}, "inputs consistent")
    expect_refuse(lambda: S.a12_inputs_consistent(["F0", "MERGE_N_d1"], dirs), "its input F1b_d1 is not in the look")
    (f1 / "DONE.json").write_text(json.dumps({"per_scan_raw_sha256": {"0001": h1}, "rewritten": True}))
    msg = expect_refuse(lambda: S.a12_inputs_consistent(["F0", "F1b_d1", "MERGE_N_d1"], dirs), "built from other input DONE")
    bad = root / "pred/F1b_bad"
    (bad / "raw").mkdir(parents=True)
    (bad / "DONE.json").write_text(json.dumps({"per_scan_raw_sha256": {"0001": "0" * 64}}))
    shutil.copyfile(f1 / "raw/0001.nii.gz", bad / "raw/0001.nii.gz")
    expect_refuse(lambda: CA.build_merge_n(f0, bad, root / "pred/MERGE_N_x", ["0001"], "d1", log=lambda m: None),
                  "sha256 differs from its DONE record")
    return {"merge_n": MN.ravel().tolist(), "stats": stt, "refusal": msg[:80]}


# ---------------------------------------------------------------------------------------------------- T25 runner, A1.2 rules
def t25_runner_a12():
    out = {}
    r, c, jobs, work = runner_fixture("a12_chain_guard", cutoff=datetime.now() + timedelta(hours=5))
    mark_done(jobs, 7)
    check(run_ok(c, jobs) == 0, "the runner finishes")
    nl = notrun_lines(c)
    check([x.split()[2] for x in nl] == A.STOCK_ARMS
          and all("the registered maximum run time of the rest of its STOCK chain, 8.5 h, would pass the cutoff" in x
                  for x in nl), f"STOCK chain guard {nl}")
    check(order(work) == [f"a12_cpu_MERGE_N_{d}" for d in A.DRAWS] + [f"a12_cpu_LABELMAP_{d}" for d in A.DRAWS],
          f"5 h before the cutoff only the CPU arms run: {order(work)}")
    check(not any("EXPERIMENT" in x for x in nl), "STOCK NOT RUN never stops the experiment")
    out["chain_guard_prep"] = nl[0].split("] ", 1)[-1][:150]

    t0 = datetime.now()
    r, c, jobs, work = runner_fixture("a12_chain_guard_train", cutoff=t0 + timedelta(hours=8.55))
    mark_done(jobs, 7)
    check(run_ok(c, jobs, now_fn=lambda: t0 + timedelta(hours=0.6 * order(work).count("a12_prep_STOCK_d1"))) == 0, "run")
    nl = notrun_lines(c)
    check(nl[0].startswith("NOTRUN word_A1 STOCK_d1") and "a12_train_STOCK_d1 not launched before the waits" in nl[0]
          and "8.0 h" in nl[0] and [x.split()[2] for x in nl] == A.STOCK_ARMS, f"train guard after the prep {nl}")
    check("a12_prep_STOCK_d1" in order(work) and "a12_train_STOCK_d1" not in order(work)
          and "a12_prep_STOCK_d2" not in order(work), f"{order(work)}")
    out["chain_guard_train"] = nl[0].split("] ", 1)[-1][:150]

    r, c, jobs, work = runner_fixture("a12_stock_fails", plan={"a12_train_STOCK_d2": {"fail_times": 9}})
    check(run_ok(c, jobs) == 0, "a STOCK failure does not stop the runner")
    nl = notrun_lines(c)
    check(nl == ["NOTRUN word_A1 STOCK_d2 a12_train_STOCK_d2 failed twice (rc 1)"], f"{nl}")
    check(order(work).count("a12_train_STOCK_d2") == 2 and "a12_pred_STOCK_d2__word_test30" not in order(work)
          and all(f"a12_cpu_{k}_{d}" in order(work) for k in ("MERGE_N", "LABELMAP") for d in A.DRAWS)
          and "a12_pred_STOCK_d3__word_test30" in order(work), f"{order(work)}")
    ow = order(work)
    check(ow[:7] == A.TS_JOBS and ow[7:13] == A.A12_CPU_JOBS
          and max(ow.index(n) for n in A.A12_CPU_JOBS) < ow.index("a12_prep_STOCK_d1"),
          f"the six CPU arm jobs run before the first STOCK job: {ow}")
    out["stock_fails"] = nl[0].split("] ", 1)[-1][:120]

    r, c, jobs, work = runner_fixture("a12_stock_cutoff", plan={"a12_train_STOCK_d1": {"sleep": 60}},
                                      cutoff=datetime.now() + timedelta(seconds=8))
    [j for j in jobs if j["name"] == "a12_train_STOCK_d1"][0]["est_max_h"] = 0.0001
    check([j["name"] for j in jobs][13:15] == ["a12_prep_STOCK_d1", "a12_train_STOCK_d1"], "A1.2 order in the fixture")
    mark_done(jobs, 14)
    t1 = time.time()
    check(run_ok(c, jobs) == 0 and time.time() - t1 < 50, "a running STOCK job is stopped at the cutoff")
    nl = notrun_lines(c)
    check(nl[0].startswith("NOTRUN word_A1 STOCK_d1 not DONE by the cutoff") and "a12_train_STOCK_d1 stopped" in nl[0]
          and not any("EXPERIMENT" in x for x in nl), f"{nl}")
    st = json.load(open(c.state_dir / "a12_train_STOCK_d1.json"))
    check(st["history"][0]["stopped_at_cutoff"] and st["failures"] == 0, f"{st}")
    out["stock_cutoff"] = nl[0].split("] ", 1)[-1][:120]

    r, c, jobs, work = runner_fixture("a12_locks", plan={n: {"check_lock": True} for n in (
        "a12_prep_STOCK_d1", "a12_train_STOCK_d1", "a12_pred_STOCK_d1__word_test30", "a12_cpu_MERGE_N_d1",
        "a12_cpu_LABELMAP_d3")})
    check(run_ok(c, jobs) == 0, "run")
    ow = order(work)
    check(ow == A.RUN_JOBS and ow[7:13] == A.A12_CPU_JOBS and ow[13:] == A.A12_STOCK_JOBS
          and max(ow.index(n) for n in A.A12_CPU_JOBS) < min(ow.index(n) for n in A.A12_STOCK_JOBS)
          and min(ow.index(n) for n in A.A12_STOCK_JOBS) > max(ow.index(n) for n in A.TS_JOBS),
          f"A1.2 order: A1's seven, then the six CPU arm jobs, then the nine STOCK jobs: {ow}")
    out["order"] = "A1's 7 jobs, then the 6 CPU arm jobs, then the 9 STOCK jobs"
    lk = {n: (work / f"{n}.locked").read_text() for n in ("a12_prep_STOCK_d1", "a12_train_STOCK_d1",
                                                           "a12_pred_STOCK_d1__word_test30", "a12_cpu_MERGE_N_d1",
                                                           "a12_cpu_LABELMAP_d3")}
    check(lk == {"a12_prep_STOCK_d1": "free", "a12_train_STOCK_d1": "held", "a12_pred_STOCK_d1__word_test30": "held",
                 "a12_cpu_MERGE_N_d1": "free", "a12_cpu_LABELMAP_d3": "free"}, f"GPU-1 lock {lk}")
    hist = json.load(open(c.state_dir / "a12_cpu_MERGE_N_d1.json"))["history"][0]
    check(hist["gpu"] is False and hist["gpu1_lock"] is None and hist["gpu1_mib_before"] is None, f"{hist}")
    out["locks"] = lk

    r, c, jobs, work = runner_fixture("a12_merge_dep", plan={"pfinC_train_F1b_d2": {"fail_times": 9}})
    check(run_ok(c, jobs) == 0, "run")
    nl = notrun_lines(c)
    check(len(nl) == 2 and nl[0].startswith("NOTRUN word_A1 F1b_d2 pfinC_train_F1b_d2 failed twice")
          and nl[1].startswith("NOTRUN word_A1 MERGE_N_d2 its input arm F1b_d2 was declared NOT RUN"), f"{nl}")
    check("a12_cpu_LABELMAP_d2" in order(work) and "a12_pred_STOCK_d2__word_test30" in order(work)
          and "a12_cpu_MERGE_N_d2" not in order(work), f"{order(work)}")
    out["merge_dependency"] = nl[1].split("] ", 1)[-1][:120]

    r, c, jobs, work = runner_fixture("a12_inputs_missing")
    (c.prep_dir / "labels/manifest.json").unlink()
    out["inputs_missing"] = expect_refuse(lambda: run_ok(c, jobs), "A1.2 inputs missing")[:90]
    check(order(work) == [] and not c.state_dir.exists(), "nothing launched")
    src = (HERE / "word_a1_stock.py").read_text()
    pre, post = src[src.index("def main("):].split("A.real_job_guard(", 1)
    real_stock = ('g["main"]()', "default_dirs()", "W.train_label")
    check(all(k in post and k not in pre for k in real_stock), "word_a1_stock.py: the registration gate precedes every real "
                                                               "STOCK action (training, prediction, label read)")
    src = (HERE / "word_a1_cpu_arms.py").read_text()
    pre, post = src[src.index("def main("):].split("A.real_job_guard(", 1)
    real_cpu = ("build_merge_n(", "training_cases(", "build_labelmap(")
    check(all(k in post and k not in pre for k in real_cpu), "word_a1_cpu_arms.py: the registration gate precedes every "
                                                             "real CPU-arm action")
    out["wrappers_gate_first"] = "real_job_guard precedes every real STOCK / CPU-arm action (source order)"
    return out


# ---------------------------------------------------------------------------------------------------- T26 v5 gate
def t26_v5_gate():
    out = {}
    v4_sha = "92a9b5a397b5ad0f9b915506dfc95f2049ca45b39d8321e106e0b80f57811fd7"
    check(A.SUPERSEDED_DEPOSITS == {"v1": ("osf_deposit_2026-11-05_word.zip", A.DEPOSIT_V1_SHA256),
                                    "v2": ("osf_deposit_2026-11-05_word_v2.zip", A.DEPOSIT_V2_SHA256),
                                    "v3": ("osf_deposit_2026-11-05_word_v3.zip",
                                           "65b6b3f9f1044f866ea331322d8cd871c73584d56758ac48cd976c18f0a611f3"),
                                    "v4": ("osf_deposit_2026-11-06_word_v4.zip", v4_sha)},
          f"superseded deposits {A.SUPERSEDED_DEPOSITS}")
    v4_rel = "outputs/results/vccor/osf_deposit_2026-11-06_word_v4.zip"
    logged_v4 = [p[4] for p in (ln.split() for ln in open(F.PREREG_LOG))
                 if len(p) >= 6 and p[2] == "DEPOSIT" and p[3] == "sha256" and p[5] == v4_rel]
    check(logged_v4 == [v4_sha] and F.sha256(A.DEPOSIT_V4_ZIP) == v4_sha,
          f"the v4 zip's sha256 == its single DEPOSIT line == the file on disk: {logged_v4}")
    check(A.DEPOSIT_ZIP.name == "osf_deposit_2026-11-06_word_v5.zip" and A.Ctx.real_ctx().deposit_rel == V5_REL
          and A.DEPOSIT_VERSION == "v5", "the real gate wants the v5 zip")
    for ver, (name, h) in A.SUPERSEDED_DEPOSITS.items():
        r, c = gate_fixture(f"v5_{ver}_sha", osf=False)
        setup_osf(c, name_sha=h)
        out[f"{ver}_sha"] = run_gate(r, c, f"names the {ver} zip")[:80]
        r, c = gate_fixture(f"v5_{ver}_path", osf=False)
        setup_osf(c, name_path=f"outputs/results/vccor/{name}")
        out[f"{ver}_path"] = run_gate(r, c, f"names the {ver} zip")[:80]
    for ver in ("v3", "v4"):
        h = A.SUPERSEDED_DEPOSITS[ver][1]
        r, c = gate_fixture(f"v5_deposit_line_carries_{ver}", osf=False)
        plog_line(c, f"DEPOSIT sha256 {h} {V5_REL} (synthetic: the v5 path with the {ver} sha256)")
        setup_osf(c, deposit=False, gate_text=f"{URL}\nOSF registration (PUBLIC) of {V5_REL} sha256 {FAKE_V5}\n")
        out[f"deposit_line_carries_{ver}"] = run_gate(r, c, "carries a superseded zip sha256")[:80]
    r, c = gate_fixture("v5_only_v4_deposit", osf=False)
    plog_line(c, f"DEPOSIT sha256 {v4_sha} {v4_rel} (v4)")
    setup_osf(c, deposit=False)
    out["only_v4_deposit"] = run_gate(r, c, f"no DEPOSIT line for {V5_REL}")[:90]
    r, c = gate_fixture("v5_ok")
    with patched_word(r):
        rec_, _ = S.gates(c, check=True, log=lambda m: None, stat_zip=False)
    check(rec_["2_osf_gate"]["ok"] and rec_["2_osf_gate"]["detail"]["deposit_zip"] == V5_REL
          and rec_["2_osf_gate"]["detail"]["deposit_zip_sha256"] == FAKE_V5, f"v5 accepted {rec_['2_osf_gate']}")
    for ver in ("v3", "v4"):
        r, c, jobs, work = runner_fixture(f"v5_runner_names_{ver}", osf=False)
        setup_osf(c, name_sha=A.SUPERSEDED_DEPOSITS[ver][1])
        out[f"runner_{ver}"] = expect_refuse(lambda: run_ok(c, jobs), f"names the {ver} zip")[:80]
        check(order(work) == [] and not c.state_dir.exists(), f"nothing launched with a {ver} gate file")
        rec2 = RUN.static_gates(c, pids_fn=lambda: [], check=True, log=lambda m: None)
        check(rec2["2_osf_gate"]["ok"] is False and "only the v5 zip may be registered" in rec2["2_osf_gate"]["detail"],
              f"{rec2['2_osf_gate']}")
    out["v5_ok"] = "accepted (scorer gates, synthetic v5 DEPOSIT line)"
    return out


# ---------------------------------------------------------------------------------------------------- T27 STOCK wrapper
INSTALL_PROBE = r"""
import json, sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, sys.argv[1])
import word_a1_stock as ST
import word_a1_common as A
import fin_registry as RG
W = A.W
scr = Path(sys.argv[2])
before = ST._snapshot(W, RG)
nn0 = str(W.NN_ROOT)
orig_verify = W.verify_datasets
rec = ST.install(nn_root=scr / "nnunet_stock", prep_dir=scr / "prep_stock")
again = ST.install()
out = {"f1b_unchanged": ST._snapshot(W, RG) == before, "idempotent": again == rec, "nn_root": str(W.NN_ROOT),
       "nn_root_before": nn0, "verify_wrapped": W.verify_datasets is not orig_verify}
out["runs"] = {k: {kk: vv for kk, vv in W.RUNS[k].items()} for k in ("STOCK_d1", "STOCK_d2", "STOCK_d3")}
out["datasets"] = {str(k): W.DATASETS[k]["name"] for k in (824, 825, 826)}
out["models"] = {k: str(RG.MODELS[k]["dir"]) for k in ("word_STOCK_d1", "word_STOCK_d2", "word_STOCK_d3")}
out["labelled_d2"] = W.labelled_of("STOCK_d2")
out["draw_d2"] = W.draw("d2")
p = scr / "prep_stock"
p.mkdir(parents=True, exist_ok=True)
f = scr / "x.bin"
f.write_bytes(b"x")
(p / "DONE_stock_d1.json").write_text(json.dumps({"files_sha256": {str(f): A.F.sha256(f)},
                                                   "datasets": ["Dataset824_FINWORDSTOCK_d1"], "ids": W.draw("d1")}))
out["verify_ok"] = W.verify_datasets("STOCK_d1", "STOCK") is True
res = {}
for nm, args in (("recipe", ("STOCK_d1", "F1b")), ("missing", ("STOCK_d2", "STOCK"))):
    try:
        W.verify_datasets(*args)
        res[nm] = "accepted"
    except SystemExit as e:
        res[nm] = str(e)[:600]
f.write_bytes(b"y")
try:
    W.verify_datasets("STOCK_d1", "STOCK")
    res["tampered"] = "accepted"
except SystemExit as e:
    res["tampered"] = str(e)[:600]
out["verify_refusals"] = res
print(json.dumps(out, default=str))
"""


def t27_stock_wrapper():
    ctx = A.Ctx.real_ctx()
    t11_preconditions(ctx.osf_file, ctx.look_dir)
    before = real_snapshot()
    out = {}
    scr = Path(SCR) / "t27_install"
    shutil.rmtree(scr, ignore_errors=True)
    scr.mkdir(parents=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    r = subprocess.run([str(F.VENV), "-B", "-c", INSTALL_PROBE, str(HERE), str(scr)], capture_output=True, text=True,
                       cwd=SCR, timeout=600, env=env)
    check(r.returncode == 0, f"install probe failed: {r.stderr[-1500:]}")
    p = json.loads(r.stdout.strip().splitlines()[-1])
    check(p["f1b_unchanged"] and p["idempotent"] and p["verify_wrapped"] and p["nn_root"] == str(scr / "nnunet_stock"),
          f"install {p}")
    for d, did in (("d1", 824), ("d2", 825), ("d3", 826)):
        check(p["runs"][f"STOCK_{d}"] == {"model": f"word_STOCK_{d}", "recipe": "STOCK", "ts_datasets": {},
                                          "stock_dataset": did, "ds": "word", "n_classes": 15, "labelled": d}, f"run {d}")
        check(p["datasets"][str(did)] == f"Dataset{did}_FINWORDSTOCK_{d}", "dataset name")
        check(p["models"][f"word_STOCK_{d}"] == str(A.W.DIR / f"models/STOCK_{d}"), "model dir")
    check(p["labelled_d2"] == p["draw_d2"] == A.W.draw("d2"), "labelled = the registered draw")
    v = p["verify_refusals"]
    check(p["verify_ok"] and "uses the STOCK recipe, not F1b" in v["recipe"] and "DONE_stock_d2.json missing" in v["missing"]
          and "files changed since the prep" in v["tampered"], f"verify_datasets {v}")
    out["install"] = {"runs": sorted(p["runs"]), "verify": {k: x[:60] for k, x in v.items()}}
    for stage in ("train", "predict"):
        r = subprocess.run([str(F.VENV), "-B", str(HERE / "word_a1_stock.py"), "--stage", stage, "--draw", "d2",
                            "--load_only"], capture_output=True, text=True, cwd=SCR, timeout=900, env=env)
        check(r.returncode == 0, f"--load_only {stage}: {r.stderr[-2000:]}")
        d = json.loads(r.stdout.strip().splitlines()[-1])
        if stage == "train":
            want = {"run": "STOCK_d2", "recipe": "STOCK", "trainer": "nnUNetTrainer_250epochs_NoMirroring", "epochs": 250,
                    "lr": 0.01, "fold": "all", "seed": 1337, "plans": "TS291StockPlans",
                    "nn_root": str(A.W.DIR / "nnunet_stock"), "export": str(A.W.DIR / "models/STOCK_d2"),
                    "stock_dataset": "Dataset825_FINWORDSTOCK_d2", "n_classes": 15, "labelled": A.W.draw("d2"),
                    "train_final_sha256": "ee533e5a97068a4b845e00ec8bb4ef1923988f3ad6d23f4c6f99f96b822d4f67"}
        else:
            want = {"job": "word_STOCK_d2__word_test30", "model": "word_STOCK_d2", "target": "word_test30",
                    "pred_dir": str(A.W.DIR / "pred/word_STOCK_d2__word_test30"), "n": 30,
                    "model_dir": str(A.W.DIR / "models/STOCK_d2"),
                    "predict_final_sha256": "722d5d8f37c475a1d044449932ddec42d2d5426118d06bd044453df5561a896e"}
        check(d == want, f"--load_only {stage}: {d} != {want}")
        out[f"load_only_{stage}"] = d
    after = real_snapshot()
    check(before == after, f"real files changed: {before} -> {after}")
    out["real_files_unchanged"] = True
    return out


# ---------------------------------------------------------------------------------------------------- T28 STOCK prep
def t28_stock_prep():
    import nibabel as nib
    import numpy as np
    root = Path(SCR) / "t28_stock"
    shutil.rmtree(root, ignore_errors=True)
    imgs, labs, scr = root / "imgs", root / "labs", root / "smoke"
    imgs.mkdir(parents=True)
    labs.mkdir(parents=True)
    ids = W.draw("d1")
    rng = np.random.default_rng(28)
    aff = np.diag([2.0, 2.0, 3.0, 1.0])
    raw_labels = {}
    for i, c in enumerate(ids):
        a = rng.normal(0, 50, (48, 48, 32)).astype(np.int16)
        lab = np.zeros((48, 48, 32), np.uint8)
        for k in range(1, 17):
            lab[(k - 1) * 3:(k - 1) * 3 + 3, 8:40, 4:28] = k
            a[lab == k] += 40 * k
        if i == 3:
            lab[lab == 6] = 0
        nib.save(nib.Nifti1Image(a, aff), str(imgs / f"word_{c}.nii.gz"))
        nib.save(nib.Nifti1Image(lab, aff), str(labs / f"word_{c}_label.nii.gz"))
        raw_labels[c] = lab
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    r = subprocess.run([str(F.VENV), "-B", str(HERE / "word_a1_stock.py"), "--stage", "prep", "--draw", "d1", "--smoke",
                        str(scr), "--smoke-images", str(imgs), "--smoke-labels", str(labs), "--threads", "2"],
                       capture_output=True, text=True, cwd=SCR, timeout=1800, env=env)
    check(r.returncode == 0, f"STOCK prep smoke failed: {r.stderr[-3000:]}")
    res = json.loads(r.stdout.strip().splitlines()[-1])
    dsn = "Dataset824_FINWORDSTOCK_d1"
    check(res["dataset"] == dsn and res["ids"] == ids, f"{res}")
    rd = scr / "nnunet_stock/nnUNet_raw" / dsn
    dj = json.load(open(rd / "dataset.json"))
    check(dj["labels"] == {"background": 0} | {W.ORGANS[o - 1]: o for o in range(1, 16)} and dj["numTraining"] == 4
          and dj["overwrite_image_reader_writer"] == "NibabelIOWithReorient" and dj["file_ending"] == ".nii.gz",
          f"dataset.json {dj}")
    alut = W.analysis_lut()
    for c in ids:
        fi = rd / "imagesTr" / f"WORD_{c}_0000.nii.gz"
        check(fi.is_symlink() and Path(os.readlink(fi)) == imgs / f"word_{c}.nii.gz", f"{c}: image symlink")
        lab = np.asarray(nib.load(str(rd / "labelsTr" / f"WORD_{c}.nii.gz")).dataobj)
        check(np.array_equal(lab, alut[raw_labels[c]]), f"{c}: labels in the analysis ids")
    check(int(alut[13]) == 10 and int(alut[10]) == 10 and int(alut[14]) == 13 and int(alut[15]) == 14 and int(alut[16]) == 15,
          "WORD rectum -> colon_rectum, bladder 14 -> 13, femoral heads 15 / 16 -> 14 / 15")
    pre = scr / "nnunet_stock/nnUNet_preprocessed" / dsn
    plans = json.load(open(pre / "TS291StockPlans.json"))
    import tsft_full_common as Q
    tsp = json.load(open(Q.TS_MODEL[291] / "plans.json"))
    for k in ("patch_size", "spacing", "batch_size", "UNet_base_num_features"):
        check(plans["configurations"]["3d_fullres"].get(k) == tsp["configurations"]["3d_fullres"].get(k), f"plans {k}")
    done = json.load(open(scr / "prep_stock/DONE_stock_d1.json"))
    check(done["ids"] == ids and done["datasets"] == [dsn] and done["plans"] == "TS291StockPlans"
          and done["registered"]["trainer"] == "nnUNetTrainer_250epochs_NoMirroring" and len(done["files_sha256"]) >= 6
          and all(Path(f).is_file() and F.sha256(f) == h for f, h in done["files_sha256"].items()), "DONE record")
    n_pkl = len(list((pre / plans["configurations"]["3d_fullres"]["data_identifier"]).glob("*.pkl")))
    check(n_pkl == 4, f"4 preprocessed cases ({n_pkl})")
    check(all(str(Path(f).resolve()).startswith("/tmp/") for f in done["files_sha256"]), "everything under /tmp")
    return {"dataset": dsn, "n_files": len(done["files_sha256"]), "spacing": plans["configurations"]["3d_fullres"]["spacing"],
            "labels_present": {c: done["labels"][c]["analysis_ids_present"] for c in ids}}


# ---------------------------------------------------------------------------------------------------- T29 all 13 arms
def build_a12_arms(root):
    """STOCK d1-d3 (analysis ids: d1 perfect, d2 perfect without the liver, d3 == F0's partial map), MERGE_N d1-d3 (built by
    word_a1_cpu_arms from the synthetic F0 / F1b maps) and LABELMAP d1-d3 (the name map: no merge rule)."""
    import nibabel as nib
    import numpy as np
    root = Path(root)
    dirs = A.arm_dirs(root / "C_word/pred")
    aff = np.array(AFF)
    perfect = np.zeros(SHAPE, np.uint8)
    partial = np.zeros(SHAPE, np.uint8)
    for a in range(1, K + 1):
        perfect[3 * (a - 1):3 * a, :, :] = a
        partial[3 * (a - 1):3 * a, :9, :] = a
    noliver = perfect.copy()
    noliver[0:3] = 0
    for d, P in (("d1", perfect), ("d2", noliver), ("d3", partial)):
        dd = dirs[f"STOCK_{d}"]
        (dd / "raw").mkdir(parents=True, exist_ok=True)
        shas = {}
        for c in TEST_IDS:
            p = dd / "raw" / f"{c}.nii.gz"
            nib.save(nib.Nifti1Image(P, aff), str(p))
            shas[c] = F.sha256(p)
        json.dump({"per_scan_raw_sha256": shas, "kind": "stock"}, open(dd / "DONE.json", "w"))
    ids = list(TEST_IDS)
    for d in A.DRAWS:
        CA.build_merge_n(dirs["F0"], dirs[f"F1b_{d}"], dirs[f"MERGE_N_{d}"], ids, d, log=lambda m: None)
        rules = {"merged": {}, "lut": [int(x) for x in W.ts_lut()], "train_ids": [], "procedure": "unit test: name map"}
        CA.build_labelmap(dirs["F0"], dirs[f"LABELMAP_{d}"], ids, d, rules, log=lambda m: None)
    return dirs


A1_KEYS = ("P1_gain", "P1_organ_costs", "P2_draw_dependence", "outcome", "claims_allowed", "secondary_S_TS",
           "secondary_colon_rectum", "secondary_per_draw_per_organ_descriptive", "descriptive_contrasts")


def a1_part(res):
    return json.dumps(F.clean_nan({k: res[k] for k in A1_KEYS}), sort_keys=True, default=F._default)


REHEARSE_PROBE_ALL = REHEARSE_PROBE.replace(
    'print(json.dumps({"equal": rep["continuation_reanalysis_equal"], "n": rep["n_scans"], "workers": rep["workers"]}))',
    'print(json.dumps({"equal": rep["continuation_reanalysis_equal"], "n": rep["n_scans"], "workers": rep["workers"], '
    '"arms": rep["arms"], "a12": json.load(open(scratch / "C_word/look/results.json"))["A1_2"]["readings"]}))')


def t29_all_arms_look():
    out = {}
    r0, c0 = word_fixture("t29_ts_only")
    with patched_word(r0):
        ts_res = S.look(c0, workers=1, log=lambda m: None)
    root, ctx = word_fixture("t29_all", a12="build")
    with patched_word(root):
        res = S.look(ctx, workers=2, log=lambda m: None)
    check(res["arms_run"] == sorted(A.ALL_ARMS) and res["n_scans"] == 5, f"13 arms scored in the single look {res['arms_run']}")
    check(a1_part(res) == a1_part(ts_res), "P1 / P2 / the outcome label identical with and without the A1.2 arms")
    mp = json.loads((ctx.look_dir / "maps.json").read_text())
    check(mp["lut"] == {a_: A.ARM_LUT[a_] for a_ in A.ALL_ARMS} and set(mp["done_sha256"]) == set(A.ALL_ARMS)
          and set(mp["rules_sha256"]) == set(A.LABELMAP_ARMS), "maps.json: label maps, DONE and rules hashes")
    P = PARTIAL
    a = res["A1_2"]
    check("error" not in a and S.secondary_error(res) is None, "the A1.2 secondary analysis ran without an error")
    s1 = a["S1_F1b_minus_STOCK"]
    for d, v in {"d1": 0.0, "d2": (11 * P - 1000) / 12, "d3": 0.0}.items():
        check(close(s1["per_draw"][d]["mean"], v, 1e-9), f"S1 {d}: {s1['per_draw'][d]['mean']} != {v}")
    check(close(s1["pooled"]["M"], (11 * P - 1000) / 36, 1e-9) and s1["reading"] == "NO_DIFFERENCE_DETECTED", f"S1 {s1['pooled']}")
    s1b = a["S1b_STOCK_minus_F0"]
    check(close(s1b["per_draw"]["d1"]["M"], 100 - P, 1e-9) and close(s1b["per_draw"]["d2"]["M"], 1100 / 12 - P, 1e-9)
          and close(s1b["per_draw"]["d3"]["M"], 0.0, 1e-9) and s1b["reading"] == "STOCK_NO_GAIN", f"S1b {s1b}")
    s2 = a["S2_MERGE_N_minus_F1b"]
    check(close(s2["pooled"]["M"], 0.0, 1e-9) and s2["reading_ni"] == "MERGE_N_NON_INFERIOR"
          and s2["reading_harm"] == "NO_COST_DETECTED", f"S2 {s2['pooled']}")
    fh = s2["femoral_heads_merge_n_minus_f1b"]["femur_head_left"]
    check(close(fh["M"], (P - 100) / 3, 1e-9), f"MERGE_N restores F0's femur: {fh}")
    mc = s2["merged_classes"]
    check(mc["d1"]["flagged"] == ["femur_left", "femur_right"] and mc["d2"]["flagged"] == [] and mc["d3"]["flagged"] == []
          and close(mc["d1"]["classes"]["femur_left"]["median_dice_f1b_vs_f0"], 200 * 270 / 570, 1e-9)
          and close(mc["d1"]["classes"]["femur_left"]["median_ln_volume_ratio"], math.log(300 / 270), 1e-12)
          and mc["d1"]["voxels_merge_n_differs_from_f1b"]["median"] == 60, f"merged classes {mc['d1']}")
    s3 = a["S3_LABELMAP_minus_F0"]
    check(close(s3["pooled"]["M"], 0.0, 1e-12) and s3["reading"] == "NO_DIFFERENCE_DETECTED" and not s3["S_LM"]["present"]
          and s3["rules"] == {"d1": {}, "d2": {}, "d3": {}}, f"S3 {s3['pooled']} {s3['S_LM']}")
    s4 = a["S4_F1b_minus_LABELMAP"]
    for d, v in {"d1": 100 - P, "d2": (100 - P) / 12, "d3": 0.0}.items():
        check(close(s4["per_draw"][d]["mean"], v, 1e-9), f"S4 {d}")
    check(s4["reading"] == "NOT_DETECTED" and s4["Q_share_of_F1b_gain_recovered_by_relabelling"]["Q"] is None,
          "S4 and Q (P1 fails here, as in T5)")
    tags = log_tags(ctx)
    check(tags.count("GT_OPENED") == 1 and tags.index("GT_OPENED") < tags.index("RESULTS"), f"log {tags}")
    rl = [ln for ln in ctx.log_text_lines() if " RESULTS " in ln][0]
    check("A1.2 secondary: F1b-STOCK NO_DIFFERENCE_DETECTED" in rl and "LABELMAP-F0 NO_DIFFERENCE_DETECTED" in rl, rl)
    check("SECONDARY_ERROR" not in tags, f"no SECONDARY_ERROR line without an error: {tags}")
    out["a12_readings"] = a["readings"]

    rootc, ctxc = word_fixture("t29_crash", with_code=True, a12="build")
    with patched_word(rootc):
        ref, _ = crash_after(W.reference, 4)
        W.reference = ref
        try:
            S.look(ctxc, workers=1, log=lambda m: None)
            raise AssertionError("the injected crash did not happen")
        except RuntimeError as e:
            check("injected crash" in str(e), str(e))
    with patched_word(rootc):
        resc = S.continue_look(ctxc, workers=1, log=lambda m: None)
    check(analysis_part(resc) == analysis_part(res), "continued all-arms results == the uninterrupted look")
    lkc = resc["look"]["continuation"]
    check(lkc["reused"] == ["0014", "0016"] and lkc["rescored"] == ["0017", "0019", "0021"], f"{lkc}")
    out["continued"] = {"reused": lkc["reused"], "rescored": lkc["rescored"]}
    rootd, ctxd = word_fixture("t29_done_changed", with_code=True, a12="build")
    with patched_word(rootd):
        ref, _ = crash_after(W.reference, 4)
        W.reference = ref
        try:
            S.look(ctxd, workers=1, log=lambda m: None)
        except RuntimeError:
            pass
    dj = A.arm_dirs(ctxd.pred_dir)["LABELMAP_d2"] / "rules.json"
    dj.write_text(dj.read_text() + " ")
    with patched_word(rootd):
        out["continue_rules_changed"] = expect_refuse(lambda: S.continue_gates(ctxd), "LABELMAP rules changed")[:90]

    rootr = Path(SCR) / "t29_rehearsal_word"
    shutil.rmtree(rootr, ignore_errors=True)
    build_word(rootr)
    build_a12_arms(rootr)
    scr = Path(SCR) / "t29_rehearsal_scratch"
    shutil.rmtree(scr, ignore_errors=True)
    rr = subprocess.run([str(F.VENV), "-B", "-c", REHEARSE_PROBE_ALL, str(HERE), str(rootr), str(scr)],
                        capture_output=True, text=True, cwd=SCR, timeout=900)
    check(rr.returncode == 0, f"all-arms rehearsal failed: {rr.stderr[-1500:]}")
    g = json.loads(rr.stdout.strip().splitlines()[-1])
    check(g["equal"] and g["n"] == 5 and g["arms"] == A.ALL_ARMS and g["a12"]["S1"] != "NOT RUN", f"rehearsal {g}")
    check(not (rootr / "C_word/look").exists(), "the protected look was not created")
    out["rehearsal_all_arms"] = {k: g[k] for k in ("equal", "n", "workers")}
    return out


# ---------------------------------------------------------------------------------------------------- T30 secondary error
@contextlib.contextmanager
def patched_a12(fn):
    keep = S.analyse_a1_2
    try:
        S.analyse_a1_2 = fn
        yield
    finally:
        S.analyse_a1_2 = keep


def t30_secondary_error():
    out = {}
    injected = "RuntimeError: injected secondary failure"

    def boom(*a, **k):
        raise RuntimeError("injected secondary failure")

    def unserialisable(*a, **k):
        return {"readings": {"S1": "F1B_BETTER"}, "not_json": object()}

    def stop(*a, **k):
        raise S.Terminated("signal 15 (SIGTERM)")

    R, ids = a12_records()
    ex = merge_extras(ids)
    normal = S.analyse_all(R, ids, set(A.ALL_ARMS), None, {}, ex)
    with patched_a12(boom):
        pure = S.analyse_all(R, ids, set(A.ALL_ARMS), None, {}, ex)
    check(pure["A1_2"] == {"error": injected} and a1_part(pure) == a1_part(normal) and S.secondary_error(pure) == injected
          and S.secondary_error(normal) is None, f"pure analyse_all: {pure['A1_2']}")
    with patched_a12(stop):
        try:
            S.analyse_all(R, ids, set(A.ALL_ARMS), None, {}, ex)
            raise AssertionError("a signal (Terminated) inside the secondary analysis was swallowed")
        except S.Terminated:
            pass
    out["pure"] = "error recorded, A1 part unchanged; a Terminated still propagates"

    r0, c0 = word_fixture("t30_reference", a12="build")
    with patched_word(r0):
        ref = S.look(c0, workers=1, log=lambda m: None)
    check("error" not in ref["A1_2"] and "SECONDARY_ERROR" not in log_tags(c0), "reference look: no secondary error")
    for name, fn, want in (("raises", boom, injected), ("unserialisable", unserialisable, "TypeError: ")):
        root, ctx = word_fixture(f"t30_{name}", a12="build")
        with patched_word(root), patched_a12(fn):
            res = S.look(ctx, workers=1, log=lambda m: None)
        check(set(res["A1_2"]) == {"error"} and res["A1_2"]["error"].startswith(want), f"{name}: A1_2 block {res['A1_2']}")
        check(a1_part(res) == a1_part(ref) and res["outcome"] == ref["outcome"],
              f"{name}: P1, P2 and the outcome label identical to the look without the error")
        disk = json.loads((ctx.look_dir / "results.json").read_text())
        check(disk["P1_gain"]["M"] == ref["P1_gain"]["M"]
              and disk["P2_draw_dependence"]["k"] == ref["P2_draw_dependence"]["k"]
              and disk["outcome"] == ref["outcome"] and disk["A1_2"] == res["A1_2"],
              f"{name}: results.json holds P1, P2, the outcome label and the error")
        tags = log_tags(ctx)
        check(tags.count("GT_OPENED") == 1 and tags.count("SECONDARY_ERROR") == 1 and tags.count("RESULTS") == 1
              and tags.index("GT_OPENED") < tags.index("SECONDARY_ERROR") < tags.index("RESULTS")
              and "GT_CRASHED" not in tags, f"{name}: log order {tags}")
        el = [ln for ln in ctx.log_text_lines() if " SECONDARY_ERROR " in ln][0]
        rl = [ln for ln in ctx.log_text_lines() if ln.split()[2:3] == ["RESULTS"]][0]
        check(f" SECONDARY_ERROR word_A1 {want}" in el and "never blocks the primary results" in el
              and " RESULTS " not in el and el.split()[2] == "SECONDARY_ERROR", el)
        check(f"(word A1: {ref['outcome']['combination']}; organs {ref['outcome']['organ_costs']}" in rl
              and "A1.2 secondary: ERROR (" in rl and rl.split()[4] == F.sha256(ctx.look_dir / "results.json"), rl)
        with patched_word(root):
            expect_refuse(lambda: S.continue_gates(ctx), "the look is complete")
            expect_refuse(lambda: S.look(ctx, workers=1, log=lambda m: None), "already been made")
        out[name] = {"A1_2": res["A1_2"]["error"][:70], "log": [t for t in tags if t in ("GT_OPENED", "SECONDARY_ERROR",
                                                                                         "RESULTS")]}

    root = Path(SCR) / "t30_rehearsal_word"
    shutil.rmtree(root, ignore_errors=True)
    build_word(root)
    build_a12_arms(root)
    scratch = Path(SCR) / "t30_rehearsal_scratch"
    shutil.rmtree(scratch, ignore_errors=True)
    with patched_word(root), patched_a12(boom):
        def never(cid, marker):
            raise AssertionError("the rehearsal called the real reference reader")

        W.reference = never
        rep = RH.rehearse(scratch, pred_dir=root / "C_word/pred", workers=1, result_timeout_s=60, log=lambda m: None)
    check(rep["a12_secondary_error"] == injected and rep["continuation_reanalysis_equal"] and rep["results_json_reparsed"],
          f"rehearsal report {rep}")
    tags = [ln.split()[2] for ln in open(scratch / "logs/final_prereg.log")]
    check(tags == ["GT_OPENED", "SECONDARY_ERROR", "RESULTS"], f"rehearsal scratch log {tags}")
    out["rehearsal"] = {"a12_secondary_error": rep["a12_secondary_error"], "equal": rep["continuation_reanalysis_equal"]}
    return out


TESTS = [("T1_analyse_scenario_A", t1_scenario_a), ("T2_analyse_scenario_B_n29_one_draw", t2_scenario_b),
         ("T3_analyse_scenario_C_two_draws", t3_scenario_c), ("T4_tables", t4_tables),
         ("T5_end_to_end_look", t5_end_to_end), ("T6_abort_after_marker", t6_abort),
         ("T7_scorer_gates", t7_scorer_gates), ("T8_duplicates_kept", t8_duplicates_kept),
         ("T9_runner", t9_runner), ("T10_queue_v8", t10_queue_v8), ("T11_real_mode_check_only", t11_real_mode),
         ("T12_no_inline_comments", t12_style), ("T13_one_draw_P1_bootstrap", t13_one_draw_bootstrap),
         ("T14_M_unequal_scan_sets", t14_unequal_scan_sets), ("T15_crash_and_continue", t15_crash_continue),
         ("T16_killed_or_stuck_worker", t16_killed_worker), ("T17_runner_cutoff_done_handstop", t17_runner_extras),
         ("T18_gpu1_flock", t18_gpu_flock), ("T19_queue_v8_match", t19_queue_v8_match),
         ("T20_T11_refusal", t20_t11_refusal), ("T21_rehearsal", t21_rehearsal),
         ("T22_A12_contrasts", t22_a12_contrasts), ("T23_labelmap_rules", t23_labelmap_rules),
         ("T24_merge_n", t24_merge_n), ("T25_runner_A12_rules", t25_runner_a12), ("T26_v5_gate", t26_v5_gate),
         ("T27_stock_wrapper", t27_stock_wrapper), ("T28_stock_prep_smoke", t28_stock_prep),
         ("T29_all_arms_look", t29_all_arms_look), ("T30_secondary_error_never_blocks", t30_secondary_error)]


def main():
    global SCR
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    SCR = str(Path(a.scratch).resolve())
    assert SCR.startswith("/tmp/") and not SCR.startswith(str(A.REPO)), "scratch must be under /tmp"
    Path(SCR).mkdir(parents=True, exist_ok=True)
    F.be_nice(10)
    res, t_all = {}, time.time()
    for name, fn in TESTS:
        if a.only and name.split("_")[0] not in a.only.split(","):
            continue
        t0 = time.time()
        try:
            res[name] = {"pass": True, "detail": fn(), "s": round(time.time() - t0, 1)}
        except Exception as e:
            res[name] = {"pass": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:],
                         "s": round(time.time() - t0, 1)}
        print(f"{name}: {'PASS' if res[name]['pass'] else 'FAIL ' + res[name]['error'][:300]}", flush=True)
    code = {str(p.relative_to(A.REPO)): F.sha256(p) for p in sorted(HERE.iterdir())
            if p.suffix in (".py", ".service")}
    out = {"suite": "word A1 / A1.1 / A1.2 unit tests (synthetic data only)", "written": F.now(), "scratch": SCR,
           "python": sys.version.split()[0], "tests": res, "n_tests": len(res),
           "n_pass": sum(v["pass"] for v in res.values()), "all_pass": all(v["pass"] for v in res.values()),
           "code_sha256": code, "wall_s": round(time.time() - t_all, 1),
           "data": "synthetic only: no real WORD archive, image or label was opened, listed or stat-ed; the real look "
                   "directory was not created, opened or listed (T11 / T20 / T27 check only that it does not exist); no "
                   "GPU, no training, no inference (T27 executes train_final.py / predict_final.py module code only; T28 "
                   "preprocesses synthetic volumes under /tmp)",
           "authorship": "written and run by an AI coding assistant (Claude) on the owner's instruction of 2026-11-05 (A1, "
                         "A1.1) and 2026-11-06 (A1.2)"}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(F.clean_nan(out), fh, indent=1, default=str)
        fh.write("\n")
    print(f"{out['n_pass']} / {out['n_tests']} pass -> {a.out}")
    return 0 if out["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
