"""Fact tables for storyline B (adopted on 2026-11-20) and three completions from stored per-scan records.

Tags: [R] registered reading, [RD] registered descriptive, [PH] post hoc (computed here, or copied from an earlier post
hoc table), [DEV] development phase. Every number is copied from, or computed from, the record named in its row; the
results.json files are sha256-checked against their RESULTS lines (disp_common.results). Facts for the author, not paper
text.

Outputs (this folder): table1_cohorts, table2_volumes, table3_update_options, table4_deviations (.md and .csv each),
table4b_deviations_full.csv, completions_posthoc.md / .json, TRACE_tables.csv, sources_tables.json.
usage: python3 -I -B build_tables.py
Version of 2026-11-20.
"""
import csv
import json
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import disp_common as C  # noqa: E402

RES = {k: C.results(k) for k in C.LOOK_DIRS}
DISP = "tables"


# ------------------------------------------------------------------ helpers
def est(d):
    for k in ("M", "estimate", "mean"):
        if k in d:
            return d[k]
    raise KeyError(list(d))


def ci(d):
    for k in ("ci", "scan_boot_ci", "boot_ci"):
        if k in d:
            return d[k]
    raise KeyError(list(d))


def pooled(d):
    return d.get("pooled", d)


def reading(*ds):
    for d in ds:
        if isinstance(d, dict):
            for k in ("reading", "decision", "outcome", "combination"):
                if isinstance(d.get(k), str):
                    return d[k]
    found = []
    for d in ds:
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, str) and ("reading" in k or "decision" in k) and v not in found:
                    found.append(v)
    assert found, [list(d) for d in ds if isinstance(d, dict)]
    return " / ".join(found)


def f2(x, d=2):
    return C.sg(x, d)


def fci(c, d=2):
    return C.ci_txt(c, d)


def cell(where, quantity, text, source):
    C.trace(DISP, where, quantity, text, source)
    return text


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        out.append("| " + " | ".join(str(c).replace("|", "\\|").replace("\n", "<br>") for c in r) + " |")
    return "\n".join(out)


def write_table(name, title, notes, header, rows):
    with open(C.OUT / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    with open(C.OUT / f"{name}.md", "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        for n in notes:
            f.write(f"- {n}\n")
        f.write("\n" + md_table(header, rows) + "\n")


def log_time(kind, path):
    r = C.rel(path)
    rx = re.compile(r"^\[([0-9-]+ [0-9:]+)\] %s sha256 [0-9a-f]{64} (\S+)" % kind)
    hits = []
    with open(C.LOG, encoding="utf-8") as f:
        for ln in f:
            m = rx.match(ln)
            if m and m.group(2) == r:
                hits.append(m.group(1))
    assert len(hits) == 1, (kind, r, hits)
    return hits[0]


def interval(a, b):
    s = int((datetime.strptime(b, "%Y-%m-%d %H:%M:%S") - datetime.strptime(a, "%Y-%m-%d %H:%M:%S")).total_seconds())
    return f"{s // 3600} h {s % 3600 // 60:02d} min {s % 60:02d} s"


def col(header, name):
    if name in header:
        return header.index(name)
    hits = [i for i, h in enumerate(header) if name in h and "status" not in h.lower() and "mean over draws" not in h.lower()]
    assert len(hits) == 1, (name, header)
    return hits[0]


# ------------------------------------------------------------------ completion 1: WORD-N solid-organ volume error level
def signed_e(rec, arm, i):
    n = rec["NATIVE"]
    if not n["ref"]["present"][i]:
        return None
    return 100.0 * (n["arms"][arm]["ml"][i] - n["ref"]["ml"][i]) / n["ref"]["ml"][i]


PV_N = C.pervol("WORD-N")
VD = RES["WORD-N"]["volumes_descriptive"]
SOLID = RES["WORD-N"]["solid_organs"]
L_N = len(next(iter(PV_N.values()))["NATIVE"]["ref"]["ml"])
IDX = {}
for name, st in VD["F0"]["organs"].items():          # map names to list positions by the stored F0 mean signed error
    cands = []
    for i in range(L_N):
        vals = [v for v in (signed_e(r, "F0", i) for r in PV_N.values()) if v is not None]
        if len(vals) == st["n"] and abs(statistics.fmean(vals) - st["mean_signed"]) < 1e-9:
            cands.append(i)
    assert len(cands) == 1, (name, cands)
    IDX[name] = cands[0]
DRAWS_N = ["d1", "d2", "d3"]


def level(arm):
    per_scan = []
    for r in PV_N.values():
        es = [abs(signed_e(r, arm, IDX[o])) for o in SOLID if signed_e(r, arm, IDX[o]) is not None]
        if es:
            per_scan.append(statistics.fmean(es))
    return statistics.fmean(per_scan), len(per_scan)


LVL = {a: level(a) for a in ["F0"] + [f"F1b_{d}" for d in DRAWS_N] + [f"STOCK_{d}" for d in DRAWS_N]}
PV_CHECK = statistics.fmean(LVL[f"F1b_{d}"][0] - LVL["F0"][0] for d in DRAWS_N)
assert abs(PV_CHECK - RES["WORD-N"]["P_V"]["estimate"]) < 1e-9, (PV_CHECK, RES["WORD-N"]["P_V"]["estimate"])
COMP1 = {"definition": "mean over scans of the mean over the solid organs present in the reference (>= 1 mL) of "
                       "|e|, e = 100 (V_pred - V_ref) / V_ref, V = all voxels of the class (mL); the estimand of the "
                       "registered primary P-V is the mean over draws of (F1b level - F0 level)",
         "solid_organs": SOLID, "pervol_positions": {o: IDX[o] for o in SOLID},
         "levels_pp": {a: {"mean_abs_e": v[0], "n_scans": v[1]} for a, v in LVL.items()},
         "check": f"mean over draws of (F1b - F0 level) = {PV_CHECK:.12f} = registered P_V.estimate (to 1e-9)"}

# ------------------------------------------------------------------ completion 2: LCTSC precision / recall / volume ratio
STRUCT = ["esophagus", "heart", "lung_left", "lung_right", "spinal_cord"]   # plan label_rules ids 1-5 = list index 0-4
ARMS_L = ["F0", "F1b_S1", "F1b_S2", "F1b_S3"]


def lctsc_stats(look):
    pv = C.pervol(look)
    out = {}
    for i, s in enumerate(STRUCT):
        row = {}
        for a in ARMS_L:
            P, Rc, VR, DC = [], [], [], []
            for rec in pv.values():
                n = rec["NATIVE"]
                if not n["ref"]["present"][i]:
                    continue
                x = n["arms"][a]
                vp, vr, it = x["vox"][i], n["ref"]["vox"][i], x["inter"][i]
                P.append(100.0 * it / vp if vp else 0.0)
                Rc.append(100.0 * it / vr)
                VR.append(x["ml"][i] / n["ref"]["ml"][i])
                DC.append(x["dice"][i])
            row[a] = {"precision": statistics.fmean(P), "recall": statistics.fmean(Rc),
                      "median_volume_ratio": statistics.median(VR), "mean_dice": statistics.fmean(DC), "n": len(P)}
        out[s] = row
    oc = RES[look]["organ_costs"]["per_organ"]
    for s in STRUCT[:4]:
        m = statistics.fmean(out[s][a]["mean_dice"] - out[s]["F0"]["mean_dice"] for a in ARMS_L[1:])
        assert abs(m - oc[s]["M"]) < 1e-9, (look, s, m, oc[s]["M"])
        for a in ARMS_L:
            assert abs(out[s][a]["median_volume_ratio"] - oc[s]["median_volume_ratio"][a]) < 1e-9, (look, s, a)
    for a in ARMS_L[1:]:
        assert out["spinal_cord"][a] == out["spinal_cord"]["F0"], (look, a, "spinal cord must equal F0")
    return out


COMP2 = {"LCTSC": lctsc_stats("LCTSC"), "LCTSC-T23": lctsc_stats("LCTSC-T23"),
         "definition": "per patient: precision = 100 inter / predicted voxels, recall = 100 inter / reference voxels, "
                       "volume ratio = predicted mL / reference mL; means (median for the ratio) over patients; "
                       "checks: mean Dice change equals the registered organ costs (to 1e-9), median volume ratios "
                       "equal the registered descriptive values, spinal cord identical in every arm"}

# ------------------------------------------------------------------ completion 3: LCTSC-T23 per-structure rows (copied)
T23 = RES["LCTSC-T23"]
COMP3 = {"organ_costs": T23["organ_costs"], "per_draw": {
    d: {"fp_events": T23["descriptive_per_draw_contrasts"][d]["fp_events"],
        "failures": T23["descriptive_per_draw_contrasts"][d]["failures"]} for d in ("S1", "S2", "S3")}}

json.dump({"completion_1_wordn_volume_level": COMP1, "completion_2_lctsc_precision_recall": COMP2,
           "completion_3_t23_rows_copied_RD": COMP3}, open(C.OUT / "completions_posthoc.json", "w"), indent=1)

# ------------------------------------------------------------------ Table 1: cohorts and tiers
LOOK_TXT = {
    "WORD": dict(
        reg="osf.io/c5fyj",
        data="WORD (Luo et al. 2022): abdominal CT of patients before radiotherapy, one centre, one SIEMENS scanner, "
             "'without appearance enhancement'; official split 100 / 20 / 30",
        ref="one senior oncologist, checked by an expert; annotations follow the RTOG radiotherapy delineation "
            "guideline (WORD paper, section 3.2)",
        test="29 test scans (30 minus one duplicate)",
        train="3 draws x 4 scans of WORD's training split",
        models="3 updated models (one per draw), trained after the registration was public; default path: 3 models",
        tier="internal: a new dataset, one centre",
        later="2 test labels opened for a figure on 2026-11-16 (C3-D15)"),
    "FLARE-T50": dict(
        reg="osf.io/ajfrc",
        data="50 never-opened scans of the FLARE22 cohort (Hugging Face FLARE-MedFM/FLARE-Task2-LaptopSeg); "
             "centres not stated in the records",
        ref="FLARE challenge labels; protocol not read (register 6.9)",
        test="50 scans",
        train="development training scans: 3 draws x 4 (also K = 1: 4 models; K = 16: 3 models)",
        models="development-phase models, trained before registration; default path: 1 model",
        tier="internal: fresh scans of a development dataset ('internal replication ..., not external validation', "
             "plan data.role)",
        later="post hoc second read of all 50 labels (C3-D24); 3 labels for a figure (C3-D27)"),
    "LCTSC": dict(
        reg="osf.io/d6ayj",
        data="LCTSC (TCIA; 2017 AAPM challenge): thoracic radiotherapy planning CT, three institutions",
        ref="clinical contours checked against the RTOG 1106 atlas by one organiser (Yang et al. 2018); RTSTRUCT "
            "polygons rasterised by the study's code (plan note N3)",
        test="24 test patients",
        train="one institution per draw: 4 training patients each (S1, S2, S3)",
        models="3 updated models, trained before registration (disclosed); default path not run",
        tier="internal: a new dataset (thorax, three institutions)",
        later="none recorded (C4: read 1 of 1)"),
    "WORD-N": dict(
        reg="osf.io/7jser",
        data="the 20 scans of WORD's validation split: same centre, scanner and protocol as WORD",
        ref="as WORD",
        test="20 scans",
        train="none: the WORD models",
        models="the 3 WORD models; default path: 3 models; VISTA3D tier NOT RUN (out of memory)",
        tier="internal: same centre and models as WORD (an extension, not a second cohort)",
        later="none recorded (C4: read 1 of 1)"),
    "LCTSC-T23": dict(
        reg="osf.io/69kwv",
        data="23 LCTSC patients of the collection's training folder whose labels no analysis had read (36 minus 13); "
             "8 / 7 / 8 per institution",
        ref="as LCTSC",
        test="23 patients",
        train="none: the frozen LCTSC models",
        models="the 3 frozen LCTSC models",
        tier="internal: further patients of the same institutions, same models ('precision, not independence')",
        later="labels of 5 patients opened for figures on 2026-11-18 and 2026-11-19 (logged NOTE lines)"),
}
T_SUB = {"WORD": "2026-11-06 14:23:12", "FLARE-T50": "2026-11-12 10:11:07", "LCTSC": "2026-11-16 09:14:43",
         "WORD-N": "2026-11-16 09:11:05", "LCTSC-T23": "2026-11-18 09:18:48"}
EXPECT_INTERVAL = {"WORD": "22 h 20 min 55 s", "FLARE-T50": "22 h 47 min 17 s", "LCTSC": "1 h 55 min 08 s",
                   "WORD-N": "3 h 56 min 55 s", "LCTSC-T23": "1 h 38 min 44 s"}
g23 = C.read_json(C.FINAL / "K_lctsc_t23/gate_osf_public.json")
assert g23["api_attributes"]["date_registered"].startswith("2026-11-18T00:18:48")


def public_time(look):
    if look == "WORD":
        return "2026-11-06 16:55:06", "R/C_word/independent_check_2026-11-07/REPORT.md section 2; T/process/C2"
    g = C.read_json(C.FINAL / C.LOOK_DIRS[look] / "gate_osf_public.json")
    return g["public_checked_kst"], f"R/{C.LOOK_DIRS[look]}/gate_osf_public.json public_checked_kst"


def primary(look):
    r = RES[look]
    if look == "WORD":
        d = r["P1_gain"]
        return ("P1: macro Dice gain, 12 organs, with each organ's cost at -1.0",
                f"{reading(d, r)}: {f2(est(d))} {fci(ci(d))}; {d['scans_up']} of {r['P1_gain'].get('n_scans', 29)} up "
                f"(need {d['need_up']}); organ costs {r['P1_organ_costs']['reading']}", "R/C_word/look/results.json P1_gain, P1_organ_costs")
    if look == "FLARE-T50":
        d = r["R1"]
        return ("R1: macro Dice gain, 13 organs (the WORD rule on fresh scans)",
                f"{reading(d)}: {f2(est(d))} {fci(ci(d))}; {d['scans_up']} of 50 up (need {d['need_up']}); organ costs "
                f"{r['organ_costs']['reading']} [RD]", "R/I_flaret50/look/results.json R1, organ_costs")
    if look == "LCTSC":
        d = r["P1"]
        return ("P1: macro Dice gain, 4 structures (esophagus, heart, lungs)",
                f"{reading(d)}: {f2(est(d))} {fci(ci(d))}; {d['scans_up']} of 24 up (need {d['need_up']}); organ costs "
                f"{r['organ_costs']['reading']}", "R/K_lctsc/look/results.json P1, organ_costs")
    if look == "WORD-N":
        d = r["P_V"]
        return ("P-V: change in mean |volume error| of 5 solid organs (pp); the Dice gain is a Holm secondary",
                f"{d['reading']}: {f2(d['estimate'])} pp {fci(d['ci'])}; p {d['p_two_sided']:.4f}; {d['scans_below_0']} "
                f"of 20 scans below 0", "R/I_wordn/look/results.json P_V")
    d = r["P1_HF"]
    return ("P1_HF: macro Dice gain without the heart (esophagus, lungs)",
            f"{d['reading']}: {f2(d['M'])} {fci(d['ci'])}; {d['scans_up']} of 23 up (need {d['need_up']})",
            "R/K_lctsc_t23/look/results.json P1_HF")


H1 = ["Look (registration)", "Data", "Reference standard", "Test set", "Labelled scans for the update", "Models",
      "Tier (the call's terms)", "Registration submitted (KST)", "Public (KST)", "Reference labels opened (KST)",
      "Public to labels", "Later label loads", "Primary endpoint", "Primary reading [R]"]
ROWS1 = []
for look, t in LOOK_TXT.items():
    pub, pub_src = public_time(look)
    gto = log_time("GT_OPENED", C.FINAL / C.LOOK_DIRS[look] / "look/GT_OPENED")
    iv = interval(pub, gto)
    assert iv == EXPECT_INTERVAL[look], (look, iv)
    pe, pr, psrc = primary(look)
    ROWS1.append([f"{look} ({t['reg']})", t["data"], t["ref"], t["test"], t["train"], t["models"], t["tier"],
                  cell(f"T1 {look}", "submitted", T_SUB[look], "OSF date_registered + 9 h (NUMBERS.md table 0)"),
                  cell(f"T1 {look}", "public", pub, pub_src),
                  cell(f"T1 {look}", "labels opened", gto, f"GT_OPENED line of the registration log ({look})"),
                  cell(f"T1 {look}", "public to labels", iv, "computed from the two times"),
                  t["later"], pe, cell(f"T1 {look}", "primary reading", pr, psrc)])
C4 = {r["Evaluation set"][:12]: r for r in C.read_csv(C.TAB / "process/C4_label_read_ledger_summary.csv")}
ROWS1.append(["Development phase (no registration) [DEV]",
              "reused internal sets: FLARE held18 / test10, MACT-81, AMOS-240 / 56, BTCV-28", "as each dataset", "-",
              "development draws", "development models", "internal; development (exploratory)", "-", "-", "-", "-",
              "scoring reads: FLARE held18 >= 23; MACT-81 7; AMOS-240 5; BTCV-28 >= 4 loads (C4)",
              "-", "the recipe and the hypotheses of the looks come from here"])
ROWS1.append(["AMOS site design E3 / E3b (osf.io/me5w2)", "AMOS-240 / AMOS-56, two sites", "AMOS labels", "-", "-", "-",
              "not pre-registered: public only after both looks (C3-D30); E3's model and predictions predate the "
              "submission (C3-D31)", "2026-11-04 15:57:59", "2026-11-04 23:51:56", "E3 17:14:41; E3b 21:56:26", "-",
              "-", "site-specific component", "E3 held; the second draw E3b did not replicate (+0.23 [-0.18, +0.67])"])
write_table("table1_cohorts", "Table 1 (facts): cohorts, reference standards, tiers and timing of the registered looks",
            ["Every look is internal testing in the call's terms: two new datasets (WORD, LCTSC) and three repeats by "
             "design (FLARE-T50, WORD-N, LCTSC-T23).",
             "Times KST. 'Public' = first anonymous OSF API 200 (gate records; WORD: its check report). 'Labels opened' "
             "= the GT_OPENED line of the registration log. GATE log lines are first-use times, not registration "
             "times.",
             "Too wide for the 6.0 in text block at body size: split it (for example data and tier / timing and "
             "endpoint) or move columns to the appendix. No smaller font is allowed in tables.",
             "Sources per cell: TRACE_tables.csv; data descriptions from the plans, the WORD paper (arXiv 2111.02403) "
             "and REGISTER / T/process C2-C4."], H1, ROWS1)

# ------------------------------------------------------------------ Table 2: volumes
H2 = ["Quantity", "Look", "Unmodified TS (F0)", "Updated, draw 1 / 2 / 3", "Default path, draw 1 / 2 / 3",
      "Registered contrast", "Tag", "Source"]
ROWS2 = []
b4c = {r[0]: r for r in csv.reader(open(C.TAB / "flare_word/B4_flaret50_volumes_B4c.csv", encoding="utf-8"))}
C.SOURCES[C.rel(C.TAB / "flare_word/B4_flaret50_volumes_B4c.csv")] = C.sha256(C.TAB / "flare_word/B4_flaret50_volumes_B4c.csv")
s6 = RES["FLARE-T50"]["secondary_holm_family"]["contrasts"]["S6"]
ROWS2.append(["Mean |volume error|, 5 solid organs (percentage points)", "FLARE-T50",
              cell("T2", "FLARE-T50 F0 level", f"{float(b4c['F0'][1]):.2f}", "T/flare_word/B4c F0"),
              cell("T2", "FLARE-T50 F1b levels", " / ".join(f"{float(b4c[a][1]):.2f}" for a in ("FULL_s1337", "FULL_s2025", "FULL_s3407")), "T/flare_word/B4c"),
              cell("T2", "FLARE-T50 STOCK level", f"{float(b4c['STOCK'][1]):.2f} (one model)", "T/flare_word/B4c"),
              cell("T2", "FLARE-T50 S6", f"S6 {f2(s6['estimate'])} pp {fci(s6['ci'])}; Holm p {s6['p_holm']:.4f}; {s6['reading']}", "secondary_holm_family.contrasts.S6"),
              "levels [PH]; contrast [R]", "T/flare_word/B4c; R/I_flaret50/look/results.json"])
pv = RES["WORD-N"]["P_V"]
ROWS2.append(["Mean |volume error|, 5 solid organs (percentage points)", "WORD-N",
              cell("T2", "WORD-N F0 level", f"{LVL['F0'][0]:.2f}", "completion 1"),
              cell("T2", "WORD-N F1b levels", " / ".join(f"{LVL['F1b_' + d][0]:.2f}" for d in DRAWS_N), "completion 1"),
              cell("T2", "WORD-N STOCK levels", " / ".join(f"{LVL['STOCK_' + d][0]:.2f}" for d in DRAWS_N), "completion 1"),
              cell("T2", "WORD-N P-V", f"P-V (primary) {f2(pv['estimate'])} pp {fci(pv['ci'])}; p {pv['p_two_sided']:.4f}; {pv['scans_below_0']} of 20 scans below 0; {pv['reading']}", "P_V"),
              "levels [PH] (completion 1); contrast [R]", "R/I_wordn/look/pervol; results.json P_V"])
b4b_rows = list(csv.reader(open(C.TAB / "flare_word/B4_flaret50_volumes_B4b.csv", encoding="utf-8")))
C.SOURCES[C.rel(C.TAB / "flare_word/B4_flaret50_volumes_B4b.csv")] = C.sha256(C.TAB / "flare_word/B4_flaret50_volumes_B4b.csv")
hb = b4b_rows[0]
ci_f0, ci_sg = col(hb, "F0 mean |e|"), col(hb, "F0 mean signed e")
ci_d = [col(hb, f"{a} mean |e|") for a in ("FULL_s1337", "FULL_s2025", "FULL_s3407")]
ci_st = col(hb, "STOCK mean |e|")
for r in b4b_rows[1:]:
    ROWS2.append([f"{r[0]}: mean |e| (%); F0 mean signed e", "FLARE-T50",
                  cell("T2", f"FLARE-T50 {r[0]} F0", f"|e| {float(r[ci_f0]):.2f}; signed {f2(float(r[ci_sg]))}", "T/flare_word/B4b"),
                  cell("T2", f"FLARE-T50 {r[0]} F1b", " / ".join(f"{float(r[i]):.2f}" for i in ci_d), "T/flare_word/B4b"),
                  cell("T2", f"FLARE-T50 {r[0]} STOCK", f"{float(r[ci_st]):.2f}", "T/flare_word/B4b"), "-", "[PH]",
                  "T/flare_word/B4b"])
for o in SOLID:
    f0 = VD["F0"]["organs"][o]
    ROWS2.append([f"{o}: mean |e| (%); mean signed e (%)", "WORD-N",
                  cell("T2", f"WORD-N {o} F0", f"|e| {f0['mean_abs']:.2f}; signed {f2(f0['mean_signed'])}", f"volumes_descriptive.F0.organs.{o}"),
                  cell("T2", f"WORD-N {o} F1b", " / ".join(f"{VD['F1b_' + d]['organs'][o]['mean_abs']:.2f} ({f2(VD['F1b_' + d]['organs'][o]['mean_signed'])})" for d in DRAWS_N), f"volumes_descriptive.F1b_d*.organs.{o}"),
                  cell("T2", f"WORD-N {o} STOCK", " / ".join(f"{VD['STOCK_' + d]['organs'][o]['mean_abs']:.2f}" for d in DRAWS_N), f"volumes_descriptive.STOCK_d*.organs.{o}"),
                  "-", "[RD]", "R/I_wordn/look/results.json volumes_descriptive"])
ROWS2.append(["Share of the 240 primary-organ volumes within +-10 % (12 organs x 20 scans)", "WORD-N",
              cell("T2", "WORD-N within 10 F0", f"{100 * VD['F0']['share_within_10pct_primary']:.1f} %", "volumes_descriptive.F0"),
              cell("T2", "WORD-N within 10 F1b", " / ".join(f"{100 * VD['F1b_' + d]['share_within_10pct_primary']:.1f} %" for d in DRAWS_N), "volumes_descriptive"),
              cell("T2", "WORD-N within 10 STOCK", " / ".join(f"{100 * VD['STOCK_' + d]['share_within_10pct_primary']:.1f} %" for d in DRAWS_N), "volumes_descriptive"),
              "-", "[RD]", "R/I_wordn/look/results.json volumes_descriptive"])
for look in ("LCTSC", "LCTSC-T23"):
    oc = RES[look]["organ_costs"]["per_organ"]
    for s in STRUCT[:4]:
        vr = oc[s]["median_volume_ratio"]
        ROWS2.append([f"{s}: median V_pred / V_ref", look,
                      cell("T2", f"{look} {s} F0 ratio", f"{vr['F0']:.3f}", f"organ_costs.per_organ.{s}.median_volume_ratio"),
                      cell("T2", f"{look} {s} F1b ratio", " / ".join(f"{vr[a]:.3f}" for a in ARMS_L[1:]), f"organ_costs.per_organ.{s}.median_volume_ratio"),
                      "not run", "-", "[RD]", f"R/{C.LOOK_DIRS[look]}/look/results.json"])
    sc = COMP2[look]["spinal_cord"]["F0"]
    ROWS2.append(["spinal_cord (not adapted): median V_pred / V_ref", look,
                  cell("T2", f"{look} cord ratio", f"{sc['median_volume_ratio']:.3f}", "completion 2"),
                  "identical to F0 (not retrained)", "not run", "-", "[PH] (completion 2)", f"R/{C.LOOK_DIRS[look]}/look/pervol"])
hol = RES["WORD-N"]["holm_family"]["results"]
for o, key in (("duodenum", "OS-D"), ("esophagus", "OS-E")):
    tv = RES["WORD-N"]["tubular_volumes"][o]
    f0 = tv["F0_signed_e"]
    ROWS2.append([f"{o}: mean signed volume error (%); scans above +25 % (of 20)", "WORD-N",
                  cell("T2", f"WORD-N {o} F0 signed", f"{f2(f0['estimate'])} {fci(f0['ci'])}; {f0['down']} of 20 below 0; above +25 %: {tv['n_over_25pct']['F0']}", f"tubular_volumes.{o}.F0_signed_e"),
                  cell("T2", f"WORD-N {o} F1b signed", " / ".join(f2(tv['per_draw_mean_signed_e'][d]) for d in DRAWS_N) + "; above +25 %: " + " / ".join(str(tv['n_over_25pct']['F1b_' + d]) for d in DRAWS_N), f"tubular_volumes.{o}"),
                  "-",
                  cell("T2", f"WORD-N {key}", f"{key}: signed e_F1b {f2(tv['signed_e_F1b']['estimate'])} {fci(tv['signed_e_F1b']['ci'])}; shift {f2(tv['shift_e_F1b_minus_e_F0']['estimate'])} pp {fci(tv['shift_e_F1b_minus_e_F0']['ci'])}; Holm p {hol[key]['p_holm']:.4f}; {hol[key]['reading']}", f"tubular_volumes.{o}; holm_family.results.{key}"),
                  "[R] contrast; [RD] levels and counts", "R/I_wordn/look/results.json"])
write_table("table2_volumes", "Table 2 (facts): volumes a clinician reads",
            ["e = 100 (V_pred - V_ref) / V_ref per scan and organ (organs present in the reference, >= 1 mL); |e| in "
             "percentage points.",
             "Registered volume endpoints exist on FLARE-T50 (S6, Holm secondary) and WORD-N (P-V, primary) only. "
             "Allowed phrase: 'reduced solid-organ volume error'; never 'improves volumetry' in general.",
             "WORD-N over-segmentation of duodenum and esophagus was registered and NOT_CONFIRMED: report the shift, "
             "never 'over-segments'.",
             "LCTSC rows: the heart ratio is the registered descriptive value; a heart-definition mechanism is never "
             "a finding ('consistent with')."], H2, ROWS2)

# ------------------------------------------------------------------ Table 3: update options
F = RES["FLARE-T50"]
FC = F["secondary_holm_family"]["contrasts"]
A12 = RES["WORD"]["A1_2"]
J = C.read_json(C.FINAL / "J_t50_posthoc/results_posthoc.json")["contrasts"]
absm = F["absolute_macro"]


def am(k):
    v = absm[k]
    return v if isinstance(v, (int, float)) else est(v)


ssr = F["tier2"]["single_scan_risk"]["per_draw"]


def share(v):
    if isinstance(v, (int, float)):
        return v
    for k, x in v.items():
        if isinstance(x, (int, float)) and ("share" in k or "frac" in k or "below" in k):
            return x
    raise KeyError(list(v))


k1 = {k: am(k) - am("F0") for k in ("K1_0001", "K1_0006", "K1_0014", "K1_0038")}
wn = RES["WORD-N"]
s1f = pooled(wn["S1_fresh"])
DIFF_STOCK = ("re-initialised segmentation head; the dataset's own classes only (TS's other classes dropped); "
              "learning rate 1e-2; 250 x 250 iterations (trainer default 1,000 epochs); nnU-Net preprocessing of the raw "
              "image; nnU-Net predictor instead of TS's pipeline (C5b)")
H3 = ["Route", "Differs from the update in", "Look", "Contrast", "Estimate [95% CI]", "Scans; trained models",
      "Reading; tag", "What the row cannot show"]
ROWS3 = [
    ["Default nnU-Net fine-tune path (STOCK)", DIFF_STOCK, "WORD", "STOCK - F0",
     cell("T3", "WORD STOCK-F0", f"{f2(est(A12['S1b_STOCK_minus_F0']))} {fci(ci(A12['S1b_STOCK_minus_F0']))}", "A1_2.S1b_STOCK_minus_F0"),
     f"29; 3", f"{reading(A12['S1b_STOCK_minus_F0'])}; [R] (no direction registered)",
     "which factor matters; on the 15-structure macro STOCK is above F0 (82.05 vs 81.28 [RD], through the femoral heads)"],
    ["", "", "WORD", "update - STOCK",
     cell("T3", "WORD F1b-STOCK", f"{f2(est(pooled(A12['S1_F1b_minus_STOCK'])))} {fci(ci(pooled(A12['S1_F1b_minus_STOCK'])))}", "A1_2.S1_F1b_minus_STOCK.pooled"),
     "29; 3 + 3", f"{reading(A12['S1_F1b_minus_STOCK'], pooled(A12['S1_F1b_minus_STOCK']))}; [R] (no direction registered)", "recipe-level comparison only"],
    ["", "", "FLARE-T50", "STOCK - F0",
     cell("T3", "FLARE STOCK-F0", f"{f2(est(F['stock_vs_F0']))} {fci(ci(F['stock_vs_F0']))}", "stock_vs_F0"),
     "50; 1", f"{reading(F['stock_vs_F0'])}; [R] (no direction registered)",
     "one model; the loss grew with scan length (rho -0.73) [PH], part unexplained"],
    ["", "", "FLARE-T50", "update (draw s1337) - STOCK",
     cell("T3", "FLARE S3", f"{f2(FC['S3']['estimate'])} {fci(FC['S3']['ci'])}; Holm p {FC['S3']['p_holm']:.4f}", "secondary_holm_family.contrasts.S3"),
     "50; 1 + 1", f"{FC['S3']['reading']}; [R] (Holm, direction registered)", "one model each side"],
    ["", "", "WORD-N", "STOCK - F0",
     cell("T3", "WORD-N STOCK-F0", f"{f2(est(wn['A1_2_readings_fresh_descriptive']['S1b_STOCK_minus_F0']))} {fci(ci(wn['A1_2_readings_fresh_descriptive']['S1b_STOCK_minus_F0']))}", "A1_2_readings_fresh_descriptive.S1b_STOCK_minus_F0"),
     "20; 3", f"{reading(wn['A1_2_readings_fresh_descriptive']['S1b_STOCK_minus_F0'])}; [RD]", "same models as WORD"],
    ["", "", "WORD-N", "update - STOCK",
     cell("T3", "WORD-N S1-fresh", f"{f2(est(s1f))} {fci(ci(s1f))}; Holm p {wn['holm_family']['results']['S1-fresh']['p_holm']:.4f}", "S1_fresh; holm_family.results.S1-fresh"),
     "20; 3 + 3", f"{wn['holm_family']['results']['S1-fresh']['reading']}; [R] (Holm, no direction registered)", "same models as WORD"],
    ["One labelled scan (K = 1)", "one scan instead of four (same recipe)", "FLARE-T50", "K = 1 - K = 4",
     cell("T3", "K1-K4", f"{f2(est(F['tier2']['K1_minus_K4']))} {fci(ci(F['tier2']['K1_minus_K4']))}", "tier2.K1_minus_K4"),
     "50; 4 + 3", f"{F['tier2']['K1_minus_K4']['reading']}; [R]",
     cell("T3", "K1 single models", "the four single scans are the four scans of one draw; single models minus F0: " + ", ".join(f2(v) for v in k1.values()) + " [RD]; share of scans below F0: " + ", ".join(f"{share(v):.2f}" for v in ssr.values()) + " (risk REPLICATED [R])", "absolute_macro.K1_*; tier2.single_scan_risk")],
    ["Sixteen labelled scans (K = 16)", "16 scans instead of 4; the 16 contain the 4", "FLARE-T50", "K = 16 - K = 4",
     cell("T3", "S4", f"{f2(FC['S4']['estimate'])} {fci(FC['S4']['ci'])}; Holm p {FC['S4']['p_holm']:.4f}", "contrasts.S4"),
     "50; 3 + 3", f"{FC['S4']['reading']}; [R] (Holm)", "the weakest registered secondary; one cohort; nested draws"],
    ["A model adapted on another dataset", "four scans of AMOS, MACT or WORD instead of FLARE's", "FLARE-T50",
     "local - borrowed",
     cell("T3", "S5", f"{f2(FC['S5']['estimate'])} {fci(FC['S5']['ci'])}; Holm p {FC['S5']['p_holm']:.4f}", "contrasts.S5"),
     "50; 3 + 3", f"{FC['S5']['reading']}; [R] (Holm)",
     cell("T3", "borrowed-F0", "borrowed - F0: " + "; ".join(f"{k} {f2(F['borrowed_vs_F0'][k]['mean'])} {fci(F['borrowed_vs_F0'][k]['ci'])} {F['borrowed_vs_F0'][k]['reading']}" for k in ("amos", "mact", "word")) + " [RD]; three fixed borrowed models, two retrained one network only", "borrowed_vs_F0")],
    ["Adding pseudo-labels from unmodified TS", "4 labelled scans plus 36 scans labelled by zero-shot TS", "FLARE-T50",
     "pseudo-labels added - update",
     cell("T3", "S1", f"{f2(FC['S1']['estimate'])} {fci(FC['S1']['ci'])}; Holm p {FC['S1']['p_holm']:.4f}", "contrasts.S1"),
     "50; 1 + 1", f"{FC['S1']['reading']}; [R] (Holm, harm predicted)",
     cell("T3", "SSFT_B abs", f"one draw (s1337); this arm is still above F0 ({am('SSFT_B'):.2f} vs {am('F0'):.2f} [RD])", "absolute_macro")],
    ["", "the same 36 scans labelled by the updated model instead", "FLARE-T50", "own labels - TS labels",
     cell("T3", "S2", f"{f2(FC['S2']['estimate'])} {fci(FC['S2']['ci'])}; Holm p {FC['S2']['p_holm']:.4f}", "contrasts.S2"),
     "50; 1 + 1", f"{FC['S2']['reading']}; [R] (Holm)", "one draw"],
    ["Relabelling only (no training)", "zero-shot TS plus a class-level label map learned from the four scans", "WORD",
     "relabelling - F0",
     cell("T3", "S3 labelmap", f"{f2(est(pooled(A12['S3_LABELMAP_minus_F0'])), 3)} {fci(ci(pooled(A12['S3_LABELMAP_minus_F0'])), 3)}", "A1_2.S3_LABELMAP_minus_F0"),
     "29; 3 rules", f"{reading(A12['S3_LABELMAP_minus_F0'], pooled(A12['S3_LABELMAP_minus_F0']))}; [R] (no direction registered)",
     "boundary changes inside a class; the rules differ by draw and one draw has none"],
    ["", "", "WORD", "update - relabelling",
     cell("T3", "S4 labelmap", f"{f2(est(pooled(A12['S4_F1b_minus_LABELMAP'])))} {fci(ci(pooled(A12['S4_F1b_minus_LABELMAP'])))}", "A1_2.S4_F1b_minus_LABELMAP"),
     "29; 3 + 3", f"{reading(A12['S4_F1b_minus_LABELMAP'], pooled(A12['S4_F1b_minus_LABELMAP']))}; [R] (no direction registered)",
     "allowed wording: 'more than a class-level relabelling'"],
    ["Restore TS's other classes after the update", "every non-target class set back to zero-shot TS's output", "WORD",
     "restored - update",
     cell("T3", "S2 merge", f"{f2(est(pooled(A12['S2_MERGE_N_minus_F1b'])), 4)} {fci(ci(pooled(A12['S2_MERGE_N_minus_F1b'])), 4)}", "A1_2.S2_MERGE_N_minus_F1b"),
     "29; 3", f"{reading(A12['S2_MERGE_N_minus_F1b'], pooled(A12['S2_MERGE_N_minus_F1b']))}; [R]",
     "scored on the 12 target organs only; the restored classes were never scored against labels"],
    ["Low-rank adapters (LoRA)", "adapters instead of all weights", "FLARE-T50", "LoRA - update (draw s1337)",
     cell("T3", "LoRA", f"{f2(J['LORA_B - FULL_s1337']['mean'])}; bootstrap {fci(J['LORA_B - FULL_s1337']['ci'], 3)}; t {fci(J['LORA_B - FULL_s1337']['t_ci'], 3)}", "J_t50_posthoc contrasts['LORA_B - FULL_s1337']"),
     "50; 1 + 1", "[PH] (second read of the labels)",
     cell("T3", "LoRA-F0", f"one draw; the sign of the interval depends on its type; LoRA - F0 {f2(J['LORA_B - F0']['mean'])} {fci(J['LORA_B - F0']['ci'])}", "J_t50_posthoc")],
    ["Segmentation head only", "head weights only", "FLARE-T50", "head only - update (draw s1337)",
     cell("T3", "HEAD", f"{f2(J['HEAD - FULL_s1337']['mean'])} {fci(J['HEAD - FULL_s1337']['ci'])}", "J_t50_posthoc contrasts['HEAD - FULL_s1337']"),
     "50; 1 + 1", "[PH] (second read of the labels)",
     cell("T3", "HEAD-F0", f"one draw; head only - F0 {f2(J['HEAD - F0']['mean'])} {fci(J['HEAD - F0']['ci'])}", "J_t50_posthoc")],
    ["All weights at the default rate", "learning rate 1e-2 instead of 1e-4 (all else as the update)", "FLARE-T50",
     "lr 1e-2 - F0",
     cell("T3", "LR1E2", f"{f2(J['LR1E2 - F0']['mean'])} {fci(J['LR1E2 - F0']['ci'])}", "J_t50_posthoc contrasts['LR1E2 - F0']"),
     "50; 1", "[PH] (second read of the labels)", "one draw; explains part of the default path's loss, not all"],
]
write_table("table3_update_options", "Table 3 (facts): update options against the four-scan update and the unmodified model",
            ["Rows marked [R] are registered contrasts; 'no direction registered' rows are readings, not tested "
             "predictions. [PH] rows come from a post hoc second read of the FLARE-T50 labels (C3-D24).",
             "Never 'the default fine-tune fails' or 'worse than zero-shot': the registered reading is STOCK_NO_GAIN. "
             "Never 'full fine-tuning beats LoRA'. No ingredient of the update is credited: the routes differ in "
             "several factors at once.",
             "CI types: two-level (WORD, WORD-N pooled; FLARE-T50 Holm family as registered); scan bootstrap for one "
             "model (FLARE-T50 STOCK - F0, the post hoc rows)."], H3, ROWS3)

# ------------------------------------------------------------------ Table 4: deviations and disclosures
c3 = C.read_csv(C.TAB / "process/C3_deviations_disclosures.csv")
byid = {r["ID"]: r for r in c3}
X = [
    ["X1", "LCTSC-T23", "2026-11-17", "The heart-free primary was chosen after the LCTSC test look (post hoc analysis A2); "
     "the designer knew the LCTSC test results", "disclosure (hypothesis source)", "The new patients protect the test, not "
     "the choice of endpoint", "plan hypothesis_source", "yes: 69kwv sections 1 and 5"],
    ["X3", "LCTSC-T23", "2026-11-13 to 2026-11-18", "The 23 label archives were on disk from 2026-11-13; 'never "
     "extracted' rests on file metadata and code rules", "disclosure (label seal)", "A weaker seal than LCTSC test, whose "
     "labels were not on disk", "plan data.labels; check report note 5", "yes: 69kwv section 2"],
    ["X4", "LCTSC-T23", "2026-11-17 13:15-13:59", "Label-free heart check on the 24 LCTSC test patients ran after the "
     "freeze and before the registration", "disclosure", "No LCTSC-T23 file was opened", "H/README.md; NOTE lines",
     "yes: 69kwv section 5"],
    ["X6", "LCTSC-T23", "2026-11-18 09:22", "Both GATE log lines were written by the coordinator's gate tool, outside "
     "the frozen code", "deviation (record)", "The log shows the order; the lines are not products of the frozen code",
     "check report note 2", "no"],
    ["X8", "LCTSC-T23", "2026-11-18 15:11; 2026-11-19 11:54", "The reference labels of five patients were opened again "
     "for the qualitative figures, after the single look ('read 1 of 1')", "post hoc load (logged)", "No number was "
     "computed or changed; selection rules fixed from stored scores first", "NOTE lines 2026-11-18 15:03:40, 15:11:28; "
     "2026-11-19 11:51:50, 11:54:19", "no"],
]
H4 = ["ID", "Look / scope", "Date (KST)", "What happened", "Kind", "Effect on results", "Where recorded",
      "In a registration?"]
C3COLS = ["ID", "Look / scope", "Date (KST)", "What happened", "Kind", "Effect on results", "Where recorded",
          "Disclosed in a registration?"]
FULL = [[r[h] for h in C3COLS] for r in c3] + X
with open(C.OUT / "table4b_deviations_full.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(H4)
    w.writerows(FULL)
MAIN_IDS = ["C3-D01", "C3-D02", "C3-D03", "C3-D06", "C3-D07", "C3-D08", "C3-D15", "C3-D16", "C3-D24", "C3-D27",
            "C3-D30", "C3-D31", "C3-D40", "C3-D41"]
missing = [i for i in MAIN_IDS if i not in byid]
assert not missing, missing
ROWS4 = [[byid[i][h] for h in C3COLS] for i in MAIN_IDS] + [x for x in X if x[0] in ("X1", "X3", "X6", "X8")]
write_table("table4_deviations", "Table 4 (facts): deviations and disclosures that matter for inference",
            ["Main-text selection: every item that is in no registration (C3-D06, D07, D08, D15, D24, D27, D30, D31, "
             "D40, D41; X6, X8) plus the registered disclosures that bear on the claims (LCTSC models trained before "
             "registration, D01-D03; hypothesis sources, D16 and X1; the LCTSC-T23 label seal, X3).",
             f"All {len(c3)} rows of T/process/C3 plus the LCTSC-T23 items: table4b_deviations_full.csv.",
             "The 'What happened' cells are the compiled record text: shorten them, keep the facts."], H4, ROWS4)

# ------------------------------------------------------------------ completions report
L = []
L.append("# Completions from stored per-scan records (post hoc [PH]), 2026-11-20\n")
L.append("Computed by `build_tables.py` from the per-scan pervol records (hashes in sources_tables.json). No label, image "
         "or prediction map was opened. Each completion is checked against a registered value before use.\n")
L.append("## 1. WORD-N: level of the solid-organ volume error\n")
L.append(f"- Definition: {COMP1['definition']}.")
L.append(f"- Check: {COMP1['check']}.")
L.append("- Levels (pp): " + "; ".join(f"{a} {v[0]:.2f} (n {v[1]})" for a, v in LVL.items()) + ".\n")
L.append("## 2. LCTSC and LCTSC-T23: precision, recall and volume ratio per structure\n")
L.append(f"- Definition and checks: {COMP2['definition']}.\n")
for look in ("LCTSC", "LCTSC-T23"):
    L.append(f"**{look}**\n")
    L.append("| structure | arm | precision | recall | median V_pred / V_ref | mean Dice | n |")
    L.append("|---|---|---|---|---|---|---|")
    for s in STRUCT:
        for a in ARMS_L:
            x = COMP2[look][s][a]
            L.append(f"| {s} | {a} | {x['precision']:.2f} | {x['recall']:.2f} | {x['median_volume_ratio']:.3f} | "
                     f"{x['mean_dice']:.2f} | {x['n']} |")
            C.trace(DISP, f"completion 2 {look}", f"{s} {a}", f"{x['precision']:.2f} / {x['recall']:.2f} / "
                    f"{x['median_volume_ratio']:.3f} / {x['mean_dice']:.2f}", f"R/{C.LOOK_DIRS[look]}/look/pervol")
    L.append("")
L.append("## 3. LCTSC-T23 per-structure rows\n")
L.append("Copied, not computed: `organ_costs` and the per-draw false-positive and failure counts of "
         "`R/K_lctsc_t23/look/results.json` [RD], in completions_posthoc.json. Per draw: false-positive events "
         + "; ".join(f"{d} {len(COMP3['per_draw'][d]['fp_events'].get('arm', []))}" for d in ("S1", "S2", "S3"))
         + "; Dice-50 failures created " + "; ".join(f"{d} {COMP3['per_draw'][d]['failures'].get('n_created')}" for d in ("S1", "S2", "S3")) + ".")
open(C.OUT / "completions_posthoc.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
C.write_trace(DISP)
print("tables written; TRACE rows:", len(C.TRACE))
print("WORD-N levels:", {a: round(v[0], 2) for a, v in LVL.items()}, "positions", COMP1["pervol_positions"])
for look in ("LCTSC", "LCTSC-T23"):
    print(look, {s: (round(COMP2[look][s]["F0"]["precision"], 2), round(COMP2[look][s]["F0"]["recall"], 2),
                     round(statistics.fmean(COMP2[look][s][a]["precision"] for a in ARMS_L[1:]), 2),
                     round(statistics.fmean(COMP2[look][s][a]["recall"] for a in ARMS_L[1:]), 2)) for s in STRUCT})
