"""Figure 6 (pick 2 of 2026-11-20): the 22 registered predictions.

One box per directional prediction registered for the five looks, one row per look (primary endpoint first), coloured
by its reading: held, not held, not run. Bold outline: part of the look's primary endpoint (WORD's primary had two
parts, the gain and the organ costs).
Members and readings: docs/midl_tables_2026-11-16/process/C1_registered_prediction_scorecard.csv (four looks; column
'Held?') and the LCTSC-T23 rows X01 and X02 of panel v8's registered audit (P1A section 4.1), whose readings are read
here from the LCTSC-T23 results.json. Each C1 reading token is checked to occur as a stored reading in that look's
results.json. Counts asserted: 22 directional, 21 decided, 18 held, 3 not held, 1 not run.
No caption: the author writes it. Facts in TRACE_fig6_predictions.csv.
usage: SSL/.venv/bin/python -I -B fig6_predictions.py
Version of 2026-11-20.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import disp_common as C  # noqa: E402

plt = C.setup_mpl()
from matplotlib.patches import Rectangle  # noqa: E402

DISP = "fig6_predictions"
RES = {k: C.results(k) for k in C.LOOK_DIRS}
C1 = {r["ID"].replace("C1-", ""): r for r in C.read_csv(C.TAB / "process/C1_registered_prediction_scorecard.csv")}

ROWS = [  # name, OSF id, n, look key, members in display order
    ("WORD test", "c5fyj", "29 scans", "WORD", [["W01", "W02", "W04", "W08"]]),
    ("WORD-N", "7jser", "20 scans", "WORD-N", [["N01", "N04", "N02", "N03", "N06"]]),
    ("FLARE-T50", "ajfrc", "50 scans", "FLARE-T50", [["T01", "T02", "T03", "T04", "T05"], ["T06", "T07", "T08", "T09"]]),
    ("LCTSC test", "d6ayj", "24 patients", "LCTSC", [["L01", "L02"]]),
    ("LCTSC-T23", "69kwv", "23 patients", "LCTSC-T23", [["X01", "X02"]]),
]
PRIMARY = {"W01", "W02", "N01", "T01", "L01", "X01"}
LABEL = {"W01": "Dice gain", "W02": "all 12 organs\nnon-inferior", "W04": "share of error\nremoved",
         "W08": "class-merge variant\nnon-inferior", "N01": "solid-organ\nvolume error", "N04": "Dice gain",
         "N02": "duodenum\nover-segmentation", "N03": "esophagus\nover-segmentation", "N06": "VISTA3D\ncomparison",
         "T01": "Dice gain", "T02": "pseudo-labels\nlower the gain", "T03": "pseudo-label\nsource",
         "T04": "update vs\ndefault path", "T05": "16 vs 4\nlabelled scans", "T06": "own vs\nborrowed models",
         "T07": "solid-organ\nvolume error", "T08": "1 vs 4\nlabelled scans", "T09": "risk with one\nlabelled scan",
         "L01": "Dice gain", "L02": "all 4 structures\nnon-inferior", "X01": "Dice gain,\nheart excluded",
         "X02": "Dice gain,\nwith heart"}
REASON = {"W02": "3 inconclusive, 0 inferior", "N02": "not confirmed", "N03": "not confirmed",
          "N06": "out of GPU memory"}


def strings(o):
    if isinstance(o, dict):
        for v in o.values():
            yield from strings(v)
    elif isinstance(o, list):
        for v in o:
            yield from strings(v)
    elif isinstance(o, str):
        yield o


def norm(s):
    return re.sub(r"[\s_]+", "_", s.strip().upper())


STORED = {k: {norm(s) for s in strings(r)} for k, r in RES.items()}
T2 = RES["FLARE-T50"]["tier2"]
T08 = [k for k, v in T2.items() if isinstance(v, dict) and v.get("reading") == "DETECTED"]
assert len(T08) == 1, T08
KEY = {  # prediction -> (look, key path in results.json, stored reading expected for the C1 / P1A status)
    "W01": ("WORD", ("P1_gain", "reading"), "GAIN"), "W02": ("WORD", ("P1_organ_costs", "reading"), "INCONCLUSIVE"),
    "W04": ("WORD", ("secondary_S_TS", "claim_systematic_share_present"), True),
    "W08": ("WORD", ("A1_2", "S2_MERGE_N_minus_F1b", "reading_ni"), "MERGE_N_NON_INFERIOR"),
    "T01": ("FLARE-T50", ("R1", "reading"), "REPLICATED"),
    **{f"T0{i + 1}": ("FLARE-T50", ("secondary_holm_family", "contrasts", f"S{i}", "reading"), "DETECTED")
       for i in range(1, 7)},
    "T08": ("FLARE-T50", ("tier2", T08[0], "reading"), "DETECTED"),
    "T09": ("FLARE-T50", ("tier2", "single_scan_risk", "reading"), "REPLICATED"),
    "L01": ("LCTSC", ("P1", "reading"), "GAIN"), "L02": ("LCTSC", ("organ_costs", "reading"), "ALL_NI"),
    "N01": ("WORD-N", ("P_V", "reading"), "VOLUME_ERROR_REDUCED"),
    "N02": ("WORD-N", ("holm_family", "results", "OS-D", "reading"), "NOT_CONFIRMED"),
    "N03": ("WORD-N", ("holm_family", "results", "OS-E", "reading"), "NOT_CONFIRMED"),
    "N04": ("WORD-N", ("holm_family", "results", "P1-fresh", "reading"), "GAIN"),
    "N06": ("WORD-N", ("tier2", "V_P1", "reading"), "NOT RUN"),
    "X01": ("LCTSC-T23", ("P1_HF", "reading"), "GAIN"), "X02": ("LCTSC-T23", ("S_4S", "reading"), "GAIN")}


def stored(look, path):
    o = RES[look]
    for k in path:
        o = o[k]
    return o


for pid, (look, path, want) in KEY.items():
    assert stored(look, path) == want, (pid, path, stored(look, path), want)
STATUS, TOKENS = {}, {}
for name, osf, n, look, lines in ROWS:
    for pid in [p for ln in lines for p in ln]:
        if pid.startswith("X"):
            blk = RES["LCTSC-T23"]["P1_HF" if pid == "X01" else "S_4S"]
            tok = blk["reading"]
            STATUS[pid] = "held" if tok == "GAIN" and blk["pass"] else "not held"
            src = f"LCTSC-T23 results.json {'P1_HF' if pid == 'X01' else 'S_4S'}.reading = {tok}; P1A section 4.1"
        else:
            r = C1[pid]
            tok = re.split(r"[:;]", r["Outcome reading: estimate [95% CI]"])[0].strip()
            STATUS[pid] = {"yes": "held", "no": "not held", "not run": "not run"}[r["Held?"]]
            lk, path, want = KEY[pid]
            src = (f"C1 row C1-{pid} (Held? = {r['Held?']}); {lk} results.json {'.'.join(path)} = {want} "
                   f"(asserted)")
        TOKENS[pid] = tok
        C.trace(DISP, f"{name} / {pid}", "prediction; reading", f"{LABEL[pid].replace(chr(10), ' ')}; {STATUS[pid]}",
                src)
count = {s: list(STATUS.values()).count(s) for s in ("held", "not held", "not run")}
assert len(STATUS) == 22 and count == {"held": 18, "not held": 3, "not run": 1}, count
assert {p for p, s in STATUS.items() if s != "held"} == {"W02", "N02", "N03", "N06"}
assert RES["WORD"]["P1_gain"]["reading"] == "GAIN" and RES["FLARE-T50"]["R1"]["reading"] == "REPLICATED"
assert RES["LCTSC"]["P1"]["reading"] == "GAIN" and RES["WORD-N"]["P1_fresh"]["p1"]["pass"]

FILL = {"held": "#d3ebda", "not held": "#f6e1b0", "not run": "white"}
TXT = {"held": "#1d5b33", "not held": "#6a4800", "not run": C.INK2}
X0, BW, BH, GX, GY = 1.12, 0.89, 0.33, 0.06, 0.07
HEAD, ROWGAP = 0.10, 0.10
nlines = sum(len(r[4]) for r in ROWS)
H = HEAD + nlines * (BH + GY) + (len(ROWS) - 1) * ROWGAP + 0.50
fig = plt.figure(figsize=(C.W_IN, H))


def fx(x):
    return x / C.W_IN


def fy(y):
    return 1 - y / H


y = HEAD
for name, osf, n, look, lines in ROWS:
    ids = [p for ln in lines for p in ln]
    k = {s: sum(STATUS[p] == s for p in ids) for s in ("held", "not held", "not run")}
    summ = ", ".join(f"{v} {s}" for s, v in k.items() if v)
    fig.text(fx(0.02), fy(y + 0.06), name, ha="left", va="top", fontsize=7.4, fontweight="bold", color=C.INK)
    fig.text(fx(0.02), fy(y + 0.19), f"osf.io/{osf} · {n}", ha="left", va="top", fontsize=6.1, color=C.INK2)
    fig.text(fx(0.02), fy(y + 0.31), summ, ha="left", va="top", fontsize=5.9, color=C.INK2)
    for ln in lines:
        for j, pid in enumerate(ln):
            st = STATUS[pid]
            x = X0 + j * (BW + GX)
            prim = pid in PRIMARY
            fig.add_artist(Rectangle((fx(x), fy(y + BH)), fx(BW), BH / H, transform=fig.transFigure, fc=FILL[st],
                                     ec=C.INK if prim else (C.RULE if st != "not run" else C.INK2),
                                     lw=1.3 if prim else 0.6, ls="-" if st != "not run" else (0, (2, 1.5))))
            txt = LABEL[pid]
            nl = txt.count("\n") + 1 + (1 if pid in REASON else 0)
            yc = y + BH / 2 - (0.045 if pid in REASON else 0) + (0.0 if nl < 3 else 0.0)
            fig.text(fx(x + BW / 2), fy(yc), txt, ha="center", va="center", fontsize=6.2, color=TXT[st],
                     fontweight="bold" if prim else "normal", linespacing=1.05)
            if pid in REASON:
                fig.text(fx(x + BW / 2), fy(y + BH - 0.055), REASON[pid], ha="center", va="center", fontsize=5.6,
                         color=TXT[st], style="italic")
        y += BH + GY
    y += ROWGAP
y += 0.04
lx = 0.02
for st, lab in (("held", f"held ({count['held']})"), ("not held", f"not held ({count['not held']})"),
                ("not run", f"not run ({count['not run']})")):
    fig.add_artist(Rectangle((fx(lx), fy(y + 0.13)), fx(0.24), 0.13 / H, transform=fig.transFigure, fc=FILL[st],
                             ec=C.INK2 if st == "not run" else C.RULE, lw=0.6,
                             ls="-" if st != "not run" else (0, (2, 1.5))))
    fig.text(fx(lx + 0.30), fy(y + 0.065), lab, ha="left", va="center", fontsize=6.6, color=C.INK)
    lx += 1.05
fig.add_artist(Rectangle((fx(lx), fy(y + 0.13)), fx(0.24), 0.13 / H, transform=fig.transFigure, fc="white",
                         ec=C.INK, lw=1.3))
fig.text(fx(lx + 0.30), fy(y + 0.065), "bold outline: part of the look's primary endpoint", ha="left", va="center",
         fontsize=6.6, color=C.INK)
fig.text(fx(0.02), fy(y + 0.25), f"22 directional predictions registered for the five looks: 21 decided, "
         f"{count['held']} held, {count['not held']} not held, {count['not run']} not run.", ha="left", va="center",
         fontsize=6.6, color=C.INK2)
C.save(fig, DISP)
C.write_trace(DISP)
print(f"{DISP}: {C.W_IN:.2f} x {H:.2f} in; {count}")
