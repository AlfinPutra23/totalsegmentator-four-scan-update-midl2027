"""Figure 3, third design (2026-11-20: "too many blanks" in the one-grid table, 50 of 95 cells empty).

Two compact colour-coded tables side by side: abdomen (WORD test, WORD-N, FLARE-T50) and thorax (LCTSC test,
LCTSC-T23). Each cell gives the Dice change of the four-scan update against unmodified TotalSegmentator (mean of the three
draws), coloured by its registered class at the pre-set -1.0 margin. FLARE-T50 labels the adrenals separately, so its
adrenal cell is split (L / R); WORD labels them as one class. 6 cells stay empty: structures a dataset does not label.
Values and classes are copied from the organ-cost blocks of the results.json files (45 readings). No caption.
usage: SSL/.venv/bin/python -I -B fig3_table.py
Version of 2026-11-20.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import disp_common as C  # noqa: E402

plt = C.setup_mpl()
from matplotlib.patches import Rectangle  # noqa: E402

DISP = "fig3_table"
RES = {k: C.results(k) for k in C.LOOK_DIRS}
PER = {"WORD": RES["WORD"]["P1_organ_costs"]["per_organ"],
       "WORD-N": RES["WORD-N"]["A1_readings_fresh_descriptive"]["P1_organ_costs"]["per_organ"],
       "FLARE-T50": RES["FLARE-T50"]["organ_costs"]["per_organ"],
       "LCTSC": RES["LCTSC"]["organ_costs"]["per_organ"],
       "LCTSC-T23": RES["LCTSC-T23"]["organ_costs"]["per_organ"]}
LEFT_COLS = [("WORD", "WORD test", "29 scans", "registered"), ("WORD-N", "WORD-N", "20 scans", "descriptive"),
             ("FLARE-T50", "FLARE-T50", "50 scans", "descriptive")]
RIGHT_COLS = [("LCTSC", "LCTSC test", "24 patients", "registered"),
              ("LCTSC-T23", "LCTSC-T23", "23 patients", "descriptive")]
LEFT_ROWS = [
    ("liver", {"WORD": "liver", "WORD-N": "liver", "FLARE-T50": "liver"}),
    ("spleen", {"WORD": "spleen", "WORD-N": "spleen", "FLARE-T50": "spleen"}),
    ("left kidney", {"WORD": "kidney_left", "WORD-N": "kidney_left", "FLARE-T50": "l.kid"}),
    ("right kidney", {"WORD": "kidney_right", "WORD-N": "kidney_right", "FLARE-T50": "r.kid"}),
    ("stomach", {"WORD": "stomach", "WORD-N": "stomach", "FLARE-T50": "stom"}),
    ("gallbladder", {"WORD": "gallbladder", "WORD-N": "gallbladder", "FLARE-T50": "gall"}),
    ("esophagus", {"WORD": "esophagus", "WORD-N": "esophagus", "FLARE-T50": "esoph"}),
    ("pancreas", {"WORD": "pancreas", "WORD-N": "pancreas", "FLARE-T50": "panc"}),
    ("duodenum", {"WORD": "duodenum", "WORD-N": "duodenum", "FLARE-T50": "duod"}),
    ("adrenal glands", {"WORD": "adrenal", "WORD-N": "adrenal", "FLARE-T50": ("l.adr", "r.adr")}),
    ("small intestine", {"WORD": "intestine", "WORD-N": "intestine"}),
    ("bladder", {"WORD": "bladder", "WORD-N": "bladder"}),
    ("aorta", {"FLARE-T50": "aorta"}),
    ("IVC", {"FLARE-T50": "ivc"}),
]
RIGHT_ROWS = [("esophagus", {"LCTSC": "esophagus", "LCTSC-T23": "esophagus"}),
              ("heart", {"LCTSC": "heart", "LCTSC-T23": "heart"}),
              ("left lung", {"LCTSC": "lung_left", "LCTSC-T23": "lung_left"}),
              ("right lung", {"LCTSC": "lung_right", "LCTSC-T23": "lung_right"})]
used = set()
for _, m in LEFT_ROWS + RIGHT_ROWS:
    for lk, key in m.items():
        for k in (key if isinstance(key, tuple) else (key,)):
            assert (lk, k) not in used
            used.add((lk, k))
assert used == {(lk, o) for lk, d in PER.items() for o in d}, "every structure reading must appear exactly once"


def klass(b):
    if b["ni"] == "NON-INFERIOR" and b["harm"] == "BENEFICIAL":
        return "ben"
    if b["ni"] == "NON-INFERIOR":
        return "ni"
    if b["ni"] == "INCONCLUSIVE":
        return "inc"
    raise ValueError(b)


FILL = {"ben": "#d3ebda", "ni": "#e7ebf1", "inc": "#f6e1b0"}
TXT = {"ben": "#1d5b33", "ni": C.INK, "inc": "#6a4800"}
COUNT = {"ben": 0, "ni": 0, "inc": 0}
ROW_H, W_COL, HEAD = 0.205, 0.80, 0.72
XL, XR = 1.04, 4.20                       # left edge of the first value column of each table (inches)
H = HEAD + ROW_H * len(LEFT_ROWS) + 0.08
fig = plt.figure(figsize=(C.W_IN, H))


def fx(x):
    return x / C.W_IN


def fy(y):
    return 1 - y / H


def rect(x, y, w, h, fc, ec="white", lw=1.2):
    fig.add_artist(Rectangle((fx(x), fy(y + h)), fx(w), h / H, transform=fig.transFigure, fc=fc, ec=ec, lw=lw))


def cell(x, y, w, b, look, key, label, size=7.0, prefix=""):
    k = klass(b)
    COUNT[k] += 1
    rect(x, y, w, ROW_H, FILL[k])
    t = C.sg(b["M"], 2, C.MINUS)
    fig.text(fx(x + w / 2), fy(y + ROW_H / 2), prefix + t, ha="center", va="center", fontsize=size, color=TXT[k],
             fontweight="bold" if k == "ben" else "normal")
    C.trace(DISP, f"{label} / {look}", "Dice change (mean of three draws); class", f"{prefix}{t}; {k}",
            f"{look} results.json per_organ.{key}: M {b['M']:.4f}, ci {b['ci']}, {b['ni']}, {b['harm']}")


def table(x0, title, cols, rows):
    fig.text(fx(x0 - 0.06), fy(0.09), title, ha="right", va="center", fontsize=7.8, fontweight="bold", color=C.INK)
    for j, (lk, name, n, tag) in enumerate(cols):
        xc = x0 + j * W_COL + W_COL / 2
        fig.text(fx(xc), fy(0.20), name, ha="center", va="center", fontsize=7.2, fontweight="bold", color=C.INK)
        fig.text(fx(xc), fy(0.36), n, ha="center", va="center", fontsize=6.3, color=C.INK2)
        fig.text(fx(xc), fy(0.51), tag, ha="center", va="center", fontsize=6.0, color=C.INK2, style="italic")
    y = HEAD
    for label, m in rows:
        fig.text(fx(x0 - 0.07), fy(y + ROW_H / 2), label, ha="right", va="center", fontsize=7.0, color=C.INK)
        for j, (lk, *_r) in enumerate(cols):
            x = x0 + j * W_COL
            key = m.get(lk)
            if key is None:
                rect(x, y, W_COL, ROW_H, "white")
            elif isinstance(key, tuple):
                for s, (kk, side) in enumerate(zip(key, ("L ", "R "))):
                    cell(x + s * W_COL / 2, y, W_COL / 2, PER[lk][kk], lk, kk, label, size=6.2, prefix=side)
            else:
                cell(x, y, W_COL, PER[lk][key], lk, key, label)
        y += ROW_H
    return y


yl_end = table(XL, "Abdomen", LEFT_COLS, LEFT_ROWS)
yr_end = table(XR, "Thorax", RIGHT_COLS, RIGHT_ROWS)
assert COUNT == {"ben": 26, "ni": 13, "inc": 6}, COUNT
y = yr_end + 0.22
for k, s in (("ben", f"improved: 95% CI above 0 ({COUNT['ben']})"),
             ("ni", f"no gain shown: not worse than {C.MINUS}1 ({COUNT['ni']})"),
             ("inc", f"inconclusive: CI crosses {C.MINUS}1 ({COUNT['inc']})")):
    rect(XR - 0.60, y, 0.22, 0.13, FILL[k], ec=C.RULE, lw=0.6)
    fig.text(fx(XR - 0.32), fy(y + 0.065), s, ha="left", va="center", fontsize=6.5, color=C.INK)
    y += 0.21
fig.text(fx(XR - 0.60), fy(y + 0.06), "Dice change (points): updated minus unmodified\nTotalSegmentator, mean of "
         f"three draws. Pre-set\nmargin {C.MINUS}1.0 Dice; no structure read inferior.\nBlank: not labelled in "
         "that dataset. FLARE-T50\nlabels the adrenals separately (L / R).", ha="left", va="top", fontsize=6.1,
         color=C.INK2, linespacing=1.25)
C.save(fig, DISP)
C.write_trace(DISP)
blank = sum(1 for _, m in LEFT_ROWS for lk, *_ in LEFT_COLS if lk not in m) + \
    sum(1 for _, m in RIGHT_ROWS for lk, *_ in RIGHT_COLS if lk not in m)
print(f"{DISP}: {C.W_IN:.2f} x {H:.2f} in; classes {COUNT}; empty cells {blank}")
