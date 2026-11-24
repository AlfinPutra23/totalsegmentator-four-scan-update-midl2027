"""Figure 1, second design (2026-11-20: the first study-flow figure was hard to understand).

(a) A timeline in time order (not to scale): the development phase, the AMOS site design that was not pre-registered in
time, and the five registered single looks with their dates, sizes and tiers; brackets join the looks that share
models. (b) The update for one draw, in four steps. Dates from table1_cohorts.csv (registration log and gate records).
No caption: the author writes it.
usage: SSL/.venv/bin/python -I -B fig1_timeline.py   (run build_tables.py first)
Version of 2026-11-20.
"""
import csv
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import disp_common as C  # noqa: E402

plt = C.setup_mpl()
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

DISP = "fig1_timeline"
T1 = {r["Look (registration)"].split(" (")[0]: r for r in csv.DictReader(open(C.OUT / "table1_cohorts.csv", encoding="utf-8"))}
C.SOURCES[C.rel(C.OUT / "table1_cohorts.csv")] = C.sha256(C.OUT / "table1_cohorts.csv")


def day(look):
    t = datetime.strptime(T1[look]["Reference labels opened (KST)"], "%Y-%m-%d %H:%M:%S")
    return f"{t.day} Nov"


LOOKS = [("WORD", "WORD test", ["29 scans", "new dataset"], 2.45),
         ("FLARE-T50", "FLARE-T50", ["50 scans", "development", "dataset"], 3.25),
         ("LCTSC", "LCTSC test", ["24 patients", "new dataset", "(thorax)"], 4.05),
         ("WORD-N", "WORD-N", ["20 scans", "same centre", "as WORD test"], 4.85),
         ("LCTSC-T23", "LCTSC-T23", ["23 patients", "same institutions", "as LCTSC test"], 5.62)]
H = 2.24
fig = plt.figure(figsize=(C.W_IN, H))


def fx(x):
    return x / C.W_IN


def fy(y):
    return 1 - y / H


def text(x, y, s, size=6.2, weight="normal", color=C.INK, ha="left", va="top", **kw):
    fig.text(fx(x), fy(y), s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, linespacing=1.15, **kw)


def line(xs, ys, color=C.INK2, lw=0.8, ls="-"):
    fig.add_artist(Line2D([fx(x) for x in xs], [fy(y) for y in ys], color=color, lw=lw, ls=ls))


text(0.0, 0.04, "The study in time order (not to scale)", 8.0, "bold")
YL = 0.78
fig.add_artist(FancyBboxPatch((fx(0.0), fy(YL + 0.30)), fx(1.32), 0.60 / H, boxstyle="square,pad=0",
                              transform=fig.transFigure, fc=C.BAND, ec=C.INK2, ls=(0, (3, 2)), lw=0.8))
text(0.66, YL - 0.22, "development", 6.8, "bold", ha="center")
text(0.66, YL - 0.06, "reused internal sets,\nexploratory", 6.2, color=C.INK2, ha="center")
fig.add_artist(FancyArrowPatch((fx(1.36), fy(YL)), (fx(5.98), fy(YL)), transform=fig.transFigure,
                               arrowstyle="-|>", mutation_scale=8, lw=0.9, color=C.INK2))
fig.add_artist(Line2D([fx(1.80)], [fy(YL)], marker="D", ms=5.5, mfc="white", mec=C.INK2, mew=1.0, ls="none"))
text(1.80, YL - 0.13, "AMOS site\ndesign", 6.4, "bold", ha="center", va="bottom")
text(1.80, YL + 0.12, "4 Nov\nnot pre-registered\nin time", 6.0, color=C.INK2, ha="center")
for key, name, lines, x in LOOKS:
    fig.add_artist(Line2D([fx(x)], [fy(YL)], marker="o", ms=7.5, mfc=C.GREEN, mec="white", mew=0.8, ls="none"))
    text(x, YL - 0.13, name, 6.8, "bold", ha="center", va="bottom")
    text(x, YL + 0.12, day(key), 6.4, "bold", ha="center")
    text(x, YL + 0.25, "\n".join(lines), 6.0, color=C.INK2, ha="center")
    C.trace(DISP, f"(a) {name}", "reference labels opened (day)", day(key), "table1_cohorts.csv")
# brackets for the looks that share models
for x1, x2, yb, s in ((2.45, 4.85, 1.52, "same centre, same three models"),
                      (4.05, 5.62, 1.80, "same institutions, same three models")):
    line([x1, x1, x2, x2], [yb - 0.07, yb, yb, yb - 0.07])
    text((x1 + x2) / 2, yb + 0.03, s, 6.0, color=C.INK2, ha="center")
fig.add_artist(Line2D([fx(2.0)], [fy(2.12)], marker="o", ms=6, mfc=C.GREEN, mec="white", ls="none"))
text(2.10, 2.12, "registered single look: the registration was public before the reference labels were opened",
     6.0, color=C.INK2, va="center")

C.save(fig, DISP)
C.write_trace(DISP)
print(f"{DISP}: {C.W_IN:.2f} x {H:.2f} in")
