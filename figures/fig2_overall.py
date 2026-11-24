"""Figure 2, second design (2026-11-20: the first forest was hard to understand).

Before -> after per look: mean Dice of unmodified TotalSegmentator (grey) and of the four-scan update (green, mean of the
three draws), with the default nnU-Net fine-tune path (brown square) where it was run; the registered change and its
95% CI on the right. Absolute values from the results.json files; each change is checked against (updated - unmodified).
No caption: the author writes it. Facts in TRACE_fig2_overall.csv.
usage: SSL/.venv/bin/python -I -B fig2_overall.py
Version of 2026-11-20.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import disp_common as C  # noqa: E402

plt = C.setup_mpl()
from matplotlib.lines import Line2D  # noqa: E402

DISP = "fig2_overall"
RES = {k: C.results(k) for k in C.LOOK_DIRS}
GREYDOT = "#9aa0a8"


def num(v):
    return v if isinstance(v, (int, float)) else (v.get("mean") if isinstance(v, dict) else None)


def mean(xs):
    return sum(xs) / len(xs)


w = RES["WORD"]["A1_2"]["descriptive_absolute_dice_vista3d_link"]
wn = RES["WORD-N"]["A1_2_readings_fresh_descriptive"]["descriptive_absolute_dice_vista3d_link"]
fa = RES["FLARE-T50"]["absolute_macro"]
la = RES["LCTSC"].get("absolute_macro")
ta = RES["LCTSC-T23"]["absolute_macro"]
if isinstance(la, dict) and "four_structures" in la:
    la = la["four_structures"]
assert isinstance(la, dict) and "F0" in la, la


def lmean(d):
    return mean([num(d[k]) for k in ("F1b_S1", "F1b_S2", "F1b_S3")])


ROWS = [  # name, line 2, unmodified, updated, default path, registered change, role
    ("WORD test", "29 scans · 12 organs", w["F0"]["macro_penalised_12"], w["F1b"]["macro_penalised_12"],
     w["STOCK"]["macro_penalised_12"], RES["WORD"]["P1_gain"], "primary"),
    ("WORD-N", "20 scans · same centre as WORD", wn["F0"]["macro_penalised_12"], wn["F1b"]["macro_penalised_12"],
     wn["STOCK"]["macro_penalised_12"], RES["WORD-N"]["P1_fresh"]["p1"], "secondary"),
    ("FLARE-T50", "50 scans · 13 organs", num(fa["F0"]),
     mean([num(fa[k]) for k in ("FULL_s1337", "FULL_s2025", "FULL_s3407")]), num(fa["STOCK"]),
     RES["FLARE-T50"]["R1"], "primary"),
    ("LCTSC test", "24 patients · 4 structures", num(la["F0"]), lmean(la), None, RES["LCTSC"]["P1"], "primary"),
    ("LCTSC-T23", "23 patients · heart excluded", num(ta["heart_free"]["F0"]), lmean(ta["heart_free"]), None,
     RES["LCTSC-T23"]["P1_HF"], "primary"),
    ("LCTSC-T23, with heart", "same 23 patients · 4 structures", num(ta["four_structures"]["F0"]),
     lmean(ta["four_structures"]), None, RES["LCTSC-T23"]["S_4S"], "secondary"),
]
for name, _, f0, up, st, reg, role in ROWS:
    assert abs((up - f0) - reg["M"]) < 0.01, (name, up - f0, reg["M"])

XMIN, XMAX = 78.0, 92.5
PITCH, TOP, BOT = 0.43, 0.62, 0.42
H = TOP + PITCH * len(ROWS) + BOT
fig = plt.figure(figsize=(C.W_IN, H))
X_AX, W_AX, X_VAL = 1.78, 3.02, 4.98
ax = fig.add_axes([X_AX / C.W_IN, BOT / H, W_AX / C.W_IN, PITCH * len(ROWS) / H])
ax.set_xlim(XMIN, XMAX)
ax.set_ylim(len(ROWS) - 0.5, -0.5)
ax.set_yticks([])
ax.patch.set_alpha(0)
for sp in ("left", "right", "top"):
    ax.spines[sp].set_visible(False)
ticks = list(range(78, 93, 2))
ax.set_xticks(ticks)
for t in ticks:
    ax.axvline(t, color=C.RULE, lw=0.45, zorder=0)
ax.set_xlabel("Dice (points), mean over the look's structures", fontsize=7)
for yb in (1.5, 2.5):
    ax.axhline(yb, color=C.RULE, lw=0.7, zorder=0)


def fyrow(i):
    return (BOT + PITCH * (len(ROWS) - 0.5 - i)) / H


M = C.MINUS
for i, (name, l2, f0, up, st, reg, role) in enumerate(ROWS):
    ax.annotate("", xy=(up, i), xytext=(f0, i), arrowprops=dict(arrowstyle="-|>", color=C.GREEN, lw=1.6,
                                                                 shrinkA=4.5, shrinkB=4.5, mutation_scale=9), zorder=3)
    ax.plot([f0], [i], "o", ms=7, mfc=GREYDOT, mec="white", mew=0.7, zorder=4)
    ax.plot([up], [i], "o", ms=7, mfc=C.GREEN, mec="white", mew=0.7, zorder=4)
    ax.text(f0 - 0.28, i + 0.02, f"{f0:.1f}", ha="right", va="center", fontsize=6.4, color=C.INK2)
    ax.text(up + 0.28, i + 0.02, f"{up:.1f}", ha="left", va="center", fontsize=6.4, color=C.GREEN)
    if st is not None:
        if st < XMIN:
            ax.plot([XMIN + 0.12], [i], "<", ms=5, color=C.BROWN, zorder=4)
            ax.text(XMIN + 0.45, i - 0.27, f"{st:.1f}", ha="left", va="center", fontsize=6.2, color=C.BROWN)
        else:
            ax.plot([st], [i], "s", ms=5.5, mfc=C.BROWN, mec="white", mew=0.6, zorder=4)
            ax.text(st, i - 0.27, f"{st:.1f}", ha="center", va="center", fontsize=6.2, color=C.BROWN)
    fig.text(0.02 / C.W_IN, fyrow(i) + 0.055 / H, name, ha="left", va="center", fontsize=7.4, fontweight="bold",
             color=C.INK)
    fig.text(0.02 / C.W_IN, fyrow(i) - 0.095 / H, l2, ha="left", va="center", fontsize=6.3, color=C.INK2)
    txt = f"{C.sg(reg['M'], 2, M)} [{reg['ci'][0]:.2f}, {reg['ci'][1]:.2f}]"
    fig.text(X_VAL / C.W_IN, fyrow(i) + 0.055 / H, txt, ha="left", va="center", fontsize=7.0, color=C.INK)
    fig.text(X_VAL / C.W_IN, fyrow(i) - 0.095 / H, role + (", heart included" if "with heart" in name else ""),
             ha="left", va="center", fontsize=6.3, color=C.INK2)
    C.trace(DISP, name, "unmodified / updated (mean of three draws) / default path",
            f"{f0:.2f} / {up:.2f} / {'-' if st is None else f'{st:.2f}'}", "results.json absolute macros")
    C.trace(DISP, name, "registered change [95% CI], two-level", txt, f"{name}: registered reading ({role})")

fig.text(X_VAL / C.W_IN, 1 - (TOP - 0.10) / H, "Change [95% CI]", ha="left", va="center", fontsize=7.2,
         fontweight="bold", color=C.INK)
hand = [Line2D([], [], ls="none", marker="o", ms=7, mfc=GREYDOT, mec="white"),
        Line2D([], [], ls="none", marker="o", ms=7, mfc=C.GREEN, mec="white"),
        Line2D([], [], ls="none", marker="s", ms=5.5, mfc=C.BROWN, mec="white")]
lab = ["unmodified TotalSegmentator", "updated with four labelled scans (mean of three draws)",
       "default nnU-Net fine-tune path (FLARE-T50: one model; not run on LCTSC)"]
fig.legend(hand, lab, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=1, fontsize=6.6, frameon=False,
           handletextpad=0.4, labelspacing=0.25, borderaxespad=0.05)
C.save(fig, DISP)
C.write_trace(DISP)
print(f"{DISP}: {C.W_IN:.2f} x {H:.2f} in; rows " + "; ".join(f"{r[0]} {r[2]:.2f}->{r[3]:.2f}" for r in ROWS))
