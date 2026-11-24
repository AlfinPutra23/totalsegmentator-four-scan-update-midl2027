"""Figure 4 (storyline B): two LCTSC-T23 patients, coronal slice through the centre of the reference heart.

A re-layout of docs/midl_qualitative_2026-11-18/paper_fig_typical_coronal (2026-11-19) from the cached slices only
(slices_typical.npz / slices_typical.json, sha256-checked); no image, label or prediction map is opened. Rows: the
75th- and 25th-percentile patients of the heart-free change (the median row of the 2026-11-19 version is dropped to
save height; display choice, post hoc). Legend reworded from 'TS wrong / fine-tuned right' to agreement with the
reference, and patient IDs shortened (full IDs in README.md). Drawing code: make_paper_figure.py (sha256-checked).
No caption: the author writes it. Facts in TRACE_fig4_coronal.csv.
usage: SSL/.venv/bin/python -I -B fig_coronal.py
Version of 2026-11-20.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import disp_common as C  # noqa: E402

EXPECT = {"make_paper_figure.py": "14fe265fb90c28255e689192a24c0de68c991cddc90a1070152bb6966f208d3b",
          "slices_typical.npz": "6d72f0db64536e4bbd0c84b99156aba631b141bdadc20010391d6e22f3640074",
          "slices_typical.json": "356af44a3ed53cdad29bb9762160546ea33ef47d8d2cca09a316aada2abca609"}
for f, h in EXPECT.items():
    assert C.sha256(C.QUAL / f) == h, f"{f} differs from its recorded sha256"
    C.SOURCES[C.rel(C.QUAL / f)] = h
sys.path.insert(0, str(C.QUAL))
import make_paper_figure as PF  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DISP = "fig4_coronal"
VIEW = "coronal_heartc"
ROWS = ["p75", "p25"]
PCT_TXT = {"p75": "75th percentile (heart-free)", "p25": "25th percentile (heart-free)"}
RULE_TXT = "slice through the centre\nof the reference heart"
PF.COLS = ["Reference", "TotalSegmentator\n(unmodified)", "Updated\n(four scans)", "Gained / lost\nagreement"]
PF.FIXED_TXT = "gained agreement with the reference"
PF.NEW_TXT = "lost agreement with the reference"

meta = json.load(open(C.QUAL / "slices_typical.json"))
arr = dict(np.load(C.QUAL / "slices_typical.npz"))
cw = (PF.W_IN - 3 * PF.GAP) / 4
rows = [(case, cw * PF.aspect(arr, meta, case, VIEW)) for case in ROWS]
HEAD, BAND, RGAP, LEG = 0.30, 0.04 + 2 * PF.LINE + 0.03, 0.05, 0.36
H = HEAD + sum(h + BAND for _, h in rows) + RGAP * (len(rows) - 1) + LEG
fig = plt.figure(figsize=(PF.W_IN, H))
for j, t in enumerate(PF.COLS):
    PF.ftext(fig, H, j * (cw + PF.GAP) + cw / 2, 0.02, t, ha="center", va="top", fontsize=PF.F_HEAD,
             fontweight="bold", color=PF.INK, linespacing=1.1)
y = HEAD
for i, (case, h) in enumerate(rows):
    r = meta["cases"][case]
    assert r["views"][VIEW]["rule"] == "heart_centroid", (case, VIEW)
    assert f"p{r['selection']['percentile']}" == case
    PF.draw_row(fig, H, meta, arr, case, VIEW, 0.0, y, cw, h, DISP, letter="ab"[i])
    yb = y + h + 0.04
    short = r["pid"].replace("LCTSC-Train-", "patient ")
    PF.ftext(fig, H, cw / 2, yb, f"{short}, {PF.Q.base(VIEW)}", ha="center", va="top", fontsize=PF.F_BAND,
             color=PF.INK)
    PF.ftext(fig, H, cw / 2, yb + PF.LINE, PCT_TXT[case], ha="center", va="top", fontsize=PF.F_BAND, color=PF.INK2)
    C.trace(DISP, f"row {'ab'[i]}", "patient and rank", f"{r['pid']}; percentile {r['selection']['percentile']}, "
            f"ascending rank {r['selection']['rank_ascending']} of 23", "slices_typical.json cases.%s.selection" % case)
    for j, a in ((1, "F0"), (2, r["shown_draw"])):
        d = r["pervol_dice"][a]
        PF.ftext(fig, H, j * (cw + PF.GAP) + cw / 2, yb, PF.dice_text(d), ha="center", va="top", fontsize=PF.F_BAND,
                 color=PF.INK, linespacing=1.3)
        for k, nm in enumerate(PF.ORGANS):
            C.trace(DISP, f"row {'ab'[i]} column {j + 1}", f"Dice {a} {nm}", f"{d[k]:.1f}",
                    f"slices_typical.json cases.{case}.pervol_dice.{a}[{k}]")
    PF.ftext(fig, H, 3 * (cw + PF.GAP) + cw / 2, yb, RULE_TXT, ha="center", va="top", fontsize=PF.F_BAND,
             color=PF.INK2, linespacing=1.3)
    y += h + BAND + RGAP
PF.legend(fig, H, y - RGAP, 0.0, PF.W_IN)
PF.save(fig, C.OUT, DISP)
C.TRACE.extend(PF.TRACE)
C.write_trace(DISP)
print(f"{DISP}: {PF.W_IN:.2f} x {H:.2f} in; TRACE rows {len(C.TRACE)}")
