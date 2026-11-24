"""LCTSC-T23 qualitative figure, paper version (2026-11-19; prereg log NOTE 2026-11-19 11:08:34).

The same two patients, shown draws and slices as the 2026-11-18 figures, laid out for the MIDL page, so the figures go
in with \\includegraphics[width=\\textwidth]{...} unscaled and every font prints at the size set here:
  text width 6.0 in = 432 pt and text height about 8.5 in, measured from the official template
  (github.com/MIDL-Conference/MIDLLatexTemplate, midl-fullpaper.pdf: text block x = 90 to 522 pt on US Letter;
  body and captions Computer Modern 10 pt).
Reads slices.npz and slices.json only, after checking their sha256; opens no CT, label, map or RTSTRUCT. Display rules
(orientation, CT window, crop, colours, overlays, change map) are imported unchanged from make_qualitative.py.
No caption text: the caption is the author's. Post hoc, descriptive [PH].

Outputs (PDF for the paper, raster layers at 600 dpi; PNG previews at 300 dpi):
  paper_fig_qualitative.*     main text: 2 rows (best and worst patient of 23) x 4 columns, full text width
  paper_fig_appendix_good.*   appendix: the five 2026-11-18 slices of the best patient (LCTSC-Train-S1-012)
  paper_fig_appendix_bad.*    appendix: the five 2026-11-18 slices of the worst patient (LCTSC-Train-S1-009)
                              (both appendix figures share one image grid, sized so each is at most H_MAX tall)
  TRACE_paper.csv             every number printed, with its source
usage: SSL/.venv/bin/python -I -B make_paper_figure.py
Version of 2026-11-19.
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_qualitative as Q  # noqa: E402  (display rules only; its main() does not run on import)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

EXPECT = {"slices.npz": "aa5db9308050126caf120559fb198873aaacdc629de8445d9f548307d37a9b71",
          "slices.json": "7a1e55238b14ed9dc1a754615cdbb159a8718097834d38a6fcddc1b9b9e96e96",
          "make_qualitative.py": "2c8e7c0c2cc6a51af5b60951d18156ae524e63bab5505187e88903daf09ac7e0"}

# ---- Wording of the labels (no number lives here): rename freely and re-run. ----
COLS = ["Reference", "TotalSegmentator (TS)", "Fine-tuned", "Fixed / new errors"]
RANK = {"good": "best of {n} (heart-free)", "bad": "worst of {n} (heart-free)"}
RULE_TXT = {"heart": "slice with the most\nheart change", "hf_good": "slice with the largest\nesophagus-and-lung gain",
            "hf_bad": "slice with the largest\nesophagus-and-lung loss"}
GROUP_TXT = {"hf_good": "Slices with the largest esophagus-and-lung gain",
             "hf_bad": "Slices with the largest esophagus-and-lung loss",
             "heart": "Slices with the most heart change"}
FIXED_TXT = "fixed: TS wrong, fine-tuned right"
NEW_TXT = "new error: TS right, fine-tuned wrong"
OUTLINE_TXT = "reference outline (last column)"
NUM_TXT = "numbers: whole-scan Dice; L / R = left / right lung"

# ---- Rows (display choice made after viewing the 2026-11-18 figures [PH]). ----
MAIN_ROWS = [("good", "axial_heart"), ("bad", "axial")]
APPX_GROUPS = [("hf", ["axial", "coronal", "sagittal"]), ("heart", ["axial_heart", "coronal_heart"])]

W_IN = 6.0                       # MIDL \textwidth
H_MAX = 7.2                      # appendix figures: leaves about 1.3 in (7 caption lines) of the 8.5 in text height
GAP = 0.03                       # between panels, inches
F_HEAD, F_BAND, F_LEG, F_MARK, F_LETTER, F_GROUP = 8.0, 6.8, 6.8, 6.0, 8.0, 7.0
LINE = F_BAND * 1.3 / 72         # one band line, inches
LW_SEG = 0.6
INK, INK2 = Q.INK, Q.INK2
plt.rcParams.update({"font.family": ["Liberation Sans", "DejaVu Sans"], "pdf.fonttype": 42, "ps.fonttype": 42,
                     "font.size": 7, "axes.unicode_minus": True, "savefig.pad_inches": 0})
TRACE = []
ORGANS = ["esophagus", "heart", "lung_left", "lung_right"]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


class Panel(Q.Panel):
    """make_qualitative's panel (CT, overlays, change map unchanged) with paper-size marks."""

    def marks(self):
        a, b = Q.EDGE[Q.base(self.view)]
        for x, t, ha in ((0.02, a, "left"), (0.98, b, "right")):
            self.ax.text(x, 0.5, t, transform=self.ax.transAxes, ha=ha, va="center", fontsize=F_MARK, color="white",
                         bbox=dict(boxstyle="square,pad=0.12", fc="black", ec="none", alpha=0.55))

    def bar(self, mm=50.0):
        x0, x1 = self.ax.get_xlim()
        y1, y0 = self.ax.get_ylim()
        bx, by = x0 + 0.05 * (x1 - x0), y1 - 0.075 * (y1 - y0)
        self.ax.plot([bx, bx + mm], [by, by], color="white", lw=1.0, solid_capstyle="butt")
        self.ax.text(bx + mm + 0.025 * (x1 - x0), by, "5 cm", ha="left", va="center", fontsize=F_MARK, color="white",
                     bbox=dict(boxstyle="square,pad=0.12", fc="black", ec="none", alpha=0.55))

    def letter(self, s):
        self.ax.text(0.03, 0.97, s, transform=self.ax.transAxes, ha="left", va="top", fontsize=F_LETTER,
                     fontweight="bold", color="white",
                     bbox=dict(boxstyle="square,pad=0.15", fc="black", ec="none", alpha=0.55))


def ftext(fig, H, x, y, s, **kw):
    """Text at (x, y) inches from the figure's top-left corner."""
    return fig.text(x / W_IN, 1 - y / H, s, **kw)


def add_ax(fig, H, x, y, w, h):
    return fig.add_axes((x / W_IN, (H - y - h) / H, w / W_IN, h / H))


def crop(arr, meta, case, view):
    r = meta["cases"][case]
    return Q.crop_box(view, r["spacing_ras_mm"], [arr[f"{case}_{view}_{a}"] for a in ("ref", "F0", r["shown_draw"])])


def aspect(arr, meta, case, view):
    x0, x1, y0, y1 = crop(arr, meta, case, view)
    return (y1 - y0) / (x1 - x0)


def rule_key(meta, case, view):
    rule = meta["cases"][case]["views"][view]["rule"]
    return "heart" if rule == "heart" else f"hf_{case}"


def draw_row(fig, H, meta, arr, case, view, x, y, cw, h, tag, letter=None):
    r = meta["cases"][case]
    sp, shown = r["spacing_ras_mm"], r["shown_draw"]
    box = crop(arr, meta, case, view)
    P = [Panel(add_ax(fig, H, x + j * (cw + GAP), y, cw, h), arr[f"{case}_{view}_ct"], view, sp, box) for j in range(4)]
    P[0].seg(arr[f"{case}_{view}_ref"], lw=LW_SEG)
    P[1].seg(arr[f"{case}_{view}_F0"], lw=LW_SEG)
    P[2].seg(arr[f"{case}_{view}_{shown}"], lw=LW_SEG)
    c = P[3].change(arr[f"{case}_{view}_ref"], arr[f"{case}_{view}_F0"], arr[f"{case}_{view}_{shown}"])
    vi = r["views"][view]
    for k in ("fixed_hf_vox", "broken_hf_vox", "fixed_heart_vox", "broken_heart_vox"):
        assert c[k] == vi[k], (case, view, k, c[k], vi[k])       # same slice, same change map as 2026-11-18
    P[0].bar()
    P[0].marks()
    if letter:
        P[0].letter(letter)
    TRACE.append([tag, f"{case} {view}", "slice shown (not printed; for the record)",
                  f"{vi['index'] + 1} of {vi['n_slices']}", f"slices.json cases.{case}.views.{view}.index + 1"])


def dice_text(d, lines=2):
    if lines == 2:
        return f"esophagus {d[0]:.1f} · heart {d[1]:.1f}\nlungs L {d[2]:.1f} · R {d[3]:.1f}"
    return f"esophagus {d[0]:.1f}\nheart {d[1]:.1f}\nlungs L {d[2]:.1f} · R {d[3]:.1f}"


def trace_dice(tag, where, case, arm, d):
    for i, nm in enumerate(ORGANS):
        TRACE.append([tag, where, f"Dice {arm} {nm}", f"{d[i]:.1f}",
                      f"slices.json cases.{case}.pervol_dice.{arm}[{i}] (copy of look/pervol)"])


def id_text(fig, H, meta, case, view, xc, y, tag, n):
    r = meta["cases"][case]
    head = r["pid"] + (f", {Q.base(view)}" if view else "")
    ftext(fig, H, xc, y, head, ha="center", va="top", fontsize=F_BAND, color=INK)
    ftext(fig, H, xc, y + LINE, RANK[case].format(n=n), ha="center", va="top", fontsize=F_BAND, color=INK2)
    TRACE.append([tag, f"{case} label", "rank / number of patients", RANK[case].format(n=n),
                  f"slices.json cases.{case}.selection.rank_of_23 = {r['selection']['rank_of_23']}; "
                  f"len(selection.ranking) = {n}"])


def legend(fig, H, y, x0, x1):
    """Two lines from y (inches from the top): left-aligned with the image grid (x0 to x1) when they fit under it,
    otherwise centred under it (kept inside the figure)."""
    rend = fig.canvas.get_renderer()
    sw, sh, pad, sep = 0.10, 0.075, 0.04, 0.17
    lines = [[("fill", Q.ESO, "esophagus"), ("fill", Q.HEART, "heart"), ("fill", Q.LUNG, "lungs"),
              ("fill", Q.FIXED, FIXED_TXT), ("fill", Q.BROKEN, NEW_TXT)],
             [("outline", None, OUTLINE_TXT), (None, None, NUM_TXT)]]

    def width(label):
        t = fig.text(0, 0, label, fontsize=F_LEG)
        w = t.get_window_extent(rend).width / fig.dpi
        t.remove()
        return w

    widths = [sum((sw + pad if kind else 0.0) + width(label) for kind, _, label in items) + sep * (len(items) - 1)
              for items in lines]
    wmax = max(widths)
    assert wmax <= W_IN, f"legend is {wmax:.2f} in wide"
    x_start = x0 if wmax <= x1 - x0 else min(max((x0 + x1) / 2 - wmax / 2, 0.0), W_IN - wmax)
    for k, items in enumerate(lines):
        x, yc = x_start, y + 0.07 + k * 0.16
        for kind, color, label in items:
            if kind == "fill":
                fig.add_artist(Rectangle((x / W_IN, 1 - (yc + sh / 2) / H), sw / W_IN, sh / H,
                                         transform=fig.transFigure, facecolor=color, edgecolor="none"))
            elif kind == "outline":
                fig.add_artist(Rectangle((x / W_IN, 1 - (yc + sh / 2) / H), sw / W_IN, sh / H,
                                         transform=fig.transFigure, facecolor="black", edgecolor="none"))
                fig.add_artist(Line2D([(x + 0.015) / W_IN, (x + sw - 0.015) / W_IN], [1 - yc / H, 1 - yc / H],
                                      transform=fig.transFigure, color="white", lw=0.6))
            tx = x + (sw + pad if kind else 0.0)
            t = ftext(fig, H, tx, yc, label, ha="left", va="center", fontsize=F_LEG, color=INK)
            x = tx + t.get_window_extent(rend).width / fig.dpi + sep


def save(fig, out, name):
    fig.savefig(out / f"{name}.pdf", dpi=600, facecolor="white", metadata={"CreationDate": None})  # rebuilds identical
    fig.savefig(out / f"{name}.png", dpi=300, facecolor="white")
    plt.close(fig)


def main_figure(meta, arr, out, n):
    tag = "paper_fig_qualitative"
    cw = (W_IN - 3 * GAP) / 4
    rows = [(case, view, cw * aspect(arr, meta, case, view)) for case, view in MAIN_ROWS]
    HEAD, BAND, RGAP, LEG = 0.18, 0.04 + 2 * LINE + 0.03, 0.05, 0.36
    H = HEAD + sum(h + BAND for *_, h in rows) + RGAP * (len(rows) - 1) + LEG
    fig = plt.figure(figsize=(W_IN, H))
    for j, t in enumerate(COLS):
        ftext(fig, H, j * (cw + GAP) + cw / 2, 0.02, t, ha="center", va="top", fontsize=F_HEAD, fontweight="bold",
              color=INK)
    y = HEAD
    for i, (case, view, h) in enumerate(rows):
        draw_row(fig, H, meta, arr, case, view, 0.0, y, cw, h, tag, letter="ab"[i])
        r = meta["cases"][case]
        D, shown = r["pervol_dice"], r["shown_draw"]
        yb = y + h + 0.04
        id_text(fig, H, meta, case, view, cw / 2, yb, tag, n)
        for j, a in ((1, "F0"), (2, shown)):
            ftext(fig, H, j * (cw + GAP) + cw / 2, yb, dice_text(D[a]), ha="center", va="top", fontsize=F_BAND,
                  color=INK, linespacing=1.3)
            trace_dice(tag, f"row {'ab'[i]} under column {j + 1}", case, a, D[a])
        ftext(fig, H, 3 * (cw + GAP) + cw / 2, yb, RULE_TXT[rule_key(meta, case, view)], ha="center", va="top",
              fontsize=F_BAND, color=INK2, linespacing=1.3)
        y += h + BAND + RGAP
    legend(fig, H, y - RGAP, 0.0, W_IN)
    save(fig, out, tag)
    return W_IN, H


def appendix_layout(meta, arr):
    """One image grid for both appendix figures: the widest columns with which each stays within H_MAX."""
    L = dict(L_IN=0.16, HEAD=0.18, BAND=0.04 + 3 * LINE + 0.05, GH=0.18, LEG=0.36)
    cw_full = (W_IN - L["L_IN"] - 3 * GAP) / 4
    fit = []
    for case in ("good", "bad"):
        views = [v for _, vs in APPX_GROUPS for v in vs]
        fixed = L["HEAD"] + L["BAND"] + L["GH"] * len(APPX_GROUPS) + GAP * len(views) + L["LEG"]
        fit.append((H_MAX - fixed) / sum(aspect(arr, meta, case, v) for v in views))
    L["cw"] = min([cw_full] + fit)
    L["x0"] = (W_IN - (L["L_IN"] + 4 * L["cw"] + 3 * GAP)) / 2 + L["L_IN"]     # grid left edge; block centred
    return L


def appendix_figure(meta, arr, out, case, n, L):
    tag = f"paper_fig_appendix_{case}"
    cw, x0 = L["cw"], L["x0"]
    r = meta["cases"][case]
    D, shown = r["pervol_dice"], r["shown_draw"]
    heights = {v: cw * aspect(arr, meta, case, v) for _, vs in APPX_GROUPS for v in vs}
    H = L["HEAD"] + L["BAND"] + sum(L["GH"] + sum(heights[v] + GAP for v in vs) for _, vs in APPX_GROUPS) + L["LEG"]
    assert H <= H_MAX + 1e-9, (case, H)
    fig = plt.figure(figsize=(W_IN, H))
    xc = [x0 + j * (cw + GAP) + cw / 2 for j in range(4)]
    for j, t in enumerate(COLS):
        ftext(fig, H, xc[j], 0.02, t, ha="center", va="top", fontsize=F_HEAD, fontweight="bold", color=INK)
    yb = L["HEAD"] + 0.04
    id_text(fig, H, meta, case, None, xc[0], yb, tag, n)
    for j, a in ((1, "F0"), (2, shown)):
        ftext(fig, H, xc[j], yb, dice_text(D[a], lines=3), ha="center", va="top", fontsize=F_BAND, color=INK,
              linespacing=1.3)
        trace_dice(tag, f"under column {j + 1} header", case, a, D[a])
    y = L["HEAD"] + L["BAND"]
    for rule, vs in APPX_GROUPS:
        key = "heart" if rule == "heart" else f"hf_{case}"
        ftext(fig, H, x0, y + L["GH"] / 2, GROUP_TXT[key], ha="left", va="center", fontsize=F_GROUP,
              fontweight="bold", color=INK2)
        y += L["GH"]
        for v in vs:
            h = heights[v]
            assert rule_key(meta, case, v) == key, (case, v)
            ftext(fig, H, x0 - L["L_IN"] / 2 - 0.01, y + h / 2, Q.base(v).capitalize(), ha="center", va="center",
                  rotation=90, fontsize=F_BAND, color=INK2)
            draw_row(fig, H, meta, arr, case, v, x0, y, cw, h, tag)
            y += h + GAP
    legend(fig, H, y - GAP + 0.02, x0, x0 + 4 * cw + 3 * GAP)
    save(fig, out, tag)
    return W_IN, H


def main():
    for name, want in EXPECT.items():
        got = sha256(HERE / name)
        assert got == want, f"{name}: sha256 {got} differs from the recorded {want}"
    meta = json.load(open(HERE / "slices.json"))
    arr = dict(np.load(HERE / "slices.npz"))
    n = len(meta["selection"]["ranking"])
    assert n == 23, n
    for case, rank in (("good", 1), ("bad", n)):
        r = meta["cases"][case]
        assert r["selection"]["rank_of_23"] == rank, (case, r["selection"]["rank_of_23"])
        for a, d in r["pervol_dice"].items():
            assert np.allclose(d, r["recomputed"][a]["dice"], atol=1e-9), (case, a)
    sizes = {"paper_fig_qualitative": main_figure(meta, arr, HERE, n)}
    L = appendix_layout(meta, arr)
    for case in ("good", "bad"):
        sizes[f"paper_fig_appendix_{case}"] = appendix_figure(meta, arr, HERE, case, n, L)
    with open(HERE / "TRACE_paper.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["figure", "where", "quantity", "printed value", "source"])
        w.writerows(TRACE)
    for k, (w_, h_) in sizes.items():
        print(f"{k}: {w_:.2f} x {h_:.2f} in")
    print(f"appendix grid: columns {L['cw']:.3f} in, grid left edge {L['x0']:.3f} in")
    print(f"TRACE rows: {len(TRACE)}")


if __name__ == "__main__":
    main()
