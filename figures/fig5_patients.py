"""Figure 5 (pick 1 of 2026-11-20): who improved.

One dot per scan or patient: the change in the look's macro Dice of the four-scan update against unmodified
TotalSegmentator, averaged over the three draws (the registered per-scan quantity of each primary or secondary rule).
Filled green: above 0; hollow amber: 0 or below. The black tick is the look's registered mean. At the right: the
registered count of scans or patients above 0 and the count the registered rule needed.
Per-scan values are recomputed from the per-scan records (pervol: stored per-structure Dice of every arm), with the
registered macro rule (a structure counts when the reference or the prediction contains it; Dice 0 when only one does).
Checks: each look's mean equals the registered M (to 1e-9), each count equals the registered scans_up, and WORD-N's
values equal its stored scan_values (to 1e-12). No image, label or prediction map is opened.
No caption: the author writes it. Facts in TRACE_fig5_patients.csv.
usage: SSL/.venv/bin/python -I -B fig5_patients.py
Version of 2026-11-20.
"""
import statistics as S
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import disp_common as C  # noqa: E402

plt = C.setup_mpl()
from matplotlib.lines import Line2D  # noqa: E402

DISP = "fig5_patients"
RES = {k: C.results(k) for k in C.LOOK_DIRS}
OC = {"WORD": RES["WORD"]["P1_organ_costs"]["per_organ"],
      "WORD-N": RES["WORD-N"]["A1_readings_fresh_descriptive"]["P1_organ_costs"]["per_organ"],
      "FLARE-T50": RES["FLARE-T50"]["organ_costs"]["per_organ"],
      "LCTSC": RES["LCTSC"]["organ_costs"]["per_organ"], "LCTSC-T23": RES["LCTSC-T23"]["organ_costs"]["per_organ"]}
DRAWS = {"WORD": ["F1b_d1", "F1b_d2", "F1b_d3"], "WORD-N": ["F1b_d1", "F1b_d2", "F1b_d3"],
         "FLARE-T50": ["FULL_s1337", "FULL_s2025", "FULL_s3407"], "LCTSC": ["F1b_S1", "F1b_S2", "F1b_S3"],
         "LCTSC-T23": ["F1b_S1", "F1b_S2", "F1b_S3"]}
AMBER_D = "#a86f00"   # validated against GREEN (dataviz validator, 2026-11-20); hollow marker as second encoding


def nat(rec):
    return rec["NATIVE"] if "NATIVE" in rec else rec["frames"]["NATIVE"]


def positions(look, recs):
    """List position of each organ-cost structure: the one whose draw-mean Dice change equals the registered M."""
    L = len(recs[0]["ref"]["vox"])
    idx = {}
    for name, b in OC[look].items():
        cands = []
        for i in range(L):
            def md(a):
                v = [n["arms"][a]["dice"][i] for n in recs if n["ref"]["present"][i]]
                return S.fmean(v) if v and None not in v else None
            f0 = md("F0")
            if f0 is not None and abs(S.fmean(md(a) - f0 for a in DRAWS[look]) - b["M"]) < 1e-9:
                cands.append(i)
        assert len(cands) == 1, (look, name, cands)
        idx[name] = cands[0]
    return idx


def per_scan(look, structures):
    pv = C.pervol(look)
    recs = {k: nat(v) for k, v in pv.items()}
    idx = positions(look, list(recs.values()))
    cols = [idx[s] for s in (structures or OC[look])]

    def macro(n, a):
        vals = []
        for i in cols:
            if n["ref"]["present"][i] or n["arms"][a]["present"][i]:
                d = n["arms"][a]["dice"][i]
                vals.append(0.0 if d is None else d)
        return S.fmean(vals)
    return {sid: S.fmean(macro(n, a) - macro(n, "F0") for a in DRAWS[look]) for sid, n in recs.items()}


ROWS = [  # name, line 2, look, registered block path, structures (None = the look's organ-cost set), role
    ("WORD test", "29 scans · 12 organs", "WORD", ("P1_gain",), None, "primary", "scans"),
    ("WORD-N", "20 scans · same centre as WORD", "WORD-N", ("P1_fresh", "p1"), None, "secondary", "scans"),
    ("FLARE-T50", "50 scans · 13 organs", "FLARE-T50", ("R1",), None, "primary", "scans"),
    ("LCTSC test", "24 patients · 4 structures", "LCTSC", ("P1",), None, "primary", "patients"),
    ("LCTSC-T23", "23 patients · heart excluded", "LCTSC-T23", ("P1_HF",), ("esophagus", "lung_left", "lung_right"),
     "primary", "patients"),
    ("LCTSC-T23, with heart", "same 23 patients · 4 structures", "LCTSC-T23", ("S_4S",), None, "secondary",
     "patients"),
]
DATA = []
for name, l2, look, path, structs, role, unit in ROWS:
    reg = RES[look]
    for k in path:
        reg = reg[k]
    ch = per_scan(look, structs)
    m, up = S.fmean(ch.values()), sum(v > 0 for v in ch.values())
    assert abs(m - reg["M"]) < 1e-9, (name, m, reg["M"])
    assert up == reg["scans_up"] and len(ch) == reg["n_scans"], (name, up, reg["scans_up"], len(ch))
    if "scan_values" in reg:
        assert max(abs(ch[s] - v) for s, v in reg["scan_values"].items()) < 1e-12, name
    DATA.append((name, l2, ch, reg, role, unit))

XMIN, XMAX = -5.0, 9.0
PITCH, TOP, BOT = 0.43, 0.46, 0.42
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
ticks = list(range(-4, 9, 2))
ax.set_xticks(ticks)
ax.set_xticklabels([C.sg(t, 0, C.MINUS) if t else "0" for t in ticks])
for t in ticks:
    ax.axvline(t, color=C.RULE, lw=0.45, zorder=0)
ax.axvline(0, color=C.INK2, lw=0.9, zorder=1)
ax.set_xlabel("Change in mean Dice per scan or patient (points): updated minus unmodified, mean of three draws",
              fontsize=7)
for yb in (1.5, 2.5):
    ax.axhline(yb, color=C.RULE, lw=0.7, zorder=0)

MS = 3.6                                      # marker size (pt)
RX = (MS / 72) / (W_AX / (XMAX - XMIN)) * 1.05   # marker diameter in x data units
RY = (MS / 72) / PITCH * 1.05                     # marker diameter in row units


LIM = 0.42                                        # largest vertical offset inside a row (row units)


def swarm(xs):
    """Deterministic 1-D beeswarm: each point takes the smallest vertical offset that does not overlap; a row whose
    swarm would leave its band is compressed vertically (all offsets scaled by one factor, order kept)."""
    placed, out = [], {}
    for k, x in sorted(xs.items(), key=lambda kv: (kv[1], kv[0])):
        off = 0.0
        for j in range(0, 60):
            off = (j + 1) // 2 * RY * (1 if j % 2 else -1) if j else 0.0
            if all(((x - px) / RX) ** 2 + ((off - py) / RY) ** 2 >= 1 for px, py in placed):
                break
        placed.append((x, off))
        out[k] = off
    m = max(abs(v) for v in out.values())
    f = min(1.0, LIM / m) if m else 1.0
    return {k: v * f for k, v in out.items()}, f


def fyrow(i):
    return (BOT + PITCH * (len(ROWS) - 0.5 - i)) / H


for i, (name, l2, ch, reg, role, unit) in enumerate(DATA):
    off, squeeze = swarm(ch)
    C.trace(DISP, name, "vertical compression of the dot swarm (display only)", f"{squeeze:.2f}", "fig5_patients.py")
    for sid, x in ch.items():
        up = x > 0
        ax.plot([x], [i + off[sid]], "o", ms=MS, mfc=C.GREEN if up else "white", mec="white" if up else AMBER_D,
                mew=0.35 if up else 0.9, zorder=3)
        C.trace(DISP, f"{name} / {sid}", "change (draw mean), Dice points", f"{x:+.3f}",
                "pervol NATIVE arms dice; registered macro rule")
    ax.plot([reg["M"], reg["M"]], [i - 0.33, i + 0.33], color=C.INK, lw=1.4, zorder=4, solid_capstyle="butt")
    fig.text(0.02 / C.W_IN, fyrow(i) + 0.055 / H, name, ha="left", va="center", fontsize=7.4, fontweight="bold",
             color=C.INK)
    fig.text(0.02 / C.W_IN, fyrow(i) - 0.095 / H, l2, ha="left", va="center", fontsize=6.3, color=C.INK2)
    n = reg["n_scans"]
    fig.text(X_VAL / C.W_IN, fyrow(i) + 0.055 / H, f"{reg['scans_up']} of {n} {unit}", ha="left", va="center",
             fontsize=7.0, color=C.INK)
    fig.text(X_VAL / C.W_IN, fyrow(i) - 0.095 / H, f"rule ≥ {reg['need_up']} · {role}", ha="left",
             va="center", fontsize=6.3, color=C.INK2)
    lo, hi = min(ch.values()), max(ch.values())
    C.trace(DISP, name, "above 0 / n / needed by the rule; mean; lowest; highest",
            f"{reg['scans_up']} / {n} / {reg['need_up']}; {reg['M']:+.2f}; {lo:+.2f}; {hi:+.2f}",
            "results.json scans_up, n_scans, need_up, M (asserted equal to the recomputed values)")

fig.text(X_VAL / C.W_IN, 1 - (TOP - 0.10) / H, "Above 0", ha="left", va="center", fontsize=7.2, fontweight="bold",
         color=C.INK)
hand = [Line2D([], [], ls="none", marker="o", ms=MS + 0.6, mfc=C.GREEN, mec=C.GREEN),
        Line2D([], [], ls="none", marker="o", ms=MS + 0.6, mfc="white", mec=AMBER_D, mew=1.0),
        Line2D([], [], color=C.INK, lw=1.4)]
lab = ["scan or patient above 0 (improved)", "0 or below", "mean of the look (the registered estimate)"]
fig.legend(hand, lab, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=3, fontsize=6.6, frameon=False,
           handletextpad=0.4, columnspacing=1.4, borderaxespad=0.05)
C.save(fig, DISP)
C.write_trace(DISP)
print(f"{DISP}: {C.W_IN:.2f} x {H:.2f} in; " + "; ".join(f"{d[0]} {d[3]['scans_up']}/{d[3]['n_scans']}" for d in DATA))
