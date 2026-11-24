"""Shared helpers for the MIDL display set of 2026-11-20 (storyline B, adopted on 2026-11-20 at 11:10 KST;
the go-ahead for the displays).

Reads stored records only: the results.json files of the five registered looks (each sha256-checked against its
RESULTS line in the registration log), their per-scan pervol records, the post hoc tables of
docs/midl_posthoc_2026-11-17, the compiled tables of docs/midl_tables_2026-11-16 and, for fig_coronal.py only, the
cached slices of docs/midl_qualitative_2026-11-18 (sha256-checked). No image, label or prediction map is opened.
Figures and fact tables for planning: no paper text and no captions (the author writes those).
Version of 2026-11-20.
"""
import csv
import hashlib
import json
import re
from pathlib import Path

SSL = Path("/home/user/Documents/Medical Imagining SSL")
FINAL = SSL / "outputs/results/vccor/final"
PH = SSL / "docs/midl_posthoc_2026-11-17"
TAB = SSL / "docs/midl_tables_2026-11-16"
QUAL = SSL / "docs/midl_qualitative_2026-11-18"
LOG = SSL / "outputs/logs/vccor/final_prereg.log"
OUT = Path(__file__).resolve().parent

LOOK_DIRS = {"WORD": "C_word", "FLARE-T50": "I_flaret50", "LCTSC": "K_lctsc", "WORD-N": "I_wordn",
             "LCTSC-T23": "K_lctsc_t23"}
SOURCES = {}
TRACE = []


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def rel(p):
    return str(Path(p).resolve().relative_to(SSL))


def logged_hash(kind, p):
    """sha256 of the last '[t] KIND sha256 <h> <rel path>' line of the registration log for this file."""
    rx = re.compile(r"^\[[^\]]+\] %s sha256 ([0-9a-f]{64}) (\S+)" % kind)
    r = rel(p)
    hits = []
    with open(LOG, encoding="utf-8") as f:
        for ln in f:
            m = rx.match(ln)
            if m and m.group(2) == r:
                hits.append(m.group(1))
    assert hits, f"no {kind} line for {r}"
    return hits[-1]


_RES = {}


def results(look):
    if look not in _RES:
        p = FINAL / LOOK_DIRS[look] / "look/results.json"
        h = sha256(p)
        assert h == logged_hash("RESULTS", p), f"{look}: results.json differs from its RESULTS line"
        SOURCES[rel(p)] = h
        _RES[look] = json.load(open(p))
    return _RES[look]


def pervol(look):
    d = FINAL / LOOK_DIRS[look] / "look/pervol"
    hs = hashlib.sha256()
    out = {}
    for p in sorted(d.glob("*.json")):
        hs.update(f"{p.name} {sha256(p)}\n".encode())
        out[p.stem] = json.load(open(p))
    SOURCES[rel(d) + "/*.json (sha256 of the sorted 'name sha256' list)"] = hs.hexdigest()
    return out


def read_csv(p):
    SOURCES[rel(p)] = sha256(p)
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(p):
    SOURCES[rel(p)] = sha256(p)
    return json.load(open(p))


def walk_numbers(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from walk_numbers(v, f"{path}.{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from walk_numbers(v, f"{path}[{i}]")
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        yield path, float(o)


def find_value(o, value, tol):
    """Paths in a record whose number equals value within tol (used to tie a printed value to its key)."""
    return [p for p, v in walk_numbers(o) if abs(v - value) <= tol]


def trace(display, where, quantity, printed, source):
    TRACE.append([display, where, quantity, printed, source])


def write_trace(name):
    with open(OUT / f"TRACE_{name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["display", "where", "quantity", "printed value", "source"])
        w.writerows(TRACE)
    with open(OUT / f"sources_{name}.json", "w", encoding="utf-8") as f:
        json.dump(dict(sorted(SOURCES.items())), f, indent=1)


def sg(x, d=2, minus="-"):
    return f"{x:+.{d}f}".replace("-", minus)


def ci_txt(c, d=2, minus="-"):
    return f"[{sg(c[0], d, minus)}, {sg(c[1], d, minus)}]"


# ---- figure style (MIDL text width 6.0 in, body 10.95 pt; measured from the official template PDF)
INK, INK2, RULE, BAND = "#1e2328", "#5b6169", "#c9cdd2", "#f4f1ea"
GREEN, BROWN, GREY, AMBER, RED = "#2f7d4a", "#8a4b14", "#7d838c", "#c98a00", "#b03a2e"
W_IN = 6.0
MINUS = "−"


def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import rcParams
    rcParams.update({"font.family": "Liberation Sans", "font.size": 7.5, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.labelsize": 7.5,
                     "savefig.facecolor": "white", "figure.facecolor": "white", "axes.unicode_minus": True})
    import matplotlib.pyplot as plt
    return plt


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", dpi=600, metadata={"CreationDate": None})
    fig.savefig(OUT / f"{name}.png", dpi=300)
