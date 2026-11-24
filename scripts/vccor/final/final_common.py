"""vccor/final (2026-10-22): the remaining pre-registered experiments of the MIDL 2027 Special Track validation study
(blind-panel AC meta-review fix items 4, 5, 7, 8; outputs/results/vccor/blind_panel_2026-10-22/reviews/):

  C  word           KEY: zero-shot vs four-scan-aligned ranking of three foundation models on WORD, the
                    never-opened cohort (fix 8; "is it the model, or the labels?")                 plan_word.json
  D  reader         "convention or real error": blinded rating of the regions where alignment changed
                    the prediction + labelling time (fix 7)                                         plan_reader.json
  E  decomposition  definition-fixed systematic vs random error (volume + signed surface distance),
                    TS zero-shot vs aligned, computed post hoc on sets already opened              plan_decomposition.json
  A  transfer       cross-site transfer matrix of existing adapted models (fix 4)                 plan_transfer.json
  B  baselines      re-specified baselines: amendments of the peft / scratch_nm / amos_draws blocks (fix 5)  plan_baselines.json

PREPARED ONLY: no GPU job of this block has run; the scheduler patches are written and dry-run on COPIES of
scripts/queue_v8.py, never applied; outputs/logs/q8/decisions.json is never written by this block.

Scoring of every NEW look follows outputs/results/vccor/paper_tables_v2/analysis_conventions_v1.json (its sha256 is pinned in
every plan and re-checked by every scorer): native frames (MACT: NATIVE_BOX primary, == conventions sets.MACT-81; the other
cohorts: the raw grid, full field of view), presence = volume >= 1 mL on BOTH sides, FP organs score Dice 0 in the primary
(penalised) macro, ONE HD95 convention (TP: symmetric HD95 of surface-voxel centres; FN / FP: the field-of-view diagonal),
patient-cluster bootstrap CIs (10,000, seed 20261022) and, for multi-draw designs, the crossed two-level bootstrap.

This module: layout, logging, integrity guards (PREREG / AMENDMENT / CODE lines of outputs/logs/vccor/final_prereg.log), the
v1 metric rules (a verbatim copy of paper_tables_v2's v2_common.Scorer, so a later edit of that file cannot change these looks;
equality is asserted by smoke_final.py), statistics, label maps and reference readers. NO import-time side effects beyond
sys.path (safe before nnunetv2 / torch / TotalSegmentator).
"""
import hashlib
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path("/home/user/Documents/Medical Imagining SSL")
HERE = Path(__file__).resolve().parent
for _p in ("scripts/vccor/tsft_full", "scripts/vccor/xdata", "scripts/vccor/xdata/a3", "scripts/vccor/m2", "scripts/vccor/e2",
           "scripts/vccor/amos240"):
    if str(REPO / _p) not in sys.path:
        sys.path.append(str(REPO / _p))       # appended: those dirs also hold smoke / prereg scripts of their own blocks

VC = REPO / "outputs/results/vccor"
OUT = VC / "final"
LOGD = REPO / "outputs/logs/vccor"
PREREG_LOG = LOGD / "final_prereg.log"
DATA = REPO / "MagicNet/data"
VENV = REPO / ".venv/bin/python"
CONV = VC / "paper_tables_v2/analysis_conventions_v1.json"
CONV_LOG = LOGD / "paper_v2.log"
SMOKE = OUT / "smoke"

PLANS = {"transfer": "A", "baselines": "B", "word": "C", "reader": "D", "decomposition": "E"}
PLAN_DIR = {"transfer": OUT / "A_transfer", "baselines": OUT / "B_baselines", "word": OUT / "C_word", "reader": OUT / "D_reader",
            "decomposition": OUT / "E_decomposition"}

FLOOR_ML = 1.0                 # conventions presence_rule.threshold_ml
NBOOT = 10000                  # conventions statistics (cluster bootstrap)
SEED_CLUSTER = 20261022        # conventions statistics.single_draw_sets
SEED_TWO_LEVEL = 20261020      # conventions statistics.flare (m2_cluster)
NI_MARGIN = 1.0                # per-organ non-inferiority margin (Dice points), justified in each plan
NI_SENSITIVITY = (0.5, 2.0)
ORGAN_FLOOR = -1.00            # G1 organ floor (prereg_e1_mact_addendum_G1.json), == conventions statistics.organ_floor
MAX_WORKERS = 6


# ---------------------------------------------------------------------------------------------------- helpers
def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def sha256(p, n=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(n), b""):
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def arr_sha(a):
    import numpy as np
    a = np.ascontiguousarray(a)
    return hashlib.sha256(f"{a.dtype}|{a.shape}|".encode() + a.tobytes()).hexdigest()


def rel(p):
    p = Path(p)
    try:
        return str(p.resolve().relative_to(REPO))
    except ValueError:
        return str(p)


def _default(o):
    import numpy as np
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def clean_nan(o):
    if isinstance(o, float):
        return None if not math.isfinite(o) else o
    if isinstance(o, dict):
        return {str(k): clean_nan(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean_nan(v) for v in o]
    return o


WRITE_ROOTS = [OUT]            # this block writes only here (+ the declared new data dirs, see allow_write_root)


def allow_write_root(p):
    p = Path(p).resolve()
    if p not in WRITE_ROOTS:
        WRITE_ROOTS.append(p)


def _check_write(path):
    rp = Path(path).resolve()
    if not any(str(rp).startswith(str(r)) for r in WRITE_ROOTS) and not str(rp).startswith("/tmp"):
        raise SystemExit(f"[REFUSE] vccor/final writes only under {[rel(r) for r in WRITE_ROOTS]}: {path}")


def write_json(path, obj):
    path = Path(path)
    _check_write(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    with open(tmp, "w") as fh:
        json.dump(clean_nan(obj), fh, indent=1, default=_default)
    os.replace(tmp, path)
    return path


def write_text(path, txt):
    path = Path(path)
    _check_write(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    open(tmp, "w").write(txt)
    os.replace(tmp, path)
    return path


class Log:
    """Append-only text log (one line per message, time-stamped)."""

    def __init__(self, path, also_print=True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.also_print = also_print

    def __call__(self, msg):
        line = f"[{now()}] {msg}"
        if self.also_print:
            print(line, flush=True)
        with open(self.path, "a") as f:
            f.write(line + "\n")


def job_log(name):
    return Log(LOGD / f"final_{name}.log")


PLOG = Log(PREREG_LOG)


def be_nice(target=15):
    try:
        cur = os.nice(0)
        if cur < target:
            os.nice(target - cur)
    except OSError:
        pass


def cap_threads(n=1, cpu_only=True):
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = str(n)
    if cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


def nanmean(x, axis=None):
    import numpy as np
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmean(x, axis=axis)


def nanmedian(x, axis=None):
    import numpy as np
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmedian(x, axis=axis)


# ---------------------------------------------------------------------------------------------------- integrity
# final_prereg.log lines (whitespace-separated; parsed by position):
#   [date time] PREREG sha256 <hex> <repo-relative path>        a plan (OSF-ready JSON)
#   [date time] AMENDMENT sha256 <hex> <repo-relative path>     an amendment of an earlier block's pre-registration
#   [date time] DOC sha256 <hex> <path>                         the plain-English one-page version of a plan
#   [date time] CODE <hex> <path>  /  IMPORT <hex> <path>        code freeze (this block's code / frozen imports)
#   [date time] SMOKE sha256 <hex> <path>                       smoke results
#   [date time] PATCHDRYRUN sha256 <hex> <path>                 scheduler patch dry-run record (on copies)
def _log_lines():
    return [ln.split() for ln in open(PREREG_LOG)] if PREREG_LOG.is_file() else []


def logged_sha(kind, path):
    want, h = rel(path), None
    for p in _log_lines():
        if len(p) >= 6 and p[2] == kind and p[3] == "sha256" and p[5] == want:
            h = p[4]
    return h


def frozen_code_hashes():
    out = {}
    for p in _log_lines():
        if len(p) >= 5 and p[2] in ("CODE", "IMPORT") and len(p[3]) == 64:
            out[" ".join(p[4:])] = p[3]
    return out


def check_code_frozen(paths, log=print, strict=True):
    fz = frozen_code_hashes()
    bad = {rel(p): (sha256(REPO / rel(p))[:16], (fz.get(rel(p)) or "NOT-FROZEN")[:16]) for p in paths
           if fz.get(rel(p)) != sha256(REPO / rel(p))}
    if bad:
        msg = f"code differs from the CODEFREEZE hashes in {rel(PREREG_LOG)}: {bad}"
        if strict:
            raise SystemExit("[REFUSE] " + msg)
        log("[warn] " + msg)
    return not bad


def plan_path(name):
    assert name in PLANS, name
    return OUT / f"plan_{name}.json"


def check_plan(name):
    """The plan JSON must hash to its (last) PREREG line; returns the plan."""
    p = plan_path(name)
    if not p.is_file():
        raise SystemExit(f"[REFUSE] {rel(p)} missing")
    h, lg = sha256(p), logged_sha("PREREG", p)
    if h != lg:
        raise SystemExit(f"[REFUSE] {rel(p)} sha256 {h[:16]} != logged PREREG {str(lg)[:16]}")
    return json.load(open(p))


def conventions_sha_logged():
    """The last PREREG line of the conventions file in paper_v2.log (the other block's own registration)."""
    want, h = rel(CONV), None
    if CONV_LOG.is_file():
        for line in open(CONV_LOG):
            p = line.split()
            if len(p) >= 6 and p[2] == "PREREG" and p[3] == "sha256" and p[5] == want:
                h = p[4]
    return h


def check_conventions(pinned):
    """The conventions file must still hash to the value pinned in the plan (and to its own PREREG line)."""
    h = sha256(CONV)
    if h != pinned:
        raise SystemExit(f"[REFUSE] {rel(CONV)} sha256 {h[:16]} != the value pinned in the plan {str(pinned)[:16]}")
    lg = conventions_sha_logged()
    if lg != h:
        raise SystemExit(f"[REFUSE] {rel(CONV)} sha256 {h[:16]} != its PREREG line in {rel(CONV_LOG)} ({str(lg)[:16]})")
    return json.load(open(CONV))


FROZEN_IMPORTS = [REPO / "scripts/vccor/tsft_full/tsft_full_common.py", REPO / "scripts/vccor/tsft_full/ts_pipeline.py",
                  REPO / "scripts/vccor/xdata/xdata_common.py", REPO / "scripts/vccor/xdata/a3/a3_common.py",
                  REPO / "scripts/vccor/m2/m2_cluster.py", REPO / "scripts/vccor/e2/e2_common.py",
                  REPO / "scripts/vccor/amos240/a240_common.py", REPO / "scripts/ts_fullres/teacher_common.py",
                  REPO / "scripts/extras/ts_finetune/tsft_common.py"]


EXTRA_IMPORTS = [REPO / p for p in ("scripts/vccor/peft/lora.py", "scripts/vccor/vista/vista_engine.py",
                                      "scripts/vccor/vista/vista_common.py", "scripts/vccor/m2/m2_common.py",
                                      "scripts/vccor/m2/mrseg_pipeline.py", "scripts/vccor/m2/m2_train_lib.py")]


def code_files():
    return sorted(HERE.glob("*.py"))


def preflight(name, script, log=print, extra=()):
    """Every real (non-smoke) job: the plan is the registered one, and EVERY code file of this block (scripts/vccor/final/*.py,
    whatever the job imports) and every frozen import (FROZEN_IMPORTS + EXTRA_IMPORTS) hashes to its CODE / IMPORT line."""
    plan = check_plan(name)
    files = sorted(set(code_files()) | {Path(script).resolve()} | {Path(e).resolve() for e in extra})
    check_code_frozen(files + FROZEN_IMPORTS + EXTRA_IMPORTS, log)
    return plan


# ---------------------------------------------------------------------------------------------------- GT discipline
def open_marker(path, record):
    """Create the GT_OPENED marker of a look (O_EXCL: a look is made once); returns its path. Every scorer hashes every
    prediction map it will score into maps.json BEFORE calling this, and reads no reference before it."""
    path = Path(path)
    _check_write(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, json.dumps(dict(record, opened_at=now())).encode())
    os.close(fd)
    return path


FIRST_LOOKS = {"mact81": "the single MACT look of 2026-10-21", "amos240": "the AMOS-240 look of 2026-10-22"}


def repeat_look(set_key):
    """'DECLARED REPEAT LOOK n of <set>': n = 2 + the number of GT_OPENED markers of this block that already opened the set."""
    n = 0
    for p in OUT.rglob("GT_OPENED*"):
        try:
            if json.loads(p.read_text()).get("set_key") == set_key:
                n += 1
        except (OSError, ValueError):
            pass
    return f"DECLARED REPEAT LOOK {n + 2} of {set_key} (first: {FIRST_LOOKS[set_key]})"


def require_marker(path):
    if not Path(path).is_file():
        raise SystemExit(f"[REFUSE] reference read outside a look (no {rel(path)})")


# ---------------------------------------------------------------------------------------------------- v1 metric rules
# Verbatim copy (2026-10-22 16:08) of scripts/vccor/paper_v2/v2_common.py: joint_counts_chunked, bincount_chunked,
# _surface_points_lab, diag_mm, ref_record, Scorer (== analysis_conventions_v1 metrics). smoke_final.py asserts equal
# records against v2_common.Scorer on synthetic and real maps and logs v2_common's sha256 at that time.
def bincount_chunked(A, minlength, step=24):
    import numpy as np
    out = np.zeros(minlength, np.int64)
    for i in range(0, A.shape[0], step):
        out += np.bincount(A[i:i + step].ravel(), minlength=minlength)[:minlength]
    return out


def joint_counts_chunked(Y, G, k, step=24):
    import numpy as np
    n = k + 1
    H = np.zeros(n * n, np.int64)
    for i in range(0, Y.shape[0], step):
        idx = G[i:i + step].astype(np.int32) * n + Y[i:i + step].astype(np.int32)
        H += np.bincount(idx.ravel(), minlength=n * n)
    return H.reshape(n, n)


def diag_mm(shape, sp):
    import numpy as np
    return float(np.sqrt(((np.asarray(shape[:3], float) * np.asarray(sp, float)) ** 2).sum()))


def ref_record(G, k, vox_ml):
    import numpy as np
    g = bincount_chunked(G, 256)[1:k + 1].astype(np.int64)
    return {"vox": g.tolist(), "ml": [float(v * vox_ml) for v in g], "present": [bool(v * vox_ml >= FLOOR_ML) for v in g]}


def _surface_points_lab(L, o, sl):
    import numpy as np
    from scipy import ndimage
    lo = [max(s.start - 2, 0) for s in sl]
    hi = [min(s.stop + 2, n) for s, n in zip(sl, L.shape)]
    c = L[tuple(slice(a, b) for a, b in zip(lo, hi))] == o
    s = c & ~ndimage.binary_erosion(c)
    return np.argwhere(s) + np.asarray(lo)


class Scorer:
    """v1 scoring of many arms against ONE reference label map (values 0..k) on one grid (== v2_common.Scorer).
    Presence: volume >= 1 mL on both sides. R & P: Dice from voxels, HD95 / NSD 2 mm from surfaces (KD-tree, surface-voxel
    centres in mm); R & ~P (FN) or ~R & P (FP): Dice 0, NSD 0, HD95 = the field-of-view diagonal; ~R & ~P: undefined."""

    def __init__(self, G, k, sp, vox_ml=None, diag=None):
        import numpy as np
        from scipy import ndimage
        self.G, self.k, self.sp = G, int(k), np.asarray(sp, float)
        self.vox_ml = float(np.prod(self.sp)) / 1000.0 if vox_ml is None else float(vox_ml)
        self.diag = diag_mm(G.shape, self.sp) if diag is None else float(diag)
        self.ref = ref_record(G, self.k, self.vox_ml)
        self.slices = ndimage.find_objects(G, max_label=self.k)
        self._tree = {}

    def _ref_tree(self, o):
        from scipy.spatial import cKDTree
        if o not in self._tree:
            p = _surface_points_lab(self.G, o, self.slices[o - 1]) * self.sp
            self._tree[o] = (p, cKDTree(p))
        return self._tree[o]

    def score(self, Y, surface=True, tol=2.0):
        import numpy as np
        from scipy import ndimage
        from scipy.spatial import cKDTree
        k = self.k
        assert Y.shape == self.G.shape, (Y.shape, self.G.shape)
        H = joint_counts_chunked(Y, self.G, k)
        g, y = H.sum(1)[1:].astype(np.int64), H.sum(0)[1:].astype(np.int64)
        inter = np.array([H[o, o] for o in range(1, k + 1)], np.int64)
        gml, yml = g * self.vox_ml, y * self.vox_ml
        R, P = gml >= FLOOR_ML, yml >= FLOOR_ML
        dice, hd, nsd = np.full(k, np.nan), np.full(k, np.nan), np.full(k, np.nan)
        status = []
        ysl = ndimage.find_objects(Y, max_label=k) if surface else None
        for j in range(k):
            o = j + 1
            if R[j] and P[j]:
                dice[j] = 200.0 * inter[j] / (g[j] + y[j])
                status.append("TP")
                if surface:
                    pb, tb = self._ref_tree(o)
                    pa = _surface_points_lab(Y, o, ysl[j]) * self.sp
                    d_ab = tb.query(pa, k=1)[0]
                    d_ba = cKDTree(pa).query(pb, k=1)[0]
                    nsd[j] = 100.0 * float(((d_ab <= tol).sum() + (d_ba <= tol).sum()) / (len(d_ab) + len(d_ba)))
                    hd[j] = float(max(np.percentile(d_ab, 95), np.percentile(d_ba, 95)))
            elif R[j] or P[j]:
                dice[j], nsd[j], hd[j] = 0.0, 0.0, self.diag
                status.append("FN" if R[j] else "FP")
            else:
                status.append("TN")
        dice_vox = [float(200.0 * inter[j] / (g[j] + y[j])) if g[j] > 0 else float("nan") for j in range(k)]
        return {"vox": y.tolist(), "inter": inter.tolist(), "ml": [float(v) for v in yml], "present": P.tolist(),
                "status": status, "dice": dice.tolist(), "hd95": hd.tolist(), "nsd": nsd.tolist(), "surface": bool(surface),
                "dice_vox_old": dice_vox}

    def frame_record(self, extra=None):
        r = {"shape": list(self.G.shape), "spacing": [float(s) for s in self.sp], "vox_ml": self.vox_ml, "diag_mm": self.diag,
             "ref": self.ref, "arms": {}}
        if extra:
            r.update(extra)
        return r


def macro(rec_arm, ref, organs, penalised=True):
    """Per-scan macro Dice over `organs` (1-based ids): penalised = mean over the organs with R or P (FP organs score 0);
    reference-present = mean over the organs with R. nan if no organ qualifies."""
    vals = []
    for o in organs:
        j = o - 1
        r, p = ref["present"][j], rec_arm["present"][j]
        if (r or p) if penalised else r:
            vals.append(rec_arm["dice"][j])
    return float(sum(vals) / len(vals)) if vals else float("nan")


def organ_dice(rec_arm, ref, o):
    """Per-organ Dice for the per-organ contrasts: defined iff R (FN = 0); FP events are counted separately."""
    j = o - 1
    return float(rec_arm["dice"][j]) if ref["present"][j] else float("nan")


def volume_ml(rec_arm, o):
    j = o - 1
    return float(rec_arm["ml"][j]) if rec_arm["present"][j] else 0.0


# ---------------------------------------------------------------------------------------------------- statistics
def cluster_boot(values, clusters=None, nboot=NBOOT, seed=SEED_CLUSTER, level=0.95):
    """Percentile patient-cluster bootstrap of the scan-level mean (conventions statistics.single_draw_sets): clusters are
    resampled with replacement; the statistic is the mean over all scans of the resampled clusters. values {cid: float}
    (nan dropped); clusters {cid: key} (default: one cluster per scan). Returns dict(mean, ci, n, n_clusters, up, down)."""
    import numpy as np
    ids = [c for c in sorted(values) if values[c] is not None and np.isfinite(values[c])]
    if not ids:
        return {"mean": None, "ci": [None, None], "n": 0, "n_clusters": 0, "up": 0, "down": 0}
    key = {c: (clusters or {}).get(c, c) for c in ids}
    groups = sorted(set(key.values()))
    gi = {g: i for i, g in enumerate(groups)}
    sums, cnts = np.zeros(len(groups)), np.zeros(len(groups))
    for c in ids:
        sums[gi[key[c]]] += values[c]
        cnts[gi[key[c]]] += 1
    rng = np.random.default_rng(seed)
    G = len(groups)
    reps = np.empty(nboot)
    for b0 in range(0, nboot, 1000):
        nb = min(1000, nboot - b0)
        w = rng.multinomial(G, np.full(G, 1.0 / G), size=nb).astype(float)    # cluster multiplicities
        reps[b0:b0 + nb] = (w @ sums) / (w @ cnts)
    a = (1 - level) / 2
    v = np.array([values[c] for c in ids])
    return {"mean": float(v.mean()), "ci": [float(np.quantile(reps, a)), float(np.quantile(reps, 1 - a))], "n": len(ids),
            "n_clusters": G, "up": int((v > 0).sum()), "down": int((v < 0).sum()), "method": "patient-cluster percentile bootstrap",
            "nboot": nboot, "seed": seed}


def cluster_boot_diff(va, ca, vb, cb, nboot=NBOOT, seed=SEED_CLUSTER, level=0.95):
    """Difference of two scan-level means from DISJOINT groups (e.g. AMOS-240 site 'people' minus site 'center'): patient
    clusters resampled independently within each group (conventions item2.amos_strata.interaction_primary). va / vb
    {cid: value}; ca / cb {cid: cluster}. Returns dict(mean_a, mean_b, diff, ci, n_a, n_b) + a two-sided permutation p of
    the group label across clusters (10,000 permutations, same seed; descriptive)."""
    import numpy as np

    def prep(v, c):
        ids = [k for k in sorted(v) if v[k] is not None and np.isfinite(v[k])]
        g = sorted({c.get(k, k) for k in ids})
        gi = {x: i for i, x in enumerate(g)}
        s, n = np.zeros(len(g)), np.zeros(len(g))
        for k in ids:
            s[gi[c.get(k, k)]] += v[k]
            n[gi[c.get(k, k)]] += 1
        return s, n

    sa, na = prep(va, ca or {})
    sb, nb = prep(vb, cb or {})
    if not len(sa) or not len(sb):
        return {"diff": None, "ci": [None, None]}
    rng = np.random.default_rng(seed)
    reps = np.empty(nboot)
    for b in range(nboot):
        wa = rng.multinomial(len(sa), np.full(len(sa), 1.0 / len(sa)))
        wb = rng.multinomial(len(sb), np.full(len(sb), 1.0 / len(sb)))
        reps[b] = (wa @ sa) / (wa @ na) - (wb @ sb) / (wb @ nb)
    obs = sa.sum() / na.sum() - sb.sum() / nb.sum()
    S, Nn = np.r_[sa, sb], np.r_[na, nb]
    k = len(sa)
    prng = np.random.default_rng(seed)
    perm = np.empty(nboot)
    for b in range(nboot):
        idx = prng.permutation(len(S))
        perm[b] = S[idx[:k]].sum() / Nn[idx[:k]].sum() - S[idx[k:]].sum() / Nn[idx[k:]].sum()
    a = (1 - level) / 2
    return {"mean_a": float(sa.sum() / na.sum()), "mean_b": float(sb.sum() / nb.sum()), "diff": float(obs),
            "ci": [float(np.quantile(reps, a)), float(np.quantile(reps, 1 - a))], "n_a": int(na.sum()), "n_b": int(nb.sum()),
            "n_clusters_a": len(sa), "n_clusters_b": len(sb),
            "perm_p_two_sided": float((np.abs(perm) >= abs(obs) - 1e-12).mean()), "nboot": nboot, "seed": seed}


def t_ci(values, level=0.95):
    import numpy as np
    from scipy import stats
    v = np.array([x for x in values if x is not None and np.isfinite(x)], float)
    if len(v) < 2:
        return {"mean": float(v.mean()) if len(v) else None, "ci": [None, None], "n": int(len(v))}
    se = v.std(ddof=1) / np.sqrt(len(v))
    q = stats.t.ppf(1 - (1 - level) / 2, len(v) - 1)
    return {"mean": float(v.mean()), "ci": [float(v.mean() - q * se), float(v.mean() + q * se)], "n": int(len(v))}


def two_level(per_draw, nboot=NBOOT, seed=SEED_TWO_LEVEL):
    """Crossed two-level (scan x draw) bootstrap of M = mean of the draw means (frozen m2_cluster; conventions
    statistics.flare). per_draw {draw: {cid: value}}. Returns dict(M, ci, scan_values, up, n_scans, draws)."""
    import numpy as np
    import m2_cluster as MC
    draws, scans, D = MC.matrix(per_draw)
    if not draws or not scans:
        return {"M": None, "ci": [None, None], "n_scans": 0, "draws": draws}
    m = [float(np.nanmean(D[j])) for j in range(len(draws))]
    ci, _ = MC.boot2(D, nboot, seed) if len(draws) > 1 else MC.boot2(D, nboot, seed, resample_draws=False)
    dv, _n = MC.scan_values(D)
    return {"M": float(np.mean(m)), "ci": [None if x is None else float(x) for x in ci], "draw_means": dict(zip(draws, m)),
            "scan_values": dict(zip(scans, [float(x) for x in dv])), "up": int((dv > 0).sum()), "down": int((dv < 0).sum()),
            "n_scans": len(scans), "draws": draws, "method": "crossed two-level percentile bootstrap (m2_cluster.boot2)"}


def ni_class(ci, margin=NI_MARGIN):
    """Per-organ non-inferiority reading of a two-sided 95% CI of the mean paired difference (arm - comparator):
    NON-INFERIOR iff lower bound > -margin (one-sided alpha 0.025); INFERIOR iff upper bound < -margin; else INCONCLUSIVE."""
    lo, hi = ci
    if lo is None or hi is None:
        return "NOT EVALUABLE"
    if lo > -margin:
        return "NON-INFERIOR"
    if hi < -margin:
        return "INFERIOR"
    return "INCONCLUSIVE"


def harm_class(ci):
    """conventions statistics.harm_classes: HARMFUL iff upper < 0; BENEFICIAL iff lower > 0; NEUTRAL otherwise."""
    lo, hi = ci
    if lo is None or hi is None:
        return "NOT EVALUABLE"
    return "HARMFUL" if hi < 0 else ("BENEFICIAL" if lo > 0 else "NEUTRAL")


def p1_rule(values, min_mean=0.30, min_frac=2.0 / 3.0, boot=None):
    """The MACT P1 / AMOS-240 rule (prereg_e1_mact.json P1, prereg_amos240.json RULE): PASS iff M >= +0.30 Dice points AND
    the 95% scan-level t-CI lower bound > 0 AND positive paired differences on >= ceil(2/3 n) scans. boot (optional) = an
    additional bootstrap CI whose lower bound must also be > 0 (the multi-draw form, plan C)."""
    import numpy as np
    v = [x for x in values if x is not None and np.isfinite(x)]
    t = t_ci(v)
    need = int(math.ceil(min_frac * len(v)))
    up = int(sum(x > 0 for x in v))
    ok = (t["mean"] is not None and t["mean"] >= min_mean and t["ci"][0] is not None and t["ci"][0] > 0 and up >= need)
    if boot is not None:
        ok = ok and boot[0] is not None and boot[0] > 0
    return {"pass": bool(ok), "mean": t["mean"], "t_ci": t["ci"], "up": up, "need_up": need, "n": len(v),
            "boot_ci": boot, "rule": f"M >= +{min_mean:.2f} AND t-CI lower > 0 AND up >= ceil({min_frac:.3f} n)"
                                      + (" AND bootstrap-CI lower > 0" if boot is not None else "")}


# ---------------------------------------------------------------------------------------------------- label maps
FLARE13 = ['liver', 'r.kid', 'spleen', 'panc', 'aorta', 'ivc', 'r.adr', 'l.adr', 'gall', 'esoph', 'stom', 'duod', 'l.kid']
MACT8 = ["spleen", "l.kid", "gall", "esoph", "liver", "stom", "panc", "duod"]
BTCV13 = ["spleen", "r.kid", "l.kid", "gall", "esoph", "liver", "stom", "aorta", "ivc", "portal_splenic_vein", "panc", "r.adr",
          "l.adr"]
# TotalSegmentator 'total' class names per dataset id (kidney cysts -> same-side kidney, the project's v2 rule)
TS_NAMES = {
    "flare": {1: ["liver"], 2: ["kidney_right", "kidney_cyst_right"], 3: ["spleen"], 4: ["pancreas"], 5: ["aorta"],
              6: ["inferior_vena_cava"], 7: ["adrenal_gland_right"], 8: ["adrenal_gland_left"], 9: ["gallbladder"],
              10: ["esophagus"], 11: ["stomach"], 12: ["duodenum"], 13: ["kidney_left", "kidney_cyst_left"]},
    "mact": {1: ["spleen"], 2: ["kidney_left", "kidney_cyst_left"], 3: ["gallbladder"], 4: ["esophagus"], 5: ["liver"],
             6: ["stomach"], 7: ["pancreas"], 8: ["duodenum"]},
    "btcv": {1: ["spleen"], 2: ["kidney_right", "kidney_cyst_right"], 3: ["kidney_left", "kidney_cyst_left"], 4: ["gallbladder"],
             5: ["esophagus"], 6: ["liver"], 7: ["stomach"], 8: ["aorta"], 9: ["inferior_vena_cava"],
             10: ["portal_vein_and_splenic_vein"], 11: ["pancreas"], 12: ["adrenal_gland_right"], 13: ["adrenal_gland_left"]},
}
TS_NAMES["amos"] = TS_NAMES["flare"]          # AMOS is scored in FLARE ids (xdata AMOS_TO_FLARE; bladder / prostate -> 0)
ORGANS = {"flare": FLARE13, "amos": FLARE13, "mact": MACT8, "btcv": BTCV13}
# organ name (the project's short names) -> id per label space
ORGAN_ID = {ds: {n: i + 1 for i, n in enumerate(v)} for ds, v in ORGANS.items()}


def n_org(ds):
    return len(ORGANS[ds])


def ts_lut(ds):
    """uint8 LUT TotalSegmentator 'total' id -> dataset id (by class name). flare / amos / mact == xdata_common.total_lut;
    btcv == e2_common.BTCV_TS (asserted in smoke_final.py)."""
    import numpy as np
    from totalsegmentator.map_to_binary import class_map
    inv = {v: k for k, v in class_map["total"].items()}
    lut = np.zeros(256, np.uint8)
    for o, names in TS_NAMES[ds].items():
        for n in names:
            lut[inv[n]] = o
    return lut


def identity_lut(k):
    import numpy as np
    lut = np.zeros(256, np.uint8)
    lut[1:k + 1] = np.arange(1, k + 1)
    return lut


# ---------------------------------------------------------------------------------------------------- references
def flare_ref(cid, marker):
    """FLARE22 raw label (13 FLARE ids) on the raw grid + spacing (nibabel order). Only inside a look."""
    import nibabel as nib
    import numpy as np
    import xdata_common as X
    require_marker(marker)
    lbl = nib.load(str(X.FLARE_RAW_LBL).format(cid=cid))
    G = np.asarray(lbl.dataobj).astype(np.uint8)
    assert G.max() <= 13, cid
    return G, tuple(float(s) for s in lbl.header.get_zooms()[:3])


def amos_ref(cid, marker):
    """AMOS raw label (labelsTr / labelsVa) mapped to FLARE ids (xdata AMOS_TO_FLARE; 14 bladder / 15 prostate-uterus -> 0)."""
    import nibabel as nib
    import numpy as np
    import a240_common as A240
    import xdata_common as X
    require_marker(marker)
    allc = A240.all_labelled_ct()
    lbl = nib.load(str(allc[cid][1]))
    raw = np.asarray(lbl.dataobj)
    assert raw.min() >= 0 and raw.max() <= 15, cid
    G = np.zeros(raw.shape, np.uint8)
    for a, f in X.AMOS_TO_FLARE.items():
        G[raw == a] = f
    return G, tuple(float(s) for s in lbl.header.get_zooms()[:3])


def btcv_ref(mid, marker):
    """BTCV's own Training label (13 classes; 10 = portal + splenic vein) of a BTCV-28 image keyed by its MACT id."""
    import nibabel as nib
    import numpy as np
    import e2_common as E2
    require_marker(marker)
    oid = E2.btcv28()[mid]
    lbl = nib.load(str(E2.BTCV_LBL).format(oid=oid))
    G = np.rint(np.asarray(lbl.dataobj)).astype(np.uint8)
    assert G.max() <= 13, mid
    return G, tuple(float(s) for s in lbl.header.get_zooms()[:3])


def mact_native(mid, marker):
    """MACT NATIVE frames (== conventions sets.MACT-81.native_frames; == score_a3.py): the Zenodo label on the raw grid
    reoriented to LPS; returns (fr, Z_lps, box slices, LPS spacing) or None when the scan is UNVERIFIED (A3.9)."""
    import nibabel as nib
    import a3_common as A3
    import xdata_common as X
    require_marker(marker)
    img = nib.load(str(X.raw_img("mact", mid)))
    fr = A3.frame(mid, img)
    try:
        Z = A3.zenodo_raw_mact(mid, img, marker=marker)
    except A3.Unverified:
        return None
    return fr, A3.to_lps(Z, fr), A3.box_slices(fr), tuple(float(x) for x in fr["spacing"]), img


def mact_pred_lps(seg_raw, fr):
    import a3_common as A3
    return A3.to_lps(seg_raw, fr)


# ---------------------------------------------------------------------------------------------------- AMOS patients
def amos_patient_clusters():
    """{amos id: cluster key} over all 300 labelled AMOS CT (conventions item2.deduplication.amos_key): key = (Patient's
    Birth Date, Patient's Sex) from AMOS's metadata CSV (sha256-pinned in a240_common); valid iff the birth date has 8
    digits, is not all zero, and sex is M / F; scans sharing a valid key form one cluster (connected components);
    otherwise the scan is its own cluster. Cluster key = the lowest amos id of the cluster."""
    import a240_common as A240
    m = A240.meta()
    allc = sorted(A240.all_labelled_ct())
    parent = {c: c for c in allc}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    by_key = {}
    for c in allc:
        r = m[c]
        bd, sx = str(r.get("Patient's Birth Date", "")).strip(), str(r.get("Patient's Sex", "")).strip()
        if len(bd) == 8 and bd.isdigit() and set(bd) != {"0"} and sx in ("M", "F"):
            by_key.setdefault((bd, sx), []).append(c)
    for cs in by_key.values():
        for c in cs[1:]:
            a, b = find(cs[0]), find(c)
            if a != b:
                parent[max(a, b)] = min(a, b)
    comp = {}
    for c in allc:
        comp.setdefault(find(c), []).append(c)
    return {c: min(comp[find(c)]) for c in allc}


def amos_site(cid):
    import a240_common as A240
    r = A240.meta()[cid]
    return r["Site"], r["Manufacturer"], r["Manufacturer's Model Name"]
