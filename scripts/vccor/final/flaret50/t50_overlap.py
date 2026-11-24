"""vccor/final FLARE-T50 K1b: near-duplicate check of the 50 FLARE-T50 images against every project image (IMAGES ONLY;
before registration; disclosed). The rule is E0's registered overlap rule, run with E0's own frozen code
(scripts/vccor/e0/overlap.py: LPS -> window [-125, 275] HU -> Gaussian anti-alias -> 8 mm (stage 1, all pairs) / 4 mm
(stage 2, every volume's top-2 candidates, plus every pair whose raw header fingerprint matches) -> masked NCC over all
integer translations, overlap >= 40 % of the smaller volume); near-duplicate iff NCC4 >= 0.90. Header fingerprints (raw
shape + spacing) are reported and fed into stage 2.

Project images (every image set the project has used, code review B1):
  FLARE22 training 50 (raw); MACT 90 (MACT_h5 'image' only); BTCV Training 30 + Testing 20 (raw, read-only from the
  external drive); AMOS 300 = the 60 of amos.txt (E0) + the 240 of AMOS-240 (scripts/vccor/amos240/a240_common.py:
  all labelled CT ids minus amos.txt; file names only, no label file is opened); every WORD image already extracted to
  word_common.IMAGES (nothing is extracted from the WORD archive).
Positive control (detector check): BTCV vs MACT must reproduce E0's 47 near-duplicate pairs
(outputs/results/vccor/e0/overlap/pairs.json); otherwise K1b is K1B_FAILED_CONTROL (no exclusion list is written).

Decision (AC.md GO-1): a FLARE-T50 scan is EXCLUDED iff its NCC4 >= 0.90 with any project image, or its file failed to load
(images_record.json). Kill test K1b (AC.md:155): more than 10 of the 50 are near-duplicates -> STOP (K1B_STOP; no
registration). Output: I_flaret50/k1b/k1b.json and I_flaret50/exclusions.json (both read-only; every image set and its count,
and the sha256 of the code that produced them), with K1B and EXCLUSIONS lines in final_prereg.log. The prepared 8 / 4 mm
arrays live in data_flaret50/k1b_cache/ and are deleted afterwards (logged).

usage:  CUDA_VISIBLE_DEVICES="" nice -n 15 taskset -c 16-23 .venv/bin/python -I -B scripts/vccor/final/flaret50/t50_overlap.py
        (normally via the ssl-flaret50-k1b unit)

Version of 2026-11-09; revised after the code review (M1b).
"""
import json
import os
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import t50_common as T  # noqa: E402

E0_DIR = T.REPO / "scripts/vccor/e0"
A240_DIR = T.REPO / "scripts/vccor/amos240"
E0_PAIRS = T.VC / "e0/overlap/pairs.json"
NEAR_DUP = 0.90
STOP_ABOVE = 10


def k1b_json():
    return T.K1B / "k1b.json"


def word_images():
    import word_common as W
    d = Path(W.IMAGES)
    return sorted(d.glob("word_*.nii.gz")) if d.is_dir() else []


def project_volumes(OV):
    """E0's volumes (FLARE 50, MACT 90, BTCV 50, AMOS 60) + AMOS-240 + the extracted WORD images -> (V, sets)."""
    V, X = OV.volumes()
    sys.path.insert(0, str(A240_DIR))
    import a240_common as A240
    allc = A240.all_labelled_ct()
    ids = A240.ids240()
    for c in ids:
        k = f"AMOS:{c}"
        assert k not in V, k
        V[k] = dict(kind="raw", path=str(allc[c][0]))
    for p in word_images():
        V[f"WORD:{p.name[5:9]}"] = dict(kind="raw", path=str(p))
    amos60 = sorted(k for k in V if k.startswith("AMOS:") and k[5:] not in set(ids))
    sets = {"FLARE22_train": sum(k.startswith("FLARE:") for k in V), "MACT": sum(k.startswith("MACT:") for k in V),
            "BTCV": sum(k.startswith("BTCV:") for k in V), "AMOS_amos_txt_60": len(amos60), "AMOS_240": len(ids),
            "WORD_extracted": sum(k.startswith("WORD:") for k in V)}
    return V, sets


def main():
    if k1b_json().exists() or T.EXCLUSIONS.exists():
        raise T.Refuse("K1b records exist (read-only); not rewritten")
    if not T.IMAGES_RECORD.is_file():
        raise T.Refuse("images_record.json missing: download the images first")
    if T.LBL_DIR.exists():
        raise T.Refuse("a FLARE-T50 label directory exists before the look")
    imgrec = json.load(open(T.IMAGES_RECORD))["images"]
    sys.path.insert(0, str(E0_DIR))
    import overlap as OV                                    # E0's frozen code (images only)
    from multiprocessing import Pool
    cache = T.DATA / "k1b_cache"
    T.disk_gate(4.0, "K1b cache")
    cache.mkdir(parents=True, exist_ok=True)
    log = (lambda m: T.log_event(m, T.K1B / "k1b.log"))
    V, sets = project_volumes(OV)
    for c in T.IDS:
        if imgrec[c].get("load_ok"):
            if T.sha256(T.image_path(c)) != imgrec[c]["sha256"]:
                raise T.Refuse(f"{c}: image changed after images_record.json")
            V[f"T50:{c[-4:]}"] = dict(kind="raw", path=str(T.image_path(c)))
    sets["FLARE_T50"] = sum(k.startswith("T50:") for k in V)
    log(f"K1b start: {len(V)} volumes {sets}; cache {T.rel(cache)}")
    info = {}
    os.environ["OMP_NUM_THREADS"] = "1"
    with Pool(6) as p:
        for k, inf in p.imap_unordered(OV.prep_one, [(k, it, str(cache)) for k, it in V.items()], chunksize=1):
            info[k] = inf
    log("prepared every volume")
    keys = {s: sorted(k for k in V if k.startswith(s + ":")) for s in ("T50", "FLARE", "MACT", "BTCV", "AMOS", "WORD")}
    hdr = []
    for ka in keys["T50"]:
        ia = info[ka]
        for kb in [k for k in V if not k.startswith("T50:")]:
            ib = info.get(kb, {})
            if "raw_shape" in ib and ia.get("raw_shape") == ib["raw_shape"] and \
                    all(abs(x - y) < 1e-3 for x, y in zip(ia["raw_spacing"], ib["raw_spacing"])):
                hdr.append({"a": ka, "b": kb, "raw_shape": ia["raw_shape"], "raw_spacing": ia["raw_spacing"]})
    comps = [("T50_vs_FLARE", "FLARE"), ("T50_vs_MACT", "MACT"), ("T50_vs_BTCV", "BTCV"), ("T50_vs_AMOS", "AMOS")]
    if keys["WORD"]:
        comps.append(("T50_vs_WORD", "WORD"))
    res = {}
    S1, rows = OV.compare("BTCV_vs_MACT", keys["BTCV"], keys["MACT"], str(cache), log)
    found = sorted((r["a"], r["b"]) for r in rows if r["ncc4"] >= NEAR_DUP)
    e0 = json.load(open(E0_PAIRS))["results"]["BTCV_vs_MACT"]["near_duplicates"]
    expected = sorted((r["a"], r["b"]) for r in e0)
    control_ok = found == expected and len(expected) == 47
    res["positive_control_BTCV_vs_MACT"] = {"expected_pairs": len(expected), "found_pairs": len(found),
                                            "identical_to_e0": found == expected, "pass": control_ok,
                                            "missed": sorted(set(expected) - set(found)),
                                            "false_positives": sorted(set(found) - set(expected))}
    log(f"positive control BTCV vs MACT: found {len(found)}, expected {len(expected)}, identical {found == expected}")
    near = {}
    for name, other in comps:
        extra = [(h["a"], h["b"]) for h in hdr if h["b"].startswith(other + ":")]
        S1, rows = OV.compare(name, keys["T50"], keys[other], str(cache), log, extra)
        rows = sorted(rows, key=lambda r: -r["ncc4"])
        nd = [r for r in rows if r["ncc4"] >= NEAR_DUP]
        res[name] = {"n_a": len(keys["T50"]), "n_b": len(keys[other]), "top5": rows[:5], "near_duplicates": nd,
                     "header_pairs_added_to_stage2": len(extra), "best_ncc4": rows[0]["ncc4"] if rows else None}
        for r in nd:
            near.setdefault(r["a"], []).append({"with": r["b"], "ncc4": r["ncc4"]})
        log(f"[{name}] near-duplicates {len(nd)}; top NCC4 " + ", ".join(f"{r['a']}~{r['b']} {r['ncc4']:.3f}"
                                                                         for r in rows[:3]))
    excluded = {}
    for ka, lst in near.items():
        excluded[f"FLARETs_{ka[4:]}"] = {"reason": "near-duplicate (NCC4 >= 0.90)", "pairs": lst}
    for c in T.IDS:
        if not imgrec[c].get("load_ok"):
            excluded[c] = {"reason": "file failed to load", "error": imgrec[c].get("load_error")}
    n_near = len(near)
    stop = n_near > STOP_ABOVE
    out = {"rule": "E0 overlap rule (scripts/vccor/e0/overlap.py, unchanged): near-duplicate iff NCC4 >= 0.90",
           "image_sets": sets, "n_project_volumes": len(V) - sets["FLARE_T50"], "results": res,
           "header_fingerprint_matches": hdr, "excluded": excluded, "n_excluded": len(excluded),
           "n_near_duplicates": n_near,
           "kill_test": f"K1b stops if more than {STOP_ABOVE} of the 50 are near-duplicates (AC.md:155); unloadable files "
                        f"are excluded but not counted by the kill test",
           "decision": "K1B_FAILED_CONTROL" if not control_ok else ("K1B_STOP" if stop else "K1B_PASS"),
           "scale_extension": "not run (E0's in-plane scale search was itself not pre-registered)",
           "code_sha256": dict(T.code_hashes(["t50_overlap.py", "t50_common.py"]),
                               **{"e0/overlap.py": T.sha256(E0_DIR / "overlap.py"),
                                  "e0/e0_common.py": T.sha256(E0_DIR / "e0_common.py"),
                                  "amos240/a240_common.py": T.sha256(A240_DIR / "a240_common.py")}),
           "written": T.now()}
    T.write_json(k1b_json(), out, mode=0o444)
    T.prereg_line("K1B", k1b_json(), f"flaret50 K1b near-duplicate check (images only, pre-registration): {out['decision']}, "
                                     f"{n_near} near-duplicates, {len(excluded)} excluded, {out['n_project_volumes']} "
                                     f"project images, positive control {'PASS' if control_ok else 'FAIL'}")
    if control_ok:
        T.write_json(T.EXCLUSIONS, {"excluded": excluded, "from": T.rel(k1b_json()), "decision": out["decision"],
                                    "written": T.now()}, mode=0o444)
        T.prereg_line("EXCLUSIONS", T.EXCLUSIONS, f"flaret50 exclusions fixed before any FLARE-T50 label: "
                                                  f"{sorted(excluded) or 'none'}")
    shutil.rmtree(cache, ignore_errors=True)
    log(f"K1b done: {out['decision']}; cache deleted")
    return 0 if out["decision"] == "K1B_PASS" else 4


if __name__ == "__main__":
    try:
        sys.exit(main())
    except T.Refuse as e:
        print(f"[REFUSE] {e.msg}", file=sys.stderr)
        sys.exit(e.code)
