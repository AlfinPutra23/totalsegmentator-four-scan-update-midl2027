"""vccor/final FLARE-T50 SMOKE RECORD (CPU; reads a smoke prediction root only; diff check F1).

Builds I_flaret50/smoke_predict.json from a t50_predict.py --smoke prediction root (one folder per arm, each with its
DONE.json and rec/<id>.json): per arm the kind, scans, replay misses, captured parts, each part's source (ts arms; per
scan only if the scans differ), foreground voxels (summed over scans), model sha256, the code the arm ran on and its
finish time; plus the stand-in images' sha256, the tier-1 arms missing from the root and this code's sha256. Logs a SMOKE
line in final_prereg.log. Every DONE.json must say mode 'smoke'; the root must lie outside the study folders.
Refuses after the freeze: smoke_predict.json is then a frozen, read-only record and is never rewritten (diff check F1);
a later smoke keeps its outputs in its own scratch root.

usage:  nice -n 15 taskset -c 16-23 .venv/bin/python -I -B scripts/vccor/final/flaret50/t50_smoke_record.py \\
            --root PRED_ROOT --images DIR --note TEXT

Version of 2026-11-09 (diff check F1).
"""
import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import t50_common as T  # noqa: E402


def record_path():
    return T.OUT / "smoke_predict.json"


def arm_entry(d):
    done = json.load(open(d / "DONE.json"))
    if done.get("mode") != "smoke":
        raise T.Refuse(f"{d / 'DONE.json'}: mode {done.get('mode')!r}, not 'smoke'")
    recs = {c: json.load(open(d / "rec" / f"{c}.json")) for c in done["ids"]}
    per = {c: {str(p["tid"]): p["source"] for p in r["parts"]} for c, r in recs.items() if r.get("parts")}
    same = len({json.dumps(v, sort_keys=True) for v in per.values()}) == 1
    return {"kind": done["kind"], "n": done["n"], "replay_misses": done["replay_misses_this_run"],
            "captured_parts": done["captured_parts"],
            "sources": (next(iter(per.values())) if same else per) if per else None,
            "fg_vox": sum(r["fg_vox"] for r in recs.values()), "model_sha256": done["model_sha256"],
            "code_sha256": done.get("code_sha256"), "finished": done["finished"]}


def record(root, images, note):
    if T.FREEZE.exists():
        raise T.Refuse("code_freeze.json exists: smoke_predict.json is a frozen record and is not rewritten")
    root, images = Path(root).resolve(), Path(images).resolve()
    for q in (T.OUT, T.DATA):
        q = Path(q).resolve()
        if root == q or q in root.parents or images == q or q in images.parents:
            raise T.Refuse(f"the smoke root and images must lie outside the study folder {q}")
    arms = {d.name: arm_entry(d) for d in sorted(root.iterdir()) if (d / "DONE.json").is_file()}
    if not arms:
        raise T.Refuse(f"no arm with a DONE.json under {root}")
    out = {"what": note, "run": T.now(), "root": str(root), "arms": arms,
           "tier1_arms_missing": [a for a in T.TIER1_ARMS if a not in arms],
           "images_sha256": {p.name: T.sha256(p) for p in sorted(images.glob("*.nii.gz"))},
           "code_sha256": T.code_hashes(["t50_smoke_record.py", "t50_predict.py", "t50_common.py"]),
           "code_note": "each arm's code_sha256 is the code it ran on, from its DONE.json (null: written before DONE.json "
                        "recorded it)"}
    p = record_path()
    T.write_json(p, out)
    T.prereg_line("SMOKE", p, f"flaret50 CPU smoke of t50_predict.py --smoke on stand-in images: {len(arms)} arms, "
                              f"replay misses {sum(a['replay_misses'] for a in arms.values())}, tier-1 arms missing "
                              f"{out['tier1_arms_missing'] or 'none'}")
    return p


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="the smoke prediction root (one folder per arm)")
    ap.add_argument("--images", required=True, help="the stand-in images the smoke read")
    ap.add_argument("--note", required=True, help="what the stand-in is (stored as 'what')")
    a = ap.parse_args(argv)
    try:
        T.kst_gate()
        record(a.root, a.images, a.note)
    except T.Refuse as e:
        print(f"[REFUSE] {e.msg}", file=sys.stderr)
        return e.code
    return 0


if __name__ == "__main__":
    sys.exit(main())
