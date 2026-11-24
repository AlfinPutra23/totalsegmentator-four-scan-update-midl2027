"""vccor/final FLARE-T50 POST HOC REPORT tables (no label read; reads J_t50_posthoc/results_posthoc.json only).

  report:  .venv/bin/python -I -B scripts/vccor/final/t50_posthoc/ph_report.py           -> J_t50_posthoc/REPORT.md (tables;
           the reading section is written by hand afterwards)
  finish:  .venv/bin/python -I -B scripts/vccor/final/t50_posthoc/ph_report.py --finish  -> J_t50_posthoc/MANIFEST.json
           (sha256 of every output file) + DOC lines in final_prereg.log (t50_common.prereg_line) for REPORT.md,
           results_posthoc.json and MANIFEST.json
Version of 2026-11-13.
"""
import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import ph_common as PH  # noqa: E402

T = PH.T
NEW = list(PH.ORDER)
PH_TAG = "[post hoc]"


def f2(x, s="+"):
    return "NA" if x is None else (f"{x:{s}.2f}" if s else f"{x:.2f}")


def ci(c):
    return "NA" if c is None or c[0] is None else f"[{c[0]:+.2f}, {c[1]:+.2f}]"


def tables(res):
    L = []
    am = res["absolute_macro_dice"]
    C = res["contrasts"]
    L.append("## Table 1. Macro Dice and paired differences (POST HOC, exploratory; n = 50 scans)\n")
    L.append("Scan-bootstrap CI: final_common.cluster_boot, 10,000 resamples, seed 20261022. F0, FULL_s1337 and STOCK are the "
             "stored registered look/pervol records; the three post hoc arms are new. Up = scans with a positive difference.\n")
    L.append("| arm | trained | lr | macro Dice [95% CI] | minus F0 [95% CI] | up / 50 | minus FULL_s1337 [95% CI] | up / 50 |")
    L.append("|---|---|---|---|---|---|---|---|")
    meta = {"F0": ("nothing (zero-shot)", "-"), "FULL_s1337": ("all weights, parts 291+293", "1e-4"),
            "STOCK": ("all weights, new 13-class net", "1e-2"), "LR1E2": ("all weights, parts 291+293", "1e-2"),
            "LORA_B": ("LoRA adapters, parts 291+293", "1e-3"), "HEAD": ("seg heads only, parts 291+293", "1e-2")}
    for a in ["F0", "FULL_s1337", "STOCK"] + NEW:
        tag = f" {PH_TAG}" if a in NEW else " (registered look)"
        m = am[a]
        c0 = C.get(f"{a} - F0")
        c1 = C.get(f"{a} - FULL_s1337")
        L.append(f"| {a}{tag} | {meta[a][0]} | {meta[a][1]} | {f2(m['mean'], '')} {ci(m['ci']).replace('+', '')} | "
                 f"{f2(c0['mean']) + ' ' + ci(c0['ci']) if c0 else '-'} | {c0['scans_up'] if c0 else '-'} | "
                 f"{f2(c1['mean']) + ' ' + ci(c1['ci']) if c1 else '-'} | {c1['scans_up'] if c1 else '-'} |")
    L.append("")
    names = res["organs"]
    po = res["per_organ_mean_dice"]
    L.append("## Table 2. Per-organ mean Dice (POST HOC for the three new arms; organs present in the reference)\n")
    arms = ["F0", "FULL_s1337", "STOCK"] + NEW
    L.append("| organ | n | " + " | ".join(arms) + " |")
    L.append("|---|---|" + "---|" * len(arms))
    for o in names:
        L.append(f"| {o} | {po['F0'][o]['n']} | " + " | ".join(f2(po[a][o]['mean_dice'], '') for a in arms) + " |")
    L.append("")
    L.append("## Table 3. Per-organ mean Dice difference of the post hoc arms (POST HOC)\n")
    cols = [f"{a} - {b}" for a in NEW for b in ("F0", "FULL_s1337")]
    L.append("| organ | " + " | ".join(cols) + " |")
    L.append("|---|" + "---|" * len(cols))
    for o in names:
        L.append(f"| {o} | " + " | ".join(f2(C[k]['per_organ'][o]['mean_delta']) for k in cols) + " |")
    L.append("")
    vr = res["volume_ratio"]["arms"]
    L.append("## Table 4. Median volume ratio V_pred / V_ref (POST HOC for the three new arms)\n")
    L.append("| organ | " + " | ".join(arms) + " |")
    L.append("|---|" + "---|" * len(arms))
    for o in names:
        L.append(f"| {o} | " + " | ".join(f"{vr[a][o]['median_ratio']:.3f}" for a in arms) + " |")
    L.append("")
    fx = res["fn_fp"]["arms"]
    L.append("## Table 5. FN / FP organ counts over 50 scans x 13 organs (POST HOC for the three new arms)\n")
    L.append("| arm | FN | FP | scans with any FN | scans with any FP | FN organs | FP organs |")
    L.append("|---|---|---|---|---|---|---|")
    for a in arms:
        fno = ", ".join(f"{o} {n}" for o, n in fx[a]["FN_per_organ"].items() if n) or "-"
        fpo = ", ".join(f"{o} {n}" for o, n in fx[a]["FP_per_organ"].items() if n) or "-"
        L.append(f"| {a} | {fx[a]['FN']} | {fx[a]['FP']} | {len(fx[a]['FN_scans'])} | {len(fx[a]['FP_scans'])} | {fno} | {fpo} |")
    L.append("")
    return "\n".join(L)


def report():
    res = json.load(open(PH.OUT / "results_posthoc.json"))
    pr = json.load(open(PH.OUT / "predict_run.json"))
    rc = res["rescore_check"]
    head = f"""# FLARE-T50 post hoc: LR1E2, LORA_B and HEAD (EXPLORATORY, NOT REGISTERED)

Every number in this file that involves LR1E2, LORA_B or HEAD is **post hoc**. It comes from a second reading of the 50
FLARE-T50 labels, which the registered look first read on 2026-11-13 (GT_OPENED 09:01:23). The owner approved this
reading on 2026-11-13. The registered results in I_flaret50/look/results.json are unchanged.

**Question.** On T50, STOCK minus F0 is -13.5 Dice points. Does that collapse come from the learning rate (1e-2), from
training the whole network, or from neither?

**Models** (all TS-full parts 291 + 293, K = 4, draw s1337, 50 x 250 it; sha256 as in the PEFT records):
""" + "\n".join(f"- {a}: {res['models'][a]['label']}; checkpoints part291 `{res['models'][a]['checkpoint_sha256']['291']}`, "
                f"part293 `{res['models'][a]['checkpoint_sha256']['293']}` ({res['models'][a]['record']})" for a in NEW) + f"""

**How.** Predicted on GPU 1 with each model's registered TS-part predict path (frozen ts_pipeline; parts 292/294/295
replayed from F0's T50 cache, {sum(json.load(open(PH.PRED / a / 'DONE.json'))['replay_misses_this_run'] for a in NEW)} replay misses).
The predict scripts of the PEFT records then map to the FLARE22 h5 frame. That step does not apply to T50, so the raw maps
are scored like every 'ts' arm of the look: final_common.Scorer, conventions v1, native grid, FLARE's 13 organs,
ts_lut('flare'). Labels were read only through the verified copy. The archive was checked against look/labels.json and
each file was checked against look/labels.json before use. The temp folder was deleted afterwards. As a check of the
scoring path, F0 and FULL_s1337 were re-scored from their look maps and compared with the stored look records:
{'identical on all 50 scans' if rc['identical'] else 'MISMATCHES: ' + json.dumps(rc['mismatches'])[:300]}.
GPU wall time: {pr['gpu_wall_minutes_total']:.1f} min ({', '.join(f'{a} {m:.1f}' for a, m in pr['minutes_per_arm_this_run'].items())}).

## Reading (post hoc, exploratory)

READING_PLACEHOLDER

"""
    p = PH.OUT / "REPORT.md"
    assert str(p.resolve()).startswith(str(PH.OUT.resolve()))
    p.write_text(head + tables(res))
    print(f"wrote {T.rel(p)}")


def finish():
    rp = PH.OUT / "REPORT.md"
    assert "READING_PLACEHOLDER" not in rp.read_text(), "write the reading first"
    assert not PH.TMP_LABELS.exists(), "temp label folder still exists"
    files = sorted(p for p in PH.OUT.rglob("*") if p.is_file() and p.name != "MANIFEST.json")
    man = {"post_hoc": True, "status": "EXPLORATORY, NOT REGISTERED", "written": T.now(),
           "files": {str(p.relative_to(PH.OUT)): T.sha256(p) for p in files},
           "code_sha256": {T.rel(p): T.sha256(p) for p in sorted(HERE.glob("*.py"))}}
    PH.write_json(PH.OUT / "MANIFEST.json", man)
    T.prereg_line("DOC", rp, "flaret50 POST HOC t50_posthoc (EXPLORATORY, NOT REGISTERED; owner-approved 2026-11-13): "
                             "REPORT.md of the second reading of the FLARE-T50 labels, LR1E2 / LORA_B / HEAD vs F0 and FULL_s1337")
    T.prereg_line("DOC", PH.OUT / "results_posthoc.json", "flaret50 POST HOC t50_posthoc (EXPLORATORY, NOT REGISTERED): "
                                                          "results_posthoc.json")
    T.prereg_line("DOC", PH.OUT / "MANIFEST.json", f"flaret50 POST HOC t50_posthoc (EXPLORATORY, NOT REGISTERED): sha256 of "
                                                   f"all {len(files)} output files and the {len(man['code_sha256'])} code files")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--finish", action="store_true")
    a = ap.parse_args()
    finish() if a.finish else report()
