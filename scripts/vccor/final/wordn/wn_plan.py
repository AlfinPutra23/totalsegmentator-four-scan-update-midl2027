"""vccor/final WORD-N PLAN: writes I_wordn/plan_wordn.json, the machine-readable rules of the registered look (AC.md GO-2
with the author's decisions of 2026-11-13 and fixed deviations). Structured technical content only: every
outcome sentence is the author's to write (placeholders). Rewriting is allowed only before code_freeze.json exists.

usage:  .venv/bin/python -I -B scripts/vccor/final/wordn/wn_plan.py [--force]

Version of 2026-11-13.
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import wn_common as T  # noqa: E402

OWNER = "[OWNER WRITES]"

PLAN = {
    "id": "wordn",
    "title": "WORD-N: a registered look at WORD's 20 never-opened validation scans with the existing WORD models",
    "design_source": ("GO-2 of the AI-simulated area chair's review of 2026-11-09 (not a human review), with the volume "
                      "endpoints from the AI-simulated reviewer R1; adopted by the owner on 2026-11-13 ('yes'; VISTA3D tier: "
                      "'do vista tier in word N'; overshoot reading: option A, the split); the owner writes the "
                      "registration text"),
    "overshoot_mode": "SPLIT",
    "overshoot_mode_note": ("SPLIT = the owner's option A of 2026-11-13: signed over-segmentation of the duodenum and the "
                            "oesophagus is confirmatory (Holm family); the |e|-larger volume-accuracy cost is descriptive "
                            "only. The split was chosen after an AI agent's post hoc analysis of the stored WORD test-look "
                            "records (C_word/look/pervol, 2026-11-13): there the AC's combined rule held for the duodenum in "
                            "about 48 % of 20-scan subsamples and failed for the oesophagus on all 29 test scans, while the "
                            "signed shift held in about 98-100 %. The code also implements the AC's combined rule "
                            "('AC_COMBINED'); the plan fixes SPLIT."),
    "data": {
        "source": "WORD-V0.1.0 (Luo et al., Medical Image Analysis 82:102642, 2022), downloaded 2026-10-22 (C_word/"
                  "download.json; archive sha256 1ef1e48f...)",
        "set": "the official validation split, 20 scans: " + " ".join(T.VAL_IDS),
        "same_centre": "same centre, scanner and protocol as WORD's test set (one centre, SIEMENS, non-contrast CT)",
        "never_opened": "no validation image or label was ever extracted or read (records: download.json labels_opened "
                        "NONE; no WORD log names imagesVal / labelsVal; word_common cannot read labelsVal; 42 images "
                        "extracted = 12 draw + 30 test); the owner confirmed on 2026-11-13 that no validation label was "
                        "extracted by hand (final_prereg.log OWNER lines 09:31:37 and 09:49:49)",
        "sealed": "the 88 WORD training scans outside the three draws stay sealed (never extracted); the WORD test labels "
                  "are not read again",
        "header_twin": "0083 (validation) shares its header with 0095, one of the 88 sealed scans: 0095 is used in no role",
        "pinning": "archive_members.json: the 7z listing (sizes, packed sizes, modified times) of the 40 whitelisted "
                   "members, made before registration without decrypting any member; the archive's sha256; 7z checks "
                   "each AES member's authentication code on extraction",
        "licence": "WORD: GPL-3.0 repository licence; research use, no redistribution of the data",
        "plan_c_change": "Plan C (c5fyj, public 2026-11-06) said 'validation: not used'; this registration changes that",
    },
    "order": [
        "before registration: code, unit tests, the look rehearsal (stand-in data), the VISTA3D smoke test (12 draw scans "
        "only), the archive member pin, the freeze and the deposit",
        "the OSF registration is initiated (timestamped) before any validation image is extracted and before any "
        "prediction or tier-2 training; the upload gate names the deposit zip and the code freeze",
        "the registration is public (OSF API, checked between the upload and the stop rule) before any validation label "
        "is extracted or read",
        "one look: GT_OPENED marker before the label extraction, labels.json then one hashed label tar, one RESULTS line",
        "cutoff Thu 2026-11-19 18:00 KST: an arm not DONE by then is NOT RUN; a DONE.json written at or after the cutoff "
        "counts as NOT RUN",
        "stop rule: if the registration is not public by Sat 2026-11-21 12:00 KST the validation labels stay sealed (the "
        "images may have been extracted at the upload gate) and the paper states the registration and its status",
        "arm accounting at the look: a runner NOT RUN declaration is binding (a later DONE is ignored and logged); tier 2 "
        "counts only with VSMOKE_PASS; an arm not DONE by the cutoff is NOT RUN, so a stopped runner cannot block the look "
        "after the cutoff; a tier-2 integrity failure at the look (VISTA3D bundle, frozen weights) makes only that arm NOT "
        "RUN, a tier-1 integrity failure refuses the look",
        "GPU: every GPU job runs on GPU 1 only (CUDA_VISIBLE_DEVICES = GPU 1's UUID, bus 00000000:05:00.0, checked by the job) "
        "and takes the shared flock /home/user/Documents/Handpose_project/runs/method_pilot3/gpu1.lock itself; the waits for "
        "the lock, for GPU 1 to be idle (waited for without holding the lock, re-checked after taking it), for disk space and "
        "through GPU-infrastructure outages (nvidia-smi failing, GPU 1 or its UUID not listed, the lock file missing; the "
        "lock file is never created by WORD-N) are bounded by the cutoff only; a wrong CUDA_VISIBLE_DEVICES is a "
        "configuration error (a failure); a resource refusal is not a failure: the job is retried after a 10-minute "
        "back-off until the cutoff and never declares an arm NOT RUN by itself; a failed job is rerun once, after a "
        "10-minute back-off, and a second failure declares its arm NOT RUN; the VF arms are NOT RUN when V0 "
        "is NOT RUN",
        "a CPU arm counts at the look only if its inputs count; a tier-2 map that differs from its DONE record makes only "
        "that arm NOT RUN (a tier-1 map mismatch refuses the look)",
    ],
    "operational_notes": ["do not move, rename or delete the deposit zip (or its folder) before the look: every gate reopens "
                          "and hashes it",
                          "every job runs with PYTHONDONTWRITEBYTECODE=1 and python -I -B"],
    "disclosures": ["archive_members.json was written by the pre-review wn_intake.py / wn_common.py (listing only; the "
                    "freeze records the producer-code difference)",
                    "a unit test of the pre-fix code wrote 4 test lines to I_wordn/logs/run.log on 2026-11-13 10:25:26; the "
                    "file was moved aside to logs/run.log.unit_test_stray_2026-11-13 (not hashed, not deposited)",
                    "the pre-review --pin wrote one events line to outputs/logs/vccor/final_wordn.log (the old log path), "
                    "left as it is",
                    "VISTA3D smoke attempt 1 was stopped by hand before it acquired the GPU lock (NOTE 2026-11-13 10:20:24);"
                    " kept in vsmoke/attempt_1/"],
    "write_area": ["outputs/results/vccor/final/I_wordn/ (records, predictions, the look, event logs in I_wordn/logs/)",
                   "MagicNet/data/word_v010/images/ (the 20 validation images only)",
                   "outputs/logs/vccor/final_prereg.log (registration lines)",
                   "outputs/results/vccor/osf_deposit_<date>_wordn/ and its .zip (the one deposit)"],
    "exclusions": {"rule": "Pearson r of the intensities on a 4-voxel stride grid >= 0.999 between same-shape images "
                           "(validation vs validation, draw and test scans), applied to the images after the upload gate "
                           "and before any prediction: a validation duplicate of a draw or test scan is dropped and "
                           "reported; of two duplicate validation scans the higher id is dropped"},
    "arms": {"tier_1": {a: T.arms()[a]["label"] for a in T.TIER1},
             "tier_2": {a: T.arms()[a]["label"] for a in T.TIER2},
             "required": T.REQUIRED,
             "tier_2_condition": "tier 2 runs only if the VISTA3D smoke test (vsmoke/smoke.json, before registration, 12 "
                                 "draw scans) says VSMOKE_PASS; otherwise every tier-2 arm is NOT RUN",
             "no_training_of_tier_1": "every tier-1 model exists and is hashed in the freeze (A1 / A1.2 models)"},
    "endpoints": {
        "volume_error": "e = 100 (V_pred - V_ref) / V_ref per organ present in the reference (>= 1 mL); V_pred = the arm's "
                        "total predicted organ volume",
        "P_V": {"status": "PRIMARY, alpha 0.05, alone",
                "estimand": "mean over the draws of the mean over scans of the per-scan mean over liver, spleen, "
                            "kidney_left, kidney_right, pancreas (present) of |e_F1b_d| - |e_F0|",
                "ci": "crossed two-level (scan x draw) percentile bootstrap, 10,000, seed 20261020",
                "reading": "VOLUME_ERROR_REDUCED iff CI upper < 0; VOLUME_ERROR_INCREASED iff CI lower > 0; else NOT_SHOWN"},
        "holm_family": {
            "m": 4, "alpha": 0.05, "p": "two-sided bootstrap p from the replicates of each two-level contrast; "
                                        "intersection-union: the larger p and both estimates > 0 (each one-sided part at "
                                        "0.025: conservative); with one draw run, P1-fresh's p comes from the two-level "
                                        "helper with one draw (seed 20261020) while its CI is A1's one-draw scan bootstrap "
                                        "(seed 20261022)",
            "missing_member": "a member that cannot be evaluated (an arm NOT RUN) enters Holm with p = 1.0, so m stays 4; "
                              "its reading is NOT RUN / NOT EVALUABLE; the evaluable members are reported",
            "OS_D": "duodenum OVER-SEGMENTATION: mean e_F1b > 0 AND mean (e_F1b - e_F0) > 0 (two-level) -> OVERSEGMENTED "
                    "iff Holm-adjusted p < 0.05 and both estimates > 0, else NOT_CONFIRMED",
            "OS_E": "the same for the oesophagus",
            "P1_fresh": "WORD A1's P1 rule on F1b_d - F0 (M >= +0.30, scan t-CI lower > 0, two-level CI lower > 0, "
                        ">= ceil(2n/3) scans up) AND Holm-adjusted p < 0.05 -> GAIN, else NO_GAIN",
            "S1_fresh": "F1b_d - STOCK_d paired by draw, two-level, -> F1B_BETTER / STOCK_BETTER / NOT_DETECTED"},
        "AC_D_AC_E_descriptive": "duodenum / oesophagus volume-accuracy cost |e_F1b| - |e_F0| (two-level CI): COST_SHOWN "
                                 "iff CI lower > 0, NO_COST_SHOWN otherwise; descriptive only (owner's option A)",
        "tier_2": "its own family, alpha 0.05, one test: V-P1 = A1's P1 rule on VF5k_d - V0 -> V_GAIN_EVERY_DRAW / "
                  "V_GAIN_POOLED / V_NO_GAIN ('and lost Dice' iff CI upper < 0) / NOT RUN; descriptive only, no ranking: "
                  "V0 - F0, VF5k_d - F1b_d, VF5k_lrA_d1 - V0, VF5k_lrA_d1 - VF5k_d1",
        "descriptive": ["A1's full readings on the fresh scans (P1, organ costs at -1.0 Dice, P2 per draw)",
                        "A1.2's secondary contrasts (STOCK, MERGE_N, LABELMAP) on the fresh scans",
                        "the acceptance check: k of 6 correct decisions (accept F1b_d, reject STOCK_d) on all scans and on "
                        "the first 10 of a seeded order (seed 20261022)",
                        "volume SD of e (limits of agreement), share within +-10 %, per-organ signed e of every arm",
                        "failures fixed / created (Dice-50 rule), HD95, NSD",
                        "two panels: the median-gain scan and the largest d1 duodenum overshoot"],
        "multiplicity": "P-V alone at 0.05; Holm over 4; tier 2 its own family of one; descriptive readings unadjusted",
    },
    "claim_rules": {
        "allowed_only_if_read": {"VOLUME_ERROR_REDUCED": "reduced solid-organ volume error",
                                 "OVERSEGMENTED": "shifted the [organ] to larger-than-reference volumes",
                                 "COST_SHOWN (descriptive)": "made the [organ] volume less accurate (descriptive)"},
        "never": ["'improves volumetry' in general", "'external validation' or a 'second cohort'",
                  "pooling WORD-N with the WORD test scans", "a ranking of TotalSegmentator against VISTA3D",
                  "'STOCK is worse' beyond S1-fresh's reading", "'every scan' unless k = n"],
        "always": ["20 validation scans from the same centre, scanner and protocol as WORD's test set: an extension, not "
                   "an external cohort; WORD cannot split a gain into convention and image domain",
                   "the volume hypotheses and the overshoot split come from an AI agent's post hoc analysis of the stored "
                   "WORD test records; the designer knew the WORD test results; Plan C said 'validation: not used'"],
    },
    "wording": {k: OWNER for k in ("P_V_REDUCED", "P_V_NOT_SHOWN", "P_V_INCREASED", "OS_OVERSEGMENTED", "OS_NOT_CONFIRMED",
                                   "AC_COST_descriptive", "P1_fresh", "S1_fresh", "acceptance_check", "V_P1", "always")},
    "look_counters": {"WORD validation labels": "0 reads before this look; this is read 1 of 1",
                      "WORD test labels": "1 look (2026-11-07); not read again", "88 training scans": "sealed"},
    "stats": {"two_level": "m2_cluster.boot2 crossed scan x draw, 10,000, seed 20261020",
              "scan": "final_common.cluster_boot stream, 10,000, seed 20261022",
              "helpers": "flaret50/t50_analysis.py (frozen): replicates, p-values, Holm"},
    "gpu_hours": {"tier_1": "about 2.2 (F0 0.49, F1b 3 x 0.20, STOCK 3 x 0.38)",
                  "tier_2": "about 4 (V0 about 0.05, VF5k 3 x 0.9 + lrA 0.9 training, 4 x about 0.05 prediction)",
                  "smoke": "about 0.3 before registration"},
    "disk": "about 1.2 GB of validation images on / plus about 0.3 GB of predictions; / kept >= 25 GiB free",
    "ai_assistance": "the code, the records and the structured content were produced by an AI coding assistant (Claude); "
                     "the design came from an AI-simulated review; the registration text is the owner's",
}


def main():
    force = sys.argv[1:] == ["--force"]
    if T.FREEZE.exists():
        raise SystemExit("[REFUSE] the freeze exists: the plan is fixed")
    if T.PLAN.exists() and not force:
        raise SystemExit(f"[REFUSE] {T.rel(T.PLAN)} exists (use --force before the freeze)")
    T.write_json(T.PLAN, dict(PLAN, written=T.now(), code_sha256=T.code_hashes(["wn_plan.py", "wn_common.py"])))
    print(f"wrote {T.rel(T.PLAN)} ({T.sha256(T.PLAN)[:16]})")


if __name__ == "__main__":
    main()
