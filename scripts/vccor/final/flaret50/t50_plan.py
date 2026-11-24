"""vccor/final FLARE-T50 PLAN: writes I_flaret50/plan_flaret50.json, the machine-readable rules of the registered look
(AC.md GO-1 with fixed deviations and the code-review fixes of 2026-11-09). Structured technical content
only: every outcome sentence is the author's to write (placeholders). Rewriting is allowed only before code_freeze.json exists.

usage:  .venv/bin/python -I -B scripts/vccor/final/flaret50/t50_plan.py [--force]

Version of 2026-11-09; revised after the code review (M1b).
"""
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import t50_common as T  # noqa: E402

PLAN = {
    "id": "flaret50",
    "title": "FLARE-T50: a registered look at 50 never-opened FLARE scans with frozen FLARE models",
    "design_source": ("GO-1 of the AI-simulated area chair's review of 2026-11-09 (not a human review), adopted on the "
                      "owner's instruction of 2026-11-09 ('clear out the space and do the experiment'); revised after an "
                      "independent AI code review of 2026-11-09; the owner writes the registration text"),
    "data": {
        "source": "Hugging Face dataset FLARE-MedFM/FLARE-Task2-LaptopSeg (MICCAI FLARE 2024-25 Task 2)",
        "revision": T.HF_REVISION, "subset": "validation/Validation-Public-Images and -Labels, 50 cases FLARETs_0001-0050",
        "licence": "CC BY-NC 4.0 (non-commercial research use; cite FLARE22 and the 2024-25 release)",
        "access": "gated 'auto': the owner's own account accepts the access conditions",
        "pinning": "hf_manifest.json: size and LFS-pointer git oid of every image and label file at the pinned revision, "
                   "fetched from the public listing before any file was downloaded; each downloaded file must reproduce "
                   "its pointer oid",
        "identity": {"inferred": "FLARE22's 50-case validation (tuning) set",
                     "card_lines_verbatim": ["The old validation set and testing set are merged as a new validation set "
                                             "with 250 cases.", "200 hidden validation set and 50 public validation set."],
                     "reasoning": "FLARE22 had a 50-case validation set and a 200-case test set",
                     "still_inferred": "that the 50 public cases are exactly FLARE22's validation cases; their centres "
                                       "and label ids are not stated on the card"},
        "use_before_registration": "no project doc, script, data folder, Hub cache or Downloads folder held these files "
                                   "when checked on 2026-11-09 without opening any label (shell history, other accounts "
                                   "and deleted files were not checked); the images are downloaded before registration "
                                   "for the disclosed steps below; the labels only after the registration is public",
        "role": "internal replication on never-opened scans of the FLARE22 cohort, not external validation",
    },
    "order": [
        "the OSF registration is initiated (timestamped) before any tier-1 prediction on the set; the upload gate names "
        "the deposit zip (with its DEPOSIT line) and the code freeze, and the upload must follow the freeze",
        "images-only steps before the registration (disclosed): the image download, hashing, K1b and K1c",
        "the registration is public (OSF API, checked between the upload and the stop rule) before any label of the set "
        "is downloaded or read",
        "one look: GT_OPENED marker before the label download, labels.json then one hashed label archive, one RESULTS line",
        "cutoff Fri 2026-11-13 18:00 KST: an arm not DONE by then is NOT RUN; a DONE.json written at or after the cutoff "
        "counts as NOT RUN",
        "the look may start before the cutoff once every arm is DONE or declared NOT RUN",
        "stop rule: if the registration is not public by Tue 2026-11-17 12:00 KST the set stays sealed and the paper "
        "states the registration and its status",
    ],
    "exclusions": {
        "rule": "fixed before registration: (a) near-duplicates of any project image under E0's registered overlap rule "
                "(scripts/vccor/e0/overlap.py unchanged: NCC4 >= 0.90; pairs with matching raw header fingerprints are "
                "added to stage 2), with the BTCV-MACT positive control reproducing E0's 47 pairs; (b) image files that "
                "fail to load",
        "project_images": "FLARE22 training 50; MACT 90 (images only); BTCV 50; AMOS 300 (the 60 of amos.txt and the 240 "
                          "of AMOS-240); every WORD image already extracted (42); each set and its count recorded in "
                          "k1b.json",
        "record": "exclusions.json and k1b/k1b.json",
    },
    "kill_tests": {
        "K1b": "more than 10 of the 50 are near-duplicates -> stop (no registration); unloadable files are excluded but not "
               "counted by the kill test",
        "K1c": "run once, on exactly FLARETs_0030 (smallest) and FLARETs_0050 (largest): F0 or FULL_s1337 takes more than "
               "2x the expected minutes (rate x max(1, MB/29.5), rate 1.44 F0 / 0.65 FULL), fails, or reserves more than "
               "22 GiB GPU memory -> stop; the voxel count at 1.5 mm is recorded",
        "label_sanity": "inside the look, before any Dice: every included label must load, hold only exact integers in "
                        "0..13, and have the image's shape; otherwise GT_ABORTED and no result",
    },
    "arms": {a: {"tier": v["tier"], "role": v["role"], "label": v["label"]} for a, v in T.ARMS.items()},
    "arm_hashes": "code_freeze.json: every checkpoint with its export / plans / dataset json, and TotalSegmentator 2.13.0's "
                  "own weights (TS_WEIGHTS) for F0 and the frozen parts of every 'ts' arm; K = 1 models trained after the "
                  "freeze are hashed by a freeze addendum before their own arm's prediction (its time relative to the "
                  "upload is disclosed); package versions pinned",
    "prediction": "TotalSegmentator 2.13.0 'total' pipeline (frozen ts_pipeline.py), fine-tuned parts redirected, frozen "
                  "parts replayed from F0's capture; nnU-Net arms with nnUNetPredictor (tile 0.5, gaussian, no mirroring)",
    "endpoint": {"metric": "penalised macro Dice over FLARE's 13 organs (conventions v1)",
                 "frame": "native grid of the reference", "presence": ">= 1 mL on both sides; FP organ scores 0; organs "
                 "absent on both sides are left out", "pairing": "paired per scan against F0 unless stated"},
    "primary": {"id": "R1", "alpha": 0.05, "adjusted": False,
                "contrast": "mean over the three K = 4 FULL draws (s1337, s2025, s3407) of FULL_d - F0 per scan",
                "rule": "WORD A1's P1: M (mean of the draw means) >= +0.30 AND the scan t-CI lower bound > 0 AND the crossed "
                        "two-level (scan x draw) bootstrap CI lower bound > 0 (10,000 resamples, seed 20261020) AND at least "
                        "ceil(2n/3) scans with a positive draw-averaged difference",
                "per_draw": "the single-draw form (scan bootstrap, 10,000, seed 20261022) -> k of 3 -> WORD A1's outcome "
                            "combinations (GAIN_EVERY_DRAW, ...)",
                "note": "with three draws, each single-draw composition (one draw resampled three times) carries about "
                        "1/27 (3.7%) of the two-level replicates, more than the 2.5% tail, so in practice the two-level CI "
                        "excludes 0 only when the three draws agree in direction"},
    "secondary": {
        "family": "Holm, m = 6, family-wise alpha 0.05; detected iff the Holm-adjusted p < 0.05 AND the estimate lies in "
                  "the stated direction; adjusted p < 0.05 in the opposite direction reads REVERSED; unadjusted 95% CIs "
                  "always reported; a contrast with a NOT RUN arm is listed and leaves m (m = the contrasts that ran)",
        "p_value": "two-sided bootstrap p = min(1, 2 (min(#{r* <= 0}, #{r* >= 0}) + 1) / (B + 1)) over the B = 10,000 "
                   "replicates of the contrast's own bootstrap",
        "contrasts": {
            "S1": {"name": "pull-back", "contrast": "SSFT_B - FULL_s1337", "direction": "< 0",
                   "bootstrap": "scan, seed 20261022"},
            "S2": {"name": "label source", "contrast": "SSFT_D - SSFT_B (same 36 images)", "direction": "> 0",
                   "bootstrap": "scan, seed 20261022"},
            "S3": {"name": "gentle vs stock", "contrast": "FULL_s1337 - STOCK", "direction": "> 0",
                   "bootstrap": "scan, seed 20261022",
                   "also": "STOCK - F0 read only as STOCK_GAINS or STOCK_NO_GAIN (P1 single-draw form), as WORD S1b"},
            "S4": {"name": "more labels",
                   "contrast": "K16_d - FULL_d paired by draw (each K16 draw contains its K4 draw's four scans), mean over "
                               "the draws", "direction": "> 0", "bootstrap": "crossed two-level, seed 20261020"},
            "S5": {"name": "local vs borrowed",
                   "contrast": "mean of the three FLARE K = 4 draws - mean of the three borrowed models (AMOS, MACT, WORD), "
                               "each minus F0", "direction": "> 0",
                   "bootstrap": "crossed two-level: scans resampled jointly for both families, the FLARE draws resampled, "
                                "the three borrowed models held FIXED; seed 20261020",
                   "inference_target": "these three specific borrowed models, not a population of cohorts",
                   "also": "borrowed - F0 per cohort descriptive: HELPS / HARMS / NEITHER by its scan-bootstrap CI; MACT "
                           "and WORD tune part 291 only"},
            "S6": {"name": "solid-organ volume error",
                   "contrast": "per scan, mean over liver, spleen, both kidneys and pancreas (present in the reference) "
                               "of |e_FULL_d| - |e_F0|, e = 100 (V_pred - V_ref) / V_ref; mean over the three draws",
                   "direction": "< 0", "bootstrap": "crossed two-level, seed 20261020"},
        },
    },
    "tier2": {"arms": "FULL K = 1 retrained on the four fair-fight draws (0014, 0001, 0006, 0038) with the frozen "
                      "fair-fight code",
              "training_rules": "kill rule: the first training must finish within 3 h of wall time (all its attempts, an "
                                "interrupted attempt included), else K = 1 is NOT RUN; a draw that fails twice is NOT RUN; "
                                "no training is launched unless it can end 1 h before the cutoff",
              "test": "K1 pooled - K4 < 0: crossed two-level with each family's draws resampled independently, seed "
                      "20261020; detected iff p < 0.05 and the estimate < 0 (its own family, one test)",
              "count_reading": "SINGLE_SCAN_RISK replicated iff at least one K = 1 draw has >= 25% of its scans below F0"},
    "descriptive_registered": [
        "every scan: k of n with a positive draw-averaged difference; the words 'every scan' only if k = n",
        "organ costs: WORD A1's per-organ rule (two-level CI of each organ's Dice change; NON-INFERIOR / INFERIOR / "
        "INCONCLUSIVE at -1.0 Dice); any INFERIOR organ is named in every branch",
        "SCRATCH_NM1000 - F0; per-draw values; absolute macro Dice per arm",
        "two qualitative panels by fixed rules: the scan with the median of the three-draw average difference (lower "
        "median) and the scan with its minimum (largest loss); each drawn with F0 and FULL_s1337",
        "label diagnostics inside the look (descriptive only): affine difference to the image; whether the liver is the "
        "largest labelled organ",
    ],
    "outcome_wording": {
        "author": "the owner writes every sentence (MIDL's LLM policy); this plan gives only the branch structure",
        "branches": {
            "R1 REPLICATED": "[OWNER WRITES]; FLARE-T50 becomes the FLARE primary; held18 and test10 become exploratory "
                             "history",
            "R1 NOT REPLICATED": "[OWNER WRITES]; held18 and test10 results are labelled exploratory; the abstract rests on "
                                 "WORD and MACT; add the loss if the CI upper bound < 0",
            "S1-S6 each": "[OWNER WRITES] detected / not detected / reversed, with M [CI] Holm-adjusted; a failure is "
                          "named in the failure-mode section",
            "every scan": "[OWNER WRITES] k of n; if k < n the every-scan wording is retired everywhere",
            "organ costs": "[OWNER WRITES] INFERIOR organs named in the abstract next to the MACT pancreas; INCONCLUSIVE "
                           "organs listed",
            "tier 2": "[OWNER WRITES] SINGLE_SCAN_RISK replicated / not replicated / NOT RUN",
            "16 Oct declaration": "[OWNER WRITES] the reason for adding FLARE-T50 (and WORD) after the 16 Oct declaration "
                                  "that no other datasets would be added for the MIDL submission",
        },
        "always_disclose_facts": [
            "the 50 public validation cases of the FLARE 2024-25 Task 2 release, inferred to be FLARE22's tuning set, "
            "released with labels in 2025 (CC BY-NC 4.0)",
            "the images were downloaded for the disclosed pre-registration steps; the labels only after the registration "
            "was public",
            "same challenge reference standard; centres not stated by FLARE22",
            "internal replication on never-opened scans, not external validation",
            "the hypotheses come from at least 23 scoring reads of held18 and from the development set test10",
            "models and TotalSegmentator's weights frozen and hashed before registration (K = 1 models possibly by a "
            "dated addendum)",
            "the design came from an AI-simulated review; the code was written with AI assistance and reviewed by an "
            "independent AI agent",
            "images downloaded, hashed, overlap-checked (K1b) and timing-tested on two images (K1c) before registration",
        ],
    },
    "deviations_from_AC_md": [
        "images stored on the system disk, not the external drive (the external drive is under the owner's write-nothing "
        "rule)",
        "GO-2 (WORD-N) is not part of this registration (the owner has not ruled on the WORD rule)",
        "K1c's 'expected time or memory' made operational (see kill_tests.K1c)",
        "the label sanity gate adds the shape check, the exact-integer test and the unreadable-file case",
        "the Holm p-value and the two-family bootstrap for tier 2 defined as above; S5 holds the borrowed models fixed",
        "the NEITHER reading of borrowed - F0",
        "the look may start before the cutoff once every arm is DONE or NOT RUN",
        "K1b: header-fingerprint pairs are added to E0's stage 2; AMOS-240 and the extracted WORD images are compared too",
        "tier-2 K = 1 models trained with moonshot/ff_launch.py (the frozen fair-fight code) into moonshot/ff/models",
        "provenance: the tier-2 runner started at 11:09:08 KST on 2026-11-09 imported t50_common.py version a75b24a5..., "
        "which was later extended; that version is kept byte-exactly as tier2/t50_common_a75b24a5.py; the running runner "
        "charges wall time only after a normal exit (the fix applies from its next start)",
    ],
    "compute": {"gpu": "GPU 1 only, shared GPU-1 flock, idle check; GPU 0 not used", "disk_floor_gib": T.FLOOR_GIB,
                "cpu": "K1b, the look and checks on cores 16-23 under nice", "python": "-I for every step that reads the "
                "downloaded data"},
    "independent_check": "by an agent that did not write the code, recomputing every reading from look/pervol records",
}


def main():
    force = "--force" in sys.argv
    if T.FREEZE.exists():
        raise SystemExit("[REFUSE] code_freeze.json exists: the plan is frozen")
    if T.PLAN.exists() and not force:
        raise SystemExit(f"[REFUSE] {T.rel(T.PLAN)} exists; --force rewrites it before the freeze")
    if T.PLAN.exists():
        os.chmod(T.PLAN, 0o644)
    T.write_json(T.PLAN, dict(PLAN, written=T.now()), mode=0o444)
    T.prereg_line("PLAN", T.PLAN, "flaret50 plan (rules; outcome sentences left to the owner); pre-freeze version")


if __name__ == "__main__":
    main()
