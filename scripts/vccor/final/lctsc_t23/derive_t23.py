"""LCTSC-T23: derive six scripts from the FROZEN LCTSC code (scripts/vccor/final/lctsc/, frozen 2026-11-13 18:16:25 in
K_lctsc/code_freeze.json) by explicit replacements. Each replacement must occur exactly the stated number of times (else the
derivation stops). The source hashes are checked against the LCTSC freeze first. Run once, before the T23 freeze; the derived
files and this script are frozen and deposited together, so every difference from the reviewed LCTSC code is listed here.

usage:  .venv/bin/python -I -B scripts/vccor/final/lctsc_t23/derive_t23.py
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC_DIR = HERE.parent / "lctsc"
REPO = Path("/home/user/Documents/Medical Imagining SSL")
LCTSC_FREEZE = REPO / "outputs/results/vccor/final/K_lctsc/code_freeze.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def head(text, doc, marker):
    """replace the module docstring (everything before the first `marker`) with `doc`."""
    i = text.index(marker)
    return doc.rstrip("\n") + "\n" + text[i:]


def apply(text, reps, name):
    for old, new, n in reps:
        k = text.count(old)
        if k != n:
            raise SystemExit(f"{name}: expected {n} occurrence(s), found {k}: {old[:90]!r}")
        text = text.replace(old, new)
    return text


DOC_LOG = '''"""vccor/final LCTSC-T23 registration-log helpers (DERIVED from the frozen lctsc_log.py by derive_t23.py): lines appended
to the project's final_prereg.log in its existing formats ('[time] KIND sha256 <h> <rel path> (note)' and
'[time] KIND lctsc_t23 <text>'), plus a plain event log. Standard library only.
"""'''

DOC_GATES = '''"""vccor/final LCTSC-T23 gates and arms (standard library only, so that both the SSL .venv and external/lctsc_env can
import it). DERIVED from the frozen lctsc_gates.py by derive_t23.py (every change is listed there), with LCTSC-T23's paths
and dates.

  freeze        K_lctsc_t23/code_freeze.json lists the sha256 of every study file (code, frozen imports, records, the frozen
                d6ayj models); freeze_ok() re-hashes them (a change refuses every gated step).
  upload gate   K_lctsc_t23/gate_osf_uploaded.json, written by the COORDINATOR only after the owner's OSF registration was
                submitted, keys osf_url, uploaded_kst, written_by, evidence, deposit_zip_sha256, freeze_sha256; the deposit
                sha must equal the last 'DEPOSIT sha256 <h> ...K_lctsc_t23/...lctsc_t23...zip' line of final_prereg.log,
                the freeze sha the current code_freeze.json, and the upload must be after the freeze. Required before any
                prediction on the 23 LCTSC-T23 CTs.
  public gate   K_lctsc_t23/gate_osf_public.json (coordinator only): keys osf_url, public_checked_kst, api_http_status
                (200), public (true), written_by, evidence; checked after the upload, BEFORE the public deadline Thu
                2026-11-19 12:00 KST and before the stop rule. Required before any of the 23 RTSTRUCT zips is extracted.
  cutoff        Fri 2026-11-20 10:00 KST: no prediction starts at or after it; a DONE.json written at or after it makes the
                arm NOT RUN. Stop rule Fri 2026-11-20 16:00 KST: no look (GT_OPENED) at or after it.
  look          K_lctsc_t23/look/GT_OPENED (O_EXCL) is created before the first of the 23 RTSTRUCT zips is extracted; one look.
  models        MODELS = K_lctsc/models: the frozen d6ayj models (no training in LCTSC-T23).
"""'''

DOC_QUEUE = '''"""vccor/final LCTSC-T23 PREDICTION RUNNER (run by the systemd user unit ssl-lctsc-t23-predict). DERIVED from the frozen
lctsc_predict_queue.py by derive_t23.py (every change is listed there). The registered order: F0, then F1b_S1, F1b_S2,
F1b_S3 (the frozen d6ayj models), each one job of t23_predict.py (real mode) on the 23 LCTSC-T23 CTs. Rules as LCTSC:

  * the runner waits (cutoff-aware, polled every 5 min) until the coordinator's upload gate holds, then checks the freeze
    and that no look exists before every job;
  * each GPU job takes GPU 1 itself (GPU-1 UUID identity gate, idle wait WITHOUT the lock, lock, re-check; every wait
    bounded by the cutoff Fri 2026-11-20 10:00 KST) and holds the shared flock until it exits;
  * exit 0 + DONE.json = DONE; exit 3 = a resource refusal (retried after the back-off until the cutoff); exit 4 = the
    cutoff passed while waiting; any other exit = a failure; a failure waits a 10-minute back-off before its second
    attempt; a second failure declares the arm NOT RUN ('NOTRUN lctsc_t23 <arm> ...', binding);
  * a job still running at the cutoff is stopped; after the cutoff every arm not DONE is declared NOT RUN;
  * an F1b arm needs F0 DONE (F0 NOT RUN -> 'NOTRUN lctsc_t23 EXPERIMENT', no look);
  * a hand stop (SIGTERM / SIGHUP / SIGINT) stops the running job and is logged (HANDSTOP).
The look is NOT started here. State: K_lctsc_t23/run/<arm>.json; events: outputs/logs/vccor/final_lctsc_t23.log.

usage:  .venv/bin/python -I -B scripts/vccor/final/lctsc_t23/t23_predict_queue.py [--check]

Written by an AI coding assistant (Claude) on the owner's instruction of 2026-11-13; LCTSC-T23 derivation 2026-11-17.
"""'''

DOC_LOOK = '''"""vccor/final LCTSC-T23 LOOK, part 1 of 2: the labels of the 23 LCTSC-T23 patients (CPU, no network; runs in
external/lctsc_env; ONE look). DERIVED from the frozen lctsc_look_labels.py by derive_t23.py (every change is listed there).

Gates, in this order, before any of the 23 RTSTRUCT zips is extracted (a new look):
  1. no second look: look/GT_OPENED and look/results.json must not exist;
  2. the freeze (t23_gates.freeze_ok) and the public gate (t23_gates.gate_public: upload gate, logged deposit holding the
     current freeze, public after the upload and before the public deadline Thu 2026-11-19 12:00 KST, before the stop rule
     Fri 2026-11-20 16:00 KST);
  3. ARMS: BEFORE the cutoff (Fri 2026-11-20 10:00 KST) every arm must be DONE in mode 'real' or declared NOT RUN by the
     prediction runner's binding NOTRUN line; AFTER the cutoff an arm not DONE by the cutoff is NOT RUN; F0 and at least
     one F1b arm must be DONE;
  4. the arms' maps are hashed into look/maps.json (each raw map against its DONE record).
Then look/GT_OPENED is created O_EXCL and a 'GT_OPENED sha256 ...' line is appended (re-appended by --continue if a crash
left it out). The look is defined by the GT_OPENED time: --continue may finish it after the stop rule.

Labels (NO download; the one change of substance from the LCTSC look): per patient, the RTSTRUCT zip downloaded on
2026-11-13 with the other training series (data_lctsc/tcia_zips/<pid>_RTSTRUCT.zip) is used. Before it is extracted it
must equal its K_lctsc/download.json record (sha256 and size) and the TCIA manifest (dcm member count and bytes, zip CRCs),
and no extracted copy may exist in data_lctsc/dicom; anything else logs GT_ABORTED and stops the look for good. The zip is
extracted under look/rtstruct/<pid>/ only after GT_OPENED. Conversion == the LCTSC look: lctsc_raster (frozen) on the CT
DICOM series re-downloaded on 2026-11-17 (data_lctsc/dicom/<pid>_CT; t23_ct.json shows each reproduces the stored NIfTI
voxel for voxel); the RTSTRUCT's referenced series is checked against the CT's SeriesInstanceUID; per patient a descriptive
sanity record; MISSING structures are listed (registered scoring rule: not evaluable for that patient). labels.json and a
'LABELS sha256 ...' line close the step.

usage:  external/lctsc_env/bin/python -I -B scripts/vccor/final/lctsc_t23/t23_look_labels.py [--check | --continue]

Written by an AI coding assistant (Claude) on the owner's instruction of 2026-11-13; LCTSC-T23 derivation 2026-11-17.
"""'''

DOC_SCORE = '''"""vccor/final LCTSC-T23 LOOK, part 2 of 2: SCORING and ANALYSIS (CPU, <= 6 workers, nice 15; SSL .venv; the same look).
DERIVED from the frozen lctsc_score.py by derive_t23.py (every change is listed there).

Requires (else refuses): look/GT_OPENED (from t23_look_labels.py) and no GT_ABORTED line; look/labels.json finished, with
a logged 'LABELS sha256' line equal to its hash; every label file equal to labels.json; the freeze unchanged
(t23_gates.freeze_ok); every arm map equal to look/maps.json; results.json absent (--continue reuses finished per-patient
records whose map and label hashes match).
Per patient: the reference = look/labels/<pid>.nii.gz (LCTSC's 5 ids, native CT grid); every arm's raw map (TS total ids)
-> LCTSC ids by lctsc_common.ts_total_lut(); final_common.Scorer (conventions v1; Dice, HD95, NSD 2 mm) ->
look/pervol/<pid>.json; t23_analysis.analyse -> look/results.json and one 'RESULTS sha256 ...' line.

usage:  CUDA_VISIBLE_DEVICES="" nice -n 15 .venv/bin/python -I -B scripts/vccor/final/lctsc_t23/t23_score.py [--continue]

Written by an AI coding assistant (Claude) on the owner's instruction of 2026-11-13; LCTSC-T23 derivation 2026-11-17.
"""'''

NEW_FETCH = '''def fetch(pid):
    """LCTSC-T23: NO download. The RTSTRUCT zip downloaded on 2026-11-13 with the other training series is used only if it
    equals its K_lctsc/download.json record (sha256 and size) and the TCIA manifest (dcm member count and uncompressed
    bytes, zip CRCs), and (real look) no extracted copy exists in data_lctsc/dicom; anything else logs GT_ABORTED."""
    if G.REHEARSAL:                         # review S2: rehearsal stand-ins only (site-draw patients read on 2026-11-13)
        assert pid in T.STANDINS and pid not in T.ids() and not C.is_test(pid), pid
    row = series_of(pid)
    dst = ZDIR / f"{pid}_RTSTRUCT.zip"
    rec = json.load(open(C.DOWNLOAD))["files"].get(f"{pid}_RTSTRUCT.zip")
    why = None
    if rec is None or not dst.is_file():
        why = "no stored zip or no K_lctsc/download.json record"
    elif not G.REHEARSAL and any((C.DICOM / n).exists() for n in (f"{pid}_RTSTRUCT", f".tmp_{pid}_RTSTRUCT")):
        why = "an extracted copy exists in data_lctsc/dicom (the label was opened before the look)"
    elif fault("mismatch", pid) or dst.stat().st_size != rec["bytes"] or G.sha256(dst) != rec["sha256"]:
        why = "sha256 or size differ from the K_lctsc/download.json record of 2026-11-13"
    else:
        try:
            why = check_zip(dst, row)
        except Temporary as e:
            why = str(e)
    if why:
        L.prereg_note("GT_ABORTED", f"RTSTRUCT {pid}: {why}; the look is aborted")
        raise G.Refuse(f"{pid}: {why}: look aborted")
    return dst, row


'''

PRECHECK_FN = '''def precheck_zips():
    """LCTSC-T23 (review S1): before GT_OPENED, by file metadata only (no byte of any RTSTRUCT is read): every look
    patient's stored zip exists with the size of its K_lctsc/download.json record and (real look) no extracted copy or
    partial extraction exists in data_lctsc/dicom; else the look is refused before it starts."""
    dl = json.load(open(C.DOWNLOAD))["files"]
    for pid in G.test_ids():
        if G.REHEARSAL:
            assert pid in T.STANDINS and pid not in T.ids() and not C.is_test(pid), pid
        rec, z = dl.get(f"{pid}_RTSTRUCT.zip"), ZDIR / f"{pid}_RTSTRUCT.zip"
        if rec is None or not z.is_file() or z.stat().st_size != rec["bytes"] or fault("precheck_size", pid):
            raise G.Refuse(f"{pid}: stored RTSTRUCT zip missing or its size differs from K_lctsc/download.json: no look")
        if not G.REHEARSAL and any((C.DICOM / n).exists() for n in (f"{pid}_RTSTRUCT", f".tmp_{pid}_RTSTRUCT")):
            raise G.Refuse(f"{pid}: an extracted RTSTRUCT copy exists in data_lctsc/dicom: no look")


'''

SPECS = {
    "t23_log.py": ("lctsc_log.py", DOC_LOG, "\nimport hashlib", [
        ('EVENTS = REPO / "outputs/logs/vccor/final_lctsc.log"', 'EVENTS = REPO / "outputs/logs/vccor/final_lctsc_t23.log"', 1),
        ('TAG = "lctsc"', 'TAG = "lctsc_t23"', 1),
        ("LCTSC_REHEARSAL_ROOT", "LCTSC_T23_REHEARSAL_ROOT", 3),
    ]),
    "t23_gates.py": ("lctsc_gates.py", DOC_GATES, "\nimport hashlib", [
        ('OUT = REPO / "outputs/results/vccor/final/K_lctsc"\n',
         'OUT = REPO / "outputs/results/vccor/final/K_lctsc_t23"\n'
         'SRC = REPO / "outputs/results/vccor/final/K_lctsc"           # the frozen LCTSC (d6ayj) records and models\n', 1),
        ('MODELS = OUT / "models"', 'MODELS = SRC / "models"              # the frozen d6ayj models: no training in LCTSC-T23', 1),
        ('os.environ.get("LCTSC_REHEARSAL_ROOT")', 'os.environ.get("LCTSC_T23_REHEARSAL_ROOT")', 1),
        ('if REHEARSAL and os.environ.get("LCTSC_REHEARSAL_NOW"):\n        return os.environ["LCTSC_REHEARSAL_NOW"]',
         'if REHEARSAL and os.environ.get("LCTSC_T23_REHEARSAL_NOW"):\n        return os.environ["LCTSC_T23_REHEARSAL_NOW"]', 1),
        ("LCTSC_REHEARSAL_NOW may set it", "LCTSC_T23_REHEARSAL_NOW may set it", 1),
        ('CUTOFF = "2026-11-19 18:00:00"\nSTOP = "2026-11-21 12:00:00"',
         'CUTOFF = "2026-11-20 10:00:00"\nSTOP = "2026-11-20 16:00:00"\nPUBLIC_DEADLINE = "2026-11-19 12:00:00"', 1),
        ("TS_EPOCHS = 50\n", 'TS_EPOCHS = 50\nDEPOSIT_RX = r"K_lctsc_t23/.*lctsc_t23.*\\.zip$"\n', 1),
        ('    dep = logged("DEPOSIT", r"lctsc.*\\.zip$")\n', '    dep = logged("DEPOSIT", DEPOSIT_RX)\n', 1),
        ('"upload gate: deposit sha256 does not match the logged LCTSC DEPOSIT line"',
         '"upload gate: deposit sha256 does not match the logged LCTSC-T23 DEPOSIT line"', 1),
        ('    zp = REPO / logged_path("DEPOSIT", r"lctsc.*\\.zip$")', '    zp = REPO / logged_path("DEPOSIT", DEPOSIT_RX)', 1),
        ('        raise Refuse("public gate: the OSF API did not report the registration public")\n',
         '        raise Refuse("public gate: the OSF API did not report the registration public")\n'
         '    if not ts(p["public_checked_kst"]) < ts(PUBLIC_DEADLINE):\n'
         '        raise Refuse("public gate: not checked public before the public deadline (Thu 2026-11-19 12:00 KST): no look")\n', 1),
        ('"stop rule passed (Sat 2026-11-21 12:00 KST): no look"', '"stop rule passed (Fri 2026-11-20 16:00 KST): no look"', 1),
        ('    """The look\'s patients: LCTSC\'s 24 test patients; in the rehearsal, the stand-in TRAINING patients of standin.json."""\n'
         '    if REHEARSAL:\n'
         '        return json.load(open(Path(REHEARSAL_ROOT) / "standin.json"))["ids"]\n'
         '    m = json.load(open(OUT / "tcia_manifest.json"))\n'
         '    return sorted({r["PatientID"] for r in m["series"] if r["PatientID"].startswith("LCTSC-Test-")})\n',
         '    """The look\'s patients: the 23 LCTSC-T23 patients of ids_t23.json; in the rehearsal, the stand-in TRAINING patients\n'
         '    of standin.json."""\n'
         '    if REHEARSAL:\n'
         '        return json.load(open(Path(REHEARSAL_ROOT) / "standin.json"))["ids"]\n'
         '    return list(json.load(open(OUT / "ids_t23.json"))["ids"])\n', 1),
        ("'NOTRUN lctsc <arm> ...' line", "'NOTRUN lctsc_t23 <arm> ...' line", 1),
        ('f" NOTRUN lctsc {arm} "', 'f" NOTRUN lctsc_t23 {arm} "', 1),
    ]),
    "t23_predict.py": ("lctsc_predict.py", None, None, [
        ('"""vccor/final LCTSC PREDICT (GPU 1; idempotent, resumable per patient): one arm (F0, F1b_S1..S3) on the 24 LCTSC test CTs.\n'
         'NO ground truth exists on disk before the look.',
         '"""vccor/final LCTSC-T23 PREDICT (DERIVED from the frozen lctsc_predict.py by derive_t23.py; GPU 1; idempotent,\n'
         'resumable per patient): one arm (F0, F1b_S1..S3 = the frozen d6ayj models) on the CTs of the 23 LCTSC-T23 patients.\n'
         'Their RTSTRUCT zips are on disk but are never extracted before the look.', 1),
        ("only after the freeze (lctsc_gates.freeze_ok)", "only after the freeze (t23_gates.freeze_ok)", 1),
        ("started only before the cutoff (Thu 2026-11-19 18:00 KST)", "started only before the cutoff (Fri 2026-11-20 10:00 KST)", 1),
        ("refuses any LCTSC-Test id and any --images folder inside data_lctsc/images.",
         "refuses any LCTSC-Test id, any LCTSC-T23 id and any --images folder inside data_lctsc/images.", 1),
        ("usage:  run only through lctsc_predict_queue.py (unit ssl-lctsc-predict)",
         "usage:  run only through t23_predict_queue.py (unit ssl-lctsc-t23-predict)", 1),
        ("Written by an AI coding assistant (Claude) on the owner's instruction of 2026-11-13.",
         "Written by an AI coding assistant (Claude) on the owner's instruction of 2026-11-13; LCTSC-T23 derivation 2026-11-17.", 1),
        ('for _p in (str(HERE), str(HERE.parent), str(HERE.parent.parent / "tsft_full")):',
         'for _p in (str(HERE), str(HERE.parent / "lctsc"), str(HERE.parent), str(HERE.parent.parent / "tsft_full")):', 1),
        ("import lctsc_gates as G  # noqa: E402\n", "import t23_gates as G  # noqa: E402\nimport t23_common as T  # noqa: E402\n", 1),
        ('assert not any(C.is_test(c) for c in ids), "smoke never runs on an LCTSC test patient"',
         'assert not any(C.is_test(c) or c in T.ids() for c in ids), "smoke never runs on an LCTSC test or LCTSC-T23 patient"', 1),
        ('"cutoff passed (Thu 2026-11-19 18:00 KST): no prediction is started"',
         '"cutoff passed (Fri 2026-11-20 10:00 KST): no prediction is started"', 1),
        ('I.wait_disk(0.5, f"prediction {arm}", I.CUTOFF)', 'I.wait_disk(0.5, f"prediction {arm}", T.CUTOFF_DT)', 1),
        ('_LOCK["fd"] = I.gpu_session(I.CUTOFF)', '_LOCK["fd"] = I.gpu_session(T.CUTOFF_DT)', 1),
        ('root, ids, image = G.PRED, C.patients("test"), C.image_path', 'root, ids, image = G.PRED, G.test_ids(), C.image_path', 1),
        ('for f in ("lctsc_predict.py", "lctsc_gates.py", "lctsc_common.py")',
         'for f in ("t23_predict.py", "t23_gates.py", "t23_common.py")', 1),
    ]),
    "t23_predict_queue.py": ("lctsc_predict_queue.py", DOC_QUEUE, "\nimport argparse", [
        ("sys.path.insert(0, str(HERE))\nimport lctsc_gates as G  # noqa: E402\nimport lctsc_infra as I  # noqa: E402\n"
         "import lctsc_log as L  # noqa: E402\nimport lctsc_runs as R  # noqa: E402\n",
         'sys.path.insert(0, str(HERE.parent / "lctsc"))\nsys.path.insert(0, str(HERE))\nimport t23_gates as G  # noqa: E402\n'
         "import lctsc_infra as I  # noqa: E402\nimport t23_log as L  # noqa: E402\nimport t23_common as T  # noqa: E402\n", 1),
        ('str(HERE / "lctsc_predict.py")', 'str(HERE / "t23_predict.py")', 1),
        ('I.wait_disk(0.5, f"prediction {arm}", I.CUTOFF, log=L.event)', 'I.wait_disk(0.5, f"prediction {arm}", T.CUTOFF_DT, log=L.event)', 1),
        ("if not I.backoff(I.CUTOFF):", "if not I.backoff(T.CUTOFF_DT):", 1),
        ('"not DONE by the cutoff (Thu 2026-11-19 18:00 KST)"', '"not DONE by the cutoff (Fri 2026-11-20 10:00 KST)"', 1),
    ]),
    "t23_look_labels.py": ("lctsc_look_labels.py", DOC_LOOK, "\nimport argparse", [
        ("sys.path.insert(0, str(HERE))\nimport lctsc_common as C  # noqa: E402\nimport lctsc_gates as G  # noqa: E402\n"
         "import lctsc_log as L  # noqa: E402\n",
         'sys.path.insert(0, str(HERE.parent / "lctsc"))\nsys.path.insert(0, str(HERE))\nimport lctsc_common as C  # noqa: E402\n'
         "import t23_gates as G  # noqa: E402\nimport t23_log as L  # noqa: E402\nimport t23_common as T  # noqa: E402\n", 1),
        ('API = "https://services.cancerimagingarchive.net/nbia-api/services/v1"',
         "API = None                           # LCTSC-T23: no download in the look", 1),
        ('ZDIR = (Path(os.environ["LCTSC_REHEARSAL_ROOT"]) / "zips") if G.REHEARSAL else C.DATA / "test_rtstruct_zips"',
         "ZDIR = C.ZIPS                        # the training RTSTRUCT zips of 2026-11-13 (K_lctsc/download.json); read only", 1),
        ('os.environ.get("LCTSC_REHEARSAL_FAULTS", "")', 'os.environ.get("LCTSC_T23_REHEARSAL_FAULTS", "")', 1),
        ('" GT_ABORTED lctsc "', '" GT_ABORTED lctsc_t23 "', 1),
        ('" NOTE lctsc look not completed"', '" NOTE lctsc_t23 look not completed"', 1),
        ("assert len(rows) == 1 and (C.is_test(pid) or G.REHEARSAL), pid",
         "assert len(rows) == 1 and (pid in T.ids() or G.REHEARSAL), pid", 1),
        ('r"K_lctsc/(rehearsal_\\d+/)?look/GT_OPENED$"', 'r"K_lctsc_t23/(rehearsal_\\d+/)?look/GT_OPENED$"', 1),
        ('f"lctsc look: test RTSTRUCTs are requested only after this line; arms DONE {done}; "',
         'f"lctsc_t23 look: the 23 RTSTRUCT zips are extracted only after this line; arms DONE {done}; "', 1),
        ("f\"lctsc look: {len(rec['patients'])} test labels converted (lctsc_raster); missing structures \"",
         "f\"lctsc_t23 look: {len(rec['patients'])} labels converted (lctsc_raster); missing structures \"", 1),
        ("def gates(cont=False):\n", PRECHECK_FN + "def gates(cont=False):\n", 1),
        ("    G.gate_public()\n    return arm_states()\n", "    G.gate_public()\n    precheck_zips()\n    return arm_states()\n", 1),
        ('f"test label {pid}: id out of range or grid differs from the CT"', 'f"label {pid}: id out of range or grid differs from the CT"', 1),
        ('    if not a.cont:\n        maps = {"arms_done": done, "not_run": not_run, "maps": maps_record(done), "written": G.now()}\n',
         '    if not a.cont:\n        maps = {"arms_done": done, "not_run": not_run, "maps": maps_record(done), "written": G.now()}\n'
         '        if not G.before_stop():                   # LCTSC-T23 (review N3): no GT_OPENED at or after the stop rule\n'
         '            raise G.Refuse("stop rule passed (Fri 2026-11-20 16:00 KST): no look")\n', 1),
    ]),
    "t23_score.py": ("lctsc_score.py", DOC_SCORE, "\nimport argparse", [
        ('for _p in (str(HERE), str(HERE.parent), str(HERE.parent / "flaret50"), str(HERE.parent.parent / "m2")):',
         'for _p in (str(HERE), str(HERE.parent / "lctsc"), str(HERE.parent), str(HERE.parent / "flaret50"),\n'
         '           str(HERE.parent.parent / "m2")):', 1),
        ("import lctsc_gates as G  # noqa: E402\nimport lctsc_log as L  # noqa: E402\n",
         "import t23_gates as G  # noqa: E402\nimport t23_log as L  # noqa: E402\n", 1),
        ('os.environ.get("LCTSC_REHEARSAL_FAULTS", "")', 'os.environ.get("LCTSC_T23_REHEARSAL_FAULTS", "")', 1),
        ("(LCTSC_REHEARSAL_FAULTS 'kind:n')", "(LCTSC_T23_REHEARSAL_FAULTS 'kind:n')", 1),
        ('r"K_lctsc/(rehearsal_\\d+/)?look/results\\.json$"', 'r"K_lctsc_t23/(rehearsal_\\d+/)?look/results\\.json$"', 1),
        ('"no GT_OPENED marker: run lctsc_look_labels.py first"', '"no GT_OPENED marker: run t23_look_labels.py first"', 1),
        ('if any(" GT_ABORTED " in line and "lctsc" in line for line in open(G.PREREG_LOG)):',
         'if any(" GT_ABORTED lctsc_t23 " in line for line in open(G.PREREG_LOG)):', 1),
        ('r"K_lctsc/(rehearsal_\\d+/)?look/labels\\.json$"', 'r"K_lctsc_t23/(rehearsal_\\d+/)?look/labels\\.json$"', 1),
        ("f\"lctsc: P1 {r['P1']['reading']} M {r['P1']['M']:+.2f} \"",
         "f\"lctsc_t23: P1_HF {r['P1_HF']['reading']} M {r['P1_HF']['M']:+.2f} \"", 1),
        ("    import lctsc_analysis as A\n", "    import t23_analysis as A\n", 1),
        ('for f in ("lctsc_score.py", "lctsc_analysis.py", "lctsc_gates.py",\n'
         '                                                                    "lctsc_common.py", "lctsc_look_labels.py")',
         'for f in ("t23_score.py", "t23_analysis.py", "t23_gates.py",\n'
         '                                                                    "t23_common.py", "t23_look_labels.py")', 1),
        ('    p1 = res["P1"]\n    L.prereg_line("RESULTS", G.LOOK / "results.json",\n'
         "                  f\"lctsc: P1 {p1['reading']} M {p1['M']:+.2f}; {res['P1_combination']['combination']}; organs \"\n"
         "                  f\"{res['organ_costs']['reading']}\")",
         '    p1, s4 = res["P1_HF"], res["S_4S"]\n    L.prereg_line("RESULTS", G.LOOK / "results.json",\n'
         "                  f\"lctsc_t23: P1_HF {p1['reading']} M {p1['M']:+.2f}; S_4S {s4['reading']} M {s4['M']:+.2f}; \"\n"
         "                  f\"per-draw P1_HF {res['P1_HF_combination_descriptive']['combination']} (descriptive); organs \"\n"
         "                  f\"{res['organ_costs']['reading']}\")", 1),
    ]),
}


def main():
    if (REPO / "outputs/results/vccor/final/K_lctsc_t23/code_freeze.json").exists():     # review S4
        raise SystemExit("an LCTSC-T23 freeze exists: no re-derivation")
    fz = json.load(open(LCTSC_FREEZE))["files"]
    rec = {"derived_kst": None, "sources": {}, "outputs": {}}
    for out, (src, doc, marker, reps) in SPECS.items():
        sp = SRC_DIR / src
        key = f"scripts/vccor/final/lctsc/{src}"
        if fz.get(key) != sha(sp):
            raise SystemExit(f"{src}: differs from the LCTSC freeze")
        text = sp.read_text()
        if doc is not None:
            text = head(text, doc, marker)
        if out == "t23_look_labels.py":
            i, j = text.index("def download_part(pid, row, part):"), text.index("def convert(pid, zpath):")
            text = text[:i] + NEW_FETCH + text[j:]
        text = apply(text, reps, out)
        (HERE / out).write_text(text)
        rec["sources"][src] = sha(sp)
        rec["outputs"][out] = sha(HERE / out)
        print(f"{out} <- {src}: {len(reps)} replacement rule(s)" + (" + docstring" if doc else "")
              + (" + fetch() replaced (no download)" if out == "t23_look_labels.py" else ""))
    import time
    rec["derived_kst"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (HERE / "derived.json").write_text(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
