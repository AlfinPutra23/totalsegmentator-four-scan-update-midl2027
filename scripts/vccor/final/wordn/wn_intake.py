"""vccor/final WORD-N INTAKE (CPU).

  --pin   BEFORE the registration: the 7z technical listing (-slt) of the 40 whitelisted members (imagesVal and labelsVal of
          the 20 validation scans): size, packed size, modified time, method, encryption. Central-directory metadata only;
          no member content is decrypted or read. The archive's size and sha256 are checked against C_word/download.json.
          Writes I_wordn/archive_members.json (pinned by the freeze). The AES-zip members carry no CRC (AE-2); the
          authentication code that 7z checks on extraction, the pinned member sizes and the archive's sha256 pin them.
  --run   AFTER the gate_osf_uploaded.json record (the runner's first job): the 20 validation IMAGES are extracted
          from the archive with a member whitelist (imagesVal only; every member logged), each checked against its pinned
          size, hashed into intake/images_record.json; then the registered duplicate rule on the images (Plan C / A1's
          rule extended to the validation scans, fixed before registration): every validation scan is compared with
          every other validation scan and with the 12 draw and 30 test scans whose image shape is identical, by the
          Pearson r of the intensities on a 4-voxel stride grid; r >= 0.999 -> the same image: a validation scan that
          duplicates a draw or test scan is dropped (reported), and of two duplicate validation scans the higher id is
          dropped. Writes intake/exclusions.json (decision INTAKE_PASS) and an INTAKE line. No label member is touched.

usage:  .venv/bin/python -I -B scripts/vccor/final/wordn/wn_intake.py --pin    (before registration; done 2026-11-13)
        --run: runner only (wn_run.py's first job, after the upload gate)

Version of 2026-11-13.
"""
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import wn_common as T  # noqa: E402

W = T.W


def pin():
    if T.MEMBERS.exists():
        raise T.Refuse(f"{T.rel(T.MEMBERS)} exists: the pin is made once")
    if T.look_made():
        raise T.Refuse("a look exists")
    T.archive_ok()
    names = [T.member(k, c) for c in T.VAL_IDS for k in ("imagesVal", "labelsVal")]
    lst = T.list_members(names)
    bad = [n for n, v in lst.items() if v.get("Encrypted") != "+"]
    rec = {"what": "7z -slt central-directory listing of the 40 whitelisted WORD validation members (no content read)",
           "archive": T.rel(T.ARCHIVE), "archive_sha256": T.sha256_cached(T.ARCHIVE),
           "archive_bytes": os.stat(T.ARCHIVE).st_size, "members": lst,
           "sizes": {n: int(v["Size"]) for n, v in lst.items()}, "not_encrypted": bad, "written": T.now(),
           "code_sha256": T.code_hashes(["wn_intake.py", "wn_common.py"])}
    T.write_json(T.MEMBERS, rec)
    T.log_event(f"archive members pinned: {len(lst)} members, archive sha256 {rec['archive_sha256'][:16]}")
    return rec


def _stride(cid, cache):
    import nibabel as nib
    import numpy as np
    if cid not in cache:
        cache[cid] = np.asarray(nib.load(str(T.image_path(cid))).dataobj)[::4, ::4, ::4].astype(np.float64).ravel()
    return cache[cid]


def duplicates(val_ids, other_ids, log=print):
    """The registered duplicate rule (see the module docstring) -> (pairs, excluded {cid: reason})."""
    import nibabel as nib
    import numpy as np
    shape = {c: tuple(nib.load(str(T.image_path(c))).shape[:3]) for c in list(val_ids) + list(other_ids)}
    pairs, excl, cache = [], {}, {}
    cands = []
    vs = sorted(val_ids)
    for i, a in enumerate(vs):
        for b in vs[i + 1:]:
            if shape[a] == shape[b]:
                cands.append((a, b, "val-val"))
        for b in sorted(other_ids):
            if shape[a] == shape[b]:
                cands.append((a, b, "val-draw" if b in set(W.draw_scans()) else "val-test"))
    for a, b, kind in cands:
        x, y = _stride(a, cache), _stride(b, cache)
        r = float(np.corrcoef(x, y)[0, 1])
        same = r >= T.DUP_R
        pairs.append({"pair": [a, b], "kind": kind, "r_stride4": r, "same_image": same})
        if same:
            if kind == "val-val":
                excl.setdefault(b, f"duplicate of validation scan {a} (r {r:.5f})")
            else:
                excl.setdefault(a, f"duplicate of {kind.split('-')[1]} scan {b} (r {r:.5f})")
    log(f"duplicate rule: {len(cands)} same-shape pairs checked; excluded {excl}")
    return pairs, excl


def run():
    T.kst_gate()
    if T.look_made():
        raise T.Refuse("a look exists")
    if not T.REHEARSAL:
        T.freeze_ok()
        T.gate_uploaded()
    if T.EXCLUSIONS.is_file():
        h = T.sha256(T.EXCLUSIONS)
        if not any(f"INTAKE sha256 {h} " in ln for ln in T.prereg_lines()):
            T.prereg_line("INTAKE", T.EXCLUSIONS, "wordn intake (line written on rerun: the first run stopped before it)")
        print(f"{T.rel(T.EXCLUSIONS)} exists: intake DONE")
        return 0
    T.archive_ok()
    pinned = json.load(open(T.MEMBERS))
    if pinned["archive_sha256"] != T.sha256_cached(T.ARCHIVE):
        raise T.Refuse("the archive differs from the pinned listing")
    T.wait_disk(1.5, "WORD-N image extraction")
    Path(T.IMAGES).mkdir(parents=True, exist_ok=True)
    rec = {}
    import nibabel as nib
    for c in T.VAL_IDS:
        m = T.member("imagesVal", c)
        p = T.image_path(c)
        if p.is_file() and p.stat().st_size == pinned["sizes"][m]:
            T.log_event(f"[images] {m} present ({p.stat().st_size} bytes)")
        else:
            b = T.extract_member_bytes(m)
            if len(b) != pinned["sizes"][m]:
                raise T.Refuse(f"{m}: {len(b)} bytes, pinned {pinned['sizes'][m]}")
            tmp = p.with_name(f".{p.name}.tmp{os.getpid()}")
            tmp.write_bytes(b)
            os.replace(tmp, p)
            T.log_event(f"[images] extracted {m} ({len(b)} bytes)")
        im = nib.load(str(p))
        rec[c] = {"member": m, "sha256": T.sha256(p), "bytes": p.stat().st_size, "shape": list(im.shape[:3]),
                  "spacing": [float(z) for z in im.header.get_zooms()[:3]]}
    T.write_json(T.IMAGES_RECORD, {"images": rec, "written": T.now(), "archive_sha256": pinned["archive_sha256"],
                                   "code_sha256": T.code_hashes(["wn_intake.py", "wn_common.py"])})
    others = sorted(set(W.draw_scans()) | set(W.split()["test"]))
    miss = [c for c in others if not T.image_path(c).is_file()]
    if miss:
        raise T.Refuse(f"draw / test images missing for the duplicate rule: {miss}")
    pairs, excl = duplicates(T.VAL_IDS, others, log=T.log_event)
    ex = {"decision": "INTAKE_PASS", "rule": f"Pearson r of the intensities on a 4-voxel stride grid >= {T.DUP_R} between "
                                             "same-shape images (validation vs validation / draw / test)",
          "pairs": pairs, "excluded": excl, "included": [c for c in T.VAL_IDS if c not in excl],
          "sealed_twin_note": "0083's header twin 0095 is one of the 88 sealed training scans: never extracted or used",
          "images_record_sha256": T.sha256(T.IMAGES_RECORD), "written": T.now(),
          "code_sha256": T.code_hashes(["wn_intake.py", "wn_common.py"])}
    T.write_json(T.EXCLUSIONS, ex, mode=0o444)
    os.chmod(T.IMAGES_RECORD, 0o444)
    T.prereg_line("INTAKE", T.EXCLUSIONS, f"wordn intake: 20 validation images extracted (imagesVal whitelist) and the "
                                          f"duplicate rule applied; excluded {sorted(excl) or 'none'}")
    return 0


def main():
    a = sys.argv[1:]
    try:
        if a == ["--pin"]:
            pin()
            return 0
        if a == ["--run"]:
            return run()
        raise SystemExit(__doc__)
    except T.Refuse as e:
        print(f"[REFUSE] {e.msg}", file=sys.stderr)
        return e.code


if __name__ == "__main__":
    sys.exit(main())
