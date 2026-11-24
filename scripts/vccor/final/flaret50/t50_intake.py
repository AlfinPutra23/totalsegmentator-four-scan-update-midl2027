"""vccor/final FLARE-T50 INTAKE (CPU; network).

  --metadata  Public metadata only, no login and no file download: the Hub's tree listing of
              FLARE-MedFM/FLARE-Task2-LaptopSeg at the pinned commit (t50_common.HF_REVISION) for
              validation/Validation-Public-Images (50 images) and validation/Validation-Public-Labels (50 labels): path,
              size and the git oid of each file's LFS pointer, plus the dataset card's sha256 and the repo's gating and
              licence. Written to I_flaret50/hf_manifest.json (read-only) with a MANIFEST line in final_prereg.log. The
              pointer oid pins each file's content (sha256 + size) before any byte of it is downloaded; the label oids are
              therefore fixed before registration without reading a label.
  --images    IMAGES ONLY (before registration; disclosed): needs the author's own Hugging Face login (the author accepts
              the dataset's access conditions and runs 'hf auth login'; this script never asks for, prints or writes a
              token). Downloads exactly the 50 manifest image files (allow-list by exact path under
              validation/Validation-Public-Images/; any other path refuses) at the pinned commit into
              data_flaret50/hf/, checks each file's size and that its sha256 reproduces the manifest's pointer oid, and
              writes I_flaret50/images_record.json (read-only; per-image sha256, size, shape, spacing, load status) with an
              IMAGES line. Refuses if any label file or label directory exists under data_flaret50/, or if / would fall
              below 35 GiB free.

usage:  .venv/bin/python -I -B scripts/vccor/final/flaret50/t50_intake.py --metadata
        .venv/bin/python -I -B scripts/vccor/final/flaret50/t50_intake.py --images

Version of 2026-11-09.
"""
import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import t50_common as T  # noqa: E402

os.environ.update(T.HF_ENV)          # before huggingface_hub is imported (telemetry off; no Xet chunk cache)
API = "https://huggingface.co/api/datasets"
UA = {"User-Agent": "vccor-flaret50-intake/1.0"}


def _get(url, timeout=120):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


def listing(prefix):
    url = f"{API}/{T.HF_REPO}/tree/{T.HF_REVISION}/{prefix.rstrip('/')}?recursive=1"
    st, body = _get(url)
    if st != 200:
        raise T.Refuse(f"tree listing {url}: HTTP {st}")
    return [e for e in json.loads(body) if e.get("type") == "file"]


def metadata():
    if T.MANIFEST.exists():
        raise T.Refuse(f"{T.rel(T.MANIFEST)} exists (read-only record); not rewritten")
    st, info = _get(f"{API}/{T.HF_REPO}/revision/{T.HF_REVISION}")
    info = json.loads(info)
    if info.get("sha") != T.HF_REVISION:
        raise T.Refuse(f"revision {T.HF_REVISION} not found (got {info.get('sha')})")
    imgs = {e["path"]: e for e in listing(T.HF_IMG_PREFIX)}
    lbls = {e["path"]: e for e in listing(T.HF_LBL_PREFIX)}
    out = {"repo": T.HF_REPO, "revision": T.HF_REVISION, "gated": info.get("gated"),
           "license": (info.get("cardData") or {}).get("license"), "last_modified": info.get("lastModified"),
           "images": {}, "labels": {}, "fetched": T.now(),
           "how": "public Hub API tree listing at the pinned revision, no login; 'pointer_oid' is the git oid of the LFS "
                  "pointer (sha1 of the pointer text 'version ... oid sha256:<content sha256> size <bytes>'), which fixes "
                  "the content before download"}
    for cid in T.IDS:
        pi, pl = f"{T.HF_IMG_PREFIX}{cid}_0000.nii.gz", f"{T.HF_LBL_PREFIX}{cid}.nii.gz"
        if pi not in imgs or pl not in lbls:
            raise T.Refuse(f"{cid}: image or label missing from the listing")
        for key, e in (("images", imgs[pi]), ("labels", lbls[pl])):
            if not e.get("lfs"):
                raise T.Refuse(f"{e['path']}: not an LFS file")
            out[key][cid] = {"path": e["path"], "size": int(e["size"]), "pointer_oid": e["oid"]}
    if len(imgs) != 50 or len(lbls) != 50:
        raise T.Refuse(f"expected 50 images and 50 labels, listing has {len(imgs)} and {len(lbls)}")
    st, card = _get(f"https://huggingface.co/datasets/{T.HF_REPO}/resolve/{T.HF_REVISION}/README.md")
    import hashlib
    out["dataset_card_sha256"] = hashlib.sha256(card).hexdigest()
    out["dataset_card_says"] = [ln.strip() for ln in card.decode("utf-8", "replace").splitlines()
                                if "validation" in ln.lower() and ("merged" in ln.lower() or "public" in ln.lower())][:4]
    out["image_bytes_total"] = sum(v["size"] for v in out["images"].values())
    out["label_bytes_total"] = sum(v["size"] for v in out["labels"].values())
    T.write_json(T.MANIFEST, out, mode=0o444)
    T.prereg_line("MANIFEST", T.MANIFEST, f"flaret50 public Hub metadata at revision {T.HF_REVISION}: 50 image and 50 "
                                          f"label pointer oids, sizes; no file downloaded")
    print(f"images {out['image_bytes_total'] / 1e9:.3f} GB, labels {out['label_bytes_total'] / 1e6:.1f} MB")


def no_labels_on_disk():
    if T.LBL_DIR.exists():
        raise T.Refuse(f"{T.rel(T.LBL_DIR)} exists before the look")
    if T.DATA.exists():
        for root, dirs, files in os.walk(T.DATA):
            for n in dirs + files:
                if "Label" in n or n.endswith("Labels"):
                    raise T.Refuse(f"label-like path under {T.rel(T.DATA)}: {os.path.join(root, n)}")


def images():
    import hashlib
    from huggingface_hub import get_token, hf_hub_download
    m = T.load_manifest()
    if T.IMAGES_RECORD.exists():
        raise T.Refuse(f"{T.rel(T.IMAGES_RECORD)} exists (read-only record)")
    if get_token() is None:
        raise T.Refuse("no Hugging Face login on this machine: the owner logs in and accepts the dataset's access "
                       "conditions first ('hf auth login'); this script never handles a token")
    no_labels_on_disk()
    need = sum(v["size"] for v in m["images"].values()) / 1024 ** 3 + 0.5
    T.disk_gate(need, "FLARE-T50 image download")
    T.HF_LOCAL.mkdir(parents=True, exist_ok=True)
    rec = {"revision": T.HF_REVISION, "started": T.now(), "images": {}, "versions": T.versions(),
           "code_sha256": T.code_hashes(["t50_intake.py", "t50_common.py"]), "hf_env": T.HF_ENV}
    import nibabel as nib
    for cid in T.IDS:
        e = m["images"][cid]
        assert e["path"].startswith(T.HF_IMG_PREFIX) and e["path"].endswith("_0000.nii.gz"), e["path"]
        p = Path(hf_hub_download(repo_id=T.HF_REPO, repo_type="dataset", filename=e["path"], revision=T.HF_REVISION,
                                 local_dir=str(T.HF_LOCAL)))
        if p.resolve() != T.image_path(cid).resolve():
            raise T.Refuse(f"{cid}: downloaded to {p}, expected {T.image_path(cid)}")
        size = p.stat().st_size
        h = T.sha256(p)
        ok_size, ok_oid = size == e["size"], T.pointer_oid(h, size) == e["pointer_oid"]
        r = {"sha256": h, "size": size, "size_ok": ok_size, "pointer_oid_ok": ok_oid}
        try:
            im = nib.load(str(p))
            r.update(load_ok=True, shape=[int(x) for x in im.shape[:3]],
                     spacing=[float(x) for x in im.header.get_zooms()[:3]])
        except Exception as ex:                       # noqa: BLE001  (an unloadable file is an exclusion, recorded)
            r.update(load_ok=False, load_error=f"{type(ex).__name__}: {ex}")
        rec["images"][cid] = r
        T.log_event(f"image {cid}: {size / 1e6:.1f} MB sha256 {h[:16]} size_ok {ok_size} pointer_ok {ok_oid} "
                    f"load_ok {r['load_ok']}")
        if not (ok_size and ok_oid):
            raise T.Refuse(f"{cid}: downloaded image does not match the pinned listing")
    no_labels_on_disk()
    rec["finished"] = T.now()
    rec["all_pointer_oids_match"] = all(v["pointer_oid_ok"] for v in rec["images"].values())
    rec["unloadable"] = [c for c, v in rec["images"].items() if not v.get("load_ok")]
    T.write_json(T.IMAGES_RECORD, rec, mode=0o444)
    T.prereg_line("IMAGES", T.IMAGES_RECORD, "flaret50 images-only download (pre-registration, disclosed): 50 images, "
                                             "each sha256 reproduces its pinned pointer oid; no label downloaded")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--metadata", action="store_true")
    g.add_argument("--images", action="store_true")
    a = ap.parse_args()
    try:
        T.kst_gate()
        metadata() if a.metadata else images()
    except T.Refuse as e:
        print(f"[REFUSE] {e.msg}", file=sys.stderr)
        return e.code
    return 0


if __name__ == "__main__":
    sys.exit(main())
