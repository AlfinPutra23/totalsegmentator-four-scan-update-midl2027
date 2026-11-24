"""FLARE pseudo-labels from the frozen TotalSegmentator prior, aligned to the h5 frame (Lever #3 distillation).

The cached TotalSeg masks (outputs/results/seg_flare/{cid}.nii.gz, TS v2 'total' ids) already exist, so
NO TotalSegmentator re-run is needed. We reuse the proven alignment from gen_prior_seeds.py:
  recompute the v1 importance from the seg, resample to 1.5mm, template-match to the STORED v1 importance
  (= the flare_h5 crop frame) to recover the crop offset (peak ~1.0), then crop the resampled+remapped
  seg with that offset. Remap TS v2 ids -> FLARE ids.

Output: MagicNet/data/flare_pseudo/{cid}.h5  dataset 'pseudo' (uint8, FLARE ids 0-13), h5-aligned.
SELF-VALIDATION: FLARE is fully labeled, so we score Dice(pseudo, flare_h5 GT) per organ -> this is the
distillation teacher's ceiling. If a class Dice ~0, its TS id or the alignment is wrong.

Run: .venv/bin/python scripts/build_flare_pseudo.py   (CPU-only, minutes)
"""
import glob, os, numpy as np, nibabel as nib, h5py
from scipy import ndimage as ndi
from skimage.feature import match_template

SPACING = 1.5; K_MM = 4.0
IMPORTANCE_TS = [7, 19, 4, 15, 9, 8, 6]                      # MUST match original v1 build (for the template match)
# TotalSegmentator v2 'total' id -> FLARE id
TS_TO_FLARE = {1: 3, 2: 2, 3: 13, 4: 9, 5: 1, 6: 11, 7: 4, 8: 7, 9: 8, 15: 10, 19: 12, 52: 5, 63: 6}
FLARE = ["_", "liver", "Rkid", "spleen", "panc", "aorta", "IVC", "Radr", "Ladr", "gallb", "esoph", "stom", "duod", "Lkid"]
TAIL = [4, 7, 8, 9, 10, 12]
OUT = "MagicNet/data/flare_pseudo"


def spacing(aff): return np.sqrt((aff[:3, :3] ** 2).sum(0))
def resample(a, sp, lab): return ndi.zoom(a, np.asarray(sp, float) / SPACING, order=0 if lab else 1, mode='nearest')


def build_importance(seg, sp, k_mm=K_MM):
    grad = ndi.grey_dilation(seg, size=3) != ndi.grey_erosion(seg, size=3)
    prox = np.exp(-ndi.distance_transform_edt(~grad, sampling=sp) / k_mm)
    cw = np.ones_like(seg, np.float32); cw[np.isin(seg, IMPORTANCE_TS)] = 3.0
    return (cw * prox).astype(np.float32)


def remap(seg):
    out = np.zeros_like(seg, np.uint8)
    for ts, fl in TS_TO_FLARE.items():
        out[seg == ts] = fl
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    segs = sorted(glob.glob('outputs/results/seg_flare/*.nii.gz'))
    print(f"building {len(segs)} FLARE pseudo-labels -> {OUT}")
    dsc = {i: [] for i in range(1, 14)}                      # per-organ Dice vs GT
    ok = 0
    for sp in segs:
        cid = sp.split('/')[-1][:4]
        try:
            nii = nib.load(sp); seg = np.asarray(nii.dataobj).astype(np.int16); vs = spacing(nii.affine)
            imp_rs = resample(build_importance(seg, vs), vs, lab=False)
            with h5py.File(f'MagicNet/data/flare_importance/{cid}.h5', 'r') as f:
                stored = np.asarray(f['importance']).astype(np.float32)
            if any(s > r for s, r in zip(stored.shape, imp_rs.shape)):
                print(f'[skip] {cid}: template larger than recomputed'); continue
            cc = match_template(imp_rs, stored); off = np.unravel_index(np.argmax(cc), cc.shape); peak = float(cc.max())
            sl = tuple(slice(o, o + s) for o, s in zip(off, stored.shape))
            pseudo = remap(resample(seg, vs, lab=True)[sl])
            with h5py.File(f'{OUT}/{cid}.h5', 'w') as f:
                f.create_dataset('pseudo', data=pseudo, compression='gzip')
            # validate vs real GT (same h5 frame)
            with h5py.File(f'MagicNet/data/flare_h5/{cid}.h5', 'r') as f:
                gt = np.asarray(f['label'])
            line = []
            for oid in range(1, 14):
                p = pseudo == oid; g = gt == oid
                if g.sum() > 0:
                    d = 2 * (p & g).sum() / (p.sum() + g.sum() + 1e-6); dsc[oid].append(d); line.append(f"{FLARE[oid]}={d*100:.0f}")
            ok += 1
            print(f'[ok] {cid} peak={peak:.2f} shape={pseudo.shape} | ' + ' '.join(line))
        except Exception as e:
            print(f'[err] {cid}: {type(e).__name__}: {e}')
    print(f"\nDONE: {ok}/{len(segs)} pseudo-labels -> {OUT}")
    print("\n=== TEACHER QUALITY: mean Dice(pseudo, GT) per organ (n cases) ===")
    allm = []
    for oid in range(1, 14):
        if dsc[oid]:
            m = np.mean(dsc[oid]) * 100; allm.append(m)
            tag = " <-TAIL" if oid in TAIL else ""
            print(f"  {FLARE[oid]:7} : {m:5.1f}  (n={len(dsc[oid])}){tag}")
    tailm = np.mean([np.mean(dsc[o]) for o in TAIL if dsc[o]]) * 100
    print(f"\n  MEAN all13 = {np.mean(allm):.1f}   TAIL(6) = {tailm:.1f}")
    print("  (compare to base C1: all~81.0, tail~70.8. If teacher tail >> 70.8, distillation has headroom.)")


if __name__ == '__main__':
    main()
