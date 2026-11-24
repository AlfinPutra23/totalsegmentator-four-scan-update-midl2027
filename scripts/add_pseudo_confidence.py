"""Add a per-voxel TEACHER-CONFIDENCE field to each flare_pseudo/{cid}.h5 (for soft/weighted distillation).

The frozen teacher (TotalSeg) is reliable in organ INTERIORS and error-prone at BOUNDARIES. We can read that
off the hard pseudo-label alone: conf(v) = 1 - exp(-dist_to_boundary(v)/d0)  ∈ [0,1], ~0 at label boundaries,
~1 deep inside organs/background. Weighting the distillation loss by conf lets the student TRUST the teacher's
confident interiors while relying on its 4 real GT labels + SSL at boundaries -- the regions where the student
can EXCEED the teacher (our test-split analysis: student already beats teacher on R-adrenal/esophagus).

Non-destructive: only ADDS 'conf' to each flare_pseudo/{cid}.h5. Run: .venv/bin/python scripts/add_pseudo_confidence.py
"""
import glob, numpy as np, h5py
from scipy import ndimage as ndi

D0 = 3.0   # voxels; confidence saturates ~3 voxels from any boundary


def confidence(pseudo):
    boundary = ndi.grey_dilation(pseudo, size=3) != ndi.grey_erosion(pseudo, size=3)
    dist = ndi.distance_transform_edt(~boundary)
    return (1.0 - np.exp(-dist / D0)).astype(np.float32)


def main():
    files = sorted(glob.glob("MagicNet/data/flare_pseudo/*.h5"))
    print(f"adding 'conf' to {len(files)} pseudo files (d0={D0} vox)")
    for f in files:
        with h5py.File(f, "r") as h:
            pseudo = h["pseudo"][:]
            has = "conf" in h
        conf = confidence(pseudo)
        with h5py.File(f, "a") as h:
            if has:
                del h["conf"]
            h.create_dataset("conf", data=conf, compression="gzip")
    # quick sanity on one case
    with h5py.File(files[0], "r") as h:
        c = h["conf"][:]; p = h["pseudo"][:]
        print(f"sample {files[0].split('/')[-1]}: conf range [{c.min():.2f},{c.max():.2f}] mean={c.mean():.2f} "
              f"| mean conf inside organs={c[p>0].mean():.2f} at fg-boundary~low")
    print("done: 'conf' written to all flare_pseudo files.")


if __name__ == "__main__":
    main()
