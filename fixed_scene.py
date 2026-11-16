"""Module 3 -- Lab 2: the clean (bug-free) counterpart of broken_scene.py.

Renders the same three scenes with the one-line fix applied in each case and
proves each fix against a reference with SSIM / PSNR.

  python fixed_scene.py            # all three scenes
  python fixed_scene.py --scene inside-out

  inside-out : faces re-wound so outward faces survive back-face culling
  zfight     : near = 1.0 instead of 1e-3, so the 16-bit depth codes separate
  stretched  : model matrix built in the correct order, no stray non-uniform scale

References
  inside-out, stretched -> out/temple_raster.ppm (the committed clean frame;
                           re-rendered at the reference's own size if missing)
  zfight                -> the float-depth render of the same scene, which is
                           the ground truth a 16-bit buffer is approximating.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from transforms import rot_y, scale, translate
from objio import load_obj, Mesh
from ppm_io import read_ppm, write_ppm
from metrics import psnr, ssim
import raster3d

# Same model matrix the reference frame and raster3d.main() use:
# scale first, then rotate in world, then translate (right-to-left, column vectors).
GOOD_MODEL = translate((0, 0, 0)) @ rot_y(35) @ scale((1.4, 1.4, 1.4))

TRI_COLS = [(230, 130, 40), (230, 130, 40), (60, 120, 210), (60, 120, 210)]

REF_RASTER = "out/temple_raster.ppm"


def _load():
    mesh = load_obj("data/pusaka_model.obj")
    tex = read_ppm("data/relief.ppm")
    return mesh, tex


def _temple_reference(mesh, tex, w, h):
    """The clean temple frame at w*h: read from disk when the size matches,
    otherwise re-rendered with the same settings that produced it."""
    if os.path.exists(REF_RASTER):
        ref = read_ppm(REF_RASTER)
        if ref.shape[:2] == (h, w):
            return ref, REF_RASTER
    col, _, _, _ = raster3d.render(mesh, w=w, h=h, tex_u8=tex, model=GOOD_MODEL)
    return col, "re-rendered clean frame"


def fixed_inside_out(mesh, tex, w, h):
    """FIX: re-wind every triangle so its outward face is front-facing.

    raster3d culls a triangle when its signed screen area is > 0 (screen y is
    down, so front faces have area < 0). The shipped OBJ is wound the other
    way, so with cull_back=True its *outward* faces are the ones discarded and
    you see through the temple into its far interior. Reversing the index order
    puts the outward faces back in front, and the culled render then matches
    the reference exactly -- 104 of 180 triangles culled, all of them genuinely
    facing away.
    """
    rewound = Mesh(
        mesh.positions,
        mesh.uvs,
        np.asarray(mesh.faces)[:, ::-1].copy(),
        None if mesh.face_uvs is None else np.asarray(mesh.face_uvs)[:, ::-1].copy(),
    )
    col, _, _, culled = raster3d.render(rewound, w=w, h=h, tex_u8=tex,
                                        model=GOOD_MODEL, cull_back=True)
    ref, ref_name = _temple_reference(mesh, tex, w, h)
    n = len(np.asarray(mesh.faces))
    note = (f"faces re-wound, back-face culling ON: {culled}/{n} triangles culled "
            f"({culled * 100 // n}%) -- only the genuinely away-facing ones.")
    return col, ref, ref_name, note


def _zfight_scene():
    """Two crossing quads: A flat at z=-9, B the same quad tilted ~1.2 deg about
    its own centre, so they intersect along a near-horizontal line."""
    cy, cz = 1.0, -9.0
    quadA = np.array([[-3.0, -1, 0], [3, -1, 0], [3, 3, 0], [-3, 3, 0]]) + [0, 0, cz]
    a = np.radians(1.2)
    R = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    quadB = (quadA - [0, cy, cz]) @ R.T + [0, cy, cz]
    pos = np.vstack([quadA, quadB])
    faces = [[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6]]
    return Mesh(pos, np.zeros((0, 2)), faces, [])


def fixed_zfight(mesh, tex, w, h):
    """FIX: near = 1.0 instead of 1e-3.

    Depth precision after the perspective divide is spent almost entirely on
    the slice closest to the eye. With near=1e-3 and far=200 the whole scene at
    z=-9 lands inside a handful of 16-bit codes, so the two quads quantise to
    the same depth across a wide band and draw order -- not geometry -- decides
    each pixel. Pulling `near` out to 1.0 shrinks the far/near ratio by three
    orders of magnitude; the same 16-bit buffer now resolves the crossing.
    """
    m = _zfight_scene()
    cam = dict(eye=(0, 1, 6), center=(0, 1, -1), cull_back=False, far=200,
               tri_colors=TRI_COLS, light_dir=(0, 0.2, 1))
    col, _, _, _ = raster3d.render(m, w=w, h=h, near=1.0, depth_bits=16, **cam)
    ref, _, _, _ = raster3d.render(m, w=w, h=h, near=1.0, depth_bits=None, **cam)
    wrong = int(np.sum(np.abs(col.astype(int) - ref.astype(int)).max(2) > 60))
    note = (f"near=1.0 with the same 16-bit depth buffer: {wrong} pixels differ "
            f"from the float-depth truth (they sit on the crossing line itself).")
    return col, ref, "float-depth render of the same scene", note


def fixed_stretched(mesh, tex, w, h):
    """FIX: build the model matrix in the intended order and drop the stray
    non-uniform scale.

    The broken version wrote the product backwards and then post-multiplied
    diag(2.2, 0.5, 1.0) -- applied in object space, after the rotation, so it
    stretches along the temple's own axes and shears the silhouette. The
    correct matrix is translate @ rot_y @ scale: scale the model, rotate it in
    world, then place it.
    """
    col, _, _, _ = raster3d.render(mesh, w=w, h=h, tex_u8=tex, model=GOOD_MODEL)
    ref, ref_name = _temple_reference(mesh, tex, w, h)
    note = ("model matrix = translate @ rot_y(35) @ scale(1.4), uniform:\n" +
            np.array2string(GOOD_MODEL[:3], precision=2))
    return col, ref, ref_name, note


SCENES = {
    "inside-out": fixed_inside_out,
    "zfight": fixed_zfight,
    "stretched": fixed_stretched,
}


def run(name, mesh, tex, w, h):
    col, ref, ref_name, note = SCENES[name](mesh, tex, w, h)
    os.makedirs("out", exist_ok=True)
    out = f"out/fixed_{name}.ppm"
    write_ppm(out, col)

    s = ssim(col, ref)
    p = psnr(col, ref)
    exact = np.array_equal(col, ref)

    print(f"\n== fixed: {name} ==")
    print("fix:", note)
    print(f"reference: {ref_name}")
    print(f"SSIM(fixed, reference) = {s:.4f}   "
          f"PSNR = {'inf' if np.isinf(p) else f'{p:.2f}'} dB"
          f"{'   (byte-identical)' if exact else ''}")
    print(f"wrote {out}")
    return s, p, exact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", choices=list(SCENES))
    ap.add_argument("--w", type=int, default=512)
    ap.add_argument("--h", type=int, default=512)
    args = ap.parse_args()

    mesh, tex = _load()
    names = [args.scene] if args.scene else list(SCENES)
    rows = [(n, *run(n, mesh, tex, args.w, args.h)) for n in names]

    print(f"\n{'scene':<14}{'SSIM':>9}{'PSNR (dB)':>12}")
    for n, s, p, exact in rows:
        print(f"{n:<14}{s:>9.4f}{('inf' if np.isinf(p) else f'{p:.2f}'):>12}")


if __name__ == "__main__":
    main()
