# src/data_fieldlevel_healpix.py
import os
import re
import json
import argparse
import numpy as np
import healpy as hp
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

SHELL_RE = re.compile(r"Tb_shell(\d+)_(\d+)\.fits$")

def list_shell_fits(sim_dir: Path, shell_min: int, shell_max: int):
    files = []
    for fp in sim_dir.iterdir():
        if not fp.name.endswith(".fits"):
            continue
        m = SHELL_RE.match(fp.name)
        if not m:
            continue
        shell_in, shell_out = int(m.group(1)), int(m.group(2))
        if shell_in >= shell_min and shell_out <= shell_max:
            files.append((shell_in, shell_out, fp))
    files.sort(key=lambda t: (t[0], t[1]))
    return files

def process_one_sim(sim_dir: Path, outdir: Path, nside_out: int, asinh_k: float,
                    shell_min: int, shell_max: int, save_float32: bool):
    sim_id = int(sim_dir.name)

    fits_list = list_shell_fits(sim_dir, shell_min, shell_max)
    if not fits_list:
        return None

    maps = []
    meta_all = []

    mask_path = sim_dir / "octant_mask.fits"
    mask = None
    if mask_path.exists():
        mask_raw = hp.read_map(str(mask_path), dtype=np.float32, verbose=False)
        mask = (mask_raw > 0).astype(np.float32)
        if hp.get_nside(mask) != nside_out:
            mask = hp.ud_grade(mask, nside_out=nside_out, order_in="RING", order_out="RING").astype(np.float32)

    for shell_in, shell_out, fp in fits_list:
        try:
            # field 1 => DTB
            dTb = hp.read_map(str(fp), field=1, verbose=False)
            dTb = np.nan_to_num(dTb, nan=0.0)

            # downgrade if needed
            if hp.get_nside(dTb) != nside_out:
                dTb = hp.ud_grade(dTb, nside_out=nside_out, order_in="RING", order_out="RING")
              
            if mask is not None:
                dTb = dTb * mask

            # asinh scaling 
            if asinh_k > 0:
                dTb = np.arcsinh(dTb / asinh_k)

            dTb = dTb.astype(np.float32) if save_float32 else dTb.astype(np.float64)
            maps.append(dTb)

            meta_fp = fp.with_suffix("").as_posix() + "_meta.json"
            with open(meta_fp, "r") as f:
                meta_all.append(json.load(f))

        except Exception as e:
            print(f"[sim {sim_id}] skip {fp.name}: {e}")

    if not maps:
        return None

    # X: (S, Npix)
    X = np.stack(maps, axis=0)

    cosmo = meta_all[0]["cosmo_params"]
    y = np.array([cosmo["Omega_m"], cosmo["Omega_b"], cosmo["sigma_8"]],
                 dtype=np.float32)

    out = dict(
        X=X,
        y=y,
        sim_id=sim_id,
        nside=nside_out,
        asinh_k=float(asinh_k),
        shell_min=int(shell_min),
        shell_max=int(shell_max),
    )
    outfile = outdir / f"sim_{sim_id:04d}_dtb_healpix.npz"
    np.savez_compressed(outfile, **out)
    return str(outfile)

def main():
    ap = argparse.ArgumentParser(description="Create field-level HEALPix DTB dataset for DeepSphere/SBI.")
    ap.add_argument("--fits-root", required=True, help="Root with per-sim folders containing Tb_shell*.fits")
    ap.add_argument("--outdir", required=True, help="Output dataset directory (ignored by git)")
    ap.add_argument("--nside-out", type=int, default=128)
    ap.add_argument("--asinh-k", type=float, default=0.05)
    ap.add_argument("--shell-min", type=int, default=240)
    ap.add_argument("--shell-max", type=int, default=990)
    ap.add_argument("--float32", action="store_true")
    ap.add_argument("--n-procs", type=int, default=0, help="0 => use cpu_count()")
    args = ap.parse_args()

    fits_root = Path(args.fits_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sim_dirs = sorted(
        [d for d in fits_root.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda p: int(p.name)
    )
    print("Found simulations:", len(sim_dirs))

    n_procs = args.n_procs if args.n_procs > 0 else cpu_count()

    worker = partial(
        process_one_sim,
        outdir=outdir,
        nside_out=args.nside_out,
        asinh_k=args.asinh_k,
        shell_min=args.shell_min,
        shell_max=args.shell_max,
        save_float32=args.float32,
    )

    results = []
    with Pool(processes=n_procs) as pool:
        for res in pool.imap_unordered(worker, sim_dirs):
            if res:
                results.append(res)
    with open(outdir / "index.txt", "w") as f:
        for r in sorted(results):
            f.write(r + "\n")

    print(f"[ok] wrote {len(results)} sims to {outdir}")

if __name__ == "__main__":
    main()
