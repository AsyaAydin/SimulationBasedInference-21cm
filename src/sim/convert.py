# src/convert.py
import os
import argparse
import numpy as np

# Pylians3: user must make readfof importable (e.g. via PYTHONPATH)
import readfof

Z_MAP = {4: 0.0, 3: 0.5, 2: 1.0, 1: 2.0, 0: 3.0}

def process_sim(i: int, in_root: str, out_root: str, snapnum: int,
                mass_cut: float, box_size: float,
                Om_m: np.ndarray, hubble: np.ndarray):

    in_dir = os.path.join(in_root, str(i))
    out_dir = os.path.join(out_root, str(i))

    if not os.path.isdir(in_dir):
        print(f"[skip] missing input: {in_dir}")
        return

    os.makedirs(out_dir, exist_ok=True)
    z = Z_MAP[snapnum]

    FoF = readfof.FoF_catalog(
        in_dir, snapnum,
        long_ids=False, swap=False, SFR=False, read_IDs=False
    )

    pos_h = FoF.GroupPos / 1e3        # Mpc/h
    mass  = FoF.GroupMass * 1e10      # Msun/h
    vel_h = FoF.GroupVel * (1.0 + z)  # km/s (Quijote convention)

    sel = (mass >= mass_cut)
    N = int(sel.sum())
    if N == 0:
        print(f"[warn] {i}: no halos above MASS_CUT.")
        return

    outcat = np.hstack([
        np.arange(N, dtype=int).reshape(N, 1),
        pos_h[sel],
        vel_h[sel],
        mass[sel].reshape(N, 1),
    ])

    header = f"{box_size}\n{Om_m[i]}\n{hubble[i]}\n{z}\n{N}\n"
    outfile = os.path.join(out_dir, "halos.txt")

    np.savetxt(
        outfile,
        outcat,
        fmt=("%1i %10.5e %10.5e %10.5e %10.5e %10.5e %10.5e %10.5e"),
        header=header,
        comments="",
    )

    print(f"[ok] {i} -> {outfile}")
    print(f"    number density ~ {N/(box_size**3):.4e} (h^3 Mpc^-3), N={N}")

def main():
    ap = argparse.ArgumentParser(description="Convert Quijote FoF catalogs to halos.txt per simulation.")
    ap.add_argument("--cosmo-txt", required=True, help="Path to latin_hypercube_params.txt")
    ap.add_argument("--in-root", required=True, help="Root dir containing Quijote simulations (folders 0..N)")
    ap.add_argument("--out-root", required=True, help="Output root dir to write halos.txt")
    ap.add_argument("--snapnum", type=int, default=2, help="Snapshot number (default: 2 for z=1)")
    ap.add_argument("--mass-cut", type=float, default=3.1622777e12, help="Mass cut [Msun/h]")
    ap.add_argument("--box-size", type=float, default=1000.0, help="Box size [Mpc/h]")
    ap.add_argument("--sim-min", type=int, default=0)
    ap.add_argument("--sim-max", type=int, default=2000)
    args = ap.parse_args()

    cosmo = np.loadtxt(args.cosmo_txt)
    Om_m = cosmo[:, 0]
    hubble = cosmo[:, 2]

    for sim_id in range(args.sim_min, args.sim_max + 1):
        process_sim(
            sim_id,
            in_root=args.in_root,
            out_root=args.out_root,
            snapnum=args.snapnum,
            mass_cut=args.mass_cut,
            box_size=args.box_size,
            Om_m=Om_m,
            hubble=hubble,
        )

if __name__ == "__main__":
    main()
