# src/data_preparation_2C.py
import os, re, json, argparse
import numpy as np
from pathlib import Path

NPZ_NAME_RE = re.compile(r"Tb_shell(\d+)_(\d+)_dTB_cl\.npz$")

def find_shell_files(sim_dir: Path, shell_rmin: int, shell_rmax: int):
    shells = []
    for fname in os.listdir(sim_dir):
        m = NPZ_NAME_RE.match(fname)
        if not m:
            continue
        shell_in, shell_out = map(int, m.groups())
        if shell_rmin <= shell_in < shell_rmax:
            shells.append((shell_in, shell_out, sim_dir / fname))
    return sorted(shells)

def read_shell_mean_tb(fits_root: Path, sim_id: int, shell_in: int, shell_out: int) -> float:
    meta_path = fits_root / f"{sim_id}" / f"Tb_shell{shell_in}_{shell_out}_meta.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return float(meta["mean_Tb"])

def load_Dl_matrix(sim_dir: Path, fits_root: Path, ell_band, shell_rmin, shell_rmax, floor, ell_ref: np.ndarray):
    sim_id = int(sim_dir.name)
    shells = find_shell_files(sim_dir, shell_rmin, shell_rmax)

    X_rows = []
    tb_means = []

    for shell_in, shell_out, fpath in shells:
        d = np.load(fpath)
        ell, Dl = d["ell"], d["Dl"]

        m = (ell >= ell_band[0]) & (ell <= ell_band[1])
        ell, Dl = ell[m], Dl[m]

        if ell_ref.size == 0:
            ell_ref = ell.copy()
        else:
            if not np.allclose(ell, ell_ref, rtol=1e-3, atol=1e-3):
                raise ValueError(f"Inconsistent ell grid in {fpath}")

        Dl = np.clip(Dl, floor, None)
        X_rows.append(np.log10(Dl).astype(np.float32))

        tb_means.append(read_shell_mean_tb(fits_root, sim_id, shell_in, shell_out))

    X = np.array(X_rows, dtype=np.float32)        # (S, L)
    tb = np.array(tb_means, dtype=np.float32)     # (S,)
    return X, tb, ell_ref

def main():
    ap = argparse.ArgumentParser(description="Prepare ML dataset from power spectrum outputs.")
    ap.add_argument("--power-root", required=True, help="Root with per-sim power spectra folders")
    ap.add_argument("--fits-root", required=True, help="Root with per-sim fits meta json (mean_Tb)")
    ap.add_argument("--latin-file", required=True, help="Latin hypercube params txt")
    ap.add_argument("--outdir", required=True, help="Dataset output directory")
    ap.add_argument("--ell-min", type=float, default=82.5)
    ap.add_argument("--ell-max", type=float, default=620.5)
    ap.add_argument("--shell-rmin", type=int, default=240)
    ap.add_argument("--shell-rmax", type=int, default=990)
    ap.add_argument("--floor", type=float, default=1e-12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-frac", type=float, default=1.0)
    ap.add_argument("--stage", choices=["per_sim", "final", "all"], default="all")
    args = ap.parse_args()

    power_root = Path(args.power_root)
    fits_root  = Path(args.fits_root)
    outdir     = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    P = np.loadtxt(Path(args.latin_file), comments="#")
    ell_band = (args.ell_min, args.ell_max)

    sim_dirs = [power_root / f for f in os.listdir(power_root) if f.isdigit()]
    sim_dirs = sorted(sim_dirs, key=lambda p: int(p.name))
    if args.stage in ("per_sim", "all"):
        ell_ref = np.array([])
        all_X, all_tb, all_ids = [], [], []

        for sim_dir in sim_dirs:
            try:
                X, tb, ell_ref = load_Dl_matrix(
                    sim_dir, fits_root,
                    ell_band=ell_band,
                    shell_rmin=args.shell_rmin,
                    shell_rmax=args.shell_rmax,
                    floor=args.floor,
                    ell_ref=ell_ref,
                )
                all_X.append(X)
                all_tb.append(tb)
                all_ids.append(int(sim_dir.name))
            except Exception as e:
                print(f"[skip] {sim_dir}: {e}")

        all_X  = np.array(all_X, dtype=np.float32)   # (N, S, L)
        all_tb = np.array(all_tb, dtype=np.float32)  # (N, S)
        all_ids = np.array(all_ids, dtype=int)

        print("Loaded:", all_X.shape, all_tb.shape, "N sims:", len(all_ids))

        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(len(all_X))
        n_train = int(args.train_frac * len(all_X))
        train_idx, val_idx = perm[:n_train], perm[n_train:]

        Xtr  = all_X[train_idx]
        tbtr = all_tb[train_idx]

      
        mu_X    = Xtr.mean(axis=(0,2))         # (S,)
        sigma_X = Xtr.std(axis=(0,2))          # (S,)
        sigma_X[sigma_X < 1e-8] = 1e-8

        log_tb  = np.log10(tbtr)
        mu_tb   = log_tb.mean(axis=0)          # (S,)
        sigma_tb = log_tb.std(axis=0)          # (S,)
        sigma_tb[sigma_tb < 1e-8] = 1e-8

        # write per sim
        for X, tb, sid in zip(all_X, all_tb, all_ids):
            Xn = (X - mu_X[:, None]) / sigma_X[:, None]                 # (S, L)
            tbn = (np.log10(tb) - mu_tb) / sigma_tb                     # (S,)
            tb_chan = np.broadcast_to(tbn[:, None], X.shape).astype(np.float32)  # (S, L)

            np.savez_compressed(
                outdir / f"sim_{sid:04d}.npz",
                Dl_norm=Xn.astype(np.float32),
                Tbch_norm=tb_chan.astype(np.float32),
            )

        meta = dict(
            ell=ell_ref.tolist(),
            ell_band=ell_band,
            shell_range=(args.shell_rmin, args.shell_rmax),
            mu_X=mu_X.tolist(),
            sigma_X=sigma_X.tolist(),
            mu_tb=mu_tb.tolist(),
            sigma_tb=sigma_tb.tolist(),
            train_ids=[int(x) for x in all_ids[train_idx]],
            val_ids=[int(x) for x in all_ids[val_idx]],
            note="Dl_norm: log10(Dl) normalized per-shell using (mu_X, sigma_X). "
                 "Tbch_norm: per-shell log10(meanTb) normalized using (mu_tb, sigma_tb) and broadcast over ell.",
        )
        with open(outdir / "preproc_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[ok] per-sim dataset written to {outdir}")

    if args.stage in ("final", "all"):
        files = sorted([f for f in os.listdir(outdir) if f.startswith("sim_") and f.endswith(".npz")])
        if not files:
            raise RuntimeError("No per-sim files found. Run with --stage per_sim first (or --stage all).")

        X_list, ids = [], []
        for fn in files:
            sid = int(fn.split("_")[1].split(".")[0])
            d = np.load(outdir / fn)
            Dl = d["Dl_norm"]          # (S, L)
            Tb = d["Tbch_norm"]        # (S, L)
            X_list.append(np.stack([Dl, Tb], axis=0))  # (2, S, L)
            ids.append(sid)

        X = np.stack(X_list, axis=0).astype(np.float32)      # (N, 2, S, L)
        ids = np.array(ids, dtype=int)

        # Y: Omega_m, Omega_b, sigma_8
        Y = P[ids][:, [0, 1, 4]].astype(np.float32)

        np.save(outdir / "X.npy", X)
        np.save(outdir / "Y.npy", Y)

        print("[ok] Final shapes:", X.shape, Y.shape)
        print(f"[ok] Saved X.npy and Y.npy to {outdir}")

if __name__ == "__main__":
    main()
