# src/power_spectrum.py
import os
import re
import json
import hashlib
import argparse
import numpy as np
import healpy as hp
import pymaster as nmt


def mask_key(mk: np.ndarray) -> str:
    return hashlib.sha1(mk.view(np.uint8)).hexdigest()

def fsky_eff(mask: np.ndarray) -> float:
    return float(np.mean(mask**2))

def knox_sigma(ell_b: np.ndarray, Cb: np.ndarray, bin_width: int, fsky: float) -> np.ndarray:
    lb = np.maximum(ell_b, 1.0)
    var = 2.0 / ((2.0 * lb + 1.0) * bin_width * max(fsky, 1e-6)) * (Cb**2)
    return np.sqrt(var)

def find_mask(folder_path: str, shell_name: str, tb_map: np.ndarray) -> np.ndarray:
    m1 = os.path.join(folder_path, "octant_mask.fits")
    if os.path.exists(m1):
        mask_raw = hp.read_map(m1, dtype=np.float64, verbose=False)
        return (mask_raw > 0).astype(np.float64)
      
    m2 = os.path.join(folder_path, f"{shell_name}_mask.fits")
    if os.path.exists(m2):
        mask_raw = hp.read_map(m2, dtype=np.float64, verbose=False)
        return (mask_raw > 0).astype(np.float64)

    mk = np.isfinite(tb_map) & (tb_map != hp.UNSEEN)
    return mk.astype(np.float64)

def main():
    ap = argparse.ArgumentParser(description="Compute angular power spectra from Tb/DTB HEALPix FITS maps using NaMaster.")
    ap.add_argument("--input-root", required=True, help="Root directory containing fits/<sim_id>/Tb_shell*_*.fits")
    ap.add_argument("--output-root", required=True, help="Output root for spectra npz (and optional plots)")
    ap.add_argument("--nside", type=int, default=512)
    ap.add_argument("--bin-width", type=int, default=6)
    ap.add_argument("--sim-min", type=int, default=0)
    ap.add_argument("--sim-max", type=int, default=2000)
    ap.add_argument("--shell-min", type=int, default=240, help="Keep shells with shell_in >= this")
    ap.add_argument("--shell-max", type=int, default=990, help="Keep shells with shell_out <= this")
    ap.add_argument("--apod-deg", type=float, default=1.0, help="Mask apodization scale in degrees")
    ap.add_argument("--make-plots", action="store_true", help="Optional: save binned D_ell plots (slow)")
    args = ap.parse_args()

    NSIDE = args.nside
    LMAX = 3 * NSIDE - 1
    BIN_WIDTH = args.bin_width

    shell_re = re.compile(r"Tb_shell(\d+)_(\d+)\.fits$")
    binning = nmt.NmtBin.from_lmax_linear(lmax=LMAX, nlb=BIN_WIDTH)
    ell_b = binning.get_effective_ells()

    os.makedirs(args.output_root, exist_ok=True)


    workspace_cache = {}

    if args.make_plots:
        import matplotlib.pyplot as plt

    # Expect structure: input-root/fits/<sim_id>/*.fits
    # But allow passing directly the fits root too.
    # We'll detect if input-root has "fits" inside.
    fits_root = args.input_root
    if os.path.isdir(os.path.join(args.input_root, "fits")):
        fits_root = os.path.join(args.input_root, "fits")

    for folder in sorted(os.listdir(fits_root), key=lambda x: int(x) if x.isdigit() else 10**9):
        if not folder.isdigit():
            continue
        fid = int(folder)
        if not (args.sim_min <= fid <= args.sim_max):
            continue

        folder_path = os.path.join(fits_root, folder)
        if not os.path.isdir(folder_path):
            continue
          
        fits_files = []
        for f in os.listdir(folder_path):
            if not f.endswith(".fits"):
                continue
            m = shell_re.match(f)
            if not m:
                continue
            shell_in = int(m.group(1))
            shell_out = int(m.group(2))
            if (shell_in >= args.shell_min) and (shell_out <= args.shell_max):
                fits_files.append((shell_in, shell_out, f))

        fits_files.sort()
        if not fits_files:
            continue

        out_dir = os.path.join(args.output_root, str(fid))
        os.makedirs(out_dir, exist_ok=True)

        for shell_in, shell_out, fname in fits_files:
            fpath = os.path.join(folder_path, fname)
            shell_name = fname[:-5]  # strip .fits

            TB_mK, DTB = hp.read_map(fpath, field=(0, 1), dtype=np.float64, verbose=False)

            # data
            meta_path = os.path.join(folder_path, f"{shell_name}_meta.json")
            mean_Tb_mK = None
            if os.path.exists(meta_path):
                with open(meta_path, "r") as _f:
                    meta = json.load(_f)
                mean_Tb_mK = float(meta.get("mean_Tb", np.nan))

            mask = find_mask(folder_path, shell_name, TB_mK)
            mask = nmt.mask_apodization(mask, aposize=args.apod_deg, apotype="C2")

            if (mask > 0).sum() < 10:
                continue

            mkey = mask_key(mask.astype(np.float32))
            if mkey in workspace_cache:
                w = workspace_cache[mkey]
            else:
                f_tmp = nmt.NmtField(mask, [mask * 0.0])
                w = nmt.NmtWorkspace()
                w.compute_coupling_matrix(f_tmp, f_tmp, binning)
                workspace_cache[mkey] = w

            # dTB field 
            dtb = np.where(np.isfinite(DTB), DTB, 0.0)
            mval = (mask > 0) & np.isfinite(dtb)
            dtb = dtb - (dtb[mval].mean() if np.any(mval) else 0.0)

            f_dtb = nmt.NmtField(mask, [dtb])
            cl_coup_dtb = nmt.compute_coupled_cell(f_dtb, f_dtb)
            cl_dtb = w.decouple_cell(cl_coup_dtb)[0]

            # TB field (mean-subtracted on observed region)
            tb = np.where(np.isfinite(TB_mK) & (TB_mK != hp.UNSEEN) & (mask > 0), TB_mK, 0.0)
            val = (mask > 0) & np.isfinite(tb)
            tb = tb - (tb[val].mean() if np.any(val) else 0.0)

            f_tb = nmt.NmtField(mask, [tb])
            cl_coup_tb = nmt.compute_coupled_cell(f_tb, f_tb)
            cl_tb = w.decouple_cell(cl_coup_tb)[0]

            #  errors (Knox) for visualization
            fsky = fsky_eff(mask)
            sigC_dtb = knox_sigma(ell_b, cl_dtb, BIN_WIDTH, fsky)
            sigC_tb  = knox_sigma(ell_b, cl_tb,  BIN_WIDTH, fsky)

            Dl_dtb = ell_b * (ell_b + 1) * cl_dtb / (2 * np.pi)
            Dl_tb  = ell_b * (ell_b + 1) * cl_tb  / (2 * np.pi)
            Dlerr_dtb = ell_b * (ell_b + 1) * sigC_dtb / (2 * np.pi)
            Dlerr_tb  = ell_b * (ell_b + 1) * sigC_tb  / (2 * np.pi)

            # save
            np.savez_compressed(
                os.path.join(out_dir, f"{shell_name}_dTB_cl.npz"),
                ell=ell_b, cl=cl_dtb, Dl=Dl_dtb, Dl_err=Dlerr_dtb,
                field="DTB", nside=NSIDE, lmax=LMAX, bin_width=BIN_WIDTH,
                shell_in=int(shell_in), shell_out=int(shell_out),
                mean_Tb_mK=mean_Tb_mK,
                fsky=fsky,
            )
            np.savez_compressed(
                os.path.join(out_dir, f"{shell_name}_TB_cl.npz"),
                ell=ell_b, cl=cl_tb, Dl=Dl_tb, Dl_err=Dlerr_tb,
                field="TB_mK_minus_mean", nside=NSIDE, lmax=LMAX, bin_width=BIN_WIDTH,
                shell_in=int(shell_in), shell_out=int(shell_out),
                mean_Tb_mK=mean_Tb_mK,
                fsky=fsky,
            )

            # optional plot
            if args.make_plots:
                import matplotlib.pyplot as plt
                m = np.isfinite(Dl_tb) & np.isfinite(Dlerr_tb) & (Dl_tb > 0) & (Dlerr_tb > 0)
                if np.any(m):
                    plt.figure(figsize=(5.8, 4.2), dpi=200)
                    yerr_low = np.minimum(Dlerr_tb[m], 0.99 * Dl_tb[m])
                    yerr_high = Dlerr_tb[m]
                    plt.errorbar(
                        ell_b[m], Dl_tb[m],
                        yerr=np.vstack([yerr_low, yerr_high]),
                        fmt="o", ms=3, lw=1, capsize=2,
                        label="Knox errors",
                    )
                    plt.xscale("log"); plt.yscale("log")
                    plt.xlabel(r"$\ell$")
                    plt.ylabel(r"$D_\ell$ [mK$^2$]")
                    plt.grid(True, which="major", alpha=0.25)
                    plt.title(f"{shell_name}  Δℓ={BIN_WIDTH}  z=1")
                    plt.legend(frameon=False, loc="upper left")
                    plt.tight_layout()
                    plt.savefig(os.path.join(out_dir, f"{shell_name}_Dl.png"))
                    plt.close()

        print(f"[ok] sim {fid}")

if __name__ == "__main__":
    main()
