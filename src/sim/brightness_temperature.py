# src/brightness_temperature.py
import os
import json
import argparse
import numpy as np
import healpy as hp
from scipy.constants import h as h_planck, c, k, physical_constants


Msun_to_kg = 1.988e30
Mpc_to_m   = 3.086e22
A10        = 2.869e-15
f21        = 1.4204065e9
m_p        = physical_constants["proton mass"][0]
k_b        = k
Y_p   = 0.24
alpha = 0.2
beta  = -0.58
v_c0  = 10**1.56


def v_circ(M_msun, zred, Omega_m, h):
    """Circular velocity in km/s"""
    factor = (200 * Omega_m * h**2 / 24.4) ** (1 / 6)
    return 96.6 * factor * ((1 + zred) / 3.3) ** 0.5 * (M_msun / 1e11) ** (1 / 3)


def M_HI(M_msun_h, zred, Omega_m, f_H_c, h):
    """HI mass prescription. Input halo mass is Msun/h."""
    mass_ratio = (M_msun_h / (1e11 / h)) ** beta
    vc = v_circ(M_msun_h / h, zred, Omega_m, h)  
    return alpha * f_H_c * M_msun_h * mass_ratio * np.exp(-(v_c0 / vc) ** 3)


def main():
    ap = argparse.ArgumentParser(
        description="Generate 21cm brightness temperature and fluctuation maps from halos.txt."
    )
    ap.add_argument("--base-input", required=True, help="Root with per-sim halos.txt (output of convert.py)")
    ap.add_argument("--latin-file", required=True, help="Latin hypercube params txt (Omega_m Omega_b h n_s sigma8)")
    ap.add_argument("--output-base", required=True, help="Output root (will create fits/)")
    ap.add_argument("--nside", type=int, default=512)
    ap.add_argument("--z", type=float, default=1.0)
    ap.add_argument("--shell-start", type=float, default=0.0)
    ap.add_argument("--shell-end", type=float, default=1000.0)
    ap.add_argument("--shell-thickness", type=float, default=30.0)
    ap.add_argument("--sim-min", type=int, default=0)
    ap.add_argument("--sim-max", type=int, default=2000)
    ap.add_argument("--make-plots", action="store_true", help="Optional: make png plots (slow, not recommended by default)")
    args = ap.parse_args()

    latin = np.loadtxt(args.latin_file, comments="#", dtype=np.float64)

    Npix = hp.nside2npix(args.nside)
    theta_p, phi_p = hp.pix2ang(args.nside, np.arange(Npix))
    octant_mask = (theta_p <= np.pi / 2) & (phi_p <= np.pi / 2)
    Npix_oct = int(octant_mask.sum())
    f_sky = float(Npix_oct / Npix)

    shells = np.arange(args.shell_start, args.shell_end, args.shell_thickness)

    out_fits_root = os.path.join(args.output_base, "fits")
    os.makedirs(out_fits_root, exist_ok=True)

    if args.make_plots:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

    for sim in range(args.sim_min, args.sim_max + 1):
        sim_dir = os.path.join(args.base_input, str(sim))
        halo_file = os.path.join(sim_dir, "halos.txt")
        if not os.path.exists(halo_file):
            continue

        if sim >= latin.shape[0]:
            print(f"[skip] sim {sim}: latin index out of bounds")
            continue

        Omega_m, Omega_b, h, n_s, sigma_8 = latin[sim]
        Omega_L = 1 - Omega_m
        f_H_c = (1 - Y_p) * Omega_b / Omega_m

        H0 = 100 * h * 1e3 / Mpc_to_m
        Hz = H0 * np.sqrt(Omega_m * (1 + args.z) ** 3 + Omega_L)
        Tb_pre = (3 * h_planck * c**3 * A10) / (32 * np.pi * k_b * f21**2 * m_p)

        data = np.loadtxt(halo_file, skiprows=6)
        pos = data[:, 1:4]          # Mpc/h, comoving
        mass = data[:, 7]           # Msun/h

        r_all = np.linalg.norm(pos, axis=1)

        out_sim = os.path.join(out_fits_root, str(sim))
        os.makedirs(out_sim, exist_ok=True)
-
        mask_path = os.path.join(out_sim, "octant_mask.fits")
        if not os.path.exists(mask_path):
            hp.write_map(mask_path, octant_mask.astype(np.float32), overwrite=True)

        for shell_in in shells:
            shell_out = min(shell_in + args.shell_thickness, args.shell_end)
            shell_mask = (r_all >= shell_in) & (r_all < shell_out)
            if not np.any(shell_mask):
                continue

            pos_s = pos[shell_mask]
            mass_s = mass[shell_mask]
            r_s = np.linalg.norm(pos_s, axis=1)

            x, y, zc = pos_s.T
            cosang = np.clip(zc / r_s, -1.0, 1.0)
            theta = np.arccos(cosang)
            phi = np.mod(np.arctan2(y, x), 2 * np.pi)
            pix = hp.ang2pix(args.nside, theta, phi)

            MHI = M_HI(mass_s, args.z, Omega_m, f_H_c, h)  # Msun/h
            mass_map = np.bincount(pix, weights=MHI, minlength=Npix).astype(np.float64)
            mass_map[~octant_mask] = 0.0

            r_center = 0.5 * (shell_in + shell_out)
            dr = shell_out - shell_in

            shell_volume = (4 * np.pi * r_center**2 * dr) * f_sky / (1 + args.z) ** 3
            vol_pix = shell_volume / Npix_oct
            vol_pix_m3 = vol_pix * (Mpc_to_m**3) / (h**3)

            mass_map_kg = mass_map * Msun_to_kg / h
            rho_HI = mass_map_kg / vol_pix_m3

            Tb = Tb_pre * ((1 + args.z) ** 2 / Hz) * rho_HI * 1e3  # mK

            mean_Tb = float(Tb[octant_mask].mean())
            delta = np.zeros_like(Tb, dtype=np.float32)
            delta[octant_mask] = (Tb[octant_mask] - mean_Tb) / mean_Tb

            Tb_map = Tb.astype(np.float32)
            Tb_map[~octant_mask] = 0.0

            shell_name = f"Tb_shell{int(shell_in)}_{int(shell_out)}"

            # write FITS (two columns)
            fits_path = os.path.join(out_sim, f"{shell_name}.fits")
            hp.write_map(
                fits_path,
                [Tb_map.astype(np.float32), delta.astype(np.float32)],
                column_names=["TB_mK", "DTB"],
                dtype=np.float32,
                overwrite=True,
            )

            # data
            meta = dict(
                shell_in=float(shell_in),
                shell_out=float(shell_out),
                N_halos=int(len(mass_s)),
                f_sky=float(f_sky),
                mean_Tb=mean_Tb,
                M_HI_tot_Msun_h=float(MHI.sum()),
                model_params=dict(alpha=float(alpha), beta=float(beta), v_c0=float(v_c0), Y_p=float(Y_p)),
                cosmo_params=dict(
                    Omega_m=float(Omega_m),
                    Omega_b=float(Omega_b),
                    h=float(h),
                    n_s=float(n_s),
                    sigma_8=float(sigma_8),
                    z=float(args.z),
                ),
            )
            with open(os.path.join(out_sim, f"{shell_name}_meta.json"), "w") as f:
                json.dump(meta, f, indent=2, sort_keys=True)

            # optional plots (off by default)
            if args.make_plots:
                plot_dir = os.path.join(args.output_base, "brightness_temp", str(sim))
                os.makedirs(plot_dir, exist_ok=True)

                Tb_plot = Tb_map.copy()
                valid = Tb_plot != hp.UNSEEN
                if np.any(valid):
                    p1, p99 = np.percentile(Tb_plot[valid], [1, 99])
                else:
                    p1, p99 = 1e-4, 1.0

                Tb_plot[valid] = np.clip(Tb_plot[valid], 5e-4, None)
                Tb_plot = hp.smoothing(Tb_plot, fwhm=np.radians(0.5), verbose=False)

                title = f"T_b shell {shell_in:.0f}-{shell_out:.0f} | Om={Omega_m:.3f}, Ob={Omega_b:.3f}, h={h:.3f}"
                hp.mollview(Tb_plot + 1e-6, cbar=False, title=title, unit="mK", cmap="inferno", min=p1, max=p99)

                fig = plt.gcf()
                ax = plt.gca()
                image = ax.get_images()[0]
                image.set_norm(LogNorm(vmin=1e-4, vmax=1))
                cbar = fig.colorbar(image, ax=ax, orientation="horizontal", pad=0.05, fraction=0.05)
                cbar.set_label("Brightness Temperature [mK] (log)", fontsize=10, labelpad=6)

                plt.savefig(os.path.join(plot_dir, f"{shell_name}.png"), dpi=150, bbox_inches="tight")
                plt.close()

        print(f"[ok] sim {sim}")

if __name__ == "__main__":
    main()
