"""
Reproduce Figures 1, 6, 7, and 8 using multi-turn geometry smoothing.
Now includes a high-resolution, smooth 3D rectangular surface rendering.

"""

import os
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

from hsx_utilities import (
    get_geom, run_force_profile, compute_delta, read_coil_geometry,
    simplify_multiturn_coil, B_reg_centerline, B_internal_parts, style_axes
)

def main():

    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    geom_file = 'HSX_coil_geometry.txt'

    xyz_coil, currents = read_coil_geometry(geom_file)
    coil_idx = 0
    Rx_full = xyz_coil[coil_idx, 0, :-1]
    Ry_full = xyz_coil[coil_idx, 1, :-1]
    Rz_full = xyz_coil[coil_idx, 2, :-1]

    # Base Parameters
    I_current = 150000.0  # 150 kA
    a_true, b_true = 0.13, 0.06
    delta_true = compute_delta(a_true, b_true)

    print("Fig1 Plotting")

    n_points_3d = 1024 #dense points to make smooth
    Rx_3d, Ry_3d, Rz_3d = simplify_multiturn_coil(Rx_full, Ry_full, Rz_full, n_turns=14, n_out=n_points_3d)
    phi_3d = np.linspace(0, 2 * np.pi, n_points_3d, endpoint=False)

    F_3d = run_force_profile(jnp.array(Rx_3d), jnp.array(Ry_3d), jnp.array(Rz_3d),
                             jnp.array(phi_3d), I_current, a_true, b_true, delta_true)
    F_3d = np.array(F_3d)
    F_mag_3d = np.linalg.norm(F_3d, axis=1) / 1000.0  # Force in kN/m

    # Extract the {p, q} frame vectors to build the rectangle
    _, _, _, _, p_hat, q_hat, _, _, _, _ = get_geom(
        jnp.array(Rx_3d), jnp.array(Ry_3d), jnp.array(Rz_3d), jnp.array(phi_3d))

    p_hat = np.array(p_hat)
    q_hat = np.array(q_hat)

    # Calculate the 4 corners of the rectangular cross-section
    u_corners = np.array([1, -1, -1, 1, 1])
    v_corners = np.array([1, 1, -1, -1, 1])

    X = np.zeros((n_points_3d, 5))
    Y = np.zeros((n_points_3d, 5))
    Z = np.zeros((n_points_3d, 5))

    for i in range(5):
        corner = np.stack([Rx_3d, Ry_3d, Rz_3d], axis=1) + \
                 (u_corners[i] * a_true / 2.0) * p_hat + \
                 (v_corners[i] * b_true / 2.0) * q_hat
        X[:, i] = corner[:, 0]
        Y[:, i] = corner[:, 1]
        Z[:, i] = corner[:, 2]

    # Close the toroidal loop (connect end back to start)
    X_closed = np.vstack([X, X[0:1, :]])
    Y_closed = np.vstack([Y, Y[0:1, :]])
    Z_closed = np.vstack([Z, Z[0:1, :]])
    F_mag_closed = np.append(F_mag_3d, F_mag_3d[0])

    # Map force magnitude to surface color
    norm = Normalize(vmin=F_mag_closed.min(), vmax=F_mag_closed.max())
    cmap = cm.jet

    face_colors = cmap(norm(F_mag_closed[:-1]))
    surface_colors = np.tile(face_colors[:, None, :], (1, 4, 1))

    fig1 = plt.figure(figsize=(10, 8), dpi=150)
    ax1 = fig1.add_subplot(111, projection='3d')

    surf = ax1.plot_surface(X_closed, Y_closed, Z_closed,
                            facecolors=surface_colors,
                            rstride=1, cstride=1,
                            shade=True,
                            linewidth=0, antialiased=False)

    skip = 25
    Ox = Rx_3d[::skip] + (a_true / 2.0) * p_hat[::skip, 0]
    Oy = Ry_3d[::skip] + (a_true / 2.0) * p_hat[::skip, 1]
    Oz = Rz_3d[::skip] + (a_true / 2.0) * p_hat[::skip, 2]

    F_direction = F_3d[::skip] / (F_mag_3d[::skip, None] * 1000.0)
    s_scale = 0.05

    q = ax1.quiver(Ox, Oy, Oz,
                   F_direction[:, 0] * s_scale,
                   F_direction[:, 1] * s_scale,
                   F_direction[:, 2] * s_scale,
                   colors=cmap(norm(F_mag_3d[::skip])),
                   linewidth=1.5, arrow_length_ratio=0.4)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig1.colorbar(sm, ax=ax1, shrink=0.5, aspect=15)
    cbar.set_label('Self-Force Magnitude [kN/m]', fontsize=11)

    # Viewpoint using SVD perpendicular to average plane
    coords = np.column_stack((Rx_3d, Ry_3d, Rz_3d))
    centroid = np.mean(coords, axis=0)
    centered_coords = coords - centroid
    _, _, vh = np.linalg.svd(centered_coords)
    nx, ny, nz = vh[2, :]  # Normal vector

    if nz < 0: # Ensure the coil isn't drawn upside down
        nx, ny, nz = -nx, -ny, -nz

    elev = np.degrees(np.arcsin(nz))
    azim = np.degrees(np.arctan2(ny, nx))
    ax1.view_init(elev=elev, azim=azim)

    ax1.set_title("Figure 1: HSX Coil with Self-Force Vectors", fontsize=13)

    # Equal aspect ratio bounding box
    max_range = np.array([Rx_3d.max() - Rx_3d.min(), Ry_3d.max() - Ry_3d.min(), Rz_3d.max() - Rz_3d.min()]).max() / 2.0
    mid_x, mid_y, mid_z = centroid
    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
    ax1.set_zlim(mid_z - max_range, mid_z + max_range)

    fig1.savefig(os.path.join(output_dir, f'Fig1_{coil_idx}.png'), bbox_inches='tight')
    plt.close(fig1)

    # Back to 256 pts for other plots

    n_points = 256
    Rx, Ry, Rz = simplify_multiturn_coil(Rx_full, Ry_full, Rz_full, n_turns=14, n_out=n_points)
    phi = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

    F_base = run_force_profile(jnp.array(Rx), jnp.array(Ry), jnp.array(Rz),
                               jnp.array(phi), I_current, a_true, b_true, delta_true)
    F_np = np.array(F_base)

    print("Generating Figure 6 (Internal Field)...")
    r_geom, rp, rpp, t_hat, p_hat, q_hat, b_hat, kappa, k1, k2 = get_geom(
        jnp.array(Rx), jnp.array(Ry), jnp.array(Rz), jnp.array(phi))

    Bcl = B_reg_centerline(r_geom, rp, rpp, b_hat, kappa, I_current, a_true, b_true, delta_true, jnp.array(phi))

    k2_np = np.array(k2)
    positive_k2_indices = np.where(k2_np > 0.5)[0]
    if len(positive_k2_indices) > 0:
        target_k2 = 1.5
        cross_idx = positive_k2_indices[np.argmin(np.abs(k2_np[positive_k2_indices] - target_k2))]
    else:
        cross_idx = int(np.argmax(F_np[:, 2]))

    n_uv = 80
    almost_one = 1 - 1e-6
    u = np.linspace(-almost_one, almost_one, n_uv)
    v = np.linspace(-almost_one, almost_one, n_uv)
    vv, uu = np.meshgrid(v, u, indexing='xy')

    B0, Bk, Bb = B_internal_parts(p_hat[cross_idx], q_hat[cross_idx], b_hat[cross_idx],
                                  kappa[cross_idx], k1[cross_idx], k2[cross_idx],
                                  I_current, a_true, b_true, delta_true, jnp.array(uu), jnp.array(vv))

    B_total = B0 + Bk + Bb + Bcl[cross_idx][None, None, :]
    B_total_np = np.array(B_total)
    Bmag = np.linalg.norm(B_total_np, axis=-1)
    Bz = B_total_np[:, :, 2]

    x_cm = vv * (b_true / 2) * 100
    y_cm = uu * (a_true / 2) * 100

    fig6, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=150)

    levels1 = np.linspace(0, float(Bmag.max()), 20)
    cs1 = axes[0].contour(-x_cm, y_cm, Bmag, colors='k', levels=levels1, linewidths=0.8)
    axes[0].clabel(cs1, fmt='%.2f', fontsize=8)
    axes[0].set_xlabel(r'$vb/2$ [cm]')
    axes[0].set_ylabel(r'$ua/2$ [cm]')
    axes[0].set_title(r'Reduced model $|B|$ [Tesla]')
    axes[0].set_aspect('equal')
    style_axes(axes[0])

    levels2 = np.linspace(float(Bz.min()), float(Bz.max()), 20)
    cs2 = axes[1].contour(x_cm, y_cm, Bz, colors='k', levels=levels2, linewidths=0.8, linestyles='--')
    axes[1].clabel(cs2, fmt='%.2f', fontsize=8)
    axes[1].set_xlabel(r'$vb/2$ [cm]')
    axes[1].set_ylabel(r'$ua/2$ [cm]')
    axes[1].set_title(r'Reduced model $B_z$ [Tesla]')
    axes[1].set_aspect('equal')
    style_axes(axes[1])

    plt.tight_layout()
    fig6.savefig(os.path.join(output_dir, f'Fig6_{coil_idx}.png'), bbox_inches='tight')
    plt.close(fig6)

    print("Generating Figure 7 (Thickness Sweep)...")
    phi_star = int(np.argmax(F_np[:, 2]))
    a_list = [1e-3, 1e-2, 1e-1]
    b_vals = np.logspace(-4, 0, 60)

    fig7, ax = plt.subplots(figsize=(8, 6), dpi=150)
    colors = ['salmon', 'limegreen', 'deepskyblue']

    for a, col in zip(a_list, colors):
        Fz_list = []
        for b in b_vals:
            delta = compute_delta(a, b)
            F = run_force_profile(jnp.array(Rx), jnp.array(Ry), jnp.array(Rz),
                                  jnp.array(phi), I_current, a, b, delta)
            Fz_list.append(float(F[phi_star, 2]) / 1000)

        a_mm = a * 1000
        label = f'a = {a_mm:.0f} mm, 1D' if a >= 0.01 else f'a = {a_mm:.1f} mm, 1D'
        ax.semilogx(b_vals, Fz_list, lw=2, ls='--', color=col, label=label)

    ax.set_xlabel(r'Conductor thickness $b$ [meters]', fontsize=12)
    ax.set_ylabel(r'z component of self-force per length, $dF_z/d\ell$ [kN/m]', fontsize=12)
    ax.set_xlim(1e-4, 1)
    ax.set_title("Figure 7: Self-force vs. Conductor Thickness", fontsize=13)
    ax.legend(loc='upper right')
    style_axes(ax)

    fig7.savefig(os.path.join(output_dir, f'Fig7_{coil_idx}.png'), bbox_inches='tight')
    plt.close(fig7)

    print("Generating Figure 8 (Force Profile)...")
    fig8, ax = plt.subplots(figsize=(10, 5), dpi=150)
    ax.plot(phi, F_np[:, 2] / 1000.0, lw=2, color='C0', label='Reduced (1D)')
    ax.set_xlim(0, 2 * np.pi)
    ax.set_xlabel(r'Location along the coil $\phi$', fontsize=12)
    ax.set_ylabel(r'z component of self-force per length, $dF_z/d\ell$ [kN/m]', fontsize=12)
    ax.set_title('Figure 8: Self-force $F_z$ along HSX coil (a=13 cm, b=6 cm)', fontsize=13)
    ax.axhline(0, color='gray', ls='--', lw=0.5)
    ax.legend()
    style_axes(ax)

    fig8.savefig(os.path.join(output_dir, f'Fig8_{coil_idx}.png'), bbox_inches='tight')
    plt.close(fig8)

if __name__ == "__main__":
    main()
