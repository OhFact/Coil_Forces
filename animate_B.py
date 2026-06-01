"""
Generate a smooth animated GIF showing the 2D Internal Magnetic Field
(|B| and Bz) as we travel around the 3D rectangular HSX coil.
"""

import os
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.animation import FuncAnimation, PillowWriter

from hsx_utilities import (
    get_geom, B_reg_centerline, B_internal_parts, compute_delta,
    read_coil_geometry, simplify_multiturn_coil, style_axes
)


def main():
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    geom_file = 'HSX_coil_geometry.txt'

    print(f"Loading from {geom_file}...")
    xyz_coil, currents = read_coil_geometry(geom_file)
    coil_idx = 5
    Rx_full = xyz_coil[coil_idx, 0, :-1]
    Ry_full = xyz_coil[coil_idx, 1, :-1]
    Rz_full = xyz_coil[coil_idx, 2, :-1]

    n_points = 128
    Rx, Ry, Rz = simplify_multiturn_coil(Rx_full, Ry_full, Rz_full, n_turns=14, n_out=n_points)
    phi = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

    # Base Parameters
    I_current = 150000.0  # 150 kA
    a_true, b_true = 0.13, 0.06
    delta_true = compute_delta(a_true, b_true)

    print("Calculating local frames and regularized centerline field...")
    r_geom, rp, rpp, t_hat, p_hat, q_hat, b_hat, kappa, k1, k2 = get_geom(
        jnp.array(Rx), jnp.array(Ry), jnp.array(Rz), jnp.array(phi))

    Bcl = B_reg_centerline(r_geom, rp, rpp, b_hat, kappa, I_current, a_true, b_true, delta_true, jnp.array(phi))
    Bcl_np = np.array(Bcl)

    # Extract frame vectors for 3D surface rendering
    p_hat = np.array(p_hat)
    q_hat = np.array(q_hat)

    # Calculate the 4 corners of the rectangular cross-section for the 3D plot
    u_corners = np.array([1, -1, -1, 1, 1])
    v_corners = np.array([1, 1, -1, -1, 1])

    X = np.zeros((n_points, 5))
    Y = np.zeros((n_points, 5))
    Z = np.zeros((n_points, 5))

    for i in range(5):
        corner = np.stack([Rx, Ry, Rz], axis=1) + \
                 (u_corners[i] * a_true / 2.0) * p_hat + \
                 (v_corners[i] * b_true / 2.0) * q_hat
        X[:, i] = corner[:, 0]
        Y[:, i] = corner[:, 1]
        Z[:, i] = corner[:, 2]

    X_closed = np.vstack([X, X[0:1, :]])
    Y_closed = np.vstack([Y, Y[0:1, :]])
    Z_closed = np.vstack([Z, Z[0:1, :]])

    # Define the 2D cross-section grid
    n_uv = 60
    almost_one = 1 - 1e-6
    u = np.linspace(-almost_one, almost_one, n_uv)
    v = np.linspace(-almost_one, almost_one, n_uv)
    vv, uu = np.meshgrid(v, u, indexing='xy')

    u_jax, v_jax = jnp.array(uu), jnp.array(vv)

    global_bmag_max = 0.0
    global_bz_max_abs = 0.0

    for i in range(n_points):
        B0, Bk, Bb = B_internal_parts(p_hat[i], q_hat[i], b_hat[i],
                                      kappa[i], k1[i], k2[i],
                                      I_current, a_true, b_true, delta_true,
                                      u_jax, v_jax)
        B_tot = B0 + Bk + Bb + Bcl_np[i][None, None, :]
        B_tot_np = np.array(B_tot)

        bmag = np.linalg.norm(B_tot_np, axis=-1)
        bz = B_tot_np[:, :, 2]

        global_bmag_max = max(global_bmag_max, bmag.max())
        global_bz_max_abs = max(global_bz_max_abs, np.abs(bz).max())

    # Create static levels and norms
    levels_bmag = np.linspace(0, global_bmag_max, 20)
    levels_bz = np.linspace(-global_bz_max_abs, global_bz_max_abs, 20)

    print("Setting up animation canvas...")
    fig = plt.figure(figsize=(16, 6), dpi=120)

    # 3D Coil Tracker
    ax1 = fig.add_subplot(131, projection='3d')

    # Plot the smooth rectangular surface
    ax1.plot_surface(X_closed, Y_closed, Z_closed,
                     color='lightsteelblue', alpha=0.5,
                     rstride=1, cstride=1,
                     shade=True, linewidth=0, antialiased=False)

    point3d, = ax1.plot([], [], [], 'ro', markersize=10, zorder=10)

    coords = np.column_stack((Rx, Ry, Rz))
    centroid = np.mean(coords, axis=0)
    _, _, vh = np.linalg.svd(coords - centroid)
    nx, ny, nz = vh[2, :]
    if nz < 0: nx, ny, nz = -nx, -ny, -nz
    ax1.view_init(elev=np.degrees(np.arcsin(nz)), azim=np.degrees(np.arctan2(ny, nx)))

    max_range = np.array([Rx.max() - Rx.min(), Ry.max() - Ry.min(), Rz.max() - Rz.min()]).max() / 2.0
    ax1.set_xlim(centroid[0] - max_range, centroid[0] + max_range)
    ax1.set_ylim(centroid[1] - max_range, centroid[1] + max_range)
    ax1.set_zlim(centroid[2] - max_range, centroid[2] + max_range)
    ax1.axis('off')

    # Middle Subplot: |B|
    ax2 = fig.add_subplot(132)
    ax2.set_aspect('equal')
    style_axes(ax2)

    # Right Subplot: Bz
    ax3 = fig.add_subplot(133)
    ax3.set_aspect('equal')
    style_axes(ax3)

    # Convert normalized u,v back to physical cm for the axes
    x_cm = vv * (b_true / 2.0) * 100.0
    y_cm = uu * (a_true / 2.0) * 100.0

    # Draw static colorbars using ScalarMappables
    sm_bmag = cm.ScalarMappable(cmap='viridis', norm=Normalize(vmin=0, vmax=global_bmag_max))
    sm_bmag.set_array([])
    fig.colorbar(sm_bmag, ax=ax2, label='|B| [T]', shrink=0.8)

    sm_bz = cm.ScalarMappable(cmap='RdBu_r',
                              norm=TwoSlopeNorm(vcenter=0., vmin=-global_bz_max_abs, vmax=global_bz_max_abs))
    sm_bz.set_array([])
    fig.colorbar(sm_bz, ax=ax3, label='Bz [T]', shrink=0.8)

    def animate(frame):
        # 1. Update 3D tracking dot (shifted to the outer face of the rectangle)
        ox = Rx[frame] + (a_true / 2.0) * p_hat[frame, 0]
        oy = Ry[frame] + (a_true / 2.0) * p_hat[frame, 1]
        oz = Rz[frame] + (a_true / 2.0) * p_hat[frame, 2]

        point3d.set_data_3d([ox], [oy], [oz])
        ax1.set_title(f'Coil Position: $\\phi$ = {phi[frame]:.2f} rad', fontsize=12)

        # 2. Compute the field for this specific cross-section
        B0, Bk, Bb = B_internal_parts(p_hat[frame], q_hat[frame], b_hat[frame],
                                      kappa[frame], k1[frame], k2[frame],
                                      I_current, a_true, b_true, delta_true,
                                      u_jax, v_jax)

        B_tot = B0 + Bk + Bb + Bcl_np[frame][None, None, :]
        B_tot_np = np.array(B_tot)

        bmag = np.linalg.norm(B_tot_np, axis=-1)
        bz = B_tot_np[:, :, 2]

        # 3. Clear and redraw ax2 (|B|)
        ax2.clear()
        cs1 = ax2.contourf(x_cm, y_cm, bmag, levels=levels_bmag, cmap='viridis', extend='both')
        ct1 = ax2.contour(x_cm, y_cm, bmag, levels=levels_bmag, colors='k', linewidths=0.5, alpha=0.5)
        ax2.set_xlabel(r'$vb/2$ [cm]')
        ax2.set_ylabel(r'$ua/2$ [cm]')
        ax2.set_title(r'Reduced model $|B|$ [Tesla]', fontsize=12)
        ax2.set_aspect('equal')
        style_axes(ax2)

        # 4. Clear and redraw ax3 (Bz)
        ax3.clear()
        cs2 = ax3.contourf(x_cm, y_cm, bz, levels=levels_bz, cmap='RdBu_r', extend='both')
        ct2 = ax3.contour(x_cm, y_cm, bz, levels=levels_bz, colors='k', linewidths=0.5, linestyles='solid', alpha=0.5)
        ax3.set_xlabel(r'$vb/2$ [cm]')
        ax3.set_ylabel(r'$ua/2$ [cm]')
        ax3.set_title(r'Reduced model $B_z$ [Tesla]', fontsize=12)
        ax3.set_aspect('equal')
        style_axes(ax3)

        return point3d,

    fps = 16
    total_frames = n_points
    print(f"Rendering {total_frames} frames at {fps} FPS")

    anim = FuncAnimation(fig, animate, frames=total_frames, interval=1000 / fps, blit=False)

    output_file = os.path.join(output_dir, f'internal_field_{coil_idx}.gif')
    writer = PillowWriter(fps=fps)
    anim.save(output_file, writer=writer)
    plt.close(fig)


if __name__ == '__main__':
    main()
