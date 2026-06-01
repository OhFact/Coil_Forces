import os
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.animation import FuncAnimation, PillowWriter

from hsx_utilities import (
    get_geom, B_reg_centerline, B_internal_parts, compute_delta,
    read_coil_geometry, simplify_multiturn_coil, style_axes
)


@jax.jit
def compute_B_intercoil(r_target, r_source, I_source):
    mu_0 = 4.0 * jnp.pi * 1e-7
    diff = r_target[:, None, :] - r_source[None, :, :]
    dist_sq = jnp.sum(diff ** 2, axis=2)
    dist_inv3 = jnp.where(dist_sq > 1e-12, dist_sq ** (-1.5), 0.0)
    dr_source = jnp.roll(r_source, -1, axis=0) - r_source
    cross_prod = jnp.cross(dr_source[None, :, :], diff)
    integrand = cross_prod * dist_inv3[:, :, None]
    return (mu_0 * I_source / (4.0 * jnp.pi)) * jnp.sum(integrand, axis=1)


def main():
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    geom_file = 'HSX_coil_geometry.txt'
    if not os.path.exists(geom_file):
        raise FileNotFoundError(f"Missing {geom_file}.")

    xyz_coil, currents = read_coil_geometry(geom_file)
    coil_idx = 0
    n_points = 128
    I_current = 150000.0
    a_true, b_true = 0.13, 0.06
    delta_true = compute_delta(a_true, b_true)

    print("Processing all 48 coils for background fields...")
    coils_processed = []
    for i in range(len(currents)):
        Rx_f, Ry_f, Rz_f = xyz_coil[i, 0, :], xyz_coil[i, 1, :], xyz_coil[i, 2, :]

        # 1. Filter out the trailing zeros (padding from the read function)
        valid = ~((Rx_f == 0.0) & (Ry_f == 0.0) & (Rz_f == 0.0))
        Rx_f, Ry_f, Rz_f = Rx_f[valid], Ry_f[valid], Rz_f[valid]

        # 2. Chop off closing duplicate point if it exists to prevent spline overlap errors
        if len(Rx_f) > 1 and np.allclose([Rx_f[-1], Ry_f[-1], Rz_f[-1]], [Rx_f[0], Ry_f[0], Rz_f[0]]):
            Rx_f, Ry_f, Rz_f = Rx_f[:-1], Ry_f[:-1], Rz_f[:-1]

        rx, ry, rz = simplify_multiturn_coil(Rx_f, Ry_f, Rz_f, n_turns=14, n_out=128 if i != coil_idx else n_points)
        coils_processed.append(np.stack([rx, ry, rz], axis=1))

    r_target = coils_processed[coil_idx]
    Rx, Ry, Rz = r_target[:, 0], r_target[:, 1], r_target[:, 2]
    phi = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

    r_geom, rp, rpp, t_hat, p_hat, q_hat, b_hat, kappa, k1, k2 = get_geom(
        jnp.array(Rx), jnp.array(Ry), jnp.array(Rz), jnp.array(phi))
    Bcl_np = np.array(
        B_reg_centerline(r_geom, rp, rpp, b_hat, kappa, I_current, a_true, b_true, delta_true, jnp.array(phi)))

    print("Calculating External B-field from Mutual Interaction...")
    B_inter_total = np.zeros((n_points, 3))
    for i in range(len(currents)):
        if i == coil_idx: continue
        B_inter_total += np.array(compute_B_intercoil(jnp.array(r_target), jnp.array(coils_processed[i]), I_current))

    p_hat, q_hat = np.array(p_hat), np.array(q_hat)
    u_corners, v_corners = np.array([1, -1, -1, 1, 1]), np.array([1, 1, -1, -1, 1])
    X, Y, Z = np.zeros((n_points, 5)), np.zeros((n_points, 5)), np.zeros((n_points, 5))

    for i in range(5):
        corner = r_target + (u_corners[i] * a_true / 2.0) * p_hat + (v_corners[i] * b_true / 2.0) * q_hat
        X[:, i], Y[:, i], Z[:, i] = corner[:, 0], corner[:, 1], corner[:, 2]

    X_closed = np.vstack([X, X[0:1, :]])
    Y_closed = np.vstack([Y, Y[0:1, :]])
    Z_closed = np.vstack([Z, Z[0:1, :]])

    n_uv = 60
    u = np.linspace(-(1 - 1e-6), (1 - 1e-6), n_uv)
    v = np.linspace(-(1 - 1e-6), (1 - 1e-6), n_uv)
    vv, uu = np.meshgrid(v, u, indexing='xy')
    u_jax, v_jax = jnp.array(uu), jnp.array(vv)

    global_bmag_max = 0.0
    global_bz_max_abs = 0.0

    for i in range(n_points):
        B0, Bk, Bb = B_internal_parts(p_hat[i], q_hat[i], b_hat[i], kappa[i], k1[i], k2[i], I_current, a_true, b_true,
                                      delta_true, u_jax, v_jax)
        B_tot = B0 + Bk + Bb + Bcl_np[i][None, None, :] + B_inter_total[i][None, None, :]
        B_tot_np = np.array(B_tot)

        global_bmag_max = max(global_bmag_max, np.linalg.norm(B_tot_np, axis=-1).max())
        global_bz_max_abs = max(global_bz_max_abs, np.abs(B_tot_np[:, :, 2]).max())

    levels_bmag = np.linspace(0, global_bmag_max, 20)
    levels_bz = np.linspace(-global_bz_max_abs, global_bz_max_abs, 20)

    print("Setting up animation canvas...")
    fig = plt.figure(figsize=(16, 6), dpi=120)

    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(X_closed, Y_closed, Z_closed, color='lightsteelblue', alpha=0.5, rstride=1, cstride=1, shade=True,
                     linewidth=0, antialiased=False)
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

    ax2 = fig.add_subplot(132)
    ax2.set_aspect('equal')
    style_axes(ax2)

    ax3 = fig.add_subplot(133)
    ax3.set_aspect('equal')
    style_axes(ax3)

    x_cm, y_cm = vv * (b_true / 2.0) * 100.0, uu * (a_true / 2.0) * 100.0

    sm_bmag = cm.ScalarMappable(cmap='viridis', norm=Normalize(vmin=0, vmax=global_bmag_max))
    sm_bmag.set_array([])
    fig.colorbar(sm_bmag, ax=ax2, label='Total |B| [T]', shrink=0.8)

    sm_bz = cm.ScalarMappable(cmap='RdBu_r',
                              norm=TwoSlopeNorm(vcenter=0., vmin=-global_bz_max_abs, vmax=global_bz_max_abs))
    sm_bz.set_array([])
    fig.colorbar(sm_bz, ax=ax3, label='Total Bz [T]', shrink=0.8)

    def animate(frame):
        ox = Rx[frame] + (a_true / 2.0) * p_hat[frame, 0]
        oy = Ry[frame] + (a_true / 2.0) * p_hat[frame, 1]
        oz = Rz[frame] + (a_true / 2.0) * p_hat[frame, 2]
        point3d.set_data_3d([ox], [oy], [oz])
        ax1.set_title(f'Coil Position: $\\phi$ = {phi[frame]:.2f} rad', fontsize=12)

        B0, Bk, Bb = B_internal_parts(p_hat[frame], q_hat[frame], b_hat[frame], kappa[frame], k1[frame], k2[frame],
                                      I_current, a_true, b_true, delta_true, u_jax, v_jax)
        B_tot = B0 + Bk + Bb + Bcl_np[frame][None, None, :] + B_inter_total[frame][None, None, :]
        B_tot_np = np.array(B_tot)

        bmag = np.linalg.norm(B_tot_np, axis=-1)
        bz = B_tot_np[:, :, 2]

        ax2.clear()
        ax2.contourf(x_cm, y_cm, bmag, levels=levels_bmag, cmap='viridis', extend='both')
        ax2.contour(x_cm, y_cm, bmag, levels=levels_bmag, colors='k', linewidths=0.5, alpha=0.5)
        ax2.set_xlabel(r'$vb/2$ [cm]')
        ax2.set_ylabel(r'$ua/2$ [cm]')
        ax2.set_title(r'Total $|B|$ [Tesla]', fontsize=12)
        ax2.set_aspect('equal')
        style_axes(ax2)

        ax3.clear()
        ax3.contourf(x_cm, y_cm, bz, levels=levels_bz, cmap='RdBu_r', extend='both')
        ax3.contour(x_cm, y_cm, bz, levels=levels_bz, colors='k', linewidths=0.5, linestyles='solid', alpha=0.5)
        ax3.set_xlabel(r'$vb/2$ [cm]')
        ax3.set_ylabel(r'$ua/2$ [cm]')
        ax3.set_title(r'Total $B_z$ [Tesla]', fontsize=12)
        ax3.set_aspect('equal')
        style_axes(ax3)

        return point3d,

    fps = 16
    total_frames = n_points
    print(f"Rendering {total_frames} frames at {fps} FPS")

    anim = FuncAnimation(fig, animate, frames=total_frames, interval=1000 / fps, blit=False)
    anim.save(os.path.join(output_dir, f'internal_field_TOTAL_{coil_idx}.gif'), writer=PillowWriter(fps=fps))
    plt.close(fig)


if __name__ == '__main__':
    main()