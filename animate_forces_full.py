import os
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.animation import FuncAnimation, PillowWriter

from hsx_utilities import (
    get_geom, run_force_profile, compute_delta, read_coil_geometry,
    simplify_multiturn_coil, style_axes
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
    geom_file = 'HSX_coil_geometry.txt'
    print(f"Loading and simplifying geometry from {geom_file}...")

    if not os.path.exists(geom_file):
        raise FileNotFoundError(f"Missing {geom_file}.")

    xyz_coil, currents = read_coil_geometry(geom_file)
    coil_idx = 0
    n_points = 256
    I_current = 150000.0
    a_true, b_true = 0.13, 0.06
    delta_true = compute_delta(a_true, b_true)

    coils_processed = []
    for i in range(len(currents)):
        Rx_f, Ry_f, Rz_f = xyz_coil[i, 0, :], xyz_coil[i, 1, :], xyz_coil[i, 2, :]
        # Filter out the trailing zeros (padding from the read function)
        valid = ~((Rx_f == 0.0) & (Ry_f == 0.0) & (Rz_f == 0.0))
        Rx_f, Ry_f, Rz_f = Rx_f[valid], Ry_f[valid], Rz_f[valid]

        # Chop off closing duplicate point if it exists to prevent spline overlap errors
        if len(Rx_f) > 1 and np.allclose([Rx_f[-1], Ry_f[-1], Rz_f[-1]], [Rx_f[0], Ry_f[0], Rz_f[0]]):
            Rx_f, Ry_f, Rz_f = Rx_f[:-1], Ry_f[:-1], Rz_f[:-1]

        rx, ry, rz = simplify_multiturn_coil(Rx_f, Ry_f, Rz_f, n_turns=14, n_out=128 if i != coil_idx else n_points)
        coils_processed.append(np.stack([rx, ry, rz], axis=1))

    r_target = coils_processed[coil_idx]
    Rx, Ry, Rz = r_target[:, 0], r_target[:, 1], r_target[:, 2]
    phi = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

    print("Computing Self & Inter-coil forces...")
    F_self = np.array(
        run_force_profile(jnp.array(Rx), jnp.array(Ry), jnp.array(Rz), jnp.array(phi), I_current, a_true, b_true,
                          delta_true))

    B_inter_total = np.zeros((n_points, 3))
    for i in range(len(currents)):
        if i == coil_idx: continue
        B_inter_total += np.array(compute_B_intercoil(jnp.array(r_target), jnp.array(coils_processed[i]), I_current))

    _, _, _, t_hat, p_hat, q_hat, _, _, _, _ = get_geom(jnp.array(Rx), jnp.array(Ry), jnp.array(Rz), jnp.array(phi))
    t_hat, p_hat, q_hat = np.array(t_hat), np.array(p_hat), np.array(q_hat)

    F_inter = I_current * np.cross(t_hat, B_inter_total)
    F_tot = F_self + F_inter

    F_tot_mag_kN = np.linalg.norm(F_tot, axis=1) / 1000.0
    Fz_tot_kN = F_tot[:, 2] / 1000.0
    Fz_self_kN = F_self[:, 2] / 1000.0

    u_corners = np.array([1, -1, -1, 1, 1])
    v_corners = np.array([1, 1, -1, -1, 1])
    X, Y, Z = np.zeros((n_points, 5)), np.zeros((n_points, 5)), np.zeros((n_points, 5))

    for i in range(5):
        corner = r_target + (u_corners[i] * a_true / 2.0) * p_hat + (v_corners[i] * b_true / 2.0) * q_hat
        X[:, i], Y[:, i], Z[:, i] = corner[:, 0], corner[:, 1], corner[:, 2]

    X_closed = np.vstack([X, X[0:1, :]])
    Y_closed = np.vstack([Y, Y[0:1, :]])
    Z_closed = np.vstack([Z, Z[0:1, :]])
    F_mag_closed = np.append(F_tot_mag_kN, F_tot_mag_kN[0])

    norm = Normalize(vmin=F_mag_closed.min(), vmax=F_mag_closed.max())
    cmap = cm.jet
    face_colors = cmap(norm(F_mag_closed[:-1]))
    surface_colors = np.tile(face_colors[:, None, :], (1, 4, 1))

    fig = plt.figure(figsize=(15, 6), dpi=120)

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X_closed, Y_closed, Z_closed, facecolors=surface_colors, rstride=1, cstride=1, shade=True,
                     linewidth=0, antialiased=False)

    coords = np.column_stack((Rx, Ry, Rz))
    centroid = np.mean(coords, axis=0)
    _, _, vh = np.linalg.svd(coords - centroid)
    nx, ny, nz = vh[2, :]
    if nz < 0: nx, ny, nz = -nx, -ny, -nz
    ax1.view_init(elev=np.degrees(np.arcsin(nz)), azim=np.degrees(np.arctan2(ny, nx)))
    ax1.axis('off')

    max_range = np.array([Rx.max() - Rx.min(), Ry.max() - Ry.min(), Rz.max() - Rz.min()]).max() / 2.0
    ax1.set_xlim(centroid[0] - max_range, centroid[0] + max_range)
    ax1.set_ylim(centroid[1] - max_range, centroid[1] + max_range)
    ax1.set_zlim(centroid[2] - max_range, centroid[2] + max_range)

    ax2 = fig.add_subplot(122)
    ax2.plot(phi, Fz_self_kN, 'k--', lw=1.5, alpha=0.5, label='Self Force')
    ax2.plot(phi, Fz_tot_kN, 'b-', lw=2.5, alpha=0.8, label='Total Force')
    ax2.set_xlim(0, 2 * np.pi)
    ax2.set_xlabel(r'Location along the coil $\phi$ [rad]', fontsize=12)
    ax2.set_ylabel(r'$F_z$ [kN/m]', fontsize=12)
    ax2.axhline(0, color='gray', ls='--', lw=1)
    ax2.legend(loc='upper right')
    style_axes(ax2)

    point3d, = ax1.plot([], [], [], 'ro', markersize=10, zorder=5)
    quiver_container = [None]
    point2d, = ax2.plot([], [], 'ro', markersize=10)
    vline2d = ax2.axvline(0, color='red', ls='--', alpha=0.5)

    def animate(frame):
        ox = Rx[frame] + (a_true / 2.0) * p_hat[frame, 0]
        oy = Ry[frame] + (a_true / 2.0) * p_hat[frame, 1]
        oz = Rz[frame] + (a_true / 2.0) * p_hat[frame, 2]
        point3d.set_data_3d([ox], [oy], [oz])

        if quiver_container[0] is not None:
            quiver_container[0].remove()

        f_dir = F_tot[frame] / (F_tot_mag_kN[frame] * 1000.0)
        s_scale = 0.12
        quiver_container[0] = ax1.quiver(ox, oy, oz, f_dir[0] * s_scale, f_dir[1] * s_scale, f_dir[2] * s_scale,
                                         color='red', linewidth=3, arrow_length_ratio=0.4)

        point2d.set_data([phi[frame]], [Fz_tot_kN[frame]])
        vline2d.set_xdata([phi[frame]])

        ax1.set_title(f'Coil Position: $\\phi$ = {phi[frame]:.2f} rad', fontsize=13)
        ax2.set_title(f'Total Force: $F_z$ = {Fz_tot_kN[frame]:.1f} kN/m', fontsize=13)
        return point3d, point2d, vline2d

    fps = 15
    total_frames = len(phi)
    print(f"Rendering GIF ({total_frames} frames)")

    os.makedirs('results', exist_ok=True)
    anim = FuncAnimation(fig, animate, frames=total_frames, interval=1000 / fps, blit=False)
    anim.save(f'results/coil{coil_idx}_total_force.gif', writer=PillowWriter(fps=fps))
    print("Saved animation!")
    plt.close(fig)


if __name__ == '__main__':
    main()