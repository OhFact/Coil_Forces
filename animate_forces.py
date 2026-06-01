import os
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.animation import FuncAnimation, PillowWriter

from hsx_utilities import (
    get_geom, run_force_profile, compute_delta, read_coil_geometry,
    simplify_multiturn_coil, style_axes
)


def main():
    geom_file = 'HSX_coil_geometry.txt'
    xyz_coil, currents = read_coil_geometry(geom_file)
    coil_idx = 0
    Rx_full = xyz_coil[coil_idx, 0, :-1]
    Ry_full = xyz_coil[coil_idx, 1, :-1]
    Rz_full = xyz_coil[coil_idx, 2, :-1]

    n_points = 256
    Rx, Ry, Rz = simplify_multiturn_coil(Rx_full, Ry_full, Rz_full, n_turns=14, n_out=n_points)
    phi = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

    I_current = 150000.0  # 150 kA
    a_true, b_true = 0.13, 0.06
    delta_true = compute_delta(a_true, b_true)

    print("Computing force profile...")
    F = run_force_profile(jnp.array(Rx), jnp.array(Ry), jnp.array(Rz),
                          jnp.array(phi), I_current, a_true, b_true, delta_true)
    F = np.array(F)
    F_mag_kN = np.linalg.norm(F, axis=1) / 1000.0  # Force in kN/m
    Fz_kN = F[:, 2] / 1000.0

    # Extract {p, q} vectors to build the 3D rectangular surface
    _, _, _, _, p_hat, q_hat, _, _, _, _ = get_geom(
        jnp.array(Rx), jnp.array(Ry), jnp.array(Rz), jnp.array(phi))
    p_hat = np.array(p_hat)
    q_hat = np.array(q_hat)

    # Calculate the 4 corners of the rectangular cross-section
    u_corners = np.array([1, -1, -1, 1, 1])
    v_corners = np.array([1, 1, -1, -1, 1])

    X, Y, Z = np.zeros((n_points, 5)), np.zeros((n_points, 5)), np.zeros((n_points, 5))

    for i in range(5):
        corner = np.stack([Rx, Ry, Rz], axis=1) + \
                 (u_corners[i] * a_true / 2.0) * p_hat + \
                 (v_corners[i] * b_true / 2.0) * q_hat
        X[:, i] = corner[:, 0]
        Y[:, i] = corner[:, 1]
        Z[:, i] = corner[:, 2]

    # Close the toroidal loop
    X_closed = np.vstack([X, X[0:1, :]])
    Y_closed = np.vstack([Y, Y[0:1, :]])
    Z_closed = np.vstack([Z, Z[0:1, :]])
    F_mag_closed = np.append(F_mag_kN, F_mag_kN[0])

    # Map force magnitude to surface color
    norm = Normalize(vmin=F_mag_closed.min(), vmax=F_mag_closed.max())
    cmap = cm.jet
    face_colors = cmap(norm(F_mag_closed[:-1]))
    surface_colors = np.tile(face_colors[:, None, :], (1, 4, 1))

    fig = plt.figure(figsize=(15, 6), dpi=120)

    # Left Subplot: 3D Coil
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(X_closed, Y_closed, Z_closed,
                            facecolors=surface_colors,
                            rstride=1, cstride=1,
                            shade=True, linewidth=0, antialiased=False)

    # SVD Viewpoint for Left Subplot
    coords = np.column_stack((Rx, Ry, Rz))
    centroid = np.mean(coords, axis=0)
    _, _, vh = np.linalg.svd(coords - centroid)
    nx, ny, nz = vh[2, :]
    if nz < 0: nx, ny, nz = -nx, -ny, -nz
    ax1.view_init(elev=np.degrees(np.arcsin(nz)), azim=np.degrees(np.arctan2(ny, nx)))

    ax1.set_xlabel("X [m]")
    ax1.set_ylabel("Y [m]")
    ax1.set_zlabel("Z [m]")
    ax1.axis('off')

    max_range = np.array([Rx.max() - Rx.min(), Ry.max() - Ry.min(), Rz.max() - Rz.min()]).max() / 2.0
    ax1.set_xlim(centroid[0] - max_range, centroid[0] + max_range)
    ax1.set_ylim(centroid[1] - max_range, centroid[1] + max_range)
    ax1.set_zlim(centroid[2] - max_range, centroid[2] + max_range)

    # Right Subplot: 2D Force Profile
    ax2 = fig.add_subplot(122)
    ax2.plot(phi, Fz_kN, 'b-', lw=2.5, alpha=0.6)
    ax2.set_xlim(0, 2 * np.pi)
    ax2.set_xlabel(r'Location along the coil $\phi$ [rad]', fontsize=12)
    ax2.set_ylabel(r'$F_z$ [kN/m]', fontsize=12)
    ax2.axhline(0, color='gray', ls='--', lw=1)
    style_axes(ax2)


    # 3D Trackers (Dot and Quiver)
    point3d, = ax1.plot([], [], [], 'ro', markersize=10, zorder=5)
    quiver_container = [None]

    # 2D Trackers (Dot and Line)
    point2d, = ax2.plot([], [], 'ro', markersize=10)
    vline2d = ax2.axvline(0, color='red', ls='--', alpha=0.5)

    def animate(frame):
        # Position slightly outwards so the red dot sits on the face of the coil
        ox = Rx[frame] + (a_true / 2.0) * p_hat[frame, 0]
        oy = Ry[frame] + (a_true / 2.0) * p_hat[frame, 1]
        oz = Rz[frame] + (a_true / 2.0) * p_hat[frame, 2]

        # Update 3D Point
        point3d.set_data_3d([ox], [oy], [oz])

        # Update 3D Quiver Arrow
        if quiver_container[0] is not None:
            quiver_container[0].remove()

        f_dir = F[frame] / (F_mag_kN[frame] * 1000.0)
        s_scale = 0.12  # Make the tracking arrow nice and large
        quiver_container[0] = ax1.quiver(ox, oy, oz,
                                         f_dir[0] * s_scale,
                                         f_dir[1] * s_scale,
                                         f_dir[2] * s_scale,
                                         color='red', linewidth=3, arrow_length_ratio=0.4)

        # Update 2D Plot
        point2d.set_data([phi[frame]], [Fz_kN[frame]])
        vline2d.set_xdata([phi[frame]])

        # Update Titles
        ax1.set_title(f'Coil Position: $\\phi$ = {phi[frame]:.2f} rad', fontsize=13)
        ax2.set_title(f'Force: $F_z$ = {Fz_kN[frame]:.1f} kN/m', fontsize=13)

        return point3d, point2d, vline2d


    fps = 15
    total_frames = len(phi)  # 256 frames
    print(f"Rendering GIF ({total_frames} frames at {fps} FPS)")

    anim = FuncAnimation(fig, animate, frames=total_frames, interval=1000 / fps, blit=False)

    output_file = f'results/coil{coil_idx}.gif'
    writer = PillowWriter(fps=fps)
    anim.save(output_file, writer=writer)

    print(f"Saved animation to {output_file}")
    plt.close(fig)


if __name__ == '__main__':
    main()
