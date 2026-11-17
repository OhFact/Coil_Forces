"""Plot length-averaged TPQ forces over eight equal arc-length coil regions."""

import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from hsx_utilities import (
    compute_delta,
    get_geom,
    read_coil_geometry,
    run_force_profile,
    simplify_multiturn_coil,
)


N_MAIN_COILS = 48
N_UNIQUE_COIL_SHAPES = 6
N_REGIONS = 8


@jax.jit
def compute_B_intercoil(r_target, r_source, I_source):
    """Biot-Savart field using the same quadrature as the TPQ scripts."""
    mu_0 = 4.0 * jnp.pi * 1e-7
    r_next = jnp.roll(r_source, -1, axis=0)
    dr_source = r_next - r_source

    # Evaluate each segment at its midpoint (second-order accurate)
    diff = r_target[:, None, :] - (0.5 * (r_source + r_next))[None, :, :]
    dist_sq = jnp.sum(diff ** 2, axis=2)
    dist_inv3 = jnp.where(dist_sq > 1e-12, dist_sq ** (-1.5), 0.0)

    cross_prod = jnp.cross(dr_source[None, :, :], diff)

    integrand = cross_prod * dist_inv3[:, :, None]
    return (mu_0 * I_source / (4.0 * jnp.pi)) * jnp.sum(integrand, axis=1)


def prepare_main_coils(xyz_coil, coil_index, target_points, source_points):
    """Convert the 14-turn main-coil paths to smooth centerlines."""
    coils = []
    for index in range(min(N_MAIN_COILS, xyz_coil.shape[0])):
        coordinates = xyz_coil[index].T
        valid = ~np.all(coordinates == 0.0, axis=1)
        coordinates = coordinates[valid]
        if len(coordinates) > 1 and np.allclose(coordinates[-1], coordinates[0]):
            coordinates = coordinates[:-1]

        n_out = target_points if index == coil_index else source_points
        rx, ry, rz = simplify_multiturn_coil(
            coordinates[:, 0],
            coordinates[:, 1],
            coordinates[:, 2],
            n_turns=14,
            n_out=n_out,
        )
        coils.append(np.column_stack((rx, ry, rz)))
    return coils


def region_averages(r, force_p, force_q, p_hat, q_hat):
    """Return equal-arc-length regional averages and representative points."""
    next_r = np.roll(r, -1, axis=0)
    segment_length = np.linalg.norm(next_r - r, axis=1)
    segment_mid_s = np.cumsum(segment_length) - 0.5 * segment_length
    total_length = float(np.sum(segment_length))
    boundaries = np.linspace(0.0, total_length, N_REGIONS + 1)

    rows = []
    for region in range(N_REGIONS):
        if region == N_REGIONS - 1:
            mask = (segment_mid_s >= boundaries[region]) & (
                segment_mid_s <= boundaries[region + 1]
            )
        else:
            mask = (segment_mid_s >= boundaries[region]) & (
                segment_mid_s < boundaries[region + 1]
            )

        weights = segment_length[mask]
        fp_average = float(np.average(force_p[mask], weights=weights))
        fq_average = float(np.average(force_q[mask], weights=weights))
        position = np.average(r[mask], axis=0, weights=weights)
        p_direction = np.average(p_hat[mask], axis=0, weights=weights)
        q_direction = np.average(q_hat[mask], axis=0, weights=weights)
        p_direction /= np.linalg.norm(p_direction)
        q_direction /= np.linalg.norm(q_direction)

        rows.append(
            (
                region + 1,
                boundaries[region],
                boundaries[region + 1],
                fp_average,
                fq_average,
                *position,
                *p_direction,
                *q_direction,
            )
        )

    return np.asarray(rows), segment_mid_s, boundaries, total_length


def q_zero_bounded_averages(r, force_q, p_hat, q_hat):
    """Take signed peak |Fq| between zero crossings on the closed coil."""
    next_r = np.roll(r, -1, axis=0)
    segment_length = np.linalg.norm(next_r - r, axis=1)
    point_s = np.concatenate(([0.0], np.cumsum(segment_length[:-1])))
    segment_mid_s = point_s + 0.5 * segment_length
    total_length = float(np.sum(segment_length))

    crossings = []
    for index in range(len(force_q)):
        next_index = (index + 1) % len(force_q)
        f0, f1 = force_q[index], force_q[next_index]
        if f0 == 0.0:
            crossings.append(point_s[index])
        elif f0 * f1 < 0.0:
            fraction = abs(f0) / (abs(f0) + abs(f1))
            crossings.append(
                (point_s[index] + fraction * segment_length[index]) % total_length
            )

    if len(crossings) < 2:
        origin = 0.0
        shifted_mid_s = segment_mid_s
        boundaries = np.asarray([0.0, total_length])
    else:
        crossings = np.asarray(sorted(crossings))
        origin = float(crossings[0])
        shifted_crossings = np.mod(crossings - origin, total_length)
        shifted_crossings = np.unique(np.round(shifted_crossings, decimals=12))
        boundaries = np.concatenate((shifted_crossings, [total_length]))
        shifted_mid_s = np.mod(segment_mid_s - origin, total_length)

    rows = []
    for region in range(len(boundaries) - 1):
        mask = (shifted_mid_s >= boundaries[region]) & (
            shifted_mid_s < boundaries[region + 1]
        )
        region_indices = np.flatnonzero(mask)
        peak_index = region_indices[np.argmax(np.abs(force_q[region_indices]))]
        fq_peak = float(force_q[peak_index])
        position = r[peak_index]
        p_direction = p_hat[peak_index].copy()
        q_direction = q_hat[peak_index].copy()
        p_direction /= np.linalg.norm(p_direction)
        q_direction /= np.linalg.norm(q_direction)
        physical_start = (origin + boundaries[region]) % total_length
        physical_end = (origin + boundaries[region + 1]) % total_length
        rows.append(
            (region + 1, physical_start, physical_end, 0.0, fq_peak,
             *position, *p_direction, *q_direction)
        )

    return np.asarray(rows), shifted_mid_s, boundaries, total_length


def equal_3d_axes(ax, coordinates):
    center = np.mean(coordinates, axis=0)
    half_range = 0.5 * np.ptp(coordinates, axis=0).max()
    ax.set_xlim(center[0] - half_range, center[0] + half_range)
    ax.set_ylim(center[1] - half_range, center[1] + half_range)
    ax.set_zlim(center[2] - half_range, center[2] + half_range)
    ax.set_box_aspect((1, 1, 1))


def plot_component(
    ax, r, p_hat, q_hat, a, b, segment_mid_s, boundaries, rows, component
):
    region_count = len(rows)
    colors = colormaps["tab20"](np.linspace(0.0, 1.0, region_count))
    signs = ((1, 1), (-1, 1), (-1, -1), (1, -1))
    corners = np.stack(
        [
            r + 0.5 * sign_p * a * p_hat + 0.5 * sign_q * b * q_hat
            for sign_p, sign_q in signs
        ],
        axis=1,
    )

    for region in range(region_count):
        mask = (segment_mid_s >= boundaries[region]) & (
            segment_mid_s < boundaries[region + 1]
        )
        indices = np.flatnonzero(mask)
        faces = []
        for index in indices:
            next_index = (index + 1) % len(r)
            for side in range(4):
                next_side = (side + 1) % 4
                faces.append(
                    [
                        corners[index, side],
                        corners[index, next_side],
                        corners[next_index, next_side],
                        corners[next_index, side],
                    ]
                )
        if faces:
            ax.add_collection3d(
                Poly3DCollection(
                    faces,
                    facecolor=colors[region],
                    edgecolor=(*colors[region][:3], 0.18),
                    linewidth=0.2,
                    alpha=0.28,
                )
            )

        row = rows[region]
        position = row[5:8]
        average = row[3] if component == "p" else row[4]
        direction = row[8:11] if component == "p" else row[11:14]
        signed_direction = np.sign(average) * direction
        surface_offset = 0.5 * (a if component == "p" else b)
        arrow_origin = position + signed_direction * surface_offset
        arrow_length = 0.20 * abs(average) / max(
            np.max(np.abs(rows[:, 3 if component == "p" else 4])), 1.0
        )
        ax.scatter(
            *arrow_origin,
            color="black",
            s=22,
            depthshade=False,
            zorder=20,
        )
        ax.quiver(
            *arrow_origin,
            *(signed_direction * arrow_length),
            color=colors[region],
            linewidth=4,
            arrow_length_ratio=0.24,
            alpha=1.0,
            normalize=False,
        )
        label_position = arrow_origin + signed_direction * 0.015
        ax.text(
            *label_position,
            str(region + 1),
            fontsize=10,
            fontweight="bold",
            color="black",
            bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none"},
        )

    ax.plot(r[:, 0], r[:, 1], r[:, 2], color="black", lw=0.5, alpha=0.45)
    ax.set_title(
        f"Rectangular coil with {region_count} regions: average F_{component} vectors"
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    equal_3d_axes(ax, r)


def compute_tpq_force_data(
    geometry_file="HSX_coil_geometry.txt",
    coil_index=4,
    current=190050.0,
    a=0.13,
    b=0.06,
    target_points=256,
    source_points=256,
):
    """Compute the data used by the TPQ plot without creating output files."""
    xyz_coil, currents = read_coil_geometry(geometry_file)
    coils = prepare_main_coils(
        xyz_coil,
        coil_index=coil_index,
        target_points=target_points,
        source_points=source_points,
    )

    target = coils[coil_index]
    phi = np.linspace(0.0, 2.0 * np.pi, target_points, endpoint=False)
    delta = compute_delta(a, b)
    force_self = np.asarray(
        run_force_profile(
            jnp.asarray(target[:, 0]),
            jnp.asarray(target[:, 1]),
            jnp.asarray(target[:, 2]),
            jnp.asarray(phi),
            current,
            a,
            b,
            delta,
        )
    )

    magnetic_field_intercoil = np.zeros_like(target)
    for index, source in enumerate(coils):
        if index == coil_index:
            continue
        magnetic_field_intercoil += np.asarray(
            compute_B_intercoil(jnp.asarray(target), jnp.asarray(source), current)
        )

    _, _, _, t_hat, p_hat, q_hat, _, _, _, _ = get_geom(
        jnp.asarray(target[:, 0]),
        jnp.asarray(target[:, 1]),
        jnp.asarray(target[:, 2]),
        jnp.asarray(phi),
    )
    t_hat = np.asarray(t_hat)
    p_hat = np.asarray(p_hat)
    q_hat = np.asarray(q_hat)
    force_total = force_self + current * np.cross(t_hat, magnetic_field_intercoil)
    force_p = np.sum(force_total * p_hat, axis=1)
    force_q = np.sum(force_total * q_hat, axis=1)

    return {
        "target": target,
        "phi": phi,
        "t_hat": t_hat,
        "p_hat": p_hat,
        "q_hat": q_hat,
        "force_self": force_self,
        "magnetic_field_intercoil": magnetic_field_intercoil,
        "force_total": force_total,
        "force_p": force_p,
        "force_q": force_q,
    }


def main():
    geometry_file = "HSX_coil_geometry.txt"
    coil_index = 4
    coil_number = coil_index + 1
    current = 190050.0
    a, b = 0.13, 0.06
    target_points = 256
    source_points = 256

    data = compute_tpq_force_data(
        geometry_file=geometry_file,
        coil_index=coil_index,
        current=current,
        a=a,
        b=b,
        target_points=target_points,
        source_points=source_points,
    )
    target = data["target"]
    p_hat = data["p_hat"]
    q_hat = data["q_hat"]
    force_p = data["force_p"]
    force_q = data["force_q"]

    rows, segment_mid_s, boundaries, total_length = region_averages(
        target, force_p, force_q, p_hat, q_hat
    )
    q_rows, q_segment_mid_s, q_boundaries, _ = q_zero_bounded_averages(
        target, force_q, p_hat, q_hat
    )
    pressure_p_mpa = rows[:, 3] / b / 1.0e6
    pressure_q_mpa = q_rows[:, 4] / a / 1.0e6

    fig = plt.figure(figsize=(16, 12), dpi=150)
    ax_p_3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax_q_3d = fig.add_subplot(2, 2, 2, projection="3d")
    ax_p_bar = fig.add_subplot(2, 2, 3)
    ax_q_bar = fig.add_subplot(2, 2, 4)

    plot_component(
        ax_p_3d, target, p_hat, q_hat, a, b,
        segment_mid_s, boundaries, rows, "p"
    )
    plot_component(
        ax_q_3d, target, p_hat, q_hat, a, b,
        q_segment_mid_s, q_boundaries, q_rows, "q"
    )
    ax_q_3d.set_title(
        f"Rectangular coil with {len(q_rows)} zero-bounded regions: peak F_q vectors"
    )

    p_region_numbers = rows[:, 0].astype(int)
    q_region_numbers = q_rows[:, 0].astype(int)
    p_colors = colormaps["tab20"](np.linspace(0.0, 1.0, len(rows)))
    q_colors = colormaps["tab20"](np.linspace(0.0, 1.0, len(q_rows)))
    ax_p_bar.bar(p_region_numbers, pressure_p_mpa, color=p_colors)
    ax_q_bar.bar(q_region_numbers, pressure_q_mpa, color=q_colors)
    for ax, label, region_numbers, region_kind in (
        (ax_p_bar, "p", p_region_numbers, "Equal arc-length region"),
        (ax_q_bar, "q", q_region_numbers, "Zero-bounded Fq peak region"),
    ):
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel(region_kind)
        ax.set_ylabel(f"Equivalent average P_{label} [MPa]")
        ax.set_xticks(region_numbers)
        ax.grid(axis="y", alpha=0.3)

    ax_q_bar.set_ylabel("Equivalent regional peak P_q [MPa]")

    fig.suptitle(
        f"Coil {coil_number} of {N_UNIQUE_COIL_SHAPES} "
        f"(index {coil_index}): Fp in 8 equal regions; "
        "Fq in zero-bounded regions\n"
        f"I = {current / 1000:.2f} kA, coil length = {total_length:.4f} m",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    os.makedirs("results", exist_ok=True)
    os.makedirs("results_txt", exist_ok=True)
    figure_path = os.path.join(
        "results", f"TPQ_8_region_forces_coil_{coil_number}.png"
    )
    table_path = os.path.join(
        "results_txt", f"TPQ_8_region_forces_coil_{coil_number}.txt"
    )
    q_table_path = os.path.join(
        "results_txt", f"TPQ_Fq_zero_bounded_regions_coil_{coil_number}.txt"
    )
    fig.savefig(figure_path, bbox_inches="tight")

    # Keep the automatically saved PNG translucent, but show an opaque coil
    # in the interactive window for easier rotation and depth perception.
    for axis in (ax_p_3d, ax_q_3d):
        for collection in axis.collections:
            if isinstance(collection, Poly3DCollection):
                collection.set_alpha(1.0)

    header = (
        "Eight equal arc-length Fp regions; total force = self + 47 main-coil interactions.\n"
        f"coil_number={coil_number}, coil_index={coil_index}, "
        f"current_A={current}, total_length_m={total_length}\n"
        "region s_start_m s_end_m Fp_avg_N_per_m unused_Fq "
        "x_m y_m z_m p_x p_y p_z q_x q_y q_z Pp_avg_MPa_using_b_0.06m"
    )
    p_export_rows = np.column_stack((rows, pressure_p_mpa))
    np.savetxt(
        table_path, p_export_rows, fmt=["%d"] + ["%.10e"] * 14, header=header
    )

    q_header = (
        "Signed maximum-magnitude Fq in regions bounded by interpolated zero crossings.\n"
        f"coil_number={coil_number}, coil_index={coil_index}, "
        f"current_A={current}, total_length_m={total_length}\n"
        "region s_start_m s_end_m unused_Fp Fq_peak_N_per_m "
        "x_m y_m z_m p_x p_y p_z q_x q_y q_z Pq_peak_MPa_using_a_0.13m"
    )
    q_export_rows = np.column_stack((q_rows, pressure_q_mpa))
    np.savetxt(
        q_table_path, q_export_rows, fmt=["%d"] + ["%.10e"] * 14, header=q_header
    )

    print(f"Saved {figure_path}")
    print(f"Saved {table_path}")
    print(f"Saved {q_table_path}")
    print("Eight-region Fp averages:")
    for index, row in enumerate(rows):
        print(
            f"  {int(row[0])}: Fp={row[3] / 1000.0: .3f} kN/m; "
            f"Pp={pressure_p_mpa[index]: .3f} MPa"
        )
    print("Zero-bounded Fq signed peaks:")
    for index, row in enumerate(q_rows):
        print(
            f"  {int(row[0])}: Fq={row[4] / 1000.0: .3f} kN/m; "
            f"Pq={pressure_q_mpa[index]: .3f} MPa"
        )

    print("Opening interactive figure; close the window to finish.")
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
