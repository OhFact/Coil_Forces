import os
import numpy as np
import jax.numpy as jnp

# Import physics and geometry utilities
from hsx_utilities import (
    read_coil_geometry,
    simplify_multiturn_coil,
    compute_delta,
    run_force_profile,
    get_geom
)


def main():
    print("=" * 60)
    print("Calculating Forces in the {t, p, q} Frame (256 points)")
    print("=" * 60)

    geom_file = 'HSX_coil_geometry.txt'
    if not os.path.exists(geom_file):
        raise FileNotFoundError(f"Missing {geom_file}. Please ensure it is in the directory.")

    print("Loading multi-turn geometry...")
    xyz_coil, currents = read_coil_geometry(geom_file)

    coil_idx = 0
    Rx_full = xyz_coil[coil_idx, 0, :-1]
    Ry_full = xyz_coil[coil_idx, 1, :-1]
    Rz_full = xyz_coil[coil_idx, 2, :-1]

    n_points = 256
    Rx, Ry, Rz = simplify_multiturn_coil(Rx_full, Ry_full, Rz_full, n_turns=14, n_out=n_points)
    phi = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

    # Base Parameters
    I_current = 150000.0  # 150 kA
    a_true, b_true = 0.13, 0.06
    delta_true = compute_delta(a_true, b_true)

    F_cartesian = run_force_profile(jnp.array(Rx), jnp.array(Ry), jnp.array(Rz),
                                    jnp.array(phi), I_current, a_true, b_true, delta_true)
    F_cartesian = np.array(F_cartesian)
    F_mag = np.linalg.norm(F_cartesian, axis=1)

    _, _, _, t_hat, p_hat, q_hat, _, _, _, _ = get_geom(
        jnp.array(Rx), jnp.array(Ry), jnp.array(Rz), jnp.array(phi))

    t_hat = np.array(t_hat)
    p_hat = np.array(p_hat)
    q_hat = np.array(q_hat)

    print("Projecting forces into the local frame...")
    F_t = np.sum(F_cartesian * t_hat, axis=1)
    F_p = np.sum(F_cartesian * p_hat, axis=1)
    F_q = np.sum(F_cartesian * q_hat, axis=1)
    max_ft = np.max(np.abs(F_t))
    print(f"-> Maximum Tangential Force (F_t): {max_ft:.3e} N/m (Should be ~0)")
    output_filename = f"hsx_forces_tpq_{n_points}_{coil_idx}.txt"
    print(f"Exporting data to {output_filename}...")

    # Stack the arrays into columns
    export_data = np.column_stack((
        phi,
        Rx, Ry, Rz,
        F_t, F_p, F_q,
        F_mag
    ))

    # Define a clean header
    header = (
        "Force data projected onto the local {t, p, q} coil frame.\n"
        f"Coil Index: {coil_idx}, Current: {I_current} A, Points: {n_points}\n"
        "phi [rad] | x [m] | y [m] | z [m] | F_t [N/m] | F_p [N/m] | F_q [N/m] | |F| [N/m]"
    )

    # Save to text file with high-precision scientific notation
    np.savetxt(
        f"results_txt/{output_filename}",
        export_data,
        fmt='%.8e',
        delimiter='\t',
        header=header,
        comments='# '
    )


if __name__ == "__main__":
    main()