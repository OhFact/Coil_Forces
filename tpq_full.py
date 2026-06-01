import os
import numpy as np
import jax
import jax.numpy as jnp

# Import physics and geometry utilities
from hsx_utilities import (
    read_coil_geometry,
    simplify_multiturn_coil,
    compute_delta,
    run_force_profile,
    get_geom
)


@jax.jit
def compute_B_intercoil(r_target, r_source, I_source):
    #intercoil magnetic field
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
    xyz_coil, currents = read_coil_geometry(geom_file)

    coil_idx = 0
    n_points = 256
    I_current = 150000.0  # 150 kA
    a_true, b_true = 0.13, 0.06
    delta_true = compute_delta(a_true, b_true)

    print("Simplifying geometry for all 48 coils (takes a few seconds)...")
    coils_processed = []
    for i in range(len(currents)):
        Rx_f, Ry_f, Rz_f = xyz_coil[i, 0, :], xyz_coil[i, 1, :], xyz_coil[i, 2, :]

        #Filter out the trailing zeros
        valid = ~((Rx_f == 0.0) & (Ry_f == 0.0) & (Rz_f == 0.0))
        Rx_f, Ry_f, Rz_f = Rx_f[valid], Ry_f[valid], Rz_f[valid]

        if len(Rx_f) > 1 and np.allclose([Rx_f[-1], Ry_f[-1], Rz_f[-1]], [Rx_f[0], Ry_f[0], Rz_f[0]]):
            Rx_f, Ry_f, Rz_f = Rx_f[:-1], Ry_f[:-1], Rz_f[:-1]

        rx, ry, rz = simplify_multiturn_coil(Rx_f, Ry_f, Rz_f, n_turns=14, n_out=128 if i != coil_idx else n_points)
        coils_processed.append(np.stack([rx, ry, rz], axis=1))

    r_target = coils_processed[coil_idx]
    Rx, Ry, Rz = r_target[:, 0], r_target[:, 1], r_target[:, 2]
    phi = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

    F_self = run_force_profile(jnp.array(Rx), jnp.array(Ry), jnp.array(Rz),
                               jnp.array(phi), I_current, a_true, b_true, delta_true)
    F_self = np.array(F_self)

    print("Calculating Inter-coil Forces ")
    B_inter_total = np.zeros((n_points, 3))
    for i in range(len(currents)):
        if i == coil_idx: continue
        B_inter_total += np.array(compute_B_intercoil(jnp.array(r_target), jnp.array(coils_processed[i]), I_current))

    _, _, _, t_hat, p_hat, q_hat, _, _, _, _ = get_geom(
        jnp.array(Rx), jnp.array(Ry), jnp.array(Rz), jnp.array(phi))
    t_hat, p_hat, q_hat = np.array(t_hat), np.array(p_hat), np.array(q_hat)

    # Calculate I * t x B
    F_inter = I_current * np.cross(t_hat, B_inter_total)
    F_tot = F_self + F_inter
    F_tot_mag = np.linalg.norm(F_tot, axis=1)

    F_tot_t = np.sum(F_tot * t_hat, axis=1)
    F_tot_p = np.sum(F_tot * p_hat, axis=1)
    F_tot_q = np.sum(F_tot * q_hat, axis=1)

    print(f"-> Maximum Tangential Force (F_tot_t): {np.max(np.abs(F_tot_t)):.3e} N/m (Should be ~0)")

    output_filename = f"hsx_forces_tpq_TOTAL_{n_points}_{coil_idx}.txt"
    print(f"Exporting data to {output_filename}...")

    export_data = np.column_stack((
        phi, Rx, Ry, Rz,
        F_tot_t, F_tot_p, F_tot_q, F_tot_mag
    ))

    header = (
        "TOTAL FORCE (Self + Inter-coil) projected onto the {t, p, q} coil frame.\n"
        f"Coil Index: {coil_idx}, Current: {I_current} A, Points: {n_points}\n"
        "phi [rad] | x [m] | y [m] | z [m] | F_t [N/m] | F_p [N/m] | F_q [N/m] | |F_tot| [N/m]"
    )

    os.makedirs("results_txt", exist_ok=True)
    np.savetxt(f"results_txt/{output_filename}", export_data, fmt='%.8e', delimiter='\t', header=header, comments='# ')


if __name__ == "__main__":
    main()