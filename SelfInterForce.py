import os
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from hsx_utilities import (
    read_coil_geometry,
    simplify_multiturn_coil,
    compute_delta,
    run_force_profile,
    get_geom,
    style_axes
)

@jax.jit
def compute_B_intercoil(r_target, r_source, I_source):
    #intercoil magnetic field
    mu_0 = 4.0 * jnp.pi * 1e-7
    diff = r_target[:, None, :] - r_source[None, :, :]
    dist_sq = jnp.sum(diff**2, axis=2)
    dist_inv3 = jnp.where(dist_sq > 1e-12, dist_sq**(-1.5), 0.0)

    dr_source = jnp.roll(r_source, -1, axis=0) - r_source
    cross_prod = jnp.cross(dr_source[None, :, :], diff)

    integrand = cross_prod * dist_inv3[:, :, None]
    return (mu_0 * I_source / (4.0 * jnp.pi)) * jnp.sum(integrand, axis=1)

def main():
    geom_file = 'HSX_coil_geometry.txt'

    coil_idx = 0
    n_points = 128
    current_A = 150000.0  # 150 kA
    a, b = 0.13, 0.06
    delta = compute_delta(a, b)
    raw_xyz, raw_currents = read_coil_geometry(geom_file)
    coils_processed = []

    for i in range(len(raw_currents)):
        Rx_f, Ry_f, Rz_f = raw_xyz[i, 0, :], raw_xyz[i, 1, :], raw_xyz[i, 2, :]

        # Filter out trailing zeros
        valid = ~((Rx_f == 0.0) & (Ry_f == 0.0) & (Rz_f == 0.0))
        Rx_f, Ry_f, Rz_f = Rx_f[valid], Ry_f[valid], Rz_f[valid]

        # Use multiturn simplifier
        rx, ry, rz = simplify_multiturn_coil(Rx_f, Ry_f, Rz_f, n_turns=14,
                                             n_out=128 if i != coil_idx else n_points)

        coils_processed.append(np.stack([rx, ry, rz], axis=1))

    r_target = coils_processed[coil_idx]
    Rx, Ry, Rz = r_target[:, 0], r_target[:, 1], r_target[:, 2]
    phi = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

    F_self = run_force_profile(jnp.array(Rx), jnp.array(Ry), jnp.array(Rz),
                               jnp.array(phi), current_A, a, b, delta)
    F_self = np.array(F_self)

    print("Calculating Inter-coil Forces")
    B_inter_total = np.zeros((n_points, 3))
    for i in range(len(raw_currents)):
        if i == coil_idx: continue
        B_inter_total += np.array(compute_B_intercoil(jnp.array(r_target), jnp.array(coils_processed[i]), current_A))

    _, _, _, t_hat, p_hat, q_hat, _, _, _, _ = get_geom(
        jnp.array(Rx), jnp.array(Ry), jnp.array(Rz), jnp.array(phi)
    )
    t_hat, p_hat, q_hat = np.array(t_hat), np.array(p_hat), np.array(q_hat)

    # Calculate I * t x B
    F_inter = current_A * np.cross(t_hat, B_inter_total)
    F_tot = F_self + F_inter

    # F_p = F dot p_hat (Normal outward force pushing on the wide face 'a')
    F_self_p  = np.sum(F_self  * p_hat, axis=1)
    F_inter_p = np.sum(F_inter * p_hat, axis=1)
    F_tot_p   = np.sum(F_tot   * p_hat, axis=1)

    # F_q = F dot q_hat (Binormal lateral shear force pushing on narrow face 'b')
    F_self_q  = np.sum(F_self  * q_hat, axis=1)
    F_inter_q = np.sum(F_inter * q_hat, axis=1)
    F_tot_q   = np.sum(F_tot   * q_hat, axis=1)

    print("Generating structural frame plots...")
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True, dpi=150)

    # --- Subplot 1: Normal Force (p-frame) ---
    axes[0].plot(phi, F_self_p / 1000, 'b--', lw=2, label=r'Self Force ($F_p$)')
    axes[0].plot(phi, F_inter_p / 1000, 'r--', lw=2, label=r'Inter-coil Force ($F_p$)')
    axes[0].plot(phi, F_tot_p / 1000, 'k-', lw=2.5, alpha=0.9, label=r'Total Force ($F_p$)')
    axes[0].set_title(r'Normal Force ($F_p$) Pushing on Wide Face (Outward Bursting)', fontsize=14)
    axes[0].set_ylabel('Force [kN/m]', fontsize=12)
    axes[0].axhline(0, color='gray', ls=':', lw=1.5)
    axes[0].legend(loc='upper right')
    style_axes(axes[0])

    # --- Subplot 2: Binormal Force (q-frame) ---
    axes[1].plot(phi, F_self_q / 1000, 'b--', lw=2, label=r'Self Force ($F_q$)')
    axes[1].plot(phi, F_inter_q / 1000, 'r--', lw=2, label=r'Inter-coil Force ($F_q$)')
    axes[1].plot(phi, F_tot_q / 1000, 'k-', lw=2.5, alpha=0.9, label=r'Total Force ($F_q$)')
    axes[1].set_title(r'Binormal Force ($F_q$) Pushing on Narrow Face (Lateral Shear)', fontsize=14)
    axes[1].set_ylabel('Force [kN/m]', fontsize=12)
    axes[1].set_xlabel(r'Location along the coil $\phi$ [rad]', fontsize=12)
    axes[1].axhline(0, color='gray', ls=':', lw=1.5)
    axes[1].set_xlim(0, 2 * np.pi)
    axes[1].legend(loc='upper right')
    style_axes(axes[1])

    plt.tight_layout()

    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f'Full_TPQ_Forces_{coil_idx}.png')
    fig.savefig(out_file, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    main()