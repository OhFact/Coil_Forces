import os

os.environ.setdefault('JAX_ENABLE_X64', '1')
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import jit
from scipy.interpolate import splprep, splev, interp1d

jax.config.update("jax_enable_x64", True)

mu_0 = 4.0 * np.pi * 1e-7

#Finite Difference Derivatives
@jit
def central_diff_periodic(arr, dx):
    return (jnp.roll(arr, -1, axis=0) - jnp.roll(arr, 1, axis=0)) / (2.0 * dx)


@jit
def second_diff_periodic(arr, dx):
    return (jnp.roll(arr, -1, axis=0) - 2.0 * arr + jnp.roll(arr, 1, axis=0)) / (dx * dx)


@jit
def get_geom(Rx, Ry, Rz, phi):
    """
    Compute coil geometry and the coil-centroid frame {t, p, q}.
    """
    dphi = phi[1] - phi[0]
    r = jnp.stack([Rx, Ry, Rz], axis=1)

    # Finite derivatives for stable calculations on splines
    rp = jnp.stack([central_diff_periodic(Rx, dphi),
                    central_diff_periodic(Ry, dphi),
                    central_diff_periodic(Rz, dphi)], axis=1)

    rpp = jnp.stack([second_diff_periodic(Rx, dphi),
                     second_diff_periodic(Ry, dphi),
                     second_diff_periodic(Rz, dphi)], axis=1)

    rp_norm = jnp.linalg.norm(rp, axis=1)
    t_hat = rp / rp_norm[:, None]

    # Curvature vector curv_vec = dt/ds
    curv_vec = (rpp * rp_norm[:, None] ** 2 - rp * jnp.sum(rp * rpp, axis=1)[:, None]) / (rp_norm[:, None] ** 4)
    kappa = jnp.linalg.norm(curv_vec, axis=1)

    # Coil-centroid frame (C)
    C = (jnp.sum(r * rp_norm[:, None], axis=0) / jnp.sum(rp_norm))
    w = r - C
    w_par = (jnp.sum(w * t_hat, axis=1)[:, None]) * t_hat
    p_raw = w - w_par
    p_hat = p_raw / jnp.linalg.norm(p_raw, axis=1, keepdims=True)
    q_hat = jnp.cross(t_hat, p_hat)

    # Binormal: b = t cross n, n = curv_vec / kappa
    kappa_safe = jnp.where(kappa == 0.0, 1.0, kappa)
    curv_dir = curv_vec / kappa_safe[:, None]
    b_hat = jnp.cross(t_hat, curv_dir)

    # Components of curvature in the {p, q} frame
    k1 = jnp.sum(curv_vec * p_hat, axis=1)
    k2 = jnp.sum(curv_vec * q_hat, axis=1)

    return r, rp, rpp, t_hat, p_hat, q_hat, b_hat, kappa, k1, k2


@jit
def B_reg_centerline(r, rp, rpp, b_hat, kappa, I, a, b, delta, phi):
    """
    Compute the regularized magnetic field along the coil centerline.
    """
    dphi = phi[1] - phi[0]
    rp2 = jnp.sum(rp ** 2, axis=1)

    pref = (mu_0 * I * kappa) / (8.0 * jnp.pi)
    logt = -2.0 + jnp.log(64.0 * rp2 / (delta * a * b))
    Bloc = pref[:, None] * logt[:, None] * b_hat

    # CORRECTED VMAP: Passing all evaluation point vectors explicitly
    def single_obs_B(r_obs, rp_obs, rpp_obs, rp2_obs, phi_obs):
        diff = r_obs - r
        term1_num = jnp.cross(rp, diff)
        term1_den = (jnp.sum(diff ** 2, axis=1) + delta * a * b) ** 1.5

        dph = phi_obs - phi
        rp_obs_x_rpp_obs = jnp.cross(rp_obs, rpp_obs)
        term2_num = -rp_obs_x_rpp_obs * (1.0 - jnp.cos(dph))[:, None]
        term2_den = ((2.0 - 2.0 * jnp.cos(dph)) * rp2_obs + delta * a * b) ** 1.5

        safe_term1_den = jnp.where(term1_den == 0, 1.0, term1_den)
        safe_term2_den = jnp.where(term2_den == 0, 1.0, term2_den)

        integ1 = term1_num / safe_term1_den[:, None]
        integ2 = term2_num / safe_term2_den[:, None]

        return jnp.sum(integ1 + integ2, axis=0) * dphi

    Bint_unscaled = jax.vmap(single_obs_B)(r, rp, rpp, rp2, phi)
    Bint = (mu_0 * I / (4.0 * jnp.pi)) * Bint_unscaled

    return Bloc + Bint


@jit
def B_internal_parts(p_hat, q_hat, b_hat, kappa, k1, k2, I, a, b, delta, u, v):
    p = p_hat[None, None, :]
    q = q_hat[None, None, :]

    def G_func(x_val, y_val):
        return y_val * jnp.arctan(x_val / y_val) + 0.5 * x_val * jnp.log1p((y_val / x_val) ** 2)

    B0_p = jnp.zeros_like(u, dtype=jnp.float64)
    B0_q = jnp.zeros_like(u, dtype=jnp.float64)

    for su in (-1.0, 1.0):
        adu = a * (u - su)
        for sv in (-1.0, 1.0):
            bdv = b * (v - sv)
            G_bdv_adu = G_func(bdv, adu)
            G_adu_bdv = G_func(adu, bdv)
            B0_q = B0_q + su * sv * G_bdv_adu
            B0_p = B0_p - su * sv * G_adu_bdv

    pref0 = mu_0 * I / (4.0 * jnp.pi * a * b)
    B0_p = pref0 * B0_p
    B0_q = pref0 * B0_q
    B0 = B0_p[..., None] * p + B0_q[..., None] * q

    Bk_p = jnp.zeros_like(u, dtype=jnp.float64)
    Bk_q = jnp.zeros_like(u, dtype=jnp.float64)

    for su in (-1.0, 1.0):
        for sv in (-1.0, 1.0):
            U = u - su
            V = v - sv
            temp = a * U * U / b + b * V * V / a
            temp = jnp.maximum(temp, 1e-15)
            log_factor = jnp.log(temp)

            atan_bV_aU = jnp.arctan2(b * V, a * U)
            atan_aU_bV = jnp.arctan2(a * U, b * V)

            K_p = (
                    -2.0 * U * V * (-k2) * log_factor
                    - k1 * temp * log_factor
                    + 4.0 * a * U * U * k2 / b * atan_bV_aU
            )
            K_q = (
                    -2.0 * U * V * k1 * log_factor
                    + k2 * temp * log_factor
                    - 4.0 * b * V * V * k1 / a * atan_aU_bV
            )

            Bk_p = Bk_p + su * sv * K_p
            Bk_q = Bk_q + su * sv * K_q

    prefk = mu_0 * I / (64.0 * jnp.pi)
    Bk_p = prefk * Bk_p
    Bk_q = prefk * Bk_q
    Bk = Bk_p[..., None] * p + Bk_q[..., None] * q

    Bb = (mu_0 * I * kappa / (8.0 * jnp.pi)) * (4.0 + 2.0 * jnp.log(2.0) + jnp.log(delta)) * b_hat

    return B0, Bk, Bb


@jit
def A_reg_centerline(r, rp, rpp, t_hat, kappa, I, a, b, delta, phi):
    """
    Compute the regularized vector potential along the coil centerline.
    """
    dphi = phi[1] - phi[0]
    rp2 = jnp.sum(rp ** 2, axis=1)

    pref = (mu_0 * I) / (8.0 * jnp.pi)
    logt = -3.0 + jnp.log(64.0 * rp2 / (delta * a * b))
    Aloc = pref * logt[:, None] * t_hat

    # CORRECTED VMAP: Passing explicit evaluation points
    def single_obs_A(r_obs, rp2_obs, phi_obs):
        diff = r_obs - r
        term1_num = rp
        term1_den = jnp.sqrt(jnp.sum(diff ** 2, axis=1) + delta * a * b)

        dph = phi_obs - phi
        term2_num = rpp * (1.0 - jnp.cos(dph))[:, None]
        term2_den = jnp.sqrt((2.0 - 2.0 * jnp.cos(dph)) * rp2_obs + delta * a * b)

        safe_term1_den = jnp.where(term1_den == 0, 1.0, term1_den)
        safe_term2_den = jnp.where(term2_den == 0, 1.0, term2_den)

        integ1 = term1_num / safe_term1_den[:, None]
        integ2 = term2_num / safe_term2_den[:, None]

        return jnp.sum(integ1 + integ2, axis=0) * dphi

    Aint_unscaled = jax.vmap(single_obs_A)(r, rp2, phi)
    Aint = (mu_0 * I / (4.0 * jnp.pi)) * Aint_unscaled

    return Aloc + Aint


@jit
def self_force(I, t_hat, B_reg):
    return I * jnp.cross(t_hat, B_reg)


@jit
def run_force_profile(Rx, Ry, Rz, phi, I, a, b, delta):
    r, rp, rpp, t_hat, p_hat, q_hat, b_hat, kappa, k1, k2 = get_geom(Rx, Ry, Rz, phi)
    Bcl = B_reg_centerline(r, rp, rpp, b_hat, kappa, I, a, b, delta, phi)
    return self_force(I, t_hat, Bcl)


@jit
def self_inductance(r, rp, rpp, t_hat, kappa, I, a, b, delta, phi):
    A = A_reg_centerline(r, rp, rpp, t_hat, kappa, I, a, b, delta, phi)
    rp_norm = jnp.linalg.norm(rp, axis=1)
    dphi = phi[1] - phi[0]
    integrand = jnp.sum(A * t_hat, axis=1) * rp_norm
    L_raw = jnp.sum(integrand) * dphi / I
    return (2.0 / jnp.pi) * L_raw


@jit
def self_inductance_2D(r, rp, delta, a, b, phi):
    dphi = phi[1] - phi[0]

    def single_obs_L(r_obs, rp_obs):
        diff = r_obs - r
        dist_sq = jnp.sum(diff ** 2, axis=1)
        numerator = jnp.sum(rp_obs * rp, axis=1)
        denominator = jnp.sqrt(dist_sq + delta * a * b)
        return jnp.sum(numerator / denominator)

    integrals = jax.vmap(single_obs_L)(r, rp)
    L = (mu_0 / (4 * jnp.pi)) * jnp.sum(integrals) * dphi * dphi
    return L


@jit
def compute_delta(a, b):
    k = ((4.0 * b) / (3.0 * a)) * jnp.arctan(a / b) \
        + ((4.0 * a) / (3.0 * b)) * jnp.arctan(b / a) \
        + (b ** 2 / (6.0 * a ** 2)) * jnp.log(b / a) \
        + (a ** 2 / (6.0 * b ** 2)) * jnp.log(a / b) \
        - ((a ** 4 - 6.0 * a ** 2 * b ** 2 + b ** 4) / (6.0 * a ** 2 * b ** 2)) * jnp.log((a / b) + (b / a))
    return jnp.exp(-25.0 / 6.0 + k)


@jit
def compute_k(a, b):
    return ((4.0 * b) / (3.0 * a)) * jnp.arctan(a / b) \
        + ((4.0 * a) / (3.0 * b)) * jnp.arctan(b / a) \
        + (b ** 2 / (6.0 * a ** 2)) * jnp.log(b / a) \
        + (a ** 2 / (6.0 * b ** 2)) * jnp.log(a / b) \
        - ((a ** 4 - 6.0 * a ** 2 * b ** 2 + b ** 4) / (6.0 * a ** 2 * b ** 2)) * jnp.log((a / b) + (b / a))

def read_coil_geometry(filename, nskipline=3):
    """Read multi-turn geometry file"""
    with open(filename, 'r') as f:
        lines = f.readlines()[nskipline:]
    ncoil = 400
    xyz_coil = np.zeros((ncoil, 3, 1000))
    current = np.zeros(ncoil)
    cc_arr = np.zeros(ncoil)
    cc = 0
    icoil = 0
    cbefore = 0
    for i, lin in enumerate(lines):
        if 'end' in lin:
            break
        linsplit = lin.split()
        dat = np.array(np.double(linsplit[0:4]))

        xyz_coil[icoil, :, cc] = dat[0:3]
        cc += 1
        if dat[3] == 0.:
            cc_arr[icoil] = cc
            cc = 0
            current[icoil] = cbefore
            icoil += 1
        cbefore = dat[3]

    npt = int(cc_arr[0])
    xyz_coil = xyz_coil[:, :, 0:npt]
    ncoil = np.sum(current != 0)
    xyz_coil = xyz_coil[0:ncoil, :, :]
    current = current[0:ncoil]
    return (xyz_coil, current)


def read_coils_file(filename):
    """Read single-turn filament format (FOCUS/makegrid)"""
    coils = []
    currents_list = []
    with open(filename, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip().lower()
        if line.startswith('periods'):
            pass
        elif line.startswith('begin filament'):
            i += 1
            break
        elif line.startswith('mirror'):
            pass
        i += 1

    if i < len(lines) and lines[i].strip().lower().startswith('mirror'):
        i += 1

    current_coil_x, current_coil_y, current_coil_z = [], [], []
    current_val = None

    while i < len(lines):
        line = lines[i].strip()
        if line.lower() == 'end': break
        parts = line.split()
        if len(parts) >= 4:
            try:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                curr = float(parts[3])
                current_coil_x.append(x)
                current_coil_y.append(y)
                current_coil_z.append(z)

                if curr == 0.0:
                    if len(current_coil_x) > 1 and current_val is not None:
                        coils.append([np.array(current_coil_x), np.array(current_coil_y), np.array(current_coil_z)])
                        currents_list.append(current_val)
                    current_coil_x, current_coil_y, current_coil_z = [], [], []
                    current_val = None
                else:
                    current_val = curr
            except ValueError:
                pass
        i += 1

    if len(coils) == 0:
        raise ValueError(f"No coils found in {filename}")

    ncoil = len(coils)
    max_pts = max(len(c[0]) for c in coils)
    xyz_coil = np.zeros((ncoil, 3, max_pts))
    currents = np.zeros(ncoil)

    for ic, (coil, curr) in enumerate(zip(coils, currents_list)):
        npts = len(coil[0])
        xyz_coil[ic, 0, :npts] = coil[0]
        xyz_coil[ic, 1, :npts] = coil[1]
        xyz_coil[ic, 2, :npts] = coil[2]
        if npts < max_pts:
            xyz_coil[ic, :, npts:] = xyz_coil[ic, :, npts - 1:npts]
        currents[ic] = curr

    return xyz_coil, currents


def simplify_multiturn_coil(Rx_full, Ry_full, Rz_full, n_turns=14, n_out=256):
    """Average out a multi-turn coil into a smooth 1D centerline using continuous interpolation"""
    n_total = len(Rx_full)

    # Parameterize the full multi-turn coil from 0.0 to 1.0
    u_full = np.linspace(0, 1, n_total)

    # Create continuous interpolators for the raw data (No 'mode' argument!)
    fx = interp1d(u_full, Rx_full)
    fy = interp1d(u_full, Ry_full)
    fz = interp1d(u_full, Rz_full)

    # Average the N turns together smoothly
    pts_per_turn = max(n_out, 150)
    t_vals = np.linspace(0, 1.0 / n_turns, pts_per_turn, endpoint=False)

    Rx_simple, Ry_simple, Rz_simple = [], [], []

    for t in t_vals:
        xs, ys, zs = [], [], []
        for turn in range(n_turns):
            u_eval = t + (turn / float(n_turns))
            if u_eval >= 1.0:
                u_eval -= 1.0
            xs.append(fx(u_eval))
            ys.append(fy(u_eval))
            zs.append(fz(u_eval))

        Rx_simple.append(np.mean(xs))
        Ry_simple.append(np.mean(ys))
        Rz_simple.append(np.mean(zs))

    Rx_simple = np.array(Rx_simple)
    Ry_simple = np.array(Ry_simple)
    Rz_simple = np.array(Rz_simple)

    # Explicitly append the first point to the end to satisfy scipy's per=True requirement
    Rx_simple = np.append(Rx_simple, Rx_simple[0])
    Ry_simple = np.append(Ry_simple, Ry_simple[0])
    Rz_simple = np.append(Rz_simple, Rz_simple[0])

    # Generate the final smooth spline
    tck, u = splprep([Rx_simple, Ry_simple, Rz_simple], s=0.001, per=True)
    u_new = np.linspace(0, 1, n_out, endpoint=False)
    Rx_smooth, Ry_smooth, Rz_smooth = splev(u_new, tck)

    return np.array(Rx_smooth), np.array(Ry_smooth), np.array(Rz_smooth)


def load_hsx_coil(filename, coil_index=0, use_single_filament=None):
    if use_single_filament is None:
        use_single_filament = 'coils.' in filename.lower() or filename.endswith('.original')

    if use_single_filament:
        xyz_coil, currents = read_coils_file(filename)
        Rx, Ry, Rz = xyz_coil[coil_index, 0, :], xyz_coil[coil_index, 1, :], xyz_coil[coil_index, 2, :]

        npts = len(Rx)
        while npts > 1 and (
                Rx[npts - 1] == Rx[npts - 2] and Ry[npts - 1] == Ry[npts - 2] and Rz[npts - 1] == Rz[npts - 2]):
            npts -= 1
        if npts > 1 and np.allclose([Rx[npts - 1], Ry[npts - 1], Rz[npts - 1]], [Rx[0], Ry[0], Rz[0]], rtol=1e-10):
            npts -= 1
        return Rx[:npts], Ry[:npts], Rz[:npts], currents[coil_index], False

    else:
        xyz_coil, currents = read_coil_geometry(filename)
        Rx_full = xyz_coil[coil_index, 0, :-1]
        Ry_full = xyz_coil[coil_index, 1, :-1]
        Rz_full = xyz_coil[coil_index, 2, :-1]
        Rx, Ry, Rz = simplify_multiturn_coil(Rx_full, Ry_full, Rz_full, n_turns=14, n_out=256)
        return Rx, Ry, Rz, currents[coil_index], True

#Plotting
def style_axes(ax):
    ax.grid(True, which='both', alpha=0.3, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_force_profile(phi_np, F, component='z'):
    comp_idx = {'x': 0, 'y': 1, 'z': 2}[component]
    fig, ax = plt.subplots(figsize=(7, 4), dpi=100)
    ax.plot(phi_np, F[:, comp_idx], lw=2, color='#2E86AB')
    ax.set_xlim(0.0, 2.0 * np.pi)
    ax.set_xlabel(r'$\phi$ (rad)', fontsize=11)
    ax.set_ylabel(f'$F_{component}$ (kN/m)', fontsize=11)
    ax.set_title(f'Self-force {component}-component along coil', fontsize=12)
    style_axes(ax)
    plt.tight_layout()
    return fig


def plot_thickness_sweep(Rx, Ry, Rz, phi, I, a_list=(1e-3, 1e-2, 1e-1)):
    a_ref, b_ref = 0.13, 0.06
    delta_ref = compute_delta(a_ref, b_ref)
    F_ref = run_force_profile(jnp.array(Rx), jnp.array(Ry), jnp.array(Rz),
                              jnp.array(phi), I, a_ref, b_ref, delta_ref)
    phi_star = int(np.argmax(np.abs(np.array(F_ref[:, 2]))))
    b_vals = np.logspace(-4, 0, 80)

    fig, ax = plt.subplots(figsize=(10, 4), dpi=100)
    colors = ['salmon', 'limegreen', 'deepskyblue']
    linestyles = ['--', '--', '--']

    for idx, (a, col, ls) in enumerate(zip(a_list, colors, linestyles)):
        Fz = []
        for b_val in b_vals:
            delta = compute_delta(a, b_val)
            F = run_force_profile(jnp.array(Rx), jnp.array(Ry), jnp.array(Rz),
                                  jnp.array(phi), I, a, b_val, delta)
            Fz.append(-float(F[phi_star, 2]) * 1e-3)

        a_mm = a * 1000
        label = f'a = {a_mm:.0f} mm, 1D' if a >= 0.01 else f'a = {a_mm:.1f} mm, 1D'
        ax.semilogx(b_vals, Fz, lw=2, color=col, ls=ls, label=label)

    ax.set_xlabel(r'Conductor thickness $b$ [meters]', fontsize=11)
    ax.set_ylabel(r'z component of self-force per length, $dF_z/d\ell$ [kN/m]', fontsize=11)
    ax.legend(loc='upper right', fontsize=10, frameon=True, framealpha=0.9)
    ax.set_xlim(1e-4, 1)
    style_axes(ax)
    plt.tight_layout()
    return fig


def plot_internal_field(Rx, Ry, Rz, phi, I, a=0.13, b=0.06, idx=None):
    delta = compute_delta(a, b)
    r, rp, rpp, t_hat, p_hat, q_hat, b_hat, kappa, k1, k2 = get_geom(
        jnp.array(Rx), jnp.array(Ry), jnp.array(Rz), jnp.array(phi))
    Bcl = B_reg_centerline(r, rp, rpp, b_hat, kappa, I, a, b, delta, jnp.array(phi))

    if idx is None:
        F = self_force(I, t_hat, Bcl)
        idx = int(np.argmax(np.abs(np.array(F[:, 2]))))

    n = 120
    almost_one = 1 - 1e-6
    u = np.linspace(-almost_one, almost_one, n)
    v = np.linspace(-almost_one, almost_one, n)
    vv, uu = np.meshgrid(v, u, indexing="xy")

    B0, Bk, Bb = B_internal_parts(p_hat[idx], q_hat[idx], b_hat[idx],
                                  kappa[idx], k1[idx], k2[idx],
                                  I, a, b, delta,
                                  jnp.array(uu), jnp.array(vv))
    B_local = B0 + Bk + Bb

    B_total = B_local + Bcl[idx][None, None, :]
    Bmag = np.linalg.norm(np.array(B_total), axis=-1)
    Bz = np.array(B_total[:, :, 2])

    x_cm = vv * (b / 2.0) * 100.0
    y_cm = uu * (a / 2.0) * 100.0

    Lmag = np.linspace(0, float(Bmag.max()), 20)
    Lz = np.linspace(float(Bz.min()), float(Bz.max()), 20)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), dpi=100)

    cs0 = axes[0].contour(x_cm, y_cm, Bmag, colors='k', levels=Lmag, linewidths=0.8)
    axes[0].clabel(cs0, fmt="%.2f", fontsize=7, inline=True)
    axes[0].set_xlabel(r'$vb/2$ [cm]', fontsize=11)
    axes[0].set_ylabel(r'$ua/2$ [cm]', fontsize=11)
    axes[0].set_title(r'Reduced model $|B|$ [Tesla]', fontsize=12)
    axes[0].set_aspect('equal', 'box')

    cs1 = axes[1].contour(x_cm, y_cm, Bz, colors='k', levels=Lz, linewidths=0.8, linestyles='--')
    axes[1].clabel(cs1, fmt="%.2f", fontsize=7, inline=True)
    axes[1].set_xlabel(r'$vb/2$ [cm]', fontsize=11)
    axes[1].set_ylabel(r'$ua/2$ [cm]', fontsize=11)
    axes[1].set_title(r'Reduced model $B_z$ [Tesla]', fontsize=12)
    axes[1].set_aspect('equal', 'box')

    style_axes(axes[0])
    style_axes(axes[1])
    plt.tight_layout()
    return fig