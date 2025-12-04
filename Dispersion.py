import numpy as np
from numba import njit, prange
from scipy.interpolate import CubicSpline

ITERATIONS = 50
C_TOL = 1e-20
D_TOL = 1e-20
GAMMA = 1e-20

STEP = 1e-4
INCREASE_RATE = 1.05
ROOT_F_TOL = 1e-10

@njit(cache=True)
def _f0_single(k, c, dz, Um, Uzzm):
    n_z = dz.size
    k2 = k * k
    ce = c + GAMMA * 1j
    f = 0.0 + 0.0j
    for i in range(n_z):
        s = np.sqrt(k2 + Uzzm[i] / (Um[i] - ce))
        sdz = s * dz[i]
        t = np.tanh(sdz)
        f = (f + t / s) / (1.0 + t * s * f)
    return f

@njit(cache=True)
def _f0(k, c, dz, Um, Uzzm):
    n_c = c.size
    out = np.zeros(n_c, np.complex128)
    for j in range(n_c):
        out[j] = _f0_single(k, c[j], dz, Um, Uzzm)
    return out

@njit(cache=True)
def _Disp(k, c, dz, Um, Uzzm, U_last, Uz0, Bo):
    n = c.size
    D = np.empty(n, np.float64)
    k2 = k * k
    f = _f0(k, c, dz, Um, Uzzm)
    for i in range(n):
        cm = c[i] - U_last
        D[i] = cm * cm - (1.0 + Bo * k2 - cm * Uz0) * f[i].real
    return D

@njit(cache=True)
def _Disp_scalar(k, c, dz, Um, Uzzm, U_last, Uz0, Bo):
    k2 = k * k
    f = _f0_single(k, c, dz, Um, Uzzm)
    cm = c - U_last
    return cm * cm - (1.0 + Bo * k2 - cm * Uz0) * f.real

@njit(cache=True)
def _refine_and_classify_root(k, cl0, cr0, Dl0, Dr0, dz, Um, Uzzm, U_last, Uz0, Bo):
    cl = cl0
    cr = cr0
    Dl = Dl0
    Dr = Dr0
    best_c = cl
    best_D_abs = abs(Dl)

    if abs(Dr) < best_D_abs:
        best_c = cr
        best_D_abs = abs(Dr)
    for _ in range(ITERATIONS):
        cm = 0.5 * (cl + cr)
        Dm = _Disp_scalar(k, cm, dz, Um, Uzzm, U_last, Uz0, Bo)
        Dm_abs = abs(Dm)
        if Dm_abs < best_D_abs:
            best_D_abs = Dm_abs
            best_c = cm
        if Dl * Dm <= 0.0:
            cr = cm
            Dr = Dm
        else:
            cl = cm
            Dl = Dm
        if (cr - cl) < C_TOL or best_D_abs < D_TOL:
            break

    is_true_root = best_D_abs < ROOT_F_TOL
    return best_c, is_true_root

@njit(cache=True)
def _c_approx(k, dz, Um, Uzzm, U_last, Uz0, Bo):
    c_start = U_last + 1e-3
    c_max = U_last + 5.0
    #c_start = -1e-2
    #c_max = 1e-2
    Dc = _Disp_scalar(k, c_start, dz, Um, Uzzm, U_last, Uz0, Bo)

    best_c = c_start
    best_D_abs = abs(Dc)

    step = STEP
    c = c_start
    while c < c_max:
        if c + step > c_max:
            step = c_max - c
        if step <= 0.0:
            break

        c_next = c + step
        D_next = _Disp_scalar(k, c_next, dz, Um, Uzzm, U_last, Uz0, Bo)

        D_next_abs = abs(D_next)
        if D_next_abs < best_D_abs:
            best_D_abs = D_next_abs
            best_c = c_next

        if Dc * D_next <= 0.0:
            root_c, is_root = _refine_and_classify_root(
                k, c, c_next, Dc, D_next, dz, Um, Uzzm, U_last, Uz0, Bo
            )
            if is_root:
                return root_c
            else:
                c = c_next
                Dc = D_next
                step = STEP / 10
                continue

        c = c_next
        Dc = D_next
        step *= INCREASE_RATE

    return best_c

@njit(parallel=True)
def _all_roots(k_vec, dz, Um, Uzzm, U_last, Uz0, Bo):
    n = k_vec.size
    out = np.empty(n, np.float64)
    for i in prange(n):
        out[i] = _c_approx(k_vec[i], dz, Um, Uzzm, U_last, Uz0, Bo)
    return out

def Get_dispersion(U_vec, z_vec, k_vec, Bo=0.0):
    z = np.asarray(z_vec, np.float64)
    U = np.asarray(U_vec, np.float64)
    if z[0] > z[-1]:
        z, U = z[::-1], U[::-1]

    cs = CubicSpline(z, U, bc_type="natural", extrapolate=True)
    Uz0 = float(cs(0.0, 1))

    dz = np.diff(z)
    Um = 0.5 * (U[:-1] + U[1:])
    Uzz = cs(z, 2)
    Uzzm = 0.5 * (Uzz[:-1] + Uzz[1:])
    k_vec = np.asarray(k_vec, np.float64)
    return _all_roots(k_vec, dz, Um, Uzzm, U[-1], Uz0, float(Bo))

def Get_dispersion_Magnus(U_vec, z_vec, k_vec, Bo = 0, interpolate = False):

    ####################################################################################################################
    ### Define variables
    ####################################################################################################################
    z = np.asarray(z_vec, float)
    U = np.asarray(U_vec, float)
    if z[0] > z[-1]:
        z, U = z[::-1], U[::-1]

    cubic_spline = CubicSpline(z, U, bc_type="natural", extrapolate=True)
    Uz0 = cubic_spline(0.0, 1)

    if interpolate and len(z) < 5000:
        z_refine = 5000 / len(z_vec)
    else:
        z_refine = 1

    Nf = (len(z) - 1) * max(1, int(z_refine)) + 1
    n_steps = Nf -1
    zf = np.linspace(z[0], z[-1], Nf)
    Uf = cubic_spline(zf)
    Uzzf = cubic_spline(zf, 2)

    dz = np.diff(zf)
    Um = 0.5 * (Uf[:-1] + Uf[1:])
    Uzzm = 0.5 * (Uzzf[:-1] + Uzzf[1:])
    ####################################################################################################################
    ###
    ####################################################################################################################

    def Dispersion(k, c, gamma=1e-10, cheb_order=6, magnus_order=3):
        try:
            interp_flag = bool(interpolate)
        except NameError:
            interp_flag = True

        c = np.asarray(c, dtype=np.complex128)
        ce = c + 1j * gamma

        try:
            import qiskit_dynamics, traceback
            from qiskit_dynamics.solvers import MagnusSolver
            from qiskit_dynamics.signals import Signal

            O0 = np.array([[0, 1], [0, 0]], dtype=complex)
            O1 = np.array([[0, 0], [1, 0]], dtype=complex)

            s0 = Signal(envelope=lambda t: np.ones_like(np.asarray(t, float)))

            def a_signal(c_eff):
                def a_env(t):
                    tt = np.asarray(t, float)
                    return k ** 2 + cubic_spline(tt, 2) / (cubic_spline(tt) - c_eff)

                return Signal(envelope=a_env)

            solver_kwargs = dict(
                operators=[O0, O1],
                rotating_frame=None,
                dt=float((zf[-1] - zf[0]) / n_steps),
                carrier_freqs=np.array([0.0, 0.0]),
                chebyshev_orders=[0, int(cheb_order)],
                expansion_order=int(magnus_order),
                integration_method='RK45',
                include_imag=[True, True]
            )
            try:
                solver = MagnusSolver(**solver_kwargs, include_imag=[True, True])
            except TypeError:
                solver = MagnusSolver(**solver_kwargs)

            y0 = np.array([0.0 + 0.0j, 1.0 + 0.0j])

            f_end = np.empty_like(ce, dtype=complex)
            for j, cj in enumerate(ce):
                try:
                    res = solver.solve(t0=float(zf[0]), n_steps=int(n_steps), y0=y0,
                                       signals=[s0, a_signal(cj)])
                    yf = getattr(res, "yf", None)
                    if yf is None:
                        yarr = getattr(res, "y", None) or getattr(res, "ys", None)
                        if yarr is None:
                            raise RuntimeError("MagnusSolver result missing yf/y/ys.")
                        yf = np.asarray(yarr)[-1]
                    u_end, v_end = np.asarray(yf)
                    f_end[j] = u_end / v_end
                except Exception as e:
                    f_end[j] = np.nan + 1j * np.nan

            cintr = c - U[-1]
            D = cintr ** 2 - (1.0 + Bo * k **2 - cintr * Uz0) * f_end
            return np.real(D)

        except Exception as e_outer:
            print("[Magnus unavailable] Falling back. Reason:", e_outer)

        ce = np.asarray(c, dtype=np.complex128) + 1e-5j
        f = np.zeros_like(ce)
        for i in range(dz.size):
            s = np.sqrt(k ** 2 + Uzzm[i] / (Um[i] - ce))
            f = (1 / s) * np.tanh(s * dz[i] + np.arctanh(s * f))

        D = (c - U[-1]) ** 2 - (1.0 + Bo * k ** 2 - (c - U[-1]) * Uz0) * f
        return np.real(D)

    def zero_crossing_pairs(x):
        i = np.where(x[:-1] * x[1:] < 0)[0]
        return np.stack((i, i + 1), axis=1)

    def get_new_bounds(D, c):
        D = np.asarray(D)
        c = np.asarray(c)
        pairs = zero_crossing_pairs(D)
        p_index = np.where(np.sum(np.abs(D[pairs]), axis=1) == min(np.sum(np.abs(D[pairs]), axis=1)))[0][0]
        return c[pairs][p_index]

    def c_approximation(k):
        c_first = np.linspace(-1e-2,1e-2, 1000)
        D_first = Dispersion(k, c_first)

        c_bounds = get_new_bounds(D_first, c_first)
        c_new = np.linspace(c_bounds[0], c_bounds[-1], 500)
        D_new = Dispersion(k, c_new)

        c_bounds = get_new_bounds(D_new, c_new)
        c_new = np.linspace(c_bounds[0], c_bounds[-1], 500)
        D_new = Dispersion(k, c_new)

        cs = get_new_bounds(D_new, c_new)
        return np.mean(cs)

    return np.array([c_approximation(ki) for ki in k_vec])