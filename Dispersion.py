import numpy as np
from numba import njit, prange

GAMMA = 0.
ITERATIONS = 50
ROOT_F_TOL = 1e-8
N_SCAN = 50

@njit(cache=True, inline="always")
def _gamma(c, U):
    if GAMMA == 0.0:
        return 0.0
    i = np.searchsorted(U, c)
    dl = c - U[i - 1]
    dr = U[i] - c
    d2 = dl*dl if dl < dr else dr*dr
    x = 1e2 * d2
    return 0.0 if x > 25.0 else GAMMA * np.exp(-x)


@njit(cache=True, inline="always", fastmath=True)
def _f0_single(k2, ce, dz, Um, Uzzm):
    f = 0.0j
    for i in range(dz.size):
        s = np.sqrt(k2 + Uzzm[i] / (Um[i] - ce))
        t = np.tanh(s * dz[i])
        f = (f + t / s) / (1.0 + t * s * f)
    return f

@njit(cache=True, inline="always")
def _Disp_scalar(k2, gk2, c, dz, Um, Um_sorted, Uzzm, U_last, Uz0):
    if GAMMA == 0.0:
        gam = 0.0
    else:
        gam = _gamma(c, Um_sorted)
    ce = c * (1.0 + gam * 1j)
    f = _f0_single(k2, ce, dz, Um, Uzzm)
    cm = ce - U_last
    return (cm * cm - (gk2 - cm * Uz0) * f).real

@njit(cache=True, inline="always")
def _absf(x):
    return -x if x < 0.0 else x

@njit(cache=True, inline="always", fastmath=True)
def _refine_root(k2, gk2, cl, cr, Dl, dz, Um, Um_sorted, Uzzm, U_last, Uz0):
    cm = 0.5 * (cl + cr)
    Dm = _Disp_scalar(k2, gk2, cm, dz, Um, Um_sorted, Uzzm, U_last, Uz0)
    for _ in range(ITERATIONS):
        if Dl * Dm <= 0.0:
            cr = cm
        else:
            cl = cm
            Dl = Dm
        cm = 0.5 * (cl + cr)
        Dm = _Disp_scalar(k2, gk2, cm, dz, Um, Um_sorted, Uzzm, U_last, Uz0)
    return cm, _absf(Dm)

@njit(cache=True, inline="always", fastmath=True)
def _largest_root(k, dz, Um, Um_sorted, Uzzm, U_last, Uz0, Yps, c_left, c_right):
    k2 = k * k
    gk2 = 9.81 + Yps * k2
    step = (c_right - c_left) / N_SCAN

    c1 = c_right
    D1 = _Disp_scalar(k2, gk2, c1, dz, Um, Um_sorted, Uzzm, U_last, Uz0)

    best_c = c1
    best_D = _absf(D1)

    for _ in range(N_SCAN):
        c0 = c1 - step
        D0 = _Disp_scalar(k2, gk2, c0, dz, Um, Um_sorted, Uzzm, U_last, Uz0)

        a0 = _absf(D0)
        if a0 < best_D:
            best_D = a0
            best_c = c0

        if D0 * D1 <= 0.0:
            rc, rd = _refine_root(k2, gk2, c0, c1, D0, dz, Um, Um_sorted, Uzzm, U_last, Uz0)
            if rd < ROOT_F_TOL:
                return rc
            if rd < best_D:
                best_D = rd
                best_c = rc

        c1 = c0
        D1 = D0

    return best_c

@njit(parallel=True, cache=True)
def _all_roots(k_vec, dz, Um, Um_sorted, Uzzm, U_last, Uz0, Yps, brackets):
    out = np.empty_like(k_vec, np.float64)
    for i in prange(k_vec.size):
        out[i] = _largest_root(k_vec[i], dz, Um, Um_sorted, Uzzm, U_last, Uz0, Yps, brackets[i, 0], brackets[i, 1])
    return out - U_last

def Get_dispersion(U_vec, z_vec, k_vec, Yps=0.0, bracket=None):
    if z_vec[0] > z_vec[-1]:
        z_vec = z_vec[::-1].copy()
        U_vec = U_vec[::-1].copy()

    z = np.ascontiguousarray(z_vec)
    U = np.ascontiguousarray(U_vec)
    k = np.ascontiguousarray(k_vec)

    Uz = np.ascontiguousarray(np.gradient(U, z, edge_order=2))
    Uzz = np.ascontiguousarray(np.gradient(Uz, z, edge_order=2))
    Uz0 = float(Uz[-1])

    dz = np.ascontiguousarray(np.diff(z))
    Um = np.ascontiguousarray(0.5 * (U[:-1] + U[1:]))
    Uzzm = np.ascontiguousarray(0.5 * (Uzz[:-1] + Uzz[1:]))
    Um_sorted = np.ascontiguousarray(np.sort(Um.copy()))
    U_last = float(U[-1])

    if bracket is None:
        delta = np.array([np.trapezoid(Uz * np.sinh(2.0 * ki * (z + 1.0)) / np.sinh(2.0 * ki), z) for ki in k])
        c0 = np.sqrt((9.81 / k + Yps * k) * np.tanh(k))
        c_KC = U_last + c0 - delta
        c_EL = U_last + np.sqrt(c0 * c0 + delta * delta) - delta
        brackets = np.empty((k.size, 2), np.float64)
        brackets[:, 0] = 1.0 * c_KC
        brackets[:, 1] = 2.0 * c_EL
    else:
        bracket = np.asarray(bracket, np.float64)
        if bracket.shape == (2,):
            brackets = np.empty((k.size, 2), np.float64)
            brackets[:, 0] = bracket[0]
            brackets[:, 1] = bracket[1]
        else:
            brackets = bracket

    return _all_roots(k, dz, Um, Um_sorted, Uzzm, U_last, Uz0, float(Yps), np.ascontiguousarray(brackets))

def Get_dispersion_Magnus(U_vec, z_vec, k_vec, Yps = 0, interpolate = False):
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
            D = cintr ** 2 - (1.0 + Yps/9.81 * k **2 - cintr * Uz0) * f_end
            return np.real(D)

        except Exception as e_outer:
            print("[Magnus unavailable] Falling back. Reason:", e_outer)

        ce = np.asarray(c, dtype=np.complex128) + 1e-5j
        f = np.zeros_like(ce)
        for i in range(dz.size):
            s = np.sqrt(k ** 2 + Uzzm[i] / (Um[i] - ce))
            f = (1 / s) * np.tanh(s * dz[i] + np.arctanh(s * f))

        D = (c - U[-1]) ** 2 - (1.0 + Yps/9.81 * k ** 2 - (c - U[-1]) * Uz0) * f
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


