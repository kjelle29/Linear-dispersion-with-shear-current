import numpy as np
from numba import njit, prange

GAMMA = 1e-2
ITERATIONS = 50
ROOT_F_TOL = 1e-8
N_SCAN = 1000

@njit(cache=True, inline="always")
def _gamma(c, U):
    if GAMMA == 0.0:
        return 0.0

    i = np.searchsorted(U, c)

    if i <= 0:
        d = U[0] - c
    elif i >= U.size:
        d = c - U[-1]
    else:
        dl = c - U[i - 1]
        dr = U[i] - c
        d = dl if dl < dr else dr

    x = 1e2 * d * d
    return 0.0 if x > 25.0 else GAMMA * np.exp(-x)


@njit(cache=True, inline="always")
def _f0_single(k2, ce, dz, Um, Uzzm):
    k = np.sqrt(k2)
    g = 0.0 + 0.0j

    for i in range(dz.size):
        d = dz[i]

        q = -Uzzm[i] / (k2 * (ce - Um[i]))
        beta = k * d * q

        sh = np.sinh(0.5 * k * d)
        ch = np.cosh(0.5 * k * d)

        A = ch * ch + sh * sh + beta * sh * ch
        B = 2.0 * sh * ch + beta * sh * sh
        C = 2.0 * sh * ch + beta * ch * ch

        g = (A * g + B) / (C * g + A)

    return g / k


@njit(cache=True, inline="always")
def _Disp_scalar(k2, gk2, c, dz, Um, Um_sorted, Uzzm, U_last, Uz0):
    gam = _gamma(c, Um_sorted)

    ce = c * (1 + 1j * gam)

    f = _f0_single(k2, ce, dz, Um, Uzzm)
    cm = ce - U_last
    D = cm * cm - (gk2 - cm * Uz0) * f

    return D.real, f

@njit(cache=True, inline="always")
def _absf(x):
    return -x if x < 0.0 else x

@njit(cache=True, inline="always", fastmath=True)
def _refine_root(k2, gk2, cl, cr, Dl, dz, Um, Um_sorted, Uzzm, U_last, Uz0):
    cm = 0.5 * (cl + cr)
    Dm, _ = _Disp_scalar(k2, gk2, cm, dz, Um, Um_sorted, Uzzm, U_last, Uz0)
    for _ in range(ITERATIONS):
        if Dl * Dm <= 0.0:
            cr = cm
        else:
            cl = cm
            Dl = Dm
        cm = 0.5 * (cl + cr)
        Dm, f = _Disp_scalar(k2, gk2, cm, dz, Um, Um_sorted, Uzzm, U_last, Uz0)
    return cm, _absf(Dm), f

@njit(cache=True, inline="always", fastmath=True)
def _largest_root(k, dz, Um, Um_sorted, Uzzm, U_last, Uz0, Yps, c_left, c_right):
    k2 = k * k
    gk2 = 9.81 + Yps * k2
    step = (c_right - c_left) / N_SCAN

    c1 = c_right
    D1, _ = _Disp_scalar(k2, gk2, c1, dz, Um, Um_sorted, Uzzm, U_last, Uz0)

    best_c = c1
    best_D = _absf(D1)

    for _ in range(N_SCAN):
        c0 = c1 - step
        D0, _ = _Disp_scalar(k2, gk2, c0, dz, Um, Um_sorted, Uzzm, U_last, Uz0)

        a0 = _absf(D0)
        if a0 < best_D:
            best_D = a0
            best_c = c0

        if D0 * D1 <= 0.0:
            rc, rd, f = _refine_root(k2, gk2, c0, c1, D0, dz, Um, Um_sorted, Uzzm, U_last, Uz0)
            if rd < ROOT_F_TOL:
                return rc, f.real
            if rd < best_D:
                best_D = rd
                best_c = rc

        c1 = c0
        D1 = D0

    _, fbest = _Disp_scalar(k2, gk2, best_c, dz, Um, Um_sorted, Uzzm, U_last, Uz0)
    return best_c, fbest.real

@njit(parallel=True, cache=True)
def _all_roots(k_vec, dz, Um, Um_sorted, Uzzm, U_last, Uz0, Yps, brackets):
    out = np.empty(k_vec.size, np.float64)
    f = np.empty(k_vec.size, np.float64)

    for i in prange(k_vec.size):
        out[i], f[i] = _largest_root(k_vec[i], dz, Um, Um_sorted, Uzzm, U_last, Uz0, Yps, brackets[i, 0], brackets[i, 1])

    r = (np.tanh(k_vec) - k_vec*f) / (k_vec * f * np.tanh(k_vec) - 1.0)
    #r = f
    return out - U_last, r, Uz0

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
