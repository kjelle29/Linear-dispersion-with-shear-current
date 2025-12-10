import numpy as np
from numba import njit, prange
from scipy.interpolate import CubicSpline

GAMMA = 1e-10
STEP = 1e-3
INCREASE_RATE = 1.1

ITERATIONS = 50
C_TOL = 1e-20
D_TOL = 1e-20
ROOT_F_TOL = 1e-10

@njit(cache=True)
def _f0_single(k, c, dz, Um, Uzzm):
    k2 = k * k
    ce = c - GAMMA * 1j
    f = 0.0 + 0.0j
    for i in range(dz.size):
        s = np.sqrt(k2 + Uzzm[i] / (Um[i] - ce))
        sdz = s * dz[i]
        t = np.tanh(sdz)
        f = (f + t / s) / (1.0 + t * s * f)
    return f

@njit(cache=True)
def _f0(k, c, dz, Um, Uzzm):
    out = np.zeros_like(c, np.complex128)
    for j in range(c.size):
        out[j] = _f0_single(k, c[j], dz, Um, Uzzm)
    return out

@njit(cache=True)
def _Disp(k, c, dz, Um, Uzzm, U_last, Uz0, Yps):
    D = np.empty_like(c, np.float64)
    k2 = k * k
    f = _f0(k, c, dz, Um, Uzzm)
    for i in range(c.size):
        cm = c[i] - U_last
        D[i] = cm * cm - (9.81 + Yps * k2 - cm * Uz0) * f[i].real
    return D

@njit(cache=True)
def _Disp_scalar(k, c, dz, Um, Uzzm, U_last, Uz0, Yps):
    k2 = k * k
    f = _f0_single(k, c, dz, Um, Uzzm)
    cm = c - U_last
    return cm * cm - (9.81 + Yps * k2 - cm * Uz0) * f.real

@njit(cache=True)
def _refine_and_classify_root(k, cl, cr, Dl, Dr, dz, Um, Uzzm, U_last, Uz0, Yps):
    best_c = cl
    best_D_abs = abs(Dl)

    if abs(Dr) < best_D_abs:
        best_c = cr
        best_D_abs = abs(Dr)
    for _ in range(ITERATIONS):
        cm = 0.5 * (cl + cr)
        Dm = _Disp_scalar(k, cm, dz, Um, Uzzm, U_last, Uz0, Yps)
        Dm_abs = abs(Dm)
        if Dm_abs < best_D_abs:
            best_D_abs = Dm_abs
            best_c = cm
        if Dl * Dm <= 0.0:
            cr = cm
        else:
            cl = cm
            Dl = Dm
        if (cr - cl) < C_TOL or best_D_abs < D_TOL:
            break

    is_true_root = best_D_abs < ROOT_F_TOL
    return best_c, is_true_root

@njit(cache=True)
def _c_approx(k, dz, Um, Uzzm, U_last, Uz0, Yps, bracket):
    c_min = bracket[0]
    c_start = bracket[1]

    Dc = _Disp_scalar(k, c_start, dz, Um, Uzzm, U_last, Uz0, Yps)

    best_c = c_start
    best_D_abs = abs(Dc)

    step = -STEP
    c = c_start

    while c > c_min:
        c_next = c + step
        if c_next < c_min:
            c_next = c_min

        D_next = _Disp_scalar(k, c_next, dz, Um, Uzzm, U_last, Uz0, Yps)

        D_next_abs = abs(D_next)
        if D_next_abs < best_D_abs:
            best_D_abs = D_next_abs
            best_c = c_next

        if Dc * D_next <= 0.0 and c != c_next:
            if c_next < c:
                cl, cr = c_next, c
                Dl, Dr = D_next, Dc
            else:
                cl, cr = c, c_next
                Dl, Dr = Dc, D_next

            root_c, is_root = _refine_and_classify_root(k, cl, cr, Dl, Dr, dz, Um, Uzzm, U_last, Uz0, Yps)
            if is_root:
                return root_c
            else:
                c = c_next
                Dc = D_next
                step = -STEP
                continue

        c = c_next
        Dc = D_next
        step *= INCREASE_RATE

        if c == c_min:
            break

    return best_c

@njit(parallel=True)
def _all_roots(k_vec, dz, Um, Uzzm, U_last, Uz0, Yps, bracket):
    out = np.empty_like(k_vec, np.float64)
    for i in prange(k_vec.size):
        out[i] = _c_approx(k_vec[i], dz, Um, Uzzm, U_last, Uz0, Yps, bracket[i])
    return out - U_last

def Get_dispersion(U_vec, z_vec, k_vec, Yps=0.0, bracket=None):
    z = np.asarray(z_vec, np.float64)
    U = np.asarray(U_vec, np.float64)
    k = np.asarray(k_vec, np.float64)

    if z[0] > z[-1]:
        z, U = z[::-1], U[::-1]

    cs = CubicSpline(z, U, bc_type="natural", extrapolate=True)
    Uz = cs(z, 1)
    Uz0 = float(cs(0.0, 1))

    if bracket is None:
        delta = np.array([np.trapezoid(Uz * np.sinh(2.0 * ki * (z + 1.0)) / np.sinh(2.0 * ki), z)for ki in k])
        c0 = np.sqrt((9.81 / k + Yps * k) * np.tanh(k))

        U0 = U[-1]
        c_KC = U0 + c0 - delta
        c_EL = U0 + np.sqrt(c0 * c0 + delta * delta) - delta

        bracket = np.array([[1.0 * c_KC[i], 1.2 * c_EL[i]] for i in range(len(k))])
    else:
        bracket = np.array([bracket for _ in range(len(k))])

    dz = np.diff(z)
    Um = 0.5 * (U[:-1] + U[1:])
    Uzz = cs(z, 2)
    Uzzm = 0.5 * (Uzz[:-1] + Uzz[1:])
    return _all_roots(k, dz, Um, Uzzm, U[-1], Uz0, float(Yps), bracket)
