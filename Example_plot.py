import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],                              # Times look
    "text.latex.preamble": r"\usepackage{newtxtext}\usepackage{newtxmath}",
    "font.size": 8,                                       # base size
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,

    "figure.figsize": [6.9, 4.1],                         # width fits JFM double column
    "figure.dpi": 300,
    "savefig.dpi": 600,

    # Journal-ish axes/ticks
    "axes.linewidth": 0.7,
    "lines.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.titlepad": 2.0,
    "axes.labelpad": 1.5,
    "axes.unicode_minus": False,
})
from Get_dispersion import Get_dispersion

g = 9.81
h = 1
Yps = 0.073/1000
c0 = np.sqrt(g*h)

z = np.linspace(-1, 0, 100000)

P1 = np.array([0.9884, 5.367, 10.48, 8.784, 2.684, 0, 0])
P2 = np.array([1.098, 4.275, 3.041, -0.0086, 0.1212, 0, 0])
P3 = np.array([1.509, 2.999, 3.811, 2.172, 0.4921, 0, 0])

U1 = np.polynomial.polynomial.polyval(z, P1)
U2 = np.polynomial.polynomial.polyval(z, P2)
U3 = np.polynomial.polynomial.polyval(z, P3)

kv1 = np.loadtxt("DIM/kv1.dat")
kv2 = np.loadtxt("DIM/kv2.dat")
kv3 = np.loadtxt("DIM/kv3.dat")

w01 = np.sqrt((9.81 + Yps * kv1**2) * kv1 * np.tanh(kv1))
w02 = np.sqrt((9.81 + Yps * kv2**2) * kv2 * np.tanh(kv2))
w03 = np.sqrt((9.81 + Yps * kv3**2) * kv3 * np.tanh(kv3))

c_DIM1  = np.loadtxt("DIM/c_dim1.dat") / c0
c_DIM2  = np.loadtxt("DIM/c_dim2.dat") / c0
c_DIM3  = np.loadtxt("DIM/c_dim3.dat") / c0

c_KC1 = np.loadtxt("DIM/c_kc1.dat") / c0
c_KC2 = np.loadtxt("DIM/c_kc2.dat") / c0
c_KC3 = np.loadtxt("DIM/c_kc3.dat") / c0

c1_ = Get_dispersion(U1, z, kv1, Yps=Yps) / c0
print('1')
c2_ = Get_dispersion(U2, z, kv2,Yps=Yps) / c0
print('2')
c3_ = Get_dispersion(U3, z, kv3, Yps=Yps) / c0
print('3')

#######################################################################################################################

fig, ax = plt.subplots(3, 3, sharex='col', sharey='col', figsize=(6.9, 4.5))

ax[0, 0].plot(U1 / c0, z, '-k', lw=1)
ax[0, 1].plot(kv1, c_DIM1 * kv1 / w01, c='darkgray', lw=1.5, label='DIM', alpha=0.75)
ax[0, 1].plot(kv1, c_KC1 * kv1 / w01, ls='--', c='royalblue', lw=1, label=r'K\&C$_\textrm{1st}$')
ax[0, 1].plot(kv1, c1_ * kv1 / w01, ':', c='tomato', lw=1.5, label='Magnus 1st')
ax[0, 2].loglog(kv1, np.abs((c_KC1-c_DIM1)/c_DIM1), ls='--', c='royalblue', lw=1, label=r'K\&C$_\textrm{1st}$')
ax[0, 2].loglog(kv1, np.abs((c1_-c_DIM1)/c_DIM1), c='tomato', lw=1, label='Magnus 1st')
ax[0, 1].legend(frameon=False, loc='upper left'), ax[0, 2].legend(frameon=False, loc='center left')

ax[1, 0].plot(U2 / c0, z, '-k', lw=1)
ax[1, 1].plot(kv2, c_DIM2 * kv2 / w02, c='darkgray', lw=1.5, label='DIM', alpha=0.75)
ax[1, 1].plot(kv2, c_KC2 * kv2 / w02, ls='--', c='royalblue', lw=1, label=r'K\&C$_\textrm{1st}$')
ax[1, 1].plot(kv2, c2_ * kv2 / w02, ':', c='tomato', lw=1.5, label='Magnus 1st')
ax[1, 2].loglog(kv2, np.abs((c_KC2-c_DIM2)/c_DIM2), ls='--', c='royalblue', lw=1, label=r'K\&C$_\textrm{1st}$')
ax[1, 2].loglog(kv2, np.abs((c2_-c_DIM2)/c_DIM2), c='tomato', lw=1, label='Magnus 1st')
ax[1, 1].legend(frameon=False, loc='upper left'), ax[1, 2].legend(frameon=False, loc='center left')

ax[2, 0].plot(U3 / c0, z, '-k', lw=1)
ax[2, 1].plot(kv3, c_DIM3 * kv3 / w03, c='darkgray', lw=1.5, label='DIM', alpha=0.75)
ax[2, 1].plot(kv3, c_KC3 * kv3 / w03, ls='--', c='royalblue', lw=1, label=r'K\&C$_\textrm{1st}$')
ax[2, 1].semilogx(kv3, c3_ * kv3 / w03, ':', c='tomato', lw=1.5, label='Magnus 1st')
ax[2, 2].loglog(kv3, np.abs((c_KC3-c_DIM3)/c_DIM3), ls='--', c='royalblue', lw=1, label=r'K\&C$_\textrm{1st}$')
ax[2, 2].loglog(kv3, np.abs((c3_-c_DIM3)/c_DIM3), c='tomato', lw=1, label='Magnus 1st')
ax[2, 1].legend(frameon=False, loc='upper left'), ax[2, 2].legend(frameon=False, loc='center left')

for i in range(3):
    ax[i, 0].set_ylabel(r'$z/h$')
    ax[i, 1].set_ylabel(r'$\Omega_0 / \omega_0$')
    ax[i, 2].set_ylabel(r'$R$')
    ax[i, 2].set_ylim(1e-13, 1)

ax[2, 0].set_xlabel(r'$U(z) / \sqrt{gh}$')
ax[2, 1].set_xlabel(r'$kh$'), ax[2, 2].set_xlabel(r'$kh$')

plt.tight_layout()
plt.show()


