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

from Dispersion import Get_dispersion

g = 9.81
h = 1
Yps = 0.073/1000
c0 = np.sqrt(g*h)

z = np.linspace(-1, 0, 1000)

P1 = np.array([0.9884, 5.367, 10.48, 8.784, 2.684, 0, 0])
P2 = np.array([1.098, 4.275, 3.041, -0.0086, 0.1212, 0, 0])
P3 = np.array([1.509, 2.999, 3.811, 2.172, 0.4921, 0, 0])

U1 = np.polynomial.polynomial.polyval(z, P1) / c0
U2 = np.polynomial.polynomial.polyval(z, P2) / c0
U3 = np.polynomial.polynomial.polyval(z, P3) / c0

kv1 = np.loadtxt("DIM/kv1.dat")
kv2 = np.loadtxt("DIM/kv2.dat")
kv3 = np.loadtxt("DIM/kv3.dat")

w01 = np.sqrt((1 + Yps/9.81 * kv1**2) * kv1 * np.tanh(kv1))
w02 = np.sqrt((1 + Yps/9.81 * kv2**2) * kv2 * np.tanh(kv2))
w03 = np.sqrt((1 + Yps/9.81 * kv3**2) * kv3 * np.tanh(kv3))

c_DIM1  = np.loadtxt("DIM/c_dim1.dat") / c0 + U1[-1]
c_DIM2  = np.loadtxt("DIM/c_dim2.dat") / c0 + U2[-1]
c_DIM3  = np.loadtxt("DIM/c_dim3.dat") / c0 + U3[-1]

c_KC1 = np.loadtxt("DIM/c_kc1.dat") / c0 + U1[-1]
c_KC2 = np.loadtxt("DIM/c_kc2.dat") / c0 + U2[-1]
c_KC3 = np.loadtxt("DIM/c_kc3.dat") / c0 + U3[-1]

c1_Heinrich = Get_dispersion(U1, z, kv1, Yps=Yps)
c2_Heinrich = Get_dispersion(U2, z, kv2, Yps=Yps)
c3_Heinrich = Get_dispersion(U3, z, kv3, Yps=Yps)

#######################################################################################################################

fig, ax = plt.subplots(3, 3, sharex='col', sharey='col', figsize=(6.9, 5))

ax[0, 0].plot(U1, z, '-k', lw=1)
ax[0, 1].plot(kv1, c_DIM1 * kv1 / w01, c='darkgray', lw=1.5, label='DIM', alpha=0.75)
ax[0, 1].plot(kv1, c_KC1 * kv1 / w01, ls='--', c='royalblue', lw=1, label=r'K\&C$_\textrm{1st}$')
ax[0, 1].plot(kv1, c1_Heinrich * kv1 / w01, ':', c='tomato', lw=1.5, label='Magnus 1st')
ax[0, 2].loglog(kv1, np.abs((c_KC1-c_DIM1)/c_DIM1), ls='--', c='royalblue', lw=1, label=r'K\&C$_\textrm{1st}$')
ax[0, 2].loglog(kv1, np.abs((c1_Heinrich - c_DIM1) / c_DIM1), c='tomato', lw=1, label='Magnus 1st')
ax[0, 1].legend(frameon=False), ax[0, 2].legend(frameon=False)

ax[1, 0].plot(U2, z, '-k', lw=1)
ax[1, 1].plot(kv2, c_DIM2 * kv2 / w02, c='darkgray', lw=1.5, label='DIM', alpha=0.75)
ax[1, 1].plot(kv2, c_KC2 * kv2 / w02, ls='--', c='royalblue', lw=1, label=r'K\&C$_\textrm{1st}$')
ax[1, 1].plot(kv2, c2_Heinrich * kv2 / w02, ':', c='tomato', lw=1.5, label='Magnus 1st')
ax[1, 2].loglog(kv2, np.abs((c_KC2-c_DIM2)/c_DIM2), ls='--', c='royalblue', lw=1, label=r'K\&C$_\textrm{1st}$')
ax[1, 2].loglog(kv2, np.abs((c2_Heinrich-c_DIM2)/c_DIM2), c='tomato', lw=1, label='Magnus 1st')
ax[1, 1].legend(frameon=False), ax[1, 2].legend(frameon=False)

ax[2, 0].plot(U3, z, '-k', lw=1)
ax[2, 1].plot(kv3, c_DIM3 * kv3 / w03, c='darkgray', lw=1.5, label='DIM', alpha=0.75)
ax[2, 1].plot(kv3, c_KC3 * kv3 / w03, ls='--', c='royalblue', lw=1, label=r'K\&C$_\textrm{1st}$')
ax[2, 1].loglog(kv3, c3_Heinrich * kv3 / w03, ':', c='tomato', lw=1.5, label='Magnus 1st')
ax[2, 2].loglog(kv3, np.abs((c_KC3-c_DIM3)/c_DIM3), ls='--', c='royalblue', lw=1, label=r'K\&C$_\textrm{1st}$')
ax[2, 2].loglog(kv3, np.abs((c3_Heinrich - c_DIM3) / c_DIM3), c='tomato', lw=1, label='Magnus 1st')
ax[2, 1].legend(frameon=False), ax[2, 2].legend(frameon=False)

for i in range(3):
    ax[i, 0].set_ylabel(r'$z/h$')
    ax[i, 1].set_ylabel(r'$\Omega / \omega_0$')
    ax[i, 2].set_ylabel(r'$R$')
    ax[i, 2].set_ylim(1e-12, 1)

ax[2, 0].set_xlabel(r'$U(z) / \sqrt{gh}$')
ax[2, 1].set_xlabel(r'$kh$'), ax[2, 2].set_xlabel(r'$kh$')

plt.tight_layout()
plt.show()


