## Exact dispersion relation for linear surface waves on arbitrary vertical shear

<p align="center">
        <img width="520" height="290" alt="image" src="https://github.com/user-attachments/assets/ce495baa-f0ba-4473-ab1f-42136ed3b992" />
</p>

For the physical geometry in the figure above, we have derived the dispersion relation 

$$\Omega_0^2 = \big[(g+\Upsilon k^2) k^2 - \Omega_0 |\partial_z U(0)| \cos \gamma \big] \frac{G_-(0)}{\partial_z G_-(0)} $$

for an arbitrary shear current $U(z)$. The Green's function and its derivative are given by

$$G_-(z) = \frac{\cosh k(z+h)}{\cosh kh} Z_1(z) + \frac{\sinh k (z+h)}{\cosh kh} Z_2(z), $$

$$\partial_z G_-(z) = k \frac{\sinh k(z+h)}{\cosh kh} Z_1(z) + k \frac{\cosh k(z+h)}{\cosh kh} Z_2(z), $$

where $Z_1(z)$ and $Z_2(z)$ are the vector elements of

$$\vec{Z}(z) = \mathcal{P} \exp \bigg( k \int_{-h}^z q(\zeta) C(\zeta+h) d\zeta \bigg) \begin{pmatrix} 0\\\ 1\end{pmatrix}, \quad C(x) = \frac{1}{2}\begin{bmatrix} - \sinh 2kx & 1 - \cosh 2kx \\\ 1 + \cosh 2kx & \sinh 2kx \end{bmatrix} . $$

### Example code

In `Example_plot.py` is the code to reproduce `Figure 2.` in the manuscript. The dispersion relations for DIM are saved in the `/DIM` folder as intrinsic phase velocities $\tilde{c} = c - U(0)$. These are only a factor $k$ away from the frequency. These are obtained with the discretisation and iterations explained in the manuscript. 

In the file `Dispersion.py`, we have implemented the product approximation for the path-ordered exponential. Below, we explain how they are implemented, along with some practical tips.

#### Dispersion function
The function `Get_dispersion` in `Dispersion.py` takes the shear current $U$, the vertical grid $z/h$, and wavenumber(s) $kh$. It is assumed that $h=1$. 

To compare it to DIM, it solves the dispersion relation (assuming $\vec{U} \parallel \vec{k}$)

$$\tilde{c}^2 = \big(g + \Upsilon k^2  - \tilde{c}|\partial_z U(0)| \big) f(0),$$

where $\tilde{c}=\Omega_0/k$ is the intrinsic phase velocity and

$$\frac{d}{dz} f(z) = 1 - k^2\big[1 + q(z)\big] f(z)^2,\qquad f(-h) =0.$$

#### Critical layers

The dispersion function $D(k,c)$ is a real-valued function where the zeros represent freely propagating waves. However, there exist two types of singularities on the real axis; when evaluating $D(k, c)$ in the complex plane $`c\in \mathbb{C}`$, it is clear that it contains poles when $\partial_z G_-(0)=0$ and a branch cut when there is a critical layer, $z_c\in(-h,0)$ such that $|c-U(z_c)| = 0$. To handle the branch cut, we utilise the analytic continuation of the dispersion function and evaluate it slightly above the real axis. If, however, there exists a pole that diverges to $\pm \infty$ depending on whether it is approached from below or above, a small imaginary part can introduce a false zero crossing, as shown in the Figure 1. and 2. below.

<table>
  <tr>
    <td colspan="2" align="center">
      <b>Dispersion for </b>$U(z) = \sqrt{gh}$ $\exp( 8 z/h)$ <b>and</b> $kh = 9$
    </td>
  </tr>
  <tr>
    <td align="center">
      <img width="490" alt="Pole on real axis"
           src="https://github.com/kjelle29/Linear-dispersion-with-shear-current/blob/main/Figs/Pole_on_real_axis.png" />
      <br/>
      <sub><b>Figure 1.</b> Pole on the real axis.</sub>
    </td>
    <td align="center">
      <img width="490" alt="Pole small imaginary part"
           src="https://github.com/kjelle29/Linear-dispersion-with-shear-current/blob/main/Figs/Pole_small_imaginary_part.png" />
      <br/>
      <sub><b>Figure 2.</b> Pole with a small imaginary part, $c \to c(1 + i / 1000)$.</sub>
    </td>
  </tr>
</table>
<table>
  <tr>
    <td colspan="2" align="center">
      <b>Dispersion for </b>$U(z) = \sqrt{gh}$ $\sin( 10 z/h)$ <b>and</b> $kh = 10$
    </td>
  </tr>
  <tr>
    <td align="center">
      <img width="490" alt="Pole on real axis"
           src="https://github.com/kjelle29/Linear-dispersion-with-shear-current/blob/main/Figs/Branch_cut.png" />
      <br/>
      <sub><b>Figure 3.</b> Dispersion function with branch cut on the real axis.</sub>
    </td>
    <td align="center">
      <img width="490" alt="Pole small imaginary part"
           src="https://github.com/kjelle29/Linear-dispersion-with-shear-current/blob/main/Figs/Near_branch_cut.png" />
      <br/>
      <sub><b>Figure 4.</b> Dispersion slightly above the real axis, $c \to c(1 + 3i / 20)$.</sub>
    </td>
  </tr>
</table>

On the other hand, evaluating the dispersion function on the real axis near/ or along the branch cut yields nonsense values, as can be seen from Figure 3. above. Since $D(k, c)$ is analytic, we evaluate it above the branch cut and take the real part. This gives a smoother function for the root-finding algorithm. <b>But</b>, this is the opposite conclusion of the handling of the poles; a small imaginary part introduced false zeros. To avoid the branch cut, but not the poles, we include only an imaginary part to the phase velocity if there exists a critical layer for that phase velocity. We move gradually away from the real axis as we approach a critical layer, as illustrated in the figure below.

<table align="center">
  <tr>
    <td colspan="1" align="center">
      <b>Example positions of branch cut and poles </b>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img width="490" alt="Pole on real axis"
           src="https://github.com/kjelle29/Linear-dispersion-with-shear-current/blob/main/Figs/Contour_dispersion_function.png" />
      <br/>
      <sub><b>Figure 5.</b>The blue markings are the poles and branch cut, and the red line shows where/how the code evaluates the dispersion function $D(k, c)$ in the presence of the difference singularities.</sub>
    </td>
  </tr>
</table>

#### Practical tips

In the file `Dispersion.py`, there are four constants one can tune: `GAMMA` controls imaginary-part smoothing near critical layers. Too-large values smear the root and can shift the physical branch, so keep it as small as you can get away with. `ITERATIONS` sets the bisection refinement depth—choose it so the final bracket width is below your required phase-speed resolution. `ROOT_F_TOL` is the acceptance threshold on $|D(k,c)|$ after refinement. `N_SCAN` controls how aggressively the initial bracket is searched for sign changes and the right-most root.

A good workflow is to pick the smallest stable `GAMMA`, set `N_SCAN` just high enough to reliably catch the desired branch, then tune `ITERATIONS` and `ROOT_F_TOL` together so refinement effort matches the accuracy your $z$ grid can actually support.

### Citing
        @misc{heinrich2026dispersion,
        title={Exact dispersion relation for linear surface waves on arbitrary vertical shear}, 
        author={Kjell S. Heinrich and Simen Å. Ellingsen},
        year={2026},
        eprint={2604.24484},
        archivePrefix={arXiv},
        url={https://arxiv.org/abs/2604.24484}, 
        }
