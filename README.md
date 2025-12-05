## A closed linear dispersion relation for surface waves on a shear current 

<p align="center">
        <img width="520" height="290" alt="image" src="https://github.com/user-attachments/assets/ce495baa-f0ba-4473-ab1f-42136ed3b992" />
</p>

For the physical geometry in the figure above, we have derived the dispersion relation 

$$\Omega_0^2 = \big[(g+\Upsilon k^2) k^2 - \Omega_0 |\partial_z U(0)| \cos \gamma \big] \frac{G_-(0)}{\partial_z G_-(0)} $$

for an arbitrary shear current $U(z)$. The Green's function and its derivative are given by

$$G_-(z) = \cosh k(z+h) Z_1(z) + \frac{\sinh k (z+h)}{k} Z_2(z), $$

$$\partial_z G_-(z) = k \sinh k(z+h) Z_1(z) + \cosh k(z+h) Z_2(z), $$

where $Z_1(z)$ and $Z_2(z)$ are the vector elements of

$$\vec{Z}(z) = \mathcal{P} \exp \bigg( \int_{-h}^z q(\zeta) C(\zeta+h) d\zeta \bigg) \begin{pmatrix} 0\\\ 1\end{pmatrix}, \quad C(x) = \frac{1}{2}\begin{bmatrix} - k \sinh 2kx & 1 - \cosh 2kx \\\ k^2 (1 + \cosh 2kx) & k \sinh 2kx \end{bmatrix} . $$

### Example code

In `Example_plot.py` is the code to reproduce `Figure 3.` in the manuscript. The dispersion relations for DIM and K&C1 are saved in the `/DIM` folder as intrinsic phase velocities $\tilde{c} = c - U(0)$. These are only a factor $k$ away from the frequency. These are obtained with the discretization and iterations explained in the manuscript. 

In the file `Dispersion.py`, we have implemented the first and third order Magnus approximation for the path-ordered exponential. Below, we explain how they are implemented, along with some practical tips.

#### Dispersion function
The function `Get_dispersion` in `Dispersion.py` takes the shear current $U$, nondimensionalized by $\sqrt{gh}$, the vertical grid $z/h$, and wavenumber(s) $kh$. It is assumed that $h=1$. 

To compare it to DIM, it solves the dispersion relation (assuming $\vec{U} \parallel \vec{k}$)

$$\tilde{c}^2 = \big(1 + \Upsilon k^2/g  - \tilde{c}|\partial_z U(0)| \big) f(0),$$

where $\tilde{c}=\Omega_0/k$ is the intrinsic phase velocity and

$$\frac{d}{dz} f(z) = 1 - k^2\big[1 + q(z)\big] f(z),\qquad f(-h) =0.$$

The equation above is the Rayleigh equation cast in a Riccati equation for $f(z) = G_-(z) / \partial_z G_-(z)$. Defining $s^2 = k^2(1+q)$, the derivative can be expressed as

$$\frac{d}{dz} f(z) = 1 - s^2 f(z)^2.$$

Without the interaction picture, we can write the evolution of the state ${\vec{\chi} (z) = (G_-(z), {\ } \partial_z G_-(z))^T}$ as 

$$\partial_z \vec{\chi}(z) = \begin{bmatrix}
        0 & 1\\\ s^2 & 0
    \end{bmatrix} \vec{\chi} (z) .$$

Next, we discretize the depth $\Delta = z_{i+i} - z_{i}$ and $s(z_i) = s_i$. Taking the first order Magnus term, so the ordinary exponential, we can write

$$\vec{\chi} _{i+1} = \begin{bmatrix}
        \cosh (s_i\Delta) & \frac{\sinh (s_i\Delta)}{s_i}\\
        s_i \sinh (s_i\Delta) & \cosh (s_i\Delta)
    \end{bmatrix} \vec{\chi} _i .$$

 We insert this into the equation for $f(z)$, and simplify it. The equation that solves the Rayleigh/Ricatti equation numerically is

 $$ f_{i+1} = \frac{f_i + s^{-1}_i \tanh (s_i\Delta)}{1 + s_i f_i \tanh (s_i \Delta)}= s^{-1}_i \tanh\big(s_i \Delta + \tanh^{-1}(s_i f_i) \big), \quad f_0=0.$$

 #### Magnus expansion

In `Dispersion.py`, there is also a function `Get_dispersion_Magnus`, which calculates the Magnus expansion up to third order. It has the same inputs and assumptions as `Get_dispersion(**args)`. The order is set equal to 3, but can be changed manually within the function if desirable. 
 
To implement it, we have used the class `MagnusSolver` from Qiskit, which is documented here: https://qiskit-community.github.io/qiskit-dynamics/stubs/qiskit_dynamics.solvers.MagnusSolver.html

#### Practical tips

The interval for the bisection method is found by scanning the phase speed `c` starting at `U_last + 1e-3` with an initial step size `STEP`. On each iteration, the code checks for a sign change in `D(c)` between `c` and `c + step`; if there is no sign change, it advances `c` to `c + step` and multiplies `step` by `INCREASE_RATE` so the scan speeds up. If a sign change is detected, `[c, c + step]` is treated as a bracket containing a root and passed to `_refine_and_classify_root`.

`_refine_and_classify_root` refines a candidate bracket using bisection for at most `ITERATIONS` steps. It stops early if the bracket width `cr - cl` falls below `C_TOL` or if the best function value is smaller than `D_TOL`. After refinement, the root is only accepted if `abs(D)` is below `ROOT_F_TOL`, so smaller `ROOT_F_TOL` means stricter acceptance.


### Citing
TBD

