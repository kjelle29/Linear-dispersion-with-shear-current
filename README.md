## Dispersion function
The function Get_dispersion in  'Dispersion.py'  takes the shear current $U$, nondimensionalized by $\sqrt{gh}$, the vertical grid $z/h$, and wavenumber(s) $kh$. It is assumed that $h=1$. 

To compare it to DIM, it solves the dispersion relation

$$\tilde{c}^2 = \big(1 + \Upsilon k^2/g  - \tilde{c}|U'_0| \big) f(0),$$

where $\tilde{c}=\Omega_0/k$ is the intrinsic phase velocity and

$$\frac{d}{dz} f(z) = 1 - k^2\big[1 + q(z)\big] f(z),\qquad f(-h) =0.$$

The equation above is the Rayleigh equation cast in a Riccati equation for $f(z) = G_-(z) / G_-'(z)$. Defining $s^2 = k^2(1+q)$, the derivative can be expressed as

$$\frac{d}{dz} f(z) = 1 - s^2 f(z)^2.$$

Without the interaction picture, we can write the evolution of the state ${\chi(z) = (G_-(z), G_-'(z))^T}$ as 

$$\chi'(z) = \begin{bmatrix}
        0 & 1\\\ s^2 & 0
    \end{bmatrix} \chi(z) .$$

Next, we discretize the depth $\Delta = z_{i+i} - z_{i}$ and $s(z_i) = s_i$. Taking the first order Magnus term, so the ordinary exponential, we can write

$$\chi_{i+1} = \begin{bmatrix}
        \cosh (s_i\Delta) & \frac{\sinh (s_i\Delta)}{s_i}\\
        s_i \sinh (s_i\Delta) & \cosh (s_i\Delta)
    \end{bmatrix} \chi_i .$$

 We insert this into the equation for $f(z)$, and simplify it. The equation that solves the Rayleigh/Ricatti equation numerically is

 $$ f_{i+1} = \frac{f_i + s^{-1}_i \tanh (s_i\Delta)}{1 + s_i f_i \tanh (s_i \Delta)}= s^{-1}_i \tanh\big(s_i \Delta + \tanh^{-1}(s_i f_i) \big), \quad f_0=0.$$

 ## Magnus expansion

In 'Dispersion.py', there is also a function Get_dispersion_Magnus, which calculates the Magnus expansion up to third order. It has the same inputs and assumptions as Get_dispersion(**args).
 
To implement it, we have used the class MagnusSolver from Qiskit, which is documented here: https://qiskit-community.github.io/qiskit-dynamics/stubs/qiskit_dynamics.solvers.MagnusSolver.html
