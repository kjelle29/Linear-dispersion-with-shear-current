The function Get_dispersion$(U, z, k, \Upsilon) \to c(k)$ in Dispersion.py takes the shear current, nondimensionalized by $\sqrt{gh}$, the vertical grid $z/h$, and wavenumber(s) $kh$. It is assumed that $h=1$. 

To compare it to DIM, it solves the dispersion relation 

$\tilde{c}^2 = \big(1 + \Upsilon k^2/g  - \tilde{c}|U'_0| \big) f(0)$,
