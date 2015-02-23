import numpy as np
import uncertainties as uc
import uncertainties.unumpy as un


# All quantities are notated in mm!!
# positions of left and right first maxima on screen
xl = 48
xr = -42
yl = -1
yr = 7
X = un.uarray([xl, xr, yl, yr], 1)
#
# distance between zeroth and first order maximum on screen
d = 0.5 * un.sqrt((X[0] - X[1]) ** 2 + (X[2] - X[3]) ** 2) 

m = 1               # order of maximum
l = uc.ufloat(56, 2)# distance between grating and screen
lamb = 632.8e-6     # wavelength of HeNe laser

# lattice constant of sine grating:
K_sin = m * lamb * un.sqrt((l / d) ** 2 + 1)

