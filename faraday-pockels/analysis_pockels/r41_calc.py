import numpy as np
import uncertainties as uc
from uncertainties.umath import sqrt

lamb = uc.ufloat(632.8E-9, 1E-9)
n1   = uc.ufloat(1.522, 0.001)
n3   = uc.ufloat(1.477, 0.001)
l    = uc.ufloat( 20E-3, 0.1E-3)
d    = uc.ufloat(2.4E-3, 0.1E-3)
U    = uc.ufloat(135  , 15)


r41 = lamb * d / (4 * l *U * 2) * sqrt(0.5*(1/n1**2 + 1/n3**2))**3 
print('{:L}'.format(r41*1e12))
