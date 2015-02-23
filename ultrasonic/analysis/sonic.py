import numpy as np
from scipy import constants as co
f = 10
R = co.R
T = 273.15 + 21
kappa = (f+2)/f 
M = 114.23E-4

c_gas = np.sqrt(kappa * R * T / M)
print(c_gas)
