# quick calculation of the expected value of U for the given cristall parameters:
# (in SI)
import numpy as np

r_41 = 23.4e-12
n1 = 1.522
n3 = 1.477
lam = 632.8e-9
l = 20e-3
d = 2.4e-3

u = lam * d / (4 * l * r_41) * (np.sqrt(0.5 * (1 / n1 ** 2 + 1 / n3 ** 2))) ** 3
