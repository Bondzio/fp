import numpy as np
import pylab as plt
import os


input_dir = 'data_npy/'
f_i = input_dir + 'jod_3_I.npy'
f_l = input_dir + 'jod_3_l.npy'
iod_i = np.load(f_i)    # intensity
iod_l = np.load(f_l)    # wavelenght

# searching for local minima
b_max = 4   # maximum number of neighbours 
mins = np.r_[True, iod_i[1:] < iod_i[:-1]]
for i in range(b_max):
        b = i + 1
        x1 = np.r_[[True]*b, iod_i[b:] < iod_i[:-b]] * \
                np.r_[iod_i[:-b] < iod_i[b:], b * [True]]
        if i:   mins *= x1
        else: mins = x1
l_min = 500     # minimum wavelenght at which to search for minima
l_max = 500     # maximum wavelenght at which to search for minima
mins *= iod_l > l_min


plt.close('all')
title ="$J_2$-Molecule"
fig = plt.figure()
plt.plot(iod_l,iod_i, '-')
plt.plot(iod_l[mins],iod_i[mins], 'o')
plt.grid(True)
plt.title(title)
plt.xlabel("Wavelength $\lambda$")
plt.ylabel("Relative Intensity $I(\lambda)$")
fig.show()
