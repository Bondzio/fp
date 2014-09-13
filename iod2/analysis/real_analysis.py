import numpy as np
import pylab as plt
import os


input_dir = 'data_npy/'

# intensity and waves for iodine
f_iod_i = input_dir + 'jod_3_I.npy'
f_iod_l = input_dir + 'jod_3_l.npy'
iod_i = np.load(f_iod_i)    # intensity
iod_l = np.load(f_iod_l)    # wavelenght
l_min = 500     # minimum wavelenght at which to search for minima
l_max = 620     # maximum wavelenght at which to search for minima
l_range = (l_min < iod_l) & (iod_l < l_max)
iod_l = iod_l[l_range]
iod_i = iod_i[l_range]

# intensity and wavelength for halogene
f_halog_i = input_dir + 'halogen_02_I.npy'
f_halog_l = input_dir + 'halogen_02_l.npy'
halog_i = np.load(f_halog_i)    # intensity
halog_l = np.load(f_halog_l)    # wavelenght

# correction for air, linearly interpolated with  
# (n_l - 1) * 10 **(4) = [2.79, 2.78, 2.77, 2.76]  at corrisponding wavelenght [500, 540, 600, 680] nm   
l_lower = np.array([500, 540, 600])         # lower bounds for linear interpolation
l_upper = np.array([540, 600, 680])         # upper bounds for linear interpolation
for i in range(len(iod_l)):
    l = iod_l[i]                        
    j = np.where((l_lower < l) & (l <= l_upper))[0][0]  # find the correct intervall
    n_l = (2.79 - 0.01 * j) * 10 ** (-4) + 1  - (l - l_lower[j]) / (10 ** 6 * (l_upper[j] - l_lower[j]))    # diffraction index
    iod_l[i] = n_l * l

# searching for local minima
b_max = 4   # maximum number of neighbours 
iod_cm = 1 / iod_l * 10 ** 7
mins = np.r_[True, iod_i[1:] < iod_i[:-1]] & np.r_[iod_i[:-1] < iod_i[1:], [True]]
for b in range(2, b_max + 1):
        mins *= np.r_[[False]*b, iod_i[b:] < iod_i[:-b]] * \
                np.r_[iod_i[:-b] < iod_i[b:], b * [False]]
lm_all = iod_l[mins]
cmm_all = iod_cm[mins]
dl_all = lm_all[1:] - lm_all[:-1]
lm = [[], [], []]
cmm = [[], [], []]
dl = [[], [], []]
prog = [[], [], []]
i_prog = 0 # choose progression to display



# progression v'' = 0 -> v'
prog01 = np.where((cmm_all > 18200) * (cmm_all < 19500))[0]           # selectred point before intersect
prog02 = np.array([24, 26, 28, 30, 32, 34])      # selectred point in intersect
prog[0] = np.r_[prog01,  prog02]      # indices of progression 0, reffering to minima
lm[0] = lm_all[prog[0]] 
cmm[0] = cmm_all[prog[0]]
dl[0] = lm[0][1:] - lm[0][:-1]

# progression v'' = 1 -> v'
prog[1] = np.array([23, 25, 27, 29, 31, 33, 35, 36, 37, 38, 39, 40])      # selectred point in intersect
lm[0] = lm_all[prog[0]] 
cmm[0] = cmm_all[prog[0]]
dl[0] = lm[0][1:] - lm[0][:-1]

# progression v'' = 2 -> v'
prog[2] = np.arange(41, 49)      # until the last minima found.
lm[0] = lm_all[prog[0]] 
cmm[0] = cmm_all[prog[0]]
dl[0] = lm[0][1:] - lm[0][:-1]

# Print array
print(cmm[i_prog][::-1])

plt.close('all')
title ="$J_2$-Molecule"
fig = plt.figure()
plt.plot(iod_cm,iod_i, '-')
#plt.plot(iod_l[mins],iod_i[mins], 'o')
plt.plot(cmm[i_prog],iod_i[mins][prog[i_prog]], 'o')
#plt.plot(iod_cm,iod_i, '-')
#plt.plot(iod_cm,iod_i, '-')
#plt.plot(iod_cm[mins],iod_i[mins], 'o')
#plt.plot(halog_l,halog_i, '-')
plt.grid(True)
plt.title(title)
plt.xlabel("Wavelength $\lambda$")
plt.ylabel("Relative Intensity $I(\lambda)$")
fig.show()

title ="$J_2$-Molecule, differences in wavelenght between maxima"
fig2 = plt.figure()
#plt.plot(lm0[:-1],dl0, 'b.')
#plt.plot(lm1[:-1],dl1, 'r.')
plt.plot(cmm[i_prog][:-1],dl[i_prog], 'g.')
plt.grid(True)
plt.title(title)
plt.xlabel("Wavelength $\lambda$")
plt.ylabel("$\Delta \lambda$")
fig2.show()
