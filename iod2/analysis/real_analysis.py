import numpy as np
import pylab as plt
import os
from uncertainties import unumpy as un


input_dir = 'data_npy/'

# intensity and waves for iodine
f_iod_i = input_dir + 'jod_3_I.npy'
f_iod_l = input_dir + 'jod_3_l.npy'
iod_i = np.load(f_iod_i)    # intensity
iod_l = un.uarray(np.load(f_iod_l), 0.2)    # wavelenght
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
    j = np.where((l_lower < l.n) * (l.n <= l_upper))[0][0]  # find the correct intervall
    n_l = (2.79 - 0.01 * j) * 10 ** (-4) + 1  - (l - l_lower[j]) / (10 ** 6 * (l_upper[j] - l_lower[j]))    # diffraction index
    iod_l[i] = n_l * l

# searching for local minima
b_max = 4   # maximum number of neighbours 
iod_cm = 1 / iod_l * 10 ** 7
mins = np.r_[True, iod_i[1:] < iod_i[:-1]] & np.r_[iod_i[:-1] < iod_i[1:], [True]]
for b in range(2, b_max + 1):
        mins *= np.r_[[False]*b, iod_i[b:] < iod_i[:-b]] * \
                np.r_[iod_i[:-b] < iod_i[b:], b * [False]]
mins[119] = True
mins[479] = True
mins[702] = True
mins[754] = True
mins[789] = True
mins[825] = True
mins[863] = True

lm_all = iod_l[mins]
cmm_all = iod_cm[mins]
dl_all = lm_all[1:] - lm_all[:-1]
lm = [[], [], []]
dl = [[], [], []]
cmm = [[], [], []]
dcm = [[], [], []]
prog = [[], [], []]
nums = [[], [], []]
i_prog = 2# choose progression to display



# progression v'' = 0 -> v'
prog01 = np.where((cmm_all > 18300) * (cmm_all < 197000))[0]           # selectred point before intersect
prog02 = np.array([24, 26, 28, 30, 32, 34, 36, 38])      # selectred point in intersect
prog[0] = np.r_[prog01,  prog02]      # indices of progression 0, reffering to minima

# progression v'' = 1 -> v'
prog[1] = np.array([23, 25, 27, 29, 31, 33, 35, 37, 39, 40, 42, 44, 46])      # selectred point in intersect

# progression v'' = 2 -> v'
prog21 = np.array([41, 43, 45])      # until the last minima found.
prog22 = np.arange(47, 56)      # until the last minima found.
prog[2] = np.r_[prog21,  prog22]      # indices of progression 0, reffering to minima

# calcutlating differences 
for i in range(3):
    lm[i] = lm_all[prog[i]] 
    dl[i] = lm[i][1:] - lm[i][:-1]
    cmm[i] = cmm_all[prog[i]]
    dcm[i] = cmm[i][:-1] - cmm[i][1:]
    print(un.nominal_values(cmm[i][::-1]))


# calculating energy differences
# dG[0] := G''(1) - G''(0) = (G'(n) - G''(0)) - (G'(n) - G''(1)) = w_e'' - 2 w_e x_e''
# dG[1] := G''(2) - G''(1) = (G'(n) - G''(1)) - (G'(n) - G''(2)) = w_e'' - 4 w_e x_e''
# For associating the correct 'n' , we need to numerate the progressions (find the corrisponding v') first
# This is done manually comparing with Tab 2 & 3, p. 47a, b in Staatsex-Jod2-Molekül.pdf
nums[0] = np.arange(17, 48)
nums[1] = np.arange(15, 28)
nums[2] = np.arange(9, 21)
min_n = [[],[]]         # min and max number for coorisponding v'
max_n = [[],[]]
dG = [[],[]]
for i in range(2):      # comparing v'' = (0, 1) and v'' = (1, 2), respectively 
    min_n[i] = max(min(nums[i]), min(nums[i + 1]))
    max_n[i] = min(max(nums[i]), max(nums[i + 1]))
    where0 = np.where((min_n[i] <= nums[i]) * (nums[i] <= max_n[i]))    # translating to the correct indices
    where1 = np.where((min_n[i] <= nums[i + 1]) * (nums[i + 1] <= max_n[i]))
    dG[i] = cmm[i][where0]- cmm[i + 1][where1]              # differences
    #dG_mean[i] = np.mean(dG[i])                     # take mean from all corrisponding values
    #dG_std[i] = np.std(dG[i])                       # standart deviation

# Now switch to a more OO interface to exercise more features.
fig = plt.figure(figsize = [15.0, 7.0])
ax = plt.subplot(121)
title_spectrum = "Absorption spectrum"
ax.set_title = (title_spectrum)
ax.plot(un.nominal_values(iod_cm),iod_i, '-')
for i in range(3):
    ax.plot(un.nominal_values(cmm[i]),iod_i[mins][prog[i]], 'o')
#ax.plot(iod_cm[mins],iod_i[mins], 'o')
#ax.plot(halog_l,halog_i, '-')
ax.set_xlabel("Wavenumber $\sigma$ / $\mathrm{cm^{-1}}$")
ax.set_ylabel("Relative Intensity $I(\sigma)$")

ax = plt.subplot(122)
title_dl ="differences in wavelenght between maxima"
ax.set_title(title_dl)
for i in range(3):
    ax.errorbar(un.nominal_values(cmm[i][:-1]), un.nominal_values(dcm[i]), yerr=un.std_devs(cmm[i][:-1]), xerr=un.std_devs(dcm[i]), fmt=None )
ax.set_xlabel("Wavenumber $\sigma$ / $\mathrm{cm^{-1}}$")
ax.set_ylabel("$\Delta \sigma$/ $\mathrm{cm^{-1}}$")

fig.suptitle('Iodine 2 molecule')

plt.show()


'''
plt.close('all')
title ="$J_2$-Molecule"
fig = plt.figure()
plt.plot(iod_cm,iod_i, '-')
for i in range(3):
    plt.plot(cmm[i],iod_i[mins][prog[i]], 'o')
#plt.plot(iod_cm[mins],iod_i[mins], 'o')
#plt.plot(halog_l,halog_i, '-')
plt.grid(True)
plt.title(title)
plt.xlabel("Wavenumber $\sigma$ / $\mathrm{cm^{-1}}$")
plt.ylabel("Relative Intensity $I(\sigma)$")
fig.show()

fig2 = plt.figure()
for i in range(3):
    plt.plot(cmm[i][:-1],dcm[i], '.')
plt.grid(True)
plt.title(title)
plt.xlabel("Wavenumber $\sigma$ / $\mathrm{cm^{-1}}$")
plt.ylabel("$\Delta \sigma$/ $\mathrm{cm^{-1}}$")
fig2.show()
'''
