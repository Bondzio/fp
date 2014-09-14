import numpy as np
import pylab as plt
import os
from uncertainties import unumpy as un

def unv(uarray):        # returning nominal values of a uarray
    return un.nominal_values(uarray)

def usd(uarray):        # returning the standard deviations of a uarray
    return un.std_devs(uarray)

def weighted_avg_and_std(uarray):
    """
    Return the weighted average and standard deviation.
    Input: uncertainties.unumpy.uarray(nominal_values, std_devs)
    """
    values = unv(uarray)
    weights = 1 / (usd(uarray) ** 2)
    average = np.average(values, weights=weights)
    variance_biased = np.average((values-average)**2, weights=weights)  # Fast and numerically precise
    variance = variance_biased / (1 - np.sum(weights ** 2)/(np.sum(weights) **2))
    return (average, np.sqrt(variance))

def str_label_fit(i, j):
    """
    creates a latex string of the polynomial created by the fit
    """
    for j in range(deg + 1):     # write strings for poly fit label
        c = coeff[i][deg - j]
        if not j: 
            if c != 0:
                str_p =  '%.2f$' %abs(c)
                if c < 0: str_p = '- ' + str_p
                else: str_p = '+ ' + str_p
        elif j == 1: 
            if c != 0:
                str_p =  '%.2f x' %abs(c) + str_p
                if c < 0: str_p = '- ' + str_p
                else: str_p = '+ ' + str_p
        elif j == deg:
            if c != 0:
                str_p =  '%.2f x^%i' %(abs(c), j) + str_p
                if c < 0: str_p = '- ' + str_p
        else:
            if c != 0:
                str_p =  '%.2f x^%i' %(abs(c), j) + str_p
                if c < 0: str_p = '- ' + str_p
                else: str_p = '+ ' + str_p
    return( '$p_%i(v\') = '%deg + str_p)

#############################################################################################

# Plotting, general instructions
prog_n = (0, 3)     # variable needed for 'range'
deg = 2                                   # degree of polynomial fit
plt.close('all')
colors = ['b', 'g', 'r', 'pink']
labels = ['$v\'\' = %i \\rightarrow v\'$' %i for i in range(3)]


# intensity and waves for iodine
# the lab data is saved for increasing wavelength lambda. As we are working in cm^-1, we need to keep in mind this reversed order
lambda_error = 0.2                              # uncertainty of spectrometer
input_dir = 'data_npy/'
f_iod_i = input_dir + 'jod_3_I.npy'
f_iod_l = input_dir + 'jod_3_l.npy'
iod_i = np.load(f_iod_i)                  # intensity
iod_l = un.uarray(np.load(f_iod_l), lambda_error)    # wavelenght
l_min = 500     # minimum wavelenght at which to search for minima
l_max = 620     # maximum wavelenght at which to search for minima
l_range = (l_min < iod_l) & (iod_l < l_max)
iod_l = iod_l[l_range]
iod_i = iod_i[l_range]

# intensity and wavelength for halogene
f_halog_i = input_dir + 'halogen_02_I.npy'
f_halog_l = input_dir + 'halogen_02_l.npy'
halog_i = np.load(f_halog_i)    # intensity in counts
halog_l = un.uarray(np.load(f_halog_l), lambda_error)    # wavelenght
l_range = (l_min < halog_l) & (halog_l < l_max)
halog_l = halog_l[l_range]
halog_i = halog_i[l_range]

# correction for air, linearly interpolated with  
# (n_l - 1) * 10 **(4) = [2.79, 2.78, 2.77, 2.76]  at corrisponding wavelenght [500, 540, 600, 680] nm   
l_lower = np.array([500, 540, 600])         # lower bounds for linear interpolation
l_upper = np.array([540, 600, 680])         # upper bounds for linear interpolation
for i in range(len(iod_l)):
    l = iod_l[i]                 
    j = np.where((l_lower < l.n) * (l.n <= l_upper))[0][0]  # find the correct intervall
    n_l = (2.79 - 0.01 * j) * 10 ** (-4) + 1  - (l - l_lower[j]) / (10 ** 6 * (l_upper[j] - l_lower[j]))    # diffraction index
    iod_l[i] = n_l * l
iod_cm = 1 / iod_l * 10 ** 7
for i in range(len(halog_l)):
    l = halog_l[i]                 
    j = np.where((l_lower < l.n) * (l.n <= l_upper))[0][0]  # find the correct intervall
    n_l = (2.79 - 0.01 * j) * 10 ** (-4) + 1  - (l - l_lower[j]) / (10 ** 6 * (l_upper[j] - l_lower[j]))    # diffraction index
    halog_l[i] = n_l * l
halog_cm = 1 / iod_l * 10 ** 7

# searching for local minima
b_max = 4   # maximum number of neighbours 
mins = np.r_[True, iod_i[1:] < iod_i[:-1]] & np.r_[iod_i[:-1] < iod_i[1:], [True]]
for b in range(2, b_max + 1):
        mins *= np.r_[[False]*b, iod_i[b:] < iod_i[:-b]] * \
                np.r_[iod_i[:-b] < iod_i[b:], b * [False]]
mins[[119, 479, 702, 754, 789, 825, 863]] = True        # manually adding band heads

lm_all = iod_l[mins]
cmm_all = iod_cm[mins]
dl_all = lm_all[1:] - lm_all[:-1]
# since there are three progressions, we define all used quantities as dictionaries 
# which are then index with the number of progression ( = v'')
prog    = {}        # indices of elements of progression (with respect to array of all minima, "mins")
lm, cmm = {}, {}    # wavelength in nm / wavenumber in cm^-1  of points of a progression
dl, dcm = {}, {}    # differences of wavelength in nm / wavenumber in cm^-1 between to successive points of a progression 
nums    = {}        # numbering of progression (i = v')
coeff, covA = {}, {}     # coefficients and covariance matrix from polinomial fit


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

# Plotting the absorbtion spectrum
fig = plt.figure(figsize = [7.0, 7.0])
ax = plt.subplot(111)
title_spectrum = "Absorption spectrum"
ax.set_title(title_spectrum)
ax.plot(un.nominal_values(iod_cm),iod_i, '-', color='chocolate')
#ax.plot(unv(halog_cm),halog_i, '-')    # does not fit the intensitiy with the iodine!
for i in range(prog_n[0], prog_n[1]):
    ax.plot(unv(cmm[i]),iod_i[mins][prog[i]], '.', label=labels[i])
ax.set_xlabel("Wavenumber $\sigma$ / $\mathrm{cm^{-1}}$")
ax.set_ylabel("Intensity $I(\sigma)$ / counts" )
ax.legend(loc='lower left')

# calculating energy differences
# For associating the correct 'n' , we need to numerate the progressions (find the corrisponding v') first
# This is done manually comparing with Tab 2 & 3, p. 47a, b in Staatsex-Jod2-Molekül.pdf
# As data is saved in the reversed order (increasing lambda, not energy), we need to invert the numbering
nums[0] = np.arange(46, 16, -1) + 0.5
nums[1] = np.arange(26, 14, -1) + 0.5
nums[2] = np.arange(19,  8, -1) + 0.5

# Calculating constants w_e', w_e x_e' and w_e y_e' with the Birge sponer diagram
# Polynomial fit for dcm
for i in range(prog_n[0], prog_n[1]):
    weights = 1 / (usd(dcm[i]) ** 2)
    coeff[i], covA[i] = np.polyfit(nums[i], unv(dcm[i]), deg, full=False, w=weights, cov=True)

# Birge Sponer plots
fig2 = plt.figure(figsize = [21.0, 7.0])
n_min = 0
n_max = 70
n_grid = np.linspace(n_min, n_max, 200)
for i in range(prog_n[0], prog_n[1]):
    ax = plt.subplot(1, 3, i + 1)
    title_dl = '$v\'\' = %i \\rightarrow v\'$' %i
    label_fit = str_label_fit(i, j)
    ax.set_title(title_dl)
    ax.plot(nums[i], unv(dcm[i]), '.', color=colors[i] )
#    ax.errorbar(unv(cmm[i]), unv(dcm[i]), yerr=usd(cmm[i][:-1]), xerr=usd(dcm[i]), fmt=None )
    ax.plot(n_grid, np.polyval(coeff[i], n_grid), '-', color=colors[i], label=label_fit)
    ax.fill_between(n_grid, \
            np.polyval(coeff[i] - np.sqrt(np.diag(covA[i])), n_grid), \
            np.polyval(coeff[i] + np.sqrt(np.diag(covA[i])), \
            n_grid), facecolor=colors[i], color=colors[i], alpha=0.3 )
#    ax.plot(unv(cmm[i]), unv(dcm[i]), '.' )
    ax.set_xlim(n_min, n_max + 0.5)
    ax.set_xlabel("$v\' + \\frac{1}{2}$")
    ax.set_ylim(0)
    ax.set_ylabel("$\Delta \sigma \, / \, \mathrm{cm^{-1}}$")
    ax.legend(loc='upper center')
    xticks = np.arange(n_min, n_max + 1, 10) + 0.5  # assign values, at which xticks appear
    ax.set_xticks(xticks)               # set xticks
    #xticklabels = ["", "", ...]        # specify, what is written at each xtick
    #ax.set_xticklabels(xticklabels, fontsize=20)


# Calculating constants for ground level
# dG[0] := G''(1) - G''(0) = (G'(n) - G''(0)) - (G'(n) - G''(1)) = w_e'' - 2 w_e x_e''
# dG[1] := G''(2) - G''(1) = (G'(n) - G''(1)) - (G'(n) - G''(2)) = w_e'' - 4 w_e x_e''
min_n, max_n = {}, {}         # min and max number for coorisponding v'
dG, dG_avg, dG_std = {}, {}, {}
for j in range(2):      # comparing v'' = (0, 1) and v'' = (1, 2), respectively 
    min_n[j] = max(min(nums[j]), min(nums[j + 1]))
    max_n[j] = min(max(nums[j]), max(nums[j + 1]))
    where0 = np.where((min_n[j] <= nums[j]) * (nums[j] <= max_n[j]))    # translating to the correct indices
    where1 = np.where((min_n[j] <= nums[j + 1]) * (nums[j + 1] <= max_n[j]))
    dG[j] = cmm[j][where0]- cmm[j + 1][where1]              # differences
    dG_avg[j], dG_std[j] = weighted_avg_and_std(dG[j])      # weighted mean and standard deviation


fig.suptitle('Iodine 2 molecule - absorbtion spectrum')
fig2.suptitle('Iodine 2 molecule - Birge-Sponer plots')
plt.show()

