import numpy as np
import pylab as plt
import os
import uncertainties as uc
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

def chi2_theta(x, y, sigma, coeff, dx, dy, deg=1):
    return np.sum(((np.polyval(coeff + [dx, dy], x) - y) / sigma) ** 2 )

def chi2_beta(x, y, sigma, eigvec, beta, deg=1):
    a = np.dot(beta, eigvec.T)
    summ = 0
    for j in range(len(x)):
        xj = x[j]
        yj = y[j]
        sigmaj = sigma[j]
        summ +=  ((yj - a[0] * xj - a[1]) / sigmaj)** 2
    return(summ)

def str_label_fit(i, j):
    """
    creates a latex string of the polynomial created by the fit
    """
    for j in range(deg[i] + 1):     # write strings for poly fit label
        c = coeff[i][deg[i] - j]
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
        elif j == deg[i]:
            if c != 0:
                str_p =  '%.2f x^%i' %(abs(c), j) + str_p
                if c < 0: str_p = '- ' + str_p
        else:
            if c != 0:
                str_p =  '%.2f x^%i' %(abs(c), j) + str_p
                if c < 0: str_p = '- ' + str_p
                else: str_p = '+ ' + str_p
    return( '$p_%i(v\') = '%deg[i] + str_p)

#############################################################################################

# general parameters
prog_n = (0, 3)     # variable needed for 'range'
deg = [1, 1, 1]                                # degree of polynomial fit
lambda_error = 0.05                        # uncertainty of spectrometer in lambda (nonimal = 0.6)
plt.close('all')
colors = ['b', 'g', 'r', 'pink']
labels = ['$v\'\' = %i \\rightarrow v\'$' %i for i in range(3)]


# intensity and waves for iodine
# the lab data is saved for increasing wavelength lambda. As we are working in cm^-1, we need to keep in mind this reversed order
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
coeff, covA = {}, {}    # coefficients and covariance matrix from polinomial fit
chi2_min, n_d, gof = {}, {}, {}    # chi square for poly fit, number of ind. data points, goodness-of-fit
beta, beta_sd = {}, {}      # uncorrelated parameters and their std devs
eigvec = {}


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


# Calculating w_e' = w1 and w_e x_e' = wx1 for first electronic level with Birge-Sponer plots
# dG(v' + 1/2) = w_e' - w_e x_e' * (2v + 2) = a + b v (latter one is solution of polinomial fit)
# =>    w_e' = a + b
#       w_e x_e' = (a - w_e') / 2
# similar for deg = 2
w1      = un.uarray([0]*3, 0)
wx1     = un.uarray([0]*3, 0)
wy1     = un.uarray([0]*3, 0)
v_diss  = un.uarray([0]*3, 0)
D_0     = un.uarray([0]*3, 0)
label_c = [[],[],[]]
for i in range(prog_n[0], prog_n[1]):   # Polynomial fit for dG(v' + 1/2)
    weights = 1 / (usd(dcm[i]) ** 2)
    coeff[i], covA[i] = np.polyfit(nums[i], unv(dcm[i]), deg[i], full=False, w=weights, cov=True)
    c = un.uarray(coeff[i], np.sqrt(np.diag(covA[i])))
    # diagonalize covariance matrix
    eigval, eigvec[i] = np.linalg.eig(covA[i])
    beta[i] = np.linalg.solve(eigvec[i], coeff[i]) # uncorrelated parameters
    A = np.matrix(eigvec[i]).I * np.matrix(covA[i]) * np.matrix(eigvec[i])
    beta_sd[i] = np.sqrt(np.diag(A))            # standard deviation for beta
    # minimized chi square
    chi2_min[i] = np.sum(((np.polyval(coeff[i], nums[i]) - unv(dcm[i])) / usd(dcm[i])) ** 2 )
    n_d[i] = len(dcm[i]) - (deg[i] + 1)
    gof[i] = chi2_min[i] / n_d[i]
    if deg[i] == 1:
        wx1[i] = -0.5 * c[0]
        w1[i] = c[1] - c[0]
        v_diss[i] = -c[1] / c[0]        # solving for  p(v) = 0
    if deg[i] == 2:
        wy1[i] = c[0] / 3
        wx1[i] = c[0] - 0.5 * c[1]
        w1[i] = 11 / 12 * c[0] - c[1] + c[2]
        v_diss[i] = (-c[1] - un.sqrt([c[1] ** 2 - 4 * c[0] * c[2]])[0] )/ (2 * c[0]) # solving for p(v) = 0
    v_grid_n = np.arange(0, v_diss[i].n) + 0.5        # v grid for summation over all dG values, nominal value
    #v_grid_upper = np.arange(0, v_diss[i].n + v_diss[i].s) + 0.5        # v grid for v_diss = upper value
    #v_grid_lower = np.arange(0, v_diss[i].n - v_diss[i].s) + 0.5        # v grid for v_diss = lower value
    D_0_n = np.sum(np.polyval(c, v_grid_n))     # dissipation energy D_0, nominal value
    #D_0_upper = np.sum(np.polyval(c, v_grid_upper))     # dissipation energy D_0, nominal value
    #D_0_lower = np.sum(np.polyval(c, v_grid_lower))     # dissipation energy D_0, nominal value
    # create labels for plot
    label_c[i] = '$\omega_e\' \\ = ' + '{:L}'.format(w1[i]) + '$\n' \
            '$\omega_e x_e\' = ' + '{:L}'.format(wx1[i]) + '$'
    if deg[i] == 2:
        label_c[i] += '\n$\omega_e y_e\' = ' + '{:L}'.format(wy1[i]) + '$'

# Birge Sponer plots
fig2 = plt.figure(figsize = [21.0, 7.0])
for i in range(prog_n[0], prog_n[1]):
    ax = plt.subplot(1, 3, i + 1)
    v_min = 0
    v_max = [70, 70, 80][i]
    v_grid = np.linspace(v_min, v_max + 1, 200)
    title_dl = '$v\'\' = %i \\rightarrow v\'$' %i
    label_fit = str_label_fit(i, j)
    ax.set_title(title_dl)
    ax.errorbar(nums[i], unv(dcm[i]), yerr=usd(cmm[i][:-1]), color=colors[i], ls='dots', marker='.' )
    ax.plot(v_grid, np.polyval(coeff[i], v_grid), '-', color=colors[i], label=label_fit)
    ax.plot([0], [0], ',', alpha=0, label=label_c[i])      # add calculated parameters to plot
    # plot errors of poly-fit as shaded areas
    ax.fill_between(v_grid, \
            np.polyval(coeff[i] - np.sqrt(np.diag(covA[i])), v_grid), \
            np.polyval(coeff[i] + np.sqrt(np.diag(covA[i])), v_grid), \
            facecolor=colors[i], color=colors[i], alpha=0.3 )
    ax.set_xlim(v_min, v_max + 0.5)
    ax.set_xlabel("$v\' + \\frac{1}{2}$")
    xticks = np.arange(v_min, v_max + 1, 10) + 0.5  # assign values, at which xticks appear
    ax.set_xticks(xticks)               # set xticks
    #xticklabels = ["", "", ...]        # specify, what is written at each xtick
    #ax.set_xticklabels(xticklabels, fontsize=20)
    ax.set_ylim(0, [150, 250, 150][i])
    ax.set_ylabel("$\Delta \sigma \, / \, \mathrm{cm^{-1}}$")
    leg = ax.legend(loc='upper center', fancybox=True)
    leg.get_frame().set_visible(False)

# chi2 plots
fig3 = plt.figure(figsize = [21.0, 7.0])
for i in range(3):
    if deg[i] == 1:
        ax = plt.subplot(1, 3, i + 1)
        n = 11
        L = 2 * beta_sd[i]
        x = np.linspace(-L[0], L[0], n)
        y = np.linspace(-L[1], L[1], n)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros([n, n])
        Z2 = np.zeros([n, n])
        for j in range(n):
            for k in range(n):
                beta_jk = beta[i] + np.array([X[j,k], Y[j,k]])
                Z[j,k] = chi2_beta(nums[i], unv(dcm[i]), usd(dcm[i]), eigvec[i], beta_jk, deg=1)
                Z2[j,k] = chi2_theta(nums[i], unv(dcm[i]), usd(dcm[i]), coeff[i], X[j,k], Y[j,k], deg=1)
        Z -= chi2_min[i]
        Z2 -= chi2_min[i]
        levs = [1]
        CS = ax.contour(X, Y, Z, levels=levs)
        ax.plot([-beta_sd[i][0], -beta_sd[i][0]],[-L[1], L[1]], 'k--')
        ax.plot([beta_sd[i][0], beta_sd[i][0]],[-L[1], L[1]], 'k--')
        ax.plot([-L[1], L[1]], [-beta_sd[i][1], -beta_sd[i][1]], 'k--')
        ax.plot([-L[1], L[1]], [beta_sd[i][1], beta_sd[i][1]], 'k--')
        ax.plot([0],[0], 'k.')
        ax.clabel(CS, inline=1, fontsize=10)
        title_dl = '$v\'\' = %i \\rightarrow v\'$' %i
        ax.set_title(title_dl)
        ax.set_xlim(-L[0], L[0])
        ax.set_xlabel("$\\beta_1$")
        ax.set_ylim(-L[1], L[1])
        ax.set_ylabel("$\\beta_2$")


fig.suptitle('Iodine 2 molecule - absorbtion spectrum')
fig2.suptitle('Iodine 2 molecule - Birge-Sponer plots')
fig3.suptitle('Iodine 2 molecule - $\chi^2$ plots, uncorr. poly-fit parameters')
plt.show()



# Calculating w_e and w_e x_e for ground level = w0, wx0
# dG[0] := G''(1) - G''(0) = (G'(n) - G''(0)) - (G'(n) - G''(1)) = w_e'' - 2 w_e x_e''
# dG[1] := G''(2) - G''(1) = (G'(n) - G''(1)) - (G'(n) - G''(2)) = w_e'' - 4 w_e x_e''
dG = un.uarray([0]*2, 0)
for j in range(2):      # comparing v'' = (0, 1) and v'' = (1, 2), respectively 
    min_n = max(min(nums[j]), min(nums[j + 1])) # min and max number for coorisponding v'
    max_n = min(max(nums[j]), max(nums[j + 1]))
    where0 = np.where((min_n <= nums[j]) * (nums[j] <= max_n))    # translating to the correct indices
    where1 = np.where((min_n <= nums[j + 1]) * (nums[j + 1] <= max_n))
    dGj = cmm[j][where0]- cmm[j + 1][where1]              # differences
    dG_avg, dG_std = weighted_avg_and_std(dGj)      # weighted mean and standard deviation
    dG[j] = uc.ufloat(dG_avg, dG_std)
wx0 = 0.5 * (dG[0] - dG[1]) 
w0  = 2 * (dG[0] - 0.5 * dG[1])

# Calculating dissociation energies D_e' and D_0' of Pi-states
D_e = w1 ** 2 / (4 * wx1)
# D_0 = \sum_{v = 0}^{v_diss} \Delta G(v + \\frac{1}{2})
# find v_diss = intersect of p(v) with y = 0


# printing results in Latex format
print("\omega_e'' = {:L}".format(w0) )
print("\omega_e x_e'' = {:L}".format(wx0) )
print()
print("results from Birge-Sponer method:")
for i in range(3):
    print("progression: v'' = %i -> x'" %i)
    print("goodness-of-fit: $\chi^2 / n_d = %f$" %gof[i])
    print("$n_d = %i = #points - (deg + 1)$" % n_d[i])
    print("\omega_e''  = {:L}".format(w1[i]) )
    print("\omega_e x_e''  = {:L}".format(wx1[i]) )
    if wy1[i]:
        print("\omega_e y_e''  = {:L}".format(wy1[i]) )
    print("D_e = \\frac{ \omega_e'^2}{ 4  \omega_e x_e'} =" +  "{:L}".format(D_e[i]))
