import numpy as np
import pylab as plt
import uncertainties as uc
from uncertainties import unumpy as un
import sympy as sy
from pylatex import Document, Section, Subsection, Math
from pylatex.numpy import Matrix
from math import log10, floor
import scipy.constants as co

def unv(uarray):        # returning nominal values of a uarray
    return un.nominal_values(uarray)

def usd(uarray):        # returning the standard deviations of a uarray
    return un.std_devs(uarray)

def dig_err(cov, i): # returns the significant digit of the error
    dx = np.sqrt(cov[i,i])
    digit = -int(floor(log10(dx)))    
    if (dx * 10**digit) < 3.5:
        digit += 1
    return digit
    
def dig_val(x):     # returns the last significant digit of a value (error convention...)
    digit = -int(floor(log10(abs(x))))    
    if (x * 10**digit) < 3.5:
        digit += 1
    return digit

def em(str):        # rewrites 'e' for exponential but leaves other 'e's untouched
    str = str.split("e")
    for i, substr in enumerate(str):
        if i == 0:
            new_str = substr
        else:
            if substr[1].isdigit():
                new_str = r"\mathrm{e}".join([new_str, substr])
            else:
                new_str = "e".join([new_str, substr])
    return new_str

def la_coeff(f1, coeff, cov, var_names, additional_digits=0):
    """
    prints coeffients and their covariance matrix to a .tex file
    """
    f1.write(r"\begin{eqnarray}" + "\n")
    for j, co in enumerate(coeff):
        str_co = "    " + var_names[j]
        digit = dig_err(cov, j) + additional_digits
        var = round(co, digit)
        """
        if digit < 1:
            str_co  += " &=& %i "%int(var) 
        if digit < 4:
            pre_str = " &=&%." + str(digit) + r"f "
            str_co  += pre_str%(var) 
        else:
            str_co += " &=& %.3e "%var
        """
        str_co += " &=& %.3e "%co
        str_co += r"\cm \nonumber \\"
        str_co = em(str_co)
        f1.write(str_co +"\n")

    f1.write(r"    \mathrm{cov}(p_i, p_j) &=& " "\n")
    f1.write(r"    \begin{pmatrix}" + "\n")
    for row in cov:
        str_row = "        "
        for entry in row:
            digit = dig_val(entry) + additional_digits
            var = round(entry, digit)
            """
            if digit < 1:
                str_row += " %i &"%int(var)
            elif digit < 4:
                pre_str = "%." + str(digit) + "f &"
                str_row += pre_str%var
            else:
                str_row += "%.1e &"%var
            """
            str_row += "%.3e &"%entry
        str_row = str_row[:-1] + r"\\"
        str_row = em(str_row)
        f1.write(str_row + "\n")
    f1.write(r"    \end{pmatrix}" + "\n")
    f1.write(r"\\ \Rightarrow \qquad" + "\n")
    for j, co in enumerate(coeff):
        str_co = "    " + var_names[j]
        var = uc.ufloat(coeff[j], np.sqrt(cov[j,j]))
        str_co += " &=& {:L} \\cm\\\\".format(var)
        str_co = em(str_co)
        if j == len(coeff) -1:
            str_co = str_co[:-2]
        f1.write(str_co +"\n")
    f1.write(r"\end{eqnarray}" + "\n\n")

    return 0

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

def str_label_fit(i):
    """
    creates a latex string of the polynomial created by the fit
    """
    for j in range(deg[i] + 1):     # write strings for poly fit label
        c = coeff[i][deg[i] - j]
        if j == 0: 
            if c != 0:
                str_p =  "%i$" %abs(c)
                if c < 0: str_p = "- " + str_p
                else: str_p = "+ " + str_p
        elif j == 1: 
            if c != 0:
                str_p =  "%.2f v'" %abs(c) + str_p
                if c < 0: str_p = "- " + str_p
                else: str_p = "+ " + str_p
        elif j == deg[i]:
            if c != 0:
                str_p =  "%.5f {v'}^%i" %(abs(c), j) + str_p
                if c < 0: str_p = "- " + str_p
        else:
            if c != 0:
                str_p =  "%.2f v'^%i" %(abs(c), j) + str_p
                if c < 0: str_p = "- " + str_p
                else: str_p = "+ " + str_p
    return( "$P_%i(v') = "%deg[i] + str_p)
#############################################################################################

# general parameters
plot_show = 0
save_fig = 0
print_latex = 1
deg = [2, 1, 1]                                # degree of polynomial fit
plt.close('all')
colors = ['b', 'g', 'r', 'pink']
labels = ['$v\'\' = %i \\rightarrow v\'$' %i for i in range(3)]
fig_dir = 'figures/'
l1 = 7.0
figsize = [co.golden * l1, l1]

# load arrays
array_dir = "arrays/"
str_array_all = ["cmm", "dcm", "nums"]
cmm = np.load(array_dir + "cmm.npy")
dcm = np.load(array_dir + "dcm.npy")
nums = np.load(array_dir + "nums.npy")

coeff, covA = {}, {}    # coefficients and covariance matrix from polinomial fit
chi2_min, n_d, gof = {}, {}, {}    # chi square for poly fit, number of ind. data points, goodness-of-fit
beta, beta_sd = {}, {}      # uncorrelated parameters and their std devs
eigvec = {}
vib_co = {}
D_e1 = {}
v_diss, v_diss_upper, v_diss_lower = {}, {}, {}
D_0_n, D_0_upper, D_0_lower, D_0 = {}, {}, {}, {}
sigma00, E_diss = {}, {}
n = np.int_([22, 3, 3])
n_real = np.int_([47, 27, 20]) - n

# Calculating w_e' = w1 and w_e x_e' = wx1 for first electronic level with Birge-Sponer plots
# dG(v' + 1/2) = w_e' - w_e x_e' * (2v + 2) = a + b v (latter one is solution of polinomial fit)
# =>    w_e' = a + b
#       w_e x_e' = (a - w_e') / 2
# similar for deg = 2
v_diss  = un.uarray([0]*3, 0)
D_0     = un.uarray([0]*3, 0)
label_c = [[],[],[]]
f1 = open("coefficients.tex", "w+")
f2 = open("vib_coefficients.tex", "w+")
for i in range(3):   # Polynomial fit for dG(v' + 1/2)
    weights = 1 / (usd(dcm[i]) ** 2)
    coeff[i], covA[i] = np.polyfit(nums[i], unv(dcm[i]), deg[i], full=False, w=weights, cov=True)
    c = uc.correlated_values(coeff[i], covA[i])
    if deg[i] == 1:
        wx1 = -0.5 * c[0]
        w1 = c[1] - c[0]
        vib_co[i] = np.array([wx1, w1])
    if deg[i] == 2:
        wy1 = c[0] / 3
        wx1 = c[0] - 0.5 * c[1]
        w1 = 11 / 12 * c[0] - c[1] + c[2]
        vib_co[i] = np.array([wy1, wx1, w1])
    label_c[i] = '$\omega_e\' \\ = ' + '{:L}'.format(w1) + '$\n' \
        '$\omega_e x_e\' = ' + '{:L}'.format(wx1) + '$'
    if deg[i] >= 2:
        label_c[i] += '\n$\omega_e y_e\' = ' + '{:L}'.format(wy1) + '$'
# printing coefficents p_i and vibrational parameters with their covariance matrices to files
    var_names = ["p_2", "p_1", "p_0"]
    la_coeff(f1, coeff[i], covA[i], var_names, additional_digits=2)
    val = vib_co[i]
    if deg[i] == 2: var_names = [r"\omega_e y_e'", r"\omega_e x_e'", r"\omega_e'"]
    if deg[i] == 1: var_names = [r"\omega_e x_e'", r"\omega_e'"]
    cov = np.array(uc.covariance_matrix(val))
    la_coeff(f2, unv(val), cov, var_names)
# Calculating dissociation energies D_e' and D_0' of Pi-states
    D_e1[i] = vib_co[i][-1] ** 2 / (4 * vib_co[i][-2])
# Calculating D_0 = \sum_{v = 0}^{v_diss} \Delta G(v + \\frac{1}{2})
# v_diss = intersect of p(v) with y = 0
    v_grid_n = np.arange(0, 90) + 0.5           # v grid for finding zeros
    vals = unv(np.polyval(c, v_grid_n))
    errs = usd(np.polyval(c, v_grid_n))
    both = np.polyval(c, v_grid_n)
    g0 = np.where(vals > 0)     # dissipation energy D_0, nominal value
    g0_upper = np.where(vals + errs > 0)
    g0_lower = np.where(vals - errs > 0)
    if len(g0) == len(vals):
        print("no zero intersect! -> D_0 cannot be calculated")
    v_diss[i] = v_grid_n[g0[0][-1]]
    v_diss_upper[i] = v_grid_n[g0_upper[0][ -1]]
    v_diss_lower[i] = v_grid_n[g0_lower[0][ -1]]
    D_0_n[i] = np.sum(vals[g0])     # dissipation energy D_0, nominal value
    D_0_upper[i] = np.sum((vals + errs)[g0_upper])     # dissipation energy D_0, nominal value
    D_0_lower[i] = np.sum((vals - errs)[g0_lower])     # dissipation energy D_0, nominal value
    D_0[i] = uc.ufloat(D_0_n[i], max(abs(D_0_n[i] - D_0_upper[i]), abs(D_0_n[i] - D_0_lower[i])))
# Excitation energy
    sigma00[i] = cmm[i][n[i]] - (both[n[i]-1] - both[0])
# Dissociation energy in the experiment
    E_diss[i] = sigma00[i] + D_0[i]
# Birge Sponer plots
    fig1 = plt.figure(figsize = figsize)
    #fig1.suptitle('Iodine 2 molecule - Birge-Sponer plots')
    ax = plt.subplot(111)
    title_dl = '$v\'\' = %i \\rightarrow v\'$' %i
    #ax.set_title(title_dl)
    ax.errorbar(nums[i], unv(dcm[i]), yerr=usd(dcm[i]), color=colors[i], ls='dots', marker='.' )
    v_min = 0
    v_max = [70, 70, 90][i]
    v_grid = np.linspace(v_min, v_max + 1, 200)
    label_fit = str_label_fit(i)
    ax.plot(v_grid, unv(np.polyval(c, v_grid)), '-', color=colors[i], label=label_fit)
    ax.set_xlabel("$v\' + \\frac{1}{2}$")
    xticks = np.arange(v_min, v_max + 1, 10) + 0.5  # assign values, at which xticks appear
    ax.set_xticks(xticks)               # set xticks
    #xticklabels = ["", "", ...]        # specify, what is written at each xtick
    #ax.set_xticklabels(xticklabels, fontsize=20)
    ax.set_ylim(0, [150, 150, 150][i])
    ax.set_ylabel("$\Delta \sigma \, / \, \mathrm{cm^{-1}}$")
    leg = ax.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_visible(False)
    # plot errors of poly-fit as shaded areas
    ax.set_xlim(v_min, v_max + 0.5)
    ax.fill_between(v_grid, \
            unv(np.polyval(c, v_grid)) + usd(np.polyval(c, v_grid)),
            unv(np.polyval(c, v_grid)) - usd(np.polyval(c, v_grid)),
            facecolor=colors[i], color=colors[i], alpha=0.3 )
    if plot_show:
        fig1.show()
    if save_fig: 
        fig_name = "b_s_%i.pdf"%i
        fig1.savefig(fig_dir + fig_name)
f1.close()
f2.close()

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
freq0 = w0 * co.c * 10*2
mu = 127/2 * co.physical_constants["atomic mass constant"][0]
k0 = (freq0 * 2 * co.pi) ** 2 * mu

# Calculating dissociation energies D_e'' and D_0'' of Pi-states
D_e0 = w0 ** 2 / (4 * wx0)



# chi2 plots
fig2 = plt.figure(figsize = [21.0, 7.0])
fig2.suptitle('Iodine 2 molecule - $\chi^2$ plots, uncorr. poly-fit parameters')
for i in range(1):
# diagonalize covariance matrix
    eigval, eigvec[i] = np.linalg.eig(covA[i])
    beta[i] = np.linalg.solve(eigvec[i], coeff[i]) # uncorrelated parameters
    A = np.matrix(eigvec[i]).I * np.matrix(covA[i]) * np.matrix(eigvec[i])
    beta_sd[i] = np.sqrt(np.diag(A))            # standard deviation for beta
# minimized chi square
    chi2_min[i] = np.sum(((np.polyval(coeff[i], nums[i]) - unv(dcm[i])) / usd(dcm[i])) ** 2 )
    n_d[i] = len(dcm[i]) - (deg[i] + 1)     # number of degress of freedom
    gof[i] = chi2_min[i] / n_d[i]   # godness-of-fit
    if deg[i] == 1:
        ax = plt.subplot(1, 3, i + 1)
        n_grid = 11
        L = 2 * beta_sd[i]
        x = np.linspace(-L[0], L[0], n_grid)
        y = np.linspace(-L[1], L[1], n_grid)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros([n_grid, n_grid])
        Z2 = np.zeros([n_grid, n_grid])
        for j in range(n_grid):
            for k in range(n_grid):
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


# printing results in Latex format
if print_latex:
    f3 = open("results.tex", "w+")
    f3.write("\omega_e'' &=& {:L} \cm\n".format(w0) )
    f3.write("\omega_e x_e'' &=& {:L} \cm\n".format(wx0) )
    f3.write("f_e'' &=& {:L}".format(freq0)  + " \mathrm{Hz}\\\\\n")
    f3.write("k_e'' &=& {:L}".format(k0) + " \mathrm{\\frac{kg}{s^2]}\n")
    f3.write("\nresults from Birge-Sponer method: \n\n")
    for i in range(3):
        f3.write("progression: v'' = %i -> x' \n" %i)
        f3.write("D_e' = \\frac{ \omega_e'^2}{ 4  \omega_e x_e'} =" +  "{:L} \cm\n".format(D_e1[i]))
        f3.write("v_\mathrm{diss}' &=& %.1f \\\\\n"%v_diss[i])
        f3.write("D_0' &=& %i \cm \\\\\n"%round(D_0_n[i]))
        f3.write("v_\mathrm{diss,\, upper}' &=& %.1f \\\\\n"%v_diss_upper[i])
        f3.write("D_{0,\, \mathrm{upper}}' &=& %i \cm\\\\\n"%round(D_0_upper[i]))
        f3.write("v_\mathrm{diss,\, lower}' &=& %.1f \\\\\n"%v_diss_lower[i])
        f3.write("D_{0,\, \mathrm{lower}}' &=& %i \cm\n"%round(D_0_lower[i]))
        f3.write("D_0' = {:L} \cm\n".format(D_0[i]))
        f3.write("G'(v' = %i) &=& "%n_real[i] + "{:L} \cm\n".format(cmm[i][n[i]]))
        f3.write(r"\sigma_{00} &=& " + "{:L} \cm\n".format(sigma00[i]))
        f3.write(r"E_\mathrm{diss} &=& " + "{:L} \cm\n".format(E_diss[i]))
        #f3.write("goodness-of-fit: $\chi^2 / n_d = %f$" %gof[i])
        #f3.write("$n_d = %i = #points - (deg + 1)$" % n_d[i])
        f3.write("\n")
    f3.close()
    #fig2.show()
