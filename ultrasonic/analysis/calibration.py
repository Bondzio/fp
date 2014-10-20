import numpy as np
import pylab as plt
from matplotlib import rcParams
from scipy.optimize import curve_fit
import scipy.constants as co
import uncertainties as uc
import uncertainties.unumpy as un
from scipy.signal import argrelextrema as ext
import seaborn as sns

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

def la_coeff(f1, coeff, cov, var_names, digits=3):
    """
    prints coeffients and their covariance matrix to a .tex file
    """
    f1.write(r"\begin{align}" + "\n")

    for j, co in enumerate(coeff):
        str_co = "\t" + var_names[j]
        #var = uc.ufloat(coeff[j]
        #str_co += " &=& {:L} \\\\".format(var)     # uncertainties format
        str_co += (" &= %." + str(digits) + "f \\\\")%coeff[j] # specified format withouh errors
        str_co = em(str_co)
        f1.write(str_co +"\n")

    f1.write("\t\mathrm{cov} &=\n")
    f1.write("\t\\begin{pmatrix}\n")
    for row in cov:
        str_row = "\t\t"
        for var in row:
            str_row += ("%." + str(1) + "e &")%var
        str_row = str_row[:-1] + r"\\"
        str_row = em(str_row)
        f1.write(str_row + "\n")
    f1.write("\t\end{pmatrix} \n")
    f1.write("\end{align}\n\n")
    return 0

fontsize_labels = 12    # size used in latex document
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern Roman']
rcParams['text.usetex'] = True
rcParams['figure.autolayout'] = True
rcParams['axes.labelsize'] = fontsize_labels
rcParams['xtick.labelsize'] = fontsize_labels
rcParams['ytick.labelsize'] = fontsize_labels

plt.close("all")
show_fig = True
save_fig = True
fig_dir = "../figures/"
npy_dir = "./data_npy/"
plotsize = (6.2, 3.83)  # width corresponds to \textwidth in latex document (ratio = golden ratio ;))

# Gauge
# parameters of the gauge grating
fig1, ax1 = plt.subplots(1, 1, figsize=plotsize)
lamb = 632.8e-9
K = 10 ** -4 # 10 lines / mm => lattice constant K = 10^-4 m
maxis_both = []
for q in "ab":  # use both measurements to get deviation
    npy_files = npy_dir + "gauge_" + q
    t = np.load(npy_files + "_t" + ".npy")
    t = t * 10 ** 3 # time in ms!!!
    t = t - t[-1] / 2 # t = 0 doesn't lie on the physical t=0. Translate the center to t=0!
    signal = np.load(npy_files + "_ch_a" + ".npy")

    # get local maxima
    maxis = ext(signal, np.greater_equal, order=70)[0] # all maxima with next <order> points greater_equal
    maxis = maxis[signal[maxis] >  0.01]   # only those greater <float>
    maxis_j = []
    # reduce numbers of maxima to one per plateau
    i = 0
    while i<len(maxis):
        plateau = np.where((maxis >= maxis[i])*(maxis < maxis[i] + 100))[0]
        maxis_j.append(plateau[int(len(plateau) / 2)])
        i = plateau[-1] + 1
    maxis = maxis[maxis_j]
    maxis_both.append(maxis)

    peak_plot, = ax1.plot(t, signal, alpha=0.8)
    [ax1.plot([t[maxi]] * 2, [0, signal[maxi]], '--', color=peak_plot.get_color(), linewidth=1) for maxi in maxis]
    next(ax1._get_lines.color_cycle)
ax1.set_xlim(t[0], t[-1])
ax1.set_xlabel("$t$ / ms")
ax1.set_ylabel("$U$ / V")


# Linear fitting of angles corresponding to maxima
# zorder: default: 1 for patch, 2 for lines, 3 for legend
fig2, ax2 = plt.subplots(1, 1, figsize=plotsize)
# Prepare variables: take average of both maxima
t_max_both = t[np.array(maxis_both).T]
t_max = np.zeros(len(t_max_both))
std_dev = t_max * 0
for i, t_max_pair in enumerate(t_max_both):
    t_max[i] = np.average(t_max_pair)
    std_dev[i] = np.std(t_max_pair)
# caculate the angles corresponding to the maxima, using sin(theta) = m lamb / K
m = np.arange(-3, 3) # order of maxima: in this case: -3 -- 2
theta = np.arcsin(m * lamb / K)
ax2.errorbar(t_max, theta, xerr=std_dev, fmt='k,', zorder=2.9) 
# linear fit
p, cov = np.polyfit(t_max, theta, 1, cov=True)
p_uc = uc.correlated_values(p, cov)
# plotting the linear fit, including shading of errors
data_fit = np.polyval(p, t)
fit_plot, = ax2.plot(t, data_fit, alpha=1, zorder=2.1)
error_on_fit = un.std_devs(np.polyval(p_uc, t))
data_fit_min = data_fit - error_on_fit
data_fit_max = data_fit + error_on_fit
ax2.fill_between(t, data_fit_min , data_fit_max, facecolor=fit_plot.get_color(), alpha=0.2, zorder=2 )

# place a text box in upper right in axes coords, using eqnarray
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
textstr = '$\omega t + \phi_0$ with\n$\omega=%.2f$ rad/ms\n$\phi_0=%.2f$ rad'%(p[0], p[1])
textstr = '\\begin{eqnarray*}\\theta(t) &=& \omega t + \phi_0 \\\\ \
        \omega      &=&%.4f \, \mathrm{ rad/s} \\\\  \
        \phi_0    &=&%.4f \, \mathrm{rad}\end{eqnarray*}'%(p[0], p[1])
ax2.text(0.1, 0.85, textstr, transform=ax2.transAxes, fontsize=fontsize_labels, va='top', bbox=props)

ax2.set_xlim(t[0], t[-1])
ax2.set_ylim(-0.03, 0.03)
ax2.set_xlabel("$t$ / ms")
ax2.set_ylabel("$\\theta$ / rad")


# Saving relevant data and plotting

np.save(npy_dir + "gauge_fit_coeff", p)
np.save(npy_dir + "gauge_fit_cov", cov)

# print covariance matrix to file
f1 = open("coefficients.tex", "w+")
var_names = [r"\omega", r"\phi_0"]
la_coeff(f1, p, cov, var_names, digits=4)
f1.close()

if show_fig:
    fig1.show()
    fig2.show()

if save_fig:
    fig1.savefig(fig_dir + "calibrate_peaks.pdf")
    fig2.savefig(fig_dir + "calibrate_fit.pdf")

