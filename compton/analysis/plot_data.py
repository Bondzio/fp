import numpy as np
import pylab as plt
from matplotlib import rcParams
from scipy.optimize import curve_fit
import scipy.constants as co
#import uncertainties as uc
#import uncertainties.unumpy as un
from scipy.signal import argrelextrema as ext
import seaborn as sns

fontsize_labels = 22    # size used in latex document
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern Roman']
rcParams['text.usetex'] = True
rcParams['figure.autolayout'] = True
rcParams['font.size'] = fontsize_labels
rcParams['axes.labelsize'] = fontsize_labels
rcParams['xtick.labelsize'] = fontsize_labels
rcParams['ytick.labelsize'] = fontsize_labels
rcParams['legend.fontsize'] = fontsize_labels
rcParams['figure.figsize'] = (2*6.2, 2*3.83)  # in inches; width corresponds to \textwidth in latex document (golden ratio)

plt.close("all")
show_fig = True
save_fig = False # see below
if not save_fig:
    rcParams['figure.figsize'] = (15, 8)  # in inches; width corresponds to \textwidth in latex document (golden ratio)
fig_dir = "../figures/"
npy_dir = "./data_npy/"

"""
# Plastic szintillator
file_in = npy_dir + "ps_22na_01" + '.npy'
x, y = np.load(file_in)

fig1, ax1 = plt.subplots(1, 1)
sample_name = '22Na'
if not save_fig:
    fig1.suptitle("Histogram PS, Sample: " + sample_name)
ax1.semilogy(x, y, alpha=0.8, label=(r'$T_\mathrm{' + sample_name + r'}$'))
#ax1.bar(x, y, alpha=0.8, log=True, label=(r'$T_\mathrm{' + sample_name + r'}$'))
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
textstr = 'Sample: ' + sample_name
ax1 = plt.gca()
ax1.text(0.1, 0.95, textstr, transform=ax1.transAxes, va='top', bbox=props)
#ax1.set_xlim(, )
#ax1.set_ylim(0, 3)
ax1.set_xlabel("Channel")
ax1.set_ylabel("Counts")
#ax1.legend(loc=1)
if show_fig:
    fig1.show()
if save_fig:
    file_name = "histo_ps_" + sample_name
    fig1.savefig(fig_dir + file_name + ".pdf")
    fig1.savefig(fig_dir + file_name + ".png")
"""

# NaI szintillator
file_in = npy_dir + "na_22na_02" + '.npy'
y = np.load(file_in)
x = np.arange(len(y))

# Gaussian fits
# Two peaks: 511 keV and 1277 keV (22Na)

# Peak one: bias needs exponential to be added
# Define model function to be used to fit to the data above:
def gauss_plus_exp(x, *p):
    A, mu, sigma, offset, C, lamb = p
    return  A * np.exp(-(x - mu)**2 / (2. * sigma**2)) + offset + \
            C * np.exp(lamb * (x - x_min))

# Define range to be fitted
x_min = 4000    # lower bound
x_max = 8000    # upper bound
x_fit = x[(x > x_min) * (x < x_max)]
y_fit = y[(x > x_min) * (x < x_max)]

# p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
p0 = [160, 6200, 1000, 20, 60, -1 / 1000]
coeff, var_matrix = curve_fit(gauss_plus_exp, x_fit, y_fit, p0=p0)
hist_fit = gauss_plus_exp(x_fit, *coeff)


# Peak two: only offset
# Define model function to be used to fit to the data above:
def gauss(x, *p):
    A, mu, sigma, offset = p
    return  A * np.exp(-(x - mu)**2 / (2. * sigma**2)) + offset

# Define range to be fitted
x_min = 12500    # lower bound
x_max = 16000    # upper bound
x_fit2 = x[(x > x_min) * (x < x_max)]
y_fit2 = y[(x > x_min) * (x < x_max)]

# p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
p0 = [30, 14200, 400, 20]
coeff2, var_matrix2 = curve_fit(gauss, x_fit2, y_fit2, p0=p0)
hist_fit2 = gauss(x_fit2, *coeff2)

# Plotting
fig1, ax1 = plt.subplots(1, 1)
sample_name = '$^{22}\mathrm{Na}$'
if not save_fig:
    fig1.suptitle("Histogram PS, Sample: " + sample_name)
ax1.plot(x, y, '.', alpha=0.8, label=('measured counts'))
next(ax1._get_lines.color_cycle)
ax1.plot(x_fit, hist_fit, label='511 keV peak')
next(ax1._get_lines.color_cycle)
ax1.plot(x_fit2, hist_fit2, label='1277 keV peak')
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
textstr = 'Sample: ' + sample_name
ax1.text(0.1, 0.95, textstr, transform=ax1.transAxes, va='top', bbox=props)
#ax1.set_xlim(, )
#ax1.set_ylim(0, 3)
ax1.set_xlabel("Channel")
ax1.set_ylabel("Counts")
ax1.legend(loc=1)
if show_fig:
    fig1.show()
if save_fig:
    file_name = "histo_ps_" + sample_name
    fig1.savefig(fig_dir + file_name + ".pdf")
    fig1.savefig(fig_dir + file_name + ".png")
