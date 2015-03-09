"""
Calibration of plastic scintillator with two samples:
    22Na with 2 peaks and 2 visible Compton edges
    137Cs with one peak and one Compton edge
"""

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

# Background
#### 22Na sample ####
file_in = npy_dir + "ps_background_01" + '.npy'
x_background, y_background = np.load(file_in)



# Plotting
fig1, ax1 = plt.subplots(1, 1)
if not save_fig:
    fig1.suptitle("Histogram PS: Background")
ax1.semilogy(x, y, '.', alpha=0.8, label=('measured counts'))
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
#ax1.set_xlim(, )
#ax1.set_ylim(0, 3)
ax1.set_xlabel("Channel")
ax1.set_ylabel("Counts")
ax1.legend(loc=1)
if show_fig:
    fig1.show()
if save_fig:
    file_name = "histo_ps_background"
    fig1.savefig(fig_dir + file_name + ".pdf")
    fig1.savefig(fig_dir + file_name + ".png")



#choose_sample = input('choose sample: 1: 22Na, 2: 137Cs\n')
choose_sample = '2'
if choose_sample == '1':
    #### 22Na sample ####
    sample_name = '$^{22}\mathrm{Na}$'
    file_in = npy_dir + "ps_22na_04" + '.npy'
    x, y = np.load(file_in)

    # Define range to be fitted
    x_min = 4000    # lower bound
    x_max = 8000    # upper bound
    x_fit = x[(x > x_min) * (x < x_max)]
    y_fit = y[(x > x_min) * (x < x_max)]

    # p0 is the initial guess for the fitting coefficients (A, mu and sigma above)

    # Plotting
    fig1, ax1 = plt.subplots(1, 1)
    if not save_fig:
        fig1.suptitle("Histogram PS, Sample: " + sample_name)
    ax1.semilogy(x, y, '.', alpha=0.8, label=('measured counts'))
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
        file_name = "histo_ps_22na"
        fig1.savefig(fig_dir + file_name + ".pdf")
        fig1.savefig(fig_dir + file_name + ".png")
else:
    print('failed to provide the correct input')
