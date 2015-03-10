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
import sys

"""
if len(sys.argv) == 2:
    choose_sample = sys.argv[1]
else:
    choose_sample = '1'
""" 

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
file_name = "histo_ps_both"

#### Background ####
file_in = npy_dir + "ps_background_01" + '.npy'
x_bg, y_bg = np.load(file_in)
t_bg = 234116.126  # Real time
rate_bg = y_bg / t_bg

#### 22Na sample ####
na_sample = '$^{22}\mathrm{Na}$'
file_in = npy_dir + "ps_22na_04" + '.npy'
x_na, y_na = np.load(file_in)
t_na = 72000.000   
rate_na = y_na / t_na
rate_na = rate_na - rate_bg

#### 137Cs sample ####
cs_sample = '$^{137}\mathrm{Cs}$'
file_in = npy_dir + "ps_137cs_03" + '.npy'
x_cs, y_cs = np.load(file_in)
t_cs = 4861.948    
rate_cs = y_cs / t_cs
rate_cs = rate_cs - rate_bg

# Plotting
fig1, ax1 = plt.subplots(1, 1)
if not save_fig:
    fig1.suptitle("Histogram PS, both samples")
ax1.semilogy(x_na, rate_na, '.', alpha=0.8, label=(na_sample))
ax1.semilogy(x_cs, rate_cs, '.', alpha=0.8, label=(cs_sample))
ax1.set_xlabel("Channel")
ax1.set_ylabel("Counts")
ax1.legend(loc=1)
if show_fig:
    fig1.show()
if save_fig:
    fig1.savefig(fig_dir + file_name + ".pdf")
    fig1.savefig(fig_dir + file_name + ".png")
