"""
Calibration of NaI(Te) scintillator with two samples:
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
from scipy.signal import savgol_filter as sav
import seaborn as sns
from scipy.special import erfc
from scipy.integrate import quad
import sys

if len(sys.argv) == 2:
    choose_sample = sys.argv[1]
else:
    choose_sample = input('choose sample: 1: 22Na, 2: 137Cs\n')

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

# Peak: only offset
# Define model function to be used to fit to the data above:
def gauss(x, *p):
    A, mu, sigma, offset = p
    return  A * np.exp(-(x - mu)**2 / (2. * sigma**2)) + offset

# Peak one: bias needs exponential to be added
# Define model function to be used to fit to the data above:
def gauss_plus_exp(x, *p):
    A, mu, sigma, offset, C, lamb = p
    return  A * np.exp(-(x - mu)**2 / (2. * sigma**2)) + offset + \
            C * np.exp(lamb * (x - x_min))

# Compton edge: crude approximation of klein nishina by convolving step function and gaussian
def conv_analytical(x, x_max, sigma, A, offset):
    '''
    analytical solution of convolution of step and gauss
    '''
    return A / 2 * erfc((x - x_max) / (np.sqrt(2) * sigma))  + offset

###########################################################3
if choose_sample == '1':
    #### 22Na sample ####
    file_in = npy_dir + "na_22na_02" + '.npy'
    y = np.load(file_in)
    x = np.arange(len(y))
    
    # Filter data
    y_filtered = sav(y, 301, 4)

    # Gaussian fits
    # Two peaks: 511 keV and 1277 keV (22Na)
    # Compton edge at 341 keV (not visible) and 1064 keV 

    ###### PEAK 1 #################
    # Define range to be fitted
    x_min = 4000    # lower bound
    x_max = 8000    # upper bound
    x_fit = x[(x > x_min) * (x < x_max)]
    y_fit = y[(x > x_min) * (x < x_max)]

    # p0 is the initial guess for the fitting coefficients
    # p = [A, mu, sigma, offset, C, lamb]
    p0 = [160, 6200, 340, 20, 60, -1 / 1000]
    coeff, var_matrix = curve_fit(gauss_plus_exp, x_fit, y_fit, p0=p0)
    hist_fit = gauss_plus_exp(x_fit, *coeff)

    ###### PEAK 2 #################
    # Define range to be fitted
    x_min = 12500    # lower bound
    x_max = 16000    # upper bound
    x_fit2 = x[(x > x_min) * (x < x_max)]
    y_fit2 = y[(x > x_min) * (x < x_max)]

    # p0 is the initial guess for the fitting coefficients
    # p = [A, mu, sigma, offset]
    p0 = [30, 14200, 400, 20]
    coeff2, var_matrix2 = curve_fit(gauss, x_fit2, y_fit2, p0=p0)
    hist_fit2 = gauss(x_fit2, *coeff2)

    ###### COMPTON EDGE 2 #################
    # Define range to be fitted
    x_min = 11290    # lower bound
    x_max = 13150    # upper bound
    x_fit3 = x[(x > x_min) * (x < x_max)]
    y_fit3 = y[(x > x_min) * (x < x_max)]

    # p0 is the initial guess for the fitting coefficients
    # p = [x_max, A, sigma, offset]
    p0 = [12000, 6, 300, 5]
    coeff3, var_matrix3 = curve_fit(conv_analytical, x_fit3, y_fit3, p0=p0)
    hist_fit3 = conv_analytical(x_fit3, *coeff3)
    x_max = coeff3[0]
    
    """
    x_max  = 3500
    A = 130
    sigma = 1000

    x_max  = 12000
    A = 6
    sigma = 300
    offset = 5

    hist_fit3 = conv_analytical(x, x_max, sigma, A) + offset
    """
    # Plotting
    fig1, ax1 = plt.subplots(1, 1)
    sample_name = '$^{22}\mathrm{Na}$'
    if not save_fig:
        fig1.suptitle("Histogram NaI, Sample: " + sample_name)
    ax1.plot(x, y, '.', alpha=0.8, label=('measured counts'))
    ax1.plot(x, y_filtered, '-', alpha=0.8, label=('filtered data'))
    #next(ax1._get_lines.color_cycle)
    #ax1.plot(x_fit, hist_fit, label='511 keV peak')
    #next(ax1._get_lines.color_cycle)
    #ax1.plot(x_fit2, hist_fit2, label='1277 keV peak')
    ax1.plot(x_fit3, hist_fit3, label='compton edge fit')
    ax1.plot([x_max] * 2, [0, 200], '-', label='341 keV compton edge')
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
        file_name = "histo_na_22na"
        fig1.savefig(fig_dir + file_name + ".pdf")
        fig1.savefig(fig_dir + file_name + ".png")

elif choose_sample == '2':
    #### 137Cs sample ####
    file_in = npy_dir + "na_137cs_01" + '.npy'
    y = np.load(file_in)
    x = np.arange(len(y))

    # Gaussian fits
    # One peak: 662 keV
    # Compton edge: 477 keV
    # Backscattered peak: 477 keV

    ###### PEAK 1 #################
    # Define range to be fitted
    x_min = 6600    # lower bound
    x_max = 10000    # upper bound
    x_fit = x[(x > x_min) * (x < x_max)]
    y_fit = y[(x > x_min) * (x < x_max)]

    # p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
    p0 = [7000, 8000, 400, 20]
    coeff, var_matrix = curve_fit(gauss, x_fit, y_fit, p0=p0)
    hist_fit = gauss(x_fit, *coeff)

    # Plotting
    fig1, ax1 = plt.subplots(1, 1)
    sample_name = '$^{137}\mathrm{Cs}$'
    if not save_fig:
        fig1.suptitle("Histogram NaI, Sample: " + sample_name)
    ax1.plot(x, y, '.', alpha=0.8, label=('measured counts'))
    ax1.plot(x_fit, hist_fit, label='511 keV peak')
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
        file_name = "histo_na_137cs"
        fig1.savefig(fig_dir + file_name + ".pdf")
        fig1.savefig(fig_dir + file_name + ".png")
else:
    print('failed to provide the correct input')
