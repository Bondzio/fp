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

def klein_nishina_array(x, a, C, scale):
    """
    Insert photon and electron energy in m_e, constant C (in barn)
    Takes only np.arrays!
    """
    x_max = 2 * a**2 / (1 + 2 * a)                     # maximal electron energy
    y = x / scale
    dsdE =  C / a**2 * \
            (y**2 / (a * (a - y))**2 + ((y - 1)**2 - 1) / (a * (a - y)) + 2) *\
            (y <= x_max)                                     # in barn / keV
    dsdE[np.isnan(dsdE)] = 0
    return(dsdE)

def klein_nishina_float(x, a, C, scale):
    '''
    Insert photon and electron energy in m_e, constant C (in barn)
    Takes only floats!
    '''
    x_max = 2 * a**2 / (1 + 2 * a)                     # maximal electron energy
    y = x / scale
    if y > x_max:
        dsdE = 0
    else:
        dsdE =  C / a**2 * \
                (y**2 / (a * (a - y))**2 + ((y - 1)**2 - 1) / (a * (a - y)) + 2) 
    return(dsdE)

def gauss0(x, sigma):
    return  1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-x**2 / (2. * sigma**2))

def conv_kn_gauss(x, a, C, scale, sigma):
    kn_gauss = lambda y, x, a, C, sigma: klein_nishina_float(x - y, a, C, scale) * gauss0(y, sigma)
    if type(x) == float:
        conv = quad(kn_gauss, -np.inf, np.inf, args=(x, a, C, sigma))[0]
    else:
        conv = np.array([quad(kn_gauss, -np.inf, np.inf, args=(x_i, a, C, sigma))[0] for x_i in x])
    return conv

# Compton edge: crude approximation of klein nishina by convolving step function and gaussian
def conv_analytical(x, x_max, sigma, A, offset):
    '''
    analytical solution of convolution of step and gauss
    '''
    return A / 2 * erfc((x - x_max) / (np.sqrt(2) * sigma))  + offset

###########################################################3
#### 137Cs sample ####
file_in = npy_dir + "na_137cs_01" + '.npy'
y = np.load(file_in)
x = np.arange(len(y))
    
# Filter data
y_filtered = sav(y, 301, 4)

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

# p0 is the initial guess for the fitting coefficients
# p = [A, mu, sigma, offset]
p0 = [7000, 8000, 400, 20]
coeff, var_matrix = curve_fit(gauss, x_fit, y_fit, p0=p0)
hist_fit = gauss(x_fit, *coeff)

###### COMPTON EDGE  #################
# Define range to be fitted
x_min = 5000    # lower bound
x_max = 10000    # upper bound
x_fit2 = x[(x > x_min) * (x < x_max)]
y_fit2 = y[(x > x_min) * (x < x_max)]

# p0 is the initial guess for the fitting coefficients
# p = [x_max, A, sigma, offset]
p0 = [12000, 6, 300, 5]
#coeff2, var_matrix2 = curve_fit(conv_analytical, x_fit2, y_fit2, p0=p0)
#hist_fit2 = conv_analytical(x_fit2, *coeff2)
#x_max = coeff2[0]

# Parameters of convolution
x_max  = 5700
A = 1600
sigma = 500
offset = 0

hist_fit2 = conv_analytical(x_fit2, x_max, sigma, A, offset)

"""
# Manually adapted parameters
a = 1.8 
C = 2.1 * 10**3
scale = 4.1 * 10**3
sigma  = 0.12 * scale
x_c_max = 10000
n_x = 200
x_c = np.linspace(0, x_c_max, n_x)
kn = klein_nishina_array(x_c, a, C, scale)
#smeared_kn = np.convolve(klein_nishina_array(x, a, C, scale), gauss0(x, sigma), 'same') * 2 * x0 / n_x
conv = conv_kn_gauss(x_c, a, C, scale, sigma)
"""


# Plotting
fig1, ax1 = plt.subplots(1, 1)
sample_name = '$^{137}\mathrm{Cs}$'
if not save_fig:
    fig1.suptitle("Histogram NaI, Sample: " + sample_name)
ax1.plot(x, y, '.', alpha=0.8, label=('measured counts'))
ax1.plot(x_fit, hist_fit, label='511 keV peak')
next(ax1._get_lines.color_cycle)
ax1.plot(x, y_filtered, '-', alpha=0.8, label=('filtered data'))
next(ax1._get_lines.color_cycle)
#ax1.plot(x_c, conv, label='compton edge fit')
#ax1.plot(x_c, kn, label='compton edge fit')
#ax1.plot(x_fit2, conv, label='compton edge fit')
ax1.plot(x_fit2, hist_fit2, label='compton edge fit')
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
    file_name = "histo_na_137cs"
    fig1.savefig(fig_dir + file_name + ".pdf")
    fig1.savefig(fig_dir + file_name + ".png")
