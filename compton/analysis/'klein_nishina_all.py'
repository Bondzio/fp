"""
Draws the Klein-Nishina formula
"""

import numpy as np
import pylab as plt
from matplotlib import rcParams
from scipy.optimize import curve_fit
import scipy.constants as co
#import uncertainties as uc
#import uncertainties.unumpy as un
from scipy.signal import argrelextrema as ext
from scipy.special import erfc
from scipy.integrate import quad
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

def klein_nishina(x, a, C):
    """
    Insert photon and electron energy in m_e, constant C (in barn)
    """
    if type(x) == float: # remove diverging points (dsdE = 0 for x > a)
        if x == a:
            x = a + 0.1
    else: 
        x[x == a] = a + 0.1
    x_max = 2 * a**2 / (1 + 2 * a)                     # maximal electron energy
    dsdE =  C / a**2 * \
            (x**2 / (a * (a - x))**2 + ((x - 1)**2 - 1) / (a * (a - x)) + 2) *\
            (x <= x_max)                                     # in barn / keV
    return(dsdE)

def gauss(x, *p):
    mu, sigma = p
    return  1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-(x - mu)**2 / (2. * sigma**2))

def conv_tooth(x, x_m, sigma, a, b):
    '''
    convolution of gaussian of width sigma 
    with tooth-like function 
    f = (a + b * x) * heaviside(x_m, x)
    '''
    conv_f_g = (a + b*x) / 2 * erfc((x - x_m) / (np.sqrt(2) * sigma)) + \
            b / (2 * np.sqrt(np.pi)) * np.exp(-(x - x_m)**2 / (2 * sigma**2))
    return conv_f_g

"""
# Convolution
def conv_kn_gauss(x, a, C, mu, sigma):
    kn_gauss = lambda y, x, a, C, mu, sigma: klein_nishina(y, a, C) * gauss(x - y, mu, sigma)
    if type(x) == float:
        conv = quad(kn_gauss, -2*a, 2*a, args=(x, a, C, sigma))[0]
    else:
        conv = np.array([quad(kn_gauss, -np.inf, np.inf, args=(x_i, a, C, mu, sigma))[0] for x_i in x])
    return conv
"""

# Convolution
def gauss0(x, sigma):
    return  1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-x**2 / (2. * sigma**2))

def conv_kn_gauss(x, a, C, sigma):
    kn_gauss = lambda y, x, a, C, sigma: klein_nishina(y, a, C) * gauss0(x - y, sigma)
    if type(x) == float:
        conv = quad(kn_gauss, -2*a, 2*a, args=(x, a, C, sigma))[0]
    else:
        conv = np.array([quad(kn_gauss, -np.inf, np.inf, args=(x_i, a, C, sigma))[0] for x_i in x])
    return conv


# Physics
E_photon    = 1277 # in keV
m_e = co.physical_constants["electron mass energy equivalent in MeV"][0] * 10**3 # in keV / c**2
lamb_e = co.physical_constants["Compton wavelength"][0] * 10**14    # Compton wavelength of electron in sqrt(barn)
C = co.alpha**2 * lamb_e**2 / (16 * np.pi**3 * m_e)                 # constant prefactor in barn / keV
#C = 1
a = E_photon / m_e
x_max = 2 * a**2 / (1 + 2 * a)                     # maximal electron energy

# Parameters of convolution
sigma  = 0.05
mu = 0
x0 = 2 * a
n_x = 1000
x = np.linspace(-x0, x0, n_x)

"""
# Parameters of linear approximation of K-N
slope   = C / a**2 * (20 * a + 12 - 4 / a - 2 / a**2)  # slope at kn(x_max) by taylor
offset  = x_max * (2 * C / a**2 - slope) + 2 * C / a**2     
tooth   = lambda x: offset + slope * x
"""

kn = klein_nishina(x, a, C)
g = gauss(x, mu, sigma)
smeared_kn = np.convolve(klein_nishina(x, a, C), gauss(x, mu, sigma), 'same') * 2 * x0 / n_x
conv = conv_kn_gauss(x, a, C, sigma)
#smeared_tooth = conv_tooth(x, x_max, sigma, )

#Plotting
fig1, ax1 = plt.subplots(1, 1)
if not save_fig:
    fig1.suptitle("Klein-Nishina formula")
ax1.plot(x, kn, '-', alpha=0.8, label=('Klein-Nishina'))
ax1.plot(x_max, klein_nishina(np.array([x_max]), a, C), 'o', alpha=0.8, label=('max'))
ax1.plot(x, conv, '-', alpha=0.8, label=('Convolution(quad)'))
#ax1.plot(x, tooth(x), '-', alpha=0.8, label=('Klein-Nishina'))
#ax1.plot(x, C * g, '-', alpha=0.8, label=('Gauss'))
ax1.plot(x, smeared_kn, '-', alpha=0.8, label=('Convolution(numerical)'))
#ax1.plot(x, tooth, '-', alpha=0.8, label=('Convolution'))
ax1.set_xlim(0, a)
ax1.set_ylim(0, 0.000014)
#ax1.set_ylim(0, 1)
ax1.set_xlabel("$E_e$ / keV")
ax1.set_ylabel("$\\frac{d\sigma}{d E_e} / \mathrm{\\frac{barn}{keV}}$")
ax1.legend(loc=1)
if show_fig:
    fig1.show()
if save_fig:
    file_name = "klein_nishina"
    fig1.savefig(fig_dir + file_name + ".pdf")
    fig1.savefig(fig_dir + file_name + ".png")


