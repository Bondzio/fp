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

# Physics
E_photon    = 1277 # in keV
m_e = co.physical_constants["electron mass energy equivalent in MeV"][0] * 10**3 # in keV / c**2
lamb_e = co.physical_constants["Compton wavelength"][0] * 10**14    # Compton wavelength of electron in sqrt(barn)
C = co.alpha**2 * lamb_e**2 / (16 * np.pi**3 * m_e)                 # constant prefactor in barn / keV
#C = 1
a = E_photon / m_e
x_max = 2 * a**2 / (1 + 2 * a)                     # maximal electron energy
"""
# Fiction
x_max = 2 * a**2 / (1 + 2 * a)                     # maximal electron energy
kn_max = 200
C = kn_max * a**2 / (2 * (x_max + 1))
"""


# Parameters of convolution
scale = 0.1
sigma  = 0.2 / scale
x0 = 2 * a / scale
n_x = 1000
x = np.linspace(-x0, x0, n_x)

kn = klein_nishina_array(x, a, C, scale)
g = gauss0(x, sigma)
smeared_kn = np.convolve(klein_nishina_array(x, a, C, scale), gauss0(x, sigma), 'same') * 2 * x0 / n_x
conv = conv_kn_gauss(x, a, C, scale, sigma)

#Plotting
fig1, ax1 = plt.subplots(1, 1)
if not save_fig:
    fig1.suptitle("Klein-Nishina formula")
ax1.plot(x, klein_nishina_array(x, a, C, scale), '-', alpha=0.8, label=('Klein-Nishina'))
ax1.plot(x_max, klein_nishina_float(x_max, a, C, scale), 'o', alpha=0.8, label=('max'))
ax1.plot(x, conv, '-', alpha=0.8, label=('Convolution(quad)'))
ax1.plot(x, smeared_kn, '-', alpha=0.8, label=('Convolution(numerical)'))
ax1.set_xlim(0, 2 * a / scale)
#ax1.set_ylim(0, 0.000014)
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

