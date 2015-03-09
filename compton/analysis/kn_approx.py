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
            x = a + 1
    else: 
        x[x == a] = a + 1
    x_max = 2 * a**2 / (1 + 2 * a)                     # maximal electron energy
    dsdE =  C / a**2 * \
            (x**2 / (a * (a - x))**2 + ((x - 1)**2 - 1) / (a * (a - x)) + 2) *\
            (x <= x_max)                                     # in barn / keV
    return(dsdE)

def gauss(x, sigma):
    return  1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-x**2 / (2. * sigma**2))

# Convolution
def conv_kn_gauss(x, a, C, sigma):
    kn_gauss = lambda y, x, a, C, sigma: klein_nishina(y, a, C) * gauss(x - y, sigma)
    conv = quad(kn_gauss, -2*a, 2*a, args=(1, a, C, sigma))
    return conv


"""
# Plotting
E_photon = 1277 # in keV
m_e = co.physical_constants["electron mass energy equivalent in MeV"][0] * 10**3 # in keV / c**2
a = E_photon    / m_e
x = np.linspace(0, E_photon, 2000) / m_e
#sigma = 10
#g = gauss(x, E_photon / 2, sigma)

fig1, ax1 = plt.subplots(1, 1)
if not save_fig:
    fig1.suptitle("Klein-Nishina formula")
ax1.plot(x, kn, '-', alpha=0.8, label=('bla'))
#ax1.set_xlim(, )
#ax1.set_ylim(0, 3)
ax1.set_xlabel("$E_e$ / keV")
ax1.set_ylabel("$\\frac{d\sigma}{d E_e} / \mathrm{\\frac{barn}{keV}}$")
ax1.legend(loc=1)
if show_fig:
    fig1.show()
if save_fig:
    file_name = "klein_nishina"
    fig1.savefig(fig_dir + file_name + ".pdf")
    fig1.savefig(fig_dir + file_name + ".png")

"""
