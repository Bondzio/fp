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


def step(x, x_max):
    y = 1 *  (x <= x_max)
    return y

def gauss0(x, sigma):
    return  1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-x**2 / (2. * sigma**2))

def conv_step_gauss(x, x_max, sigma, A):
    step_gauss = lambda y, x, x_max, sigma: step(x - y, x_max) * gauss0(y, sigma)
    if type(x) == float:
        conv = quad(step_gauss, -2*a, 2*a, args=(x, x_max, sigma))[0]
    else:
        conv = np.array([quad(step_gauss, -np.inf, np.inf, args=(x_i, x_max, sigma))[0] for x_i in x])
    return conv * A

def conv_analytical(x, x_max, sigma, A):
    '''
    analytical solution of convolution of step and gauss
    '''
    return A / 2 * erfc((x - x_max) / (np.sqrt(2) * sigma)) 

# Parameters of convolution
x0 = 20
n_x = 1000
x = np.linspace(0, x0, n_x)

x_max  = 10
A = 3
sigma = 1

steps = 3 * step(x, x_max)
conv = conv_step_gauss(x, x_max, sigma, A)
#conv = conv_analytical(x, x_max, sigma, A)

#Plotting
fig1, ax1 = plt.subplots(1, 1)
if not save_fig:
    fig1.suptitle("Klein-Nishina formula")
ax1.plot(x, steps, '-', alpha=0.8, label=('Step'))
ax1.plot(x, conv, '-', alpha=0.8, label=('Convolution(quad)'))
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


