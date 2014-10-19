import numpy as np
import pylab as plt
from scipy.optimize import curve_fit
import scipy.constants as co
import uncertainties as uc
import uncertainties.unumpy as un
from scipy.signal import argrelextrema as ext
import seaborn as sns

plt.close("all")
plot = True

fig1, ax1 = plt.subplots(1, 1, figsize=(8.09, 5))
fig2, ax2 = plt.subplots(1, 1, figsize=(8.09, 5))
# Gauge, measure b is much more useful
maxi_both = []
for q in "ab":
    npy_files = "./data_npy/gauge_" + q
    t = np.load(npy_files + "_t" + ".npy")
    signal = np.load(npy_files + "_ch_a" + ".npy")

    # get local maxima
    maxi = ext(signal, np.greater_equal, order=70)[0] # all maxima with next <order> points greater_equal
    maxi = maxi[signal[maxi] >  0.01]   # only those greater <float>
    maxi_j = []
    # reduce numbers of maxima to one per plateau
    i = 0
    while i<len(maxi):
        plateau = np.where((maxi >= maxi[i])*(maxi < maxi[i] + 100))[0]
        maxi_j.append(plateau[int(len(plateau) / 2)])
        i = plateau[-1] + 1
    maxi = maxi[maxi_j]
    
    maxi_both.append(maxi)

    ax1.plot(t, signal, alpha=0.8)
    ax1.plot(t[maxi], signal[maxi], 'o')

# Take average of both maxima
t_max_both = t[np.array(maxi_both).T]
t_max = np.zeros(len(t_max_both))
std_dev = t_max * 0
for i, t_max_pair in enumerate(t_max_both):
    t_max[i] = np.average(t_max_pair)
    std_dev[i] = np.std(t_max_pair)
m = np.arange(-3, 3) # order of maxima: in this case: -3 -- 2
# linear fit
p, cov = np.polyfit(m, t_max, 1, w=1/std_dev, cov=True)
p_uc = uc.correlated_values(p, cov)
x_min, x_max =(-3.2, 2.2)
x_fit = np.linspace(x_min, x_max, 200)
data_fit = np.polyval(p, x_fit)
error_on_fit = un.std_devs(np.polyval(p_uc, x_fit))
data_fit_min = data_fit - error_on_fit
data_fit_max = data_fit + error_on_fit
ax2.fill_between(x_fit, data_fit_min , data_fit_max,facecolor="r", color="b", alpha=0.2 )
ax2.plot(x_fit, data_fit, alpha=0.4)

ax2.errorbar(m, t_max, yerr=std_dev, fmt='k,')
ax2.set_xlim(x_min, x_max)
ax2.set_ylim(0, 0.0008)
fig1.show()
fig2.show()




"""
# Plotting the function for the intensity of a grating with N slits of width b and lattice constant K, all quantities in mm
lamb = 632.8e-6
k = 2 * np.pi / lamb
K = 10e-3
N = 1000
b = 0.5 / K
beta    = lambda theta: k * b * np.sin(theta) / 2
gamma   = lambda theta: k * K * np.sin(theta) / 2
theta = np.linspace(-2*np.pi, 2*np.pi, 100)
def f(theta):
    return (np.sin(beta(theta)) / beta(theta)) ** 2 * (np.sin(N * gamma(theta)) / (N * np.sin(gamma(theta)))) ** 2
fig, ax = plt.subplots(1, 1, figsize=(8.09, 5))
ax.plot(theta, f(theta))
#ax.set_ylim(-1, 1)
fig.show()
"""
