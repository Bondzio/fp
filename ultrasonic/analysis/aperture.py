import numpy as np
import pylab as plt
from matplotlib import rcParams
from scipy.optimize import curve_fit
import scipy.constants as co
import uncertainties as uc
import uncertainties.unumpy as un
from scipy.signal import argrelextrema as ext
import seaborn as sns

def search_maxi(plot_suffix, neighbours=100, minimum=0.01, n_plateau=50):
    # get local maxima
    maxis = ext(signal, np.greater_equal, order=neighbours)[0] # all maxima with next <order> points greater_equal
    maxis = maxis[signal[maxis] > minimum]   # only those greater <float>
    maxis = maxis[t[maxis] > t[10]]   # only those maxima greater then 2% of the maximum time
    maxis_j = []
    # reduce numbers of maxima to one per plateau
    i = 0
    while i<len(maxis):
        plateau = np.where((maxis >= maxis[i])*(maxis < maxis[i] + n_plateau))[0]
        maxis_j.append(plateau[int(len(plateau) / 2)])
        i = plateau[-1] + 1
    maxis = maxis[maxis_j]
    return maxis

def plot_maxi(maxis, dotted=False):
    fig1, ax1 = plt.subplots(1, 1, figsize=plotsize)
    fig1.suptitle("Grating " + plot_suffix)
    peak_plot, = ax1.plot(t, signal, alpha=0.8)
    if dotted:
        [ax1.plot(t[maxi], signal[maxi], 'o', color=peak_plot.get_color(), linewidth=1) for maxi in maxis]
    [ax1.plot([t[maxi]] * 2, [0, signal[maxi]], '--', color=peak_plot.get_color(), linewidth=1) for maxi in maxis]
    #ax1.plot(t, func(t,*p), alpha=0.8)
    ax1.set_xlim(t[0], t[-1])
    ax1.set_xlabel("$t$ / ms")
    ax1.set_ylabel("$U$ / V")
    if show_fig:
        fig1.show()
    if save_fig:
        fig1.savefig(fig_dir + "aperture_" + plot_suffix[0] + ".pdf")
    return 0

fontsize_labels = 12    # size used in latex document
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern Roman']
rcParams['text.usetex'] = True
rcParams['figure.autolayout'] = True
rcParams['axes.labelsize'] = fontsize_labels
rcParams['xtick.labelsize'] = fontsize_labels
rcParams['ytick.labelsize'] = fontsize_labels

plt.close("all")
show_fig = True
save_fig = True
fig_dir = "../figures/"
npy_dir = "./data_npy/"
plotsize = (6.2, 3.83)  # width corresponds to \textwidth in latex document (ratio = golden ratio ;))

lamb = 632.8e-9 # wavelength of laser
"""
import coefficients for theta(t) = omega * t + phi_0; 
created in "./calibration.py", theta_coeff = np.array([omega, phi_0])
"""
theta_coeff = np.load(npy_dir + "gauge_fit_coeff.npy")
theta_cov = np.load(npy_dir + "gauge_fit_cov.npy")
theta_coeff_corr = uc.correlated_values(theta_coeff, theta_cov)
s_t_m = 0.01 # error on the maxima in ms
theta = lambda t: np.polyval(theta_coeff_corr, un.uarray(t, s_t_m))
#theta = lambda t: np.polyval(theta_coeff_corr, t) # strangly, this version yields smaller errors


for q1 in ["1", "2", "4", "5", "7", "8"]: # for "3", both data sets are the same :(
    q2 = "a"
    plot_suffix = q1 + q2 
    print(plot_suffix)
    n = 70
    m = 0.014
    if plot_suffix == "4b": # exclude extra peak
        m = 0.014
    npy_files = npy_dir + "aperture_" + plot_suffix
    t = np.load(npy_files + "_t" + ".npy")
    t = t * 10 ** 3 # time in ms!!!
    t = t - t[-1] / 2 # t = 0 doesn't lie on the physical t=0. Translate the center to t=0!
    signal = np.load(npy_files + "_ch_a" + ".npy")
    maxis = search_maxi(plot_suffix, n, m)
    n_peaks = len(maxis)
    print(n_peaks)
    #p0 = np.concatenate((signal[maxis], t[maxis], np.array([0.02]*n_peaks), [0]), axis=0)
    #p, cov = curve_fit(func, t, signal, p0=p0)
    plot_maxi(maxis, dotted=False)

    q1 = "b"
