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
    maxis = maxis[t[maxis] > t[-1] * 0.02]   # only those maxima greater then 2% of the maximum time
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
    fig1.suptitle("Grating " + plot_suffix[0])
    peak_plot, = ax1.plot(t, signal, alpha=0.8)
    if dotted:
        [ax1.plot(t[maxi], signal[maxi], 'o', color=peak_plot.get_color(), linewidth=1) for maxi in maxis]
    [ax1.plot([t[maxi]] * 2, [0, signal[maxi]], '--', color=peak_plot.get_color(), linewidth=1) for maxi in maxis]
    #ax1.plot(t, func(t,*p), alpha=0.8)
    ax1.set_xlabel("$t$ / ms")
    ax1.set_ylabel("$U$ / V")
    if show_fig:
        fig1.show()
    return 0

def func(x,*p):
    """
    p must be of the form: p = [a1, ..., a{n_peaks}, mu1, ..., m{n_peaks}, sigma1, ..., sigma{n_peaks}, c]
    """
    y = p[-1]
    for i in range(n_peaks):
        y += p[i] * np.exp(-(x - p[n_peaks + i]) ** 2 / (2 * p[2 * n_peaks + i])**2)
    return y

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
save_fig = False
fig_dir = "../figures/"
npy_dir = "./data_npy/"
plotsize = (6.2, 3.83)  # width corresponds to \textwidth in latex document (ratio = golden ratio ;))

lamb = 632.8e-6
K = 10e-3
p = np.load(npy_dir + "gauge_fit_coeff.npy")
cov = np.load(npy_dir + "gauge_fit_cov.npy")
p_uc = uc.correlated_values(p, cov)

# finding minima and plotting
plot_suffixes = ["1", "2b", "3", "4b", "5b"]
neighbouring = [70] * 4 + [120]
minimal = [0.01, 0.0036, 0.006, 0.0045, 0.0075]
for i in range(5):
    plot_suffix, n, m = plot_suffixes[i], neighbouring[i], minimal[i]
    npy_files = npy_dir + "grating_" + plot_suffix
    t = np.load(npy_files + "_t" + ".npy")
    t = t * 10 ** 3 # time in ms!!!
    signal = np.load(npy_files + "_ch_a" + ".npy")
    maxis = search_maxi(plot_suffix, n, m)
    n_peaks = len(maxis)
    #p0 = np.concatenate((signal[maxis], t[maxis], np.array([0.02]*n_peaks), [0]), axis=0)
    #p, cov = curve_fit(func, t, signal, p0=p0)
    plot_maxi(maxis, dotted=False)



#        a1,a2,a3,a4,a5,mu1,mu2,mu3,mu4,mu5,sigma1,sigma2,sigma3,sigma4,sigma5,c):


if save_fig:
    fig1.savefig(fig_dir + "calibrate_peaks.pdf")
    #fig2.savefig(fig_dir + "calibrate_fit.pdf")

#fig2.show()
