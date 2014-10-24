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
    fig1.suptitle("Grating 1, position " + plot_suffix)
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
show_fig = False
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

# load the lattice constant K, evaluated in "gratings.py", in m
K = np.load(npy_dir + "K_1.npy")
K_std_dev = np.load(npy_dir + "K_1_std_dev.npy")
K = uc.ufloat(K, K_std_dev)

fwhm_all = []
for q1 in ["1", "2", "4", "5", "6", "7", "8"]: # for "3", both data sets are the same :(
#for q1 in ["1"]: # for "3", both data sets are the same :(
    for direction in ["left", "right"]:
        q2 = "a"
        plot_suffix = q1 + q2 
        n = 70
        m = 0.8
        if plot_suffix == "4b": # exclude extra peak
            m = 0.014
        npy_files = npy_dir + "aperture_" + plot_suffix
        t = np.load(npy_files + "_t" + ".npy")
        t = t * 10 ** 3 # time in ms!!!
        t = t - t[-1] / 2 # t = 0 doesn't lie on the physical t=0. Translate the center to t=0!
        signal = np.load(npy_files + "_ch_a" + ".npy")
        maxi_zero = search_maxi(plot_suffix, n, m)
        #plot_maxi(maxi_zero, dotted=False)
        I_zero = signal[maxi_zero][0]   # intensity of the 0th maximum
        I_zero_err = 0.03 * 2.0 # error on intensity  = 0.03 * 0.2 (scale) (oscilloscope value)
        I_zero = uc.ufloat(I_zero, I_zero_err)

        q2 = "b"
        plot_suffix = q1 + q2 
        n = 100
        mini = 0.014
        if plot_suffix == "4b":     # exclude extra peak
            m = 0.014
        npy_files = npy_dir + "aperture_" + plot_suffix
        t = np.load(npy_files + "_t" + ".npy")
        t = t * 10 ** 3             # time in ms!!!
        t = t - t[-1] / 2           # t = 0 doesn't lie on the physical t=0. Translate the center to t=0!
        signal = np.load(npy_files + "_ch_a" + ".npy")
        maxis = search_maxi(plot_suffix, n, mini)
        maxis_red = maxis[signal[maxis] < max(signal[maxis]) * 0.9] # reduced to all but the zeroth maximum
        #plot_maxi(maxis_red, dotted=False)
        I_nonzero = signal[maxis_red]    # intensities for all maxima but the 0 order maximum
        I_nonzero_err = 0.03 * 0.2 # error on intensity  = 0.03 * 0.2 (scale) (oscilloscope value)
        I_nonzero = un.uarray(I_nonzero, I_nonzero_err)
        def g(x):
        #if True:
            I = I_nonzero
            if len(I) % 2 == 0:
                g_x = 0
            else:
                g_x = I[0]
                I = I[1:]
            m = np.arange(len(I)) - len(I) / 2
            for i in range(len(I)):
                g_x += un.sqrt(I[i]) * un.cos(2 * np.pi * m[i] * x / K)
            g_x += un.sqrt(I_zero) / 2
            g_x /= (np.sum(un.sqrt(I_nonzero)) + un.sqrt(I_zero))
            return g_x
        print('position', plot_suffix[0])

        x_max = K.n * 10**6
        x_len = 200
        x = np.linspace(-x_max, x_max, x_len)
        data_fit = un.nominal_values(g(x * 10**-6))
        # compute fwhm -> parameter b = width of windows -> ratio b / K
        height = g(0)
        left_region = (x > -50) * (x < 0)
        half_max_left = np.argmin((data_fit * left_region - height/2)**2)
        x_f_l = x[half_max_left]
        fwhm = abs(x_f_l * 2)
        fwhm_all.append(fwhm)
        if show_fig:
            fig2, ax2 = plt.subplots(1, 1, figsize=plotsize)
            fig2.suptitle("Grating 1, aperture function")
            fit_plot, = ax2.plot(x, data_fit, alpha=1, zorder=2.4)
            # zorder: default: 1 for patch, 2 for lines, 3 for legend
            #error_on_fit = un.std_devs(g(x * 10**-6))
            #data_fit_min = data_fit - error_on_fit
            #data_fit_max = data_fit + error_on_fit
            #ax2.fill_between(x, data_fit_min , data_fit_max, facecolor=fit_plot.get_color(), alpha=0.2, zorder=2 )
            ax2.plot([x_f_l, x_f_r], data_fit[[half_max_left, half_max_right]], "--", alpha=1, zorder=2.5, label='fwhm')
            ax2.set_xlim(-x_max, x_max)
            ax2.set_ylim(0, 1)
            ax2.set_xlabel("$x$ / $\mu$m")
            ax2.set_ylabel("$g(x)$")
            ax2.legend(loc=4)
            fig2.show()
            if save_fig:
                fig2.savefig(fig_dir + "aperture_function.pdf")

fwhm_all = np.array(fwhm_all)
fwhm_mean = np.mean(fwhm_all)
fwhm_std = np.std(fwhm_all)
