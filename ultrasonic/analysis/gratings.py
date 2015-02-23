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
    fig1.suptitle("Grating " + plot_suffix[0])
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
        fig1.savefig(fig_dir + "gratings_maxi" + plot_suffix[0] + ".pdf")
    return 0

def func(x,*p):
    """
    p must be of the form: p = [a1, ..., a{n_peaks}, mu1, ..., m{n_peaks}, sigma1, ..., sigma{n_peaks}, c]
    """
    y = p[-1]
    for i in range(n_peaks):
        y += p[i] * np.exp(-(x - p[n_peaks + i]) ** 2 / (2 * p[2 * n_peaks + i])**2)
    return y

def weighted_avg_and_std(uarray):
    """
    Return the weighted average and standard deviation.
    Input: uncertainties.unumpy.uarray(nominal_values, std_devs)
    """
    values = un.nominal_values(uarray)
    weights = 1 / (un.std_devs(uarray) ** 2)
    average = np.average(values, weights=weights)
    variance_biased = np.average((values-average)**2, weights=weights)  # Fast and numerically precise
    variance = variance_biased / (1 - np.sum(weights ** 2)/(np.sum(weights) **2))
    return uc.ufloat(average, np.sqrt(variance))

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

# finding minima and plotting
plot_suffixes = ["1", "2b", "3", "4b", "5b"]    # original data name
neighbouring = [70] * 4 + [120]                 # number of neighboring peaks to be tested for max
minimal = [0.01, 0.0036, 0.006, 0.0045, 0.0075] # minimal value for a found maximum to be accepted
m_min = [-4, -3, -5, -5, -2]                    # minimal order observed (only valid for gratings 1, 2, 5)

# calculating the resolution
phi = uc.ufloat(2.9, 0.5) * 10**-3              # diameter of laser in m
n_visible = np.array([10, 6, 10, 9, 6])         # visible maxima at maximal illumination

# write results into a tabular environment in file
f = open(r"./gratings_K.tex", "w+")
f.write("\t\\begin{tabular}{|p{3.82cm}|p{6.18cm}|p{3.82cm}|}\n")
f.write("\t\t\hline\n")
f.write("\t\t\\rowcolor{tabcolor}\n")
f.write("\t\tGrating & Orders visible & $\overline{K}$ / ($\mu$m)  \\\\ \hline\n")
f2 = open(r"./gratings_resolution.tex", "w+")
f2.write("\t\\begin{tabular}{|p{2cm}|p{3.82cm}|p{3.82cm}|p{3.82cm}|}\n")
f2.write("\t\t\hline\n")
f2.write("\t\t\\rowcolor{tabcolor}\n")
f2.write("\t\tGrating & $n$ orders visible & $N$ lines illuminated & resolution $a$ \\\\ \hline\n")
for i in range(5):  # i+1 = nr of grating
    f.write("\t\t$%i$  & "%(i+1))
    f2.write("\t\t$%i$  & $%i$ & "%(i+1, n_visible[i]))
    plot_suffix, n, m = plot_suffixes[i], neighbouring[i], minimal[i]
    npy_files = npy_dir + "grating_" + plot_suffix
    t = np.load(npy_files + "_t" + ".npy")
    t = t * 10 ** 3 # time in ms!!!
    signal = np.load(npy_files + "_ch_a" + ".npy")
    maxis = search_maxi(plot_suffix, n, m)
    for j, maxi in enumerate(maxis):
        m = m_min[i] + j    # maximum of m-th order
        if i+1 == 3:        # for grating nr 3, the 3rd maxima are visible
            m = [-5, -4, -2, -1, 0, 1, 2, 4, 5][j]
        if i+1 == 4:        # for grating nr 4, the 2nd and 4th maxima are not visible
            m = [-5, -3, -1, 0, 1, 3, 5][j]
        if m == 0:
            t = t - t[maxi]
            break
    n_peaks = len(maxis)
    plot_maxi(maxis, dotted=False)
    K = lambda m, t: m * lamb / un.sin(theta(t) )
    Ks = []
    for j, maxi in enumerate(maxis):
        m = m_min[i] + j    # maximum of m-th order
        if i+1 == 3:        # for grating nr 3, the 3rd maxima are visible
            m = [-5, -4, -2, -1, 0, 1, 2, 4, 5][j]
        if i+1 == 4:        # for grating nr 4, the 2nd and 4th maxima are not visible
            m = [-5, -3, -1, 0, 1, 3, 5][j]
        if m != 0:      # omitting zeroth order
            t_m = t[maxi]
            Ks += [K(m, t_m)]
            #print("m =", m, "t =", t_m, "theta =", theta(t_m), "K =", K(m, t_m))
        if j ==0:
            f.write("$%i"%(m))
        else:
            f.write(", %i"%(m))
    K_mean = weighted_avg_and_std(np.array(Ks)) # calculate weighted mean and std_dev for K
    f.write("$ & $ {0:L} $ \\\\\n".format(K_mean * 10 ** 6)) 
    if i==0:
        np.save(npy_dir + "K_1.npy", K_mean.n)
        np.save(npy_dir + "K_1_std_dev.npy", K_mean.s)
    N = phi / K_mean                    # number of lines illuminated
    a = N * n_visible[i]                # resolution 
    f2.write("${0:L}$ & $ %i \pm %i $ \\\\\n".format(N)%(a.n, a.s)) 
f.write("\t\t\hline\n")
f.write("\t\end{tabular}\n")
f.close()
f2.write("\t\t\hline\n")
f2.write("\t\end{tabular}\n")
f2.close()
