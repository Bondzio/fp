import numpy as np
import pylab as plt
from matplotlib import rcParams
from scipy.optimize import curve_fit
import scipy.constants as co
import uncertainties as uc
import uncertainties.unumpy as un
from scipy.signal import argrelextrema as ext
import seaborn as sns

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

def plot():
    fig1, ax1 = plt.subplots(1, 1, figsize=plotsize)
    fig1.suptitle(title)
    ax1.plot(U, n, ".")
    ax1.set_yscale("log")
    #ax1.set_xlim(t[0], t[-1])
    #ax1.set_xlabel("$t$ / ms")
    #ax1.set_ylabel("$U$ / V")
    if show_fig:
        fig1.show()
    #if save_fig:
    #   fig1.savefig(fig_dir + "gratings_maxi" + plot_suffix[0] + ".pdf")
    return 0

titles = ["background", "uranium"]
for title in titles:
    npy_files = npy_dir + title + "_"
    t = np.load(npy_files + "t" + ".npy")
    U = np.load(npy_files + "U" + ".npy")
    n = np.load(npy_files + "n" + ".npy")
    plot()


