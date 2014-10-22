# numpy / scipyj
import numpy as np
from scipy.optimize import curve_fit
import scipy.constants as co
from scipy.signal import argrelextrema as ext

#uncertainties
import uncertainties as uc
import uncertainties.unumpy as un

# plotting
from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern Roman']
rcParams['text.usetex'] = True
rcParams['figure.autolayout'] = True

plt.close("all")
show_fig = True
save_fig = True

fig_dir = "../figures/"
npy_dir = "./data_npy/raman_nath/"
npy_dir2 = "./data_npy/"
plotsize = (6.2, 3.83)  # width corresponds to \textwidth in latex document (ratio = golden ratio ;))

lamb = 632.8e-9 # wavelength of laser


def make_fig(fig, show=True,save=False, name="foo"):
    if show == True:
        fig.show()
    if save == True:
        fig.savefig(fig_dir + name + ".pdf")

def plot_3d():
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    U = np.load(npy_dir + "U.npy")
    for i in np.arange(20,1,-1):
        ch_a = np.load(npy_dir + "phase_%03d_ch_a.npy"%i)
        t    = np.load(npy_dir + "phase_%03d_t.npy"%i)
        ax.plot(t,t*0 + U[i-1],ch_a, c="%.3f"%(i/20))

    make_fig(fig,1,0,"3dplot")


plot_3d()
def search_maxi(signal,t,neighbours=5, minimum=0.05, n_plateau=10):
    # get local maxima
    maxis = ext(signal, np.greater_equal, order=neighbours)[0] # all maxima with next <order> points greater_equal
    maxis = maxis[signal[maxis] > minimum]   # only those greater <float>
    #maxis = maxis[t[maxis] > t[10]]   # only those maxima greater then 2% of the maximum time
    maxis_j = []
    # reduce numbers of maxima to one per plateau
    i = 0
    while i<len(maxis):
        plateau = np.where((maxis >= maxis[i])*(maxis < maxis[i] + n_plateau))[0]
        maxis_j.append(plateau[int(len(plateau) / 2)])
        i = plateau[-1] + 1
    maxis = maxis[maxis_j]
    return maxis

def plot_maxi(dotted=False):

    # without errors on x
    theta_coeff = np.load(npy_dir2 + "gauge_fit_coeff.npy")
    theta_cov = np.load(npy_dir2 + "gauge_fit_cov.npy")
    #theta_coeff_corr = uc.correlated_values(theta_coeff, theta_cov)
    theta = lambda t: np.polyval([theta_coeff[0],0], t-0.516)

    U = np.load(npy_dir + "U.npy")
    i  = 1
    signal = np.load(npy_dir + "phase_%03d_ch_a.npy"%i)
    I0 = max(signal)
    signals = []
    for i in np.arange(20,0,-1):
    #for i in [1]:
        signal = np.load(npy_dir + "phase_%03d_ch_a.npy"%i) / I0
        t    = np.load(npy_dir + "phase_%03d_t.npy"%i) *10**3
        t = theta(t)
        print(t)
        i0 = np.argmin(t[t>0]) + len(t[t<0])

        maxis = search_maxi(signal,t)
        fig1 = plt.figure()
        ax1 = plt.subplot(111)
        ax1.plot(t, signal, alpha=0.8)

        if dotted:
            [ax1.plot(t[maxi], signal[maxi], 'o', linewidth=1, label = str(k)) for k,maxi in enumerate(maxis)]
        ax1.scatter(t[i0],signal[i0],s = 500, marker= "*")
        signals += [signal[i0]]
        

        plt.title("index %d"%i)
        #ax1.plot(t, func(t,*p), alpha=0.8)
        ax1.set_xlim(-0.01,+0.01)
        ax1.set_xlabel("$\\theta$ in degree")
        ax1.set_ylabel("$U$ / V")
        #plt.legend()
        plt.close()
        make_fig(fig1,0,0,"no plot here")
    plt.figure()
    plt.scatter(U,signals)
    plt.show()


