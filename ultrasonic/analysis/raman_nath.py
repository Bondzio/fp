# numpy / scipy
import numpy as np
from scipy.optimize import curve_fit
import scipy.constants as co
from scipy.signal import argrelextrema as ext
from scipy.special import jn

#uncertainties
import uncertainties as uc
import uncertainties.unumpy as un
import stat1 as st

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

fontsize_labels = 12    # size used in latex document
rcParams['axes.labelsize'] = fontsize_labels
rcParams['xtick.labelsize'] = fontsize_labels
rcParams['ytick.labelsize'] = fontsize_labels
plotsize = (6.2, 3.83)  # width corresponds to \textwidth in latex document (ratio = golden ratio ;))
rcParams['figure.figsize'] = plotsize

plt.close("all")
show_fig = True
save_fig = True

fig_dir = "./figures/"
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

def func(U, n, alpha, A,c):
    return A*jn(n , alpha * U)**2 + c

def plot_maxi(plot_all=False):
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
        signals += [signal[i0]]

        maxis = search_maxi(signal,t)
        if plot_all:
            fig1 = plt.figure()
            ax1 = plt.subplot(111)
            ax1.plot(t, signal, alpha=0.8)

            [ax1.plot(t[maxi], signal[maxi], 'o', linewidth=1, label = str(k)) for k,maxi in enumerate(maxis)]
            ax1.scatter(t[i0],signal[i0],s = 500, marker= "*")
            

            #plt.title("index %d"%i)
            #ax1.plot(t, func(t,*p), alpha=0.8)
            ax1.set_xlim(-0.01,+0.01)
            ax1.set_xlabel("$\\theta$ in degree", fontsize = 20)
            ax1.set_ylabel("$U$ / V",fontsize = 20)

            ax1.xaxis.set_tick_params(labelsize = 20)
            ax1.yaxis.set_tick_params(labelsize = 20)

            #plt.legend()
            plt.close()
            make_fig(fig1,0,1,"raman_%03d"%i)

    signals = np.array(signals[::-1])
    sigma = (0.03 + signals*0) / I0
    p0 = [ 0.5, 1, 0.1]
    func_0= lambda U,alpha,A,c: func(U, 0, alpha, A, c)
    
    p,cov = curve_fit(func_0, U, signals, p0 = p0, sigma = sigma, absolute_sigma = True)
    p_uc = uc.correlated_values(p, cov)


    f1 = open("coefficients_bessel0.tex","a")
    st.la_coeff2(f1, p,cov, ["\\alpha","A","c"])
    f1.close()


    U_fit = np.linspace(min(U),max(U),1000)
    data_fit = func_0(U_fit, *p)


    fig = plt.figure()
    ax  = plt.subplot(111)

    plt.errorbar(U,signals, yerr = sigma, fmt="x")
    plt.plot(U_fit,data_fit)

    #error_on_fit = un.std_devs(func_0(U_fit, *p_uc))
    #data_fit_min = data_fit - error_on_fit
    #data_fit_max = data_fit + error_on_fit

    #plt.fill_between(U_fit, data_fit_min , data_fit_max,facecolor="r", color="b", alpha=0.3 )

    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    textstr = '$A + J_0^2(\\alpha \cdot U) + c$ \n$A=%.3f\pm%.3f$\n$\\alpha=(%.3f\pm%.3f)1/V$\n$c=%.3f\pm%.3f$'%(p[1],p_uc[1].s,p[0],p_uc[0].s,p[2],p_uc[2].s)
    ax.text(0.6,0.8, textstr, transform=ax.transAxes, fontsize=12, va='top', bbox=props)
    plt.xlabel("applied Voltage / V")
    plt.ylabel("relative Intensity of I_0")
    plt.grid(True)
    make_fig(fig,0,1,"besselfit_0")

plot_maxi(False)
