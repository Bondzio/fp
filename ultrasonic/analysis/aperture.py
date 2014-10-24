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
rcParams['figure.figsize'] = (6.2, 3.83)  # in inches; width corresponds to \textwidth in latex document (golden ratio)

plt.close("all")
show_fig = False
save_fig = True
fig_dir = "../figures/"
npy_dir = "./data_npy/"

#########################################################################################################

def search_maxi(suffix, neighbours=100, minimum=0.01, n_plateau=50):
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
    fig1, ax1 = plt.subplots(1, 1)
    if not save_fig:
        fig1.suptitle("Grating 1, position " + suffix)
    peak_plot, = ax1.plot(t, signal, alpha=0.8)
    if dotted:
        [ax1.plot(t[maxi], signal[maxi], 'o', color=peak_plot.get_color(), linewidth=1) for maxi in maxis]
    [ax1.plot([t[maxi]] * 2, [0, signal[maxi]], '--', color=peak_plot.get_color(), linewidth=1) for maxi in maxis]
    ax1.set_xlim(t[0], t[-1])
    ax1.set_xlabel("$t$ / ms")
    ax1.set_ylabel("$U$ / V")
    if show_fig:
        fig1.show()
    if save_fig:
        fig1.savefig(fig_dir + "aperture_" + suffix + ".pdf")
        fig1.savefig(fig_dir + "aperture_" + suffix + ".png")
    return 0

###################################################################################################

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

#for q1 in ["1", "2", "3", "4", "5", "6", "7", "8"]: # for "3", both data sets are the same; "4" cannot be used / range too small:(
plot_to_work_on = "5"
for q1 in [plot_to_work_on]: # 
    for side in ["left", "right"]:
        q2 = "a"
        suffix = q1 + q2 
        n = 200
        m = 0.8
        if suffix == "4b": # exclude extra peak
            m = 0.014
        npy_files = npy_dir + "aperture_" + suffix
        t = np.load(npy_files + "_t" + ".npy")
        t = t * 10 ** 3 # time in ms!!!
        signal = np.load(npy_files + "_ch_a" + ".npy")
        maxi_zero = search_maxi(suffix, n, m)
        t = t - t[maxi_zero]            #  t = 0 doesn't lie on the physical t=0. Translate the center to t=0!
        I_zero = signal[maxi_zero][0]   # intensity of the 0th maximum
        I_zero_err = 0.03 * 2.0 # error on intensity  = 0.03 * 0.2 (scale) (oscilloscope value)
        I_zero = uc.ufloat(I_zero, I_zero_err)
        if side == "left":
            # plotting maxima
            plot_maxi(maxi_zero, dotted=False)

        q2 = "b"
        suffix = q1 + q2 
        n = 70
        if side == "left":          # choose minimal value for accepting maxima depending on the side looked at
            mini = 0.014
        if side == "right":
            mini = 0.010
            if suffix == "6b":     # exclude extra peak
                mini = 0.014
            if suffix == "8b":     # exclude extra peak
                mini = 0.02
        npy_files = npy_dir + "aperture_" + suffix
        t = np.load(npy_files + "_t" + ".npy")
        t = t * 10 ** 3             # time in ms!!!
        t = t - t[-1] / 2           # t = 0 doesn't lie on the physical t=0. Translate the center to t=0!
        signal = np.load(npy_files + "_ch_a" + ".npy")
        maxis = search_maxi(suffix, n, mini)
        if suffix == "4b":
            maxis_at_zero =  maxis[(signal[maxis] > max(signal[maxis]) * 0.9) * (t[maxis] > -0.05)] # reduced to all but the zeroth maximum
        else:
            maxis_at_zero =  maxis[signal[maxis] > max(signal[maxis]) * 0.9] # reduced to all but the zeroth maximum
        maxis_red = maxis[signal[maxis] < max(signal[maxis]) * 0.9] # reduced to all but the zeroth maximum
        if side == "left":
            maxis_one_side = maxis_red[maxis_red < maxis_at_zero[0]]
            direction = -1
        if side == "right":
            maxis_one_side = maxis_red[maxis_red > maxis_at_zero[-1]]
            direction = +1
        I_nonzero = signal[maxis_one_side]  # intensities for all maxima but the 0 order maximum
        I_nonzero_err = 0.03 * 0.2          # error on intensity  = 0.03 * 0.2 (scale) (oscilloscope value)
        I_nonzero = un.uarray(I_nonzero, I_nonzero_err)
        # plotting maxima
        if side == "left":
            maxis_left = np.copy(maxis_one_side)
            print(q1, "left")
        if side == "right":
            maxis_both = np.concatenate((maxis_left, maxis_one_side))
            print(q1, "right")
            plot_maxi(maxis_both, dotted=False)

        if (q1 == plot_to_work_on) * (side == "left"):
            I_nonzero_left = I_nonzero[::-1]
        if (q1 == plot_to_work_on) * (side == "right"):
            Is = (I_nonzero + I_nonzero_left) / 2
            ms = np.arange(1, len(Is) + 1)
            def g(x):                           # aperture function
                g_x = 0
                for [I, m] in np.column_stack((Is, ms)):                # orders m != 0
                    g_x += un.sqrt(I) * un.cos(2 * np.pi / K * m * (x * 10 ** -6))
                g_x += un.sqrt(I_zero) / 2                              # zeroth order
                g_x /= (np.sum(un.sqrt(I_nonzero)) + un.sqrt(I_zero))   # normalize
                return g_x
            f = open(r"./aperture_I.tex", "w+")
            f.write("\t\\begin{tabular}{|p{3.82cm}|p{3.82cm}|p{3.82cm}|p{3.82cm}|}\n")
            f.write("\t\t\hline\n")
            f.write("\t\t\\rowcolor{tabcolor}\n")
            f.write("\t\tOrder $m$ & $I_\mathrm{left}$ / V & $I_\mathrm{right}$ / V & $\overline{I}$ / V  \\\\ \hline\n")
            f.write("\t\t$0$ & \t & \t& ${0:L}$ \\\\ \n".format(I_zero))
            for [m, I_l, I_r, I ] in np.column_stack((ms, I_nonzero_left, I_nonzero, Is)):                # orders m != 0
                f.write("\t\t$%i$ & ${0:L}$ & ${1:L}$ & ${2:L}$ \\\\ \n".format(I_l, I_r, I)%m)
            f.write("\t\t\hline\n")
            f.write("\t\end{tabular}\n")
            f.close()

            # compute fwhm -> parameter b = width of windows -> ratio b / K
            fwhm_error = True
            x_max = K.n * 10**6
            x_len = 200
            x = np.linspace(-x_max, x_max, x_len)
            data_fit = un.nominal_values(g(x))
            height = g(0)
            region = (x > 0) * (x < 50)
            half_max = np.argmin((data_fit * region - height/2)**2)
            x_fwhm = x[half_max]
            b = abs(x_fwhm * 2)
            # plot b = fwhm
            fig2, ax2 = plt.subplots(1, 1)
            if not save_fig:
                fig2.suptitle("Grating 1, aperture function")
            fit_plot, = ax2.plot(x, data_fit, alpha=1, zorder=2.4)
            # #zorder: default: 1 for patch, 2 for lines, 3 for legend
            # fit errors of g(x)
            #error_on_fit = un.std_devs(g(x * 10**-6))
            #data_fit_min = data_fit - error_on_fit
            #data_fit_max = data_fit + error_on_fit
            #ax2.fill_between(x, data_fit_min , data_fit_max, facecolor=fit_plot.get_color(), alpha=0.2, zorder=2 )
            xrange_fwhm = np.linspace(-x_fwhm, x_fwhm, 50)
            g_0 = xrange_fwhm * 0 
            g_fwhm = g_0 + data_fit[half_max]
            ax2.fill_between(xrange_fwhm, g_0, g_fwhm, facecolor=fit_plot.get_color(), alpha=0.3, zorder=2 )
            ax2.plot([-x_fwhm, x_fwhm], [data_fit[half_max]] * 2, "--", alpha=1, zorder=2.5, label='fwhm')
            if fwhm_error:
                relative_error = 0.1 # on x
                b = uc.ufloat(b, 2 * relative_error * b)
                textstr = "$b = ({0:L})".format(b) + r"\, \mu\mathrm{m}$"
                x_err = x_fwhm + x_fwhm * relative_error
                xrange_fwhm = np.linspace(-x_err, x_err, 50)
                g_0 = xrange_fwhm * 0 
                g_fwhm = g_0 + g(x_err).n
                ax2.fill_between(xrange_fwhm, g_0, g_fwhm, facecolor=fit_plot.get_color(), alpha=0.15, zorder=2 )
                x_err = x_fwhm - x_fwhm * relative_error 
                xrange_fwhm = np.linspace(-x_err, x_err, 50)
                g_0 = xrange_fwhm * 0 
                g_fwhm = g_0 + g(x_err).n
                ax2.fill_between(xrange_fwhm, g_0, g_fwhm, facecolor=fit_plot.get_color(), alpha=0.15, zorder=2 )
            # place a text box in upper right in axes coords, using eqnarray
            props = dict(boxstyle='round', facecolor='white', alpha=0.5, zorder=4)
            ax2.text(0.1, 0.85, textstr, transform=ax2.transAxes, fontsize=fontsize_labels, va='top', bbox=props)
            f2 = open(r"./aperture_b.tex", "w+")
            f2.write("\t" + textstr[1:-1] + "\n")
            f2.write("\t" + r"\frac{b}{K} = \frac" + "{{ {0:L} }}{{ {1:L} }} = {2:L} \, .\n".format(b, K*10**6, b/(K*10**6)))
            f2.close()

            ax2.set_xlim(-x_max, x_max)
            ax2.set_ylim(0, 1)
            ax2.set_xlabel("$x$ / $\mu$m")
            ax2.set_ylabel("$g(x)$")
            ax2.legend(loc=4)
            if show_fig:
                fig2.show()
            if save_fig:
                fig2.savefig(fig_dir + "aperture_function.pdf")
            
