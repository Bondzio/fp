import numpy as np
import pylab as plt
from matplotlib import rcParams
from scipy.optimize import curve_fit
import scipy.constants as co
import uncertainties as uc
import uncertainties.unumpy as un
from scipy.signal import argrelextrema as ext
import seaborn as sns
import stat1 as st

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
save_fig = False
do_plot_raw         = True
do_write_to_tex     = True
do_plot_gaussians   = True
do_plot_mobility    = True
do_plot_life_time   = True
do_plot_diffusion   = True
fig_dir = "../figures/"
npy_dir = "./data_npy/"

props = dict(boxstyle='round', facecolor='white', edgecolor = 'white', alpha=0.7)

# file names
original_all = np.array([2, 4, 6, 8, 10, 12, 14, 17, 18, 20, 22, 24, 26, 28, 33])
smooth_all = np.array([3, 5, 7, 9, 11, 13, 15, 16, 19, 21, 23, 25, 27, 29, 34])
d_all = np.array([1.95, 2.52, 3.93, 5.41, 7.48, 8.49, 10.27, 8.02, 7.55, 6.86, 5.86, 4.37, 3.91, 3.66, 1.08])

# choose valid plots.
choose = 0
if choose == 0:
    chosen = np.array([5, 6, 8, 9, 11, 12, 13, 14, 15]) -1
    t_grid = np.linspace(360, 380, 100) # grid for plotting the linear fit in \mu s
    box_pos = np.array([(0.02, 0.4), (0.6, 0.95), (0.6, 0.6), (0.6, 0.8), (0.6, 0.6), 
            (0.6, 0.6), (0.6, 0.96), (0.6, 0.95), (0.6, 0.95)])
elif choose == 1:
    chosen = np.array([1, 2, 3, 4]) -1
    t_grid = np.linspace(0, 20, 100) # grid for plotting the linear fit in \mu s
    box_pos = np.array([(0.02, 0.4), (0.6, 0.95), (0.6, 0.6), (0.6, 0.6)])
original = original_all[chosen]
smooth = smooth_all[chosen]
d = d_all[chosen]
# sort
order = np.argsort(d)
d = d[order]
original = original[order]
smooth = smooth[order]
box_pos = box_pos[order]

pairs = zip(original, smooth)
s_d = 0.5  # mm error on d

# mm -> m
#d = d * 10**-3
#s_d = s_d  * 10**-3

ps = np.zeros([len(d), 4])
covs = np.zeros([len(d), 4, 4])
chi2_tests = np.zeros([len(d)])
for k, (i, j) in enumerate(pairs): # 69
    npy_file = npy_dir + "haynes_shockley_%i"%i + ".npy"
    t, signal = np.load(npy_file)
    npy_file = npy_dir + "haynes_shockley_%i"%j + ".npy"
    t_s, signal_s = np.load(npy_file)

    t = t * 10**6                   # s -> \mu s!
    signal = signal * 10 ** 3       # V -> mV
    signal_s = signal_s * 10 ** 3   # V -> mV
    t_label = "$t \,/ \, \mathrm{\mu s}$"
    t_c_label = "$t_c \,/ \, \mathrm{\mu s}$"
    u_label = "$U \,/ \, \mathrm{mV}$"

    # fluctuation
    fluct_side = ["left"] * 4 + ["right"] * 5
    if fluct_side[k] == "left":
        fluct_boundary = 500
    elif fluct_side == "right":
        fluct_boundary = -500
    s_signal = np.std(t[:fluct_boundary])

    # pre-guessing
    index = np.argmax(signal) 
    t_max = t[index]
    signal_mean = np.mean(signal_s)
    A = signal_s[index] - signal_mean
    def func(t, A, t_c, sigma, c):
        return A / np.sqrt(2 * np.pi * sigma**2) * np.exp(1)**(-((t - t_c) / sigma)**2 / 2 ) + c

    # p0 is the initial guess for the fitting coefficients 
    p0_s = [A, t_max, 0.5, signal_mean]
    p_s, cov_s = curve_fit(func, t, signal_s, p0_s) # fitting on smooth data
    p0 = p_s.copy()
    p0[3] = np.mean(signal)
    p, cov = curve_fit(func, t, signal, p0, sigma=s_signal)#, absolute_sigma-True)
    p_corr = uc.correlated_values(p, cov)
    ps[k] = p
    covs[k] = cov
    # chi square
    n_d = 5
    chi2 = np.sum(((func(t, *p) - signal) / s_signal) ** 2 )
    chi2_tests[k] = chi2/n_d

    def plot_raw():
        """
        Plots raw data and fits + parameters in a box
        """
        fig1, ax1 = plt.subplots(1, 1)
        if not save_fig:
            fig1.suptitle("Haynes and Shockley, Pair: %i, %i"%(i, j))
        ax1.plot(t, func(t, *p_s), '-', alpha=0.8)
        ax1.plot(t, func(t, *p), '-', alpha=0.8)
        ax1.plot(t, signal, '.', alpha=0.8, label=(r'original signal'))
        ax1.plot(t, signal_s, '.', alpha=0.8, label=(r'smooth signal'))
        textstr = '\\begin{eqnarray*}\
                d &=& (%.1f \pm 0.1)\, \mathrm{mm} \\\\ \
                c(t) &=& \\frac{A}{\sqrt{2\pi \sigma_t}} \
                    \exp\left(-\\frac{1}{2}\left(\\frac{t - t_c}{\sigma_t}\\right)^2\\right)\\\\ \
                A           &=& \,\,(%.1f \pm %.1f) \, \mathrm{mm / \mu s} \\\\  \
                t_c         &=& \,\,(%.2f \pm %.2f) \, \mathrm{\mu s} \\\\  \
                \sigma_t    &=& (%.2f \pm %.2f) \, \mathrm{\mu s} \
                \end{eqnarray*}'%(d[k], p[0], p_corr[0].s, p[1], p_corr[1].s, p[2], p_corr[2].s)
        if k == 8:
            textstr = '\\begin{eqnarray*}\
                    d &=& %.1f \, \mathrm{mm} \\\\ \
                    c(t) &=& \\frac{A}{\sqrt{2\pi \sigma_t}} \
                        \exp\left(-\\frac{1}{2}\left(\\frac{t - t_c}{\sigma_t}\\right)^2\\right)\\\\ \
                    A           &=& \,\,(%.1f \pm %.1f) \, \mathrm{mm / \mu s} \\\\  \
                    t_c         &=& \,\,(%.3f \pm %.3f) \, \mathrm{\mu s} \\\\  \
                    \sigma_t    &=& (%.3f \pm %.3f) \, \mathrm{\mu s} \
                    \end{eqnarray*}'%(d[k], p[0], p_corr[0].s, p[1], p_corr[1].s, p[2], p_corr[2].s)
        ax1.text(box_pos[k][0], box_pos[k][1], textstr, transform=ax1.transAxes, fontsize=18, va='top', bbox=props)
        ax1.set_xlim(t[0], t[-1])
        ax1.set_xlabel(t_label)
        ax1.set_ylabel(u_label)
        if show_fig:
            fig1.show()
        if save_fig:
            file_name = "haynes_shockley_raw_%i"%i
            fig1.savefig(fig_dir + file_name + ".pdf")
            fig1.savefig(fig_dir + file_name + ".png")
        return 0
    if do_plot_raw:
        plot_raw()

def write_to_tex():
    """
    Writes result of gaussian fits to .tex file
    """
    f = open("h_s_fit_parameters.tex", "w+")
    f.write("\t\\begin{tabular}{|p{2cm}|p{3cm}|p{3cm}|p{3cm}|p{2cm}|}\n")
    f.write("\t\t\hline\n")
    f.write("\t\t\\rowcolor{tabcolor}\n")
    f.write("\t\t$d \, / \, \mathrm{mm}$        & $A \, / \, \mathrm{\\frac{mm}{\mu s}}$ & \n \
    \t\t\t$t_c \, / \, \mathrm{\mu s}$    & $\sigma_t \, / \, \mathrm{\mu s}$ & \n \
    \t\t\t$\chi^2 / n_d$ \\\\ \hline\n")
    for k in range(len(ps)):
        p = ps[k]
        s_p = np.sqrt(np.diag(covs[k]))
        chi2_test = chi2_tests[k]
        f.write('\t\t$%.1f$ & $%.1f \pm %.1f$ & $%.2f \pm %.2f$ & $%.2f \pm %.2f$ & $%.1f$\\\\ \n'
                %(d[k], p[0], s_p[0], p[1], s_p[1], p[2], s_p[2], chi2_test))
    f.write("\t\t\hline\n")
    f.write("\t\end{tabular}\n")
    f.close()
    return 0
if do_write_to_tex:
    write_to_tex()

def plot_gaussians():
    """
    plotting all gaussians in one plot (setting offset to zero)
    """
    offsets = ps.T[-1].copy()   # save ?
    ps.T[-1] = 0
    fig1, ax1 = plt.subplots(1, 1)
    if not save_fig:
        fig1.suptitle("Haynes and Shockley, all gaussians")
    [ax1.plot(t, func(t, *p), '-', alpha=0.8, label='$d = %.1f \, \mathrm{mm}$'%d[k]) for k, p in enumerate(ps)]
    ax1.set_xlim(t[0], t[-1])
    ax1.set_xlabel(t_label)
    ax1.set_ylabel(u_label)
    ax1.legend(loc=1)
    if show_fig:
        fig1.show()
    if save_fig:
        file_name = "haynes_shockley_all_gauss"
        fig1.savefig(fig_dir + file_name + ".pdf")
        fig1.savefig(fig_dir + file_name + ".png")
if do_plot_gaussians:
    plot_gaussians()

A, t_c, sigma_t, c = ps.T
s_A, s_t_c, s_sigma_t, s_c = np.array([np.sqrt(np.diag(covs[i])) for i in range(9)]).T
sigma_t_corr = un.uarray(sigma_t, s_sigma_t)

# fit mobility of free electrons x_c(t) = mu_e * E * t
def x_c_func(t, mu_e, t0):
    return mu_e* (t - t0)
x_c_fit, cov_x_c_fit = curve_fit(x_c_func, t_c, d, p0=None, sigma=s_d)
s_x_c_fit = np.sqrt(np.diag(cov_x_c_fit))
mu_e = uc.ufloat(x_c_fit[0], s_x_c_fit[0])
n_d = 3 # degrees of freedom: # fit parameter + 1
chi2 = np.sum(((x_c_func(t_c, *x_c_fit) - d) / s_d) ** 2 )

def plot_mobility():
    """
    plots linear fit for electron mobility
    """
    fig1, ax1 = plt.subplots(1, 1)
    if not save_fig:
        fig1.suptitle("Haynes and Shockley, Fit: $e^-$ mobility")
    ax1.errorbar(t_c, d, xerr=s_t_c, yerr=s_d,   fmt='.', alpha=0.8)
    next(ax1._get_lines.color_cycle)
    ax1.plot(t_grid, x_c_func(t_grid, *x_c_fit), '-', alpha=0.8)
    textstr = 'Fit parameters for $e^-$ mobility $\mu_n$: \n\
            \\begin{eqnarray*}x_c(t) &=& (\mu_n E)(t - t_0) \\\\ \
            (\mu_n E)     &=& \,\,(%.2f \pm %.2f) \, \mathrm{mm / \mu s} \\\\  \
            t_0    &=&(%.1f \pm %.1f) \, \mathrm{\mu s}\\\\ \
            \chi^2 / n_d &=& %.1f\
            \end{eqnarray*}'%(x_c_fit[0], s_x_c_fit[0], x_c_fit[1], s_x_c_fit[1], chi2 / n_d)
    ax1.text(0.1, 0.95, textstr, transform=ax1.transAxes, va='top', bbox=props)
    ax1.set_xlim(t_grid[0], t_grid[-1])
    ax1.set_xlabel(t_c_label)
    ax1.set_ylabel("d / mm")
    if show_fig:
        fig1.show()
    if save_fig:
        file_name = "haynes_shockley_mu_e_d"
        fig1.savefig(fig_dir + file_name + ".pdf")
        fig1.savefig(fig_dir + file_name + ".png")
    return 0
if do_plot_mobility:
    plot_mobility()

# transform t -> t - t_0 (no offset)
t_c_tr = t_c - x_c_fit[1]
t_grid_tr = t_grid - x_c_fit[1]

# fit life time of electrons A - C * exp(-t / tau)
def A_func(t, C, tau):
    return C * np.exp(1) ** (- t / tau)
def A_func_real(t, C, tau):
    return C /mu_e.n* np.exp(1) ** (- t / tau)
p0 = [20, 4]
C_fit, cov_C_fit = curve_fit(A_func, t_c_tr, A, p0=p0, sigma=s_A)
C_corr = uc.correlated_values(C_fit, cov_C_fit)
A_corr = np.array([C_corr[0] * mu_e, C_corr[1]])
A_fit = un.nominal_values(A_corr)
s_A_fit = un.std_devs(A_corr)
n_d = 3 # degrees of freedom: # fit parameter + 1
chi2 = np.sum(((A_func(t_c_tr, *C_fit) - A) / s_A) ** 2 )

def plot_life_time():
    """
    plots exponential fit for life time of electrons
    """
    fig1, ax1 = plt.subplots(1, 1)
    if not save_fig:
        fig1.suptitle("Haynes and Shockley, Fit: life time of $e^-$")
    ax1.errorbar(t_c_tr, A, xerr=s_t_c, yerr=s_A, fmt='.', alpha=0.8) # errors of t are not changed!
    next(ax1._get_lines.color_cycle)
    ax1.plot(t_grid_tr, A_func_real(t_grid_tr, *A_fit), '-', alpha=0.8)
    textstr = 'Fit parameters for life time of $e^-$, $\\tau_n$: \n\
            \\begin{eqnarray*}A(t) &=& \\frac{C}{\mu_n E} \exp(- t / \\tau_n) \\\\ \
            C       &=& \,\, (%.0f \pm %.0f) \, \mathrm{mV \cdot mm} \\\\  \
            \\tau_n &=& (%.1f \pm %.1f) \, \mathrm{\mu s} \\\\ \
            \chi^2 / n_d &=& %.1f\
            \end{eqnarray*}'%(A_fit[0], s_A_fit[0], A_fit[1], s_A_fit[1], chi2 / n_d)
    ax1.text(0.5, 0.95, textstr, transform=ax1.transAxes, va='top', bbox=props)
    ax1.set_xlim(t_grid_tr[0], t_grid_tr[-1])
    ax1.set_xlabel(t_c_label)
    ax1.set_ylabel("$\mathrm{A \, / \, (mV \cdot \mu s)} $")
    if show_fig:
        fig1.show()
    if save_fig:
        file_name = "haynes_shockley_tau_d"
        fig1.savefig(fig_dir + file_name + ".pdf")
        fig1.savefig(fig_dir + file_name + ".png")
    return 0
if do_plot_life_time:
    plot_life_time()

# fit diffusion constant var(t) = sigma_x**2(t) = D * 2 * t
# units: ms -> mm
def x(t, mu_e):
    return(mu_e * t)

def var_func(t, D_n):
    return 2 * D_n * t
# calculate the variance in x dir from the std dev in t
sigma_x_corr = x(sigma_t_corr, mu_e)
sigma_x = un.nominal_values(sigma_x_corr)
var_corr = sigma_x_corr**2
var = un.nominal_values(var_corr)
s_var = un.std_devs(var_corr)
var_fit, cov_var_fit = curve_fit(var_func, t_c_tr, var, p0=None, sigma=s_var)
s_var_fit = np.array([np.sqrt(cov_var_fit[i]) for i in range(len(cov_var_fit))])
n_d = 2 # degrees of freedom: # fit parameter + 1
chi2 = np.sum(((var_func(t_c_tr, *var_fit) - var) / s_var) ** 2 )

def plot_diffusion():
    """
    plots linear fit for diffusion constant 
    """
    fig1, ax1 = plt.subplots(1, 1)
    if not save_fig:
        fig1.suptitle("Haynes and Shockley, Fit: diffusion constant of $e^-$")
    ax1.errorbar(t_c_tr, var, xerr=s_t_c, yerr=s_var, fmt='.', alpha=0.8) # errors of t are not changed!
    next(ax1._get_lines.color_cycle)
    ax1.plot(t_grid_tr, var_func(t_grid_tr, *var_fit), '-', alpha=0.8)
    textstr = 'Fit parameters for $e^-$ diffusion const., $D_n$: \n\
            \\begin{eqnarray*}\sigma_x^2 &=& 2 D_n t\\\\ \
            D_n     &=& (%.3f \pm %.3f) \, \mathrm{mm^2 \, / \, \mu s} \\\\  \
            \chi^2 / n_d &=& %.1f\
            \end{eqnarray*}'%(var_fit[0], s_var_fit[0], chi2  / n_d)
    ax1.text(0.02, 0.95, textstr, transform=ax1.transAxes, va='top', bbox=props)
    ax1.set_xlim(t_grid_tr[0], t_grid_tr[-1])
    ax1.set_xlabel(t_c_label)
    ax1.set_ylabel("$\sigma_x^2\, / \,\mathrm{mm}^2$")
    if show_fig:
        fig1.show()
    if save_fig:
        file_name = "haynes_shockley_D_d"
        fig1.savefig(fig_dir + file_name + ".pdf")
        fig1.savefig(fig_dir + file_name + ".png")
    return 0
if do_plot_diffusion:
    plot_diffusion()

# mobility mu_n
l = uc.ufloat(30.0, 0.5)    #mm, length of piece of metal
U = uc.ufloat(49.6, 0.8)    # V, applied voltage
E = U / l   # resulting electric field
mu_n = mu_e / E
