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
save_fig = True
do_plot_raw         = True
do_write_to_tex     = True
do_plot_gaussians   = True
do_plot_mobility    = True
do_plot_life_time   = True
do_plot_diffusion   = True
fig_dir = "../figures/"
npy_dir = "./data_npy/"

props = dict(boxstyle='round', facecolor='white', edgecolor = 'white', alpha=0.8)

# file names
original_all = np.array([35, 39, 43, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67])
smooth_all = np.array([36, 40, 44, 46, 48, 50, 52, 54, 58, 60, 62, 64, 66, 68])
U_all = np.array([48.0, 48.4, 44.4, 39.6, 34.4, 27.6, 24.0, 22.8, 14.4, 24.4, 32.8, 41.2, 49.6, 46.4, 42.0])

# choose valid plots.
choose = 0 # 0 -> raw data, 1 -> smooth data
if choose == 0:
    chosen = np.array([1, 2, 3, 4, 10, 11, 12, 13, 14]) -1
    box_pos = np.array([(0.65, 0.4)] + [(0.02, 0.4)] * (len(chosen)-1))
elif choose == 1:
    chosen = np.array([1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14]) -1 # only if we use smooth signals
    box_pos = np.array([(0.65, 0.4)] + [(0.02, 0.4)] * (len(chosen)-1))
original = original_all[chosen]
smooth = smooth_all[chosen]
U = U_all[chosen]
t_grid = np.linspace(360, 380, 100) # grid for plotting the linear fit in \mu s

# sort
order = np.argsort(U)
U = U[order]
original = original[order]
smooth = smooth[order]
box_pos = box_pos[order]

pairs = zip(original, smooth)
s_U = 0.8  # error on d in V

l = uc.ufloat(30.0, 0.5)    #mm, length of piece of metal
d = 4.1                     # distance laser needle
s_d = 0.5
d = uc.ufloat(d, s_d)

ps = np.zeros([len(U), 4])
covs = np.zeros([len(U), 4, 4])
chi2_tests = np.zeros([len(U)])
for k, (i, j) in enumerate(pairs): # 69
    npy_file = npy_dir + "haynes_shockley_%i"%i + ".npy"
    t, signal = np.load(npy_file)
    npy_file = npy_dir + "haynes_shockley_%i"%j + ".npy"
    t_s, signal_s = np.load(npy_file)
    if choose == 1:
        signal = signal_s

    t = t * 10**6                   # s -> \mu s!
    signal = signal * 10 ** 3       # V -> mV
    signal_s = signal_s * 10 ** 3   # V -> mV
    t_label = "$t \,/ \, \mathrm{\mu s}$"
    t_c_label = "$t_c \,/ \, \mathrm{\mu s}$"
    u_label = "$U \,/ \, \mathrm{mV}$"

    # fluctuation
    fluct_side = ["left"] * len(U)
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
                U_\mathrm{acc} &=& (%.1f \pm 0.1)\, \mathrm{V} \\\\ \
                c(t) &=& \\frac{A}{\sqrt{2\pi \sigma_t}} \
                    \exp\left(-\\frac{1}{2}\left(\\frac{t - t_c}{\sigma_t}\\right)^2\\right)\\\\ \
                A           &=& \,\,(%.1f \pm %.1f) \, \mathrm{mm / \mu s} \\\\  \
                t_c         &=& \,\,(%.2f \pm %.2f) \, \mathrm{\mu s} \\\\  \
                \sigma_t    &=& (%.2f \pm %.2f) \, \mathrm{\mu s} \
                \end{eqnarray*}'%(U[k], p[0], p_corr[0].s, p[1], p_corr[1].s, p[2], p_corr[2].s)
        ax1.text(box_pos[k][0], box_pos[k][1], textstr, transform=ax1.transAxes, fontsize=18, va='top', bbox=props)
        ax1.set_xlim(t[0], t[-1])
        ax1.set_xlabel(t_label)
        ax1.set_ylabel(u_label)
        if show_fig:
            fig1.show()
        if save_fig:
            file_name = "haynes_shockley_raw_U_%i"%i
            fig1.savefig(fig_dir + file_name + ".pdf")
            fig1.savefig(fig_dir + file_name + ".png")
        return 0
    if do_plot_raw:
        plot_raw()

def write_to_tex():
    """
    Writes result of gaussian fits to .tex file
    """
    f = open("h_s_fit_parameters_U.tex", "w+")
    f.write("\t\\begin{tabular}{|p{2cm}|p{3cm}|p{3cm}|p{3cm}|p{2cm}|}\n")
    f.write("\t\t\hline\n")
    f.write("\t\t\\rowcolor{tabcolor}\n")
    f.write("\t\t$U_\mathrm{acc} \, / \, \mathrm{V}$        & $A \, / \, \mathrm{\\frac{mm}{\mu s}}$ & \n \
    \t\t\t$t_c \, / \, \mathrm{\mu s}$    & $\sigma_t \, / \, \mathrm{\mu s}$ & \n \
    \t\t\t$\chi^2 / n_d$ \\\\ \hline\n")
    for k in range(len(ps)):
        p = ps[k]
        s_p = np.sqrt(np.diag(covs[k]))
        chi2_test = chi2_tests[k]
        f.write('\t\t$%.1f$ & $%.1f \pm %.1f$ & $%.2f \pm %.2f$ & $%.2f \pm %.2f$ & $%.1f$\\\\ \n'
                %(U[k], p[0], s_p[0], p[1], s_p[1], p[2], s_p[2], chi2_test))
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
    [ax1.plot(t_grid, func(t_grid, *p), '-', alpha=0.8, label='$U_\mathrm{acc} = %.1f \, \mathrm{V}$'%U[k]) for k, p in enumerate(ps)]
    ax1.set_xlim(t_grid[0], t_grid[-1])
    ax1.set_xlabel(t_label)
    ax1.set_ylabel(u_label)
    ax1.legend(loc=1)
    if show_fig:
        fig1.show()
    if save_fig:
        file_name = "haynes_shockley_all_gauss_U"
        fig1.savefig(fig_dir + file_name + ".pdf")
        fig1.savefig(fig_dir + file_name + ".png")
if do_plot_gaussians:
    plot_gaussians()

A, t_c, sigma_t, c = ps.T
s_A, s_t_c, s_sigma_t, s_c = np.array([np.sqrt(np.diag(covs[i])) for i in range(len(ps))]).T
sigma_t_corr = un.uarray(sigma_t, s_sigma_t)
A_corr = un.uarray(A, s_A)

# fit mobility of free electrons d * l / U_acc = mu_n *t 
def mob_func(t, mu_n, t0):
    return mu_n* (t - t0)
y_corr = d * l / U
y = un.nominal_values(y_corr)
s_y = un.std_devs(y_corr)
mob_fit, cov_mob_fit = curve_fit(mob_func, t_c, y, p0=None, sigma=s_y)
s_mob_fit = np.sqrt(np.diag(cov_mob_fit))
mob_fit_corr = uc.correlated_values(mob_fit, cov_mob_fit)
mu_e = mob_fit[0] * U / l
n_d = 3 # degrees of freedom: # fit parameter + 1
chi2 = np.sum(((mob_func(t_c, *mob_fit) - y) / s_y) ** 2 )

def plot_mobility():
    """
    plots linear fit for electron mobility
    """
    fig1, ax1 = plt.subplots(1, 1)
    if not save_fig:
        fig1.suptitle("Haynes and Shockley, Fit: $e^-$ mobility")
    ax1.errorbar(t_c, y, xerr=s_t_c, yerr=s_y,   fmt='.', alpha=0.8)
    next(ax1._get_lines.color_cycle)
    ax1.plot(t_grid, mob_func(t_grid, *mob_fit), '-', alpha=0.8)
    textstr = 'Fit parameters for $e^-$ mobility $\mu_n$: \n\
            \\begin{eqnarray*}\\frac{d \, l}{U_\mathrm{acc}(t)} &=& \mu_n (t - t_0) \\\\ \
            \mu_n     &=& \,\,(%.2f \pm %.2f) \, \mathrm{mm^2 / (V \mu s)} \\\\  \
            t_0    &=&(%.1f \pm %.1f) \, \mathrm{\mu s}\\\\ \
            \chi^2 / n_d &=& %.1f\
            \end{eqnarray*}'%(mob_fit[0], s_mob_fit[0], mob_fit[1], s_mob_fit[1], chi2 / n_d)
    ax1.text(0.55, 0.5, textstr, transform=ax1.transAxes, va='top', bbox=props)
    ax1.set_xlim(t_grid[0], t_grid[-1])
    ax1.set_xlabel(t_c_label)
    ax1.set_ylabel("$\\frac{d \, l}{U} \, / \, \mathrm{\\frac{mm^2}{V}}$")
    if show_fig:
        fig1.show()
    if save_fig:
        file_name = "haynes_shockley_mu_e_U"
        fig1.savefig(fig_dir + file_name + ".pdf")
        fig1.savefig(fig_dir + file_name + ".png")
    return 0
if do_plot_mobility:
    plot_mobility()

# transform t -> t - t_0 (no offset)
t_c_tr = t_c - mob_fit[1]
t_grid_tr = np.linspace(0, 15, 100)

# fit life time of electrons A - C * exp(-t / tau)
def A_func(t, C, tau):
    return C * np.exp(1) ** (- t / tau)
p0 = [20, 4]
y_A_corr = A_corr * mu_e * U / l
y_A = un.nominal_values(y_A_corr)
s_y_A = un.std_devs(y_A_corr)
A_fit, cov_A_fit = curve_fit(A_func, t_c_tr, y_A, p0=p0, sigma=s_y_A)
A_corr = uc.correlated_values(A_fit, cov_A_fit)
s_A_fit = un.std_devs(A_corr)
n_d = 3 # degrees of freedom: # fit parameter + 1
chi2 = np.sum(((A_func(t_c_tr, *A_fit) - y_A) / s_y_A) ** 2 )
def plot_life_time():
    """
    plots exponential fit for life time of electrons
    """
    fig1, ax1 = plt.subplots(1, 1)
    if not save_fig:
        fig1.suptitle("Haynes and Shockley, Fit: life time of $e^-$")
    ax1.errorbar(t_c_tr, y_A, xerr=s_t_c, yerr=s_y_A, fmt='.', alpha=0.8) # errors of t are not changed!
    next(ax1._get_lines.color_cycle)
    ax1.plot(t_grid_tr, A_func(t_grid_tr, *A_fit), '-', alpha=0.8)
    textstr = 'Fit parameters for life time of $e^-$, $\\tau_n$: \n\
            \\begin{eqnarray*}\mu_n A(t_c) \\frac{U_\mathrm{acc}(t_c)}{l} &=& \\ C\' \exp(- t / \\tau_n) \\\\ \
            C\'       &=& \,\, (%i \pm %i) \, \mathrm{mV \cdot mm} \\\\  \
            \\tau_n &=& (%.1f \pm %.1f) \, \mathrm{\mu s} \\\\ \
            \chi^2 / n_d &=& %.1f\
            \end{eqnarray*}'%(A_fit[0], s_A_fit[0], A_fit[1], s_A_fit[1], chi2 / n_d)
    ax1.text(0.02, 0.5, textstr, transform=ax1.transAxes, va='top', bbox=props)
    ax1.set_xlim(t_grid_tr[0], t_grid_tr[-1])
    ax1.set_ylim(0, 12)
    ax1.set_xlabel(t_c_label)
    ax1.set_ylabel("$\mathrm{\mu_n A \\frac{U_\mathrm{acc}}{l} \, / \, (mV \cdot mm)} $")
    if show_fig:
        fig1.show()
    if save_fig:
        file_name = "haynes_shockley_tau_U"
        fig1.savefig(fig_dir + file_name + ".pdf")
        fig1.savefig(fig_dir + file_name + ".png")
    return 0
if do_plot_life_time:
    plot_life_time()

# fit diffusion constant var(t) = sigma_x**2(t) = D * 2 * t
# units: ms -> mm
def x(t, mu_e):
    return(t * mu_e)
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
        file_name = "haynes_shockley_D_U"
        fig1.savefig(fig_dir + file_name + ".pdf")
        fig1.savefig(fig_dir + file_name + ".png")
    return 0
if do_plot_diffusion:
    plot_diffusion()

