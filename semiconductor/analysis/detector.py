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

def err_round(coeffs, cov):
    """
    rounds the coefficients to the last significant digit
    in: coefficients and according covariance matrix
    out: rounded coefficients and their errors
    """
    s_coeffs = np.sqrt(np.diag(cov))                                        # errors on values
    digits = -np.int_(np.floor(np.log10(s_coeffs)))                         # calc no of digits
    digits += (s_coeffs * 10**np.float64(digits)) < 1.5                     # convention
    coeffs_rounded  = np.array([round(c, digit)for c, digit in zip(coeffs, digits)])          # round values
    s_coeffs_rounded = np.array([round(s_c, digit)for s_c, digit in zip(s_coeffs, digits)])    # round errors
    return coeffs_rounded, s_coeffs_rounded

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
fig_dir = "../figures/"
npy_dir = "./data_npy/"
props = dict(boxstyle='round', facecolor='white', edgecolor = 'white', alpha=0.7)

def two_gauss(x, A1, A2, mu1, mu2, sigma1, sigma2):
    return (A1 / (abs(sigma1) * np.sqrt(2 * np.pi)) * np.exp(1)**(-((x - mu1) / abs(sigma1))**2 / 2 ) +
            A2 / (abs(sigma2) * np.sqrt(2 * np.pi)) * np.exp(1)**(-((x - mu2) / abs(sigma2))**2 / 2 ) )
def gauss(x, A, mu, sigma):
    return (A / (abs(sigma) * np.sqrt(2 * np.pi)) * np.exp(1)**(-((x - mu) / sigma)**2 / 2 ) )
def gauss_c(x, A, mu, sigma, c):
    return (A / (abs(sigma) * np.sqrt(2 * np.pi)) * np.exp(1)**(-((x - mu) / sigma)**2 / 2 ) + c)

func_str = "c(x) &=& \sum_{i = 1}^{2} \\frac{A_i}{\sqrt{2\pi \sigma_i}} \
        \exp\left(-\\frac{1}{2}\left(\\frac{x - \mu_i}{\sigma_i}\\right)^2\\right)"

fit_params = []
As = [0]*6
mus = [0]*6
sigmas = [0]*6
s_As = [0]*6
s_mus = [0]*6
s_sigmas = [0]*6
def CdTe_Co():
    detector_name = "CdTe"
    sample_name = "Co"
    suff = "_" + sample_name + "_" + detector_name 
    npy_file = npy_dir + "spectra" + suff + ".npy"
    n = np.load(npy_file)[:1000]
    s_n = np.sqrt(n)
    x = np.arange(len(n))
    
    # right fit
    p0 = [4000, 580, 20]
    red = (x > 570) * (x < 600)
    n_red = n[red]
    x_red = x[red]
    s_n_red = s_n[red]
    fit, cov_fit = curve_fit(gauss, x_red, n_red, p0=p0, sigma=s_n_red)
    fit  = np.abs(fit)
    fit_corr = uc.correlated_values(fit, cov_fit)
    # chi square
    n_d = 4
    chi2 = np.sum(((gauss(x_red, *fit) - n_red) / s_n_red) ** 2 )
    chi2_test = chi2/n_d
    fit_r, s_fit_r = err_round(fit, cov_fit)
    fit_both = np.array([fit_r, s_fit_r])
    fit_both = np.reshape(fit_both.T, np.size(fit_both))
    x_red1 = x_red[:]
    fit1 = fit[:]
    fit_both1 = fit_both[:]
    chi2_test1 = chi2_test
    As[0], mus[0], sigmas[0] = fit
    s_As[0], s_mus[0], s_sigmas[0] = un.std_devs(fit_corr)

    # left fit
    p0 = [400, 648, 10]
    red = (x > 640) * (x < 663)
    n_red = n[red]
    x_red = x[red]
    s_n_red = s_n[red]
    fit, cov_fit = curve_fit(gauss, x_red, n_red, p0=p0, sigma=s_n_red)
    fit  = np.abs(fit)
    fit_corr = uc.correlated_values(fit, cov_fit)
    # chi square
    n_d = 4
    chi2 = np.sum(((gauss(x_red, *fit) - n_red) / s_n_red) ** 2 )
    chi2_test = chi2/n_d
    fit_r, s_fit_r = err_round(fit, cov_fit)
    fit_both = np.array([fit_r, s_fit_r])
    fit_both = np.reshape(fit_both.T, np.size(fit_both))
    x_red2 = x_red[:]
    fit2 = fit[:]
    fit_both2 = fit_both[:]
    chi2_test2 = chi2_test
    As[1], mus[1], sigmas[1] = fit
    s_As[1], s_mus[1], s_sigmas[1] = un.std_devs(fit_corr)
    
    all = np.concatenate((fit_both1, [chi2_test1], fit_both2, [chi2_test2]), axis=0)

    def plot_two_gauss():
        fig1, ax1 = plt.subplots(1, 1)
        if not save_fig:
            fig1.suptitle("Detector: " + detector_name + "; sample: " + sample_name)
        plot1, = ax1.plot(x, n, '.', alpha=0.3)   # histo plot
        ax1.errorbar(x, n, yerr=s_n, fmt=',', alpha=0.99, c=plot1.get_color(), errorevery=10) # errors of t are not changed!
        gauss_fit, = ax1.plot(x_red1, gauss(x_red1, *fit1))
        ax1.plot(x_red2, gauss(x_red2, *fit2), c=gauss_fit.get_color())
        ax1.set_xlim(500, 750)
        ax1.set_ylim(0, 270)
        ax1.set_xlabel("channel")
        ax1.set_ylabel("counts")
        textstr = 'Results of fit:\n\
                \\begin{eqnarray*}\
                A_1     &=& (%.0f \pm %.0f) \\\\ \
                \mu_1   &=& (%.1f \pm %.1f) \\\\ \
                \sigma_1&=& (%.1f \pm %.1f) \\\\ \
                \chi_1^2 / n_d &=& %.1f     \\\\ \
                A_2     &=& (%.0f \pm %.0f) \\\\ \
                \mu_2   &=& (%.1f \pm %.1f) \\\\ \
                \sigma_2&=& (%.1f \pm %.1f) \\\\ \
                \chi_2^2 / n_d &=& %.1f\
                \end{eqnarray*}'%tuple(all)
        ax1.text(0.65, 0.95, textstr, transform=ax1.transAxes, va='top', bbox=props)
        if show_fig:
            fig1.show()
        if save_fig:
            file_name = "detector_" + suff
            fig1.savefig(fig_dir + file_name + ".pdf")
            fig1.savefig(fig_dir + file_name + ".png")
        return 0
    plot_two_gauss()
    return 0 
def CdTe_Am():
    detector_name = "CdTe"
    sample_name = "Am"
    suff = "_" + sample_name + "_" + detector_name 
    npy_file = npy_dir + "spectra" + suff + ".npy"
    n = np.load(npy_file)[:500]
    s_n = np.sqrt(n)
    x = np.arange(len(n))
    
    # only fit
    p0 = [250000, 250, 25]
    red = (x > 235) * (x < 270)
    n_red = n[red]
    x_red = x[red]
    s_n_red = s_n[red]
    fit, cov_fit = curve_fit(gauss, x_red, n_red, p0=p0, sigma=s_n_red)
    fit  = np.abs(fit)
    fit_corr = uc.correlated_values(fit, cov_fit)
    # chi square
    n_d = 4
    chi2 = np.sum(((gauss(x_red, *fit) - n_red) / s_n_red) ** 2 )
    chi2_test = chi2/n_d
    fit_r, s_fit_r = err_round(fit, cov_fit)
    fit_both = np.array([fit_r, s_fit_r])
    fit_both = np.reshape(fit_both.T, np.size(fit_both))
    As[2], mus[2], sigmas[2] = fit
    s_As[2], s_mus[2], s_sigmas[2] = un.std_devs(fit_corr)

    all = np.concatenate((fit_both, [chi2_test]), axis=0)
    def plot_two_gauss():
        fig1, ax1 = plt.subplots(1, 1)
        if not save_fig:
            fig1.suptitle("Detector: " + detector_name + "; sample: " + sample_name)
        plot1, = ax1.plot(x, n, '.', alpha=0.3)   # histo plot
        ax1.errorbar(x, n, yerr=s_n, fmt=',', alpha=0.99, c=plot1.get_color(), errorevery=10) # errors of t are not changed!
        ax1.plot(x_red, gauss(x_red, *fit))
        ax1.set_xlabel("channel")
        ax1.set_ylabel("counts")
        textstr = 'Results of fit:\n\
                \\begin{eqnarray*}\
                A     &=& (%.0f \pm %.0f) \\\\ \
                \mu   &=& (%.1f \pm %.1f) \\\\ \
                \sigma&=& (%.1f \pm %.1f) \\\\ \
                \chi^2 / n_d &=& %.1f\
                \end{eqnarray*}'%tuple(all)
        ax1.text(0.65, 0.95, textstr, transform=ax1.transAxes, va='top', bbox=props)
        if show_fig:
            fig1.show()
        if save_fig:
            file_name = "detector_" + suff
            fig1.savefig(fig_dir + file_name + ".pdf")
            fig1.savefig(fig_dir + file_name + ".png")
        return 0
    plot_two_gauss()
def Si_Co():
    detector_name = "Si"
    sample_name = "Co"
    suff = "_" + sample_name + "_" + detector_name 
    npy_file = npy_dir + "spectra" + suff + ".npy"
    n = np.load(npy_file)[:1000]
    x = np.arange(len(n))
    # rebin
    l = int(len(n) / 2)
    n = np.array([n[2*i] + n[2*i+1] for i in range(l)]) / 2
    x = np.array([x[2*i] for i in range(l)])
    #l = int(len(n) / 2)
    #n = np.array([n[2*i] + n[2*i+1] for i in range(l)]) / 2
    #x = np.array([x[2*i] for i in range(l)])
    s_n = np.sqrt(n)
    
    # left fit
    p0 = [400, 620, 15]
    red = (x > 580) * (x < 660)
    n_red = n[red]
    x_red = x[red]
    s_n_red = s_n[red]
    s_n_red += (s_n_red == 0)
    fit, cov_fit = curve_fit(gauss, x_red, n_red, p0=p0, sigma=s_n_red)
    fit  = np.abs(fit)
    fit_corr = uc.correlated_values(fit, cov_fit)
    # chi square
    n_d = 4
    chi2 = np.sum(((gauss(x_red, *fit) - n_red) / s_n_red) ** 2 )
    chi2_test = chi2/n_d
    fit_r, s_fit_r = err_round(fit, cov_fit)
    fit_both = np.array([fit_r, s_fit_r])
    fit_both = np.reshape(fit_both.T, np.size(fit_both))
    x_red1 = x_red[:]
    fit1 = fit[:]
    fit_both1 = fit_both[:]
    chi2_test1 = chi2_test
    As[3], mus[3], sigmas[3] = fit
    s_As[3], s_mus[3], s_sigmas[3] = un.std_devs(fit_corr)

    # right fit
    p0 = [27, 687, 9]
    red = (x > 665) * (x < 710)
    n_red = n[red]
    x_red = x[red]
    s_n_red = s_n[red]
    s_n_red += (s_n_red == 0)
    fit, cov_fit = curve_fit(gauss, x_red, n_red, p0=p0, sigma=s_n_red)
    fit  = np.abs(fit)
    fit_corr = uc.correlated_values(fit, cov_fit)
    # chi square
    n_d = 4
    chi2 = np.sum(((gauss(x_red, *fit) - n_red) / s_n_red) ** 2 )
    chi2_test = chi2/n_d
    fit_r, s_fit_r = err_round(fit, cov_fit)
    fit_both = np.array([fit_r, s_fit_r])
    fit_both = np.reshape(fit_both.T, np.size(fit_both))
    x_red2 = x_red[:]
    fit2 = fit[:]
    fit_both2 = fit_both[:]
    chi2_test2 = chi2_test
    As[4], mus[4], sigmas[4] = fit
    s_As[4], s_mus[4], s_sigmas[4] = un.std_devs(fit_corr)

    all = np.concatenate((fit_both1, [chi2_test1], fit_both2, [chi2_test2]), axis=0)
    def plot_two_gauss():
        fig1, ax1 = plt.subplots(1, 1)
        if not save_fig:
            fig1.suptitle("Detector: " + detector_name + "; sample: " + sample_name)
        plot1, = ax1.plot(x, n, '-', alpha=0.3)   # histo plot
        ax1.errorbar(x, n, yerr=s_n, fmt=',', alpha=0.99, c=plot1.get_color(), errorevery=10) # errors of t are not changed!
        ax1.plot(x_red1, gauss(x_red1, *fit1))
        ax1.plot(x_red2, gauss(x_red2, *fit2))
        ax1.set_xlim(500, 800)
        ax1.set_ylim(0, 20)
        ax1.set_xlabel("channel")
        ax1.set_ylabel("counts")
        textstr = 'Results of fit:\n\
                \\begin{eqnarray*}\
                A_1     &=& (%.0f \pm %.0f) \\\\ \
                \mu_1   &=& (%.1f \pm %.1f) \\\\ \
                \sigma_1&=& (%.1f \pm %.1f) \\\\ \
                \chi_1^2 / n_d &=& %.1f     \\\\ \
                A_2     &=& (%.0f \pm %.0f) \\\\ \
                \mu_2   &=& (%.0f \pm %.0f) \\\\ \
                \sigma_2&=& (%.0f \pm %.0f) \\\\ \
                (\chi_1^2 / n_d &=& %.1f)     \
                \end{eqnarray*}'%tuple(all)
        ax1.text(0.65, 0.95, textstr, transform=ax1.transAxes, va='top', bbox=props)
        if show_fig:
            fig1.show()
        if save_fig:
            file_name = "detector_" + suff
            fig1.savefig(fig_dir + file_name + ".pdf")
            fig1.savefig(fig_dir + file_name + ".png")
        return 0
    plot_two_gauss()
def Si_Am():
    detector_name = "Si"
    sample_name = "Am"
    suff = "_" + sample_name + "_" + detector_name 
    npy_file = npy_dir + "spectra" + suff + ".npy"
    n = np.load(npy_file)[:400]
    s_n = np.sqrt(n)
    x = np.arange(len(n))
    # left fit
    p0 = [600*2.5*15, 300, 15]
    red = (x > 280) * (x < 340)
    n_red = n[red]
    x_red = x[red]
    s_n_red = s_n[red]
    fit, cov_fit = curve_fit(gauss, x_red, n_red, p0=p0, sigma=s_n_red)
    fit  = np.abs(fit)
    fit_corr = uc.correlated_values(fit, cov_fit)
    # chi square
    n_d = 4
    chi2 = np.sum(((gauss(x_red, *fit) - n_red) / s_n_red) ** 2 )
    chi2_test = chi2/n_d
    fit_r, s_fit_r = err_round(fit, cov_fit)
    fit_both = np.array([fit_r, s_fit_r])
    fit_both = np.reshape(fit_both.T, np.size(fit_both))
    As[5], mus[5], sigmas[5] = fit
    s_As[5], s_mus[5], s_sigmas[5] = un.std_devs(fit_corr)

    all = np.concatenate((fit_both, [chi2_test]), axis=0)
    def plot_two_gauss():
        fig1, ax1 = plt.subplots(1, 1)
        if not save_fig:
            fig1.suptitle("Detector: " + detector_name + "; sample: " + sample_name)
        plot1, = ax1.plot(x, n, '.', alpha=0.3)   # histo plot
        ax1.errorbar(x, n, yerr=s_n, fmt=',', alpha=0.99, c=plot1.get_color(), errorevery=10) # errors of t are not changed!
        ax1.plot(x_red, gauss(x_red, *fit))
        ax1.set_xlabel("channel")
        ax1.set_ylabel("counts")
        textstr = 'Results of fit:\n\
                \\begin{eqnarray*}\
                A_1     &=& (%.0f \pm %.0f) \\\\ \
                \mu_1   &=& (%.2f \pm %.2f) \\\\ \
                \sigma_1&=& (%.2f \pm %.2f) \\\\ \
                \chi^2 / n_d &=& %.1f\
                \end{eqnarray*}'%tuple(all)
        ax1.text(0.65, 0.95, textstr, transform=ax1.transAxes, va='top', bbox=props)
        if show_fig:
            fig1.show()
        if save_fig:
            file_name = "detector_" + suff
            fig1.savefig(fig_dir + file_name + ".pdf")
            fig1.savefig(fig_dir + file_name + ".png")
        return 0
    plot_two_gauss()
CdTe_Co()
CdTe_Am()
Si_Co()
Si_Am()

# fit parameters
As = np.array(As)
mus = np.array(mus)
sigmas = np.array(sigmas)
s_As = np.array(s_As)
s_mus = np.array(s_mus)
s_sigmas = np.array(s_sigmas)

E = np.array([122.06, 136.47, 59.5]) # peaks

# Calibration
# E(mus)
def calibration(detector_name, mu, s_mu):
    def lin_func(x, a, b):
        return(a*x + b)
    fit_, cov_fit_ = curve_fit(lin_func, E, mu, p0=None, sigma=s_mu)
    fit_corr_ = uc.correlated_values(fit_, cov_fit_)
    fit_corr  = np.array([1 / fit_corr_[0], - fit_corr_[1] / fit_corr_[0]])
    fit = un.nominal_values(fit_corr) 
    cov_fit = uc.covariance_matrix(fit_corr)
    s_fit = un.std_devs(fit_corr) 
    fit_r, s_fit_r = err_round(fit, cov_fit)
    fit_both = np.array([fit_r, s_fit_r])
    fit_both = np.reshape(fit_both.T, np.size(fit_both))
    
    fig1, ax1 = plt.subplots(1, 1)
    if not save_fig:
        fig1.suptitle("Calibration: " + detector_name)
    #plot1, = ax1.plot(mu, E, '.', alpha=0.9)   
    mu_grid = np.linspace(0, 700, 100)
    plot_fit, = ax1.plot(mu_grid, lin_func(mu_grid, *fit), alpha=0.5)
    ax1.errorbar(mu, E, xerr=s_mu, fmt='.', alpha=0.99, c=plot_fit.get_color()) # errors of t are not changed!
    ax1.set_xlabel("channel")
    ax1.set_ylabel("Energy / keV")
    textstr = 'Results of linear fit:\n\
            \\begin{eqnarray*}\
            E(\mu)&=& a \mu + E_0 \\\\ \
            a     &=& (%.4f \pm %.4f)\, \mathrm{keV / ch} \\\\ \
            E_0   &=& (%.1f \pm %.1f)\, \mathrm{keV} \\\\ \
            \end{eqnarray*}'%tuple(fit_both)
    ax1.text(0.1, 0.95, textstr, transform=ax1.transAxes, va='top', bbox=props)
    if show_fig:
        fig1.show()
    if save_fig:
        file_name = "detector_calibration_" + detector_name
        fig1.savefig(fig_dir + file_name + ".pdf")
        fig1.savefig(fig_dir + file_name + ".png")
    return 0
detector_name = "CdTe"
mu = mus[:3]
s_mu = s_mus[:3]
calibration(detector_name, mu, s_mu)
detector_name = "Si"
mu = mus[3:]
s_mu = s_mus[3:]
calibration(detector_name, mu, s_mu)


