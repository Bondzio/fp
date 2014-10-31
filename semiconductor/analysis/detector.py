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

def err_round_s(coeffs, s_coeffs):
    """
    rounds the coefficients to the last significant digit
    in: coefficients and according ERRORS
    out: rounded coefficients and their errors
    """
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
    s_fit1 = un.std_devs(fit_corr)
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
    s_fit2 = un.std_devs(fit_corr)
    fit_both2 = fit_both[:]
    chi2_test2 = chi2_test
    
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
            file_name = "detector" + suff
            fig1.savefig(fig_dir + file_name + ".pdf")
            fig1.savefig(fig_dir + file_name + ".png")
        return 0
    plot_two_gauss()
    return fit1, s_fit1, fit2, s_fit2
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
    s_fit = un.std_devs(fit_corr)
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
            file_name = "detector" + suff
            fig1.savefig(fig_dir + file_name + ".pdf")
            fig1.savefig(fig_dir + file_name + ".png")
        return 0
    plot_two_gauss()
    return fit, s_fit
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
    s_fit1 = un.std_devs(fit_corr)
    fit_both1 = fit_both[:]
    chi2_test1 = chi2_test

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
    s_fit2 = un.std_devs(fit_corr)
    fit_both2 = fit_both[:]
    chi2_test2 = chi2_test

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
            file_name = "detector" + suff
            fig1.savefig(fig_dir + file_name + ".pdf")
            fig1.savefig(fig_dir + file_name + ".png")
        return 0
    plot_two_gauss()
    return fit1, s_fit1, fit2, s_fit2
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
    s_fit = un.std_devs(fit_corr)
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
            file_name = "detector" + suff
            fig1.savefig(fig_dir + file_name + ".pdf")
            fig1.savefig(fig_dir + file_name + ".png")
        return 0
    plot_two_gauss()
    return fit, s_fit

# obtain amplitude, center and std dev for all probes
As = [0]*6
mus = [0]*6
sigmas = [0]*6
s_As = [0]*6
s_mus = [0]*6
s_sigmas = [0]*6
[As[0], mus[0], sigmas[0]], \
        [s_As[0], s_mus[0], s_sigmas[0]], \
        [As[1], mus[1], sigmas[1]], \
        [s_As[1], s_mus[1], s_sigmas[1]] = CdTe_Co()
[As[2], mus[2], sigmas[2]], \
        [s_As[2], s_mus[2], s_sigmas[2]] = CdTe_Am()
[As[3], mus[3], sigmas[3]], \
        [s_As[3], s_mus[3], s_sigmas[3]], \
        [As[4], mus[4], sigmas[4]], \
        [s_As[4], s_mus[4], s_sigmas[4]] = Si_Co()
[As[5], mus[5], sigmas[5]], \
        [s_As[5], s_mus[5], s_sigmas[5]] = Si_Am()

# fit parameters
As = np.array(As)
s_As = np.array(s_As)
As_corr = un.uarray(As, s_As)
mus = np.array(mus)
s_mus = np.array(s_mus)
sigmas = np.array(sigmas)
s_sigmas = np.array(s_sigmas)
sigmas_corr = un.uarray(sigmas, s_sigmas)

peaks = ["^{57}\mathrm{Co}_1", "^{57}\mathrm{Co}_2", "^{241}\mathrm{Am}"]
Es = np.array([122.06, 136.47, 59.5]) # peaks

# Calibration
# E(mus)
def calibration(detector_name, mu, s_mu):
    """
    does a linear fit fit three points !!!
    """
    def lin_func(x, a, b):
        return(a*x + b)
    fit_, cov_fit_ = curve_fit(lin_func, Es, mu, p0=None, sigma=s_mu)
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
    #plot1, = ax1.plot(mu, Es, '.', alpha=0.9)   
    mu_grid = np.linspace(0, 700, 100)
    plot_fit, = ax1.plot(mu_grid, lin_func(mu_grid, *fit), alpha=0.5)
    ax1.errorbar(mu, Es, xerr=s_mu, fmt='.', alpha=0.99, c=plot_fit.get_color()) # errors of t are not changed!
    ax1.set_xlabel("channel")
    ax1.set_ylabel("Energy / keV")
    if detector_name == 'Si':
        textstr = 'Results of linear fit:\n\
                \\begin{eqnarray*}\
                E(\mu)&=& a \mu + E_0 \\\\ \
                a     &=& (%.4f \pm %.4f)\, \mathrm{keV / ch} \\\\ \
                E_0   &=& (%.1f \pm %.1f)\, \mathrm{keV} \\\\ \
                \end{eqnarray*}'%tuple(fit_both)
    elif detector_name == 'CdTe':
        textstr = 'Results of linear fit:\n\
                \\begin{eqnarray*}\
                E(\mu)&=& a \mu + E_0 \\\\ \
                a     &=& (%.3f \pm %.3f)\, \mathrm{keV / ch} \\\\ \
                E_0   &=& (%.1f \pm %.1f)\, \mathrm{keV} \\\\ \
                \end{eqnarray*}'%tuple(fit_both)
    ax1.text(0.1, 0.95, textstr, transform=ax1.transAxes, va='top', bbox=props)
    if show_fig:
        fig1.show()
    if save_fig:
        file_name = "detector_calibration_" + detector_name
        fig1.savefig(fig_dir + file_name + ".pdf")
        fig1.savefig(fig_dir + file_name + ".png")
    return fit_corr[0]

linear_factor = [0] * 2
detector_name = "CdTe"
mus_CdTe = mus[:3]
s_mus_CdTe = s_mus[:3]
linear_factor[0] = calibration(detector_name, mus_CdTe, s_mus_CdTe)
detector_name = "Si"
mus_Si = mus[3:]
s_mus_Si = s_mus[3:]
linear_factor[1] = calibration(detector_name, mus_Si, s_mus_Si)

def peaks_to_tex():
    """
    Writes peaks to table (.tex file)
    """
    f = open("detector_peaks.tex", "w+")
    f.write("\t\\begin{tabular}{|p{3cm}|p{3cm}|p{3cm}|p{3cm}|}\n")
    f.write("\t\t\hline\n")
    f.write("\t\t\\rowcolor{tabcolor}\n")
    f.write("\t\tPeak   & $E_\mathrm{peak}$ / keV & $\mathrm{\mu_{CdTe}}$ / Channel & $\mathrm{\mu_{Si}}$ / Channel\\\\ \n")
    f.write("\t\t\hline\n")
    f.write('\t\t$%s$ & $%.2f$ & $%.1f \pm %.1f$ & $%.1f \pm %.1f$ \\\\ \n'
            %(peaks[0], Es[0], mus_CdTe[0], s_mus_CdTe[0], mus_Si[0], s_mus_Si[0]))
    f.write('\t\t$%s$ & $%.2f$ & $%.1f \pm %.1f$ & $%.0f \pm %.0f$ \\\\ \n'
            %(peaks[1], Es[1], mus_CdTe[1], s_mus_CdTe[1], mus_Si[1], s_mus_Si[1]))
    f.write('\t\t$%s$ & $%.1f$ & $%.1f \pm %.1f$ & $%.2f \pm %.2f$ \\\\ \n'
            %(peaks[2], Es[2], mus_CdTe[2], s_mus_CdTe[2], mus_Si[2], s_mus_Si[2]))
    f.write("\t\t\hline\n")
    f.write("\t\end{tabular}\n")
    f.close()
    return 0
peaks_to_tex()

# Calculation of absorption ratio
a_CdTe = np.array([100]*3)  # active areas of CdTe detector
a_Si = np.array([23]*3)     # active areas of Si detector
P_corr = (As_corr[3:] / a_Si) / (As_corr[:3] / a_CdTe) # absorption ratio
P = un.nominal_values(P_corr)
s_P = un.std_devs(P_corr)
P, s_P = err_round_s(P, s_P)
As, s_As = err_round_s(As, s_As)
As_CdTe = As[:3]
s_As_CdTe = s_As[:3]
As_Si = As[3:]
s_As_Si = s_As[3:]

def ratio_to_tex():
    """
    Writes absorption ratio to table (.tex file)
    """
    f = open("detector_ratio.tex", "w+")
    f.write("\t\\begin{tabular}{|p{3cm}|p{3cm}|p{3cm}|p{3cm}|}\n")
    f.write("\t\t\hline\n")
    f.write("\t\t\\rowcolor{tabcolor}\n")
    f.write("\t\tPeak   & $A_\mathrm{CdTe}$ & $A_\mathrm{Si}$ & $P$\\\\ \n")
    f.write("\t\t\hline\n")
    f.write('\t\t$%s$ & $%.0f \pm %.0f$ & $%.0f \pm %.0f$ & $%.3f \pm %.3f$\\\\ \n'
            %(peaks[0], As_CdTe[0], s_As_CdTe[0], As_Si[0], s_As_Si[0], P[0], s_P[0]))
    f.write('\t\t$%s$ & $%.0f \pm %.0f$ & $%.0f \pm %.0f$ & $%.2f \pm %.2f$\\\\ \n'
            %(peaks[1], As_CdTe[1], s_As_CdTe[1], As_Si[1], s_As_Si[1], P[1], s_P[1]))
    f.write('\t\t$%s$ & $%.0f \pm %.0f$ & $%.0f \pm %.0f$ & $%.3f \pm %.3f$\\\\ \n'
            %(peaks[2], As_CdTe[2], s_As_CdTe[2], As_Si[2], s_As_Si[2], P[2], s_P[2]))
    f.write("\t\t\hline\n")
    f.write("\t\end{tabular}\n")
    f.close()
    return 0
ratio_to_tex()

# Relative energy resolution
# RER(E) = FWHM(E) / E = 2.35 \sigma(E) / E
# \sigma_channel -> sigma_E:
sigmas_E_corr = sigmas_corr * np.array([linear_factor[0]]*3 + [linear_factor[1]]*3)
sigmas_E = un.nominal_values(sigmas_E_corr)
s_sigmas_E = un.std_devs(sigmas_E_corr)
RER_corr = 2.35 * sigmas_E_corr / np.concatenate((Es, Es), axis=0)
RER = un.nominal_values(RER_corr)
s_RER = un.std_devs(RER_corr)
def RER_to_tex():
    """
    Writes RER for both detectors to table (.tex file)
    """
    f = open("detector_RER.tex", "w+")
    f.write("\t\\begin{tabular}{|p{2cm}|p{2.5cm}|p{3cm}|p{3cm}|p{3cm}|}\n")
    f.write("\t\t\hline\n")
    f.write("\t\t\\rowcolor{tabcolor}\n")
    f.write("\t\tPeak   & $E_\mathrm{peak}$ / keV & $\sigma_\mathrm{CdTe}$ / Channel & \
            $\sigma_{E, \mathrm{CdTe}}$ /keV & $\mathrm{RER_{CdTe}}(E)$ \\\\ \n")
    f.write("\t\t\hline\n")
    f.write('\t\t$%s$ & $%.2f$ & $%.1f \pm %.1f$ & $%.2f \pm %.2f$ & $%.4f \pm %.4f$\\\\ \n'
            %(peaks[0], Es[0], sigmas[0], s_sigmas[0], sigmas_E[0], s_sigmas_E[0], RER[0], s_RER[0]))
    f.write('\t\t$%s$ & $%.2f$ & $%.1f \pm %.1f$ & $%.1f \pm %.1f$ & $%.3f \pm %.3f$\\\\ \n'
            %(peaks[1], Es[1], sigmas[1], s_sigmas[1], sigmas_E[1], s_sigmas_E[1], RER[1], s_RER[1]))
    f.write('\t\t$%s$ & $%.1f$ & $%.1f \pm %.1f$ & $%.2f \pm %.2f$ & $%.3f \pm %.3f$\\\\ \n'
            %(peaks[2], Es[2], sigmas[2], s_sigmas[2], sigmas_E[2], s_sigmas_E[2], RER[2], s_RER[2]))
    f.write("\t\t\hline &&&&\\\\ \n")
    f.write("\t\t\hline\n")
    f.write("\t\t\\rowcolor{tabcolor}\n")
    f.write("\t\tPeak   & $E_\mathrm{peak}$ / keV & $\sigma_\mathrm{Si}$ / Channel & \
            $\sigma_{E, \mathrm{Si}}$ /keV & $\mathrm{RER_{Si}}(E)$ \\\\ \n")
    f.write("\t\t\hline\n")
    f.write('\t\t$%s$ & $%.2f$ & $%.1f \pm %.1f$ & $%.2f \pm %.2f$ & $%.4f \pm %.4f$\\\\ \n'
            %(peaks[0], Es[0], sigmas[0], s_sigmas[0], sigmas_E[0], s_sigmas_E[0], RER[0], s_RER[0]))
    f.write('\t\t$%s$ & $%.2f$ & $%.1f \pm %.1f$ & $%.1f \pm %.1f$ & $%.3f \pm %.3f$\\\\ \n'
            %(peaks[1], Es[1], sigmas[1], s_sigmas[1], sigmas_E[1], s_sigmas_E[1], RER[1], s_RER[1]))
    f.write('\t\t$%s$ & $%.1f$ & $%.1f \pm %.1f$ & $%.2f \pm %.2f$ & $%.3f \pm %.3f$\\\\ \n'
            %(peaks[2], Es[2], sigmas[2], s_sigmas[2], sigmas_E[2], s_sigmas_E[2], RER[2], s_RER[2]))
    f.write("\t\t\hline\n")
    f.write("\t\end{tabular}\n")
    f.close()
    return 0
RER_to_tex()
