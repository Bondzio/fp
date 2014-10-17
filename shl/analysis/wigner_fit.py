import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from scipy.interpolate import UnivariateSpline,InterpolatedUnivariateSpline,interp1d
from smooth import savitzky_golay
import uncertainties as uc
from uncertainties import umath
import uncertainties.unumpy as un

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern Roman']
rcParams['text.usetex'] = True
rcParams['figure.autolayout'] = True


fig_dir = "./figures/"

def make_fig(fig, show=True,save=False, name="foo"):
    if show == True:
        fig.show()
    if save == True:
        fig.savefig(fig_dir + name + ".pdf")


a = uc.ufloat(0.614120164611590,0.028481101140689)
b = uc.ufloat(2.567505577230589,4.103491871131302)

data = np.load("./data/measure2_1d.npy")
data = data[17:28]
error = np.sqrt(data)
P = data / np.sum(data) 
error /= np.sum(data)
t = a.n * np.arange(17,28,1) + b.n

fig = plt.figure()
ax  = plt.subplot(111)

plt.errorbar(t,P, yerr= error, fmt="x")
plt.ylabel("Probability P")
plt.xlabel("Energy in keV")
plt.xlim(13,19)
plt.show()

def wigner(Mf, gamma, Mi):
    return gamma /(2*np.pi* (gamma**2/4 + (Mf - Mi)**2))
# p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
p0 = [1., 1.]

p, cov = curve_fit(wigner,t , P, p0=p0)
p_uc = uc.correlated_values(p, cov)

t_fit = np.linspace(13,19,1000)

data_fit = wigner(t_fit, *p) 
error_on_fit = un.std_devs(wigner(t_fit, *p_uc))
data_fit_min = data_fit - error_on_fit
data_fit_max = data_fit + error_on_fit

plt.plot(t_fit, data_fit)

plt.fill_between(t_fit, data_fit_min , data_fit_max, facecolor="r", color="b", alpha=0.3 )



make_fig(fig,1,0,name="plot_wigner")
