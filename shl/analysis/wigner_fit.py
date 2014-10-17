import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from scipy.interpolate import UnivariateSpline,InterpolatedUnivariateSpline,interp1d
from smooth import savitzky_golay
import uncertainties as uc
from uncertainties import umath

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern Roman']
rcParams['text.usetex'] = True
rcParams['figure.autolayout'] = True


fig_dir = "./figures/"



data = np.load("./data/measure2_1d.npy")
data = data[17:28]
P = data / np.sum(data) 
channel = np.arange(17,28,1)
plt.figure()
plt.plot(channel,P)
plt.ylabel("Probability P")
plt.xlim(17,27)
plt.show()
"""
To do: Plugin energy scaling for x-axis. 
Fitting Breit Wigner
gamma -> tau 

after: voigt profile


k
def gauss(x, *p):
    gamma, = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

# p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
p0 = [1., 0., 1.]

        coeff, var_matrix = curve_fit(gauss, bin_centres, hist, p0=p0)

        # Get the fitted curve
        hist_fit = gauss(bin_centres, *coeff)

        fig = plt.figure(figsize = (12,8))
        ax = fig.add_subplot(111)



        ax.errorbar(bin_centres, hist, label='Test data',yerr = np.sqrt(len(data)))
        #ax.plot(bin_centres, hist_fit, label='Fitted data')

        # Finally, lets get the fitting parameters, i.e. the mean and standard deviation:
        mu  = coeff[1]
        sigma = coeff[2]
        #ax.set_title = ("Measure 2.1%s with    $\mu=%.3f$ $\sigma = %.3f$" %(q,coeff[1],coeff[2]))

        ax.set_xlabel("Channels", fontsize = 40)
        ax.set_ylabel("counts", fontsize = 40)
        ax.xaxis.set_tick_params(labelsize = 35)
        ax.yaxis.set_tick_params(labelsize = 35)

        ax.locator_params(nbins=5)
        
        # place a text box in upper left in axes coords
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        textstr = 'Position: %s\n PM: %s'%(pos,pm)
        if q in "a,c,e": 
            xpos = 0.05 
        else:
            xpos = 0.6 
        ax.text(xpos, 0.95, textstr, transform=ax.transAxes, fontsize=40, va='top', bbox=props)

        ax.grid(True)
        ax.set_xlim(min(bin_centres),max(bin_edges))
        make_fig(fig,0,1,name="plot2_1"+q)
""" 
