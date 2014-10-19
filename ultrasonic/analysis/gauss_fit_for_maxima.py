import numpy as np
import pylab as plt
from scipy.optimize import curve_fit
import uncertainties as uc
import uncertainties.unumpy as un
plt.close("all")

# Gauge, measure b is much more useful
npy_files = "./data_npy/gauge_b"
t = np.load(npy_files + "_t" + ".npy")
signal = np.load(npy_files + "_ch_a" + ".npy")

def search_max(plot = False):
    return 0
data = signal
#error = np.sqrt(data)
channel = t
channel_fit = t

def func(x, a1, a2, a3, a4, a5, a6, \
        mu1, mu2, mu3, mu4, mu5, mu6, \
        sigma1, sigma2, sigma3, sigma4, sigma5, sigma6, \
        c):
    return a1*np.exp(1)**(- (x-mu1)**2 / (2*sigma1)**2 ) + \
           a2*np.exp(1)**(- (x-mu2)**2 / (2*sigma2)**2 ) + \
           a3*np.exp(1)**(- (x-mu3)**2 / (2*sigma3)**2 ) + \
           a4*np.exp(1)**(- (x-mu4)**2 / (2*sigma4)**2 ) + \
           a5*np.exp(1)**(- (x-mu5)**2 / (2*sigma5)**2 ) + \
           a6*np.exp(1)**(- (x-mu6)**2 / (2*sigma6)**2 ) + c

# p0 is the initial guess for the fitting coefficients 
s = 0.00002 # initial guess for sigmas
p0 = [0.05, 0.02, 0.35, 0.45, 0.20, 0.015, \
        0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007] + [s]*6 + [0]

p, cov = curve_fit(func, channel, data, p0=p0) #, sigma = error)
p_uc = uc.correlated_values(p, cov)
plot = True
if plot:
    fig, ax = plt.subplots(1, 1, figsize=(8.09, 5))
    ax.plot(t, signal)
    ax.plot(t, func(t, *p))
    fig.show(t)

"""
    mu = [p_uc[3].n,p_uc[4].n,p_uc[5].n]
    Smu = [p_uc[3].s,p_uc[4].s,p_uc[5].s]
    energies = [26.3, 33.2,59.5]
    return energies,mu,Smu 
"""
search_max(plot=True)




