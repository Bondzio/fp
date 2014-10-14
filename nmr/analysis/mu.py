import numpy as np
import pylab as plt
import scipy.constants as co
import uncertainties.unumpy as un
import seaborn as sns
sns.set(style='ticks', palette='Set2')
sns.despine()
plt.rc('text', usetex=True)
plt.rc('font',**{'family':'serif','sans-serif':['helvetica']})

# Calculating the gyromagnetic ratio and nuclea g-factor

nu = np.array([18.6905, 18.68130, 16.8503, 16.8704]) * 10**6
B = np.array([419, 423, 395, 394]) * 10**-3

s_nu = 0.0001 * 10**6
s_B = 5 * 10**-3
nu = un.uarray(nu, s_nu)
B = un.uarray(B, s_B)

def gamma(nu, B):
    gamma = 2 * np.pi * nu / B
    return gamma

gammas = gamma(nu,B)

mu_k = 5.05079e-27
g_N = gamma(nu, B) / mu_k * co.hbar

for i in range(4):
    print("{:L}".format(gammas[i]) + " & " + "{:L}".format(g_N[i]))



