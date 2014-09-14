import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, hbar , eV
I = np.load("data_npy/jod_1_I.npy")
L = np.load("data_npy/jod_1_l.npy")
 

n = 4
minima = []
k_min = np.max(np.where(L < 512))
k_max = np.min(np.where(L > 612))
for k in range(k_min,k_max):
    if np.all(I[k] < I[k-n:k]) and np.all(I[k] < I[k+1:k+n+1]):
        minima+=[k]
E = c/L*hbar / eV *10**9

def plot_lines():
    f, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.scatter(L[k_min:k_max],I[k_min:k_max])
    ax1.plot(L[k_min:k_max],I[k_min:k_max])
    m = np.max(I) * 1.3
    for k in minima:
        i = I[k]
        l = L[k]
        t = np.linspace(i,m,1000)
        ax1.plot(t*0 + l, t , c= "r")
    ax1.set_xlabel("Wavelength $\lambda$ in $nm$") 
    ax1.set_ylabel("Intensity")

    ax2.scatter(L[minima][1:],np.diff(L[minima]))
    plt.show()
plot_lines()

def plot_diff():
    plot_lines()
    plt.figure()
    plt.scatter(E[minima][1:],np.diff(E[minima]))
    plt.xlabel("Energie $E$ in eV")
    plt.ylabel("Energiedifferenz $\Delta E$ in eV")
    plt.show()

