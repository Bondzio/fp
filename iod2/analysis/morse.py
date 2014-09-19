import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc
ur = 100 * sc.c * sc.h 
De = 4391 * ur 
we = 124.6 *ur / sc.hbar 
mu = 126.09447*sc.u /2 
a  = we * np.sqrt(mu/(2* De)) * ur
Be = 0.029 
re = sc.hbar / ( 4* np.pi * mu * sc.c * Be) 
mu2 = sc.hbar / ( 4* np.pi * sc.c * Be * 3 * 10**-8) 
V  = lambda r: De*(1 - np.exp(r-re))**2
r  = np.linspace(0,2,100)

def plot_it():
    plt.figure()
    plt.plot(r,V(r), "g--")
    plt.xlabel("radius $r$")
    plt.ylabel("Potential $V(r)$ in $cm^{-1}$")
    plt.grid(True)
    plt.title("Morsepotential $V(r)$")
    plt.show()
