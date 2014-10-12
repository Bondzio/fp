import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
rcParams['axes.labelsize'] = 14
rcParams['xtick.labelsize'] = 14
rcParams['ytick.labelsize'] = 14
rcParams['legend.fontsize'] = 14
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern Roman']
rcParams['text.usetex'] = True
rcParams['figure.figsize'] = 8.3, 5.2

fig_dir = "./figures/"

def halflife():
    """plotting the halflife figure
    :returns: TODO

    """
    tau =  270 / np.log(2)
    plt.figure()
    T_max = 4*365
    t = np.arange(0,T_max,1)
    plt.plot(t,1 - np.exp(-t/tau))
    plt.xlim(0, T_max)
    plt.xlabel("time $t$ in days")
    plt.ylabel("Probability of decaying in \%")
    plt.savefig("figures/halflife.pdf")
    plt.show()

import scipy.constants as cs
eV = cs.eV
c  = cs.c

def energy_terms(Z,N):
    """Liquid Drop Model

    :Z: Number of protons 
    :N: Number of neutrons 
    :A: Number of nuclei
    """
    A = Z + N

    #Change to electron volt
    m_e = cs.m_e / cs.eV 
    m_p = cs.m_p / cs.eV 
    m_n = cs.m_n / cs.eV 

    a1 = 15.56  
    a2 = 17.23  
    a3 = 0.697  
    a4 = 93.14  
    a5 = 12.00  

    f0 = Z * (m_p + m_e) + (N) * m_n
    f1 = - a1*A
    f2 = a2 * A**(2/3)
    f3 = a3 * (Z * ( Z-1) )*A**(-1/3)
    f4 = a4 * (Z - A/2)**2 / A
    val = 1
    if (Z%2 == True) and (N%2 == True):
        val *= -1
    elif ((Z%2 == True) and (N%2 == False)) or ((Z%2 == False) and (N%2 == True)):
        val = 0
    f5 = val * a5 * A**(-1/2)
    
    return (np.abs(sum([f0,f1,f2,f3,f4,f5]) )+ 0.01, (sum([f1,f2,f3,f4,f5])))

from matplotlib.colors import LogNorm
def plotdrops():
    """ Plotting the dropplets
    """
    pass
Z = np.arange(1,150,1)
N = np.arange(1,151,1)

ZZ,NN = np.meshgrid(Z,N)

total_energies = ZZ*0.0   
binding_energies = ZZ*0.0   

for i in range(len(N)):
    for j in range(len(Z)):
        print(i,j)
        total_energies[i,j] , binding_energies[i,j]  = energy_terms(ZZ[i,j],NN[i,j])

plt.figure()
plt.pcolor(ZZ,NN,total_energies,cmap='RdBu',norm=LogNorm(total_energies.min(), vmax=total_energies.max()))
plt.xlim(min(Z),max(Z))
plt.ylim(min(N),max(N))
plt.colorbar()
plt.show()

plt.figure()
plt.pcolor(ZZ,NN,binding_energies,cmap='RdBu')
plt.xlim(min(Z),max(Z))
plt.ylim(min(N),max(N))
plt.colorbar()
plt.show()


plotdrops()
