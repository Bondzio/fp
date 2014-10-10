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

