import numpy as np
import matplotlib.pyplot as plt
import seaborn as ssn

fontsize_labels = 12    # size used in latex document

from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern Roman']
rcParams['text.usetex'] = True
rcParams['figure.autolayout'] = True
rcParams['axes.labelsize'] = fontsize_labels
rcParams['xtick.labelsize'] = fontsize_labels
rcParams['ytick.labelsize'] = fontsize_labels


def use_2pi(fig,ax):
    unit   = 0.25 
    x_tick = np.arange(-2*unit, (2+1)*unit, unit)

    x_label = [r"$-\pi$", r"$- \frac{\pi}{4}$", r"$0$", r"$\frac{\pi}{4}$",   r"$\pi$"]
    ax.set_xticks(x_tick*2*np.pi)
    ax.set_xticklabels(x_label, fontsize=20)


lamb = 632e-9

k = 1 / lamb
theta = np.linspace(-np.pi,np.pi,1000)
b = 1e-5
I = np.sin(k * b/2* np.sin(theta))**2/ (k**2 * np.sin(theta)**2) 
I/=np.sum(I)

fig = plt.figure()
ax  = plt.subplot(111)
use_2pi(fig,ax)
plt.plot(theta, I)
plt.xlabel(r"angle $\theta$")
plt.ylabel(r"density $\mathcal{P}$")
plt.show()

