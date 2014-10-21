import numpy as np
import matplotlib.pyplot as plt
import seaborn as ssn

from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern Roman']
rcParams['text.usetex'] = True
rcParams['figure.autolayout'] = True

fig_dir = "../figures/"

def use_2pi(fig,ax):
    unit   = 0.25 
    x_tick = np.arange(-2*unit, (2+1)*unit, unit)

    x_label = [r"$-\pi$", r"$- \frac{\pi}{4}$", r"$0$", r"$\frac{\pi}{4}$",   r"$\pi$"]
    ax.set_xticks(x_tick*2*np.pi)
    ax.set_xticklabels(x_label, fontsize=20)
    
def make_fig(fig, show=True,save=False, name="foo"):
    if show == True:
        fig.show()
    if save == True:
        fig.savefig(fig_dir + name + ".pdf")



lamb = 632.8e-9

k = 1 / lamb
theta = np.linspace(-np.pi,np.pi,1000)
x0 = np.linspace(-2,2,1000)
b = 1e-5
I = np.sin(k * b/2* np.sin(theta))**2/ (k**2 * np.sin(theta)**2) 
#I = np.sin(k * b/2* x0)**2/ (k**2 * x0**2) 
I/=np.sum(I)


fig = plt.figure()
ax  = plt.subplot(111)
use_2pi(fig,ax)
plt.plot(x0, I)
plt.xlabel(r"angle $\theta$")
plt.ylabel(r"density $\mathcal{P}$")
plt.show()
make_fig(fig, 1 ,1, "sinc1")
