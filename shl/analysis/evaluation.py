import numpy as np
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

import uncertainties as uc
from uncertainties import umath

"""
Ein Wort

(Gottfried Benn)

Ein Wort, ein Satz -: aus Chiffren steigen
erkanntes Leben, jäher Sinn,
die Sonne steht, die Sphären schweigen,
und alles ballt sich zu ihm hin.

Ein Wort - ein Glanz, ein Flug, ein Feuer,
ein Flammenwurf, ein Sternenstrich -
und wieder Dunkel, ungeheuer,
im leeren Raum um Welt und Ich.
"""

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

plot         = True  # Do you really want to plot?
fit          = True  # Do the fitting with scipy.curve_fit 
redistribute = False # Redistributing the background 
save         = True  # Saving the figure (You should have plotted though)
show         = True  # Showing the figure ("After all is said and done, more is said than done" Aesop)
caling       = True  # rescaling the channels to time 
flip         = True  # Flipping the plot such that e^x -> e^-x 
cut          = True  # Cutting all the data for which t<0 

total_time = 52658

if scaling == True:
    # Getting the scaling of the channels 
    delay = np.load("data/delay.npy")
    channel = np.load("data/channel.npy")

    #these errors were also confirmed by the χ² test
    x_error = delay * 0 + 1
    y_error = channel * 0 + 1

    (alpha,beta), cov= np.polyfit(channel,delay, 1, full=False, cov=True, w=y_error)
    #alpha,beta = uc.correlated_values(p, cov)
    print("scaling with %.3f * channel + %.3f"%(alpha,beta))

    # We can only fit over y-errors. Should this be changed, we can
    # adopt the method to include the error on t
    tfunc= lambda channel: alpha*channel + beta
    # this can be applied after we subtracted the background

# Loading data 4.1
data_all    = np.load("./data/measure4_1.npy")[2:]
channel_all = np.arange(0,len(data_all),1)

# find Maximum of peak and take only data which is smaller
channel_peak = np.argmax(data_all)
print("found maximum at %d"%channel_peak)

data       = data_all[channel_all < channel_peak] 
channel    = channel_all[channel_all < channel_peak]
no_channel = len(channel[channel>0])

#subtracting the background
back_data = (np.load("./data/measure5_1.npy")[2:])
total_background = np.sum(back_data[back_data < 25])
no_backgroundchannels = len(back_data[back_data < 25])
total_back_time = 6080
print("Total background: %d with %d channels"%(total_background,no_backgroundchannels))
mass = total_background / no_backgroundchannels / total_back_time * total_time

if redistribute == True:

    total_mass = mass * no_channel 
    remaining = (data > 0)

    while True:
        print("new redistributing ...")
        print("total mass: %.3f"%(total_mass))
        mass = total_mass / len(np.nonzero(remaining)[0])  
        print("=> mass per channel: %.3f"%mass)
        # find those which are smaller
        q = (data[remaining] <= mass)
        remaining = ~q
        if len(np.nonzero(q)[0]) == 0:
            data[remaining] -= mass
            break
        print("number of smaller values: %d \n"%(len(np.nonzero(q)[0])))
        # subtract the mass of this data
        total_mass -= np.sum(data[q])
        data[q] = 0
    # redistribute total remaining mass to single channels
    print("number of nonzero:",len(np.nonzero(data)[0]))
else:
    data -= mass
    data = data[data>0]
    channel = channel[data>0]

if scaling == True:
    # rescale
    t = (tfunc(channel))
else:
    t = channel

if cut == True:
    data = data[t>0]
    t    = t[t>0]

if flip == True:
    t = t[::-1]

error = np.sqrt(data)

if plot==True:
    fig = plt.figure()
    ax  = plt.subplot(111)

if fit==True:
    def func(x, *p):
        a,b,c = p
        return a + b * np.exp(-c*x)

    # p0 is the initial guess for the fitting coefficients 
    p0 = [1, 1,1]

    p, cov = curve_fit(func, t, data, p0=p0, sigma = error)



    p_uc = uc.correlated_values(p, cov)
    lamb = p_uc[2]

    T12_lit = 98 
    lamb_lit = -(np.log(2)/T12_lit)
    print("literature lambda: %.3f"%lamb_lit)
    print("fitted lambda:",lamb)
    

    t_fit = np.linspace(min(t),max(t))

    data_fit = func(t_fit,*p) 
    data_lit = func(t_fit,*(0,max(data),-lamb_lit))
    pmin = (p - np.sqrt(np.diag(cov)))
    pmax = (p + np.sqrt(np.diag(cov)))

    data_fit_min = func(t_fit, *pmin)
    data_fit_max = func(t_fit, *pmax)

    if plot == True:    

        plt.plot(t_fit,data_fit)
        plt.plot(t_fit,data_lit)
        plt.fill_between(t_fit, data_fit_min , data_fit_max,facecolor="r", color="b", alpha=0.3 )

        # place a text box in upper left in axes coords
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        textstr = '$a + b \cdot e^{-cx}$ with\n$a=%.2f$\n$b=%.2f$\n$c=%.2f$'%(p[0], p[1],p[2])
        ax.text(0.6, 0.85, textstr, transform=ax.transAxes, fontsize=18, va='top', bbox=props)

        ax.xaxis.set_tick_params(labelsize = 14)
        ax.yaxis.set_tick_params(labelsize = 14)

        ax.add_patch(plt.Rectangle((0,0.1),155,100,alpha = 0.2))

if plot == True:
    plt.errorbar(t,data, yerr=error,fmt="x")
    plt.ylim(min(data)*0.8,max(data))
    #plt.yscale("log")
    plt.xlim(min(t)*0.8,max(t))
    plt.xlabel("time in $ns$", fontsize = 14)
    plt.ylabel("counts", fontsize = 14)
    make_fig(fig,show,save,name="plot4_1_reg")

