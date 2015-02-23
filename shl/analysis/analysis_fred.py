import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


#fredsche routinen
import stat1 as st


import seaborn as sns

from matplotlib import rcParams
from scipy.interpolate import UnivariateSpline,InterpolatedUnivariateSpline,interp1d
from smooth import savitzky_golay
import uncertainties as uc
import uncertainties.unumpy as un
from uncertainties import umath


rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern Roman']
rcParams['text.usetex'] = True
rcParams['figure.autolayout'] = True


fig_dir = "./figures/"

def unv(uarray):        # returning nominal values of a uarray
    return un.nominal_values(uarray)

def usd(uarray):        # returning the standard deviations of a uarray
    return un.std_devs(uarray)


def make_fig(fig, show=True,save=False, name="foo"):
    if show == True:
        fig.show()
    if save == True:
        fig.savefig(fig_dir + name + ".pdf")


if True:
    """

    The day we saw the hawk
    On a churchyard tree
    A kite too
    Was in the far sky.

    - Hekigodo 


    """
    data_original = np.load("./data/measure4_1.npy")[2:]
    data = data_original[150:30:-1]
    x = np.arange(len(data)) * 1.3

    error = np.sqrt(data) 
    error[np.where(error == 0)] += 1
    # only fit for x < 135
    fig = plt.figure()
    ax  = plt.subplot(111)
    plt.grid(True)

    fit = True

    if fit==True:

        def func(x, a, b, c):
            #a, b, c = p
            return a + b * c**x

        # p0 is the initial guess for the fitting coefficients 
        p0 = [1., 1., 1.]

        p, cov = curve_fit(func, x, data, p0=p0)
        #, sigma=np.sqrt(1/error))
        p_uc = uc.correlated_values(p, cov)
        c = p_uc[2]
        T12_lit = 98 
        lamb_lit = -(np.log(2)/T12_lit)
        print("lit",lamb_lit)
        

        lamb = umath.log(c)
        print(lamb)
        T12 = -np.log(2) /lamb 
        print("t12=",T12)

        x_fit = np.linspace(min(x),max(x))

        data_fit = unv(func(x_fit,*p_uc) )
        data_fit_min = unv(func(x_fit, *p_uc)) - usd(func(x_fit, *p_uc))
        data_fit_max = unv(func(x_fit, *p_uc)) + usd(func(x_fit, *p_uc))

        """
        ax1.fill_between(Is, 
                unv(np.polyval(c, Is)) + usd(np.polyval(c, Is)),
                unv(np.polyval(c, Is)) - usd(np.polyval(c, Is)),
                facecolor=colors[0], color=colors[0], alpha=0.2)
        ax1.plot(Is, np.polyval(coeff, Is), '-', linewidth=1.0)                
        """
        plt.plot(x_fit,data_fit)
        plt.plot(x_fit,90*np.exp(x_fit * lamb_lit))
        plt.fill_between(x_fit, data_fit_min , data_fit_max,facecolor="r", color="b", alpha=0.3 )

        # place a text box in upper left in axes coords
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        textstr = '$a + b \cdot c^x$ with\n$a=%.2f$\n$b=%.2f$\n$c=%.2f$'%(p[0], p[1],p[2])
        ax.text(0.6, 0.85, textstr, transform=ax.transAxes, fontsize=18, va='top', bbox=props)

        ax.xaxis.set_tick_params(labelsize = 14)
        ax.yaxis.set_tick_params(labelsize = 14)

        ax.add_patch(plt.Rectangle((0,0.1),155,100,alpha = 0.2))
        """
        """
    
    plt.errorbar(x,data, yerr=error,fmt="x")
    plt.ylim(min(data)*0.8,max(data))
    #plt.yscale("log")
    plt.xlim(min(x)*0.8,max(x))
    plt.xlabel("time in $ns$", fontsize = 14)
    plt.ylabel("counts", fontsize = 14)
    fig.show()

"""

#    x2 = np.arange(0,len(data2),1)

    fit = True 
    redistribute = True 

    #x2 = 1.3149710372035508*x2 -22.617788714272098
    c2 = np.where(x2 < 135)

    data = data2[c2] 
    x  = x2[c2]
    print("datapoints:",len(data))

    mass = 79/251/6080*52658
    if redistribute == True:

        # conserving the mass
        total_mass = mass * len(data)
        remaining = (data > 0)

        while True:
            print("new redistributing ...")
            print("total mass:",total_mass)
            # find those which are smaller
            q = (data[remaining] <= mass)
            remaining = ~q
            if len(np.nonzero(q)[0]) == 0:
                data[remaining] -= mass
                break
            print("number of smaller values:",len(np.nonzero(q)[0]),"\n")
            # subtract the mass of this data
            total_mass -= np.sum(data[q])
            mass = total_mass / len(np.nonzero(~remaining)[0])  
            data[q] = 0

        # redistribute total remaining mass to single channels
        print("number of nonzero:",len(np.nonzero(data)[0]))

    c    = np.nonzero(data)  
    data = data[c]
    x = x[c]

    #scaling to time units
    x = 6.3149710372035508*x -22.617788714272098
    c = (x>0)
    x = x[c]
    data = data[c]

    x = x[::-1] - min(x)


    error = np.sqrt(data) 
    # only fit for x < 135
    fig = plt.figure()
    ax  = plt.subplot(111)
    plt.grid(True)

    if fit==True:

        def func(x, *p):
            a,b,c = p
            return a + b * c**x

        # p0 is the initial guess for the fitting coefficients 
        p0 = [1., 1., 1.]

        p, cov = curve_fit(func, x, data, p0=p0, sigma = error)
        p_uc = uc.correlated_values(p, cov)
        c = p_uc[2]

        T12_lit = 98 
        lamb_lit = -(np.log(2)/T12_lit)
        print("lit",lamb_lit)
        

        lamb = umath.log(c)
        print(lamb)
        T12 = -np.log(2) /lamb 
        print("t12=",T12)

        x_fit = np.linspace(min(x),max(x))

        data_fit = func(x_fit,*p) 
        pmin = (p - np.sqrt(np.diag(cov)))
        pmax = (p + np.sqrt(np.diag(cov)))

        data_fit_min = func(x_fit, *pmin)
        data_fit_max = func(x_fit, *pmax)

        plt.plot(x_fit,data_fit)
        plt.plot(x_fit,90*np.exp(x_fit * lamb_lit))
        plt.fill_between(x_fit, data_fit_min , data_fit_max,facecolor="r", color="b", alpha=0.3 )

        # place a text box in upper left in axes coords
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        textstr = '$a + b \cdot c^x$ with\n$a=%.2f$\n$b=%.2f$\n$c=%.2f$'%(p[0], p[1],p[2])
        ax.text(0.6, 0.85, textstr, transform=ax.transAxes, fontsize=18, va='top', bbox=props)

        ax.xaxis.set_tick_params(labelsize = 14)
        ax.yaxis.set_tick_params(labelsize = 14)

        ax.add_patch(plt.Rectangle((0,0.1),155,100,alpha = 0.2))

    plt.errorbar(x,data, yerr=error,fmt="x")
    #plt.scatter(x,data,c="blue",alpha = 0.9,s=100, marker="x")
    plt.ylim(min(data)*0.8,max(data))
    #plt.yscale("log")
    plt.xlim(min(x)*0.8,max(x))
    plt.xlabel("time in $ns$", fontsize = 14)
    plt.ylabel("counts", fontsize = 14)
    make_fig(fig,1,1,name="plot4_1_reg")
"""
