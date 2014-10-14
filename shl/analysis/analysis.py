import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


#fredsche routinen
from stat import *


import seaborn as sns

from matplotlib import rcParams
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

def plot_2_1():
    for q in "abcde": 
        data = np.load("./data/measure2_1"+q+".npy")

        # Define some test data which is close to Gaussian

        hist, bin_edges = data, np.arange(0,len(data)+1,1)
        bin_centres = (bin_edges[:-1] + bin_edges[1:])/2

        # Define model function to be used to fit to the data above:
        def gauss(x, *p):
            A, mu, sigma = p
            return A*np.exp(-(x-mu)**2/(2.*sigma**2))

        # p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
        p0 = [1., 0., 1.]

        coeff, var_matrix = curve_fit(gauss, bin_centres, hist, p0=p0)

        # Get the fitted curve
        hist_fit = gauss(bin_centres, *coeff)

        fig = plt.figure(figsize = (12,8))
        ax = fig.add_subplot(111)



        ax.errorbar(bin_centres, hist, label='Test data',yerr = np.sqrt(len(data)))
        ax.plot(bin_centres, hist_fit, label='Fitted data')

        # Finally, lets get the fitting parameters, i.e. the mean and standard deviation:
        mu  = coeff[1]
        sigma = coeff[2]
        ax.set_title = ("Measure 2.1%s with    $\mu=%.3f$ $\sigma = %.3f$" %(q,coeff[1],coeff[2]))

        ax.set_xlabel("Channels", fontsize = 40)
        ax.set_ylabel("counts", fontsize = 40)
        ax.xaxis.set_tick_params(labelsize = 35)
        ax.yaxis.set_tick_params(labelsize = 35)

        ax.locator_params(nbins=5)
        
        # place a text box in upper left in axes coords
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        textstr = '$\mu=%.2f$\n$\sigma=%.2f$'%(mu, sigma)
        if q in "a,c,e": 
            xpos = 0.05 
        else:
            xpos = 0.6 
        ax.text(xpos, 0.95, textstr, transform=ax.transAxes, fontsize=40, va='top', bbox=props)

        ax.grid(True)
        ax.set_xlim(min(bin_centres),max(bin_edges))
        make_fig(fig,0,1,name="plot2_1"+q)

from scipy.interpolate import UnivariateSpline,InterpolatedUnivariateSpline,interp1d
from smooth import savitzky_golay
def plot_4_1():
    data = np.load("./data/measure4_1.npy")[2:]
    c    = np.where(data!=0)
    x = np.arange(0,len(data),1)

    x = x[c]
    data = data[c]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.grid(True)
    #plt.yscale("log")
    plt.errorbar(x,data, yerr= np.sqrt(data),fmt="x")
    #plt.scatter(x,data)

    errorA = data + np.sqrt(data)
    errorB = data - np.sqrt(data)
    s1 = savitzky_golay(errorA, 61, 3)
    s2 = savitzky_golay(errorB, 61, 3)
    xx = np.linspace(min(x),max(x),1000)
    #plt.plot(xx,s1(xx))
    plt.fill_between(x, s1, s2, facecolor='grey', alpha=0.3,interpolate=True)

    plt.ylim(min(data)*0.8,max(data))
    plt.xlim(min(x)*0.8,max(x))
    plt.xlabel("Channels", fontsize = 14)
    plt.ylabel("counts", fontsize = 14)

    ax.xaxis.set_tick_params(labelsize = 14)
    ax.yaxis.set_tick_params(labelsize = 14)


    make_fig(fig,1,1,name="plot4_1")

def plot_4_1_log():
    data = np.load("./data/measure4_1.npy")[2:]
    c    = np.where(data!=0)
    x = np.arange(0,len(data),1)

    x = x[c]
    data = data[c]

    error = np.sqrt(data) 

    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.yscale("log")
    plt.grid(True)
    #plt.errorbar(x,data, yerr= np.sqrt(data),fmt="x")
    plt.scatter(x,data,c="blue",alpha = 0.5,s=50, marker="o")

    ax.xaxis.set_tick_params(labelsize = 14)
    ax.yaxis.set_tick_params(labelsize = 14)

    ax.add_patch(plt.Rectangle((0,0.1),135,100,alpha = 0.2))

    plt.ylim(min(data)*0.8,max(data)*2)
    plt.xlim(min(x)*0.8,max(x))
    plt.xlabel("Channels", fontsize = 14)
    plt.ylabel("counts", fontsize = 14)
    make_fig(fig,1,1,name="plot4_1_log")


def plot_4_1_reg():
    data = np.load("./data/measure4_1.npy")[2:]
    c    = np.where(data!=0)
    x = np.arange(0,len(data),1)

    x = x[c]
    data = data[c]


    error = np.sqrt(data) 
    # only fit for x < 135
    c2 = np.where(x < 135)
    data_log = np.log(data[c2])
    error_log = np.sqrt(data_log) **3
    p, cov= np.polyfit(x[c2],data_log, 1, full=False, cov=True, w=error_log)
    x_fit = np.linspace(min(x[c2]),max(x[c2]))
    data_fit = np.exp(np.polyval(p, x_fit))
    data_fit_min = np.exp(np.polyval(p - np.sqrt(np.diag(cov)),x_fit))
    data_fit_max = np.exp(np.polyval(p + np.sqrt(np.diag(cov)),x_fit))

    fig = plt.figure()
    plt.yscale("log")
    plt.grid(True)
    #plt.errorbar(x,data, yerr= np.sqrt(data),fmt="x")
    plt.scatter(x,data,c="blue",alpha = 0.9,s=100, marker="x")
    plt.plot(x_fit,data_fit)
    plt.fill_between(x_fit, data_fit_min , data_fit_max,facecolor="r", color="b", alpha=0.3 )

    # place a text box in upper left in axes coords
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    textstr = '$ax + c$ with\n$a=%.2f$\n$b=%.2f$'%(p[0], p[1])
    ax = plt.gca()
    ax.text(0.7, 0.95, textstr, transform=ax.transAxes, fontsize=40, va='top', bbox=props)

    ax.xaxis.set_tick_params(labelsize = 14)
    ax.yaxis.set_tick_params(labelsize = 14)

    ax.add_patch(plt.Rectangle((0,0.1),135,100,alpha = 0.2))

    plt.ylim(min(data)*0.8,max(data)*2)
    plt.xlim(min(x)*0.8,max(x))
    plt.xlabel("Channels", fontsize = 40)
    plt.ylabel("counts", fontsize = 40)
    make_fig(fig,1,1,name="plot4_1_reg")


def plot_5_1():
    data = (np.load("./data/measure5_1.npy")[2:])
    c    = (np.where(data!=0)[0])[2:]
    x = (np.arange(0,len(data),1))[c]
    data = data[c]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.xaxis.set_tick_params(labelsize = 14)
    ax.yaxis.set_tick_params(labelsize = 14)


    ax.grid(True)
    #plt.errorbar(x[c],data[c], yerr= np.sqrt(data[c]))
    plt.scatter(x,data,c="blue",alpha = 0.3,s=100, marker="o")

    plt.ylim(min(data)*0.8,max(data)*2)
    plt.xlim(min(x)*0.8,max(x)*1.03)
    plt.xlabel("Channels", fontsize = 14)
    plt.ylabel("counts", fontsize = 14)

    make_fig(fig,1,1,name="plot5_1")
    plt.show()

def plot_5_1_hist():
    data = (np.load("./data/measure5_1.npy")[2:])
    c    = (np.where(data!=0)[0])[2:]
    x = (np.arange(0,len(data),1))[c]
    data = data[c]
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    
    ax1.xaxis.set_tick_params(labelsize = 14)
    ax1.yaxis.set_tick_params(labelsize = 14)
    ax1.grid(True)
    ax1.scatter(x,data,c="blue",alpha = 0.3,s=100, marker="o")

    ax1.set_ylim(min(data)*0.8,max(data)*1.2)
    ax1.set_xlim(min(x)*0.8,max(x)*1.03)
    ax1.set_xlabel("Channels", fontsize = 14)
    ax1.set_ylabel("counts", fontsize = 14)

    ax1.yticks = [1,2,3]
    ax1.grid(True)
    ax1.locator_params(nbins=3)

    ax2.xaxis.set_tick_params(labelsize = 14)
    ax2.yaxis.set_tick_params(labelsize = 14)



    #plt.errorbar(x[c],data[c], yerr= np.sqrt(data[c]))
    ax2.hist(data, 3, orientation="horizontal", align='mid', rwidth=20)

    ax2.set_xlabel("occurence", fontsize = 14)

    make_fig(fig,1,1,name="plot5_1_hist")
    plt.show()

def calib_TAC_MCA():
    delay   = np.array([0,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,190.5])
    channel = np.array([20,24,29,34,41,47,53,59,66,72,78,84,90,96,102,109,115,121,127,133,139,145,151,157,162])
    x_error = delay * 0 + 1
    y_error = channel * 0 + 1

    fig = plt.figure()
    ax = fig.add_subplot(111)


    p, cov= np.polyfit(channel,delay, 1, full=False, cov=True, w=y_error)
    x_fit = np.linspace(20,162,1000)

    data_fit     = (np.polyval(p, x_fit))
    data_fit_min = (np.polyval(p - np.sqrt(np.diag(cov)),x_fit))
    data_fit_max = (np.polyval(p + np.sqrt(np.diag(cov)),x_fit))

    plt.plot(x_fit,data_fit)
    
    plt.xlim(20,162)
    plt.ylim(0,191)

    plt.ylabel("Delay",fontsize = 14)
    plt.xlabel("Channel", fontsize = 14)

    ax.xaxis.set_tick_params(labelsize = 14)
    ax.yaxis.set_tick_params(labelsize = 14)



    plt.fill_between(x_fit, data_fit_min , data_fit_max,facecolor="r", color="b", alpha=0.3 )

    plt.errorbar(channel,delay, yerr= y_error,xerr= x_error, fmt= "x")
    make_fig(fig,1,1,name = "plot7")

def calib_TAC_MCA2():
    delay   = np.array([25,108.5,162.,0.5,65.5,113.5,75.5,49,79,135.5,23.5,98.5,135.5,110.5,121,127,45,51.5,184,80,185,88.5,52,167.5,44.5])
    channel = np.array([35,100,141,22,66,104,74,54,77,121,34,92,119,102,110,114,51,56,157,78,158,84,56,145,50])
    x_error = delay * 0 + 1
    y_error = channel * 0 + 1

    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.errorbar(channel,delay, yerr= y_error,xerr= x_error, fmt= "x")

    p, cov= np.polyfit(channel,delay, 1, full=False, cov=True, w=y_error)
    x_fit = np.linspace(20,162,1000)

    data_fit     = (np.polyval(p, x_fit))
    data_fit_min = (np.polyval(p - np.sqrt(np.diag(cov)),x_fit))
    data_fit_max = (np.polyval(p + np.sqrt(np.diag(cov)),x_fit))

    plt.plot(x_fit,data_fit)
    
    plt.xlim(20,162)
    plt.ylim(0,191)

    plt.ylabel("Delay",fontsize = 14)
    plt.xlabel("Channel", fontsize = 14)

    ax.xaxis.set_tick_params(labelsize = 14)
    ax.yaxis.set_tick_params(labelsize = 14)



    plt.fill_between(x_fit, data_fit_min , data_fit_max,facecolor="r", color="b", alpha=0.3 )
    make_fig(fig,1,1,name = "plot71")

calib_TAC_MCA()
def calib_TAC_MCA_rescaled():
    delay   = np.array([0,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,190.5])
    channel = np.array([20,24,29,34,41,47,53,59,66,72,78,84,90,96,102,109,115,121,127,133,139,145,151,157,162])
    x_error = delay * 0 + 1
    y_error = channel * 0 + 1.5

    fig = plt.figure()
    ax = fig.add_subplot(111)


    p, cov= np.polyfit(channel,delay, 1, full=False, cov=True, w=y_error)
    x_fit = np.linspace(20,162,1000)

    data_fit     = (np.polyval(p, x_fit))
    data_fit_min = (np.polyval(p - np.sqrt(np.diag(cov)),x_fit))- (p[0]*x_fit + p[1])
    data_fit_max = (np.polyval(p + np.sqrt(np.diag(cov)),x_fit))- (p[0]*x_fit + p[1])
    
    delay   = delay   - (p[0]*channel + p[1])
    data_fit = data_fit - (p[0]*x_fit   + p[1])

    plt.plot(x_fit,data_fit)
    
    plt.fill_between(x_fit, data_fit_min , data_fit_max,facecolor="r", color="b", alpha=0.3 )
    plt.errorbar(channel,delay, yerr= y_error,xerr= x_error, fmt= "x")

    plt.xlim(20,162)
    plt.ylim(-3,3)

    plt.ylabel("Delay",fontsize = 14)
    plt.xlabel("Channel", fontsize = 14)

    ax.xaxis.set_tick_params(labelsize = 14)
    ax.yaxis.set_tick_params(labelsize = 14)



    make_fig(fig,1,1,name = "plot7b")



