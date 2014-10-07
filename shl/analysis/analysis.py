import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from matplotlib import rcParams
rcParams['axes.labelsize'] = 12
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['legend.fontsize'] = 12
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern Roman']
rcParams['text.usetex'] = True
rcParams['figure.figsize'] = 7.3, 4.2

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

        fig = plt.figure()

        plt.errorbar(bin_centres, hist, label='Test data',yerr = np.sqrt(len(data)))
        plt.plot(bin_centres, hist_fit, label='Fitted data')

        # Finally, lets get the fitting parameters, i.e. the mean and standard deviation:
        print( 'Fitted mean = ', coeff[1])
        print( 'Fitted standard deviation = ', coeff[2])
        plt.title = ("Measure 2.1%s with    $\mu=%.3f$ $\sigma = %.3f$" %(q,coeff[1],coeff[2]))
        plt.xlabel("Channels")
        plt.ylabel("counts")
        plt.grid()
        make_fig(fig,1,1,name="plot2_1"+q)

plot_2_1()

def plot_4_1():
    data = np.load("./data/measure4_1.npy")[2:]
    c    = np.where(data!=0)
    x = np.arange(0,len(data),1)
    fig = plt.figure()
    plt.grid(True)
    plt.yscale("log")
    plt.errorbar(x[c],data[c], yerr= np.sqrt(data[c]))
    #plt.scatter(x[c],data[c])
    make_fig(fig,1,1,name="plot4_1")


def plot_5_1():
    data = (np.load("./data/measure5_1.npy")[2:])
    c    = (np.where(data!=0)[0])[2:]
    x = np.arange(0,len(data),1)
    fig = plt.figure()
    plt.grid(True)
    #plt.errorbar(x[c],data[c], yerr= np.sqrt(data[c]))
    plt.scatter(x[c],data[c])
    make_fig(fig,1,1,name="plot5_1")
    plt.show()

