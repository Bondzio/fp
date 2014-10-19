import numpy as np
import matplotlib.pyplot as plt
import seaborn as ssn

from scipy.optimize import curve_fit

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
        fig.savefig(fig_dir + name + ".png")
        plt.close(fig)



def plot_all():
    """
    Winter seclusion -
    Listening, that evening,
    To the rain in the mountain.

    - Issa 

    """
    names  =  [ ("npy/2.%d_HM1508-2"%i,"2.%d"%i) for i in range(1,5+1)]
    names +=  [ ("npy/3.%d_HM1508-2"%i,"3.%d"%i) for i in range(1,8+1)]
    names +=  [ ("npy/4.%d_HM1508-2"%i,"4.%d"%i) for i in range(1,5+1)]
    names +=  [ ("npy/5.%d_HM1508-2"%i,"5.%d"%i) for i in range(1,5+1)]

    for name,shortname in names:
            
        t = np.load(name+"_t.npy")
        A = np.load(name+"_channelA.npy")

        fig = plt.figure()
        ax  = plt.subplot(111)



        # Define model function to be used to fit to the data above:
        def sin_fit(x, *p):
            a,b,c,d= p
            return a + b*np.sin(c*x+ d)

        # p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
        p0 = [0.1, 1., 1., 0.1]

        coeff, var_matrix = curve_fit(sin_fit, t, A, p0=p0)



        plt.scatter(t,A, alpha = 0.4)
        #plt.plot(t, sin_fit(t,*coeff))
        plt.xlabel("time $t$",fontsize = 14)
        plt.ylabel("Amplitude $U$ in Volt", fontsize = 14)
        plt.title(shortname)
        make_fig(fig,1,1,name=shortname)

plot_all()
