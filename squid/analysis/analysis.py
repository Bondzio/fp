import numpy as np
import matplotlib.pyplot as plt
import seaborn as ssn

from scipy.fftpack import fft

from smooth import savitzky_golay
from scipy.optimize import curve_fit

#uncertainties
import uncertainties as uc
import uncertainties.unumpy as un
import stat1 as st


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



def plot_all():
    """
    Winter seclusion -
    Listening, that evening,
    To the rain in the mountain.

    - Issa 

    """
    names  =  [ ("npy/2.%d_HM1508-2"%i,"2.%d"%i) for i in range(1,5+1)]
    #names +=  [ ("npy/3.%d_HM1508-2"%i,"3.%d"%i) for i in range(1,8+1)]
    #names +=  [ ("npy/4.%d_HM1508-2"%i,"4.%d"%i) for i in range(1,5+1)]
    #names +=  [ ("npy/5.%d_HM1508-2"%i,"5.%d"%i) for i in range(1,5+1)]

    for name,shortname in names:
            
        t = np.load(name+"_t.npy")
        A = np.load(name+"_channelA.npy")

        fig = plt.figure()
        ax  = plt.subplot(111)



        plt.scatter(t,A, alpha = 0.4)
        plt.xlabel("time $t$",fontsize = 14)
        plt.ylabel("Amplitude $U$ in Volt", fontsize = 14)
        plt.title(shortname)
        make_fig(fig,1,0,name=shortname)
#plot_all()
def B_res():
    names  =  [ ("npy/2.%d_HM1508-2"%i,"2.%d"%i) for i in range(1,5+1)]
    #names  =  [ ("npy/4.1_HM1508-2","4.1")]
    for name,shortname in names:
        t = np.load(name+"_t.npy")[50:-50]
        A = np.load(name+"_channelA.npy")[50:-50]
        A -= (np.max(A)+np.min(A))/2

        fig = plt.figure()
        ax  = plt.subplot(111)

        # smoothing the data in order to get a stable fit.
        # Data which 
        A_smooth = savitzky_golay(A, 101, 2)

        # Define model function to be used to fit to the data above:
        def sin_fit(x, A,w,phi):
            return A*np.sin(w*x- phi) 

        # p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
        A0 = np.abs((np.max(A_smooth) - np.min(A_smooth))/2)

        p0 = [A0, 0.83, 1.6]


        p0,cov0 = curve_fit(sin_fit, t, A_smooth, p0=p0)

        p,cov = curve_fit(sin_fit, t, A, p0=p0)
        if p[0] < 0 :
            p[2] += np.pi
            p[0] *= -1
        p_uc = uc.correlated_values(p, cov)


        props = dict(boxstyle='round', facecolor='white', alpha=0.9)
        textstr = r"\begin{eqnarray*}"
        textstr += r"&&A \sin(\omega t - \phi) \\ "
        textstr += r"A &=& (%.3f \pm %.3f)V \\ "%(p[0],p_uc[0].s)
        textstr += r"\omega &=& (%.3f \pm %.3f)s^{-1} \\"%(p[1],p_uc[1].s)
        textstr += r"\phi &=& (%.3f \pm %.3f)"%(p[2],p_uc[2].s)
        textstr += r"\end{eqnarray*}"


        ax.text(0.7,0.25, textstr, transform=ax.transAxes, fontsize=12, va='top', bbox=props)
        plt.ylabel("Voltage / V")
        plt.xlabel("time in $ns$")
        plt.grid(True)

        plt.scatter(t,A, alpha = 0.1,label="data")
        plt.ylim(min(A),max(A))
        plt.xlim(min(t),max(t))
        #plt.plot(t, A_smooth)
        plt.plot(t, sin_fit(t,*p),label="least squares fit")
        plt.xlabel("time $t$",fontsize = 14)
        plt.ylabel("Amplitude $U$ in Volt", fontsize = 14)
        make_fig(fig,1,0,name="fit"+shortname)

B_res()
