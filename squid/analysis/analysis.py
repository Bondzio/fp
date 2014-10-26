import numpy as np
import matplotlib.pyplot as plt
import seaborn as ssn

from scipy.fftpack import fft
import scipy.constants as cc

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

 
def dig_err(cov, i): # returns the significant digit of the error
    dx = np.sqrt(cov[i,i])
    digit = -int(np.floor(np.log10(dx)))    
    if (dx * 10**digit) < 3.5:
        digit += 1
    return digit
    
def dig_val(x):     # returns the last significant digit of a value (error convention...)
    digit = -int(np.floor(np.log10(abs(x))))    
    if (x * 10**digit) < 3.5:
        digit += 1
    return digit

def val_str(x,Sx):
    dig = dig_val(Sx)
    return ("%."+str(dig)+"f \pm %."+str(dig)+"f")%(x,Sx)

def magnetic():
    R = [51.46,100.8,300.8,510.6,1000]
    U = [2.53, 2.67, 2.70, 2.70, 2.71]
    R = np.array([uc.ufloat(k,0.01*k) for k in R])
    r = uc.ufloat(2.0,0.25)
    z = uc.ufloat(3.0,0.2)
    U = np.array([uc.ufloat(k,0.01) for k in U])
    B = cc.mu_0 * U * r**2 / (2 * R * z**3)  *10**9
    p = np.pi * r**2 * U / R * 1000
    print("magnetic fields")
    for k in B:
        print("%s"%val_str(k.n,k.s))
    print("p dipoles")
    for k in p:
        print("&$%.1f \pm %.1f$"%(k.n,k.s))

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
def B_res(plot,polarplot):
    names  =  [ ("npy/4.%d_HM1508-2"%i,"4_%d"%i) for i in range(1,5+1)]
    #names  =  [ ("npy/4.1_HM1508-2","4.1")]
    for name,shortname in names:
        t = np.load(name+"_t.npy")[50:-50]
        A = np.load(name+"_channelA.npy")[50:-50]
        A -= (np.max(A)+np.min(A))/2

        # smoothing the data in order to get a stable fit.
        # Data which 
        A_smooth = savitzky_golay(A, 101, 2)

        # Define model function to be used to fit to the data above:
        def sin_fit(x, A,w,phi):
            return A*np.sin(w*x- phi) 

        # p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
        A0 = np.abs((np.max(A_smooth) - np.min(A_smooth))/2)

        

        p0 = [A0, 0.83, 1.6]

        sigma = t*0 + A0 * 0.03  
        p0,cov0 = curve_fit(sin_fit, t, A_smooth, p0=p0, sigma = sigma , absolute_sigma = True)

        p,cov = curve_fit(sin_fit, t, A, p0=p0)

        
        if p[0] < 0 :
            p[2] += np.pi
            p[0] *= -1

        if polarplot:
            # Getting A_mean
            w = p[1]
            f = w / (2 * np.pi)
            T = 1 / f 
            Ti = np.argmin(np.abs(t-T))


            A_mean = np.mean(A[Ti:2*Ti])


            F = 9.3
            si = 50
            T_end = 6

            Bx = (A[Ti:T_end*Ti] - A_mean)*np.cos(w * t[Ti:T_end*Ti])/si * F * 100
            By = (A[Ti:T_end*Ti] - A_mean)*np.sin(w * t[Ti:T_end*Ti])/si * F * 100
            
            fig = plt.figure()
            ax  = plt.subplot(111)
            plt.scatter(Bx,By, alpha = 0.5)
            plt.xlim(min(Bx),max(Bx))
            plt.ylim(min(By),max(By))
            plt.show()



        if plot:

            fig = plt.figure()
            ax  = plt.subplot(111)


            F = uc.ufloat(9.3,0.1)
            si = uc.ufloat(50,1)

            p_uc = uc.correlated_values(p, cov)
            F = uc.ufloat(9.3,0.1)
            si = uc.ufloat(50,1)

            Bz = p_uc[0]  /si * F * 100
            z = uc.ufloat(3.0,0.2)/1000
            pp  = 2 *np.pi * Bz * z**3 / cc.mu_0 
            #print("&$%s$"%val_str(1000*p_uc[0].n,1000*p_uc[0].s))
            #print("&$%s$"%val_str(Bz.n,Bz.s))
            print("&%s$"%val_str(pp.n,pp.s))
            


            props = dict(boxstyle='round, pad=0.9', facecolor='white', alpha=0.9, edgecolor = 'white')
            textstr = r"\begin{eqnarray*}"
            textstr += r"&&A \sin(\omega t - \phi) \\ "
            textstr += r"A &=& (%s)V \\ "%(val_str(p[0],p_uc[0].s))
            textstr += r"\omega &=& (%s)\mathrm{rad}\cdot s^{-1}\\"%(val_str(p[1],p_uc[1].s))
            textstr += r"\phi &=& (%s) \mathrm{rad}"%(val_str(p[2],p_uc[2].s))
            textstr += r"\end{eqnarray*}"


            box = ax.text(0.5,0.35, textstr, transform=ax.transAxes, fontsize=16, va='top', bbox=props)
            plt.ylabel("Voltage / V",fontsize = 16)
            plt.xlabel("time in $ns$", fontsize = 16)
            plt.grid(True)

            plt.scatter(t,A, alpha = 0.1,label="data")
            plt.ylim(min(A),max(A))
            plt.xlim(min(t),max(t)/4)
        
            #plt.plot(t, A_smooth)
            plt.plot(t, sin_fit(t,*p),label="least squares fit")
            plt.xlabel("time $t$",fontsize = 14)
            plt.ylabel("Amplitude $U$ in Volt", fontsize = 14)
            make_fig(fig,0,0,name="fit"+shortname)

def B_squid(plot,polarplot):
    names  =  [ ("npy/5.%d_HM1508-2"%i,"5_%d"%i) for i in range(1,5+1)]
    #names  =  [ ("npy/5.2_HM1508-2","5.2")]
    freqs = [0.83,0.83,0.83,0.83,0.43]
    u = -1
    for name,shortname in names:
        u+=1
        t = np.load(name+"_t.npy")[50:-50]
        A = np.load(name+"_channelA.npy")[50:-50]
        A -= (np.max(A)+np.min(A))/2

       # smoothing the data in order to get a stable fit.
        # Data which 
        A_smooth = savitzky_golay(A, 21, 3)

        # Define model function to be used to fit to the data above:
        def sin_fit(x, A,w,phi):
            return A*np.sin(w*x- phi) 

        # p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
        A0 = np.abs((np.max(A_smooth) - np.min(A_smooth))/2)

        

        p0 = [A0, freqs[u], 1.6]

        sigma = t*0 + A0 * 0.03  
        p0,cov0 = curve_fit(sin_fit, t, A_smooth, p0=p0, sigma = sigma , absolute_sigma = True)

        p,cov = curve_fit(sin_fit, t, A, p0=p0)
        if p[0] < 0 :
            p[2] += np.pi
            p[0] *= -1
        p_uc = uc.correlated_values(p, cov)

        if polarplot:
            # Getting A_mean
            w = p[1]
            f = w / (2 * np.pi)
            T = 1 / f 
            Ti = np.argmin(np.abs(t-T))


            A_mean = np.mean(A[Ti:2*Ti])


            F = 9.3
            si = 50
            T_end = 6

            Bx = (A[Ti:T_end*Ti] - A_mean)*np.cos(w * t[Ti:T_end*Ti])/si * F * 100
            By = (A[Ti:T_end*Ti] - A_mean)*np.sin(w * t[Ti:T_end*Ti])/si * F * 100
            
            fig = plt.figure()
            ax  = plt.subplot(111)
            plt.scatter(Bx,By, alpha = 0.5)
            plt.xlim(min(Bx),max(Bx))
            plt.ylim(min(By),max(By))
            plt.show()

        if plot:

            fig = plt.figure()
            ax  = plt.subplot(111)


            props = dict(boxstyle='round, pad=0.9', facecolor='white', alpha=0.9, edgecolor = 'white')
            textstr = r"\begin{eqnarray*}"

            if u in [0,1,2]:  
                plt.plot(t, A_smooth, label= "Filtered curve")
                textstr += r"F(t) &=&A \cdot f(\omega t + \phi) \\ "
                textstr += r"A &=& (%s)V \\ "%(val_str(A0,A0*0.1))
                A0 = uc.ufloat(A0,A0*0.1)
            else:
                plt.plot(t, sin_fit(t,*p),label="least squares fit")
                textstr += r"&&A \sin(\omega t - \phi) \\ "
                textstr += r"A &=& (%s)V \\ "%(val_str(p[0],p_uc[0].s))
                A0 = uc.ufloat(p0[0],p_uc[0].s)

            textstr += r"\omega &=& (%s)\mathrm{rad}\cdot s^{-1}\\"%(val_str(p[1],p_uc[1].s))
            textstr += r"\phi &=& (%s) \mathrm{rad}"%(val_str(p[2],p_uc[2].s))
            textstr += r"\end{eqnarray*}"


            F = uc.ufloat(9.3,0.1)
            si = uc.ufloat(50,1)

            Bz = A0  /si * F * 100
            z = uc.ufloat(3.0,0.2)/1000
            pp  = 2 *np.pi * Bz * z**3 / cc.mu_0 
            #print("&$%s$"%val_str(A0.n,A0.s))
            #print("&$%s$"%val_str(Bz.n,Bz.s))
            #print("&$%s$"%val_str(pp.n,pp.s))
            




            box = ax.text(0.5,0.35, textstr, transform=ax.transAxes, fontsize=14, va='top', bbox=props)
            plt.ylabel("Voltage / V",fontsize = 16)
            plt.xlabel("time in $ns$", fontsize = 16)
            plt.grid(True)

            plt.scatter(t,A, alpha = 0.1,label="data")
            plt.ylim(min(A),max(A))
            plt.xlim(min(t),max(t)/4)
            plt.legend(fontsize = 14, loc =2)
            make_fig(fig,0,0,name="fit"+shortname)
B_squid(False,True)
