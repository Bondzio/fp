import numpy as np
import matplotlib.pyplot as plt
import seaborn as ssn

from scipy.optimize import curve_fit

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern Roman']
rcParams['text.usetex'] = True
rcParams['figure.autolayout'] = True

def plot_all():
    """
    Winter seclusion -
    Listening, that evening,
    To the rain in the mountain.

    - Issa 

    """
    names = [ "npy/2.%d_HM1508-2"%i for i in range(1,5+1)]
    names +=  ["npy/3.1_HM1508-2"]
    for name in names:
        t = np.load(name+"_t.npy")
        A = np.load(name+"_channelA.npy")


        # Define model function to be used to fit to the data above:
        def sin_fit(x, *p):
            a,b,c,d= p
            return a + b*np.sin(c*x+ d)

        # p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
        p0 = [0.1, 1., 1., 0.1]

        coeff, var_matrix = curve_fit(sin_fit, t, A, p0=p0)



        fig = plt.figure()
        plt.scatter(t,A, alpha = 0.4)
        plt.plot(t, sin_fit(t,*coeff))
        #plt.xlabel("time $t$",fontsize = 14)
        #plt.ylabel("Amplitude $U$ in Volt", fontsize = 14)
        plt.show()

plot_all()
