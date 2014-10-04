import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.signal import blackman
from scipy.signal import wiener

def save_data(name,filename):
    f = open("../data_pockels/"+name+".tab", "r", encoding = "latin-1")
    T  = []
    U1 = []
    U2 = []
    for i,line in enumerate(f): 
        if i > 1:
            t,u1,u2,dump = line.split("\t")
            T  += [float(t)] 
            U1 += [float(u1)]
            U2 += [float(u2)]
    T  = np.array(T)
    U1 = np.array(U1)
    U2 = np.array(U2)
    np.save("npy/T_"+filename,T)
    np.save("npy/U1_"+filename,U1)
    np.save("npy/U2_"+filename,U2)
#this only needs to be done once
def save_all():
    U_DC = [141.0,140.5,140.0, 139.5,139.0, 138.5, 138.0, 135.0] 
    k = 0
    for i in range(7,14+1):
        save_data("2.3.sinus%02d"%i, "%03d"%k)
        k+=1
        
    U_DC = [135.5,137.0, 138.0,138.5,139.0,139.5,140.0,140.5,141.0,142.0,144.0,137.0,137.5,138.0,138.5, 139.0,139.5, 140.0,140.5,141.0]
    for i in range(1,20+1):
        save_data("2.4.sinus%02d"%i,"%03d"%k)
        k+=1

def acf(x, length=20):
    return np.array([1]+[np.corrcoef(x[:-i], x[i:])[0,1] for i in range(1, length)])


# Two subplots, unpack the axes array immediately
plt.close('all')
# Examplary cases: k = 8, k = 12
ks = range(8, 19)       # plots
k_example = 9
freq1 = np.zeros(len(ks))
freq2 = np.zeros(len(ks))
f, axarr = plt.subplots(1,2, figsize = (12,5))
for i, k in enumerate(ks):
    U1 = np.load("npy/U1_%03d.npy"%k)
    U2 = np.load("npy/U2_%03d.npy"%k)
    t  = np.load("npy/T_%03d.npy"%k) * 1000
    t_old  = np.copy(t)
    # in order to apply fourier anaylsis, we need to specify the range first.
    # we use the first and the fifth minimum of U1
    range_lower = np.where((t > 1.0) * (t < 1.5))[0]
    range_upper = np.where((t > 8.5) * (t < 9.5))[0]
    lower = np.argmin(U1[range_lower]) + np.min(range_lower)
    upper = np.argmin(U1[range_upper]) + np.min(range_upper)
    U2 = U2[lower: upper]
    t = t[lower: upper]
    U1n = U1[lower:upper]

    Tmax = np.max(t)
    N = len(U2) 
    T = Tmax/N

    U2f = fft(U2)       # fourier transfor of U2

    psd_U2 = (2.0 / N * np.abs(U2f[0:N/2])**2)     # power spectral density
    f = np.linspace(0.0, 1.0/(2.0*T), N/2)      # frequencies
    axarr[0].semilogy(f, psd_U2)           
    freq1[i] = psd_U2[4]
    freq2[i] = psd_U2[8]

    if k == k_example:
        f3, ax3 = plt.subplots(1, 1, figsize = (15,11))
        f3.suptitle('Examplary plot for k = %i'%k)
        ymin = -0.015
        ymax = 0.025
        ax3.plot(t, U2, label='$U_\mathrm{out}(t)$')                # plots of cut-off output
        ax3.plot(t_old, U1*0.2, label='$0.2 U_\mathrm{in}(t)$')                          # plotting the incoming signal
        ax3.plot(t_old[[lower, lower]], [ymin, ymax], 'k--', label='cut-off') # plotting the cut-offs
        ax3.plot(t_old[[upper, upper]], [ymin, ymax], 'k--') # plotting the cut-offs
        ax3.legend(loc=4)
        ax3.set_ylim([ymin, ymax])
        ax3.set_xlabel('$t$')
        ax3.set_ylabel('$U(t)$')
        ax3.legend()

        f4, ax4 = plt.subplots(1, 2, figsize = (11,6))
        f4.suptitle('Fourier transform for k = %i'%k)
        ax4[0].semilogy(f, psd_U2)           
        ax4[0].set_xlim([0, 70])
        ax4[0].set_xlabel('$\\nu$')
        ax4[0].set_ylabel('$\mathrm{|FFT(U_\mathrm{out})|^2(\\nu)}$')
        ax4[1].semilogy(f, psd_U2)           
        ax4[1].set_xlim([0, 2])
        ax4[1].set_ylim([10**-7, 0.05])
        ax4[1].set_xlabel('$\\nu$')
        ax4[1].set_ylabel('$\mathrm{|FFT(U_\mathrm{out})|^2(\\nu)}$')
"""
    mx = 4
    my = 3
    m = mx * my
    if i % m == 0:
        f2, axarr2 = plt.subplots(mx, my, figsize = (15,11))
        f2.suptitle('Plots for all k')
    ax = axarr2[int(i%m/my), (i%m)%my]
    ax.plot(t, U2, label='k = %i'%k)                # plots of cut-off output
    ax.plot(t_old, U1*0.2)                          # plotting the incoming signal
    ax.plot(t_old[[lower, upper]], U1[[lower, upper]]*0.2, 'o') # plotting the cut-offs
    ax.legend(loc=4)
"""


axarr[0].set_title('Fourier Transform')
axarr[0].plot([f[4], f[4]], [10**-7, 1], 'k-.', label='$\\nu_1 = %.2f\, \mathrm{kHz}$'%(f[4]))
axarr[0].plot([f[8], f[8]], [10**-7, 1], 'k--', label='$\\nu_1 = %.2f\, \mathrm{kHz}$'%(f[8]))
axarr[0].set_xlim([0, 2])
axarr[0].set_ylim([10**-7, 0.05])
axarr[0].set_xlabel('$\\nu$')
axarr[0].set_ylabel('$\mathrm{|FFT(U_\mathrm{out})|^2(\\nu)}$')
axarr[0].legend(loc=4)

axarr[1].semilogy(ks, freq1, 'b.', label='$\\nu_1 = %.2f\, \mathrm{kHz}$'%(f[4]))
axarr[1].semilogy(ks, freq2, 'r.', label='$\\nu_2 = %.2f\, \mathrm{kHz}$'%(f[8]))
axarr[1].set_title('Amplitude at %.2f and %.2f kHz'%(f[4], f[8]))
axarr[1].set_ylim([10**-4, 0.05])
axarr[1].set_xlabel('$k$')
axarr[0].set_ylabel('$\mathrm{|FFT(U_\mathrm{out})|^2(\\nu_i)}$')
axarr[1].legend(loc=4)


for i, k in enumerate(ks):
    # Using the auto correlation function of U2
    N2 = N-2
    U2acf = acf(U2,N2)
    w2   = blackman(N2)
    w2 = np.ones(N2)
    U2acff = fft(U2acf*w2) 
    tf2 = np.linspace(0.0, 1.0/(2.0*T), (N2)/2)
'''
    axarr[1, 0].plot(t[2:],U2acf)
    axarr[1, 0].set_title('Autocorrelation function')

    axarr[1, 1].semilogy(tf2, 2.0 / N2 * np.abs(U2acff[0:N2/2])**2)
    axarr[1, 1].set_title('Fourier transform of Autocorrelation function')
'''
plt.show()


