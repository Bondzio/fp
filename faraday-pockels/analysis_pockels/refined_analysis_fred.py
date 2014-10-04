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

############################################################################################################
# Two subplots, unpack the axes array immediately
show_fig = 1
save_fig = 0
plt.close('all')
fig_dir = '../figures/'

# Examplary cases: k = 8, k = 12
ks = range(8, 19)       # plots
U_DC = [135.5,137.0, 138.0,138.5,139.0,139.5,140.0,140.5,141.0,142.0,144.0]
k_example = 9
freq1 = np.zeros(len(ks))
freq2 = np.zeros(len(ks))
fig1, axarr = plt.subplots(1,2, figsize = (12,5))
fig2, ax2 = plt.subplots(1, 1, figsize = (12,7))
fig3, ax3 = plt.subplots(1, 2, figsize = (12,6))
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
        if not save_fig:
            fig2.suptitle('Examplary plot for $U_\mathrm{DC} = %.1f$ V'%U_DC[i])
            fig3.suptitle('Fourier transform for $U_\mathrm{DC} = %.1f$ V'%U_DC[i])
        ymin = -0.015
        ymax = 0.025
        ax2.plot(t, U2, label='$U_\mathrm{out}(t)$')                # plots of cut-off output
        ax2.plot(t_old, U1*0.2, label='$0.2 \cdot U_\mathrm{in}(t)$')                          # plotting the incoming signal
        ax2.plot(t_old[[lower, lower]], [ymin, ymax], 'k--', label='$\mathrm{cut-off}$') # plotting the cut-offs
        ax2.plot(t_old[[upper, upper]], [ymin, ymax], 'k--') # plotting the cut-offs
        ax2.legend(loc=4)
        ax2.set_ylim([ymin, ymax])
        ax2.set_xlabel('$t \, / \, \mathrm{ms}$')
        ax2.set_ylabel('$U(t) \, / \, \mathrm{V}$')
        ax2.legend()

        ax3[0].semilogy(f, psd_U2)           
        ax3[0].set_xlim([0, 70])
        ax3[0].set_xlabel('$\\nu \, / \, \mathrm{kHz}$')
        ax3[0].set_ylabel('$|\mathrm{FFT}\left[U_\mathrm{out} \left(\\nu\\right)\\right]|^2 \, / \, \mathrm{V}^2$')
        ax3[1].semilogy(f, psd_U2)           
        ax3[1].set_xlim([0, 2])
        ax3[1].set_ylim([10**-7, 0.05])
        ax3[1].set_xlabel('$\\nu \, / \, \mathrm{kHz}$')
        ax3[1].set_ylabel('$|\mathrm{FFT}\left[U_\mathrm{out} \left(\\nu\\right)\\right]|^2 \, / \, \mathrm{V}^2$')

axarr[0].plot([f[4], f[4]], [10**-7, 1], 'k-.', label='$\\nu_1 = %.2f\, \mathrm{kHz}$'%(f[4]))
axarr[0].plot([f[8], f[8]], [10**-7, 1], 'k--', label='$\\nu_1 = %.2f\, \mathrm{kHz}$'%(f[8]))
axarr[0].set_xlim([0, 2])
axarr[0].set_ylim([10**-7, 0.05])
axarr[0].set_xlabel('$\\nu \, / \, \mathrm{kHz}$')
axarr[0].set_ylabel('$|\mathrm{FFT}\left[U_\mathrm{out} \left(\\nu\\right)\\right]|^2 \, / \, \mathrm{V}^2$')
axarr[0].legend(loc=4)

axarr[1].semilogy(U_DC, freq1, 'b.', label='$\\nu_1 = %.2f\, \mathrm{kHz}$'%(f[4]))
axarr[1].semilogy(U_DC, freq2, 'r.', label='$\\nu_2 = %.2f\, \mathrm{kHz}$'%(f[8]))
axarr[1].set_ylim([10**-4, 0.05])
axarr[1].set_xlabel('$U_\mathrm{DC} \, / \, \mathrm{V}$')
axarr[1].set_ylabel('$|\mathrm{FFT}\left[U_\mathrm{out} \left(\\nu_i\\right)\\right]|^2 \, / \, \mathrm{V}^2$')
axarr[1].legend(loc=4)
if not save_fig:
    axarr[0].set_title('PSD for all U_DC')
    axarr[1].set_title('Amplitude at %.2f and %.2f kHz'%(f[4], f[8]))

if show_fig:
    fig1.show()
    fig2.show()
    fig3.show()
if save_fig: 
    fig1.savefig(fig_dir + "fourier_all.pdf")
    fig2.savefig(fig_dir + "cut_off_example.pdf")
    fig3.savefig(fig_dir + "fft_example.pdf")



