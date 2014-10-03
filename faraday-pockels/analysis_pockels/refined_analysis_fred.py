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
ks = range(8, 19)
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
    N2 = N-2
    T = Tmax/N
    #w   = blackman(N)
    w = np.ones(N)

    U2f = fft(U2)

    U2f_abs = (2.0 / N * np.abs(U2f[0:N/2])**2)
    tf = np.linspace(0.0, 1.0/(2.0*T), N/2)
    axarr[0].semilogy(tf, U2f_abs)
    freq1[i] = U2f_abs[4]
    freq2[i] = U2f_abs[8]

    mx = 4
    my = 3
    m = mx * my
    if i % m == 0:
        f2, axarr2 = plt.subplots(mx, my, figsize = (15,11))
        f2.suptitle('Plots for k = %i, %i, %i, %i'%(k, k+1, k+2, k+3))
    ax = axarr2[int(i%m/my), (i%m)%my]
    ax.plot(t,U2, label='k = %i'%k)
    ax.plot(t_old,U1*0.2)
    ax.plot(t_old[[lower, upper]],U1[[lower, upper]]*0.2, 'o')
    ax.legend(loc=4)
axarr[0].set_title('Fourier Transform')
axarr[0].set_ylim([0, 0.05])
axarr[0].set_xlim([0, 2])

axarr[1].semilogy(ks, freq1, label='0.5kHz')
axarr[1].semilogy(ks, freq2, label='1.0kHz')
axarr[1].set_title('Amplitude at 0.5 and 1.0 kHz')
axarr[1].set_ylim([0, 0.05])
axarr[1].legend()


for i, k in enumerate(ks):
    # Using the auto correlation function of U2
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


