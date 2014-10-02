import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.signal import blackman


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
k = 1
U2 = np.load("npy/U2_%03d.npy"%k)
t  = np.load("npy/T_%03d.npy"%k)
Tmax = np.max(t)
N = len(U2) 
N2 = N-2
T = Tmax/N*1000 
w   = blackman(N)

U2f = fft(U2*w)

U2acf = acf(U2,N2)

w2   = blackman(N2)
U2acff = fft(U2acf*w2) 
tf = np.linspace(0.0, 1.0/(2.0*T), N/2)
tf2 = np.linspace(0.0, 1.0/(2.0*T), (N2)/2)

f, axarr = plt.subplots(2, 2, figsize = (11.5,6))
axarr[0, 0].plot(t*1000,U2)
axarr[0, 0].set_title('Normal Plot')

axarr[0, 1].semilogy(tf, 2.0 / N * np.abs(U2f[0:N/2])**2)
axarr[0, 1].set_title('Fourier Transform with blackman')

axarr[1, 0].plot(t[2:]*1000,U2acf)
axarr[1, 0].set_title('Autocorrelation function')

axarr[1, 1].semilogy(tf2, 2.0 / N2 * np.abs(U2acff[0:N2/2])**2)
axarr[1, 1].set_title('Fourier transform of Autocorrelation function')

plt.show()


