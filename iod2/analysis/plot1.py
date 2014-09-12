import numpy as np
import matplotlib.pyplot as plt
I = np.load("data_npy/jod_1_I.npy")
L = np.load("data_npy/jod_1_l.npy")

n = 4
minima = []
k_min = np.max(np.where(L < 512))
for k in range(k_min,len(I)):
    if np.all(I[k] < I[k-n:k]) and np.all(I[k] < I[k+1:k+n+1]):
        minima+=[(L[k],I[k])]
plt.figure()
plt.scatter(L,I)
plt.plot(L,I)
m = np.max(I) * 1.3
for (l,i) in minima:
    t = np.linspace(i,m,1000)
    plt.plot(t*0 + l, t)
    
plt.show()
