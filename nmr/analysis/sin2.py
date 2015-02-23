import matplotlib.pyplot as plt
import seaborn as ssn
import numpy as np

import matplotlib.pyplot as plt
import prettyplotlib as ppl


def plot_sin():
    fig = plt.figure()
    ax = plt.subplot(111)

    a = np.linspace(0,10,1000)
    b = np.linspace(0,10,1000)
    A,B = np.meshgrid(a,b)


    t = 1
    Z = A*np.sin(t*np.pi*B/A) / (A**2 - B**2)

    plt.imshow(Z)
    plt.show()

fig = plt.figure()
ax = plt.subplot(111)

w = 100 

t = np.linspace(0,2,1000)
U = t - np.floor(t) + 0.1 * np.sin(w*t)

plt.plot(t,U)
plt.xlabel("$t/T$",fontsize = 14)
plt.ylabel("$U/U_{sz}$",fontsize = 14)
plt.show()

