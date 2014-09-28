import numpy as np
import matplotlib.pyplot as plt

input_dir = "../data_faraday/"

for i in range(4):
    a = np.load(input_dir +"a_%i.npy"%(i+1))
    I = np.load(input_dir + "i_%i.npy"%(i+1))
    I_fit = np.linspace(min(I),max(I),100) 
    S_a = 0.5
    weights = a*0 + 1/S_a 
    p, cov= np.polyfit(I,a, 1, full=False, cov=True, w=weights)

    a_fit = np.polyval(p, I_fit)
    S_I = 0.05
    plt.figure()
    plt.xlim(min(I), max(I))
    plt.plot(I_fit,a_fit)
    plt.fill_between(I_fit, \
                np.polyval(p - np.sqrt(np.diag(cov)), I_fit), \
                np.polyval(p + np.sqrt(np.diag(cov)), I_fit), \
                facecolor="r", color="b", alpha=0.3 )

    plt.errorbar(I,a,yerr=S_a,xerr=S_I, fmt=".")

    plt.grid(True)
    plt.xlabel("Current $I(\\alpha)$")
    plt.ylabel("angle $\\alpha$")
    plt.show()

    
