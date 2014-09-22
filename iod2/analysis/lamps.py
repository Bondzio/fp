"""
Analyse the spectrum of the Na and Hg lamps as well as plotting the reduced spectrum of the halogen lamp
"""
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

def sa(name):
    file = open("../data/"+name+".txt","r",encoding="latin-1")
    x = file.read()
    file.close()
    x = (x.split("\n"))[17:-2]
    l = []
    I = []
    for q in x:
        b = (re.sub(",",".",q)).split("\t")
        l += [float(b[0])]
        I += [float(b[1])]
    l = np.array(l)
    I = np.array(I)
    return l,I
#############################################################################################

plt.close('all')
colors = ["orange"] + 3 *  ["steelblue"]
golden_ratio = (1 + np.sqrt(5)) / 2
l1 = 7.0
figsize = [golden_ratio * l1, l1]

# Plot all characteristic lines
fig0 = plt.figure(figsize=figsize)
l,I = sa("na_01")
plt.plot(l,I, label="Na lamp", color=colors[0])
l,I = sa("hg_02")
plt.plot(l,I, label="Hg lamp", color=colors[1])
plt.xlim(l.min()*0.99,l.max()*1.01)
plt.xlabel("Wavelength $\lambda$")
plt.ylabel("Relative Intensity $I(\lambda)$")
plt.grid(True)
plt.legend(loc=2)
#plt.title("Spectrum of Hg and Na lamp")
fig0.savefig("figures/spectrum_all.pdf")

# Searching the maxima from data, na lamp
for k in range(4):
    name = ["na_01", "hg_01", "hg_02", "hg_02"][k]
    title = "Spectrum of " + ["Na", "Hg", "Hg", "Hg"][k] + " lamp - detail to define maximum"
    figname = ["na", "hg1", "hg2", "hg3"][k] + "_max.pdf"
    l,I = sa(name)
    b_max = 10   # maximum number of neighbours
    l_lower = [587, 432, 543, 574][k]
    l_upper = [591, 438, 548, 581][k] # choose intervall manually 
    l_red = l[np.where((l_lower < l) * (l < l_upper))]
    I_red = I[np.where((l_lower < l) * (l < l_upper))]
    maxi = argrelextrema(I_red, np.greater)[0]
    for maxii in maxi:
        b_max = 4
        Imax = I_red[maxii]
        if maxii <= b_max or maxii >= len(I_red) - b_max:
            maxi = np.delete(maxi, np.where(maxi == maxii))
        else:
            for i in range(2, b_max):
                if I_red[maxii - i] > Imax or I_red[maxii + i] > Imax:
                    maxi = np.delete(maxi, np.where(maxi == maxii))
                    break
    if not k == 1:
        maxi = np.delete(maxi, 0)      # this minimum is not useful!
    l_max = l_red[maxi]
    I_max = I_red[maxi]
    dl = 0.4

    fig1 = plt.figure(figsize=figsize)
    ax = plt.subplot(111)
    ax.plot(l,I, color=colors[k])
    ax.grid(True)
    if k == 0:
        for i in range(len(maxi)):
            max_label = "$\lambda_\mathrm{max, %i} = %.1f \pm %.1f \mathrm{nm}$" \
                %(i + 1, dl, l_max[i])
            ax.plot([l_max[i]]* 2, [0, I_max[i]], 'k--', label=max_label)
    if k != 0:
        if k == 1: nl_hg = 0
        for i in range(len(maxi)):
            nl_hg += 1
            max_label = "$\lambda_\mathrm{max, %i} = %.1f \pm %.1f \mathrm{nm}$" \
                %(nl_hg, dl, l_max[i])
            ax.plot([l_max[i]]* 2, [0, I_max[i]], 'k--', label=max_label)
    ax.set_xlabel("Wavelength $\lambda$")
    ax.set_ylabel("Relative Intensity $I(\lambda)$")
    ax.grid(True)
    ax.legend(loc=2)
    ax.set_xlim(l_lower, l_upper)
 #   ax.set_title(title)
    fig1.savefig("figures/" + figname)


####################################################################################################
# intensity and wavelength for halogene
input_dir = 'data_npy/'
f_halog_i = input_dir + 'halogen_02_I.npy'
f_halog_l = input_dir + 'halogen_02_l.npy'
halog_i = np.load(f_halog_i)    # intensity in counts
halog_l = np.load(f_halog_l)   # wavelenght

# Plot spetrum without corrections
fig2 = plt.figure(figsize=figsize)
ax = plt.subplot(111)
ax.plot(halog_l, halog_i, '-')    # does not fit the intensitiy with the iodine!
ax.set_xlim = ([0, 2050])
ax.set_xlabel("Wavelength $\lambda$ / nm")
ax.set_ylabel("Intensity $I(\sigma)$ / counts" )
ax.grid(True)
title_spectrum = "Spectrum halogene lamp (not corrected)"
#ax.set_title(title_spectrum)
fig2.savefig("figures/halogen_02.pdf")
fig2.show()

l_min = 500     # minimum wavelenght at which to search for minima
l_max = 620     # maximum wavelenght at which to search for minima
l_range = (l_min < halog_l) & (halog_l < l_max)
halog_l = halog_l[l_range]
halog_i = halog_i[l_range]
# correction for air, linearly interpolated with  
# (n_l - 1) * 10 **(4) = [2.79, 2.78, 2.77, 2.76]  at corrisponding wavelenght [500, 540, 600, 680] nm   
l_lower = np.array([500, 540, 600])         # lower bounds for linear interpolation
l_upper = np.array([540, 600, 680])         # upper bounds for linear interpolation
for i in range(len(halog_l)):
    l = halog_l[i]                 
    j = np.where((l_lower < l) * (l <= l_upper))[0][0]  # find the correct intervall
    n_l = (2.79 - 0.01 * j) * 10 ** (-4) + 1  - (l - l_lower[j]) / (10 ** 6 * (l_upper[j] - l_lower[j]))    # diffraction index
    halog_l[i] = n_l * l
halog_cm = 1 / halog_l * 10 ** 7
fig3 = plt.figure(figsize=figsize)
ax = plt.subplot(111)
ax.plot(halog_cm,halog_i, '-')    # does not fit the intensitiy with the iodine!
ax.set_xlabel("Wavenumber $\sigma$ / $\mathrm{cm^{-1}}$")
ax.set_ylabel("Intensity $I(\sigma)$ / counts" )
ax.grid(True)
title_spectrum = "Spectrum halogene lamp, reduced and corrected for diffraction of air"
#ax.set_title(title_spectrum)
fig3.savefig("figures/halogen_red.pdf")
fig3.show()
