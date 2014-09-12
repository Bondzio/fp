import re
import numpy as np
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

import matplotlib.pyplot as plt
fig = plt.figure()
def make_plot(title,name):
    l,I = sa(name)
    plt.plot(l,I)
    plt.grid(True)
    plt.xlim(l.min()*0.99,l.max()*1.01)
    plt.title(title)
    plt.xlabel("Wavelength $\lambda$")
    plt.ylabel("Relative Intensity $I(\lambda)$")
    plt.savefig(name)
    fig.clear()

make_plot("Spectrum of Halogenlamp (a)","halogen_01")
make_plot("Spectrum of Halogenlamp (b)","halogen_02")
make_plot("Spectrum of Natriumlamp","natrium1")
make_plot("Spectrum of Quecksilberlamp (a)","hg_01")
make_plot("Spectrum of Quecksilberlamp (b)","hg_02")
make_plot("Spectrum of Quecksilberlamp (c)","hg_03")
make_plot("$J_2$-Molecule (a)","jod_1")
make_plot("$J_2$-Molecule (b)","jod_2")
make_plot("$J_2$-Molecule (c)","jod_3")
make_plot("Spektrum of Paper (a)","papier_einlagig_01")
make_plot("Spektrum of Paper (b)","papier_zweilagig_01")
