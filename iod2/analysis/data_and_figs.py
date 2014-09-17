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
    np.save("data_npy/"+name+"_l",l)
    np.save("data_npy/"+name+"_I",I)
    plt.plot(l,I)
    plt.grid(True)
    plt.xlim(l.min()*0.99,l.max()*1.01)
    plt.title(title)
    plt.xlabel("Wavelength $\lambda$")
    plt.ylabel("Relative Intensity $I(\lambda)$")
    plt.savefig("figures/"+name+".pdf")
    fig.clear()

make_plot("Spectrum of halogen lamp (a)","halogen_01")
make_plot("Spectrum of halogen lamp (b)","halogen_02")
make_plot("Spectrum of Na lamp","na_01")
make_plot("Spectrum of Hg lamp (a)","hg_01")
make_plot("Spectrum of Hg lamp (b)","hg_02")
make_plot("Spectrum of Hg lamp (c)","hg_03")
make_plot("$J_2$-Molecule (a)","iodine_01")
make_plot("$J_2$-Molecule (b)","iodine_02")
make_plot("$J_2$-Molecule (c)","iodine_03")
make_plot("Spectrum of paper (a)","paper_one_sheet_01")
make_plot("Spectrum of paper (b)","paper_two_sheets_01")

