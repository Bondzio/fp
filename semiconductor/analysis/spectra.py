import numpy as np
import pylab as plt
from matplotlib import rcParams
from scipy.optimize import curve_fit
import scipy.constants as co
import uncertainties as uc
import uncertainties.unumpy as un
from scipy.signal import argrelextrema as ext
import seaborn as sns

fontsize_labels = 22    # size used in latex document
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern Roman']
rcParams['text.usetex'] = True
rcParams['figure.autolayout'] = True
rcParams['font.size'] = fontsize_labels
rcParams['axes.labelsize'] = fontsize_labels
rcParams['xtick.labelsize'] = fontsize_labels
rcParams['ytick.labelsize'] = fontsize_labels
rcParams['legend.fontsize'] = fontsize_labels
rcParams['figure.figsize'] = (2*6.2, 2*3.83)  # in inches; width corresponds to \textwidth in latex document (golden ratio)

plt.close("all")
show_fig = True
save_fig = False # see below
if not save_fig:
    rcParams['figure.figsize'] = (15, 8)  # in inches; width corresponds to \textwidth in latex document (golden ratio)
save_coeff = False # do ONLY save, if scipy 0.14. is in use...
fig_dir = "../figures/"
npy_dir = "./data_npy/"


for sample_name in ["Co", "Am"]:
    for detector_name in ["CdTe", "Si"]:
        npy_file  = npy_dir + "spectra_" + sample_name + "_" + detector_name + ".npy"
        histo = np.load(npy_file)
