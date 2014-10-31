import numpy as np
import pylab as plt
from matplotlib import rcParams
from scipy.optimize import curve_fit
import scipy.constants as co
import uncertainties as uc
import uncertainties.unumpy as un
from scipy.signal import argrelextrema as ext
import seaborn as sns
import stat1 as st
 
fontsize_labels = 22 # size used in latex document
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern Roman']
rcParams['text.usetex'] = True
rcParams['figure.autolayout'] = True
rcParams['font.size'] = fontsize_labels
rcParams['axes.labelsize'] = fontsize_labels
rcParams['xtick.labelsize'] = fontsize_labels
rcParams['ytick.labelsize'] = fontsize_labels
rcParams['legend.fontsize'] = fontsize_labels
rcParams['figure.figsize'] = (2*6.2, 2*3.83) # in inches; width corresponds to \textwidth in latex document (golden ratio)
 
plt.close("all")
show_fig = True
save_fig = False
#do_plot_raw = True
#do_write_to_tex = True
#do_plot_gaussians = True
#do_plot_mobility = True
#do_plot_life_time = True
#do_plot_diffusion = True
fig_dir = "../figures/"
npy_dir = "./data_npy/"
 
props = dict(boxstyle='round', facecolor='white', edgecolor = 'white', alpha=0.7)
 
# file names
 
for sample_name in ["Co", "Am"]:
    for detector_name in ["CdTe", "Si"]:
        a = 1
 
sample_name = "Co"
detector_name = "CdTe"
suffix = "_" + sample_name + "_" + detector_name
npy_file = npy_dir + "spectra" + suffix + ".npy"
a = np.load(npy_file)
a = a[:700]
a = list(a)
fig1, ax1 = plt.subplots(1, 1)
#fig1.suptitle("Detector")
#n, bins, patches = ax1.hist(histo, bins=200)
