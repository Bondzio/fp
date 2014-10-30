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
save_fig = True # see below
if not save_fig:
    rcParams['figure.figsize'] = (15, 8)  # in inches; width corresponds to \textwidth in latex document (golden ratio)
save_coeff = False # do ONLY save, if scipy 0.14. is in use...
fig_dir = "../figures/"
npy_dir = "./data_npy/"

def polygon(x_grid, x_arr, y_arr):
    y_on_grid = x_grid * 0
    for i, x in enumerate(x_grid):
        x_less = x_arr < x
        x1 = x_arr[x_less][-1]
        y1 = y_arr[x_less][-1]
        x_greater = x_arr > x
        x2 = x_arr[x_greater][0]
        y2 = y_arr[x_greater][0]
        y_x = (y1 * (x - x1) + y2 * (x2 - x)) / (x2 - x1)
        y_on_grid[i] = y_x
    return y_on_grid

# error
angle_err = [0]*5
trans_err = [0]*5
absorp_err = [0]*5
for i in range(5):
    suffix = str(i + 1)
    npy_file = npy_dir + "band_gap_error_" + suffix + ".npy"
    angle_err[i], trans_err[i], absorp_err[i]  = np.load(npy_file)
    if angle_err[i][0] > angle_err[i][10]:
        angle_err[i], trans_err[i], absorp_err[i]  = angle_err[i][::-1], trans_err[i][::-1], absorp_err[i][::-1]

# the data is not defined on the same array of angles, we need to interpolate onto one grid!
# get the ends and length of the grid
lower_angle = max([angle_err[i][0] for i in range(5)])
upper_angle = min([angle_err[i][-1] for i in range(5)])
min_len = min([len(np.where((angle_err[i] > lower_angle) * (angle_err[i] < upper_angle))[0]) for i in range(5)])
# create a grid on a subset of the intersection of all three angles intervals
x_grid_err = np.linspace(lower_angle, upper_angle, min_len + 1) # "+1" to be able to cut of the edges
x_grid_err = x_grid_err[1:-1] # cut the edges
t = []
a = []
for i in range(5):
    trans_err[i] = polygon(x_grid_err, angle_err[i], trans_err[i])
    absorp_err[i] = polygon(x_grid_err, angle_err[i], absorp_err[i])
t_avg = np.average(trans_err, axis=0)
a_avg = np.average(absorp_err, axis=0)
t_errors = np.std(trans_err, axis=0)
a_errors = np.std(absorp_err, axis=0)

# plot error bars
fig, ax= plt.subplots(2, 1, sharex=True)
if not save_fig:
    fig.suptitle("Band Gap, Error")
fig.subplots_adjust(hspace=0)          # create overlap
xticklabels = ax[0].get_xticklabels()    
plt.setp(xticklabels, visible=False)    # hide the xticks of the upper plot

plot_t, = ax[1].plot(x_grid_err, t_errors, '-', alpha=0.8, label=(r'$\sigma(T_\mathrm{Si})$'))
next(ax[1]._get_lines.color_cycle)
plot_a, = ax[1].plot(x_grid_err, a_errors, '-', alpha=0.8, label=('$\sigma(A_\mathrm{Si})$'))

ax[0].errorbar(x_grid_err, trans_err[0], yerr=t_errors, fmt='.', label=(r'$T_\mathrm{Si}$'))
next(ax[0]._get_lines.color_cycle)
ax[0].errorbar(x_grid_err, absorp_err[0], yerr=a_errors, fmt='.', label=('$A_\mathrm{Si}$'))

xlims = -55, -25
ax[1].set_xlim(xlims)
ax[1].set_xlabel("angle / degrees")
ax[1].set_ylabel("$U$ / V")
ax[1].legend(loc=2)

ax[0].set_ylabel("$U$ / V")
ax[0].legend(loc=2)

if show_fig:
    fig.show()
if save_fig:
    file_name = "band_gap_error"
    fig.savefig(fig_dir + file_name + ".pdf")
    fig.savefig(fig_dir + file_name + ".png")


# Analysis with errors
sample_name = "Si"
for suffix in ["2"]:
    npy_file = npy_dir + "band_gap_" + sample_name + "_" + suffix + ".npy"
    angle, trans, absorp  = np.load(npy_file)
    if angle[0] > 0:
        angle, trans, absorp  = angle[::-1], trans[::-1], absorp[::-1]
for suffix in ["lamp"]:
    npy_file = npy_dir + "band_gap_" + sample_name + "_" + suffix + ".npy"
    angle_lamp, trans_lamp, dummy = np.load(npy_file)
    if angle_lamp[0] > 0:
        angle_lamp, trans_lamp, absorp_lamp = angle_lamp[::-1], trans_lamp[::-1], absorp_lamp[::-1]
for suffix in ["background"]:
    npy_file = npy_dir + "band_gap_" + sample_name + "_" + suffix + ".npy"
    angle_bg, trans_bg, absorp_bg  = np.load(npy_file)
    if angle_bg[0] > 0:
        angle_bg, trans_bg, absorp_bg  = angle_bg[::-1], trans_bg[::-1], absorp_bg[::-1]

#trans_real = (trans -  trans_background)/ trans_lamp
# the data is not defined on the same array of angles, we need to interpolate onto one grid!
# get the ends and length of the grid
lower_angle = max(angle[0], angle_bg[0], angle_lamp[0])
upper_angle = min(angle[-1], angle_bg[-1], angle_lamp[-1])
min_len = min(len(np.where((angle > lower_angle) * (angle < upper_angle))[0]),
        len(np.where((angle_bg > lower_angle) * (angle_bg < upper_angle))[0]), 
        len(np.where((angle_lamp > lower_angle) * (angle_lamp < upper_angle))[0]))
# choosing side
i = 0
# create a grid on a subset of the intersection of all three angles intervals
x_grid = np.linspace(lower_angle, upper_angle, min_len + 1) # "+1" to be able to cut of the edges
x_grid = x_grid[1:-1] # cut the edges
j = i
if sample_name == "Si":
    j += 2
    print(j)
x_grid = x_grid_err

# real transmission and absorption
lamp = polygon(x_grid, angle_lamp, trans_lamp)

t = polygon(x_grid, angle, trans)
t = un.uarray(t, t_errors)
t_bg = polygon(x_grid, angle_bg, trans_bg)
t_real = (t - t_bg) / lamp

a = polygon(x_grid, angle, absorp)
a = un.uarray(a, a_errors)
a_bg = polygon(x_grid, angle_bg, absorp_bg)
a_real = (a - a_bg) / lamp

side = ["left", "right", "left", "right"]                                   # side of the recorded spectrum
xlims = [(-40, -25), (25, 40), (-49, -35), (35, 50)]                        # limits for plotting
xlims1 = [(-60, -25), (25, 40), (-49, -35), (35, 50)]                        # limits for plotting
ylims = [(0, 0.6), (0, 1.2), (0, 1.2), (0, 1.2)]
fit_ranges_t = [(-34.6, -31.4), (30.0, 34.6), (-43.0, -38.0), (37.0, 43.0)] # data points for interpolation of transmission
fit_ranges_a = [(-36.0, -31.9), (31.3, 36.0), (-46.0, -38.0), (38.0, 44.2)] # data points for interpolation of absorption
legend_loc = [1, 2, 6, 5]                                                   # location of the legend
x_hori = [x_grid[x_grid < -30], x_grid[x_grid > 29], 
        x_grid[x_grid < -37], x_grid[x_grid > 29]][j]               # horizontal line should not cross the legend but intersect

# do the interpolation!
# normalizing
max_t = max(t_real[(x_grid > xlims[j][0]) * (x_grid < xlims[j][1])])
t_real = t_real / max_t
max_a = max(a_real[(x_grid > xlims[j][0]) * (x_grid < xlims[j][1])])
a_real = a_real / max_a
# linear fit
# Transmission
fit_range_t = fit_ranges_t[j]
index_fit = (x_grid > fit_range_t[0]) * (x_grid < fit_range_t[1])
alpha_fit_t = x_grid[index_fit]
t_fit = un.nominal_values(t_real[index_fit])
t_std_dev = un.std_devs(t_real[index_fit])
x_borders_t = alpha_fit_t[[0, -1]]
t_borders = t_fit[[0, -1]]
def t_func(alpha, a, b):
    return a * alpha + b
t_c, t_cov = curve_fit(t_func, alpha_fit_t, t_fit, p0=None) #, sigma=t_std_dev)#, absolute_sigma=True) # this will not work for scipy 0.13.
t_uc = uc.correlated_values(t_c, t_cov)

# Absorption
fit_range_a = fit_ranges_a[j]
index_fit = (x_grid > fit_range_a[0]) * (x_grid < fit_range_a[1])
alpha_fit_a = x_grid[index_fit]
a_fit = un.nominal_values(a_real[index_fit])
a_std_dev = un.std_devs(a_real[index_fit])
x_borders_a = alpha_fit_a[[0, -1]]
a_borders = a_fit[[0, -1]]
min_a = min(a_real[(x_grid > xlims[j][0]) * (x_grid < xlims[j][1])]).n
def a_func(alpha, a, b):
    return a * alpha + b
a_c, a_cov = curve_fit(a_func, alpha_fit_a, a_fit, p0=None) #, sigma=a_std_dev)#, absolute_sigma=True) # this will not work for scipy 0.13.
a_uc = uc.correlated_values(a_c, a_cov)

# find intersect of horizontals and interpolations
x_int_up_t  = (1 - t_c[1]) / t_c[0]
x_int_lo_t  = (min_a - t_c[1]) / t_c[0]
x_int_t = np.array([x_int_up_t, x_int_lo_t])
x_int_up_a  = (1 - a_c[1]) / a_c[0]
x_int_lo_a  = (min_a - a_c[1]) / a_c[0]
x_int_a = np.array([x_int_up_a, x_int_lo_a])

# intersection of the two straight lines
x_int = -(t_c[1] - a_c[1]) / (t_c[0] - a_c[0])

# plotting interpolation and horizontals
fig1, ax1 = plt.subplots(1, 1)
if not save_fig:
    fig1.suptitle("Band Gap Normalized, Sample: " + sample_name + ", " + side[j])
# plotting the linear fit, including shading of errors
ax1.errorbar(x_grid, un.nominal_values(t_real), yerr=un.std_devs(t_real), fmt='.', 
        c=plot_t.get_color(), alpha=0.8, label=r'$T_\mathrm{Si}$')
ax1.plot(x_grid, t_func(x_grid, *t_c), '-', c=plot_t.get_color(), alpha=0.8)
#        label=(r'$T_\mathrm{' + sample_name + r', interpolation}$'))
ax1.plot(x_borders_t, t_borders, 'o', c=plot_t.get_color(), alpha=0.8)
#        label=(r'lower and upper minimum for interpol.'))
ax1.plot(x_hori, x_hori*0 + 1, '--', c=plot_t.get_color(), alpha=0.8)
#        label=(r'Horizontal at $\max(T)$'))
textstr = "intersects: \n \
        \\begin{eqnarray*} \
        \\alpha_1 &=& %.1f^\circ \\\\ \
        \\alpha_2 &=& %.1f^\circ \
        \end{eqnarray*}"%(x_int_up_t, x_int_lo_t)
ax1.plot(x_int_t, t_func(x_int_t, *t_c), 's', c=plot_t.get_color(), alpha=0.8) #, label=(textstr))
ax1.errorbar(x_grid, un.nominal_values(a_real), yerr=un.std_devs(a_real), fmt='.', 
        c=plot_a.get_color(), alpha=0.8, label=r'$R_\mathrm{Si}$')
ax1.plot(x_grid, a_func(x_grid, *a_c), '-', c=plot_a.get_color(), alpha=0.8)
#        label=(r'$A_\mathrm{' + sample_name + r', interpolation}$'))
ax1.plot(x_borders_a, a_borders, 'o', c=plot_a.get_color(), alpha=0.8)
#        label=(r'lower and upper minimum for interpol.'))
ax1.plot(x_hori, x_hori*0 + min_a, '--', c=plot_a.get_color(), alpha=0.8)
#        label=(r'Horizontal at $\min(A)$'))
textstr = "intersects: \n \
        \\begin{eqnarray*} \
        \\alpha_1 &=& %.1f^\circ \\\\ \
        \\alpha_2 &=& %.1f^\circ \
        \end{eqnarray*}"%(x_int_up_t, x_int_lo_t)
ax1.plot(x_int_a, a_func(x_int_a, *a_c), 's', c=plot_a.get_color(), alpha=0.8) #, label=(textstr))
ax1.set_xlim(xlims[j])
ax1.set_ylim(-0.1, 1.1)
ax1.set_xlabel("angle / degrees")
ax1.set_ylabel("$U$ / V")
ax1.legend(loc=legend_loc[j])
if show_fig:
    fig1.show()
if save_fig:
    file_name = "band_gap_result_error"
    fig1.savefig(fig_dir + file_name + ".pdf")
    fig1.savefig(fig_dir + file_name + ".png")

