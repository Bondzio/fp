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

# angle to energy
def ang_to_E(alpha, sample_name):
    deg_to_rad = lambda phi_deg: 2 * np.pi / 360 * phi_deg
    phi = deg_to_rad(alpha)
    if sample_name == "Si":
        d = 1 / 1200 / 1000
    if sample_name == "Ge":
        d = 1 / 600 / 1000
    psi = deg_to_rad(7.5)
    E = co.h * co.c / (2 * d * abs(np.sin(phi)) * np.cos(psi))
    En = E / co.e # Energy in eV
    return(En)

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


Es = 4 * [0]
dEs = np.zeros(4)
f = open("band_gap_results.tex", "w+")
f.write("\t\\begin{tabular}{|p{1.5cm}|p{1.5cm}|p{1.5cm}|p{1.5cm}|p{2cm}|p{2cm}|p{2.7cm}|}\n")
f.write("\t\t\hline\n")
f.write("\t\t\\rowcolor{tabcolor}\n")
f.write("\t\tSample & $\phi_g / ^\circ$ & \n \
\t\t\t$\overline{\phi_\mathrm{lower}}/ ^\circ$  & $\overline{\phi_\mathrm{upper}}/ ^\circ$  &\n \
\t\t\t$E_\mathrm{lower}$/ eV & $E_\mathrm{upper}$/ eV  &  $E_g$ / eV \\\\ \hline\n")
for sample_name in ["Si", "Ge"]:
    fig1, ax1 = plt.subplots(1, 1)
    if not save_fig:
        fig1.suptitle("Band Gap, Sample: " + sample_name)
    for suffix in ["2"]:
        npy_file = npy_dir + "band_gap_" + sample_name + "_" + suffix + ".npy"
        angle, trans, absorp  = np.load(npy_file)
        if angle[0] > 0:
            angle, trans, absorp  = angle[::-1], trans[::-1], absorp[::-1]
        plot_t, = ax1.plot(angle, trans, '.', alpha=0.8, label=(r'$T_\mathrm{' + sample_name + r'}$'))
        next(ax1._get_lines.color_cycle)
        plot_a, = ax1.plot(angle, absorp, '.', alpha=0.8, label=(r'$A_\mathrm{' + sample_name + r'}$'))
        next(ax1._get_lines.color_cycle)
    for suffix in ["lamp"]:
        npy_file = npy_dir + "band_gap_" + sample_name + "_" + suffix + ".npy"
        angle_lamp, trans_lamp, dummy = np.load(npy_file)
        if angle_lamp[0] > 0:
            angle_lamp, trans_lamp, absorp_lamp = angle_lamp[::-1], trans_lamp[::-1], absorp_lamp[::-1]
        plot_lamp, = ax1.plot(angle_lamp, trans_lamp, '.', alpha=0.8, label=r'$L$')
    for suffix in ["background"]:
        npy_file = npy_dir + "band_gap_" + sample_name + "_" + suffix + ".npy"
        angle_bg, trans_bg, absorp_bg  = np.load(npy_file)
        if angle_bg[0] > 0:
            angle_bg, trans_bg, absorp_bg  = angle_bg[::-1], trans_bg[::-1], absorp_bg[::-1]
        #plot_t_bg, = ax1.plot(angle_bg, trans_bg, '.', alpha=0.8, label=r'$T_\mathrm{background}$')
        #plot_a_bg, = ax1.plot(angle_bg, absorp_bg, '.', alpha=0.8, label=r'$A_\mathrm{background}$')
    # place a text box in upper left in axes coords
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    textstr = 'Sample: ' + sample_name
    ax1 = plt.gca()
    ax1.text(0.1, 0.95, textstr, transform=ax1.transAxes, va='top', bbox=props)
    ax1.set_xlim(angle[0], angle[-1])
    ax1.set_ylim(0, 3)
    ax1.set_xlabel("angle / degrees")
    ax1.set_ylabel("$U$ / V")
    ax1.legend(loc=1)
    if show_fig:
        fig1.show()
    if save_fig:
        file_name = "band_gap_raw_" + sample_name
        fig1.savefig(fig_dir + file_name + ".pdf")
        fig1.savefig(fig_dir + file_name + ".png")

    #trans_real = (trans -  trans_background)/ trans_lamp
    # the data is not defined on the same array of angles, we need to interpolate onto one grid!
    # get the ends and length of the grid
    lower_angle = max(angle[0], angle_bg[0], angle_lamp[0])
    upper_angle = min(angle[-1], angle_bg[-1], angle_lamp[-1])
    min_len = min(len(np.where((angle > lower_angle) * (angle < upper_angle))[0]),
            len(np.where((angle_bg > lower_angle) * (angle_bg < upper_angle))[0]), 
            len(np.where((angle_lamp > lower_angle) * (angle_lamp < upper_angle))[0]))
    # create a grid on a subset of the intersection of all three angles intervals
    x_grid = np.linspace(lower_angle, upper_angle, min_len + 1) # "+1" to be able to cut of the edges
    x_grid = x_grid[1:-1] # cut the edges

    # real transmission and absorption
    lamp = polygon(x_grid, angle_lamp, trans_lamp)

    t = polygon(x_grid, angle, trans)
    t_bg = polygon(x_grid, angle_bg, trans_bg)
    t_real = (t - t_bg) / lamp

    a = polygon(x_grid, angle, absorp)
    a_bg = polygon(x_grid, angle_bg, absorp_bg)
    a_real = (a - a_bg) / lamp

    # choosing side
    for i in range(2):
        j = i
        if sample_name == "Si":
            j += 2
        print(j)
        side = ["left", "right", "left", "right"]                                   # side of the recorded spectrum
        xlims = [(-40, -25), (25, 40), (-49, -35), (35, 50)]                        # limits for plotting
        xlims1 = [(-60, -25), (25, 40), (-49, -35), (35, 50)]                        # limits for plotting
        ylims = [(0, 0.6), (0, 1.2), (0, 1.2), (0, 1.2)]
        fit_ranges_t = [(-34.6, -31.4), (30.0, 34.6), (-43.0, -38.0), (37.0, 43.0)] # data points for interpolation of transmission
        fit_ranges_a = [(-36.0, -31.9), (31.3, 36.0), (-46.0, -38.0), (38.0, 44.2)] # data points for interpolation of absorption
        legend_loc = [1, 2, 6, 5]                                                   # location of the legend
        x_hori = [x_grid[x_grid < -30], x_grid[x_grid > 29], 
                x_grid[x_grid < -37], x_grid[x_grid > 29]][j]               # horizontal line should not cross the legend but intersect

        # plotting real transmission and absorption signals
        fig1, ax1 = plt.subplots(1, 1)
        if not save_fig:
            fig1.suptitle("Band Gap, Sample: " + sample_name + ", " + side[j])
        ax1.plot(x_grid, t_real, '.', c=plot_t.get_color(), alpha=0.8, label=(r'$T_\mathrm{' + sample_name + r', corrected}$'))
        ax1.plot(x_grid, a_real, '.', c=plot_a.get_color(), alpha=0.8, label=(r'$A_\mathrm{' + sample_name + r', corrected}$'))
        ax1.set_xlim(xlims1[j])
        ax1.set_ylim(ylims[j])
        ax1.set_xlabel("angle / rad")
        ax1.set_ylabel("$U$ / V")
        ax1.legend(loc=1)
        if show_fig:
            fig1.show()
        if save_fig:
            file_name = "band_gap_detail_" + sample_name + "_" + side[j]
            fig1.savefig(fig_dir + file_name + ".pdf")
            fig1.savefig(fig_dir + file_name + ".png")

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
        t_fit = t_real[index_fit]
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
        a_fit = a_real[index_fit]
        x_borders_a = alpha_fit_a[[0, -1]]
        a_borders = a_fit[[0, -1]]
        min_a = min(a_real[(x_grid > xlims[j][0]) * (x_grid < xlims[j][1])])
        def a_func(alpha, a, b):
            return a * alpha + b
        a_c, a_cov = curve_fit(a_func, alpha_fit_a, a_fit, p0=None) #, sigma=t_std_dev)#, absolute_sigma=True) # this will not work for scipy 0.13.
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
        x_int_lo = 0.5 * (x_int_lo_t + x_int_up_a)
        x_int_up = 0.5 * (x_int_up_t + x_int_lo_a)

        # plotting interpolation and horizontals
        fig1, ax1 = plt.subplots(1, 1)
        if not save_fig:
            fig1.suptitle("Band Gap Normalized, Sample: " + sample_name + ", " + side[j])
        # plotting the linear fit, including shading of errors
        ax1.plot(x_grid, t_real, '.', c=plot_t.get_color(), alpha=0.8, \
                label=(r'$T_\mathrm{' + sample_name + r', normalized}$'))
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
        ax1.plot(x_grid, a_real, '.', c=plot_a.get_color(), alpha=0.8, \
                label=(r'$A_\mathrm{' + sample_name + r', normalized}$'))
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
            file_name = "band_gap_result_" + sample_name + "_" + side[j]
            fig1.savefig(fig_dir + file_name + ".pdf")
            fig1.savefig(fig_dir + file_name + ".png")


        E       = ang_to_E(x_int, sample_name)
        E_lo    = ang_to_E(x_int_lo, sample_name)
        E_up    = ang_to_E(x_int_up, sample_name)
        dE      = max(abs(E - E_lo), abs(E - E_up))
        Es[j]   = uc.ufloat(E, dE)
        dEs[j]  = dE

        f.write("\t\t%s & %.2f & %.2f & %.2f  & %.2f & %.2f & $(%.2f \pm %.2f)$\\\\ \n" \
                %(sample_name, x_int, x_int_lo, x_int_up, E_lo, E_up, E, dE))
f.write("\t\t\hline\n")
f.write("\t\end{tabular}\n")
f.close()

i = 0
E_Ge = (Es[i] / dEs[i]**2 + Es[i+1] / dEs[i+1]**2) / np.sum((1 / dEs[i:i+2])**2)
i = 2
E_Si = (Es[i] / dEs[i]**2 + Es[i+1] / dEs[i+1]**2) / np.sum((1 / dEs[i:i+2])**2)
