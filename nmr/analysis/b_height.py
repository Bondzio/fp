import numpy as np
import pylab as plt
import seaborn as sns
sns.set(style='ticks', palette='Set2')
sns.despine()
plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('text', usetex=True)

show_fig = 1
save_fig = 1
save_tab = 1
plt.close('all')
fig_dir = '../figures/'
tab_dir = '../tables/'
input_dir = "../data/"
output_dir = input_dir

# create arrays b (magnetic field) and z (height) from .csv files
file = open(input_dir + "b_height.csv")   # read data from csv table
b = []
for line in file:
    b.append(float(line))
file.close()
b = np.array(b)

z = np.arange(0, 38)
z = np.delete(z, [1,2])
np.save(output_dir + "b_z", b)
np.save(output_dir + "z", z)

# Plot B(z) with errorbars
fig1, ax1 = plt.subplots(1,1, figsize = (8.09,5))
ymin = 300
ymax = 360
b_error = 5     # uncertainty on B measurement, taken to be the last significant digit
z_error = 0.4     # uncertainty in measuring the depth z
ax1.errorbar(z, b, xerr=z_error, yerr=b_error, fmt='.')                # plots of cut-off output
ax1.plot([20, 20], [0, 400], 'k--', label='working point')                # plots of cut-off output
ax1.legend(loc=4)
ax1.set_xlim([0, 40])
ax1.set_ylim([ymin, ymax])
ax1.set_xlabel('$z \, / \, \mathrm{mm}$')
ax1.set_ylabel('$B(z) \, / \, \mathrm{mT}$')
if not save_fig:
    fig1.suptitle('Magnetic field $B$ at height $z$')
    ax1.set_title('$\mathrm{for} \,  I = 2.62$ A')

if show_fig:
    fig1.show()
if save_fig: 
    fig1.savefig(fig_dir + "b_height.pdf")

# give out latex table:
if save_tab:
    tab_name = "b_height"
    f1 = open(tab_dir + tab_name + ".tex", "w+")
    n_cols = 3          # number of columns
    n_rows = 12         # number of rows
    ind = "    "
    inds = 2*ind
    cc = "\cellcolor{LightCyan}" # color of name cells
    cells = ("|p{1.34cm}|p{2.16cm}|p{0.8cm}" * n_cols)[:-8]
    str_l = inds + r"\cline{1-2}\cline{4-5}\cline{7-8}" + "\n"

    f1.write(r"\begin{table}[htdp]" + "\n")
    f1.write(r"\centering" + "\n")
    #f1.write(ind + "\\begin{tabular}{|l |l |c |l |l |c |l |l |}\n")
    f1.write(ind + r"\begin{tabular}{" + cells + "}\n")
    f1.write(str_l)
    f1.write(((inds + "$z$ / mm " + cc + "& $B$ / mT " + cc +"&&\n") * n_cols)[:-3] + "\\\\ \n")
    f1.write(str_l)
    for i in range(n_rows):
        row = (inds + (("%i & %i &&" * n_cols)\
                %(z[i], b[i], z[i + n_rows], b[i + n_rows], z[i + 2*n_rows], b[i + 2*n_rows]))[:-2]
                + "\\\\ \n") 
        f1.write(row)
    f1.write(str_l)
    f1.write(ind + r"\end{tabular}" + "\n")
    f1.write(ind + r"\caption{" + "\n")
    f1.write(ind + r"    Measurement of unmodulated magnetic field $B$ at height $z$ for $I = 2.62$ A."\
            + "Corresponding uncertainties: $\Delta z = %.1f$ mm, $\Delta B = %i$ mT."%(z_error, b_error) + "\n")
    f1.write(ind + r"    }" + "\n")
    f1.write(ind + r"\label{tab:" + tab_name + "}" + "\n")
    f1.write(r"\end{table}" + "\n")
    f1.close()



