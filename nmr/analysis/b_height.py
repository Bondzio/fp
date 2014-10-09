import numpy as np
import pylab as plt

show_fig = 0
save_fig = 0
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

fig1, ax1 = plt.subplots(1,1, figsize = (12,5))
ymin = 330
ymax = 360
b_error = 1     # uncertainty on B measurement, taken to be the last significant digit
z_error = 0.4     # uncertainty in measuring the depth z
ax1.errorbar(z, b, xerr=z_error, yerr=b_error, fmt='.', label=' ')                # plots of cut-off output
ax1.plot([20, 20], [0, 400], 'k--', label='working point')                # plots of cut-off output
ax1.legend(loc=4)
ax1.set_xlim([0, 40])
ax1.set_ylim([ymin, ymax])
ax1.set_xlabel('$z \, / \, \mathrm{mm}$')
ax1.set_ylabel('$B(z) \, / \, \mathrm{mT}$')
if not save_fig:
    fig1.suptitle('Magnetic field $B$ at height $z$')
    ax1.set_title('$\mathrm{for} \,  I = 2.62$ A')

# give out latex tables:
if save_tab:
    tab_name = "b_height"
    str_l = r"\cline{1-3}\cline{5-7}\cline{9-11}" + "\n"
    col_length = 12
    if save_tab:
        f1 = open(tab_dir + tab_name + ".txt", "w+")
        f1.write(r"\begin{table}[htdp]" + "\n")
        f1.write("\\begin{tabular}{\n    " + "|l |l ||c |"*2 + "l|l |}\n")
        f1.write(str_l)
        f1.write(r"    \rowcolor{LightCyan}" + "\n")
        f1.write(r"    $B$ / mT & $z$ / mm &&$B$ / mT & $z$ / mm && $B$ / mT & $z$ / mm" + "\n")
        for i in range(col_length):
            tab = ("    " + "%i & %i &&"*2 + "%i & %i \\\\ \n") \
                    %(b[i], z[i], b[i + col_length], z[i + col_length], b[i + 2*col_length], z[i + 2*col_length])
            f1.write(tab)
        f1.write(str_l)
        f1.write(r"\end{tabular}" + "\n")
        f1.write(r"\caption{" + "\n")
        f1.write(r"    Measurement 1.1: Magnetic field $B$ at height $z$" + "\n")
        f1.write(r"    }" + "\n")
        f1.write(r"\label{tab:" + tab_name + "}" + "\n")
        f1.write(r"\end{table}" + "\n")
        f1.close()

if show_fig:
    fig1.show()
if save_fig: 
    fig1.savefig(fig_dir + "b.pdf")



