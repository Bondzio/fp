import numpy as np
import pylab as plt
import uncertainties as uc
from uncertainties import unumpy as un
from math import log10, floor
import seaborn as sns
sns.set(style='ticks', palette='Set2')
sns.despine()
plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('text', usetex=True)

def unv(uarray):        # returning nominal values of a uarray
    return un.nominal_values(uarray)

def usd(uarray):        # returning the standard deviations of a uarray
    return un.std_devs(uarray)

def csv_to_npy(n):
    # create arrays b (magnetic field) and I (current) from .csv files
    filename = "b_I" + str(n)
    file = open(input_dir + filename + ".csv")   # read data from csv table
    b = []
    I = []
    for line in file:
        x = line.split(",")
        I.append(float(x[0]))
        b.append(float(x[1]))
    file.close()
    b = np.sort(np.array(b))
    I = np.sort(np.array(I))
    np.save(output_dir + filename, b)
    np.save(output_dir + "I" + str(n), I)
    return b, I

def dig_err(cov, i): # returns the significant digit of the error
    dx = np.sqrt(cov[i,i])
    digit = -int(floor(log10(dx)))    
    if (dx * 10**digit) < 3.5:
        digit += 1
    return digit
    
def dig_val(x):     # returns the last significant digit of a value (error convention...)
    digit = -int(floor(log10(abs(x))))    
    if (x * 10**digit) < 3.5:
        digit += 1
    return digit

def em(str):        # rewrites 'e' for exponential but leaves other 'e's untouched
    str = str.split("e")
    for i, substr in enumerate(str):
        if i == 0:
            new_str = substr
        else:
            if substr[1].isdigit():
                new_str = r"\mathrm{e}".join([new_str, substr])
            else:
                new_str = "e".join([new_str, substr])
    return new_str

def la_coeff(f1, coeff, cov, var_names, additional_digits=0):
    """
    prints coeffients and their covariance matrix to a .tex file
    """
    f1.write(r"\begin{align}" + "\n")

    for j, co in enumerate(coeff):
        str_co = "    " + var_names[j]
        var = uc.ufloat(coeff[j], np.sqrt(cov[j,j]))
        str_co += " &=& {:L} \\\\".format(var)
        str_co = em(str_co)
        f1.write(str_co +"\n")

    f1.write(r"    \mathrm{cov}(p_i, p_j) &=& " "\n")
    f1.write(r"    \begin{pmatrix}" + "\n")
    for row in cov:
        str_row = "        "
        for entry in row:
            digit = dig_val(entry) + additional_digits
            var = round(entry, digit)
            """
            if digit < 1:
                str_row += " %i &"%int(var)
            elif digit < 4:
                pre_str = "%." + str(digit) + "f &"
                str_row += pre_str%var
            else:
                str_row += "%.1e &"%var
            """
            str_row += "%.1f &"%entry
        str_row = str_row[:-1] + r"\\"
        str_row = em(str_row)
        f1.write(str_row + "\n")
    f1.write(r"    \end{pmatrix}" + "\n")
    f1.write(r"\end{align}" + "\n\n")
    return 0
################################################################################

show_fig = 1
save_fig = 1
save_tab = 1
plt.close('all')
fig_dir = '../figures/'
tab_dir = '../tables/'
tab_name = "b_I"
input_dir = "../data/"
output_dir = input_dir
colors = ['b', 'g', 'r', 'pink']

b, I = csv_to_npy(1)
b2, I2 = csv_to_npy(2)

# least square fit for b(I) (polyfit of deg = 1)
i_max = 8
coeff, cov = np.polyfit(I[:i_max], b[:i_max], deg=1, cov=True)
c = uc.correlated_values(coeff, cov)
Is = np.linspace(0, 6, 100)
f1 = open("coefficients.tex", "w+")
var_names = ["p_0", "p_1"]
la_coeff(f1, coeff, cov, var_names, additional_digits=2)
f1.close()


# plot B(I) with error bars
fig1, ax1 = plt.subplots(1,1, figsize = (8.09,5))
b_error = 5    # uncertainty on B measurement, taken to be the last significant digit
I_error = 0.01     # uncertainty in measuring the current I
ax1.fill_between(Is, 
        unv(np.polyval(c, Is)) + usd(np.polyval(c, Is)),
        unv(np.polyval(c, Is)) - usd(np.polyval(c, Is)),
        facecolor=colors[0], color=colors[0], alpha=0.2)
ax1.plot(Is, np.polyval(coeff, Is), '-', linewidth=1.0)                
#ax1.errorbar(I[:i_max], b[:i_max], xerr=I_error, yerr=b_error, fmt='.') 
#ax1.errorbar(I2, b2, xerr=I_error, yerr=b_error, fmt='.')                
#ax1.legend(loc=4)
ax1.errorbar(I, b, xerr=I_error, yerr=b_error, fmt='b,', elinewidth=1.0, capsize=1.2, capthick=0.8)
ax1.set_xlim([0, 6])
xmin = 0
xmax = 5.2
ax1.set_xlim([xmin, xmax])
ymin = 0
ymax = 550
ax1.set_ylim([ymin, ymax])
ax1.set_xlabel('$I \, / \, \mathrm{A}$')
ax1.set_ylabel('$B(I) \, / \, \mathrm{mT}$')
if not save_fig:
    fig1.suptitle('Magnetic field $B$ for current $I$')
    ax1.set_title('$\mathrm{for} \,  z = 2.0$ cm')

# give out latex tables:
if save_tab:
    f1 = open(tab_dir + tab_name + ".tex", "w+")
    n_cols = 2          # number of columns
    n_rows = 12         # number of rows
    ind = "    "
    inds = 2*ind
    cc = "\cellcolor{LightCyan}" # color of name cells
    cells = ("|p{1.34cm}|p{2.16cm}|p{0.8cm}"*n_cols)[:-8]
    str_l = inds + r"\cline{1-2}\cline{4-5}" + "\n"

    f1.write(r"\begin{table}[htdp]" + "\n")
    f1.write(r"\centering" + "\n")
    #f1.write(ind + "\\begin{tabular}{|l |l |c |l |l |c |l |l |}\n")
    f1.write(ind + r"\begin{tabular}{" + cells + "}\n")
    f1.write(str_l)
    f1.write(((inds + "$I$ / A " + cc + "& $B$ / mT " + cc +"&&\n") * n_cols)[:-3] + "\\\\ \n")
    f1.write(str_l)
    for i in range(n_rows):
        row = (inds + (("%.2f & %i &&" * n_cols)\
                %(I[i], b[i], I2[i], b2[i]))[:-2]
                + "\\\\ \n") 
        f1.write(row)
    f1.write(str_l)
    f1.write(ind + r"\end{tabular}" + "\n")
    f1.write(ind + r"\caption{" + "\n")
    f1.write(ind + r"    Measurement of magnetic field $B$ for current $I$ at height $z = 2$ cm. "\
            + "Corresponding uncertainties: $\Delta I = %.2f$ A, $\Delta B = %i$ mT."%(I_error, b_error) + "\n")
    f1.write(ind + r"    }" + "\n")
    f1.write(ind + r"\label{tab:" + tab_name + "}" + "\n")
    f1.write(r"\end{table}" + "\n")
    f1.close()

if show_fig:
    fig1.show()
if save_fig: 
    fig1.savefig(fig_dir + "b_I.pdf")


