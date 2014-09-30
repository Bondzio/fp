import numpy as np
import matplotlib.pyplot as plt
from math import log10, floor
import uncertainties as uc

input_dir = "../data_faraday/"
figure_dir = "../figures/"

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
    f1.write(r"\begin{eqnarray}" + "\n")
    for j, co in enumerate(coeff):
        str_co = "    " + var_names[j]
        digit = dig_err(cov, j) + additional_digits
        var = round(co, digit)
        """
        if digit < 1:
            str_co  += " &=& %i "%int(var) 
        if digit < 4:
            pre_str = " &=&%." + str(digit) + r"f "
            str_co  += pre_str%(var) 
        else:
            str_co += " &=& %.3e "%var
        """
        str_co += " &=& %.3e "%co
        str_co += r"\cm \nonumber \\"
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
            str_row += "%.3e &"%entry
        str_row = str_row[:-1] + r"\\"
        str_row = em(str_row)
        f1.write(str_row + "\n")
    f1.write(r"    \end{pmatrix}" + "\n")
    f1.write(r"\\ \Rightarrow \qquad" + "\n")
    for j, co in enumerate(coeff):
        str_co = "    " + var_names[j]
        var = uc.ufloat(coeff[j], np.sqrt(cov[j,j]))
        str_co += " &=& {:L} \\cm\\\\".format(var)
        str_co = em(str_co)
        if j == len(coeff) -1:
            str_co = str_co[:-2]
        f1.write(str_co +"\n")
    f1.write(r"\end{eqnarray}" + "\n\n")

    return 0


for i in range(4):
    a = np.load(input_dir +"a_%i.npy"%(i+1))
    I = np.load(input_dir + "i_%i.npy"%(i+1))
    I_fit = np.linspace(min(I),max(I),100) 
    S_a = 0.5
    weights = a*0 + 1/S_a 
    p, cov= np.polyfit(I,a, 1, full=False, cov=True, w=weights)

    a_fit = np.polyval(p, I_fit)
    S_I = 0.05
    plt.figure()
    plt.xlim(min(I), max(I))
    plt.plot(I_fit,a_fit)
    plt.fill_between(I_fit, \
                np.polyval(p - np.sqrt(np.diag(cov)), I_fit), \
                np.polyval(p + np.sqrt(np.diag(cov)), I_fit), \
                facecolor="r", color="b", alpha=0.3 )
    f1 = open("coefficients.tex","a")
    la_coeff(f1, p,cov, ["p_1","p_1"])
    
    plt.errorbar(I,a,yerr=S_a,xerr=S_I, fmt=".")
    plt.title("Measurement 2.%d"%(i+1))

    plt.grid(True)
    plt.xlabel("Current $I(\\alpha)$")
    plt.ylabel("angle $\\alpha$")
    plt.savefig(figure_dir +"fig2%d.pdf"%(i+1))

    
