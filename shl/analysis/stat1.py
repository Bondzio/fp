import numpy as np
from math import log10, floor
import uncertainties as uc

def chi2_min(x,y,p,Sy):
    return  np.sum(((np.polyval(p,x) - y) / Sy) ** 2 )



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
        str_co += " &=& %.3f "%co
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
            str_row += "%.3f &"%entry
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

def unv(uarray):        # returning nominal values of a uarray
    return un.nominal_values(uarray)

def usd(uarray):        # returning the standard deviations of a uarray
    return un.std_devs(uarray)

def chi2_min(x,y,p,Sy):
    return  np.sum(((np.polyval(p,x) - y) / Sy) ** 2 )


