import numpy as np
from matplotlib.colors import colorConverter
from matplotlib import rcParams
from scipy.optimize import curve_fit
import re
from scipy.constants import c,h,eV, pi
from smooth import savitzky_golay
import uncertainties as uc
import uncertainties.unumpy as un
npy_dir = "./npy/"

# Functions for fitting
def breit_wigner(x, x0, gamma, amplitude, offset):
    """
    Breit-Wigner or Cauchy distribution with location parameter x0 
    and scale parameter gamma = 0.5 * FWHM. Amplitude: 1 / (Pi * gamma). 
    No mean, variance or higher moments exist.
    """
    return amplitude / ((pi*gamma) * (1 + ((x - x0) / gamma)**2)) + offset
   
def bw_fit(x, y_e, x_range, p0, fit=True):
    x_min, x_max = x_range
    mask = (x > x_min) * (x < x_max)
    x_fit = x[mask]
    y_fit = un.nominal_values(y_e[mask])
    y_sigma = np.sqrt(un.std_devs(y_e[mask]))
    
    if fit:
        coeff, cov = curve_fit(breit_wigner, x_fit, y_fit, p0=p0, 
                                  sigma=y_sigma, absolute_sigma=True)
        c = uc.correlated_values(coeff, cov)
        fit_peak = breit_wigner(x_fit, *coeff)
    else:
        fit_peak = breit_wigner(x_fit, *p0)
        c = un.uarray(p0, [0, 0, 0, 0])
    
    return x_fit, fit_peak, c

def gauss(x, x0, sigma, amplitude, offset):
    """
    Gauss peak plus offset
    """
    return amplitude / (np.sqrt(2 * pi) * sigma) * np.exp(1)**(-(x - x0)**2 / (2 * sigma**2)) + offset
   
def gauss_fit(x, y_e, x_range, p0, fit=True):
    x_min, x_max = x_range
    mask = (x > x_min) * (x < x_max)
    x_fit = x[mask]
    y_fit = un.nominal_values(y_e[mask])
    y_sigma = np.sqrt(un.std_devs(y_e[mask]))
    
    if fit:
        coeff, cov = curve_fit(gauss, x_fit, y_fit, p0=p0, 
                                  sigma=y_sigma, absolute_sigma=True)
        c = uc.correlated_values(coeff, cov)
        fit_peak = gauss(x_fit, *coeff)
    else:
        fit_peak = gauss(x_fit, *p0)
        c = un.uarray(p0, [0, 0, 0, 0])
    
    return x_fit, fit_peak, c

def two_breit_wigner(x, x0, gamma, amplitude, x02, gamma2, amplitude2, offset):
    """
    Breit-Wigner or Cauchy distribution with location parameter x0 
    and scale parameter gamma = 0.5 * FWHM. Amplitude: 1 / (Pi * gamma). 
    No mean, variance or higher moments exist.
    """
    tbw = amplitude / ((np.pi*gamma) * (1 + ((x - x0) / gamma)**2)) + \
    amplitude2 / ((np.pi*gamma2) * (1 + ((x - x02) / gamma2)**2)) + offset
    return tbw

def two_bw_fit(x, y_e, x_range, p0, fit=True):
    x_min, x_max = x_range
    mask = (x > x_min) * (x < x_max)
    x_fit = x[mask]
    y_fit = un.nominal_values(y_e[mask])
    y_sigma = np.sqrt(un.std_devs(y_e[mask]))
    
    if fit:
        coeff, cov = curve_fit(two_breit_wigner, x_fit, y_fit, p0=p0, 
                                  sigma=y_sigma, absolute_sigma=True)
        c = uc.correlated_values(coeff, cov)
        fit_peak = two_breit_wigner(x_fit, *coeff)
    else:
        fit_peak = two_breit_wigner(x_fit, *p0)
        c = un.uarray(p0, [0, 0, 0, 0])
    
    return x_fit, fit_peak, c

def poly(x, p0, p1, p2, p3, p4):
    x_p = x - 540 # move x value near the points to be fitted in order to lower the error!
    return p0*x_p**4 + p1*x_p**3 + p2*x_p**2 + p3*x_p + p4

def poly_fit(x, y_e, x_range, p0):
    x_min, x_max = x_range
    mask = (x > x_min) * (x < x_max)
    x_fit = x[mask]
    y_fit = un.nominal_values(y_e[mask])
    y_sigma = np.sqrt(un.std_devs(y_e[mask]))

    coeff, cov = curve_fit(poly, x_fit, y_fit, p0=p0, 
                           sigma=y_sigma, absolute_sigma=True)
    c = uc.correlated_values(coeff, cov)
    return c

def linear(x, a, b):
    return (a*x + b)

# Further functions conveniently defined
def uc_str(c, max_digit=4):
    """
    input format: uc.ufloat
    rounds float and corrisponding error to last significant digit
    returns float and error as string
    as integers with max max_digit (=4) error digits
    as floats with max 4 error digits
    as exp else
    """
    digit = -int(np.floor(np.log10(c.s)))    
    if (c.s * 10**digit) < 1.5: # convention...
        digit += 1
    c_r = round(c.n, digit)
    s_c_r = round(c.s, digit)
    if (-3 < digit) * (digit <= 0): # returns readable integers
        c_str = '%i \pm %i'%(c_r, s_c_r)
    elif (0 < digit) * (digit < (max_digit + 1)): # returns readable floats (max 3 digits)
        c_str = ('%.' + str(digit) + 'f \pm %.' + str(digit) + 'f')%(c_r, s_c_r)
    else: # returns exp
        c_str = ('(%.1f \pm %.1f)\mathrm{e}%i')%(c_r * 10**(digit-1), s_c_r * 10**(digit-1), -(digit-1))
    return c_str

def enum(arr1, *args):
    i_range = range(len(arr1))
    return zip(i_range, arr1 ,*args)

# RAMAN SPECIFIC
def t_avg(filename):
    """
    Get integration time and number of measurements
    """
    f = open("data/" + filename + ".txt", encoding='cp1252')
    lines = f.readlines()
    f.close()
    t = np.float(lines[8].split(' ')[2]) * 1e-6 # measured time in sec
    avg = np.float(lines[9].split(' ')[3]) # number of recording to take average from
    return t, avg

def lamb_to_cm(lamb_stokes):
    """
    Converts Raman peaks in nm into wavenumber (in cm^-1) of corresponding vibrational mode
    """
    #lamb_laser = 532 # nm, Nd:Yag frequency doubled
    lamb_laser = np.load(npy_dir + 'ccd_lamb_laser.npy')[0] # measured laser wavelength
    dnu_cm = abs(1 / lamb_laser - 1 / lamb_stokes) * 10**7
    return(dnu_cm)

def cm_to_lamb(dnu_cm):
    """
    Converts  wavenumber (in cm^-1) of vibrational mode into corresponding Stokes 
    peak in nm (for Nd:Yag laser)
    """
    #lamb_laser = 532 # nm, Nd:Yag frequency doubled
    lamb_laser = np.load(npy_dir + 'ccd_lamb_laser.npy')[0] # measured laser wavelength
    lamb_stokes = 1 / (1 / lamb_laser - dnu_cm * 10**-7) 
    lamb_anti_stokes =  1 / (1 / lamb_laser + dnu_cm * 10**-7) 
    return(lamb_stokes, lamb_anti_stokes)

def get_ccd_data(filename):
    t, avg = t_avg(filename)
    x = np.load("npy/"+filename+"_lamb.npy")
    y = np.load("npy/"+filename+"_count.npy")
    y_e = un.uarray(y, np.maximum(1, np.sqrt(y / avg)))
    return (t, avg, x, y, y_e)


def get_ccd_rate(filename):
    t, avg = t_avg(filename)
    x = np.load("npy/"+filename+"_lamb.npy")
    y = np.load("npy/"+filename+"_count.npy")
    y_e = un.uarray(y, np.maximum(1, np.sqrt(y / avg)))
    # transform to rates
    rate_bg = np.load(npy_dir + 'ccd_rate_bg.npy') # background rates
    rate_bg_e = np.load(npy_dir + 'ccd_rate_bg_e.npy')
    rate = y / t - rate_bg
    rate_e = y_e / t - rate_bg_e
    return (t, avg, x, rate, rate_e)

# Calibration

# Data
filename = "ccd_hg_00"
t, avg, x, y, y_e = get_ccd_data(filename)

As = []
x0s = []

# Fit ranges and initial guesses
# p = [x0, gamma, amplitude, offset]
x_ranges = [[420, 450], [520, 560], [565, 578], [578, 585]]
p0s = np.array([[435, 0.2, 14000, 0], [545, 0.2, 40000, 0], 
               [577, 0.2, 5000, 0], [579, 0.2, 5000, 0]])
labels = ['435.8 nm', '546.1 nm', '577.1 nm', '579.1 nm']
for i, x_range, p0, label in enum(x_ranges, p0s, labels):
    x_fit, fit_peak, c1 = bw_fit(x, y_e, x_range, p0, fit=True)
    x0s.append(c1[0])
    As.append(c1[2])

# Print latex table
lits = [435.8, 546.1, 577.1, 579.1]
for i, x0, lit in enum(x0s, lits):
    print("\cellcolor{LightCyan}$%i$ & $%s$ & $%.1f$   \\\\"%(
            i+1, uc_str(x0), lit))

# Get all peaks. Use HWHM = gamma as error on peaks.
x0 = un.nominal_values(x0s)
s_x0 = un.std_devs(x0s)

lit_peaks = np.array([435.8, 546.1, 577.1, 579.1])
coeff_lin, cov_lin = curve_fit(linear, lit_peaks, x0, p0=None, 
                               sigma=np.sqrt(s_x0), absolute_sigma=True)
c_lin = uc.correlated_values(coeff_lin, cov_lin)

# Switch to lambda(x0) = lit_peak(x0)
d_lin = np.array([1 / c_lin[0], -c_lin[1] / c_lin[0]])
np.save(npy_dir + 'ccd_calibration', d_lin)

# Detection probability
# Data
filename = "ccd_white_pol0"
t, avg, x, y0, y_e0 = get_ccd_data(filename)

filename = "ccd_white_pol45"
t, avg, x, y45, y_e45 = get_ccd_data(filename)

filename = "ccd_white_pol90"
t, avg, x, y90, y_e90 = get_ccd_data(filename)

corr0  = y45 / y0
corr90 = y45 / y90

# remove one outlier:
mstep = np.argmin(corr0[1:] - corr0[:-1]) + 1 # find the outlier
n = 20
low = corr0[mstep - n: mstep]
up = corr0[mstep + 1: mstep + n + 1]
corr_avg = np.mean(low + up) / 2 
corr0[mstep] = corr_avg # replace it by the mean of the n values in both directions

n_sav = 301 # number of data point for savitzki-golay filter
corr0_fil = savitzky_golay(corr0, n_sav, 4)
corr90_fil = savitzky_golay(corr90, n_sav, 4)

np.save(npy_dir + 'ccd_corr0', corr0_fil)
np.save(npy_dir + 'ccd_corr90', corr90_fil)

# Analyzing the laser
# Data
filename = "ccd_laser_ccl4_00"
t, avg, x, y, y_e = get_ccd_data(filename)

# Fitting
# p = [x0, sigma, amplitude, offset]
x_range = [510, 550]   # lower and upper bound
p0 = np.array([531, 0.4, 27000, 1420])
x_fit1, fit_peak1, c1 = gauss_fit(x, y_e, x_range, p0, fit=True)
x0 = uc.ufloat(c1[0].n, c1[1].n) # choose sigma as error on laser wavelength!
lamb_laser = linear(np.array([x0]), *d_lin) # Correct for calibration
np.save(npy_dir + 'ccd_lamb_laser', lamb_laser)
print(uc_str(lamb_laser[0]))

# Notch filter
# Data
filename = "ccd_white_notch"
t, avg, x_notch, y, y_e = get_ccd_data(filename)

# White background
y_white = np.load("npy/ccd_white_00_count.npy")
notch = y_white * 1.12 - y
np.save(npy_dir + 'ccd_notch', notch)
np.save(npy_dir + 'ccd_x_notch', x_notch)

# Background
# Data
filename = "ccd_bg_30_10"
t, avg, x, y, y_e = get_ccd_data(filename)

rate_bg = y / t
rate_bg_e = y_e / t
np.save(npy_dir + 'ccd_rate_bg', rate_bg)
np.save(npy_dir + 'ccd_rate_bg_e', rate_bg_e)

# CS2
As = []
x0s = []
suffixes = ['_notch', '_notch_l2']
labels = ['CS$_2$, no $\lambda / 2$', 'CS$_2$, $\lambda / 2$']

for i, suffix, label in enum(suffixes, labels):
    filename = "ccd_cs2" + suffix
    t, avg, x, y, y_e = get_ccd_rate(filename)

    # Remove outliers 
    # identified by largest absolute value of difference between point and its two neighbours.
    for j in range(1):
        out = np.argmax(abs(2 * y[1:-1] - y[:-2] - y[2:])) + 1
        x, y = np.delete([x, y], out, 1)
        y_e = np.delete(y_e, out)
    
    # Peak fit
    # Peaks: Stokes 2,3, Anti-Stokes 2
    x_ranges = [[544, 553],
                [553, 560], 
                [512, 517]]
    p0s = np.array([[551, 0.2, 10000, 0],
                    [555, 0.2, 2000, 0], 
                    [514, 0.2, 200, 0]])
    for x_range, p0 in zip(x_ranges, p0s):
        x_fit, fit_peak, c1 = bw_fit(x, y_e, x_range, p0, fit=True)
        x0s.append(c1[0])
        As.append(c1[2])
    
# Print Latex table
lamb_all = np.array(x0s).reshape(2, 3)

d_lin = np.load(npy_dir + 'ccd_calibration.npy') 
lamb_all = linear(lamb_all, *d_lin) # Integrate error on wavelength of CCD

lamb_mean = np.sort(np.mean(lamb_all, 0)) 
lamb_laser = np.load(npy_dir + 'ccd_lamb_laser.npy')[0]

dnu = (1 / lamb_mean - 1 / lamb_laser) * 10**7

lits = np.array(['658', '658', '?']) # S, S, AS for CS2

for i, x0, dnu_cm, lit in enum(lamb_mean, lamb_to_cm(lamb_mean), lits):
    print("\cellcolor{LightCyan}$%i$ & $%s$ & $%s$ & $%s$   \\\\"%(
            i+1, uc_str(x0), uc_str(dnu_cm), lit))

# CHCl_3

As = []
x0s = []
suffixes = ['_notch_l2'] # peaks not visible for data without l2 plate
labels = ['CHCl$_3$, $\lambda / 2$']

for i, suffix, label in enum(suffixes, labels):
    filename = "ccd_chcl3" + suffix
    t, avg, x, y, y_e = get_ccd_rate(filename)

    # Remove outliers 
    # identified by largest absolute value of difference between point and its two neighbours.
    for j in range(2):
        out = np.argmax(abs(2 * y[1:-1] - y[:-2] - y[2:])) + 1
        x, y = np.delete([x, y], out, 1)
        y_e = np.delete(y_e, out)
   
    # Peak fit
    # Peaks: Stokes 1 - 4, Anti-Stokes 1, 2
    x_ranges = [[541, 548],
                [548, 552.5], 
                [553, 560], 
                [566, 572], 
                [519, 523], 
                [512, 516], 
                [509, 512]]
    p0s = np.array([[543, 0.2, 2000, 600],
                    [551.5, 0.2, 2000, 600], 
                    [554, 0.4, 700, 600], 
                    [569, 0.3, 700, 600], 
                    [522, 0.3, 1000, 100], 
                    [514, 0.3, 1000, 100], 
                    [511, 0.3, 100, 100]])
    for j, x_range, p0 in enum(x_ranges, p0s):
        x_fit, fit_peak, c1 = bw_fit(x, y_e, x_range, p0, fit=True)
        x0s.append(uc.ufloat(c1[0].n, c1[1].n))
        As.append(c1[2])
    
# Print Latex table
lamb_all = np.sort(np.array(x0s))
d_lin = np.load(npy_dir + 'ccd_calibration.npy') 
lamb_all = linear(lamb_all, *d_lin) # Integrate error on wavelength of CCD
lamb_laser = np.load(npy_dir + 'ccd_lamb_laser.npy')[0]

dnu = (1 / lamb_all - 1 / lamb_laser) * 10**7 # in cm^-1

lits = np.array([774, 680, 366, 366, 680, 774, 1220]) # S, S, AS for CS2

for i, x0, dnu_cm, lit in enum(lamb_all, lamb_to_cm(lamb_all), lits):
    print("\cellcolor{LightCyan}$%i$ & $%s$ & $%s$ & $%i$   \\\\"%(
            i+1, uc_str(x0), uc_str(dnu_cm), lit))

# CCl4

As = []
x0s = []
corr0 = np.load(npy_dir + 'ccd_corr0.npy') # Correction factor for polarization
corr90 = np.load(npy_dir + 'ccd_corr90.npy') # Correction factor for polarization
suffixes = ['l2_pol0', 'l2_pol90']
labels = ['CCl$_4$, $0^\circ \hat{=} \perp$ polarized', 'CCl$_4$, $90^\circ \hat{=} \parallel$ polarized']

for i, suffix, label in enum(suffixes, labels):
    filename = "ccd_ccl4_" + suffix
    t, avg, x, y, y_e = get_ccd_rate(filename)
    
    # Correction for polarization
    if suffix[-2:] == '90':
        corr = corr90
    else:
        corr = corr0
    y = corr * y
    y_e = corr * y_e  

    # Remove outliers 
    # identified by largest absolute value of difference between point and its two neighbours.
    for j in range(5):
        out = np.argmax(abs(2 * y[1:-1] - y[:-2] - y[2:])) + 1
        x, y = np.delete([x, y], out, 1)
        y_e = np.delete(y_e, out)

    # Peak fit
    # Peaks: Stokes 1 - 3, Anti-Stokes 1, 2
    x_ranges = [[537, 543],
                [544, 552],
                [548, 560], 
                [575, 585], 
                [520, 526],
                [514, 520], 
                [507, 516]]
    p0s = np.array([[541, 0.2, 1000, 0],
                    [546, 0.2, 2000, 0],
                    [555, 0.2, 400, 0], 
                    [579, 0.2, 60, 0], 
                    [523, 0.2, 200, 0],
                    [519, 0.2, 200, 0], 
                    [511, 0.2, 30, 0]])
    for x_range, p0 in zip(x_ranges, p0s):
        x_fit, fit_peak, c1 = bw_fit(x, y_e, x_range, p0, fit=True)
        x0s.append(uc.ufloat(c1[0].n, c1[1].n))
        As.append(c1[2])
    
    
# Print results to latex tables
lamb_all = np.array(x0s).reshape(2, 7)
d_lin = np.load(npy_dir + 'ccd_calibration.npy') 
lamb_all = linear(lamb_all, *d_lin) # Integrate error on wavelength of CCD

lamb_mean = np.sort(np.mean(lamb_all, 0)) 
lamb_laser = np.load(npy_dir + 'ccd_lamb_laser.npy')[0]

dnu = (1 / lamb_mean - 1 / lamb_laser) * 10**7

lits = np.array(['776', '459', '314', '314', '459', '776', '?']) # S, S, S, AS, AS for CCl4

for i, x0, dnu_cm, lit in enum(lamb_mean, lamb_to_cm(lamb_mean), lits):
    print("\cellcolor{LightCyan}$%i$ & $%s$ & $%s$ & $%s$   \\\\"%(
            i+1, uc_str(x0), uc_str(dnu_cm), lit))

I_s = np.array(As).reshape(2, 7).T[[-1, -2, -3, 0, 1, 2, 3]].T

I_s = np.delete(I_s, -1, 1) # Remove the value of the unknown peak!

rhos = I_s[0] / I_s[1]
rho_lits = [0.76, 0.03, 0.71, 0.71, 0.03, 0.76]
print('\n Depolarization')
for i, lit, Iperp, Ipara, rho, rho_lit in enum(lits[:-1], I_s[0],  I_s[1], rhos, rho_lits):
    print("\cellcolor{LightCyan}$%i$ & $%s$ & $%s$ & $%s$ & $%s$ & $%.2f$ \\\\"%(
            i+1, lit, uc_str(Iperp), uc_str(Ipara), uc_str(rho) ,rho_lit))

# Ethanol

As = []   
x0s = []
two_x0 = []
concentration = [10, 20, 30, 40, 60, 70, 80, 100, 'orig']
labels = [str(conc) + r'\%' for conc in [10, 20, 30, 40, 60, 70, 80, 100]]  + ['original sample']
corr = np.load(npy_dir + 'ccd_corr0.npy') # Correction factor for polarization

for i, conc_i, label in enum(concentration, labels):
    print(str(conc_i) + '% ethanol')
    filename = "ccd_et_" + str(conc_i)
    t, avg, x, y, y_e = get_ccd_rate(filename)
    # Correction for polarization
    y = corr * y
    y_e = corr * y_e
    
    # Remove outliers 
    # identified by largest absolute value of difference between point and its two neighbours.
    for j in range(4):
        out = np.argmax(abs(2 * y[1:-1] - y[:-2] - y[2:])) + 1
        x, y = np.delete([x, y], out, 1)
        y_e = np.delete(y_e, out)

    # Flourescence: fit for 550 - 590 nm w/out peaks
    x_peaks = np.array([[556, 560], [562, 566], [569, 572], [574, 580]])
    where_peaks = [(x_low < x) * (x < x_high) for x_low, x_high in x_peaks]
    mask_no_peak = np.where(sum(where_peaks) == 0)[0]
    x_np = x[mask_no_peak]
    y_np = y[mask_no_peak]
    y_np_e = y_e[mask_no_peak]
  
    x_range_poly = [550,  610]
    p0 = np.array([0, 0, 0, 0, 0])
    p =  poly_fit(x_np, y_np_e, x_range_poly, p0)
    p_n = un.nominal_values(p)
    x_poly = np.linspace(x_range_poly[0], x_range_poly[1], 200)
    y_poly = poly(x_poly, *p_n)

    # Peak fit: Data - flourescence
    y_fit = y - poly(x, *p_n)
    y_fit_e = y_e - poly(x, *p)

    # Peak fit
    # Peaks: Stokes 2,3, Anti-Stokes 2
    x_ranges = [[554, 561],
                [569, 573], 
                [573, 580]]
    p0s = np.array([[558, 0.2, 100, 0],
                    [571, 0.2, 50, 0], 
                    [577, 0.2, 50, 0]])

    for j, x_range, p0 in enum(x_ranges, p0s):
        x_fit, fit_peak, c1 = bw_fit(x, y_fit_e, x_range, p0, fit=True)
        As.append(c1[2])
        x0s.append(c1[0])
    
    # double peak
    x_range_two = [562, 566]
    p0_two = np.array([563, 0.2, 100, 565, 0.2, 100, 0])
    x_fit2, fit_peak2, c2 = two_bw_fit(x, y_fit_e, x_range_two, p0_two, fit=True)
    x0s.append(c2[0])
    x0s.append(c2[3])
    As.append(c2[2])
    As.append(c2[5])

# Linear fit on ethanol percentage

from scipy.odr import ODR, Model, Data, RealData

def func(beta, x):
    y = beta[0] + beta[1] * x
    return y

def linear_beta(x, beta_0, beta_1):
    return beta_0 + beta_1 * x

intensity = np.array(As).reshape(9, 5).T[[0, 3, 4, 1, 2]].T # move peak 2 & 3 to the correct position
A_lambda = intensity[:-1].T # entries of each peak put together
peaks = ['558.3', '563.5', '564.7', '570.8', '576.7']
intersects = []

x = np.array(concentration[:-1])
sx = 4
for i, peak, A in enum(peaks, A_lambda):
    y = un.nominal_values(A)
    sy = un.std_devs(A)


    data = RealData(x, y, sx=sx, sy=sy)
    model = Model(func)

    odr = ODR(data, model, [6, 0])
    odr.set_job(fit_type=0)
    output = odr.run()

    beta = uc.correlated_values(output.beta, output.cov_beta)
    
    fit = linear_beta(x_fit, *beta)
    
    #Intersects
    y0 = intensity[-1][i] # peak intensities of the original sample
    x0 = (y0 - beta[0]) / beta[1]
    intersects.append(x0)

# Print Latex table
lamb_all = np.array(x0s).reshape(9, 5)

d_lin = np.load(npy_dir + 'ccd_calibration.npy') 
lamb_all = linear(lamb_all, *d_lin) # Integrate error on wavelength of CCD

lamb_mean = np.mean(lamb_all, 0) 
lamb_laser = np.load(npy_dir + 'ccd_lamb_laser.npy')[0]

dnu = (1 / lamb_mean - 1 / lamb_laser) * 10**7

lamb_mean = np.sort(lamb_mean)
lits = np.array([888, 1028, 1091, 1274, 1464]) 

for i, x0, dnu_cm, lit,intersect in enum(lamb_mean, lamb_to_cm(lamb_mean), lits, intersects):
    print("\cellcolor{LightCyan}$%i$ & $%s$ & $%s$ & $%s$ & $%s$  \\\\"%(
            i+1, uc_str(x0), uc_str(dnu_cm), lit, uc_str(intersect)))
    
# Ethanol concentration in the unknown original sample.
print('\n Ethanol concentration of unknown sample: ', uc_str(np.mean(intersects)))
print('Standard deviation: ', np.std(un.nominal_values(intersects)))

# Sulfur spectrum - Temperatur measurement

corr = np.load(npy_dir + 'ccd_corr0.npy') # Correction factor for polarization
rate_bg = np.load(npy_dir + 'ccd_rate_bg.npy')
rate_bg_e = np.load(npy_dir + 'ccd_rate_bg_e.npy')

I_laser_all = [70, 80, 90, 100, 110, 120, 130, 140, 150]
labels = ['%.2f A'%(I_l * 0.01) for I_l in I_laser_all]
    
x0s = []
As = []
print('Calculate temperature')
for i, I_laser, label in enum(I_laser_all, labels):
    print(label)
    filename = "ccd_s_" + str(I_laser).zfill(3)
    t, avg, x, y, y_e = get_ccd_rate(filename)
    # Correction for polarization
    y = corr * y
    y_e = corr * y_e
    
    x_ranges = [[545, 549],
                [510, 519]]
    p0s = np.array([[545.5, 0.2, 20000, 3500],
                    [518, 0.2, 5000, 200]])

    for j, x_range, p0 in enum(x_ranges, p0s):
        x_fit, fit_peak, c1 = bw_fit(x, y_e, x_range, p0, fit=True)
        x0s.append(c1[0])
        As.append(c1[2])

intensities = np.array(As).reshape(9, 2).T # stokes and anti-stokes
A_ratio = intensities[1] / intensities[0]
y = un.nominal_values(A_ratio)
sy = un.std_devs(A_ratio)

mean_ratio = np.mean(y)
std_ratio = np.std(y)
mean_ratio_e = uc.ufloat(mean_ratio, std_ratio)

# Print latex table
for I_l, Is, Ias, A_rat in zip(labels, intensities[0], intensities[1], A_ratio):
    print("\cellcolor{LightCyan}$%s$ A & $%s$ & $%s$ & $%s$   \\\\"%(
            I_l[:-2], uc_str(Is), uc_str(Ias), uc_str(A_rat, 5)))

print('\n')
print('Mean intensity ratio: ', uc_str(uc.ufloat(mean_ratio, std_ratio)))


# Print latex table of wavenumbers
lamb_all = np.array(x0s).reshape(9, 2)

d_lin = np.load(npy_dir + 'ccd_calibration.npy') 
lamb_all = linear(lamb_all, *d_lin) # Integrate error on wavelength of CCD

lamb_mean = np.mean(lamb_all, 0) 
lamb_laser = np.load(npy_dir + 'ccd_lamb_laser.npy')[0]

dnu = (1 / lamb_mean - 1 / lamb_laser) * 10**7

lits = np.array(['bla', 'blub']) # S, S, S, AS, AS for CCl4

print('\n Wavenumbers of both peaks:')
for i, x0, dnu_cm, lit in enum(lamb_mean, lamb_to_cm(lamb_mean), lits):
    print("\cellcolor{LightCyan}$%i$ & $%s$ & $%s$ & $%s$   \\\\"%(
            i+1, uc_str(x0), uc_str(dnu_cm), lit))
    
# Calculation of temperature (uses SI units!)
from scipy.constants import k, c, h
nu_l_si = c / lamb_laser
dnu_si = np.mean(abs(c / lamb_mean - nu_l_si)) # in Hz!!!
T = -(h * dnu_si * 10**9) / (k * un.log(mean_ratio_e * ((nu_l_si - dnu_si) / (nu_l_si + dnu_si))**4))
print('\n\Delta \\nu = ' + uc_str(dnu_si * 10**-2) + ' Hz')
print('Temperatur T = ' + uc_str(T) + ' K')
print('Temperatur T = ' + uc_str(T - 273.15) + ' ^\circ C')
