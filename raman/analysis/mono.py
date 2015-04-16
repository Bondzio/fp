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
sx0_global = 0.2

def breit_wigner(x, x0, gamma, amplitude, offset):
    """
    Breit-Wigner or Cauchy distribution with location parameter x0 
    and scale parameter gamma = 0.5 * FWHM. Amplitude: 1 / (Pi * gamma). 
    No mean, variance or higher moments exist.
    """
    return amplitude / ((np.pi*gamma) * (1 + ((x - x0) / gamma)**2)) + offset
   
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

def linear(x, a, b):
    return (a*x + b)

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
    Converts  wavenumber (in cm^-1) of vibrational mode into corresponding Stokes peak in nm (for Nd:Yag laser)
    """
    #lamb_laser = 532 # nm, Nd:Yag frequency doubled
    lamb_laser = np.load(npy_dir + 'ccd_lamb_laser.npy')[0] # measured laser wavelength
    lamb_stokes = 1 / (1 / lamb_laser - dnu_cm * 10**-7) 
    return(lamb_stokes)

def uc_str(c):
    """
    input format: uc.ufloat
    rounds float and corrisponding error to last significant digit
    returns float and error as string
    * as integers with max 4 error digits
    * as floats with max 3 error digits
    * as exp else
    """
    digit = -int(np.floor(np.log10(c.s)))    
    if (c.s * 10**digit) < 1.5: # convention...
        digit += 1
    c_r = round(c.n, digit)
    s_c_r = round(c.s, digit)
    if (-3 < digit) * (digit <= 0): # returns readable integers
        c_str = '%i \pm %i'%(c_r, s_c_r)
    elif (0 < digit) * (digit < 5): # returns readable floats (max 3 digits)
        c_str = ('%.' + str(digit) + 'f \pm %.' + str(digit) + 'f')%(c_r, s_c_r)
    else: # returns exp
        c_str = ('(%.1f \pm %.1f)\mathrm{e}%i')%(c_r * 10**(digit-1), s_c_r * 10**(digit-1), -(digit-1))
    return c_str

# Calibration
filename = "mono_hg"
x = np.load("npy/"+filename+"_lamb.npy") / 10 
y = np.load("npy/"+filename+"_count.npy")
y_e = un.uarray(y, np.sqrt(np.maximum(1, y)))

# Peaks: Hg peaks: 404.7, 407.8, 435.8, 546.1, 577.1, 579.1 nm
# Fits do not converge. We search for local maxima in the predefined ranges. 
# Errors are set to 0.2 nm, twice the resolution of the data. 
x_ranges = [[402, 406],
            [406, 410], 
            [430, 440], 
            [540, 550], 
            [570, 578], 
            [578, 584]]
labels = ['404.7 nm', '407.8 nm', '435.8 nm', '546.1 nm', '577.1 nm', '579.1 nm']
x0s = []
for i, x_range, label in zip(range(len(x_ranges)), x_ranges, labels):
    x_min, x_max = x_range
    mask = (x > x_min) * (x < x_max)
    x_fit = x[mask]
    y_fit = y[mask]
    x0 = x_fit[np.argmax(y_fit)]
    x0s.append(uc.ufloat(x0, sx0_global))

# Print results to latex table
lits = [404.7, 407.8, 435.8, 546.1, 577.1, 579.1]
i=0
for x0, lit in zip(x0s, lits):
    i += 1
    print("\cellcolor{LightCyan}$%i$ & $%s$ & $%.1f$   \\\\"%(
            i, uc_str(x0), lit))

x0 = un.nominal_values(x0s)
s_x0 = un.std_devs(x0s)

lit_peaks = np.array([404.7, 407.8, 435.8, 546.1, 577.1, 579.1])

coeff_lin, cov_lin = curve_fit(linear, lit_peaks, x0, p0=None, 
                               sigma=np.sqrt(s_x0), absolute_sigma=True)
c_lin = uc.correlated_values(coeff_lin, cov_lin)
# Switch to lambda(x0) = lit_peak(x0)
d_lin = np.array([1 / c_lin[0], -c_lin[1] / c_lin[0]])
np.save(npy_dir + 'mono_calibration', d_lin)

# Slit width
widths = [200, 150, 100, 75, 50, 40, 25]
for i, width in enumerate(widths):
    filename = 'mono_slit_' + str(width)
    x = np.load("npy/" + filename + "_lamb.npy") / 10
    x = x[1:]
    y0 = np.load("npy/" + filename + "_count.npy")
    y0 = y0[1:]

# Boundary of detection
filename = "mono_white_pol0"
x = np.load("npy/"+filename+"_lamb.npy") / 10
x = x[1:]
y0 = np.load("npy/"+filename+"_count.npy")
y0 = y0[1:]

filename = "mono_white_pol90"
y90 = np.load("npy/"+filename+"_count.npy")
y90 = y90[1:]

# CS2
filename = 'mono_cs2_04' # 04: Stokes at 551 nm
x = np.load("npy/"+filename+"_lamb.npy") / 10
x = x[1:]
y = np.load("npy/"+filename+"_count.npy")
y = y[1:]

x_ranges = [[549, 554]]
labels = ['551.3 nm']
x0s = []
for i, x_range, label in zip(range(len(x_ranges)), x_ranges, labels):
    x_min, x_max = x_range
    mask = (x > x_min) * (x < x_max)
    x_fit = x[mask]
    y_fit = y[mask]
    x0 = x_fit[np.argmax(y_fit)]
    x0s.append(uc.ufloat(x0, sx0_global))

# CHCl3
filename = 'mono_chcl3_01' # 01 & 02: 3S, 1 AS
x = np.load("npy/"+filename+"_lamb.npy") / 10
x = x[1:]
y = np.load("npy/"+filename+"_count.npy")
y = y[1:]

x_ranges = [[538, 540], [541, 544], [550, 552.7]]
x0s = []
for x_range in x_ranges:
    x_min, x_max = x_range
    mask = (x > x_min) * (x < x_max)
    x_fit = x[mask]
    y_fit = y[mask]
    x0 = x_fit[np.argmax(y_fit)]
    x0s.append(uc.ufloat(x0, sx0_global))
x0s = np.array(x0s)
lits = np.sort(np.array([680, 366, 260]))
i=0
for x0, dnu_cm, lit in zip(x0s, lamb_to_cm(x0s), lits):
    i += 1
    print("\cellcolor{LightCyan}$%i$ & $%s$ & $%s$ & $%i$   \\\\"%(
            i, uc_str(x0), uc_str(dnu_cm), lit))

filename = "mono_ccl4_01"
#filename = "mono_ccl4_overnight"
# Relevant peaks are visible in both measurements...
x = np.load("npy/"+filename+"_lamb.npy") / 10
x = x[1:]
y = np.load("npy/"+filename+"_count.npy")
y = y[1:]

x_ranges = [[525, 527], [537, 539.5], [540, 542.5], [543.5, 547]]
labels = ['551.3 nm']
x0s = []
for x_range in x_ranges:
    x_min, x_max = x_range
    mask = (x > x_min) * (x < x_max)
    x_fit = x[mask]
    y_fit = y[mask]
    x0 = x_fit[np.argmax(y_fit)]
    x0s.append(uc.ufloat(x0, sx0_global))
x0s = np.array(x0s)
lits = np.sort(np.array([459, 217, 217, 314]))
i=0
for x0, dnu_cm, lit in zip(x0s, lamb_to_cm(x0s), lits):
    i += 1
    print("\cellcolor{LightCyan}$%i$ & $%s$ & $%s$ & $%i$   \\\\"%(
            i, uc_str(x0), uc_str(dnu_cm), lit))
