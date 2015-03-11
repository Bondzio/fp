# coding: utf-8
import numpy as np
import pickle
from numpy.linalg import inv as inverse
import sympy as sy
import sys

import os.path
from scipy.optimize import curve_fit
from scipy.optimize import leastsq

import uncertainties as uc
import uncertainties.unumpy as un


chars = ["Electrons","Muons","Taons","Hadrons"]

if len(sys.argv) == 2:
    cut_type = sys.argv[1]
    print("cut type is |%s|, press any KEY to continue ..."%cut_type)
    x = input()
else:
    cut_type = "simple"

bcolors = [
    '\033[95m',
    '\033[94m',
    '\033[92m',
    '\033[93m',
    '\033[91m',
    '\033[1m',
    '\033[4m',
    '\033[0m']

pl= lambda s,i: print(bcolors[i] + s + bcolors[-1])

pl("Loading data ...",0)
all_data = np.load("data/data.npy")
E_lep = np.unique(all_data["E_lep"])*2
f = open("data/original/daten_4.lum")
lumidata = f.read()
f.close()
mean_E = []
lumi   = {}

for q in range(7):
    E = float(lumidata.split()[9+q*5])
    mean_E += [E]
    lumi[E] = uc.ufloat(float(lumidata.split()[10+q*5]),float(lumidata.split()[13+q*5]))

pl(lumidata,1)
all_data_sorted = {}

for E in mean_E:
    c_data = (all_data["E_lep"]*2 < (E + 0.5)) * (all_data["E_lep"]*2 > (E - 0.5))
    all_data_sorted[E]= all_data[c_data]
    
pl("We will now calculate everything for each energy.\n",2) 

from cut import get_c_eff
from cut import cut

always_recalculate = True

if os.path.isfile("data/cuts/%s.p"%cut_type) and always_recalculate == False:
    C_eff = pickle.load(open("data/cuts/%s.p"%cut_type,"rb"))
else:
    C_eff = get_c_eff(cut_type) 

C_eff_inv= C_eff.I

def calc_all(data, E_now):

    pl("Operating at %.3f GeV now.\n------------\n"%E_now,0)

    # Calculate all the stuff we need
    # 1. Cuts: Classification of charged particles
    # 2. Removing t channel of electrons
    # 3. Adjusting cut errors of inverse efficiency matrix from montecarlo data 
    
    ## Cuts

    pl("Cuts: Classifying the charged particles\n",2)
    
    # We impose for now that there are no intersections 

    N_all = cut(data,cut_type)

    pl("Correction with respect to the cuts ",2)
    pl("with montecarlo efficiency matrix\n ",2)
    N_all_corrected = np.array(np.dot(C_eff_inv,N_all)).reshape(4)
    pl("particle numbers:",3)
    for k in range(4):
        pl("Uncorrected vs corrected number of %s"%(chars[k]),6)
        print("%.f ± %.f "%(N_all[k].n,N_all[k].s))
        print("%.f ± %.f "%(N_all_corrected[k].n,N_all_corrected[k].s))

    return N_all_corrected / lumi[E_now]


E_now = 91.22430
crosssections = {}

for E_now in mean_E:
    crosssections[E_now] = calc_all(all_data_sorted[E_now], E_now)

pickle.dump(crosssections, open("crosssection.p","wb"))

E = (np.array(4*list(crosssections.keys())).reshape(4,7).swapaxes(0,1))
cross_total = un.nominal_values(list(crosssections.values()))
cross_total_error = un.std_devs(list(crosssections.values()))

def Breit_Wigner(s,p):
    gamma_e, gamma_m,gamma_t,gamma_h,gamma_Z,Mz = p
    gamma_f = np.array([gamma_e,gamma_m,gamma_t,gamma_h])
    return 12*np.pi / Mz**2 * (gamma_e*gamma_f*s)/((s - Mz**2)**2 +(s*gamma_Z/Mz)**2)

p0 = [80,80,80,1600,2.4,90]
def residuals(p,error):
    weights = 1 / error
    return (weights*(cross_total - Breit_Wigner(E**2,p))).flatten()

pl("Now fitting the crosssections.\n------------\n",0)

p,cov,infodict,mesg,ier = leastsq(residuals,p0,args=cross_total_error,full_output=True)
p_uc = uc.correlated_values(p, cov)
keys = ["gamma_e", "gamma_m","gamma_t","gamma_h","gamma_Z","Mz      "]
params = dict(zip(keys,p_uc))
gamma_e, gamma_m,gamma_t,gamma_h,gamma_Z,Mz = p_uc
for k in params:
    print(k,"\t",params[k])

## Leptonuniversality

V_mu = gamma_m / gamma_e
print("gamma_m / gamma_e",V_mu)

V_tau = gamma_t / gamma_e

print("gamma_t/gamma_e",V_tau)

## Number of Neutrino families

gamma_nu = 1000*gamma_Z - gamma_e - gamma_m - gamma_t - gamma_h
print("Number of neutrino families",gamma_nu/(3*250))