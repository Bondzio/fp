# coding: utf-8
import numpy as np
import pickle
from numpy.linalg import inv as inverse
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import sympy as sy
import prettyplotlib as ppl

from scipy.optimize import curve_fit


sns.set_style(style='white')
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
rcParams['axes.titlesize'] = fontsize_labels
rcParams['figure.figsize'] = (3*6.2, 3*3.83)  # in inches; width corresponds to \textwidth in latex document (golden ratio)

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
    lumi[E]   = float(lumidata.split()[10+q*5])
pl(lumidata,1)
all_data_sorted = {}

for E in mean_E:
    c_data = (all_data["E_lep"]*2 < (E + 0.5)) * (all_data["E_lep"]*2 > (E - 0.5))
    all_data_sorted[E]= all_data[c_data]
pl("We will now calculate everything for each energy.\n",2) 

C_eff = np.load("data/C_eff.npy")
C_eff_inv= inverse(C_eff)

def calc_all(data, E_now):

    pl("Operating at %.3f GeV now.\n------------\n"%E_now,0)

    # Calculate all the stuff we need
    # 1. Cuts: Classification of charged particles
    # 2. Removing t channel of electrons
    # 3. Adjusting cut errors of inverse efficiency matrix from montecarlo data 
    
    ## Cuts

    pl("Cuts: Classifying the charged particles\n",2)
    
    # We impose for now that there are no intersections 
    C_eff = np.zeros([4,4])
    N_all = np.zeros(4)
    u = data

    # Electrons
    c_ee = (u["E_ecal"]   >= 60)*(u["E_ecal"] <= 120)\
           * np.logical_or((u["Pcharged"] >= 40)*(u["Pcharged"] <= 100),(u["Pcharged"] == 0))\
           * (u["Ncharged"] >= 0)*(u["Ncharged"] <= 10)
    N_all[0] = sum(c_ee)
    pl("found %d electrons \t= %.3f %%"%(sum(c_ee),100*sum(c_ee)/len(u)),5)
    
    rest = np.invert(c_ee)
    
    # Muons 
    c_mm = rest * (u["Pcharged"] >= 0)*(u["Pcharged"] <= 120)                * (u["E_ecal"] >= 0)*(u["E_ecal"] <= 60)                * (u["Ncharged"] >= 1)*(u["Ncharged"] <= 10) 
    rest = np.invert(c_mm)*rest
    pl("found %d muons \t= %.3f %%"%(sum(c_mm),100*sum(c_mm)/len(u)),5)
    N_all[1] = sum(c_mm)
    
    # Hadrons
    c_qq = rest * (u["Ncharged"] >= 10)                 * (u["E_ecal"] >= 40)*(u["E_ecal"] <= 80)                 * (u["Pcharged"] >= 0)*(u["Pcharged"] <= 100)
    rest = np.invert(c_qq)*rest
    pl("found %d hadrons \t= %.3f %%"%(sum(c_qq),100*sum(c_qq)/len(u)),5)
    N_all[3] = sum(c_qq)
    
    # Taons 
    c_tt = rest * (u["Ncharged"] >= 0)*(u["Ncharged"] <= 40)                 * (u["E_ecal"] >= 0)*(u["E_ecal"] <= 100)                 * (u["Pcharged"] >= 0)*(u["Pcharged"] <= 100)
    rest = np.invert(c_tt)*rest
    pl("found %d tauons \t= %.3f %%"%(sum(c_tt),100*sum(c_tt)/len(u)),5)
    N_all[2] = sum(c_tt)
    
    
    pl("We did not classify %d particles = %.3f%%\n"%(sum(rest),100*sum(rest)/len(data)),4)



    ff = [data[c_ee],data[c_mm],data[c_tt],data[c_qq]]
    chars=["Electrons","Muons","Taons","Hadrons"]



    pl("Removing the t channel now.\n",2) 
    costhetamin = -0.9
    costhetamax = 0.9
    # Cutting all charged particles within costheta range
    ff = [u[(u["Pcharged"]!=0)*np.logical_or((u["cos_thet"]>costhetamin)*(u["cos_thet"]<costhetamax),[ui==3]*len(u))] for ui,u in enumerate(ff)]

    ## Cut the s and t channel

    def ts_channel(cos_theta,A,B):
        return A*(1 + cos_theta**2) + B * (1 - cos_theta)**(-2) 

    # inital guess
    A = 1
    B = 1

    N_ee, cos_thet = np.histogram(ff[0]["cos_thet"][(ff[0]["cos_thet"]>costhetamin)*(ff[0]["cos_thet"]<costhetamax)],250, density=True)

    [A,B], cov = curve_fit(ts_channel, cos_thet[1:], N_ee, p0=[A,B])

    # Indefinite Integral of s 
    F = lambda x: 0.25*(6*x + np.sin(2*x))

    thetamin =  np.arccos(costhetamin)
    thetamax =  np.arccos(costhetamax)
    
    # Definite Integral of s 
    N_s = -A*(F(thetamax)-F(thetamin))*len(ff[0])
    sigma = N_s / len(ff[0])
    pl("%.3f %% of the electrons are in s channel\n"%(100*sigma),0)



    pl("Correction with respect to the cuts ",2)
    pl("with montecarlo efficiency matrix\n ",2)

    N_all_corrected = np.round(np.dot(C_eff_inv,N_all))
    pl("corrected particle numbers:",3)
    pl(" electrons\t%.3f%%\n muons     \t%.3f%%\n taons    \t%.3f%%\n hadrons\t%.3f%%\n"%tuple(100* (N_all_corrected / N_all -1)),5)

    return N_all_corrected / lumi[E_now]

E_now = 91.22430
crosssection = {}

for E_now in mean_E:
    crosssection[E_now] = calc_all(all_data_sorted[E_now], E_now)

pickle.dump(crosssection, open("crosssection.p","wb"))
