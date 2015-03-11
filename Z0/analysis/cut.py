import numpy as np
from scipy.optimize import curve_fit

import pickle
import uncertainties as uc
import uncertainties.unumpy as un
import matplotlib.pyplot as plt
import sys

chars=["Electrons","Muons","Taons","Hadrons"]

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

costhetamin = -0.9
costhetamax = 0.9
costhrumin = -0.9
costhrumax = 0.9



def classify(u, key):

    N_all = np.zeros(4)
    C_eff = np.zeros([4,4])

    if key == "simple":

        # Electrons
        c_ee = (u["E_ecal"]   >= 60)*(u["E_ecal"] <= 120)\
               * np.logical_or((u["Pcharged"] >= 40)*(u["Pcharged"] <= 100),(u["Pcharged"] == 0))\
               * (u["Ncharged"] >= 0)*(u["Ncharged"] <= 10)
        pl("found %d electrons \t= %.3f %%"%(sum(c_ee),100*sum(c_ee)/len(u)),5)
        
        rest = np.invert(c_ee)
        
        # Muons 
        c_mm = rest * (u["Pcharged"] >= 60)*(u["Pcharged"] <= 120)\
                    * (u["E_ecal"] >= 0)*(u["E_ecal"] <= 40)\
                    * (u["Ncharged"] >= 1)*(u["Ncharged"] <= 10) 
        rest = np.invert(c_mm)*rest
        pl("found %d muons \t= %.3f %%"%(sum(c_mm),100*sum(c_mm)/len(u)),5)
        
        # Hadrons
        c_qq = rest * (u["Ncharged"] >= 10)\
                    * (u["E_ecal"] >= 20)*(u["E_ecal"] <= 100)\
                    * (u["Pcharged"] >= 0)*(u["Pcharged"] <= 100)
        rest = np.invert(c_qq)*rest
        pl("found %d hadrons \t= %.3f %%"%(sum(c_qq),100*sum(c_qq)/len(u)),5)
        
        # Taons 
        c_tt = rest * (u["Ncharged"] >= 0)*(u["Ncharged"] <= 40)\
                    * (u["E_ecal"] >= 0)*(u["E_ecal"] <= 100)\
                    * (u["Pcharged"] >= 0)*(u["Pcharged"] <= 100)
        rest = np.invert(c_tt)*rest
        pl("found %d tauons \t= %.3f %%"%(sum(c_tt),100*sum(c_tt)/len(u)),5)
        
        
        pl("We did not classify %d particles = %.3f%%\n"%(sum(rest),100*sum(rest)/len(u)),4)

    if key == "simple2":

        # Electrons
        c_ee = (u["E_ecal"]   >= 60)*(u["E_ecal"] <= 120)\
               * np.logical_or((u["Pcharged"] >= 0)*(u["Pcharged"] <= 200),(u["Pcharged"] == 0))\
               * ((u["E_ecal"]+u["E_hcal"])> 40)\
               * (u["Ncharged"] >= 1)*(u["Ncharged"] <= 10)\
               * (u["E_hcal"] >= 0)*(u["E_hcal"] <= 20)
        pl("found %d electrons \t= %.3f %%"%(sum(c_ee),100*sum(c_ee)/len(u)),5)
        
        rest = np.invert(c_ee)
        
        # Muons 
        c_mm = rest * (u["Pcharged"] >= 60)*(u["Pcharged"] <= 200)\
                    * (u["E_ecal"] >= 0)*(u["E_ecal"] <= 50)\
                    * (u["Ncharged"] >= 1)*(u["Ncharged"] <= 10) 
        rest = np.invert(c_mm)*rest
        pl("found %d muons \t= %.3f %%"%(sum(c_mm),100*sum(c_mm)/len(u)),5)

        # Taons 
        c_tt = rest * (u["Ncharged"] >= 0)*(u["Ncharged"] <= 10)\
                    * (u["E_ecal"] >= 0)*(u["E_ecal"] <= 100)\
                    * (u["Pcharged"] >= 0)*(u["Pcharged"] <= 100)
        rest = np.invert(c_tt)*rest
       
        # Hadrons
        c_qq = rest * (u["Ncharged"] >= 10)\
                    * (u["E_ecal"] >= 25)*(u["E_ecal"] <= 90)
        rest = np.invert(c_qq)*rest
        pl("found %d hadrons \t= %.3f %%"%(sum(c_qq),100*sum(c_qq)/len(u)),5)
        
        pl("found %d tauons \t= %.3f %%"%(sum(c_tt),100*sum(c_tt)/len(u)),5)
        
        
        pl("We did not classify %d particles = %.3f%%\n"%(sum(rest),100*sum(rest)/len(u)),4)

    if key == "alt":

        # Electrons
        c_ee = (u["E_ecal"]   >= 80)\
               * (u["Ncharged"] >= 1)*(u["Ncharged"] <= 5)
        pl("found %d electrons \t= %.3f %%"%(sum(c_ee),100*sum(c_ee)/len(u)),5)
        
        rest = np.invert(c_ee)
        
        # Muons 
        c_mm = rest * (u["Pcharged"] >= 80)*(u["Pcharged"] <= 200)\
                    * (u["E_ecal"] >= 0)*(u["E_ecal"] <= 10)
        rest = np.invert(c_mm)*rest
        pl("found %d muons \t= %.3f %%"%(sum(c_mm),100*sum(c_mm)/len(u)),5)

        # Taons 
        c_tt = rest * (u["Ncharged"] <= 7)*(u["Ncharged"] <= 10)\
                    * (u["E_ecal"] >= 0)*(u["E_ecal"] <= 70)\
                    * (u["Pcharged"] >= 0)*(u["Pcharged"] <= 70)
        rest = np.invert(c_tt)*rest
       
        # Hadrons
        c_qq = rest * (u["Ncharged"] >= 7)
        rest = np.invert(c_qq)*rest
        pl("found %d hadrons \t= %.3f %%"%(sum(c_qq),100*sum(c_qq)/len(u)),5)
        
        pl("found %d tauons \t= %.3f %%"%(sum(c_tt),100*sum(c_tt)/len(u)),5)
        
        
        pl("We did not classify %d particles = %.3f%%\n"%(sum(rest),100*sum(rest)/len(u)),4)

        
    ff = [u[c_ee],u[c_mm],u[c_tt],u[c_qq]]

    return ff 

def cut(u, key):

    #u = u[(u["cos_thru"]>costhrumin)\
    #        *(u["cos_thru"]<costhrumax)]

    u = u[(u["Pcharged"]!=0)]
    ff = classify(u,key)
    N_all = np.array([len(k) for k in ff])
    N_all = un.uarray(N_all, np.sqrt(N_all))
    ## Cut the s and t channel
    if len(ff[0]) > 100:
        # Cutting all electrons within costheta range

        ff[0] = ff[0][(ff[0]["Pcharged"]!=0)*(ff[0]["cos_thet"]>costhetamin)\
            *(ff[0]["cos_thet"]<costhetamax)]

        pl("Removing the t channel now.\n",2) 

        def ts_channel(cos_theta,A,B):
            return A*(1 + cos_theta**2) + B * (1 - cos_theta)**(-2) 

        # inital guess
        A = 1
        B = 1

        N_ee, cos_thet = np.histogram(ff[0]["cos_thet"],250, density=True)
        try:
            #sigma = np.sqrt(N_ee), absolute_sigma = True 
            p, cov = curve_fit(ts_channel, cos_thet[1:], N_ee, p0=[A,B],sigma = np.sqrt(N_ee+1), absolute_sigma = True)
            p_uc = uc.correlated_values(p,cov)
            A,B = p_uc


            # Indefinite Integral of s 
            F = lambda x: 0.25*(6*x + np.sin(2*x))

            thetamin =  np.arccos(costhetamin)
            thetamax =  np.arccos(costhetamax)
        
            # Definite Integral of s 
            sigma = -A*(F(thetamax)-F(thetamin))
            if sigma.n > 1 or sigma.s >1:
                raise Exception("fit failed")
            pl("%.f ± %.f %% of the electrons are in s channel\n"%((100*sigma).n,(100*sigma).s),0)
            pl("Change from %.f ± %.f to %.f ± %.f electrons. \n"%(N_all[0].n,N_all[0].s,(N_all[0]*sigma).n,(N_all[0]*sigma).s),0)
            
            N_all[0] *= sigma
        except:
            pl("Removing t-channel failed. ",4)
            pl("Press any KEY to continue...",4)
            x = input()
    return N_all
def N(N):
    return un.uarray(N,np.sqrt(N))

def get_c_eff(cut_type):

    pl("\n Creating new efficiency matrix",1)
    ee = np.load("data/ee.npy")
    mm = np.load("data/mm.npy")
    tt = np.load("data/tt.npy")
    qq = np.load("data/qq.npy")

    ff = [ee,mm,tt,qq]

    # Removing bulk 
    #for k in range(4):
    #    ff[k] = ff[k][(ff[k]["cos_thru"]>costhrumin)\
    #        *(ff[k]["cos_thru"]<costhrumax)]
    ff[0] = ff[0][(ff[0]["cos_thet"]>costhetamin)\
        *(ff[0]["cos_thet"]<costhetamax) * (ff[0]["Pcharged"]!=0)]


    # Branching Ratios from Particle Data Booklet 2015



    br   = [ 3.363, 3.366, 3.370, 69.91]
    br_s = [ 0.004, 0.007, 0.008, 0.06] 
    br = un.uarray(br,br_s)

    br/=np.sum(br)

    N_monte = N([len(k) for k in ff])
    br_monte = N_monte / np.sum(N_monte)

    ratio = br  / br_monte

    C_eff = np.zeros([4,4])

    all_ee = 0
    all_mm = 0
    all_tt = 0
    all_qq = 0

    N_all_mat = un.umatrix(np.zeros([4,4]),np.zeros([4,4]))
    C_eff = un.umatrix(np.zeros([4,4]),np.zeros([4,4]))

    for u_i,u in enumerate(ff):

        pl("\nnow looking for %s, total: %d"%(chars[u_i],len(u)),4)
        N_found = cut(u,cut_type) 
        N_all_mat[u_i,:] = N_found* ratio[u_i]
        C_eff[u_i,:] = N_found / N_monte[u_i]

    N_all = np.array(np.sum(N_all_mat,0)).reshape(4)
    C_eff = np.swapaxes(C_eff,0,1)
    N_all_corrected = np.array(np.dot(C_eff.I,N_all)).reshape(4)
    pl("Now we check whether the efficiency \n matrix does what it is supposed to do. \n Notice how the error gets smaller,\n because we tracked the correlation between \n the efficiency matrix and the numbers.",2)
    for k in range(4):
        pl("Uncorrected vs corrected number of %s"%(chars[k]),6)
        print("%.f ± %.f "%(N_all[k].n,N_all[k].s))
        print("%.f ± %.f "%(N_all_corrected[k].n,N_all_corrected[k].s))

    pickle.dump(C_eff, open("data/cuts/%s.p"%cut_type,"wb"))

    return C_eff

