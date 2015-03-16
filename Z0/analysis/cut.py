import numpy as np
from scipy.optimize import curve_fit

from sklearn import neighbors

import pickle
import os

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

def get_N(n):
    return un.uarray(n,np.sqrt(n))

def cut(u, key):

    targets = np.zeros(len(u)) 

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

    targets[c_ee] = 1
    targets[c_mm] = 2
    targets[c_tt] = 3
    targets[c_qq] = 4

    return targets 

def remove_t(u):

    ## Cut the s and t channel
    N0 = uc.ufloat(len(u),np.sqrt(len(u)))

    if len(u) > 100:
        # Cutting all electrons within costheta range
        u = u[(u["Pcharged"]!=0)*(u["cos_thet"]>costhetamin)\
            *(u["cos_thet"]<costhetamax)]

        pl("Removing the t channel now.\n",2) 

        def ts_channel(cos_theta,A,B):
            return A*(1 + cos_theta**2) + B * (1 - cos_theta)**(-2) 

        # inital guess
        A = 1
        B = 1

        N_ee, cos_thet = np.histogram(u["cos_thet"],250, density=True)
        try:
            #sigma = np.sqrt(N_ee), absolute_sigma = True 
            p, cov = curve_fit(ts_channel, cos_thet[1:], N_ee, p0=[A,B],sigma = np.sqrt(N_ee+1), absolute_sigma = True)
            p_uc = uc.correlated_values(p,cov)
            A,B = p_uc

            x0 = costhetamin
            x1 = costhetamax
            al = x1 - x0 + (x1**3 - x0**3)/3
            be = 1/(x1-1) - 1/(x0-1)
            sigma = A*al / (A*al - B*be)*0.50
            
            # Definite Integral of s 
            if sigma.n > 1 or sigma.s >1:
                raise Exception("fit failed")
            pl("%.f ± %.f %% of the electrons are in s channel\n"%((100*sigma).n,(100*sigma).s),0)
            pl("Change from %.f ± %.f to %.f ± %.f electrons. \n"%(N0.n,N0.s,(N0*sigma).n,(N0*sigma).s),0)
            
            N0 *= sigma
        except:
            pl("Removing t-channel failed. ",4)
            pl("Press any KEY to continue...",4)
            x = input()
    return N0

def classify(u, cut_type):

    if cut_type[0:3] == "sup":
        all_ff = []
        for ele in u:
            all_ff  += [[ele["Pcharged"],ele["Ncharged"],ele["Pcharged"],ele["E_ecal"],ele["E_hcal"]]]
        all_ff  = np.array(all_ff)

        if os.path.isfile("./data/classifiers/%s.p"%(cut_type[4:])):
            classifier = pickle.load( open("data/classifiers/%s.p"%(cut_type[4:]),"rb"))
        else:
            raise Exception("Classifier not trained!!!")

        pred = classifier.predict(all_ff) 
    else:
        pred = cut(u,cut_type)
        
    N0 = remove_t(u[pred == 1])

    # We have to include the error of the fitting, thats why this little hack
    return np.array([N0] + [get_N(sum(pred == k)) for k in range(2,5)] )


def get_c_eff(cut_type,check_true = False,remake_choice = False):

    pl("\n Creating new efficiency matrix",1)
    ee = np.load("data/ee.npy")
    mm = np.load("data/mm.npy")
    tt = np.load("data/tt.npy")
    qq = np.load("data/qq.npy")

    ff = [ee,mm,tt,qq]
    #ff[0] = ff[0][(ff[0]["cos_thet"]>costhetamin)\
    #    *(ff[0]["cos_thet"]<costhetamax) * (ff[0]["Pcharged"]!=0)]




    mask = {}

    p_test = 0.25
    pl("\n Choosing %.1f%% of the data for testing..."%(100*p_test),1)

    if os.path.isfile("data/choice_0.npy") and remake_choice==False: 
        for ui in range(4):
            choice = np.load("data/choice/choice_%d.npy"%ui)
            mask_ = np.array([False]*len(u)) 
            mask_[choice]= True
            mask[ui] = mask_ 
    else:
        for ui,u in enumerate(ff):
            choice = np.random.randint(0,len(u),round(p_test*len(u)))
            np.save("data/choice/choice_%d"%ui,choice)
            mask_ = np.array([False]*len(u)) 
            mask_[choice]= True
            mask[ui] = mask_
    # We take only 25 % for testing!
    ff_test = [ff[0][mask[0]],ff[1][mask[1]],ff[2][mask[2]],ff[3][mask[3]]]

    pl("Now coming to the classifying...",2)
    # Removing bulk 
    #for k in range(4):
    #    ff[k] = ff[k][(ff[k]["cos_thru"]>costhrumin)\
    #        *(ff[k]["cos_thru"]<costhrumax)]
    
    # Branching Ratios from Particle Data Booklet 2015

    N_all_mat = un.umatrix(np.zeros([4,4]),np.zeros([4,4]))
    C_eff = un.umatrix(np.zeros([4,4]),np.zeros([4,4]))

    br   = [ 3.363, 3.366, 3.370, 69.91]
    br_s = [ 0.004, 0.007, 0.008, 0.06] 

    br = un.uarray(br,br_s)

    br/=np.sum(br)

    N_monte =  np.array([remove_t(ff_test[0])] + [get_N(len(ff_test[k])) for k in range(1,4)] )
    br_monte = N_monte / np.sum(N_monte)

    ratio = br  / br_monte

    if cut_type[0:3] == "sup":

        targets = []
        all_ff  = []
        for target in range(4):
            for ele in ff[target][~mask[target]]:
                all_ff  += [[ele["Pcharged"],ele["Ncharged"],ele["Pcharged"],ele["E_ecal"],ele["E_hcal"]]]
                targets += [target+1]
        learn_targets = np.array(targets)
        learn_data  = np.array(all_ff)

        if cut_type == "sup_knn":
            classifier = neighbors.KNeighborsClassifier()
        else:
            raise Exception("unknown classifier!")

        pl("Now supervising the learner...",2)
        classifier.fit(learn_data,learn_targets)
        pickle.dump(classifier, open("data/classifiers/%s.p"%(cut_type[4:]),"wb"))

    for u_i,u in enumerate(ff_test):
        pl("\nnow looking for %s, total: %d"%(chars[u_i],len(u)),4)
        N_found = classify(u,cut_type) 
        N_all_mat[u_i,:] = N_found* ratio[u_i]
        C_eff[u_i,:] = N_found / N_monte[u_i]

    C_eff = np.swapaxes(C_eff,0,1)

    Cn = un.nominal_values(C_eff)
    Cs = un.std_devs(C_eff)

    for row in range(4):
        s = ""
        for line in range(4):
            s +="%.3f ± %.3f \t"%(Cn[row,line],Cs[row,line])
        pl(s,3)
    pickle.dump(C_eff, open("data/cuts/%s.p"%cut_type,"wb"))

    if check_true == True:
        N_all = np.array(np.sum(N_all_mat,0)).reshape(4)
        N_all_corrected = np.array(np.dot(C_eff.I,N_all)).reshape(4)
        pl("Now we check whether the efficiency \n matrix does what it is supposed to do. \n Notice how the error gets smaller,\n because we tracked the correlation between \n the efficiency matrix and the numbers.",2)
        for k in range(4):
            pl("Uncorrected vs corrected number of %s"%(chars[k]),6)
            print("%.f ± %.f "%(N_all[k].n,N_all[k].s))
            print("%.f ± %.f "%(N_all_corrected[k].n,N_all_corrected[k].s))

    return C_eff

