# numpy / scipy
import numpy as np
from scipy.optimize import curve_fit
import scipy.constants as co
from scipy.signal import argrelextrema as ext
from scipy.special import jn

#uncertainties
import uncertainties as uc
import uncertainties.unumpy as un
import stat1 as st

# plotting
from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D


import seaborn as sns

import time

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern Roman']
rcParams['text.usetex'] = True
rcParams['figure.autolayout'] = True

fontsize_labels = 12    # size used in latex document
rcParams['axes.labelsize'] = fontsize_labels
rcParams['xtick.labelsize'] = fontsize_labels
rcParams['ytick.labelsize'] = fontsize_labels
plotsize = (6.2, 3.83)  # width corresponds to \textwidth in latex document (ratio = golden ratio ;))
rcParams['figure.figsize'] = plotsize

plt.close("all")
show_fig = True
save_fig = True

fig_dir = "./figures/"
npy_dir = "./data_npy/raman_nath/"
npy_dir2 = "./data_npy/"
plotsize = (6.2, 3.83)  # width corresponds to \textwidth in latex document (ratio = golden ratio ;))

lamb = 632.8e-9 # wavelength of laser


def make_fig(fig, show=True,save=False, name="foo"):
    if show == True:
        fig.show()
    if save == True:
        fig.savefig(fig_dir + name + ".pdf")
        fig.savefig(fig_dir + name + ".png")

def plot_3d():
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    U = np.load(npy_dir + "U.npy")
    
    theta_coeff = np.load(npy_dir2 + "gauge_fit_coeff.npy")
    theta_cov = np.load(npy_dir2 + "gauge_fit_cov.npy")
    #theta_coeff_corr = uc.correlated_values(theta_coeff, theta_cov)
    theta = lambda t: np.polyval([theta_coeff[0],0], t-0.516)
    for j in [0]:
        U_fit = np.load("U_fit%03d.npy"%(j))
        data_fit = np.load("data_Fit%03d.npy"%(j))
        ax.plot(U_fit*0, U_fit[::-1], data_fit)


    for i in np.arange(20,1,-1)[::-1]:
        ch_a = np.load(npy_dir + "phase_%03d_ch_a.npy"%i)/ 0.87821
        t    = np.load(npy_dir + "phase_%03d_t.npy"%i)*10**3
        t = (theta(t))[800:1250]
        ch_a = ch_a[800:1250]
        ax.plot(t,t*0 + U[21-i-1],ch_a, c="%.3f"%(i/40))
        #ax.plot(t,t*0 + U[i-1],ch_a, c="%.3f"%(i/40))

    
    ax.xaxis.set_tick_params(labelsize = 40)
    ax.yaxis.set_tick_params(labelsize = 40)
    ax.zaxis.set_tick_params(labelsize = 40)

    ax.set_ylabel(r"$U$ in V", fontsize = 40)
    ax.set_xlabel(r"$\theta$ in rad",fontsize = 40)

    make_fig(fig,1,0,"3dplot")
    
plot_3d()

def search_maxi(signal,t,neighbours=5, minimum=0.05, n_plateau=20):
    # get local maxima
    maxis = ext(signal, np.greater_equal, order=neighbours)[0] # all maxima with next <order> points greater_equal
    maxis = maxis[signal[maxis] > minimum]   # only those greater <float>
    #maxis = maxis[t[maxis] > t[10]]   # only those maxima greater then 2% of the maximum time
    maxis_j = []
    # reduce numbers of maxima to one per plateau
    i = 0
    while i<len(maxis):
        plateau = np.where((maxis >= maxis[i])*(maxis < maxis[i] + n_plateau))[0]
        maxis_j.append(plateau[int(len(plateau) / 2)])
        i = plateau[-1] + 1
    maxis = maxis[maxis_j]
    return maxis

def func(U, n, alpha, A,c):
    return A*jn(n , alpha * U)**2 + c

def plot_maxi(plot_all=False):
    # without errors on x
    theta_coeff = np.load(npy_dir2 + "gauge_fit_coeff.npy")
    theta_cov = np.load(npy_dir2 + "gauge_fit_cov.npy")
    #theta_coeff_corr = uc.correlated_values(theta_coeff, theta_cov)
    theta = lambda t: np.polyval([theta_coeff[0],0], t-0.516)
    U = np.load(npy_dir + "U.npy")
    i  = 1
    signal = np.load(npy_dir + "phase_%03d_ch_a.npy"%i)
    I0 = max(signal)
    signals = [[],[],[],[]]
    indices = [[],[],[],[]]
    Lambs = []
    indices_Lambs = []
    theta_ar = [] 
    for i in np.arange(20,0,-1):
    #for i in [7]:
        print(I0)
        signal = np.load(npy_dir + "phase_%03d_ch_a.npy"%i) / I0
        t    = np.load(npy_dir + "phase_%03d_t.npy"%i) *10**3
        t = theta(t)

        # get maxima
        maxis = search_maxi(signal,t)
        if i in [20,19,18,17,16]:
            maxis = np.append(maxis,[870])
        if i in [14,13,12,11,10,9,8]:
            maxis = np.append(maxis,[1150])
        if i in [6]:
            maxis = np.append(maxis,[968])
            maxis = maxis[1:-3]
        if i in [7]:
            maxis = maxis[1:-1]
        if i in [5]:
            maxis = np.append(maxis,[1105])
            maxis = maxis[2:-2]
        if i in [4]:
            maxis = maxis[3:-1]
            maxis = np.append(maxis,[1070])
        if i in [3,2]:
            maxis = [1000, 1033, 1070]
        if i in [1]:
            maxis = [1033]
        #np.save("./data_npy/raman_nath/maxi_%03d"%i,maxis)
        maxis = np.load( "./data_npy/raman_nath/maxi_%03d.npy"%i)

        lamb = 632.8e-9
        Lamb = lamb / np.sin(0.001426)
        theta_n = lambda m: np.arcsin(m*lamb/Lamb)

        tt = theta_n(4)
        ii = np.argmin(np.abs(t + tt))

        maxis = np.sort(maxis)
        used = set() 
        # identify zeroth maximum
        max_index =np.argmin(np.abs(t[maxis])) 
        i0 = maxis[max_index]
        used.add(i0)
        signals[0] += [signal[i0]]
        indices[0] += [i-1]
        #print(i)
        # identify first maxima
        dist1_1 = 1 
        dist1_2 = 1
        if len(maxis) > 2:

            i1_1 = maxis[max_index -1]
            i1_2 = maxis[max_index +1]
            signals[1] += [(signal[i1_1]+signal[i1_2])/2]
            indices[1] += [i-1]
            dist1_1 = 0 
            dist1_2 = 0
            t_ = (np.abs(t[i1_1]) + np.abs(t[i1_2]))/2
            t_ = uc.ufloat(t_,0.05*t_)
            Lamb = 632.8e-9/un.sin(t_)*10**6
            Lambs += [Lamb]
            indices_Lambs += [i-1]

            #theta1 = theta_n(1) 
            #cap = list(set(maxis).difference(used))
            #i1_1 = maxis[np.argmin(np.abs(t[cap] - theta1))]
            #i1_2 = maxis[np.argmin(np.abs(t[cap] + theta1))]
            #used.add(i1_1)
            #used.add(i1_2)
            #print("theta1 %.5f"%t[i1_1])
            #print("theta1 %.5f"%t[i1_2])
            #dist1_1 = (t[i1_1] - theta1)
            #dist1_2 = (t[i1_2] + theta1)
            #theta_ar += [(t[i1_1]+t[i1_2])/2]

        dist2_1 = 1 
        dist2_2 = 1 


        if len(maxis) > 4: 

            i2_1 = maxis[max_index -2]
            i2_2 = maxis[max_index +2]
            signals[2] += [(signal[i2_1]+signal[i2_2])/2]
            indices[2] += [i-1]
            dist2_1 = 0 
            dist2_2 = 0 


            # identify second maxima
            #theta2 = np.arcsin(2 * lamb/Lamb)
            #cap = list(set(maxis).difference(used))
            #i2_1 = maxis[np.argmin(np.abs(t[cap] - theta2))]
            #i2_2 = maxis[np.argmin(np.abs(t[cap] + theta2))]
            #used.add(i2_1)
            #used.add(i2_2)

            #dist2_1 = (t[i2_1] - theta2)
            #dist2_2 = (t[i2_2] + theta2)

        dist3_1 = 1 
        dist3_2 = 1 


        if len(maxis) > 6: 
            i3_1 = maxis[max_index -3]
            i3_2 = maxis[max_index +3]
            signals[3] += [(signal[i3_1]+signal[i3_2])/2]
            indices[3] += [i-1]
            dist3_1 = 0 
            dist3_2 = 0 


            # identify third maxima
            #theta3 = np.arcsin(3 * lamb/Lamb)
            #cap = list(set(maxis).difference(used))
            #i3_1 = maxis[np.argmin(np.abs(t[cap] - theta3))]
            #i3_2 = maxis[np.argmin(np.abs(t[cap] + theta3))]
            #used.add(i3_1)
            #used.add(i3_2)

            #dist3_1 = (t[i3_1] - theta3)
            #dist3_2 = (t[i3_2] + theta2)

        if plot_all:
            fig1 = plt.figure()
            ax1 = plt.subplot(111)
            ax1.plot(t, signal, alpha=0.8)

            #[ax1.plot(t[maxi], signal[maxi], 'o', linewidth=1, c="#e74c3c") for k,maxi in enumerate(maxis)]

            # 0 maximum
            ax1.scatter(t[i0],signal[i0],s = 500, marker= "*", label="0th max.", c = '#e74c3c')
            if 1==1:

                # 1 maxima
                if np.abs(dist1_1) < 0.0004:
                    ax1.scatter(t[i1_1],signal[i1_1],s = 100, marker= "s",c='#3498db')
                if np.abs(dist1_2) < 0.0004:
                    ax1.scatter(t[i1_2],signal[i1_2],s = 100, marker= "s",c='#3498db', label="1st max.")

                # 2 maxima
                if np.abs(dist2_1) < 0.0006:
                    ax1.scatter(t[i2_1],signal[i2_1],s = 100, marker= "o",c='#34495e')
                if np.abs(dist2_2) < 0.0006:
                    ax1.scatter(t[i2_2],signal[i2_2],s = 100, marker= "o",c="#34495e",label="2nd max.")
                    
                # 3 maxima
                if np.abs(dist3_1) < 0.0006:
                    ax1.scatter(t[i3_1],signal[i3_1],s = 100, marker= "d",c='#10ce59')
                if np.abs(dist3_2) < 0.0006:
                    ax1.scatter(t[i3_2],signal[i3_2],s = 100, marker= "d",c="#10ce59",label="3rd max.")


            

            #plt.title("index %d"%i)
            #ax1.plot(t, func(t,*p), alpha=0.8)
            ax1.set_xlim(-0.01,+0.01)
            ax1.set_xlabel("$\\theta$ in rad", fontsize = 20)
            ax1.set_ylabel("relative intensity",fontsize = 20)

            ax1.xaxis.set_tick_params(labelsize = 20)
            ax1.yaxis.set_tick_params(labelsize = 20)

            props = dict(boxstyle='round', facecolor='white', alpha=0.5)
            textstr = '$U = %.1f$V'%(U[i-1])
            ax1.text(0.05,0.95, textstr, transform=ax1.transAxes, fontsize=20, va='top', bbox=props)


            plt.legend(fontsize = 14)
            make_fig(fig1,0,0,"raman_%03d"%i)
            plt.close()
    if plot_all:
        for j in range(4):
        
            signals_ = np.array(signals[j][::-1])
            indices_ = np.array(indices[j][::-1])
            sigma = (0.03 + signals_*0) / I0
            p0 = [ 0.5, 1, 0.1]
            func_j= lambda U__,alpha,A,c: func(U__, j, alpha, A, c)
            
            p,cov = curve_fit(func_j, U[indices_], signals_, p0 = p0, sigma = sigma, absolute_sigma = True)
            p[0] = np.abs(p[0])
            p_uc = uc.correlated_values(p, cov)

            f1 = open("coefficients_bessel%01d.tex"%j,"w")
            st.la_coeff2(f1, p,cov, ["\\alpha/ V^{-1}","A","c"])
            f1.close()

            chi2 = np.sum(((func_j(U[indices_], *p) - signals_)/sigma)**2)/(len(indices_)-3)


            U_fit = np.linspace(min(U),max(U),1000)
            data_fit = func_j(U_fit, *p)


            fig = plt.figure()
            ax  = plt.subplot(111)
            plt.errorbar(U[indices_],signals_, yerr = sigma, fmt="x")
            plt.plot(U_fit,data_fit)
            np.save("U_fit%03d"%j,U_fit)
            np.save("data_Fit%03d"%j,data_fit)

            #error_on_fit = un.std_devs(func_0(U_fit, *p_uc))
            data_fit_min = func_j(U_fit, p[0]+p_uc[0].s,p[1]+p_uc[1].s,p[2]+p_uc[2].s)
            data_fit_max = func_j(U_fit, p[0]-p_uc[0].s,p[1]-p_uc[1].s,p[2]-p_uc[2].s)

            plt.fill_between(U_fit, data_fit_min , data_fit_max,facecolor="g", color="g", alpha=0.3 )
            #plt.title("%d"%j)

            props = dict(boxstyle='round', facecolor='white', alpha=0.5)
            textstr = '$A \cdot J_%d^2(\\alpha \cdot U) + c$ \n$A=%.3f\pm%.3f$\n$\\alpha=(%.3f\pm%.3f)1/V$\n$c=%.3f\pm%.3f$\n$\chi^2/n_d = %.3f$'%(j,p[1],p_uc[1].s,p[0],p_uc[0].s,p[2],p_uc[2].s,chi2)
            pos = [(0.1, 0.4),(0.4,0.4),(0.6,0.4),(0.1,0.9)]
            ax.text(pos[j][0],pos[j][1], textstr, transform=ax.transAxes, fontsize=12, va='top', bbox=props)
            plt.xlabel("applied Voltage / V")
            plt.ylabel("relative Intensity of I_%d"%j)
            plt.grid(True)
            make_fig(fig,0,0,"besselfit_%03d"%j)

    fig = plt.figure()
    ax  = plt.subplot(111)
    U_  = U[indices_Lambs]
    SLambs = [k.s for k in Lambs]
    Lambs  = [k.n for k in Lambs]

    plt.errorbar(U_,Lambs, yerr =SLambs, fmt= "x")
    plt.xlabel("Voltage / V")
    plt.ylabel("$\Lambda$ in $\mu$m")
    make_fig(fig,0,1,"soundspeed")
    print(np.mean(Lambs))
    print(np.mean(SLambs))


plot_maxi(True)
