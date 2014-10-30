import numpy as np
import pylab as plt
from matplotlib import rcParams
from scipy.optimize import curve_fit
import scipy.constants as co
import uncertainties as uc
import uncertainties.unumpy as un
from scipy.signal import argrelextrema as ext
import seaborn as sns

import stat1 as st
fontsize_labels = 12    # size used in latex document
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern Roman']
rcParams['text.usetex'] = True
rcParams['figure.autolayout'] = True
rcParams['axes.labelsize'] = fontsize_labels
rcParams['xtick.labelsize'] = fontsize_labels
rcParams['ytick.labelsize'] = fontsize_labels

plt.close("all")
show_fig = True
save_fig = False
fig_dir = "./figures/"
npy_dir = "./data_npy/"
plotsize = (6.2, 3.83)  # width corresponds to \textwidth in latex document (ratio = golden ratio ;))

def make_fig(fig, show=True,save=False, name="foo"):
        if show == True:
            fig.show()
        if save == True:
            fig.savefig(fig_dir + name + ".pdf")
            fig.savefig(fig_dir + name + ".png")

def find_no(X):
    x0 = X[0]
    q  = 0
    while X[q] == x0:
        q += 1
    return q

def average(X,no):
    assert (len(X)%no == 0)
    X_mean = np.zeros(len(X)/no)
    for i in range(int(len(X)/no)):
        X_mean[i] = np.mean(X[i*no:(i+1)*no])
    return X_mean

def uranium_2_1(subtract = True):
    # load uranium
    title = "uranium"
    npy_files = npy_dir + title + "_"
    U = np.load(npy_files + "U" + ".npy")
    n = np.load(npy_files + "n" + ".npy")
    # Averaging 
    no = find_no(U)
    U_mean = average(U,no) 
    n_mean = average(n,no) 
    print(len(U_mean))
    # load background 
    title = "background2"
    npy_files = npy_dir + title + "_"
    U = np.load(npy_files + "U" + ".npy")
    n = np.load(npy_files + "n" + ".npy")
    # Averaging 
    no = find_no(U)
    U2_mean = average(U,no) 
    n2_mean = average(n,no) 
    print(len(U2_mean))
        

    dt = 100

    fig1, ax1 = plt.subplots(1, 1, figsize=plotsize)

    if subtract == True:         
        n_mean -= n2_mean
    ax1.errorbar(U_mean, n_mean, yerr = np.sqrt(n_mean/dt), fmt=".")
    ax1.set_yscale("log")

    ax1.set_xlim(U[0]*0.9, U[-1])
    ax1.set_ylabel("$n$ / counts\cdot s^{-1} ")
    ax1.set_ylim(1,1000)
    ax1.set_xlabel("$U$ / V")
    make_fig(fig1,1,1,"2_1_uranium")

def background2_2():
    title = "background2"
    npy_files = npy_dir + title + "_"
    t = np.load(npy_files + "t" + ".npy")
    U = np.load(npy_files + "U" + ".npy")
    n = np.load(npy_files + "n" + ".npy")
    dt = 100
    no = find_no(U)

    fig1, ax1 = plt.subplots(1, 1, figsize=plotsize)
    ax1.errorbar(average(U,no), average(n,no), yerr = np.sqrt(average(n,no)/dt), fmt=".")
    ax1.set_yscale("log")

    #ax1.set_xlim(U[0]*0.9, U[-1])
    ax1.set_ylabel("$n$ / counts\cdot s^{-1} ")
    #ax1.set_ylim(1,1000)
    ax1.set_xlabel("$U$ / V")
    make_fig(fig1,0,1,"2_2_background2")


def samarium_2_2_1():
    title = "measurement_2_2_1"
    npy_files = npy_dir + title + "_"
    # load samarium
    U = np.load(npy_files + "U" + ".npy")
    n = np.load(npy_files + "n" + ".npy")
    no = find_no(U)
    U_sam = average(U, no)
    n_sam = average(n, no)

    # load background 
    title = "background2"
    npy_files = npy_dir + title + "_"
    U = np.load(npy_files + "U" + ".npy")
    n = np.load(npy_files + "n" + ".npy")
    # Averaging 
    no = find_no(U)
    U_back= average(U,no) 
    n_back= average(n,no) 
    c = np.where(U_back <= max(U_sam))
    U_back = U_back[c]
    n_back = n_back[c]
    # with a little bit of luck this will be the same range
    print("U_back",U_back)
    print("U",U_sam)
    assert len(U_back) == len(U_sam) 
 
    dt = 5*40

    fig1, ax1 = plt.subplots(1, 1, figsize=plotsize)
    ax1.errorbar(U_sam,n_sam, yerr = np.sqrt((n_sam+n_back)/dt), fmt=".",label="unmodified")
    ax1.errorbar(U_sam,n_sam-n_back, yerr = np.sqrt((n_sam+n_back)/dt), fmt=".", label="background subtracted")
    ax1.set_yscale("log")
    plt.legend(loc=8)

    ax1.set_xlim(1350, U_sam[-1]*1.1)
    ax1.set_ylabel("$n$ / counts\cdot s^{-1} ")
    #ax1.set_ylim(1,1000)
    ax1.set_xlabel("$U$ / V")
    make_fig(fig1,1,1,"measurement_2_2_1")

#background2_2()
#samarium_2_2_1()

def background_long():
    title = "measurement_2_3"
    npy_files = npy_dir + title + "_"
    # load background
    t = np.load(npy_files + "t" + ".npy")
    U = np.load(npy_files + "U" + ".npy")
    n = np.load(npy_files + "n" + ".npy")

    dt = np.sum(t)/2

    no = find_no(U)
    U_mean = average(U, no)
    n_mean = average(n, no)
    #print(U_sam)
    return n_mean,dt

def samarium_long(subtract=True):
    title = "measurement_2_2_2"
    npy_files = npy_dir + title + "_"
    # load samarium
    t = np.load(npy_files + "t" + ".npy")
    U = np.load(npy_files + "U" + ".npy")
    n = np.load(npy_files + "n" + ".npy")
    no = len(U)
    dt = np.sum(t)

    U_sam = average(U, no)
    n_sam = average(n, no)
    n_back, dt_back = background_long()
    print(n_sam - n_back[0], np.sqrt(n_sam/dt + n_back[0]/dt_back))

def samarium_long2(subtract=True):
    title = "measurement_2_2_3"
    npy_files = npy_dir + title + "_"
    # load samarium
    t = np.load(npy_files + "t" + ".npy")
    U = np.load(npy_files + "U" + ".npy")
    n = np.load(npy_files + "n" + ".npy")
    no = len(U)
    dt = np.sum(t)

    U_sam = average(U, no)
    n_sam = average(n, no)
    n_back, dt_back = background_long()
    print(n_sam - n_back[0], np.sqrt(n_sam/dt + n_back[0]/dt_back))


def potassium(subtract=True, fit = True):
    fig = plt.figure()
    ax  = plt.subplot(111)
    weight = [0.7754,0.5179,0.1721,0.1173,1.159,1.9966,1.6133,1.0595]
    n_back, dt_back = background_long()
    n_all = []
    U_all = []
    m_all = []
    T_all = []
    cp = sns.color_palette()
    tt = 0
    for i in range(2,9+1):
        title =  "measurement_2_4_%d"%i
        npy_files = npy_dir + title + "_"
        # load potassium 
        t = np.load(npy_files + "t" + ".npy")
        U = np.load(npy_files + "U" + ".npy")
        n = np.load(npy_files + "n" + ".npy")
        T_all += [t]
        n_all += [n]
        U_all += [U]
        m_all += [weight[i-2]+n*0]
        n_mean = np.mean(n - n_back[1])
        
        plt.errorbar(n*0+weight[i-2],n - n_back[1], alpha = 0.9, yerr =np.sqrt(n/t + n_back[1]/dt_back), c = cp[0], xerr =0.0001 ,fmt=".")
        #plt.errorbar(n*0+weight[i-2],n , alpha = 0.9, yerr =np.sqrt(n/np.sum(t)), c = cp[1])
    if fit == True:
        n_all = np.array(n_all).flatten()
        U_all = np.array(U_all).flatten()
        T_all = np.array(T_all).flatten()
        m_all = np.array(m_all).flatten()

        def func(x, a,b):
            return a*(1 - np.exp(1)**(-b*x))
        # p0 is the initial guess for the fitting coefficients 
        p0 = [1,1]

        y = n_all - n_back[1]
        sigma = np.sqrt(n_all/T_all + n_back[1]/dt_back)
        p, cov = curve_fit(func, m_all,y, p0=p0, sigma = sigma , absolute_sigma = True)
        p_uc = uc.correlated_values(p, cov)


        chi2 = np.sum(((func(m_all,*p) - y) / sigma) ** 2 )/(len(m_all)-2)


        a = p_uc[0]
        b = p_uc[1] 
        hrel  = uc.ufloat(1.18,0.01)
        fB    = uc.ufloat(1.29,0.01)
        EC    = uc.ufloat(1.120,0.01)
        m_kcl = uc.ufloat(74.55,0.01)

        eta = co.N_A * hrel*10**(-4) * fB * np.log(2) / (EC * 2 * m_kcl)
        print(eta)
        T12 =  eta / (a * b) 
        #print("a = %.3f +- %.3f  / b = %.3f += %.3f and T12 = %.3e +- %.3e"%(a.n,a.s,b.n,b.s,T12.n,T12.s))
        print(T12)
        print(T12/ 3600 / 365.25 /24)
        

        #f1 = open("coefficients_2_4.tex","w")
        #st.la_coeff(f1, p,cov, ["a","b"])
        #f1.close()

        m_fit = np.linspace(0.0,2.1,1000)
        data_fit = func(m_fit, *p) 

        

        props = dict(boxstyle='round', facecolor='white', alpha=0.5,edgecolor = 'white')
        textstr = '\\begin{eqnarray*} n(m) &=& a(1 - e^{-bm})\\\\  a&=&\left [ %.3f \pm %.3f \\right ] \mathrm{counts}\cdot s^{-1}\\\\   b&=& \left [ %.3f \pm %.3f\\right ]   g^{-1} \\\\ \chi^2/\mathrm{dof}&=&%.3f \\end{eqnarray*}'%(p[0],p_uc[0].s,p[1],p_uc[1].s, chi2)
        #textstr = '\\begin{eqnarray*} %f  %f \\end{eqnarray*}'%(p[0],p[1])
        ax.text(0.4,0.3, textstr, transform=ax.transAxes, fontsize=14, va='top', bbox=props)
        #plt.yscale("log")
        plt.xlabel("mass $m$ / g") 
        plt.ylabel("$n(m)$ / counts$\cdot s^{-1}$")


        plt.plot(m_fit,data_fit, c = cp[1])
        plt.ylim(00,6)
        plt.xlim(00,2.1)
        error_on_fit = un.std_devs(func(m_fit, *p_uc))
        data_fit_min = data_fit - error_on_fit
        data_fit_max = data_fit + error_on_fit

        plt.fill_between(m_fit, data_fit_min , data_fit_max,facecolor="r", color="b", alpha=0.3 )

    make_fig(fig,1,1,"measurement_2_4")

#potassium() 

from uncertainties.unumpy import sqrt as sq
def masses():
    p_nit = uc.ufloat( 0.75518 , 0.00001)
    p_oxy = uc.ufloat( 0.23135 , 0.00001)
    p_arg = uc.ufloat( 0.01288 , 0.00001)
    m_nit = uc.ufloat(14.0067  , 0.0002)
    m_oxy = uc.ufloat(15.9994  , 0.0003)
    m_arg = uc.ufloat(39.948   , 0.001 )
    m_sm2 = uc.ufloat(150.36, 0.01)
    m_O3 =   uc.ufloat(16.000, 0.001)
    p_sm2 = 2*m_sm2 / (2*m_sm2 + 3* m_O3)
    p_O3 = 3*m_O3 / (2*m_sm2 + 3*m_O3)
    m_air = (p_nit*sq(m_nit)+p_oxy*sq(m_oxy)+p_arg*sq(m_arg))

    m_sm2o3 = (p_sm2*sq(m_sm2)+p_O3*sq(m_O3))

    d_air = uc.ufloat(0.001184,0.0001)
    d_sm2o3  = uc.ufloat(7.6,0.1)
    #d_sm2o3  = uc.ufloat(8.35,0.1)

    Ea = uc.ufloat(2.233, 0.001)
    f  = uc.ufloat(0.56,0.01)
    R_air  = Ea * f
    #R_air = uc.ufloat(1.13,0.01)

    R_sm2o3 = R_air * (d_air * m_sm2o3)/(d_sm2o3 * m_air) 

    m_sm2o3_mol = uc.ufloat(348.72,0.01)

    h_rel = uc.ufloat(0.1499, 0.0033)

    zeta = np.log(2) * h_rel *co.N_A * d_sm2o3 * R_sm2o3 /(2*m_sm2o3_mol)
    #zeta = uc.ufloat(3.62,0.1)*10**17

    print(zeta)

    F1 = uc.ufloat(2.58 , 0.05)
    n1 = uc.ufloat(0.54  , 0.01) 

    F2 = uc.ufloat(1.56 , 0.05)
    n2 = uc.ufloat(0.18  , 0.007) 

    print(np.pi * (F2/2)**2 /n2)
    T12 = zeta * np.pi * (F2/2)**2 /n2 
    print((T12/ 3600 / 24 / 365))



#samarium_long2()
masses()
