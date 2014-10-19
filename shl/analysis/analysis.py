    import numpy as np
    from scipy.optimize import curve_fit
    import matplotlib.pyplot as plt


    #fredsche routinen
    import stat1 as st


    import seaborn as sns

    from matplotlib import rcParams
    from scipy.interpolate import UnivariateSpline,InterpolatedUnivariateSpline,interp1d
    from smooth import savitzky_golay
    import uncertainties as uc
    import uncertainties.unumpy as un
    from uncertainties import umath


    import pickle


    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Computer Modern Roman']
    rcParams['text.usetex'] = True
    rcParams['figure.autolayout'] = True


    fig_dir = "./figures/"

    def make_fig(fig, show=True,save=False, name="foo"):
        if show == True:
            fig.show()
        if save == True:
            fig.savefig(fig_dir + name + ".pdf")


    # Energy distributions

    def plot_2_1():
        couples = [("a","Pos1","right"),("b","Pos1","left"),("c","Pos2","right")\
                ,("d","Pos2","left"),("e","Pos2","left")]

        for q,pos,pm in couples: 
            data = np.load("./data/measure2_1"+q+".npy")

            # Define some test data which is close to Gaussian

            hist, bin_edges = data, np.arange(0,len(data)+1,1)
            bin_centres = (bin_edges[:-1] + bin_edges[1:])/2

            # Define model function to be used to fit to the data above:
            def gauss(x, *p):
                A, mu, sigma = p
                return A*np.exp(-(x-mu)**2/(2.*sigma**2))

            # p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
            p0 = [1., 0., 1.]

            coeff, var_matrix = curve_fit(gauss, bin_centres, hist, p0=p0)

        # Get the fitted curve
        hist_fit = gauss(bin_centres, *coeff)

        fig = plt.figure(figsize = (12,8))
        ax = fig.add_subplot(111)



        ax.errorbar(bin_centres, hist, label='Test data',yerr = np.sqrt(len(data)))
        #ax.plot(bin_centres, hist_fit, label='Fitted data')

        # Finally, lets get the fitting parameters, i.e. the mean and standard deviation:
        mu  = coeff[1]
        sigma = coeff[2]
        #ax.set_title = ("Measure 2.1%s with    $\mu=%.3f$ $\sigma = %.3f$" %(q,coeff[1],coeff[2]))

        ax.set_xlabel("Channels", fontsize = 40)
        ax.set_ylabel("counts", fontsize = 40)
        ax.xaxis.set_tick_params(labelsize = 35)
        ax.yaxis.set_tick_params(labelsize = 35)

        ax.locator_params(nbins=5)
        
        # place a text box in upper left in axes coords
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        textstr = 'Position: %s\n PM: %s'%(pos,pm)
        if q in "a,c,e": 
            xpos = 0.05 
        else:
            xpos = 0.6 
        ax.text(xpos, 0.95, textstr, transform=ax.transAxes, fontsize=40, va='top', bbox=props)

        ax.grid(True)
        ax.set_xlim(min(bin_centres),max(bin_edges))
        make_fig(fig,0,1,name="plot2_1"+q)
# Delayed coincidences

def plot_4_1():
    data = np.load("./data/measure4_1.npy")[2:]
    c    = np.where(data!=0)
    x = np.arange(0,len(data),1)
    x = x[c]
    x = 1.3149710372035508*x -22.617788714272098
    
    data = data[c]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.grid(True)
    #plt.yscale("log")
    plt.errorbar(x,data, yerr= np.sqrt(data),fmt="x")
    #plt.scatter(x,data)

    errorA = data + np.sqrt(data)
    errorB = data - np.sqrt(data)
    s1 = savitzky_golay(errorA, 61, 3)
    s2 = savitzky_golay(errorB, 61, 3)
    xx = np.linspace(min(x),max(x),1000)
    #plt.plot(xx,s1(xx))
    #plt.fill_between(x, s1, s2, facecolor='grey', alpha=0.3,interpolate=True)
    #T12_lit = 98
    #lamb_lit = -(np.log(2)/T12_lit)
    #plt.plot(x, 70*np.exp(-lamb_lit * (x-162)))
    #plt.plot(x, 70*np.exp(-lamb_lit * (x-152)*6))


    plt.ylim(min(data)*0.8,max(data))
    plt.xlim(min(x)*0.8,max(x))
    plt.xlabel("Channel", fontsize = 14)
    plt.ylabel("counts", fontsize = 14)

    ax.xaxis.set_tick_params(labelsize = 14)
    ax.yaxis.set_tick_params(labelsize = 14)


    make_fig(fig,1,1,name="plot4_1")

def plot_4_1_log():
    data = np.load("./data/measure4_1.npy")[2:]

    # This offset is from background measurement
    offset = 79/251/6080*52658
    data -= offset

    c    = np.where(data>0)
    x = np.arange(0,len(data),1)
    x = x[c]
    data = data[c]

    error = np.sqrt(data) 

    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.yscale("log")
    plt.grid(True)
    #plt.errorbar(x,data, yerr= np.sqrt(data),fmt="x")
    plt.scatter(x,data,c="blue",alpha = 0.5,s=50, marker="o")

    ax.xaxis.set_tick_params(labelsize = 14)
    ax.yaxis.set_tick_params(labelsize = 14)

    ax.add_patch(plt.Rectangle((0,0.1),135,100,alpha = 0.2))

    plt.ylim(min(data)*0.8,max(data)*2)
    plt.xlim(min(x)*0.8,max(x))
    plt.xlabel("Channels", fontsize = 14)
    plt.ylabel("counts", fontsize = 14)
    make_fig(fig,1,1,name="plot4_1_log")

def plot_4_1_reg():
    data = np.load("./data/measure4_1.npy")[2:]

    offset = 79/251/6080*52658
    data -= offset

    c    = np.where(data>1)

    x = np.arange(0,len(data),1)

    x = x[c]
    x = 1.3149710372035508*x -22.617788714272098
    data = data[c]


    error = np.sqrt(data) 
    # only fit for x < 135
    c2 = np.where(x < 155)
    data_log = np.log(data[c2])
    error_log = np.sqrt(data_log)
    p, cov= np.polyfit(x[c2],data_log, 1, full=False, cov=True, w=error_log)



    x_fit = np.linspace(min(x[c2]),max(x[c2]))
    data_fit = np.exp(np.polyval(p, x_fit))
    data_fit_min = np.exp(np.polyval(p - np.sqrt(np.diag(cov)),x_fit))
    data_fit_max = np.exp(np.polyval(p + np.sqrt(np.diag(cov)),x_fit))

    fig = plt.figure()
    plt.yscale("log")
    plt.grid(True)
    #plt.errorbar(x,data, yerr= np.sqrt(data),fmt="x")
    plt.scatter(x,data,c="blue",alpha = 0.9,s=100, marker="x")
    plt.plot(x_fit,data_fit)
    plt.fill_between(x_fit, data_fit_min , data_fit_max,facecolor="r", color="b", alpha=0.3 )

    # place a text box in upper left in axes coords
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    textstr = '$ax + c$ with\n$a=%.2f$\n$b=%.2f$'%(p[0], p[1])
    ax = plt.gca()
    ax.text(0.7, 0.95, textstr, transform=ax.transAxes, fontsize=40, va='top', bbox=props)

    ax.xaxis.set_tick_params(labelsize = 14)
    ax.yaxis.set_tick_params(labelsize = 14)

    ax.add_patch(plt.Rectangle((0,0.1),155,100,alpha = 0.2))

    plt.ylim(min(data)*0.8,max(data)*2)
    plt.xlim(min(x)*0.8,max(x))
    plt.xlabel("time", fontsize = 40)
    plt.ylabel("counts", fontsize = 40)
    make_fig(fig,1,0,name="plot4_1_reg")


def reg2():
    """

    The day we saw the hawk
    On a churchyard tree
    A kite too
    Was in the far sky.

    - Hekigodo 


    """
    data2 = np.load("./data/measure4_1.npy")[2:]

    x2 = np.arange(0,len(data2),1)

    fit = True 
    redistribute = True 

    #x2 = 1.3149710372035508*x2 -22.617788714272098
    c2 = np.where(x2 < 135)

    data = data2[c2] 
    x  = x2[c2]
    print("datapoints:",len(data))

    mass = 79/251/6080*52658
    if redistribute == True:

        # conserving the mass
        total_mass = mass * len(data)
        remaining = (data > 0)

        while True:
            print("new redistributing ...")
            print("total mass:",total_mass)
            # find those which are smaller
            q = (data[remaining] <= mass)
            remaining = ~q
            if len(np.nonzero(q)[0]) == 0:
                data[remaining] -= mass
                break
            print("number of smaller values:",len(np.nonzero(q)[0]),"\n")
            # subtract the mass of this data
            total_mass -= np.sum(data[q])
            mass = total_mass / len(np.nonzero(~remaining)[0])  
            data[q] = 0

        # redistribute total remaining mass to single channels
        print("number of nonzero:",len(np.nonzero(data)[0]))

    c    = np.nonzero(data)  
    data = data[c]
    x = x[c]

    #scaling to time units
    x = 6.3149710372035508*x -22.617788714272098
    c = (x>0)
    x = x[c]
    data = data[c]

    x = x[::-1] - min(x)


    error = np.sqrt(data) 
    # only fit for x < 135
    fig = plt.figure()
    ax  = plt.subplot(111)
    plt.grid(True)

    if fit==True:

        def func(x, *p):
            a,b,c = p
            return a + b * c**x

        # p0 is the initial guess for the fitting coefficients 
        p0 = [1., 1., 1.]

        p, cov = curve_fit(func, x, data, p0=p0, sigma = error)
        p_uc = uc.correlated_values(p, cov)
        c = p_uc[2]

        T12_lit = 98 
        lamb_lit = -(np.log(2)/T12_lit)
        print("lit",lamb_lit)
        

        lamb = umath.log(c)
        print(lamb)
        T12 = -np.log(2) /lamb 
        print("t12=",T12)

        x_fit = np.linspace(min(x),max(x))

        data_fit = func(x_fit,*p) 
        pmin = (p - np.sqrt(np.diag(cov)))
        pmax = (p + np.sqrt(np.diag(cov)))

        data_fit_min = func(x_fit, *pmin)
        data_fit_max = func(x_fit, *pmax)

        plt.plot(x_fit,data_fit)
        plt.plot(x_fit,90*np.exp(x_fit * lamb_lit))
        plt.fill_between(x_fit, data_fit_min , data_fit_max,facecolor="r", color="b", alpha=0.3 )

        # place a text box in upper left in axes coords
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        textstr = '$a + b \cdot c^x$ with\n$a=%.2f$\n$b=%.2f$\n$c=%.2f$'%(p[0], p[1],p[2])
        ax.text(0.6, 0.85, textstr, transform=ax.transAxes, fontsize=18, va='top', bbox=props)

        ax.xaxis.set_tick_params(labelsize = 14)
        ax.yaxis.set_tick_params(labelsize = 14)

        ax.add_patch(plt.Rectangle((0,0.1),155,100,alpha = 0.2))

    plt.errorbar(x,data, yerr=error,fmt="x")
    #plt.scatter(x,data,c="blue",alpha = 0.9,s=100, marker="x")
    plt.ylim(min(data)*0.8,max(data))
    #plt.yscale("log")
    plt.xlim(min(x)*0.8,max(x))
    plt.xlabel("time in $ns$", fontsize = 14)
    plt.ylabel("counts", fontsize = 14)
    make_fig(fig,1,1,name="plot4_1_reg")

# Random coicidences
def plot_5_1():
    data = (np.load("./data/measure5_1.npy")[2:])
    c    = (np.where(data<25)[0])[2:]
    x = (np.arange(0,len(data),1))[c]
    data = data[c]
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.xaxis.set_tick_params(labelsize = 14)
    ax.yaxis.set_tick_params(labelsize = 14)


    ax.grid(True)
    #plt.errorbar(x[c],data[c], yerr= np.sqrt(data[c]))
    plt.scatter(x,data,c="blue",alpha = 0.3,s=100, marker="o")

    plt.ylim(min(data)*0.8,max(data)*2)
    plt.xlim(min(x)*0.8,max(x)*1.03)
    plt.xlabel("Channels", fontsize = 14)
    plt.ylabel("counts", fontsize = 14)

    make_fig(fig,1,0,name="plot5_1")
    plt.show()

def plot_5_1_hist():
    data = (np.load("./data/measure5_1.npy")[2:])
    #Filtering the strange event which causes some confusion
    c    = (np.where(data<=25)[0])[2:]
    x = (np.arange(0,len(data),1))[c]
    data = data[c]
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    
    ax1.xaxis.set_tick_params(labelsize = 14)
    ax1.yaxis.set_tick_params(labelsize = 14)
    ax1.grid(True)
    ax1.scatter(x,data,c="blue",alpha = 0.3,s=100, marker="o")

    ax1.set_ylim(min(data)*0.8,max(data)*1.2)
    ax1.set_xlim(min(x)*0.8,max(x)*1.03)
    ax1.set_xlabel("Channels", fontsize = 14)
    ax1.set_ylabel("counts", fontsize = 14)

    ax1.yticks = [0,1,2,3]
    ax1.grid(True)
    ax1.locator_params(nbins=4)

    ax2.xaxis.set_tick_params(labelsize = 14)
    ax2.yaxis.set_tick_params(labelsize = 14)



    #plt.errorbar(x[c],data[c], yerr= np.sqrt(data[c]))
    ax2.hist(data, 4, orientation="horizontal", align='mid', rwidth=20, log=True)

    ax2.set_xlabel("occurence", fontsize = 14)

    make_fig(fig,1,1,name="plot5_1_hist")
    plt.show()
# Time calibration

def calib_TAC_MCA():
    delay   = np.array([0,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,190.5])
    channel = np.array([20,24,29,34,41,47,53,59,66,72,78,84,90,96,102,109,115,121,127,133,139,145,151,157,162])
    np.save("data/delay",delay)
    np.save("data/chennel",channel)
    x_error = delay * 0 + 1
    y_error = channel * 0 + 1

    fig = plt.figure()
    ax = fig.add_subplot(111)


    p, cov= np.polyfit(channel,delay, 1, full=False, cov=True, w=y_error)

    #f1 = open("coefficients.tex","a")
    #st.la_coeff(f1, p,cov, ["p_1","p_2"])


    x_fit = np.linspace(20,162,1000)

    data_fit     = (np.polyval(p, x_fit))
    data_fit_min = (np.polyval(p - np.sqrt(np.diag(cov)),x_fit))
    data_fit_max = (np.polyval(p + np.sqrt(np.diag(cov)),x_fit))

    plt.plot(x_fit,data_fit)

    print(st.chi2_min(channel,delay,p,y_error)/(len(channel)-1))

    
    
    plt.xlim(20,162)
    plt.ylim(0,191)

    plt.ylabel("Delay $\Delta t$ in ns",fontsize = 14)
    plt.xlabel("Channel", fontsize = 14)

    ax.xaxis.set_tick_params(labelsize = 14)
    ax.yaxis.set_tick_params(labelsize = 14)


    plt.fill_between(x_fit, data_fit_min , data_fit_max,facecolor="r", color="b", alpha=0.3 )

    plt.errorbar(channel,delay, yerr= y_error,xerr= x_error, fmt= "x",c="0.0")
    make_fig(fig,1,1,name = "plot7")
calib_TAC_MCA()

def calib_TAC_MCA2():
    delay   = np.array([25,108.5,162.,0.5,65.5,113.5,75.5,49,79,135.5,23.5,98.5,135.5,110.5,121,127,45,51.5,184,80,185,88.5,52,167.5,44.5])
    channel = np.array([35,100,141,22,66,104,74,54,77,121,34,92,119,102,110,114,51,56,157,78,158,84,56,145,50])
    x_error = delay * 0 + 1
    y_error = channel * 0 + 1

    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.errorbar(channel,delay, yerr= y_error,xerr= x_error, fmt= "x")

    p, cov= np.polyfit(channel,delay, 1, full=False, cov=True, w=y_error)

    
    x_fit = np.linspace(20,162,1000)

    data_fit     = (np.polyval(p, x_fit))
    data_fit_min = (np.polyval(p - np.sqrt(np.diag(cov)),x_fit))
    data_fit_max = (np.polyval(p + np.sqrt(np.diag(cov)),x_fit))

    plt.plot(x_fit,data_fit)
    
    plt.xlim(20,162)
    plt.ylim(0,191)

    plt.ylabel("Delay",fontsize = 14)
    plt.xlabel("Channel", fontsize = 14)

    ax.xaxis.set_tick_params(labelsize = 14)
    ax.yaxis.set_tick_params(labelsize = 14)



    plt.fill_between(x_fit, data_fit_min , data_fit_max,facecolor="r", color="b", alpha=0.3 )
    make_fig(fig,1,1,name = "plot71")

def calib_TAC_MCA_rescaled():
    delay   = np.array([0,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,190.5])
    channel = np.array([20,24,29,34,41,47,53,59,66,72,78,84,90,96,102,109,115,121,127,133,139,145,151,157,162])
    x_error = delay * 0 + 1
    y_error = channel * 0 + 1.5

    fig = plt.figure()
    ax = fig.add_subplot(111)


    p, cov= np.polyfit(channel,delay, 1, full=False, cov=True, w=y_error)
    x_fit = np.linspace(20,162,1000)

    data_fit     = (np.polyval(p, x_fit))
    data_fit_min = (np.polyval(p - np.sqrt(np.diag(cov)),x_fit))- (p[0]*x_fit + p[1])
    data_fit_max = (np.polyval(p + np.sqrt(np.diag(cov)),x_fit))- (p[0]*x_fit + p[1])
    
    delay   = delay   - (p[0]*channel + p[1])
    data_fit = data_fit - (p[0]*x_fit   + p[1])

    plt.plot(x_fit,data_fit)
    
    plt.fill_between(x_fit, data_fit_min , data_fit_max,facecolor="r", color="b", alpha=0.3 )
    plt.errorbar(channel,delay, yerr= y_error,xerr= x_error, fmt= "x")

    plt.xlim(20,162)
    plt.ylim(-3,3)

    plt.ylabel("delay",fontsize = 14)
    plt.xlabel("channel", fontsize = 14)

    ax.xaxis.set_tick_params(labelsize = 14)
    ax.yaxis.set_tick_params(labelsize = 14)



    make_fig(fig,1,1,name = "plot7b")

def plot_6():
    """

        Rainer Marie Rilke: Magie

    Aus unbeschreiblicher Verwandlung stammen
    solche Gebilde-: Fühl! und glaub!
    Wir leidens oft: zu Asche werden Flammen;
    doch: in der Kunst: zur Flamme wird der Staub.

    Hier ist Magie. In das Bereich des Zaubers
    scheint das gemeine Wort hinaufgestuft...
    und ist doch wirklich wie der Ruf des Taubers,
    der nach der unsichtbaren Taube ruft. 

    """
    f, (ax1, ax2) = plt.subplots(1, 2, figsize = (16,8))

    data = np.load("data/measure6_1.npy")
    channel = np.arange(0,len(data),1)
    ax1.errorbar(channel,data,yerr = np.sqrt(data))

    ax1.set_xlabel("Channels", fontsize = 25)
    ax1.set_ylabel("counts", fontsize = 25)
    ax1.xaxis.set_tick_params(labelsize = 25)
    ax1.yaxis.set_tick_params(labelsize = 25)

    ax1.set_xlim(0,max(channel))

    # place a text box in upper left in axes coords
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    textstr = 'Left detector\n $t = 2682$s\n Ma Coarse gain: 500'
    ax1.text(0.1, 0.95, textstr, transform=ax1.transAxes, fontsize=25, va='top', bbox=props)
    

    data = np.load("data/measure6_2.npy")
    channel = np.arange(0,len(data),1)
    ax2.errorbar(channel,data,yerr = np.sqrt(data))

    ax2.set_xlabel("Channels", fontsize = 25)
    ax2.set_ylabel("counts", fontsize = 25)
    ax2.xaxis.set_tick_params(labelsize = 25)
    ax2.yaxis.set_tick_params(labelsize = 25)

    ax2.set_xlim(0,140)
    # place a text box in upper left in axes coords
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    textstr = 'Left detector\n $t = 2843$s\n Ma Coarse gain: 200'
    ax2.text(0.1, 0.95, textstr, transform=ax2.transAxes, fontsize=25, va='top', bbox=props)
    plt.savefig("figures/plot6_12.pdf")
    plt.show()
    

    fig = plt.figure()
    ax  = plt.subplot(111)

    data = np.load("data/measure6_3.npy")
    channel = np.arange(0,len(data),1)
    ax.errorbar(channel,data,yerr = np.sqrt(data))

    ax.set_xlabel("Channels", fontsize = 14)
    ax.set_ylabel("counts", fontsize = 14)
    ax.xaxis.set_tick_params(labelsize = 14)
    ax.yaxis.set_tick_params(labelsize = 14)

    ax.set_xlim(0,150)

    # place a text box in upper left in axes coords
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    textstr = 'Right detector\n $t = 2883$s\n Ma Coarse gain: 200'
    ax.text(0.7, 0.95, textstr, transform=ax.transAxes, fontsize=14, va='top', bbox=props)
    plt.savefig("figures/plot6_3.pdf")
    plt.show()

def reg_6_3(plot = False):

    data = np.load("data/measure6_3.npy")
    error = np.sqrt(data)
    channel = np.arange(0,len(data),1)
    channel_fit = np.linspace(0,len(data)+1,1000)

    def func(x, *p):
        a1,a2,a3,mu1,mu2,mu3,sigma1,sigma2,sigma3,c = p
        return a1*np.exp(- (x-mu1)**2 / (2*sigma1)**2 ) + \
               a2*np.exp(- (x-mu2)**2 / (2*sigma2)**2 ) + \
               a3*np.exp(- (x-mu3)**2 / (2*sigma3)**2 ) + c

    # p0 is the initial guess for the fitting coefficients 
    p0 = [1000, 8000, 22000, 22, 45, 96, 1,2,3, 500]

    p, cov = curve_fit(func, channel, data, p0=p0, sigma = error)

    f1 = open("coefficients_6_3.tex","a")
    st.la_coeff(f1, p,cov, ["A_1","A_2","A_3","\mu_1","\mu_2","\mu_3","\sigma_1","\sigma_2","\sigma_3","c"])
    f1.close()
def reg_6_3(plot = False):

    data = np.load("data/measure6_3.npy")
    error = np.sqrt(data)
    channel = np.arange(0,len(data),1)
    channel_fit = np.linspace(0,len(data)+1,1000)

    def func(x, *p):
        a1,a2,a3,mu1,mu2,mu3,sigma1,sigma2,sigma3,c = p
        return a1*np.exp(- (x-mu1)**2 / (2*sigma1)**2 ) + \
               a2*np.exp(- (x-mu2)**2 / (2*sigma2)**2 ) + \
               a3*np.exp(- (x-mu3)**2 / (2*sigma3)**2 ) + c

    # p0 is the initial guess for the fitting coefficients 
    p0 = [1000, 8000, 22000, 22, 45, 96, 1,2,3, 500]

    p, cov = curve_fit(func, channel, data, p0=p0, sigma = error)

    f1 = open("coefficients_6_3.tex","a")
    st.la_coeff(f1, p,cov, ["A_1","A_2","A_3","\mu_1","\mu_2","\mu_3","\sigma_1","\sigma_2","\sigma_3","c"])
    f1.close()



def reg_2_1a(plot = False):

    data = np.load("data/measure2_1a.npy")
    error = np.sqrt(data)
    channel = np.arange(0,len(data),1)
    channel_fit = np.linspace(0,len(data)+1,1000)

    def func(x, a1,a2,a3,a4,a5,mu1,mu2,mu3,mu4,mu5,sigma1,sigma2,sigma3,sigma4,sigma5,c):
        return a1*np.exp(1)**(- (x-mu1)**2 / (2*sigma1)**2 ) + \
               a2*np.exp(1)**(- (x-mu2)**2 / (2*sigma2)**2 ) + \
               a3*np.exp(1)**(- (x-mu3)**2 / (2*sigma3)**2 ) + \
               a4*np.exp(1)**(- (x-mu4)**2 / (2*sigma4)**2 ) + \
               a5*np.exp(1)**(- (x-mu5)**2 / (2*sigma5)**2 ) + c

    # p0 is the initial guess for the fitting coefficients 
    p0 = [1000,1000, 4000,5000, 28000,18,42,119,140,190,1, 1,1,1,1, 500]

    p, cov = curve_fit(func, channel, data, p0=p0, sigma = error)
    
    f1 = open("coefficients_5_1.tex","a")
    st.la_coeff(f1, p,cov, ["A_1","A_2","A_3","A_4","A_5","\mu_1","\mu_2","\mu_3","\mu_4","\mu_5","\sigma_1","\sigma_2","\sigma_3","\sigma_4","\sigma_5","c"])
    f1.close()


    p_uc = uc.correlated_values(p, cov)
    

    if plot:
        fig = plt.figure()
        ax  = plt.subplot(111)

        ax.errorbar(channel,data,yerr = np.sqrt(data))
        
        ax.plot(channel_fit, func(channel_fit,*p))
        ax.plot(channel_fit,channel_fit*0 + p[-1],"--")
        text =True 
        if text:
            pos = [(0.02,0.35),( 0.55,0.8),( 0.7,0.95)]
            
            props = dict(boxstyle='round', facecolor='white', alpha=0.5)

            for i,q in enumerate([0,4]):

                a,Sa = p_uc[q].n, p_uc[q].s
                mu,Smu = p_uc[q+5].n, p_uc[q+5].s
                sigma,Ssigma = p_uc[q+10].n , p_uc[q+10].s
                
                #textstr = '$A=%.3f \pm %.3f$\n$\mu = %.3f \pm %.3f$\n$\sigma = %.3f \pm %.3f$'%(a,Sa,mu,Smu,sigma,Ssigma)
                ii = i+1
                textstr = '$A_%d=%.3f$\n$\mu_%d = %.3f$\n$\sigma_%d = %.3f$'%(ii,a,ii,mu,ii,sigma)
                ax.text(pos[i][0], pos[i][1], textstr, transform=ax.transAxes, fontsize=14, va='top', bbox=props)

                ax.set_xlabel("channels", fontsize = 14)
                
        ax.add_patch(plt.Rectangle((10,0.1),18,3000,alpha = 0.2))
        ax.add_patch(plt.Rectangle((165,0.1),50,30000,alpha = 0.2))
        ax.set_ylabel("counts", fontsize = 14)
        ax.xaxis.set_tick_params(labelsize = 14)
        ax.yaxis.set_tick_params(labelsize = 14)

        data_fit = func(channel_fit, *p) 
        error_on_fit = un.std_devs(func(channel_fit, *p_uc))
        data_fit_min = data_fit - error_on_fit
        data_fit_max = data_fit + error_on_fit

        plt.fill_between(channel_fit, data_fit_min , data_fit_max,facecolor="r", color="b", alpha=0.3 )

        plt.grid(True)
        plt.xlim(0,250)


        make_fig(fig,0,0,name = "plot2_1a_reg")

    mu = [p_uc[6].n,p_uc[9].n]
    Smu =[p_uc[6].s,p_uc[9].s]
    mu2 = [p_uc[5+q] for q in range(5)]
    energies = [14.4,122.1]

    return energies,mu,Smu, mu2
def energy_scale():

    E_am, mu_am, Smu_am = reg_6_3(False)
    E_co, mu_co, Smu_co, mu2 = reg_2_1a(False)
    plot =False 

    #plt.errorbar(E_am,mu_am,yerr= Smu_am, fmt="x") 
    #plt.errorbar(E_co,mu_co,yerr= Smu_co, fmt="x") 
    E = [E_co[1]] + E_am[1:3]
    mu = [mu_co[1] ]+ mu_am[1:3]
    error = [Smu_co[1]] + Smu_am[1:3]

    def func(x, a,b):
        return a*x + b

    # p0 is the initial guess for the fitting coefficients 
    p0 = [1,1]

    p, cov = curve_fit(func, mu, E, p0=p0, sigma = error)
    p_uc = uc.correlated_values(p, cov)

    
    f1 = open("coefficients_energy.tex","a")
    st.la_coeff(f1, p,cov, ["a","b"])
    f1.close()

    for mu_ in mu2: 
        print(mu_,"Channel => ",func(mu_,*p_uc),"keV")

    if plot:

        fig = plt.figure()
        ax  = plt.subplot(111)

        plt.scatter(mu_co[1],E_co[1],c="r", label = "Cobalt source: 122.1 keV Peak") 
        plt.scatter(mu_am[1:3],E_am[1:3],c="b", label = "Americium source: 33.2 keV and 59.5 keV") 
        plt.legend(fontsize = 15)


        channel_fit = np.linspace(0,300,1000)
        plt.xlim(0,300)
        plt.ylim(0,250)

        data_fit = func(channel_fit, *p) 
        

        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        textstr = 'E = $a\cdot \mathrm{channel} + b$\n$a=%.3f$ keV/channel\n$b=%.3f$ keV'%(p[0],p[1])
        ax.text(0.6,0.3, textstr, transform=ax.transAxes, fontsize=14, va='top', bbox=props)


        plt.plot(channel_fit,data_fit)
        error_on_fit = un.std_devs(func(channel_fit, *p_uc))
        data_fit_min = data_fit - error_on_fit
        data_fit_max = data_fit + error_on_fit

        plt.fill_between(channel_fit, data_fit_min , data_fit_max,facecolor="r", color="b", alpha=0.3 )

        plt.grid(True)

        plt.ylabel("Energy $E$ / keV", fontsize = 14)
        plt.xlabel("mean $\mu$ /  channel", fontsize = 14)
        make_fig(fig,1,1,name = "plot_E")

