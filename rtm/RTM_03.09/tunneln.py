import numpy as np
import matplotlib.pyplot as plt


def pplot():
    a = 10
    E = 0.9
    V_0 = 1
    h = 1
    m = 1

    k_0 = np.sqrt(2*m*E/h**2+0J)
    k_1 = np.sqrt(-2*m*(E-V_0)/h**2+0J)

    print("k_0 =",k_0)
    print("k_1 =",k_1)


    t = (4*k_0*k_1*np.exp(-1J*a*(k_0-k_1)))/((k_0+k_1)**2 - np.exp(2*1J*a*k_1)*(k_0-k_1)**2)
    r = ((k_0**2 - k_1**2)*np.sin(a*k_1))/(2*1J*k_0*k_1*np.cos(a*k_1)+(k_0**2+k_1**2)*np.sin(a*k_1))


    print("t=",t)
    print("r=",r)

    A_r = 1
    A_l = r
    C_l = 0
    C_r = t
    B_l  = (1+r)*(k_1-k_0)/(2*k_1)
    B_r = 1+r - B_l


    psi_I =  lambda x: A_r *np.exp(1J*k_0*x) + A_l  *np.exp(-1J*k_0*x)

    psi_II = lambda x: B_r *np.exp(1J*k_1*x) + B_l  *np.exp(-1J*k_1*x) 

    psi_III= lambda x: C_r *np.exp(1J*k_0*x) + C_l  *np.exp(-1J*k_0*x) 

    x1 = np.linspace(-3,0,100)
    x2 = np.linspace(0,a,100)
    x3 = np.linspace(a,a+5,100)

    plt.figure()
    plt.plot(x1,np.absolute(psi_I(x1)))
    plt.plot(x2,np.absolute(psi_II(x2)))
    plt.plot(x3,np.absolute(psi_III(x3)))
    plt.show()
def plot_T(a,c):
    m= 1
    h = 1
    V_0 = 1
    E   = np.arange(0,1,0.001)
    E2   = np.arange(1,2,0.001)

    k = np.sqrt(2*m*E/h**2)
    kappa =   np.sqrt(2*m*(V_0-E)/h**2)
    T = 1/ (1 + ((k**2+kappa**2)/(2*k*kappa))**2 * np.sinh(kappa*a)**2)

    k2 = np.sqrt(2*m*E2/h**2)
    kappa2 =   np.sqrt(2*m*(E2-V_0)/h**2)
    T2 = 1/ (1 + ((k2**2-kappa2**2)/(2*k2*kappa2))**2 * np.sin(kappa2*a)**2)

    #T2 = (1 - E2/V_0)/((1-E2/V_0) - (V_0 / (4*E2)*np.sinh(k_1b*a)**2))
    plt.plot(E,T, c=c,label="a="+str(a))
    plt.plot(E2,T2,c=c)
plt.figure()
plot_T(3,"r")
plot_T(4,"b")
plot_T(5,"g")
plt.ylabel("Durchgangswahrscheinlichkeit $T$")
plt.xlabel("$E/V_0$")
plt.legend(loc=7)
plt.grid(True)
plt.savefig("tunnel1.pdf")
plt.show()

