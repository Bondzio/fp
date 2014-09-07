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
m= 1
h = 1
a = 1
E   = np.linspace(0,1,100)
E2   = np.linspace(1,2,100)

V_0 = 0.999
k_0 = np.sqrt(2*m*E/h**2)
k_0b = np.sqrt(2*m*E2/h**2)
k_1 = np.sqrt(2*m*(V_0-E)/h**2)
#k_1b = np.sqrt(2*m*(E2-V_0)/h**2)

T = (1 - E/V_0)/((1-E/V_0) + (V_0 / (4*E)*np.sinh(k_1*a)**2))
#T2 = (1 - E2/V_0)/((1-E2/V_0) - (V_0 / (4*E2)*np.sinh(k_1b*a)**2))
plt.figure()
plt.plot(E,T)
plt.show()

