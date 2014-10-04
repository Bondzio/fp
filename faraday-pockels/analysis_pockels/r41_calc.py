import numpy as np
import uncertainties as uc
from uncertainties.umath import sqrt

def read_data(name):
    f = open("../data_pockels/"+name+".tab", "r", encoding = "latin-1")
    T  = []
    U1 = []
    U2 = []
    for i,line in enumerate(f): 
        if i > 1:
            t,u1,u2,dump = line.split("\t")
            T  += [float(t)] 
            U1 += [float(u1)]
            U2 += [float(u2)]
    T  = np.array(T)
    U1 = np.array(U1)
    U2 = np.array(U2)
    return (T,U1,U2)


def r41(U):
    """TODO: Docstring for sinusmethod.

    :arg1: TODO
    :returns: TODO

    """
    lamb = uc.ufloat(632.8E-9, 1E-9)
    n1   = uc.ufloat(1.522, 0.001)
    n3   = uc.ufloat(1.477, 0.001)
    l    = uc.ufloat( 20E-3, 0.1E-3)
    d    = uc.ufloat(2.4E-3, 0.1E-3)

    r41 = lamb * d / (4 * l *U ) * (0.5*(1/n1**2 + 1/n3**2))**(3 / 2) 
    print('{:L}'.format(r41*1e12))

def sawtoothmethod():
    for i in range(1,5+1):
        T,U1,U2 = read_data("2.1.sawtooth"+str(i))
        t = np.argmax(U2)
        print(U1[t]*110)
        r41(uc.ufloat(U1[t]*110  , 5))

sawtoothmethod()


r41(2*uc.ufloat(139  , 5))

