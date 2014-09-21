import numpy as np
import matplotlib.pyplot as plt
import re

fig = plt.figure()
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
def plot_data(name, title):
    T,U1,U2 = read_data(name)
    plt.plot(T*1000,U1, label = "applied voltage")
    plt.plot(T*1000,U2, label = "Voltage at photodiode" )
    plt.title(title)
    plt.xlabel("time $t/ ms$")
    plt.ylabel("Voltage $U/ V$")
    plt.grid(True)
    name2 = re.sub("\.","",name) 
    plt.legend(loc = 1)
    plt.savefig("figures/"+name2+".pdf")
    fig.clear()

def plot_12():
    # Only with Rectangular voltage
    for i in range(1,10+1):
        plot_data("1.2.sawtooth"+str(i), "Oscilloscope  1.2 Sawtooth "+str(i))
    # already calibrated
    for i in range(1,5+1):
        plot_data("2.1.sawtooth"+str(i), "Oscilloscope  2.1 Sawtooth "+str(i))
        
def plot_22():
    # Sinus
    f = open("U_DC.csv")
    x = f.read()
    f.close()
    U_DC = np.array([float(k) for k in x.split("\n") if not(k == "")])
    for i in range(1,30):
        plot_data("2.2.sinus%02d"%i, "Oscilloscope  2.2 Sinus %02d with $U_{DC}=%.3f$"%(i,U_DC[i-1]))

def plot_23():
    # different frequencies
    plot_data("2.3.sinus01", "Oscilloscope  2.3 Sinus at $f=3.0$ kHz")
    plot_data("2.3.sinus02", "Oscilloscope  2.3 Sinus at $f=18.0$ kHz")
    plot_data("2.3.sinus03", "Oscilloscope  2.3 Sinus at $f=2.0$ kHz")
def plot_24():
    U_DC = [141.0,140.5,140.0, 139.5,139.0, 138.5, 138.0, 135.0] 
    print(len(U_DC))
    for i in range(7,14+1):
        plot_data("2.3.sinus%02d"%i, "Oscilloscope  2.3 Sinus %02d with $U_{DC}=%.3f$"%(i,U_DC[i-7]))
        
    U_DC = [135.5,137.0, 138.0,138.5,139.0,139.5,140.0,140.5,141.0,142.0,144.0,137.0,137.5,138.0,138.5, 139.0,139.5, 140.0,140.5,141.0]
    print(len(U_DC))
    for i in range(1,20+1):
        plot_data("2.4.sinus%02d"%i, "Oscilloscope  2.4 Sinus %02d with $U_{DC}=%.3f$"%(i,U_DC[i-1]))

plot_24()


