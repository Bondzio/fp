import numpy as np

# Gauge
for q in "ab":
    f = open("../data/2.1_gauge/2.1"+q+"_gauge_HM1508.csv")
    x = f.read()
    #data = np.array([float(k) for k in x.split("\n") if k!=""])
    #np.save("./data/measure2_1"+q,data)
"""
#3.1, 4.1, 5.1 
for q in range(3,5+1):
    f = open("./data_raw/measure"+str(q)+"_1.TKA")
    x = f.read()
    data = np.array([float(k) for k in x.split("\n") if k!=""])
    np.save("./data/measure"+str(q)+"_1",data)

for q in range(1,3+1):
    f = open("./data_raw/measure6_"+str(q)+".TKA")
    x = f.read()
    data = np.array([float(k) for k in x.split("\n") if k!=""])
    np.save("./data/measure6_"+str(q),data)

f = open("./data_raw/measure7_1.TKA")
x = f.read()
data = np.array([float(k) for k in x.split("\n") if k!=""])
np.save("./data/measure7_1",data)
"""
