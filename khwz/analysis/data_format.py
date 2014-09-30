import numpy as np

# Measures 2_1
for q in "abcde":
    f = open("./data_raw/measure2_1"+q+".TKA")
    x = f.read()
    data = np.array([float(k) for k in x.split("\n") if k!=""])
    np.save("./data/measure2_1"+q,data)

for q in range(3,5+1):
    f = open("./data_raw/measure"+str(q)+"_1.TKA")
    x = f.read()
    data = np.array([float(k) for k in x.split("\n") if k!=""])
    np.save("./data/measure"+str(q)+"_1",data)
