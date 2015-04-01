import re
import numpy as np
directory = "./data/"
file_list = ["mono_white_full_01", "mono_hg","mono_slit_100", "mono_slit_150","mono_slit_200","mono_slit_25", \
      "mono_slit_25_02","mono_slit_40", "mono_slit_50","mono_slit_75","mono_white_pol0","mono_cs2"]
for filename in file_list:
    print(filename)
    f = open(directory + filename+ ".dig")
    text = f.read()
    f.close()
    lines = text.split("\n")[1:]
    lamb = []
    count = []
    for u in lines[0:-1]:
        print(u)
        x,y = (re.sub(",",".",u)).split("\t")
        lamb += [float(x)]
        count += [float(y)]
    np.save("./npy/"+filename+"_lamb", np.array(lamb))
    np.save("./npy/"+filename+"_count", np.array(count))

