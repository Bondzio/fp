# -*- coding: utf-8 -*-
import re
import numpy as np
import os

def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False
directory = "./data/"

# Mono
for filename in os.listdir(directory):
    if filename.endswith(".dig"):
        print(filename)
        f = open(directory + filename)
        text = f.read()
        f.close()
        lines = text.split("\n")[1:]
        lamb = []
        count = []
        for u in lines[0:-1]:
            x,y = (re.sub(",",".",u)).split("\t")
            lamb += [float(x)]
            count += [float(y)]
        np.save("./npy/"+filename[0:-4]+"_lamb", np.array(lamb))
        np.save("./npy/"+filename[0:-4]+"_count", np.array(count))

# CCD
for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        print(filename)
        f = open(directory + filename, encoding = 'cp1252')
        x = f.read()
        f.close()
        lines = x.split("\n")

        int_time = 0 
        q = lines[8]
        for t in q.split():
            if is_int(t):
                int_time = int(t)
        avg = 0
        q = lines[9]
        for t in q.split():
            if is_int(t):
                avg = int(t)
        data = lines[17:-2]

        lamb = []
        count = []

        for t in data:
            x,y = (re.sub(",",".",t)).split("\t")
            lamb += [float(x)]
            count += [float(y)]
        np.save("./npy/"+filename[0:-4]+"_lamb", np.array(lamb))
        np.save("./npy/"+filename[0:-4]+"_count", np.array(count))


        
