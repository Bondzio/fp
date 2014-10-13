import numpy as np
import matplotlib.pyplot as plt
import seaborn as ssn

def csv_to_npy(filename):
    '''
    create numpy arrays for t, output channelA "channelA" and input channelA "channelB".
    '''
    input_dir =  "./data/"
    output_dir = "./npy/"

    t        = []
    channelA = []
    channelB = []

    file = open(input_dir + filename+".csv")   # read data of channelA from csv table

    for i, line in enumerate(file):
        if not i==0:
            x = line.split(",")
            t.append(float(x[0]))
            channelA.append(float(x[1]))
            channelB.append(float(x[2]))

    file.close()

    t        = np.array(t)
    channelA = np.array(channelA)
    channelB = np.array(channelB)
    np.save(output_dir + filename + "_t", t)
    np.save(output_dir + filename + "_channelA", channelA)
    np.save(output_dir + filename + "_channelB", channelB)

    return t, channelA, channelB

csv_to_npy("2.1_HM1508-2")
csv_to_npy("2.2_HM1508-2")
csv_to_npy("2.3_HM1508-2")
csv_to_npy("2.4_HM1508-2")
csv_to_npy("2.5_HM1508-2")
csv_to_npy("3.1_HM1508-2")
