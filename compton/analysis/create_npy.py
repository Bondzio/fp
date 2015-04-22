import numpy as np
import matplotlib.pyplot as plt
import os


def osci_csv_npy(file_in, file_out):
    """
    convert .csv to .npy arrays: 
    three values: time, channel A, channel B
    saves: t, ch_a, ch_b
    """
    f = open(file_in)
    x = f.read()
    lines = x.split("\n")
    data = np.float_([k.split(",") for k in lines[1:] if k!=''])
    f.close()
    t, ch_a = data.T
    np.save(file_out + "_t",t)
    np.save(file_out + "_y",ch_a)

    return 0
"""
file1 = data_dir + "preamplifier/PA75DE01.CSV"
file2 = data_dir + "preamplifier/PA75DE02.CSV"
file_out1 = npy_dir + "001_ch_a"
file_out2 = npy_dir + "001_ch_b"
osci_csv_npy(file1, file_out1)
osci_csv_npy(file2, file_out2)
"""

# generate npy file
data_dir = "./data/"
npy_dir = "./data_npy/"

for filename in os.listdir(data_dir):
    if filename.endswith(".TKA"):
        print(filename)
        file_in = data_dir + filename[:-4] + '.TKA'
        file_out = npy_dir + filename[:-4] 
        f = open(file_in)
        lines = f.readlines()
        counts = np.int_(lines[2:-1])
        f.close()
        np.save(file_out, counts)
    if filename.endswith(".txt"):
        print(filename)
        file_in = data_dir + filename[:-4] + '.txt'
        file_out = npy_dir + filename[:-4] 
        f = open(file_in)
        lines = f.readlines()
        f.close()
        data = np.array([line.split("\t") for line in lines])
        data = np.int16(data.T)
        np.save(file_out, data)
