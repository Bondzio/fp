import numpy as np
import matplotlib.pyplot as plt

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

def osci_txt_npy(file_in, file_out):
    """
    convert .txt to .npy arrays: 
    only two values: channel and counts
    saves: np.array([channel, hist])
    """
    f = open(file_in)
    x = f.read()
    lines = x.split("\n")
    data = np.int_([k.split("\t") for k in lines if k!=''])
    f.close()
    np.save(file_out, data.T)

    return 0

def osci_tka_npy(file_in, file_out):
    """
    convert .TKA to .npy arrays: 
    only one value: counts
    saves: counts
    """
    f = open(file_in)
    x = f.read()
    lines = x.split("\n")
    counts = np.int_(lines[2:-1])
    f.close()
    np.save(file_out, counts)

    return 0

# generate npy file
npy_dir = "./data_npy/"

"""
file1 = "./data_csv/preamplifier/PA75DE01.CSV"
file2 = "./data_csv/preamplifier/PA75DE02.CSV"
file_out1 = npy_dir + "001_ch_a"
file_out2 = npy_dir + "001_ch_b"
osci_csv_npy(file1, file_out1)
osci_csv_npy(file2, file_out2)
"""

txt_str = 'ps_137cs_03'
file1 = './data_txt/' + txt_str + '.txt'
file_out1 = npy_dir + txt_str
osci_txt_npy(file1, file_out1)

"""
file1 = "./data_tka/na_137cs_01.TKA"
file_out1 = npy_dir + "na_137_01"
osci_tka_npy(file1, file_out1)

"""

