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

    plt.figure()
    plt.plot(t,ch_a)
    plt.show()

    return 0

# generate npy files
npy_dir = "./data_npy/"

file1 = "./data_csv/preamplifier/PA75DE01.CSV"
file2 = "./data_csv/preamplifier/PA75DE02.CSV"
file_out1 = npy_dir + "001_ch_a"
file_out2 = npy_dir + "001_ch_b"
osci_csv_npy(file1, file_out1)
osci_csv_npy(file2, file_out2)


