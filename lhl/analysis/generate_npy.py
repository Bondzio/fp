
import numpy as np
def txt_npy(file_in, file_out):
    """
    convert .csv to .npy arrays: 
    three values: time, channel A, channel B
    saves: t, ch_a, ch_b
    """
    f = open(file_in)
    x = f.read()
    f.close()
    lines = x.split("\n")
    labels = lines[0].split(" ")[:3]
    data = np.float_([k.split(",")[:3] for k in lines[1:] if k!=''])
    t, ch_a, ch_b = data.T
    np.save(file_out + "_t",t)
    np.save(file_out + "_ch_a",ch_a)
    np.save(file_out + "_ch_b",ch_b)
    return 0

# generate npy files
npy_dir = "./data_npy/"
background = False
uranium = True

if uranium:
    file_in = "../data/uran_measurement.txt"
    file_out = npy_dir + "uranium"
    f = open(file_in)
    x = f.read()
    lines = x.split("\n")
    f.close()
    #txt_npy(file_in, file_out)
