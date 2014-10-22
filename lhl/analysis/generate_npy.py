import re
import numpy as np
def txt_npy(file_in, file_out):
    """
    convert .csv to .npy arrays: 
    three values: U, t (period), n (counts per second)
    saves: U, t, n
    The values are separated by tabs, and notaed in german nomenclature :( 
    -> exchange "," and "." !!!
    """
    f = open(file_in)
    x = f.read()
    f.close()
    x = re.sub(",", ".", x)
    lines = x.split("\n")
    labels = lines[0].split("\t")[:3]
    data = np.float_([k.split("\t")[:3] for k in lines[1:] if k!=''])
    U, t, n = data.T
    np.save(file_out + "U", U)
    np.save(file_out + "t", t)
    np.save(file_out + "n", n)
    return 0

# generate npy files
npy_dir = "./data_npy/"
background = True
uranium = False

if background:
    file_in = "../data/background_overnight.txt"
    file_out = npy_dir + "background_"
    txt_npy(file_in, file_out)

if uranium:
    file_in = "../data/uran_measurement.txt"
    file_out = npy_dir + "uranium_"
    txt_npy(file_in, file_out)


