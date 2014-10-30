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
background = False 
background2 = False 
uranium = False
measurement2_2_1 = False 
measurement2_2_2 = False
background_overnight = False
potassium = True

if background:
    file_in = "./data/background_overnight.txt"
    file_out = npy_dir + "background_"
    txt_npy(file_in, file_out)

if uranium:
    file_in = "./data/uran_measurement.txt"
    file_out = npy_dir + "uranium_"
    txt_npy(file_in, file_out)

if background2:
    file_in = "./data/background_voltage2.txt"
    file_out = npy_dir + "background2_"
    txt_npy(file_in, file_out)

if measurement2_2_1:
    file_in = "./data/sm01.txt"
    file_out = npy_dir + "measurement_2_2_1_"
    txt_npy(file_in, file_out)

if measurement2_2_2:
    file_in = "./data/2.2.2_sm_1.txt"
    file_out = npy_dir + "measurement_2_2_2_"
    txt_npy(file_in, file_out)

if background_overnight:
    file_in = "./data/background_overnight.txt"
    file_out = npy_dir + "measurement_2_3_"
    txt_npy(file_in, file_out)

if potassium:
    for i in range(2,9+1):
        file_in = "./data/2.3_Ka_%d.txt"%i
        file_out = npy_dir + "measurement_2_4_%d_"%i
        txt_npy(file_in, file_out)
