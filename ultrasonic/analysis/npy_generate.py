import numpy as np
def osci_csv_npy(file_in, file_out):
    """
    convert .csv to .npy arrays: 
    three values: time, channel A, channel B
    saves: t, ch_a, ch_b
    """
    f = open(file_in)
    x = f.read()
    lines = x.split("\n")
    labels = lines[0].split(",")[:3]
    data = np.float_([k.split(",")[:3] for k in lines[1:] if k!=''])
    f.close()
    t, ch_a, ch_b = data.T
    np.save(file_out + "_t",t)
    np.save(file_out + "_ch_a",ch_a)
    np.save(file_out + "_ch_b",ch_b)
    return 0

# generate npy files
npy_dir = "./data_npy/"
gauge = False
gratings = True

if gauge:
    for q in "ab":
        file_in = "../data/2.1_gauge/2.1" + q + "_gauge_HM1508.csv"
        file_out = npy_dir + "gauge_" + q
        osci_csv_npy(file_in, file_out)
if gratings:
    for q in ["1", "2a", "2b", "3", "4a", "4b", "5a", "5b"]:
        print('b')
        file_in = "../data/2.2_lattices/2.2_lattice" + q + "_HM1508.csv"
        file_out = npy_dir + "grating_" + q
        osci_csv_npy(file_in, file_out)
