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
    np.save(file_in + "_t",t)
    np.save(file_in + "_ch_a",ch_a)
    np.save(file_in + "_ch_b",ch_b)
    return 0

# generate npy files
gauge = False

if gauge:
    for q in "ab":
        file_in = "../data/2.1_gauge/2.1" + q + "_gauge_HM1508.csv"
        file_out = "./data_npy/gauge_" + q
        osci_csv_npy(file_in, file_out)
