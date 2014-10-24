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
gratings = False
aperture = True
raman = False

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

if aperture:
    for q1 in ["1", "2", "3", "4", "5", "6", "7", "8"]:
        for q2 in ["a", "b"]:
            plot_suffix = q1 + q2 
            print(q1 + q2)
            file_in = "../data/2.3_lattice1_apertur/2.3_pos" + q1 + "_" + q2 + "_HM1508.csv"
            file_out = npy_dir + "aperture_" + plot_suffix
            osci_csv_npy(file_in, file_out)
if raman:
    for q in range(1,20 +1,1):
        print(q)
        file_in = "../data/2.5_phase/2.5_%d_HM1508.csv"%q
        file_out = npy_dir + "raman_nath/phase_%03d"%(q) 
        osci_csv_npy(file_in, file_out)

