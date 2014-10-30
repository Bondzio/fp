import numpy as np
import re
def vernier_to_npy(file_in, file_out):
    """
    convert .txt file in the Vernier format to .npy arrays: 
    6 values:   time/s, 
                angle/deg, 
                pyro signal/V, 
                sample signal/V,  
                wavelength/nm,
                energy/eV
    saves: energy, transmission (pyro), absorption (sample) -.npy
    """
    print(file_in)
    f = open(file_in, "rt", encoding='utf-8')
    arr_all = []
    for i, line in enumerate(f):
        if (i > 6):             #first lines are only text - AND NOT EVEN ENCODED IN ASCII
            if not line[0].isdigit(): break
            line = re.sub(",", ".", line)
            arr = np.float16(line.split('\t'))[[1, 2, 3]]
            arr_all.append(arr)
    f.close()
    arr_all = np.array(arr_all).T
    angle, transmission, absorption = arr_all
    np.save(file_out, arr_all)
    return 0

def csv_TEK_to_npy(file_in, file_out):
    """
    converts .csv file to .npy arrays
    saves: signal.npy which includes both the signal and t
    """
    print(file_in)
    f = open(file_in, 'rt')
    x = f.read()
    f.close()
    x = x.split("\n")[:-1]
    signal = np.zeros([len(x), 2])
    for i, line in enumerate(x):
        signal[i] = np.float64(line.split(',')[3:5])
    np.save(file_out, signal.T)
    return 0

def asc_to_npy(file_in, file_out):
    """
    converts .asc file to .npy arrays
    omits the first 12 lines (config)
    saves: histo.npy
    """
    print(file_in)
    f = open(file_in, "rt")
    x = f.read()
    f.close()
    histo = np.int_(x.split('\n')[from_line:-2])
    np.save(file_out, histo.T)
    return 0

# generate npy files
npy_dir = "./data_npy/"
band_gap = False
haynes_shockley = True
spectra =False

if band_gap:
    for sample_name in ["Ge_", "Si_"]:
        for suffix in ["1", "2", "background", "lamp"]:
            file_in = "../data/band_gap/1_" + sample_name + suffix + ".txt"
            file_out = npy_dir + "band_gap_" + sample_name + suffix
            vernier_to_npy(file_in, file_out)

        if sample_name == "Si_":
            for i in "12345":
                file_in = "../data/band_gap/1_Si_error_" + i + ".txt"
                file_out = npy_dir + "band_gap_error_" + i
                vernier_to_npy(file_in, file_out)

if haynes_shockley:
    for i in range(69): # 69
        file_in = "../data/shockley_haynes/TEK%04d.csv"%i
        file_out = npy_dir + "haynes_shockley_%i"%i
        csv_TEK_to_npy(file_in, file_out)


if spectra:
    i = 0
    from_line = 12 # 12
    for sample_name in ["Co", "Am"]:
        for detector_name in ["CdTe", "Si"]:
            i +=1
            file_in = "../data/spectra/3.%i_"%i + sample_name + "_" + detector_name + ".asc"
            file_out = npy_dir + "spectra_" + sample_name + "_" + detector_name
            asc_to_npy(file_in, file_out)
