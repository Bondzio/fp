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

# generate npy files
npy_dir = "./data_npy/"
band_gap = True


if band_gap:
    for sample_name in ["Ge_", "Si_"]:
        for suffix in ["1", "2", "background", "lamp"]:
            file_in = "../data/band_gap/1_" + sample_name + suffix + ".txt"
            file_out = npy_dir + "band_gap_" + sample_name + suffix
            vernier_to_npy(file_in, file_out)

        if sample_name == "Si":
            for i in "12345":
                file_in = "../data/band_gap/1_Si_error_" + i + ".txt"
                file_out = npy_dir + "band_gap_error_" + i
                vernier_to_npy(file_in, file_out)
            

