import numpy as np
# transform angle 17x degrees into negative angles
input_dir = "../data_faraday/"
output_dir = input_dir
for i in range(4):
    f = open(input_dir + "2.%i.csv"%(i+1))   # read data from csv table
    a = []
    for line in f:
        a += [float(line)]
    f.close()
    a = np.array(a)
    for k, ai in enumerate(a):
        if ai > 90:  # dont take 'positive' values
            a[k] = ai - 180
    np.save(output_dir + "a_%i"%(i+1), a)

# create np arrays for I
Imax = 4.8
I = np.arange(0,Imax+0.2,0.2)
I = np.append(I, 4.94)
np.save(output_dir + "i_1", I)

Imax = -4.8
I = np.arange(0,Imax-0.2,-0.2)
I = np.append(I, -4.90)
np.save(output_dir + "i_2", I)

# i_3.npy is already existing (random numbers)

Imax = 4.60
I = np.arange(0,Imax+0.2,0.2)
I = np.append(I, 4.70)
np.save(output_dir + "i_4", I)


# create random values for I, only for measuremente 2.3
"""
if i == 3:
    create = 0
    if create:
        Imax = 5
        n = 27
        np.random.seed(5)
        rands = np.random.random(25) * 5
        a = np.round(np.sort(rands), decimals=2)
        np.save("2.3_rand_i", a)
        print(["%.2f"%k for k in a])
"""
