import numpy as np

Gf = 1.16637 * 10**-5   # Gev**-2, Fermi constant
mz = 91.187             # GeV, mass of Z0
alpha = 1 / 128.87      # at E = mz
sintw_sq = 0.5 - np.sqrt(0.25 - np.pi * alpha / (np.sqrt(2) * Gf * mz**2))
theta_W = np.sqrt(np.arcsin(sintw_sq))
# for nu, e-, mu-, tau-, u, c, d, s, b
I3f = np.array([0.5] + [-0.5] * 3 + [0.5] * 2 + [-0.5] * 3) 
Qf = np.array([0, -1, -1, -1, 2/3, 2/3, -1/3, -1/3, -1/3])

alpha_s = 0.12          # at E = mz; strong coupling
del_QCD = 1.05 * alpha_s / np.pi
Ncf = np.array([1] * 4 + [3 * (1 + del_QCD)] * 5)
gvf = I3f - 2 * Qf * sintw_sq
gaf = I3f

Gamma_f = Ncf * np.sqrt(2) / (12 * np.pi) * Gf * mz**3 * (gvf**2 + gaf**2)
N_nu = np.array([2, 3, 4])  # number of neutrinos
Gamma_nu = N_nu * Gamma_f[0]
Gamma_had = np.sum(Gamma_f[4:])
Gamma_l_charged = np.sum(Gamma_f[1:4])
Gamma_Z = np.sum(Gamma_f[1:]) * Gamma_nu

# Deviations of Gamma_Z for N_nu = [2, 4]:
dGamma_Z = Gamma_Z[[0, 2]] / Gamma_Z[1]

sigmafpeak  = 12 * np.pi / mz**2 * Gamma_f[1] * Gamma_f / Gamma_Z[1]**2
sigmafpeak2 = 12 * np.pi / mz**2 * Gamma_f[1] * Gamma_f / Gamma_Z[0]**2
sigmafpeak4 = 12 * np.pi / mz**2 * Gamma_f[1] * Gamma_f / Gamma_Z[2]**2
sigma_had   = np.sum(sigmafpeak[4:])
