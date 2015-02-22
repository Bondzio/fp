from ROOT import TF2, TH1D
from root_numpy import random_sample

# Sample a ROOT function
func = TF2('func', 'sin(x)*sin(y)/(x*y)')
arr = random_sample(func, 1E6)

# Sample a ROOT histogram
hist = TH1D('hist', 'hist', 10, -3, 3)
hist.FillRandom('gauss')
arr = random_sample(hist, 1E6)
