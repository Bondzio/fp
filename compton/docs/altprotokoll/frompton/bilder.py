#!/usr/bin/python3

import numpy as np
import pylab
import math

Eg = 661.6
me = 511
re = 2.818e-15

def P(t):
  return 1 / (1 + Eg / me * (1-np.cos(t)))

def KN(t):
  return 1/2 * re**2 * P(t) * (1-P(t) * np.sin(t)**2 + P(t)**2)

theta = np.linspace(0, 180, 200)

pylab.plot(theta, KN(theta*math.pi/180) / 10**-31)
pylab.xlabel('Scattering Angle $\\theta / {}^{\circ}$')
pylab.ylabel('Differential Cross Section $\\frac{\\mathrm{d}\\sigma}{\\mathrm{d}\\Omega}$ / $\mathrm{mbarn}$')
pylab.grid()

pylab.savefig('plots/kleinn.png', dpi=300)
pylab.clf()



pylab.plot(theta, [1]*len(theta), '-g')
pylab.plot(theta, P(theta*math.pi/180), '-b')
pylab.plot(theta, 1- P(theta*math.pi/180), '-r')

pylab.ylim(0, 1.1)
pylab.xlabel('Scattering Angle $\\theta / {}^{\circ}$')
pylab.ylabel('Relative Energy $E/E_\\gamma$')
pylab.grid()
d = {'fontsize':16, 'weight':'bold', 'horizontalalignment': 'right'}
pylab.text(160, 1.03, "$E'_e + E'_\\gamma$", color='g', **d)
pylab.text(160, 0.75, "$E'_e$", color='r', **d)
pylab.text(160, 0.2, "$E'_\\gamma$", color='b', **d)

pylab.savefig('plots/energieerhaltung.png', dpi=300)

