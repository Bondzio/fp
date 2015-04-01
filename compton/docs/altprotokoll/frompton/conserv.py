#!/usr/bin/python3

import ephys
import numpy as np
import pylab
import math
from ec import *

ephys.Quantity.yesIKnowTheDangersAndPromiseToBeCareful()

x, pvc_zu = ephys.readq('data/PVC_zufall2.txt', sep='\\t', n=2)
naj_zu    = ephys.readq('data/Natrium_Zufall2.TKA', skip=2)[0]

# add Poisson errors
def setPoissonError(q): q.variance = q.value + 1 * (q.value==0)
setPoissonError(pvc_zu)
setPoissonError(naj_zu)

# convert to rates
pvc_zu /= 57653 * ephys.Second
naj_zu /= 57639 * ephys.Second

naj_ch = ephys.Quantity(np.arange(len(naj_zu)), 1,  label='Channel')
pvc_ch = ephys.Quantity(np.arange(len(pvc_zu)), 1,  label='Channel')

k = [
(20, 150, [0.03, 24, 33, 3/3600]),
(20, 150, [0.06, -70, 70, 10/3600]),
(25, 200, None),
(25, 300, None),
(50, 600, None),
(50, 600, None),
(50, 600, None),
(50, 600, None),
(50, 600, None),
]

j = [
(4000, 6000),
(3500, 5300),
(3000, 5000),
(2500, 4000),
(2000, 4000),
(1600, 3000),
(1600, 3000),
(1500, 2500),
(1500, 2500),
]

result = []
nval = []
pval = []
perr = []
nerr = []
theta = []
ival = []
ierr = []

pvcPlot = ephys.Plot()
najPlot = ephys.Plot()
col = 'rgbymckrgbmck'






for angle in range(0, 121, 15):
  naj    = ephys.readq('data/Natrium_{}.TKA'.format(angle), skip=2)[0]
  pvc    = ephys.readq('data/PVC_{}.txt'.format(angle), n=2, sep=r'\t')[1]
  setPoissonError(naj)
  setPoissonError(pvc)
  naj   /= 3600  * ephys.Second
  pvc   /= 3600  * ephys.Second
  if angle==0: naj *= 2
  if angle==0: pvc *= 2

  naj.prefunit('1/hr')
  pvc.prefunit('1/hr')
  naj.name(latex='R_{{\mathrm{{NaJ,{}^\circ}}}}'.format(angle), label='Event Rate')
  pvc.name(latex='R_{{\mathrm{{PVC,{}^\circ}}}}'.format(angle), label='Event Rate')
 
### naj plot
  p = ephys.Plot()
  p.error(naj_ch, naj, ecolor='0.5')

  A  = ephys.Quantity('1/s').name(symbol='A')
  offset  = ephys.Quantity('1/s').name(symbol='c')
  mu_naj = ephys.Quantity('').name(latex='\\mu')
  s  = ephys.Quantity('').name(latex='\\sigma')
  gf = ephys.GaussCFit(A, mu_naj, s, offset)
  lo = j[angle//15][0]
  hi = j[angle//15][1]
  gf.fit(naj_ch[lo:hi], naj[lo:hi])
  p.fit(gf)
  I = math.sqrt(2*math.pi) * s * A

  najPlot.line(naj_ch, gf.func(naj_ch, A, mu_naj, s, offset*0), '-'+col[angle//15]) 
  p.make()
  pylab.xlim(0, 6000)
  pylab.ylim(-2, 30)
  #p.save('plots/naj_{}.png'.format(angle))

### pvc plot
  p = ephys.Plot()
  p.error(pvc_ch, pvc, ecolor='0.5')

  A  = ephys.Quantity('1/s').name(symbol='A')
  offset  = ephys.Quantity('1/s').name(symbol='c')
  mu_pvc = ephys.Quantity('').name(latex='\\mu')
  s  = ephys.Quantity('').name(latex='\\sigma')
  gf = ephys.GaussCFit(A, mu_pvc, s, offset)
  lo = k[angle//15][0]
  hi = k[angle//15][1]
  es = k[angle//15][2]
  if es:
    A.value = es[0]
    mu_pvc.value = es[1]
    s.value = es[2]
    offset.value = es[3]
    gf.fit(pvc_ch[lo:hi], pvc[lo:hi], estimate=False)
  else:
    gf.fit(pvc_ch[lo:hi], pvc[lo:hi])
  p.fit(gf)
  p.make()
  pylab.ylim(-4, 150)
  pylab.xlim(-4, 600)
  #p.save('plots/pvc_{}.png'.format(angle))
  pvcPlot.line(pvc_ch, gf.func(pvc_ch, A, mu_pvc, s, offset*0), '-'+col[angle//15])

  E = E_naj(mu_naj).prefunit('keV')
  ephys.texport.push('najC{}'.format(angle), mu_naj.tex(name=False))
  ephys.texport.push('najE{}'.format(angle), E.tex(name=False))
  nval.append(E.value)
  nerr.append(E.stddev())
  ival.append(I.value)
  ierr.append(I.stddev())
  E = E_pvc(mu_pvc).prefunit('keV')
  ephys.texport.push('pvcC{}'.format(angle), mu_pvc.tex(name=False))
  ephys.texport.push('pvcE{}'.format(angle), E.tex(name=False))
  pval.append(E.value)
  perr.append(E.stddev())


pvcPlot.save('plots/pvc.png')
najPlot.make()
najPlot._yaxis.prefunit('1/hr').name(label='Event Rate', latex='R')
pylab.xlim(0, 7000)
najPlot.save('plots/naj.png')

Eg = 661.6
me = 511
def P(t):
  return 1 / (1 + Eg / me * (1-np.cos(t)))

p = ephys.Plot()

Enaj = ephys.Quantity(nval, nerr, unit='J').prefunit('keV').name(label='Particle Energy', latex='E')
Epvc = ephys.Quantity(pval, perr, unit='J').prefunit('keV').name(label='Particle Energy', latex='E')
angle = ephys.Quantity(range(0, 121, 15), 3, unit='deg').name(label='Scattering Angle', latex='\\theta')
p.error(angle, Enaj, fmt='.b') #, ecolor='k')
p.error(angle, Epvc, fmt='.r') #, ecolor='k')
p.error(angle, Epvc+Enaj, fmt='.g') #, ecolor='k')

angle = np.linspace(0, 120/180 * np.pi, 200)
Egamma = ephys.Quantity(Eg * P(angle), unit='keV')
Eelect = ephys.Quantity(Eg - Eg * P(angle), unit='keV')
Etotal = Egamma + Eelect
angle = ephys.Quantity(angle, unit='rad')
p.line(angle, Egamma, label='Scattered Photon')
p.line(angle, Eelect, label='Scattered Electron', fmt='-r')
p.line(angle, Etotal, label='Total Energy', fmt='-g')
p.legend()
p.make()
pylab.ylim(-50, 1000)
p.save('plots/conservation.png')

p = ephys.Plot()

mu = ephys.Quantity([0.089, 0.091, 0.091, 0.098, 0.108, 0.12, 0.12,  0.12,
0.136], 0.01, '1/cm')
epsilon = ephys.Quantity([0.4, 0.41, 0.45, 0.51, 0.55, 0.63, 0.65, 0.7, 0.8],
0.03)
d = ephys.Quantity('1.45+-0.1 cm')
d2 = ephys.Quantity('0.725+-0.3 cm')

tmp = ephys.Quantity(ival, ierr, '1/s')
for angle, t in zip(range(0, 121, 15), tmp):
  ephys.texport.push('I{}'.format(angle), (t*ephys.Quantity('hr')).tex(name=False))
I = ephys.Quantity(ival, ierr, unit='1/s') 
I_in = ephys.Quantity.restore('intensity')
n = ephys.Quantity('1.34e23 / cm^3')
d_Omega = ephys.Quantity('4.756 +- 0.1 cm')**2  / ephys.Quantity('11.5 +- 0.5 cm')**2 * math.pi
dcs = I / epsilon / (I_in / epsilon[0]) / (n * d * d_Omega) * 1 / (1 - ephys.exp(-mu * d2) * ephys.exp(-mu[0] * d2)) 
for angle, t in zip(range(0, 121, 15), dcs):
  ephys.texport.push('dcs{}'.format(angle), (t / ephys.Quantity('mbarn')).tex(name=False))
print(dcs)
dcs.name(label='Differential Cross Section', latex=r'\frac{\mathrm{d}\sigma}{\mathrm{d}\Omega}')
dcs.prefunit('mbarn')
angle = ephys.Quantity(range(0, 121, 15), 3, unit='deg').name(label='Scattering Angle', latex='\\theta')
p.error(angle, dcs)

Eg = 661.6
me = 511
re = 2.818e-15

angle = ephys.Quantity(range(0, 121), 0, unit='deg')

def P(t):
  return 1 / (1 + Eg / me * (1-ephys.cos(t)))
def KN(t):
  return 1/2 * re**2 * P(t) * (1-P(t) * ephys.sin(t)**2 + P(t)**2)

theo = KN(angle) * ephys.Meter**2
p.line(angle, theo)

p.save('plots/cross.png')

