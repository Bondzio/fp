#!/usr/bin/python3

import ephys
ephys.Quantity.yesIKnowTheDangersAndPromiseToBeCareful()
import pylab
import numpy as np

# read data files
x, pvc_na = ephys.readq('data/PVC-Natrium.txt', sep='\\t', n=2)
x, pvc_cs = ephys.readq('data/PVC-Caesium.txt', sep='\\t', n=2)
x, pvc_bg = ephys.readq('data/PVC-Untergrund.txt', sep='\\t', n=2)

naj_na    = ephys.readq('data/Natrium-Natrium.TKA', skip=2)[0]
naj_cs    = ephys.readq('data/Natrium-Caesium.TKA', skip=2)[0]
naj_bg    = ephys.readq('data/Natrium-Untergrund.TKA', skip=2)[0]

# add Poisson errors
def setPoissonError(q): q.variance = q.value + 1 * (q.value==0)
setPoissonError(pvc_na)
setPoissonError(pvc_cs)
setPoissonError(pvc_bg)
setPoissonError(naj_na)
setPoissonError(naj_cs)
setPoissonError(naj_bg)

# convert to rates
pvc_na /= 4700 * ephys.Second
pvc_cs /= 3600.96 * ephys.Second
pvc_bg /= 57018.02 * ephys.Second
naj_na /= 3600 * ephys.Second
naj_cs /= 3600 * ephys.Second
naj_bg /= 54000 * ephys.Second

# subtract background
pvc_na -= pvc_bg
pvc_cs -= pvc_bg
naj_na -= naj_bg
naj_cs -= naj_bg

# prepare fit
pvc_na.name(latex='R_{\mathrm{PVC,Na}}', label='Event Rate')
pvc_cs.name(latex='R_{\mathrm{PVC,Cs}}', label='Event Rate')
naj_na.name(latex='R_{\mathrm{NaJ,Na}}', label='Event Rate')
naj_cs.name(latex='R_{\mathrm{NaJ,Cs}}', label='Event Rate')

pvc_ch = ephys.Quantity(np.arange(len(pvc_na)), 1, label='Channel', symbol='C')
naj_ch = ephys.Quantity(np.arange(len(naj_na)), 1,  label='Channel', symbol='C')

rate = [pvc_na, pvc_cs, naj_na, naj_cs]
chan = [pvc_ch, pvc_ch, naj_ch, naj_ch]
name = ['pvc_na', 'pvc_cs', 'naj_na', 'naj_cs']
log  = [(0.0001, 1), (0.001, 10), False, False]
#log = [0]*4
maxC = [900, 600, 10000, 7000]


jobs =[
#('pvc_na', 'erf', 15, 100, 341), 
('pvc_na', 'erf', 200, 500, 341), 
('pvc_na', 'erf', 500, 850, 1277), 
#('pvc_cs', 'erf', 25, 110, 477), 
('pvc_cs', 'erf', 150, 500, 477), 
#('pvc_cs', 'erf', 200, 500), 
('naj_cs', 'gauss', 1200, 1600, 183),
('naj_cs', 'erf', 2750, 3900, 477),
('naj_cs', 'gauss', 4200, 5100, 662), 
('naj_na', 'erf', 1900, 2800, 341),
('naj_na', 'gauss', 3150, 4000, 511),
('naj_na', 'erf', 6500, 8000, 1064),
('naj_na', 'gauss', 8200, 9200, 1277),
]

result = []

for c, r, n, l, m in zip(chan, rate, name, log, maxC):
  p = ephys.Plot()
  p.error(c[0:m], r[0:m], ecolor='0.5')

  if l: p.ylog()

  i = 1
  for j in jobs:
    A  = ephys.Quantity('1/s').name(symbol='A')
    offset  = ephys.Quantity('1/s').name(symbol='c')
    mu = ephys.Quantity('').name(latex='\\mu')
    s  = ephys.Quantity('').name(latex='\\sigma')
    if j[0]==n:
      if j[1]=='gauss':
        mf = ephys.GaussCFit(A, mu, s, offset, y='')
        p.fit(mf.fit(c[j[2]:j[3]], r[j[2]:j[3]]))
      elif j[1]=='erf':
        mf = ephys.ErfCFit(A, mu, s, offset, y='')
        p.fit(mf.fit(c[j[2]:j[3]], r[j[2]:j[3]]))
      E = ephys.Quantity(j[4])
      pre = n[0:3]
      #E.store('E{}{}'.format(n[0:3], j[0]))
      #mu.store('M{}{}'.format(n[0:3], j[0]))
      result.append( (n[0:3], E, mu) )
      mu.name(latex='')
      ephys.texport.push('calib_{}_{}'.format(n, i), mu.tex())
      i+=1


  p.make()
  if l: pylab.ylim(*l)
  p.save('plots/' + n + '.png')

for scin in ['naj', 'pvc']:
  p = ephys.Plot()
  x = []
  sx = []
  y = []
  for r in result:
    if r[0] == scin:
      x.append(r[2].value)
      sx.append(r[2].stddev())
      y.append(r[1].value)
  
  C = ephys.Quantity(x, sx).name(label='Channel', symbol='C')
  E = ephys.Quantity(y, 1, 'keV').name(label='Energy')
  
  p.error(C, E)
  a = ephys.Quantity('keV', latex='a')
  b = ephys.Quantity('keV', latex='b')
  mf = ephys.PolynomFit(a, b, y='')

  p.fit(mf.fit(C, E))
  ephys.texport.push('calib_{}_a'.format(scin), a.tex(name=False))
  ephys.texport.push('calib_{}_b'.format(scin), b.tex(name=False))
  a.store('Ecal_{}_a'.format(scin))
  b.store('Ecal_{}_b'.format(scin))
  p.save('plots/calib_{}.png'.format(scin))
  
