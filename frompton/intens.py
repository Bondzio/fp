#!/usr/bin/python3

from ephys import *

Quantity.yesIKnowTheDangersAndPromiseToBeCareful()

counts = Quantity()
daq('data/Natrium_Intensitaet.TKA', counts, skip=2)
counts = counts | sqrt(counts.value)
rate = counts / 3600 / Second

back = Quantity()
daq('data/Natrium-Untergrund.TKA', counts, skip=2)
back = back | sqrt(back.value)
back = back / 54000 / Second

rate -= back
rate.name(label='Event Rate', latex='R_{\\mathrm{NaJ, Int}}')

ch = Quantity(range(len(rate)),0.1,  label='Channel', symbol='C')

p = Plot()
p.error(ch, rate, ecolor='0.5')

A = Quantity('1/s', symbol='A')
mu = Quantity('', latex='\\mu')
sigma = Quantity('', latex='\\sigma')
offset = Quantity('1/s', latex='c')

f = GaussCFit(A, mu, sigma, offset)
f.fit(ch[4300:6000], rate[4300:6000])
p.fit(f)

I = math.sqrt(2*math.pi) * sigma * A
print(I)

p.save('plots/intensitaet.png')
I.store('intensity')

