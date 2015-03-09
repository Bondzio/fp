#!/usr/bin/python3

import ephys

apvc = ephys.Quantity.restore('Ecal_pvc_a')
bpvc = ephys.Quantity.restore('Ecal_pvc_b')
anaj = ephys.Quantity.restore('Ecal_naj_a')
bnaj = ephys.Quantity.restore('Ecal_naj_b')

ephys.Quantity.yesIKnowTheDangersAndPromiseToBeCareful()

def E_naj(C):
  return anaj + bnaj * C 

def E_pvc(C):
  return apvc + bpvc * C
