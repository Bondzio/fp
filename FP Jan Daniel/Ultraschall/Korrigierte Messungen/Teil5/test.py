import numpy as np
import fileinput

filename = "1k03V_0_HM1508.csv"
f = open(filename, "a")
for line in f:
	line = line.replace(","," ")
for line in f:
	line = line.replace(".",",")

print("Done")
