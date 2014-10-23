import re
import numpy

file_in = "../data/sm01_only_n.txt"
file_out = "../data/sm01.txt"
f = open(file_in)
x = f.read()
f.close()
lines = x.split("\n")
n_lines = [line for line in lines if line!='']
i = 0
f = open(file_out, "w+")
for j, line in enumerate(n_lines):
    if line[:2] == '15':
        if i >= 5:
            line = str(15 + int(i / 5)) + line[2:]
            n_lines[j] = line
        i += 1
    f.write(line + "\n")

f.close()


