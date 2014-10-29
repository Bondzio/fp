f = open("data/background_voltage1.txt")
text = []
U = 2400.000
for k, line in enumerate(f):
    if k < 60:
        text += [line]
    else:
        if (k-60)%4 == 0:
            U+=100
        text += ["%.3f \t %s  \t %s"%(U,"30.000",line)]
f.close() 
f2 = open("data/background_voltage2.txt","w")
f2.writelines(text)
f2.close()
