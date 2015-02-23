from root_numpy import root2rec
import numpy as np
x = root2rec("data/original/daten_4.root")
np.save("data/data.npy",x)
x = root2rec("data/original/ee.root")
np.save("data/ee.npy",x)
x = root2rec("data/original/qq.root")
np.save("data/qq.npy",x)
x = root2rec("data/original/mm.root")
np.save("data/mm.npy",x)
x = root2rec("data/original/tt.root")
np.save("data/tt.npy",x)

