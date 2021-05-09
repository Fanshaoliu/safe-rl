import numpy as np

tra = np.load("trajectory.npy", allow_pickle=True)

print(np.shape(tra))

print(tra[0][0])

# for i in tra:
