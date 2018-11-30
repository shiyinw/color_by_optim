import numpy as np
from scipy import sparse
row = np.array([0, 0, 1, 2, 2, 2])
col = np.array([0, 2, 2, 0, 1, 2])
data = np.array([1, 2, 3, 4, 5, 6])
mtx = sparse.csc_matrix((data, (row, col)), shape=(3, 3))
print(mtx.todense())
print(mtx.indices)
print(mtx.data)
print(mtx.nonzero())

x, y = mtx.nonzero()
print(x, y)