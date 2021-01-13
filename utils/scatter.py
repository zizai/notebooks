import numpy as np


def scatter_add(self, index, src):
    np.add.at(self, index, src)
    return self
