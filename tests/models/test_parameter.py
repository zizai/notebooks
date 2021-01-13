import scipy
from scipy.sparse import coo_matrix

from neuroblast.models import Parameter


def test_glorot():
    density = 0.1
    N = 10
    weight = Parameter(coo_matrix((N, N)))

    weight.glorot_uniform(density)
    print(scipy.sparse.linalg.norm(weight))

    weight.glorot_normal(density)
    print(scipy.sparse.linalg.norm(weight))

