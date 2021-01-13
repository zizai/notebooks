import numpy as np
import scipy.sparse as sps


def main():
    x = np.random.randn(100, 1024).astype(np.float16)
    w = np.random.rand(1024, 1024).astype(np.float16)

    x = np.matmul(x, w)
    print(x.shape, x.nbytes)

    w_dim = 100
    w = sps.rand(w_dim, w_dim, density=3/w_dim, dtype=np.float16)
    print(w.__repr__(), w.data.nbytes)
    w1 = sps.kron(w, w)
    print(w1.__repr__(), w1.data.nbytes)
    w1 = sps.kron(w1, w)
    print(w1.__repr__(), w1.data.nbytes)
    w1 = sps.kron(w1, w)
    print(w1.__repr__(), w1.data.nbytes)


if __name__ == '__main__':
    main()
