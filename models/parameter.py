import math

import scipy.sparse as sps
from scipy.stats import uniform, norm


"""
Notes
-----

We want the spectral radius (rho) of the weight matrix to be close to 1.
Otherwise, the power sequence of the matrix goes to 0 if rho < 1 or inf if rho > 1

https://en.wikipedia.org/wiki/Spectral_radius#Power_sequence
http://danielrapp.github.io/rnn-spectral-radius/
"""


class Parameter(sps.coo_matrix):
    def glorot_uniform(self, density, gain=1.):
        m, n = self.shape
        bound = gain * math.sqrt(6. / ((m + n) * density))
        uniform_rvs = uniform(-bound, bound).rvs
        mat = sps.random(m, n, density, data_rvs=uniform_rvs)
        self.__init__(mat)

    def glorot_normal(self, density, gain=1.):
        m, n = self.shape
        stdev = gain * math.sqrt(2. / ((m + n) * density))
        uniform_rvs = norm(0., stdev).rvs
        mat = sps.random(m, n, density, data_rvs=uniform_rvs)
        self.__init__(mat)

    def kron_power(self, k):
        if k == 1:
            return self
        elif k > 1:
            return sps.kron(self.kron_pow(k - 1), self)
        else:
            raise ValueError('k must be greater than 0')

    def __repr__(self):
        return 'Parameter containing:\n' + super(Parameter, self).__repr__()
