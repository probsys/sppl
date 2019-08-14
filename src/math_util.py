# Copyright 2019 MIT Probabilistic Computing Project.
# See LICENSE.txt

from math import exp
from math import fabs
from math import log

import numpy

from scipy.special import logsumexp

def logdiffexp(x1, x2):
    M = x1
    xx1 = x1 - M
    xx2 = x2 - M
    return log(exp(xx1) - exp(xx2)) + M

def lognorm(array):
    M = logsumexp(array)
    return [a - M for a in array]

def logflip(logp, array, size, rng):
    p = numpy.exp(lognorm(logp))
    return pflip(p, array, size, rng)

def pflip(p, array, size, rng):
    p = normalize(p)
    return rng.choice(array, size=size, p=p)

def normalize(p):
    return numpy.asarray(p, dtype=float) / sum(p)

def allclose(values, x):
    assert numpy.allclose(values, x)
