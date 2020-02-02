# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from math import isinf

import numpy

from scipy.special import logsumexp

# Implementation of log1mexp and logdiffexp from PyMC3 math module.
# https://github.com/pymc-devs/pymc3/blob/master/pymc3/math.py
def log1mexp(x):
    if x < 0.683:
        return numpy.log(-numpy.expm1(-x))
    else:
        return numpy.log1p(-numpy.exp(-x))

def logdiffexp(a, b):
    return a + log1mexp(a - b)

def lognorm(array):
    M = logsumexp(array)
    return [a - M for a in array]

def logflip(logp, array, size, rng):
    p = numpy.exp(lognorm(logp))
    return flip(p, array, size, rng)

def flip(p, array, size, rng):
    p = normalize(p)
    return rng.choice(array, size=size, p=p)

def normalize(p):
    return numpy.asarray(p, dtype=float) / sum(p)

def allclose(values, x):
    return numpy.allclose(values, x)

def isinf_pos(x):
    return isinf(x) and x > 0

def isinf_neg(x):
    return isinf(x) and x < 0
