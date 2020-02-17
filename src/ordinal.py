# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

import scipy.stats
import sympy

from .spn import OrdinalDistribution

from .sym_util import Integers
from .sym_util import IntegersPos
from .sym_util import IntegersPos0

def Bernoulli(symbol, **kwargs):
    """A Bernoulli discrete random variable."""
    return OrdinalDistribution(symbol, scipy.stats.bernoulli(**kwargs),
        sympy.Range(0, 2))

def Betabinom(symbol, **kwargs):
    """A beta-binomial discrete random variable."""
    return OrdinalDistribution(symbol, scipy.stats.betabinom(**kwargs),
        sympy.Range(0, kwargs['n']+1))

def Binom(symbol, **kwargs):
    """A binomial discrete random variable."""
    return OrdinalDistribution(symbol, scipy.stats.binom(**kwargs),
        sympy.Range(0, kwargs['n']+1))

def Boltzmann(symbol, **kwargs):
    """A Boltzmann (Truncated Discrete Exponential) random variable."""
    return OrdinalDistribution(symbol, scipy.stats.boltzmann(**kwargs),
        sympy.Range(0, kwargs['N']+1))

def Dlaplace(symbol, **kwargs):
    """A Laplacian discrete random variable."""
    return OrdinalDistribution(symbol, scipy.stats.dlaplace(**kwargs),
        Integers)

def Geom(symbol, **kwargs):
    """A geometric discrete random variable."""
    return OrdinalDistribution(symbol, scipy.stats.geom(**kwargs),
        Integers)

def Hypergeom(symbol, **kwargs):
    """A hypergeometric discrete random variable."""
    low = max(0, kwargs['N'], kwargs['N']-kwargs['M']+kwargs['n'])
    high = min(kwargs['n'], kwargs['N'])
    return OrdinalDistribution(symbol, scipy.stats.hypergeom(**kwargs),
        sympy.Range(low, high+1))

def Logser(symbol, **kwargs):
    """A Logarithmic (Log-Series, Series) discrete random variable."""
    return OrdinalDistribution(symbol, scipy.stats.logser(**kwargs),
        IntegersPos)

def Nbinom(symbol, **kwargs):
    """A negative binomial discrete random variable."""
    return OrdinalDistribution(symbol, scipy.stats.nbinom(**kwargs),
        IntegersPos0)

def Planck(symbol, **kwargs):
    """A Planck discrete exponential random variable."""
    return OrdinalDistribution(symbol, scipy.stats.planck(**kwargs),
        IntegersPos0)

def Poisson(symbol, **kwargs):
    """A Poisson discrete random variable."""
    return OrdinalDistribution(symbol, scipy.stats.poisson(**kwargs),
        IntegersPos0)

def Randint(symbol, **kwargs):
    """A uniform discrete random variable."""
    return OrdinalDistribution(symbol, scipy.stats.randint(**kwargs),
        sympy.Range(kwargs['low'], kwargs['high']))

def Skellam(symbol, **kwargs):
    """A Skellam discrete random variable."""
    return OrdinalDistribution(symbol, scipy.stats.skellam(**kwargs),
        Integers)

def Zipf(symbol, **kwargs):
    """A Zipf discrete random variable."""
    return OrdinalDistribution(symbol, scipy.stats.zipf(**kwargs),
        IntegersPos)

def Yulesimon(symbol, **kwargs):
    """A Yule-Simon discrete random variable."""
    return OrdinalDistribution(symbol, scipy.stats.yulesimon(**kwargs),
        IntegersPos)

def Atomic(symbol, **kwargs):
    """A Yule-Simon discrete random variable."""
    return Randint(symbol, low=kwargs['loc'], high=kwargs['loc']+1)