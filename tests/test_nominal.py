# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from fractions import Fraction
from math import log

import pytest

import numpy
import scipy.stats
import sympy

from spn.distributions import SumDistribution
from spn.distributions import NominalDistribution
from spn.distributions import NumericalDistribution
from spn.distributions import OrdinalDistribution
from spn.distributions import ProductDistribution

from spn.transforms import Identity
from spn.transforms import ExpNat as Exp
from spn.transforms import LogNat as Log
from spn.math_util import allclose
from spn.math_util import isinf_neg
from spn.math_util import logsumexp
from spn.math_util import logdiffexp
from spn.sym_util import Integers
from spn.sym_util import Reals
from spn.sym_util import RealsPos

rng = numpy.random.RandomState(1)

def test_nominal_distribution():
    X = Identity('X')
    probs = {'a': Fraction(1, 5), 'b': Fraction(1, 5), 'c': Fraction(3, 5)}
    dist = NominalDistribution(X, probs)
    assert dist.logprob(X << {'a'}) == sympy.log(Fraction(1, 5))
    assert dist.logprob(X << {'b'}) == sympy.log(Fraction(1, 5))
    assert dist.logprob(X << {'a', 'c'}) == sympy.log(Fraction(4, 5))
    assert dist.logprob((X << {'a'}) & ~(X << {'b'})) == sympy.log(Fraction(1, 5))
    assert dist.logprob((X << {'a', 'b'}) & ~(X << {'b'})) == sympy.log(Fraction(1, 5))
    assert dist.logprob((X << {'d'})) == -float('inf')
    assert dist.logprob((X << ())) == -float('inf')

    samples = dist.sample(100, rng)
    assert all(s[X] in dist.support for s in samples)

    samples = dist.sample_subset([X, 'A'], 100, rng)
    assert all(s[X] in dist.support for s in samples)

    assert dist.sample_subset(['f'], 100, rng) is None

    predicate = lambda X: (X in {'a', 'b'}) or X in {'c'}
    samples = dist.sample_func(predicate, 100, rng)
    assert all(samples)

    predicate = lambda X: (not (X in {'a', 'b'})) and (not (X in {'c'}))
    samples = dist.sample_func(predicate, 100, rng)
    assert not any(samples)

    func = lambda X: 1 if X in {'a'} else None
    samples = dist.sample_func(func, 100, rng)
    assert sum(1 for s in samples if s == 1) > 12
    assert sum(1 for s in samples if s is None) > 70

    with pytest.raises(ValueError):
        dist.sample_func(lambda Y: Y, 100, rng)

    dist_condition = dist.condition(X<<{'a', 'b'})
    assert dist_condition.support == {'a', 'b', 'c'}
    assert dist_condition.logprob(X << {'a'}) \
        == dist_condition.logprob(X << {'b'}) \
        == -sympy.log(2)
    assert dist_condition.logprob(X << {'c'}) == -float('inf')
