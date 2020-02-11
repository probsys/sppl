# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

import pytest

import numpy
import scipy.stats
import sympy

from sum_product_dsl.distributions import MixtureDistribution
from sum_product_dsl.distributions import OrdinalDistribution

from sum_product_dsl.math_util import allclose
from sum_product_dsl.math_util import logdiffexp
from sum_product_dsl.math_util import logsumexp
from sum_product_dsl.sym_util import Integers
from sum_product_dsl.transforms import Identity

rng = numpy.random.RandomState(1)

def test_ordinal_distribution_poisson():
    X = Identity('X')
    dist = OrdinalDistribution(X, scipy.stats.poisson(mu=5), Integers)

    a = dist.logprob((1 <= X) <= 7)
    b = dist.logprob(X << {1,2,3,4,5,6,7})
    c = logsumexp([dist.logprob(X << {i}) for i in range(1, 8)])
    assert allclose(a, b)
    assert allclose(a, c)
    assert allclose(b, c)

    dist_condition = dist.condition(10 <= X)
    assert dist_condition.conditioned
    assert dist_condition.support == sympy.Range(10, sympy.oo)
    assert dist_condition.logZ == logdiffexp(0, dist.logprob(X<=9))

    assert allclose(
        dist_condition.logprob(X <= 10),
        dist_condition.logprob(X << {10}))
    assert allclose(
        dist_condition.logprob(X <= 10),
        dist_condition.logpdf(10))

    samples = dist_condition.sample(100, rng)
    assert all(10 <= s[X] for s in samples)

    # Unify X = 5 with left interval to make one distribution.
    event = ((1 <= X) < 5) | ((3*X + 1) << {16})
    dist_condition = dist.condition(event)
    assert isinstance(dist_condition, OrdinalDistribution)
    assert dist_condition.conditioned
    assert dist_condition.xl == 1
    assert dist_condition.xu == 5
    assert dist_condition.support == sympy.Range(1, 6, 1)
    samples = dist_condition.sample(100, rng)

    # XXX An XFAIL, event.evaluate({X: 5.0}) is False because of floating
    # point comparison to integer.
    with pytest.raises(AssertionError):
        assert all(event.evaluate(s) for s in samples)

    # Ignore X = 14/3 as a probability zero condition.
    dist_condition = dist.condition(((1 <= X) < 5) | (3*X + 1) << {15})
    assert isinstance(dist_condition, OrdinalDistribution)
    assert dist_condition.conditioned
    assert dist_condition.xl == 1
    assert dist_condition.xu == 4
    assert dist_condition.support == sympy.Range(1, 5, 1)

    # Make a mixture of two components.
    dist_condition = dist.condition(((1 <= X) < 5) | (3*X + 1) << {22})
    assert isinstance(dist_condition, MixtureDistribution)
    assert dist_condition.distributions[0].conditioned
    assert dist_condition.distributions[0].xl == 1
    assert dist_condition.distributions[0].xu == 4
    assert dist_condition.distributions[0].support == sympy.Range(1, 5, 1)
    assert dist_condition.distributions[1].conditioned
    assert dist_condition.distributions[1].xl == 7
    assert dist_condition.distributions[1].xu == 7
    assert dist_condition.distributions[1].support == {7}

    # Condition on probability zero event.
    with pytest.raises(ValueError):
        dist.condition(((-3 <= X) < 0) | (3*X + 1) << {20})
