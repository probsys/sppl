# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from math import log

import pytest

import numpy
import scipy.stats
import sympy

from spn.distributions import NumericalDistribution
from spn.distributions import SumDistribution
from spn.math_util import allclose
from spn.math_util import isinf_neg
from spn.math_util import logdiffexp
from spn.numerical import Gamma
from spn.numerical import Norm
from spn.sym_util import Reals
from spn.transforms import Identity

rng = numpy.random.RandomState(1)

def test_numeric_distribution_normal():
    X = Identity('X')
    dist = Norm(X, loc=0, scale=1)

    assert allclose(dist.logprob(X > 0), -log(2))
    assert allclose(dist.logprob(abs(X) < 2), log(dist.dist.cdf(2) - dist.dist.cdf(-2)))

    assert allclose(dist.logprob(X**2 > 0), 0)
    assert allclose(dist.logprob(abs(X) > 0), 0)
    assert allclose(dist.logprob(~(X << {1})), 0)

    assert isinf_neg(dist.logprob(X**2 - X + 10 < 0))
    assert isinf_neg(dist.logprob(abs(X) < 0))
    assert isinf_neg(dist.logprob(X << {1}))

    dist.sample(100, rng)
    dist.sample_subset([X], 100, rng)
    assert dist.sample_subset([], 100, rng) is None
    dist.sample_func(lambda X: X**2, 1, rng)
    dist.sample_func(lambda X: abs(X)+X**2, 1, rng)
    dist.sample_func(lambda X: X**2 if X > 0 else X**3, 100, rng)

    dist_condition_a = dist.condition((X < 2) | (X > 10))
    samples = dist_condition_a.sample(100, rng)
    assert all(s[X] < 2 for s in samples)

    dist_condition_b = dist.condition((X < -10) | (X > 10))
    assert isinstance(dist_condition_b, SumDistribution)
    assert allclose(dist_condition_b.weights[0], -log(2))
    assert allclose(dist_condition_b.weights[0], dist_condition_b.weights[1])

    for event in [(X<-10), (X>3)]:
        dist_condition_c = dist.condition(event)
        assert isinstance(dist_condition_c, NumericalDistribution)
        assert isinf_neg(dist_condition_c.logprob((-1 < X) < 1))
        samples = dist_condition_c.sample(100, rng)
        assert all(s[X] in event.values for s in samples)

    with pytest.raises(ValueError):
        dist.condition((X > 1) & (X < 1))

    with pytest.raises(ValueError):
        dist.condition(X << {1})

    with pytest.raises(ValueError):
        dist.sample_func(lambda Z: Z**2, 1, rng)

    x = dist.logprob((X << {1, 2}) | (X < -1))
    assert allclose(x, dist.logprob(X < -1))

def test_numeric_distribution_gamma():
    X = Identity('X')

    dist = Gamma(X, a=1, scale=1)
    with pytest.raises(ValueError):
        dist.condition((X << {1, 2}) | (X < 0))

    # Intentionally set Reals as the domain to exercise an important
    # code path in dist.condition (Union case with zero weights).
    dist = NumericalDistribution(X, scipy.stats.gamma(a=1, scale=1), Reals)
    assert isinf_neg(dist.logprob((X << {1, 2}) | (X < 0)))
    with pytest.raises(ValueError):
        dist.condition((X << {1, 2}) | (X < 0))

    dist_condition = dist.condition((X << {1,2} | (X <= 3)))
    assert isinstance(dist_condition, NumericalDistribution)
    assert dist_condition.conditioned
    assert dist_condition.support == sympy.Interval(-sympy.oo, 3)
    assert allclose(
        dist_condition.logprob(X <= 2),
        logdiffexp(dist.logprob(X<=2), dist.logprob(X<=0))
            - dist_condition.logZ)
