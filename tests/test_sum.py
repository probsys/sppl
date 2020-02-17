# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from fractions import Fraction
from math import log

import pytest

import numpy

from spn.distributions import ExposedSumDistribution
from spn.distributions import NominalDistribution
from spn.distributions import NumericalDistribution
from spn.distributions import ProductDistribution
from spn.distributions import SumDistribution
from spn.math_util import allclose
from spn.math_util import logsumexp
from spn.numerical import Gamma
from spn.numerical import Norm
from spn.transforms import Identity

rng = numpy.random.RandomState(1)

def test_mixture_distribution_normal_gamma():
    X = Identity('X')
    weights = [
        log(Fraction(2, 3)),
        log(Fraction(1, 3))
    ]
    dist = SumDistribution(
        [Norm(X, loc=0, scale=1), Gamma(X, loc=0, a=1),], weights)

    assert dist.logprob(X > 0) == logsumexp([
        dist.weights[0] + dist.distributions[0].logprob(X > 0),
        dist.weights[1] + dist.distributions[1].logprob(X > 0),
    ])
    assert dist.logprob(X < 0) == log(Fraction(2, 3)) + log(Fraction(1, 2))
    samples = dist.sample(100, rng)
    assert all(s[X] for s in samples)
    dist.sample_func(lambda X: abs(X**3), 100, rng)
    with pytest.raises(ValueError):
        dist.sample_func(lambda Y: abs(X**3), 100, rng)

    dist_condition = dist.condition(X < 0)
    assert isinstance(dist_condition, NumericalDistribution)
    assert dist_condition.conditioned
    assert dist_condition.logprob(X < 0) == 0
    samples = dist_condition.sample(100, rng)
    assert all(s[X] < 0 for s in samples)

    assert dist.logprob(X < 0) == logsumexp([
        dist.weights[0] + dist.distributions[0].logprob(X < 0),
        dist.weights[1] + dist.distributions[1].logprob(X < 0),
    ])

def test_mixture_distribution_normal_gamma_exposed():
    X = Identity('X')
    W = Identity('W')
    weights = [log(Fraction(2, 3)), log(Fraction(1, 3))]
    dists = [Norm(X, loc=0, scale=1), Gamma(X, loc=0, a=1)]
    dist = ExposedSumDistribution(dists, weights, W)

    assert dist.logprob(W << {0}) == log(Fraction(2, 3))
    assert dist.logprob(W << {1}) == log(Fraction(1, 3))
    assert allclose(dist.logprob((W << {0}) | (W << {1})), 0)
    assert dist.logprob((W << {0}) & (W << {1})) == -float('inf')

    assert allclose(
        dist.logprob((W << {0, 1}) & (X < 0)),
        dist.logprob(X < 0))

    assert allclose(
        dist.logprob((W << {0}) & (X < 0)),
        dist.weights[0] + dists[0].logprob(X < 0))

    dist_condition = dist.condition((W << {1}) | (W << {0}))
    assert isinstance(dist_condition, SumDistribution)
    assert len(dist_condition.weights) == 2
    assert \
        allclose(dist_condition.weights[0], log(Fraction(2,3))) \
            and allclose(dist_condition.weights[0], log(Fraction(2,3))) \
        or \
        allclose(dist_condition.weights[1], log(Fraction(2,3))) \
            and allclose(dist_condition.weights[0], log(Fraction(2,3))
        )

    dist_condition = dist.condition((W << {1}))
    assert isinstance(dist_condition, ProductDistribution)
    assert isinstance(dist_condition.distributions[0], NominalDistribution)
    assert isinstance(dist_condition.distributions[1], NumericalDistribution)
    assert dist_condition.logprob(X < 5) == dists[1].logprob(X < 5)
