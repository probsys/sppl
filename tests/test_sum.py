# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from fractions import Fraction
from math import log

import pytest

import numpy

from sum_product_dsl.distributions import NumericalDistribution
from sum_product_dsl.distributions import SumDistribution
from sum_product_dsl.math_util import logsumexp
from sum_product_dsl.numerical import Gamma
from sum_product_dsl.numerical import Norm
from sum_product_dsl.transforms import Identity

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