# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from fractions import Fraction
from math import log

import pytest

import numpy

from spn.spn import ExposedSumSPN
from spn.spn import NominalDistribution
from spn.spn import NumericalDistribution
from spn.spn import ProductSPN
from spn.spn import SumSPN
from spn.math_util import allclose
from spn.math_util import logsumexp
from spn.numerical import Gamma
from spn.numerical import Norm
from spn.transforms import Identity

rng = numpy.random.RandomState(1)

def test_sum_normal_gamma():
    X = Identity('X')
    weights = [
        log(Fraction(2, 3)),
        log(Fraction(1, 3))
    ]
    spn = SumSPN(
        [Norm(X, loc=0, scale=1), Gamma(X, loc=0, a=1),], weights)

    assert spn.logprob(X > 0) == logsumexp([
        spn.weights[0] + spn.children[0].logprob(X > 0),
        spn.weights[1] + spn.children[1].logprob(X > 0),
    ])
    assert spn.logprob(X < 0) == log(Fraction(2, 3)) + log(Fraction(1, 2))
    samples = spn.sample(100, rng)
    assert all(s[X] for s in samples)
    spn.sample_func(lambda X: abs(X**3), 100, rng)
    with pytest.raises(ValueError):
        spn.sample_func(lambda Y: abs(X**3), 100, rng)

    spn_condition = spn.condition(X < 0)
    assert isinstance(spn_condition, NumericalDistribution)
    assert spn_condition.conditioned
    assert spn_condition.logprob(X < 0) == 0
    samples = spn_condition.sample(100, rng)
    assert all(s[X] < 0 for s in samples)

    assert spn.logprob(X < 0) == logsumexp([
        spn.weights[0] + spn.children[0].logprob(X < 0),
        spn.weights[1] + spn.children[1].logprob(X < 0),
    ])

def test_sum_normal_gamma_exposed():
    X = Identity('X')
    W = Identity('W')
    weights = [log(Fraction(2, 3)), log(Fraction(1, 3))]
    children = [Norm(X, loc=0, scale=1), Gamma(X, loc=0, a=1)]
    spn = ExposedSumSPN(children, weights, W)

    assert spn.logprob(W << {'b0'}) == log(Fraction(2, 3))
    assert spn.logprob(W << {'b1'}) == log(Fraction(1, 3))
    assert allclose(spn.logprob((W << {'b0'}) | (W << {'b1'})), 0)
    assert spn.logprob((W << {'b0'}) & (W << {'b1'})) == -float('inf')

    assert allclose(
        spn.logprob((W << {'b0', 'b1'}) & (X < 0)),
        spn.logprob(X < 0))

    assert allclose(
        spn.logprob((W << {'b0'}) & (X < 0)),
        spn.weights[0] + children[0].logprob(X < 0))

    spn_condition = spn.condition((W << {'b1'}) | (W << {'b0'}))
    assert isinstance(spn_condition, SumSPN)
    assert len(spn_condition.weights) == 2
    assert \
        allclose(spn_condition.weights[0], log(Fraction(2,3))) \
            and allclose(spn_condition.weights[0], log(Fraction(2,3))) \
        or \
        allclose(spn_condition.weights[1], log(Fraction(2,3))) \
            and allclose(spn_condition.weights[0], log(Fraction(2,3))
        )

    spn_condition = spn.condition((W << {'b1'}))
    assert isinstance(spn_condition, ProductSPN)
    assert isinstance(spn_condition.children[0], NominalDistribution)
    assert isinstance(spn_condition.children[1], NumericalDistribution)
    assert spn_condition.logprob(X < 5) == children[1].logprob(X < 5)

def test_sum_normal_nominal():
    X = Identity('X')
    children = [
        Norm(X, loc=0, scale=1),
        NominalDistribution(X, {'low': 0.3, 'high': 0.7}),
    ]
    weights = [log(Fraction(4,7)), log(Fraction(3, 7))]
    spn = SumSPN(children, weights)

    assert allclose(
        spn.logprob(X < 0),
        log(Fraction(4,7)) + log(Fraction(1,2))
    )

    # TODO: Need to fix.
    with pytest.raises(AssertionError):
        spn.logprob(X << {'a'})
    # TODO: Need to fix.
    with pytest.raises(AssertionError):
        spn.logprob(~(X << {'a'}))
    # TODO: Need to fix.
    with pytest.raises(ValueError):
        spn.logprob(X**2 << {1})
