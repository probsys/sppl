# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from fractions import Fraction
from math import log

import pytest

import numpy

from sppl.distributions import choice
from sppl.distributions import gamma
from sppl.distributions import norm
from sppl.math_util import allclose
from sppl.math_util import isinf_neg
from sppl.math_util import logsumexp
from sppl.sets import FiniteNominal
from sppl.sets import Interval
from sppl.sets import inf as oo
from sppl.spe import ContinuousLeaf
from sppl.spe import ExposedSumSPE
from sppl.spe import NominalLeaf
from sppl.spe import ProductSPE
from sppl.spe import SumSPE
from sppl.transforms import Id

def test_sum_normal_gamma():
    X = Id('X')
    weights = [
        log(Fraction(2, 3)),
        log(Fraction(1, 3))
    ]
    spe = SumSPE(
        [X >> norm(loc=0, scale=1), X >> gamma(loc=0, a=1),], weights)

    assert spe.logprob(X > 0) == logsumexp([
        spe.weights[0] + spe.children[0].logprob(X > 0),
        spe.weights[1] + spe.children[1].logprob(X > 0),
    ])
    assert spe.logprob(X < 0) == log(Fraction(2, 3)) + log(Fraction(1, 2))
    samples = spe.sample(100, prng=numpy.random.RandomState(1))
    assert all(s[X] for s in samples)
    spe.sample_func(lambda X: abs(X**3), 100)
    with pytest.raises(ValueError):
        spe.sample_func(lambda Y: abs(X**3), 100)

    spe_condition = spe.condition(X < 0)
    assert isinstance(spe_condition, ContinuousLeaf)
    assert spe_condition.conditioned
    assert spe_condition.logprob(X < 0) == 0
    samples = spe_condition.sample(100)
    assert all(s[X] < 0 for s in samples)

    assert spe.logprob(X < 0) == logsumexp([
        spe.weights[0] + spe.children[0].logprob(X < 0),
        spe.weights[1] + spe.children[1].logprob(X < 0),
    ])

def test_sum_normal_gamma_exposed():
    X = Id('X')
    W = Id('W')
    weights = W >> choice({
        '0': Fraction(2,3),
        '1': Fraction(1,3),
    })
    children = {
        '0': X >> norm(loc=0, scale=1),
        '1': X >> gamma(loc=0, a=1),
    }
    spe = ExposedSumSPE(children, weights)

    assert spe.logprob(W << {'0'}) == log(Fraction(2, 3))
    assert spe.logprob(W << {'1'}) == log(Fraction(1, 3))
    assert allclose(spe.logprob((W << {'0'}) | (W << {'1'})), 0)
    assert spe.logprob((W << {'0'}) & (W << {'1'})) == -float('inf')

    assert allclose(
        spe.logprob((W << {'0', '1'}) & (X < 1)),
        spe.logprob(X < 1))

    assert allclose(
        spe.logprob((W << {'0'}) & (X < 1)),
        spe.weights[0] + spe.children[0].logprob(X < 1))

    spe_condition = spe.condition((W << {'1'}) | (W << {'0'}))
    assert isinstance(spe_condition, SumSPE)
    assert len(spe_condition.weights) == 2
    assert \
        allclose(spe_condition.weights[0], log(Fraction(2,3))) \
            and allclose(spe_condition.weights[0], log(Fraction(2,3))) \
        or \
        allclose(spe_condition.weights[1], log(Fraction(2,3))) \
            and allclose(spe_condition.weights[0], log(Fraction(2,3))
        )

    spe_condition = spe.condition((W << {'1'}))
    assert isinstance(spe_condition, ProductSPE)
    assert isinstance(spe_condition.children[0], NominalLeaf)
    assert isinstance(spe_condition.children[1], ContinuousLeaf)
    assert spe_condition.logprob(X < 5) == spe.children[1].logprob(X < 5)

def test_sum_normal_nominal():
    X = Id('X')
    children = [
        X >> norm(loc=0, scale=1),
        X >> choice({'low': Fraction(3, 10), 'high': Fraction(7, 10)}),
    ]
    weights = [log(Fraction(4,7)), log(Fraction(3, 7))]

    spe = SumSPE(children, weights)

    assert allclose(
        spe.logprob(X < 0),
        log(Fraction(4,7)) + log(Fraction(1,2)))

    assert allclose(
        spe.logprob(X << {'low'}),
        log(Fraction(3,7)) + log(Fraction(3, 10)))

    # The semantics of ~(X<<{'low'}) are (X << String and X != 'low')
    assert allclose(
        spe.logprob(~(X << {'low'})),
        spe.logprob((X << {'high'})))
    assert allclose(
        spe.logprob((X<<FiniteNominal(b=True)) & ~(X << {'low'})),
        spe.logprob((X<<FiniteNominal(b=True)) & (X << {'high'})))

    assert isinf_neg(spe.logprob((X < 0) & (X << {'low'})))

    assert allclose(
        spe.logprob((X < 0) | (X << {'low'})),
        logsumexp([spe.logprob(X < 0), spe.logprob(X << {'low'})]))

    assert isinf_neg(spe.logprob(X << {'a'}))
    assert allclose(
        spe.logprob(~(X << {'a'})),
        spe.logprob(X<<{'low','high'}))

    assert allclose(
        spe.logprob(X**2 < 9),
        log(Fraction(4, 7)) + spe.children[0].logprob(X**2 < 9))

    spe_condition = spe.condition(X**2 < 9)
    assert isinstance(spe_condition, ContinuousLeaf)
    assert spe_condition.support == Interval.open(-3, 3)

    spe_condition = spe.condition((X**2 < 9) | X << {'low'})
    assert isinstance(spe_condition, SumSPE)
    assert spe_condition.children[0].support == Interval.open(-3, 3)
    assert spe_condition.children[1].support == FiniteNominal('low', 'high')
    assert isinf_neg(spe_condition.children[1].logprob(X << {'high'}))

    assert spe_condition == spe.condition((X**2 < 9) | ~(X << {'high'}))
    assert allclose(spe.logprob((X < oo) | ~(X << {'1'})), 0)
