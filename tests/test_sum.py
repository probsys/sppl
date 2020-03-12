# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from fractions import Fraction
from math import log

import pytest

import numpy
import sympy

from spn.distributions import Gamma
from spn.distributions import NominalDist
from spn.distributions import Norm
from spn.math_util import allclose
from spn.math_util import isinf_neg
from spn.math_util import logdiffexp
from spn.math_util import logsumexp
from spn.spn import ContinuousReal
from spn.spn import ExposedSumSPN
from spn.spn import NominalDistribution
from spn.spn import ProductSPN
from spn.spn import SumSPN
from spn.spn import spn_simplify_sum
from spn.sym_util import NominalSet
from spn.transforms import Identity

rng = numpy.random.RandomState(1)

def test_sum_normal_gamma():
    X = Identity('X')
    weights = [
        log(Fraction(2, 3)),
        log(Fraction(1, 3))
    ]
    spn = SumSPN(
        [X >> Norm(loc=0, scale=1), X >> Gamma(loc=0, a=1),], weights)

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
    assert isinstance(spn_condition, ContinuousReal)
    assert spn_condition.conditioned
    assert spn_condition.logprob(X < 0) == 0
    samples = spn_condition.sample(100, rng)
    assert all(s[X] < 0 for s in samples)

    assert spn.logprob(X < 0) == logsumexp([
        spn.weights[0] + spn.children[0].logprob(X < 0),
        spn.weights[1] + spn.children[1].logprob(X < 0),
    ])

def test_sum_simplify_nested_sum_1():
    X = Identity('X')
    children = [
        SumSPN(
            [X >> Norm(loc=0, scale=1), X >> Norm(loc=0, scale=2)],
            [log(0.4), log(0.6)]),
        X >> Gamma(loc=0, a=1),
    ]
    spn = SumSPN(children, [log(0.7), log(0.3)])
    assert spn.children == (
        children[0].children[0],
        children[0].children[1],
        children[1]
    )
    assert allclose(spn.weights[0], log(0.7) + log(0.4))
    assert allclose(spn.weights[1], log(0.7) + log(0.6))
    assert allclose(spn.weights[2], log(0.3))

def test_sum_simplify_nested_sum_2():
    X = Identity('X')
    W = Identity('W')
    children = [
        SumSPN([
            (X >> Norm(loc=0, scale=1)) & (W >> Norm(loc=0, scale=2)),
            (X >> Norm(loc=0, scale=2)) & (W >> Norm(loc=0, scale=1))],
            [log(0.9), log(0.1)]),
        (X >> Norm(loc=0, scale=4)) & (W >> Norm(loc=0, scale=10)),
        SumSPN([
            (X >> Norm(loc=0, scale=1)) & (W >> Norm(loc=0, scale=2)),
            (X >> Norm(loc=0, scale=2)) & (W >> Norm(loc=0, scale=1)),
            (X >> Norm(loc=0, scale=8)) & (W >> Norm(loc=0, scale=3)),],
            [log(0.4), log(0.3), log(0.3)]),
    ]
    spn = SumSPN(children, [log(0.4), log(0.4), log(0.2)])
    assert spn.children == (
        children[0].children[0],
        children[0].children[1],
        children[1],
        children[2].children[0],
        children[2].children[1],
        children[2].children[2],
    )
    assert allclose(spn.weights[0], log(0.4) + log(0.9))
    assert allclose(spn.weights[1], log(0.4) + log(0.1))
    assert allclose(spn.weights[2], log(0.4))
    assert allclose(spn.weights[3], log(0.2) + log(0.4))
    assert allclose(spn.weights[4], log(0.2) + log(0.3))
    assert allclose(spn.weights[5], log(0.2) + log(0.3))

def test_sum_simplify_leaf():
    Xd0 = Identity('X') >> Norm(loc=0, scale=1)
    Xd1 = Identity('X') >> Norm(loc=0, scale=1)
    Xd2 = Identity('X') >> Norm(loc=0, scale=1)
    spn = SumSPN([Xd0, Xd1, Xd2], [log(0.5), log(0.1), log(.4)])
    assert spn_simplify_sum(spn) == spn

    spn = SumSPN([Xd0, Xd1, Xd0, Xd1], [log(0.5), log(0.1), log(.3), log(.1)])
    spn_simplified = spn_simplify_sum(spn)
    assert len(spn_simplified.children) == 2
    assert spn_simplified.children[0] == Xd0
    assert spn_simplified.children[1] == Xd1
    assert allclose(spn_simplified.weights[0], log(0.8))
    assert allclose(spn_simplified.weights[1], log(0.2))

def test_sum_simplify_product():
    A1 = Identity('A') >> Norm(loc=0, scale=1)
    A0 = Identity('A') >> Norm(loc=0, scale=1)
    B = Identity('B') >> Norm(loc=0, scale=1)
    B1 = Identity('B') >> Norm(loc=0, scale=1)
    B0 = Identity('B') >> Norm(loc=0, scale=1)
    C = Identity('C') >> Norm(loc=0, scale=1)
    C1 = Identity('C') >> Norm(loc=0, scale=1)
    D = Identity('D') >> Norm(loc=0, scale=1)
    spn = SumSPN([
        ProductSPN([A1, B, C, D]),
        ProductSPN([A0, B1, C, D]),
        ProductSPN([A0, B0, C1, D]),
    ], [log(0.4), log(0.4), log(0.2)])

    spn_simplified = spn_simplify_sum(spn)
    assert isinstance(spn_simplified, ProductSPN)
    assert isinstance(spn_simplified.children[0], SumSPN)
    assert spn_simplified.children[1] == D

    ssc0 = spn_simplified.children[0]
    assert isinstance(ssc0.children[1], ProductSPN)
    assert ssc0.children[1].children == (A0, B0, C1)

    assert isinstance(ssc0.children[0], ProductSPN)
    assert ssc0.children[0].children[1] == C

    ssc0c0 = ssc0.children[0].children[0]
    assert isinstance(ssc0c0, SumSPN)
    assert isinstance(ssc0c0.children[0], ProductSPN)
    assert isinstance(ssc0c0.children[1], ProductSPN)
    assert ssc0c0.children[0].children == (A1, B)
    assert ssc0c0.children[1].children == (A0, B1)

def test_sum_normal_gamma_exposed():
    X = Identity('X')
    W = Identity('W')
    weights = (W >> NominalDist({
        '0': Fraction(2,3),
        '1': Fraction(1,3),
    }))
    children = {
        '0': X >> Norm(loc=0, scale=1),
        '1': X >> Gamma(loc=0, a=1),
    }
    spn = ExposedSumSPN(children, weights)

    assert spn.logprob(W << {'0'}) == log(Fraction(2, 3))
    assert spn.logprob(W << {'1'}) == log(Fraction(1, 3))
    assert allclose(spn.logprob((W << {'0'}) | (W << {'1'})), 0)
    assert spn.logprob((W << {'0'}) & (W << {'1'})) == -float('inf')

    assert allclose(
        spn.logprob((W << {'0', '1'}) & (X < 1)),
        spn.logprob(X < 1))

    assert allclose(
        spn.logprob((W << {'0'}) & (X < 1)),
        spn.weights[0] + spn.children[0].logprob(X < 1))

    spn_condition = spn.condition((W << {'1'}) | (W << {'0'}))
    assert isinstance(spn_condition, SumSPN)
    assert len(spn_condition.weights) == 2
    assert \
        allclose(spn_condition.weights[0], log(Fraction(2,3))) \
            and allclose(spn_condition.weights[0], log(Fraction(2,3))) \
        or \
        allclose(spn_condition.weights[1], log(Fraction(2,3))) \
            and allclose(spn_condition.weights[0], log(Fraction(2,3))
        )

    spn_condition = spn.condition((W << {'1'}))
    assert isinstance(spn_condition, ProductSPN)
    assert isinstance(spn_condition.children[0], NominalDistribution)
    assert isinstance(spn_condition.children[1], ContinuousReal)
    assert spn_condition.logprob(X < 5) == spn.children[1].logprob(X < 5)

def test_sum_normal_nominal():
    X = Identity('X')
    children = [
        X >> Norm(loc=0, scale=1),
        X >> NominalDist({'low': Fraction(3, 10), 'high': Fraction(7, 10)}),
    ]
    weights = [log(Fraction(4,7)), log(Fraction(3, 7))]

    spn = SumSPN(children, weights)

    assert allclose(
        spn.logprob(X < 0),
        log(Fraction(4,7)) + log(Fraction(1,2)))

    assert allclose(
        spn.logprob(X << {'low'}),
        log(Fraction(3,7)) + log(Fraction(3, 10)))

    assert allclose(
        spn.logprob(~(X << {'low'})),
        logdiffexp(0, spn.logprob((X << {'low'}))))

    assert isinf_neg(spn.logprob((X < 0) & (X << {'low'})))

    assert allclose(
        spn.logprob((X < 0) | (X << {'low'})),
        logsumexp([spn.logprob(X < 0), spn.logprob(X << {'low'})]))

    assert isinf_neg(spn.logprob(X << {'a'}))
    assert allclose(spn.logprob(~(X << {'a'})), 0)

    assert allclose(
        spn.logprob(X**2 < 9),
        log(Fraction(4, 7)) + spn.children[0].logprob(X**2 < 9))

    spn_condition = spn.condition(X**2 < 9)
    assert isinstance(spn_condition, ContinuousReal)
    assert spn_condition.support == sympy.Interval.open(-3, 3)

    spn_condition = spn.condition((X**2 < 9) | X << {'low'})
    assert isinstance(spn_condition, SumSPN)
    assert spn_condition.children[0].support == sympy.Interval.open(-3, 3)
    assert spn_condition.children[1].support == NominalSet('low', 'high')
    assert isinf_neg(spn_condition.children[1].logprob(X << {'high'}))

    # FIXME: Known issue,
    # Taking the disjoint union of the event yields
    # (-3 < X < 3)
    # | ((X << (UniversalSet() \ {NominalValue(1)})) & (X <= -3))
    # | ((X << (UniversalSet() \ {NominalValue(1)})) & (3 <= X))
    # But Nominal component assigns probability zero to these clauses.
    with pytest.raises(Exception):
        spn_condition = spn.condition((X**2 < 9) | ~(X << {'1'}))
        assert spn_condition.children == spn_condition.children

    # Probability works because inclusion-exclusion does not need
    # the disjunction to be processed into disjoint events.
    assert allclose(spn.logprob((X < sympy.oo) | ~(X << {'1'})), 0)

    # FIXME: Solving this event yields Reals, eliminating the Nominal
    # branch, even though ~(X << {1}) is satisfied by that branch.
    with pytest.raises(AssertionError):
        assert spn.condition((X < 9) | ~(X << {1})) == spn
