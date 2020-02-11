# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from math import log

import pytest

import numpy
import scipy.stats
import sympy

from sum_product_dsl.distributions import SumDistribution
from sum_product_dsl.distributions import NumericalDistribution
from sum_product_dsl.distributions import ProductDistribution

from sum_product_dsl.math_util import allclose
from sum_product_dsl.math_util import isinf_neg
from sum_product_dsl.math_util import logdiffexp
from sum_product_dsl.math_util import logsumexp
from sum_product_dsl.sym_util import Reals
from sum_product_dsl.sym_util import RealsPos
from sum_product_dsl.transforms import ExpNat as Exp
from sum_product_dsl.transforms import Identity
from sum_product_dsl.transforms import LogNat as Log

rng = numpy.random.RandomState(1)

def test_product_distribution_normal_gamma():
    X1 = Identity('X1')
    X2 = Identity('X2')
    X3 = Identity('X3')
    X4 = Identity('X4')
    dists = [
        ProductDistribution([
            NumericalDistribution(X1, scipy.stats.norm(loc=0, scale=1), Reals),
            NumericalDistribution(X4, scipy.stats.norm(loc=10, scale=1), Reals)
        ]),
        NumericalDistribution(X2, scipy.stats.gamma(loc=0, a=1), RealsPos),
        NumericalDistribution(X3, scipy.stats.norm(loc=2, scale=3), Reals),
    ]
    dist = ProductDistribution(dists)
    assert dist.distributions == [
        dists[0].distributions[0],
        dists[0].distributions[1],
        dists[1],
        dists[2],
    ]
    assert dist.get_symbols() == frozenset([X1, X2, X3, X4])

    samples = dist.sample(2, rng)
    assert len(samples) == 2
    for sample in samples:
        assert len(sample) == 4
        assert all([X in sample for X in (X1, X2, X3, X4)])

    samples = dist.sample_subset((X1, X2), 10, rng)
    assert len(samples) == 10
    for sample in samples:
        assert len(sample) == 2
        assert X1 in sample
        assert X2 in sample

    samples = dist.sample_func(lambda X1, X2, X3: (X1, (X2**2, X3)), 1, rng)
    assert len(samples) == 1
    assert len(samples[0]) == 2
    assert len(samples[0][1]) == 2

    with pytest.raises(ValueError):
        dist.sample_func(lambda X1, X5: X1 + X4, 1, rng)

def test_inclusion_exclusion_basic():
    X = Identity('X')
    Y = Identity('Y')
    dist = ProductDistribution([
        NumericalDistribution(X, scipy.stats.norm(loc=0, scale=1), Reals),
        NumericalDistribution(Y, scipy.stats.gamma(a=1), RealsPos),
    ])

    for func_prob in [dist.logprob, dist.logprob_inclusion_exclusion]:
        a = func_prob(X > 0.1)
        b = func_prob(Y < 0.5)
        c = func_prob((X > 0.1) & (Y < 0.5))
        d = func_prob((X > 0.1) | (Y < 0.5))
        e = func_prob((X > 0.1) | ((Y < 0.5) & ~(X > 0.1)))
        f = func_prob(~(X > 0.1))
        g = func_prob((Y < 0.5) & ~(X > 0.1))

        assert allclose(a, dist.distributions[0].logprob(X > 0.1))
        assert allclose(b, dist.distributions[1].logprob(Y < 0.5))

        # Pr[A and B]  = Pr[A] * Pr[B]
        assert allclose(c, a + b)
         # Pr[A or B] = Pr[A] + Pr[B] - Pr[AB]
        assert allclose(d, logdiffexp(logsumexp([a, b]), c))
        # Pr[A or B] = Pr[A] + Pr[B & ~A]
        assert allclose(e, d)
        # Pr[A and B]  = Pr[A] * Pr[B]
        assert allclose(g, b + f)
        # Pr[A or (B & ~A)] = Pr[A] + Pr[B & ~A]
        assert allclose(e, logsumexp([a, b+f]))

        # (A => B) => Pr[A or B] = Pr[B]
        # i.e.,k (X > 1) => (X > 0).
        assert allclose(func_prob((X > 0) | (X > 1)), func_prob(X > 0))

        # Positive probability event.
        # Pr[A] = 1 - Pr[~A]
        event = ((0 < X) < 0.5) | ((Y < 0) & (1 < X))
        assert allclose(
            func_prob(event),
            logdiffexp(0, func_prob(~event)))

        # Probability zero event.
        event = ((0 < X) < 0.5) & ((Y < 0) | (1 < X))
        assert isinf_neg(func_prob(event))
        assert allclose(func_prob(~event), 0)

    # Condition on (X > 0)
    dX = dist.condition(X > 0)
    assert isinstance(dX, ProductDistribution)
    assert dX.distributions[0].symbol == Identity('X')
    assert dX.distributions[0].conditioned
    assert dX.distributions[0].support == sympy.Interval.open(0, sympy.oo)
    assert dX.distributions[1].symbol == Identity('Y')
    assert not dX.distributions[1].conditioned
    assert dX.distributions[1].Fl == 0
    assert dX.distributions[1].Fu == 1

    # Condition on (Y < 0.5)
    dY = dist.condition(Y < 0.5)
    assert isinstance(dY, ProductDistribution)
    assert dY.distributions[0].symbol == Identity('X')
    assert not dY.distributions[0].conditioned
    assert dY.distributions[1].symbol == Identity('Y')
    assert dY.distributions[1].conditioned
    assert dY.distributions[1].support == sympy.Interval.Ropen(0, 0.5)

    # Condition on (X > 0) & (Y < 0.5)
    dXY_and = dist.condition((X > 0) & (Y < 0.5))
    assert isinstance(dXY_and, ProductDistribution)
    assert dXY_and.distributions[0].symbol == Identity('X')
    assert dXY_and.distributions[0].conditioned
    assert dX.distributions[0].support == sympy.Interval.open(0, sympy.oo)
    assert dXY_and.distributions[1].symbol == Identity('Y')
    assert dXY_and.distributions[1].conditioned
    assert dXY_and.distributions[1].support == sympy.Interval.Ropen(0, 0.5)

    # Condition on (X > 0) | (Y < 0.5)
    event = (X > 0) | (Y < 0.5)
    dXY_or = dist.condition((X > 0) | (Y < 0.5))
    assert isinstance(dXY_or, SumDistribution)
    assert all(isinstance(d, ProductDistribution) for d in dXY_or.distributions)
    assert allclose(dXY_or.logprob(X > 0),dXY_or.weights[0])
    samples = dXY_or.sample(100, rng)
    assert all(event.evaluate(sample) for sample in samples)

def test_product_condition_or_probabilithy_zero():
    # Condition on (exp(|3*X**2|) > 0) | (log(Y) < 0.5)
    X = Identity('X')
    Y = Identity('Y')
    dist = ProductDistribution([
        NumericalDistribution(X, scipy.stats.norm(loc=0, scale=1), Reals),
        NumericalDistribution(Y, scipy.stats.gamma(a=1), RealsPos),
    ])

    # Condition on event which has probability zero.
    event = (X > 2) & (X < 2)
    with pytest.raises(ValueError):
        dist.condition(event)
    assert dist.logprob(event) == -float('inf')

    # Condition on an event where one of the terms in the factored
    # distribution has probability zero, i.e.,
    # (X > 1) | [(log(Y) < 0.5) & (X > 2)]
    #  = (X > 1) | [(log(Y) < 0.5) & (X > 2) & ~(X > 1)]
    # The subclause (X > 2) & ~(X > 1) has probability zero, hence
    # the result will be only a product distribution where X
    # is constrained and Y is unconstrained
    dist_condition = dist.condition((X>1) | ((Log(Y) < 0.5) & (X > 2)))
    assert isinstance(dist_condition, ProductDistribution)
    assert dist_condition.distributions[0].symbol == X
    assert dist_condition.distributions[0].conditioned
    assert dist_condition.distributions[0].support == sympy.Interval.open(1, sympy.oo)
    assert dist_condition.distributions[1].symbol == Y
    assert not dist_condition.distributions[1].conditioned

    # Another case as above, where this time the subclause
    # (X < 2) & ~(1 < exp(|3X**2|) is empty.
    # Thus Y remains unconditioned
    # and X is partitioned into (-oo, 0) U (0, oo) with equal weight.
    event = (Exp(abs(3*X**2)) > 1) | ((Log(Y) < 0.5) & (X < 2))
    dist_condition = dist.condition(event)
    assert isinstance(dist_condition, ProductDistribution)
    assert isinstance(dist_condition.distributions[0], SumDistribution)
    assert dist_condition.distributions[0].weights == [-log(2), -log(2)]
    assert dist_condition.distributions[0].distributions[0].conditioned
    assert dist_condition.distributions[0].distributions[1].conditioned
    assert dist_condition.distributions[0].distributions[0].support \
        == sympy.Interval.Ropen(-sympy.oo, 0)
    assert dist_condition.distributions[0].distributions[1].support \
        == sympy.Interval.Lopen(0, sympy.oo)
    assert dist_condition.distributions[1].symbol == Y
    assert not dist_condition.distributions[1].conditioned
