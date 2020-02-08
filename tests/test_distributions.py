# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from fractions import Fraction
from math import log

import pytest

import numpy
import scipy.stats
import sympy

from sum_product_dsl.distributions import MixtureDistribution
from sum_product_dsl.distributions import NominalDistribution
from sum_product_dsl.distributions import NumericDistribution
from sum_product_dsl.distributions import ProductDistribution

from sum_product_dsl.transforms import Identity
from sum_product_dsl.transforms import ExpNat as Exp
from sum_product_dsl.transforms import LogNat as Log
from sum_product_dsl.math_util import allclose
from sum_product_dsl.math_util import isinf_neg
from sum_product_dsl.math_util import logsumexp
from sum_product_dsl.math_util import logdiffexp
from sum_product_dsl.sym_util import Reals
from sum_product_dsl.sym_util import RealsPos

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

def test_numeric_distribution_normal():
    X = Identity('X')
    probs = scipy.stats.norm(loc=0, scale=1)
    dist = NumericDistribution(X, probs, Reals)

    assert allclose(dist.logprob(X > 0), -log(2))
    assert allclose(dist.logprob(abs(X) < 2), log(probs.cdf(2) - probs.cdf(-2)))

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
    assert isinstance(dist_condition_b, MixtureDistribution)
    assert allclose(dist_condition_b.weights[0], -log(2))
    assert allclose(dist_condition_b.weights[0], dist_condition_b.weights[1])

    for event in [(X<-10), (X>3)]:
        dist_condition_c = dist.condition(event)
        assert isinstance(dist_condition_c, NumericDistribution)
        assert isinf_neg(dist_condition_c.logprob((-1 < X) < 1))
        samples = dist_condition_c.sample(100, rng)
        assert all(s[X] in event.values for s in samples)

    with pytest.raises(ValueError):
        dist.condition((X > 1) & (X < 1))

    with pytest.raises(ValueError):
        dist.condition(X << {1})

    with pytest.raises(ValueError):
        dist.sample_func(lambda Z: Z**2, 1, rng)

def test_mixture_distribution_normal_gamma():
    X = Identity('X')
    weights = [
        log(Fraction(2, 3)),
        log(Fraction(1, 3))
    ]
    dist = MixtureDistribution([
            NumericDistribution(X, scipy.stats.norm(loc=0, scale=1), Reals),
            NumericDistribution(X, scipy.stats.gamma(loc=0, a=1), RealsPos),
        ], weights)

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
    assert isinstance(dist_condition, NumericDistribution)
    assert dist_condition.conditioned
    assert dist_condition.logprob(X < 0) == 0
    samples = dist_condition.sample(100, rng)
    assert all(s[X] < 0 for s in samples)

    assert dist.logprob(X < 0) == logsumexp([
        dist.weights[0] + dist.distributions[0].logprob(X < 0),
        dist.weights[1] + dist.distributions[1].logprob(X < 0),
    ])

def test_product_distribution_normal_gamma():
    X1 = Identity('X1')
    X2 = Identity('X2')
    X3 = Identity('X3')
    X4 = Identity('X4')
    dists = [
        ProductDistribution([
            NumericDistribution(X1, scipy.stats.norm(loc=0, scale=1), Reals),
            NumericDistribution(X4, scipy.stats.norm(loc=10, scale=1), Reals)
        ]),
        NumericDistribution(X2, scipy.stats.gamma(loc=0, a=1), RealsPos),
        NumericDistribution(X3, scipy.stats.norm(loc=2, scale=3), Reals),
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
        NumericDistribution(X, scipy.stats.norm(loc=0, scale=1), Reals),
        NumericDistribution(Y, scipy.stats.gamma(a=1), RealsPos),
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
    assert isinstance(dXY_or, MixtureDistribution)
    assert all(isinstance(d, ProductDistribution) for d in dXY_or.distributions)
    assert allclose(dXY_or.logprob(X > 0),dXY_or.weights[0])
    samples = dXY_or.sample(100, rng)
    assert all(event.evaluate(sample) for sample in samples)

def test_product_condition_or_probabilithy_zero():
    # Condition on (exp(|3*X**2|) > 0) | (log(Y) < 0.5)
    X = Identity('X')
    Y = Identity('Y')
    dist = ProductDistribution([
        NumericDistribution(X, scipy.stats.norm(loc=0, scale=1), Reals),
        NumericDistribution(Y, scipy.stats.gamma(a=1), RealsPos),
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
    assert isinstance(dist_condition.distributions[0], MixtureDistribution)
    assert dist_condition.distributions[0].weights == [-log(2), -log(2)]
    assert dist_condition.distributions[0].distributions[0].conditioned
    assert dist_condition.distributions[0].distributions[1].conditioned
    assert dist_condition.distributions[0].distributions[0].support \
        == sympy.Interval.Ropen(-sympy.oo, 0)
    assert dist_condition.distributions[0].distributions[1].support \
        == sympy.Interval.Lopen(0, sympy.oo)
    assert dist_condition.distributions[1].symbol == Y
    assert not dist_condition.distributions[1].conditioned
