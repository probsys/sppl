# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from math import log

import pytest

import numpy
import sympy

from spn.spn import ProductSPN
from spn.spn import SumSPN
from spn.math_util import allclose
from spn.math_util import isinf_neg
from spn.math_util import logdiffexp
from spn.math_util import logsumexp
from spn.numerical import Gamma
from spn.numerical import Norm
from spn.transforms import ExpNat as Exp
from spn.transforms import Identity
from spn.transforms import LogNat as Log

rng = numpy.random.RandomState(1)

def test_product_distribution_normal_gamma():
    X1 = Identity('X1')
    X2 = Identity('X2')
    X3 = Identity('X3')
    X4 = Identity('X4')
    children = [
        ProductSPN([
            Norm(X1, loc=0, scale=1),
            Norm(X4, loc=10, scale=1),
        ]),
        Gamma(X2, loc=0, a=1),
        Norm(X3, loc=2, scale=3)
    ]
    spn = ProductSPN(children)
    assert spn.children == (
        children[0].children[0],
        children[0].children[1],
        children[1],
        children[2],)
    assert spn.get_symbols() == frozenset([X1, X2, X3, X4])

    samples = spn.sample(2, rng)
    assert len(samples) == 2
    for sample in samples:
        assert len(sample) == 4
        assert all([X in sample for X in (X1, X2, X3, X4)])

    samples = spn.sample_subset((X1, X2), 10, rng)
    assert len(samples) == 10
    for sample in samples:
        assert len(sample) == 2
        assert X1 in sample
        assert X2 in sample

    samples = spn.sample_func(lambda X1, X2, X3: (X1, (X2**2, X3)), 1, rng)
    assert len(samples) == 1
    assert len(samples[0]) == 2
    assert len(samples[0][1]) == 2

    with pytest.raises(ValueError):
        spn.sample_func(lambda X1, X5: X1 + X4, 1, rng)

def test_inclusion_exclusion_basic():
    X = Identity('X')
    Y = Identity('Y')
    spn = ProductSPN([
        Norm(X, loc=0, scale=1),
        Gamma(Y, a=1)
    ])

    for func_prob in [spn.logprob, spn.logprob_inclusion_exclusion]:
        a = func_prob(X > 0.1)
        b = func_prob(Y < 0.5)
        c = func_prob((X > 0.1) & (Y < 0.5))
        d = func_prob((X > 0.1) | (Y < 0.5))
        e = func_prob((X > 0.1) | ((Y < 0.5) & ~(X > 0.1)))
        f = func_prob(~(X > 0.1))
        g = func_prob((Y < 0.5) & ~(X > 0.1))

        assert allclose(a, spn.children[0].logprob(X > 0.1))
        assert allclose(b, spn.children[1].logprob(Y < 0.5))

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
    dX = spn.condition(X > 0)
    assert isinstance(dX, ProductSPN)
    assert dX.children[0].symbol == Identity('X')
    assert dX.children[0].conditioned
    assert dX.children[0].support == sympy.Interval.open(0, sympy.oo)
    assert dX.children[1].symbol == Identity('Y')
    assert not dX.children[1].conditioned
    assert dX.children[1].Fl == 0
    assert dX.children[1].Fu == 1

    # Condition on (Y < 0.5)
    dY = spn.condition(Y < 0.5)
    assert isinstance(dY, ProductSPN)
    assert dY.children[0].symbol == Identity('X')
    assert not dY.children[0].conditioned
    assert dY.children[1].symbol == Identity('Y')
    assert dY.children[1].conditioned
    assert dY.children[1].support == sympy.Interval.Ropen(0, 0.5)

    # Condition on (X > 0) & (Y < 0.5)
    dXY_and = spn.condition((X > 0) & (Y < 0.5))
    assert isinstance(dXY_and, ProductSPN)
    assert dXY_and.children[0].symbol == Identity('X')
    assert dXY_and.children[0].conditioned
    assert dX.children[0].support == sympy.Interval.open(0, sympy.oo)
    assert dXY_and.children[1].symbol == Identity('Y')
    assert dXY_and.children[1].conditioned
    assert dXY_and.children[1].support == sympy.Interval.Ropen(0, 0.5)

    # Condition on (X > 0) | (Y < 0.5)
    event = (X > 0) | (Y < 0.5)
    dXY_or = spn.condition((X > 0) | (Y < 0.5))
    assert isinstance(dXY_or, SumSPN)
    assert all(isinstance(d, ProductSPN) for d in dXY_or.children)
    assert allclose(dXY_or.logprob(X > 0),dXY_or.weights[0])
    samples = dXY_or.sample(100, rng)
    assert all(event.evaluate(sample) for sample in samples)

def test_product_condition_or_probabilithy_zero():
    # Condition on (exp(|3*X**2|) > 0) | (log(Y) < 0.5)
    X = Identity('X')
    Y = Identity('Y')
    spn = ProductSPN([
        Norm(X, loc=0, scale=1),
        Gamma(Y, a=1),
    ])

    # Condition on event which has probability zero.
    event = (X > 2) & (X < 2)
    with pytest.raises(ValueError):
        spn.condition(event)
    assert spn.logprob(event) == -float('inf')

    # Condition on an event where one of the terms in the factored
    # distribution has probability zero, i.e.,
    # (X > 1) | [(log(Y) < 0.5) & (X > 2)]
    #  = (X > 1) | [(log(Y) < 0.5) & (X > 2) & ~(X > 1)]
    # The subclause (X > 2) & ~(X > 1) has probability zero, hence
    # the result will be only a product distribution where X
    # is constrained and Y is unconstrained
    spn_condition = spn.condition((X>1) | ((Log(Y) < 0.5) & (X > 2)))
    assert isinstance(spn_condition, ProductSPN)
    assert spn_condition.children[0].symbol == X
    assert spn_condition.children[0].conditioned
    assert spn_condition.children[0].support == sympy.Interval.open(1, sympy.oo)
    assert spn_condition.children[1].symbol == Y
    assert not spn_condition.children[1].conditioned

    # Another case as above, where this time the subclause
    # (X < 2) & ~(1 < exp(|3X**2|) is empty.
    # Thus Y remains unconditioned
    # and X is partitioned into (-oo, 0) U (0, oo) with equal weight.
    event = (Exp(abs(3*X**2)) > 1) | ((Log(Y) < 0.5) & (X < 2))
    spn_condition = spn.condition(event)
    assert isinstance(spn_condition, ProductSPN)
    assert isinstance(spn_condition.children[0], SumSPN)
    assert spn_condition.children[0].weights == (-log(2), -log(2))
    assert spn_condition.children[0].children[0].conditioned
    assert spn_condition.children[0].children[1].conditioned
    assert spn_condition.children[0].children[0].support \
        == sympy.Interval.Ropen(-sympy.oo, 0)
    assert spn_condition.children[0].children[1].support \
        == sympy.Interval.Lopen(0, sympy.oo)
    assert spn_condition.children[1].symbol == Y
    assert not spn_condition.children[1].conditioned

@pytest.mark.xfail(
    reason='https://github.com/probcomp/sum-product-dsl/issues/12',
    strict=True)
def test_disjoint_union_xfail():
    X = Identity('X')
    Y = Identity('Y')
    spn = ProductSPN([Norm(X, loc=0, scale=1), Norm(Y, loc=0, scale=2)])

    event = ((X > 0) & (Y < 1)) | ((X < 1) & (Y < 3))
    lp_ie = spn.logprob_inclusion_exclusion(event)
    lp_du = spn.logprob_disjoint_union(event)

    assert allclose(lp_ie, lp_du)
