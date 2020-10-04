# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from math import log

import pytest

import numpy
import sympy

from sppl.distributions import gamma
from sppl.distributions import norm
from sppl.dnf import dnf_to_disjoint_union
from sppl.math_util import allclose
from sppl.math_util import isinf_neg
from sppl.math_util import logdiffexp
from sppl.math_util import lognorm
from sppl.math_util import logsumexp
from sppl.sets import Interval
from sppl.sets import inf as oo
from sppl.spn import LeafSPN
from sppl.spn import ProductSPN
from sppl.spn import SumSPN
from sppl.transforms import Exp
from sppl.transforms import Id
from sppl.transforms import Log

def test_product_distribution_normal_gamma_basic():
    X1 = Id('X1')
    X2 = Id('X2')
    X3 = Id('X3')
    X4 = Id('X4')
    children = [
        ProductSPN([
            X1 >> norm(loc=0, scale=1),
            X4 >> norm(loc=10, scale=1),
        ]),
        X2 >> gamma(loc=0, a=1),
        X3 >> norm(loc=2, scale=3)
    ]
    spn = ProductSPN(children)
    assert spn.children == (
        children[0].children[0],
        children[0].children[1],
        children[1],
        children[2],)
    assert spn.get_symbols() == frozenset([X1, X2, X3, X4])

    samples = spn.sample(2)
    assert len(samples) == 2
    for sample in samples:
        assert len(sample) == 4
        assert all([X in sample for X in (X1, X2, X3, X4)])

    samples = spn.sample_subset((X1, X2), 10)
    assert len(samples) == 10
    for sample in samples:
        assert len(sample) == 2
        assert X1 in sample
        assert X2 in sample

    samples = spn.sample_func(lambda X1, X2, X3: (X1, (X2**2, X3)), 1)
    assert len(samples) == 1
    assert len(samples[0]) == 2
    assert len(samples[0][1]) == 2

    with pytest.raises(ValueError):
        spn.sample_func(lambda X1, X5: X1 + X4, 1)

def test_product_inclusion_exclusion_basic():
    X = Id('X')
    Y = Id('Y')
    spn = ProductSPN([X >> norm(loc=0, scale=1), Y >> gamma(a=1)])

    a = spn.logprob(X > 0.1)
    b = spn.logprob(Y < 0.5)
    c = spn.logprob((X > 0.1) & (Y < 0.5))
    d = spn.logprob((X > 0.1) | (Y < 0.5))
    e = spn.logprob((X > 0.1) | ((Y < 0.5) & ~(X > 0.1)))
    f = spn.logprob(~(X > 0.1))
    g = spn.logprob((Y < 0.5) & ~(X > 0.1))

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
    assert allclose(spn.logprob((X > 0) | (X > 1)), spn.logprob(X > 0))

    # Positive probability event.
    # Pr[A] = 1 - Pr[~A]
    event = ((0 < X) < 0.5) | ((Y < 0) & (1 < X))
    assert allclose(
        spn.logprob(event),
        logdiffexp(0, spn.logprob(~event)))

    # Probability zero event.
    event = ((0 < X) < 0.5) & ((Y < 0) | (1 < X))
    assert isinf_neg(spn.logprob(event))
    assert allclose(spn.logprob(~event), 0)

def test_product_condition_basic():
    X = Id('X')
    Y = Id('Y')
    spn = ProductSPN([X >> norm(loc=0, scale=1), Y >> gamma(a=1)])

    # Condition on (X > 0) and ((X > 0) | (Y < 0))
    # where the second clause reduces to first as Y < 0
    # has probability zero.
    for event in [(X > 0), (X  > 0) | (Y < 0)]:
        dX = spn.condition(event)
        assert isinstance(dX, ProductSPN)
        assert dX.children[0].symbol == Id('X')
        assert dX.children[0].conditioned
        assert dX.children[0].support == Interval.open(0, oo)
        assert dX.children[1].symbol == Id('Y')
        assert not dX.children[1].conditioned
        assert dX.children[1].Fl == 0
        assert dX.children[1].Fu == 1

    # Condition on (Y < 0.5)
    dY = spn.condition(Y < 0.5)
    assert isinstance(dY, ProductSPN)
    assert dY.children[0].symbol == Id('X')
    assert not dY.children[0].conditioned
    assert dY.children[1].symbol == Id('Y')
    assert dY.children[1].conditioned
    assert dY.children[1].support == Interval.Ropen(0, 0.5)

    # Condition on (X > 0) & (Y < 0.5)
    dXY_and = spn.condition((X > 0) & (Y < 0.5))
    assert isinstance(dXY_and, ProductSPN)
    assert dXY_and.children[0].symbol == Id('X')
    assert dXY_and.children[0].conditioned
    assert dXY_and.children[0].support == Interval.open(0, oo)
    assert dXY_and.children[1].symbol == Id('Y')
    assert dXY_and.children[1].conditioned
    assert dXY_and.children[1].support == Interval.Ropen(0, 0.5)

    # Condition on (X > 0) | (Y < 0.5)
    event = (X > 0) | (Y < 0.5)
    dXY_or = spn.condition((X > 0) | (Y < 0.5))
    assert isinstance(dXY_or, SumSPN)
    assert all(isinstance(d, ProductSPN) for d in dXY_or.children)
    assert allclose(dXY_or.logprob(X > 0),dXY_or.weights[0])
    samples = dXY_or.sample(100, prng=numpy.random.RandomState(1))
    assert all(event.evaluate(sample) for sample in samples)

    # Condition on a disjoint union with one term in second clause.
    dXY_disjoint_one = spn.condition((X > 0) & (Y < 0.5) | (X <= 0))
    assert isinstance(dXY_disjoint_one, SumSPN)
    component_0 = dXY_disjoint_one.children[0]
    assert component_0.children[0].symbol == Id('X')
    assert component_0.children[0].conditioned
    assert component_0.children[0].support == Interval.open(0, oo)
    assert component_0.children[1].symbol == Id('Y')
    assert component_0.children[1].conditioned
    assert component_0.children[1].support == Interval.Ropen(0, 0.5)
    component_1 = dXY_disjoint_one.children[1]
    assert component_1.children[0].symbol == Id('X')
    assert component_1.children[0].conditioned
    assert component_1.children[0].support == Interval(-oo, 0)
    assert component_1.children[1].symbol == Id('Y')
    assert not component_1.children[1].conditioned

    # Condition on a disjoint union with two terms in each clause
    dXY_disjoint_two = spn.condition((X > 0) & (Y < 0.5) | ((X <= 0) & ~(Y < 3)))
    assert isinstance(dXY_disjoint_two, SumSPN)
    component_0 = dXY_disjoint_two.children[0]
    assert component_0.children[0].symbol == Id('X')
    assert component_0.children[0].conditioned
    assert component_0.children[0].support == Interval.open(0, oo)
    assert component_0.children[1].symbol == Id('Y')
    assert component_0.children[1].conditioned
    assert component_0.children[1].support == Interval.Ropen(0, 0.5)
    component_1 = dXY_disjoint_two.children[1]
    assert component_1.children[0].symbol == Id('X')
    assert component_1.children[0].conditioned
    assert component_1.children[0].support == Interval(-oo, 0)
    assert component_1.children[1].symbol == Id('Y')
    assert component_1.children[1].conditioned
    assert component_1.children[1].support == Interval(3, oo)

    # Some various conditioning.
    spn.condition((X > 0) & (Y < 0.5) | ((X <= 1) | ~(Y < 3)))
    spn.condition((X > 0) & (Y < 0.5) | ((X <= 1) & (Y < 3)))

def test_product_condition_or_probabilithy_zero():
    X = Id('X')
    Y = Id('Y')
    spn = ProductSPN([X >> norm(loc=0, scale=1), Y >> gamma(a=1)])

    # Condition on event which has probability zero.
    event = (X > 2) & (X < 2)
    with pytest.raises(ValueError):
        spn.condition(event)
    assert spn.logprob(event) == -float('inf')

    # Condition on event which has probability zero.
    event = (Y < 0) | (Y < -1)
    with pytest.raises(ValueError):
        spn.condition(event)
    assert spn.logprob(event) == -float('inf')
    # Condition on an event where one clause has probability
    # zero, yielding a single product.
    spn_condition = spn.condition((Y < 0) | ((Log(X) >= 0) & (1 <= Y)))
    assert isinstance(spn_condition, ProductSPN)
    assert spn_condition.children[0].symbol == X
    assert spn_condition.children[0].conditioned
    assert spn_condition.children[0].support == Interval(1, oo)
    assert spn_condition.children[1].symbol == Y
    assert spn_condition.children[1].conditioned
    assert spn_condition.children[0].support == Interval(1, oo)

    # We have (X < 2) & ~(1 < exp(|3X**2|) is empty.
    # Thus Y remains unconditioned,
    #   and X is partitioned into (-oo, 0) U (0, oo) with equal weight.
    event = (Exp(abs(3*X**2)) > 1) | ((Log(Y) < 0.5) & (X < 2))
    spn_condition = spn.condition(event)
    #
    # The most concise representation of spn_condition is:
    #   (Product (Sum [.5 .5] X|X<0 X|X>0) Y)
    assert isinstance(spn_condition, ProductSPN)
    assert isinstance(spn_condition.children[0], SumSPN)
    assert spn_condition.children[0].weights == (-log(2), -log(2))
    assert spn_condition.children[0].children[0].conditioned
    assert spn_condition.children[0].children[1].conditioned
    assert spn_condition.children[0].children[0].support \
        in [Interval.Ropen(-oo, 0), Interval.Lopen(0, oo)]
    assert spn_condition.children[0].children[1].support \
        in [Interval.Ropen(-oo, 0), Interval.Lopen(0, oo)]
    assert spn_condition.children[0].children[0].support \
        != spn_condition.children[0].children[1].support
    assert spn_condition.children[1].symbol == Y
    assert not spn_condition.children[1].conditioned
    #
    # However, now we solve the constraint and factor the query into
    # a disjoint union at the root of the SPN, not once per node, we
    # no longer have guarantees of generating the smallest network.
    # In this case, the answer is the more verbose:
    #   (Sum [.5 .5] (Product X|X<0 Y) (Product X|X>0 Y)
    # assert isinstance(spn_condition, SumSPN)
    # assert spn_condition.weights == (-log(2), -log(2))
    # assert isinstance(spn_condition.children[0], ProductSPN)
    # assert X \
    #     == spn_condition.children[0].children[0].symbol \
    #     == spn_condition.children[1].children[0].symbol
    # assert spn_condition.children[0].children[0].conditioned
    # assert spn_condition.children[1].children[0].conditioned
    # assert spn_condition.children[0].children[0].support \
    #     == Interval.Ropen(-oo, 0)
    # assert spn_condition.children[1].children[0].support \
    #     == Interval.Lopen(0, oo)
    # assert Y \
    #     == spn_condition.children[0].children[1].symbol \
    #     == spn_condition.children[1].children[1].symbol
    # assert not spn_condition.children[0].children[1].conditioned
    # assert not spn_condition.children[1].children[1].conditioned

def test_product_disjoint_union_numerical():
    X = Id('X')
    Y = Id('Y')
    Z = Id('Z')
    spn = ProductSPN([
        X >> norm(loc=0, scale=1),
        Y >> norm(loc=0, scale=2),
        Z >> norm(loc=0, scale=2),
    ])

    for event in [
        (1/X < 4) | (X > 7),
        (2*X-3 > 0) | (Log(Y) < 3),
        ((X > 0) & (Y < 1)) | ((X < 1) & (Y < 3)) | ~(X<<{1, 2}),
        ((X > 0) & (Y < 1)) | ((X < 1) & (Y < 3)) | (Z < 0),
        ((X > 0) & (Y < 1)) | ((X < 1) & (Y < 3)) | (Z < 0) | ~(X <<{1, 3}),
    ]:
        clauses = dnf_to_disjoint_union(event)
        logps = [spn.logprob(s) for s in clauses.subexprs]
        assert allclose(logsumexp(logps), spn.logprob(event))

def test_product_disjoint_union_nominal():
    N = Id('N')
    P = Id('P')

    nationality = N >> {'India': 0.5, 'USA': 0.5}
    perfect = P >> {'Imperfect': 0.99, 'Perfect': 0.01}
    student = nationality & perfect

    condition_1 = (N << {'India'}) & (P << {'Imperfect'})
    condition_2 = (N << {'India'}) & (P << {'Perfect'})
    condition_3 = (N << {'USA'}) & (P << {'Imperfect'})
    condition_4 = (N << {'USA'}) & (P << {'Perfect'})

    event_1 = condition_1
    event_2 = condition_2 & ~condition_1
    event_3 = condition_3 & ~condition_2 & ~condition_1
    event_4 = condition_4 & ~condition_3 & ~condition_2 & ~condition_1

    assert allclose(student.prob(event_1), 0.5*0.99)
    assert allclose(student.prob(event_2), 0.5*0.01)
    assert allclose(student.prob(event_3), 0.5*0.99)
    assert allclose(student.prob(event_4), 0.5*0.01)

A = Id('A')
B = Id('B')
C = Id('C')
D = Id('D')
spn_abcd \
    = norm(loc=0, scale=1)(A) \
    & norm(loc=0, scale=1)(B) \
    & norm(loc=0, scale=1)(C) \
    & norm(loc=0, scale=1)(D)
def test_product_condition_simplify_a():
    spn = spn_abcd.condition((A > 1) | (A < -1))
    assert isinstance(spn, ProductSPN)
    assert spn_abcd.children[1] in spn.children
    assert spn_abcd.children[2] in spn.children
    assert spn_abcd.children[3] in spn.children
    idx_sum = [i for i, c in enumerate(spn.children) if isinstance(c, SumSPN)]
    assert len(idx_sum) == 1
    assert allclose(spn.children[idx_sum[0]].weights[0], -log(2))
    assert allclose(spn.children[idx_sum[0]].weights[1], -log(2))
    assert spn.children[idx_sum[0]].children[0].conditioned
    assert spn.children[idx_sum[0]].children[1].conditioned

def test_product_condition_simplify_ab():
    spn = spn_abcd.condition((A > 1) | (B < 0))
    assert isinstance(spn, ProductSPN)
    assert spn_abcd.children[2] in spn.children
    assert spn_abcd.children[2] in spn.children
    idx_sum = [i for i, c in enumerate(spn.children) if isinstance(c, SumSPN)]
    assert len(idx_sum) == 1
    spn_sum = spn.children[idx_sum[0]]
    assert isinstance(spn_sum.children[0], ProductSPN)
    assert isinstance(spn_sum.children[1], ProductSPN)
    lp0 = spn_abcd.logprob(A > 1)
    lp1 = spn_abcd.logprob((B < 0) & ~(A > 1))
    weights = lognorm([lp0, lp1])
    assert allclose(spn_sum.weights[0], weights[0])
    assert allclose(spn_sum.weights[1], weights[1])

def test_product_condition_simplify_abc():
    spn = spn_abcd.condition((A > 1) | (B < 0) | (C > 2))
    assert isinstance(spn, ProductSPN)
    assert len(spn.children) == 2
    assert spn_abcd.children[3] in spn.children
    idx_sum = [i for i, c in enumerate(spn.children) if isinstance(c, SumSPN)]
    assert len(idx_sum) == 1
    spn_sum = spn.children[idx_sum[0]]
    assert len(spn_sum.children) == 2
    lp0 = spn_abcd.logprob(A > 1)
    lp1 = spn_abcd.logprob((B < 0) & ~(A > 1))
    lp2 = spn_abcd.logprob((C > 2) & ~(B < 0) & ~(A > 1))
    weights = lognorm([lp0, lp1, lp2])
    assert allclose(spn_sum.weights[1], weights[-1])
    assert isinstance(spn_sum.children[0], ProductSPN)
    assert isinstance(spn_sum.children[0].children[0], SumSPN)
    assert isinstance(spn_sum.children[0].children[1], LeafSPN)
    assert isinstance(spn_sum.children[1], ProductSPN)
    assert isinstance(spn_sum.children[1].children[0], LeafSPN)
    assert isinstance(spn_sum.children[1].children[1], LeafSPN)
    assert isinstance(spn_sum.children[1].children[2], LeafSPN)
