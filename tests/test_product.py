# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from math import log

import pytest

import numpy

from sppl.distributions import choice
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
from sppl.spe import LeafSPE
from sppl.spe import ProductSPE
from sppl.spe import SumSPE
from sppl.transforms import Exp
from sppl.transforms import Id
from sppl.transforms import Log

def test_product_distribution_normal_gamma_basic():
    X1 = Id('X1')
    X2 = Id('X2')
    X3 = Id('X3')
    X4 = Id('X4')
    children = [
        ProductSPE([
            X1 >> norm(loc=0, scale=1),
            X4 >> norm(loc=10, scale=1),
        ]),
        X2 >> gamma(loc=0, a=1),
        X3 >> norm(loc=2, scale=3)
    ]
    spe = ProductSPE(children)
    assert spe.children == (
        children[0].children[0],
        children[0].children[1],
        children[1],
        children[2],)
    assert spe.get_symbols() == frozenset([X1, X2, X3, X4])
    assert spe.size() == 5

    samples = spe.sample(2)
    assert len(samples) == 2
    for sample in samples:
        assert len(sample) == 4
        assert all([X in sample for X in (X1, X2, X3, X4)])

    samples = spe.sample_subset((X1, X2), 10)
    assert len(samples) == 10
    for sample in samples:
        assert len(sample) == 2
        assert X1 in sample
        assert X2 in sample

    samples = spe.sample_func(lambda X1, X2, X3: (X1, (X2**2, X3)), 1)
    assert len(samples) == 1
    assert len(samples[0]) == 2
    assert len(samples[0][1]) == 2

    with pytest.raises(ValueError):
        spe.sample_func(lambda X1, X5: X1 + X4, 1)

def test_product_inclusion_exclusion_basic():
    X = Id('X')
    Y = Id('Y')
    spe = ProductSPE([X >> norm(loc=0, scale=1), Y >> gamma(a=1)])

    a = spe.logprob(X > 0.1)
    b = spe.logprob(Y < 0.5)
    c = spe.logprob((X > 0.1) & (Y < 0.5))
    d = spe.logprob((X > 0.1) | (Y < 0.5))
    e = spe.logprob((X > 0.1) | ((Y < 0.5) & ~(X > 0.1)))
    f = spe.logprob(~(X > 0.1))
    g = spe.logprob((Y < 0.5) & ~(X > 0.1))

    assert allclose(a, spe.children[0].logprob(X > 0.1))
    assert allclose(b, spe.children[1].logprob(Y < 0.5))

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
    assert allclose(spe.logprob((X > 0) | (X > 1)), spe.logprob(X > 0))

    # Positive probability event.
    # Pr[A] = 1 - Pr[~A]
    event = ((0 < X) < 0.5) | ((Y < 0) & (1 < X))
    assert allclose(
        spe.logprob(event),
        logdiffexp(0, spe.logprob(~event)))

    # Probability zero event.
    event = ((0 < X) < 0.5) & ((Y < 0) | (1 < X))
    assert isinf_neg(spe.logprob(event))
    assert allclose(spe.logprob(~event), 0)

def test_product_condition_basic():
    X = Id('X')
    Y = Id('Y')
    spe = ProductSPE([X >> norm(loc=0, scale=1), Y >> gamma(a=1)])

    # Condition on (X > 0) and ((X > 0) | (Y < 0))
    # where the second clause reduces to first as Y < 0
    # has probability zero.
    for event in [(X > 0), (X  > 0) | (Y < 0)]:
        dX = spe.condition(event)
        assert isinstance(dX, ProductSPE)
        assert dX.children[0].symbol == Id('X')
        assert dX.children[0].conditioned
        assert dX.children[0].support == Interval.open(0, oo)
        assert dX.children[1].symbol == Id('Y')
        assert not dX.children[1].conditioned
        assert dX.children[1].Fl == 0
        assert dX.children[1].Fu == 1

    # Condition on (Y < 0.5)
    dY = spe.condition(Y < 0.5)
    assert isinstance(dY, ProductSPE)
    assert dY.children[0].symbol == Id('X')
    assert not dY.children[0].conditioned
    assert dY.children[1].symbol == Id('Y')
    assert dY.children[1].conditioned
    assert dY.children[1].support == Interval.Ropen(0, 0.5)

    # Condition on (X > 0) & (Y < 0.5)
    dXY_and = spe.condition((X > 0) & (Y < 0.5))
    assert isinstance(dXY_and, ProductSPE)
    assert dXY_and.children[0].symbol == Id('X')
    assert dXY_and.children[0].conditioned
    assert dXY_and.children[0].support == Interval.open(0, oo)
    assert dXY_and.children[1].symbol == Id('Y')
    assert dXY_and.children[1].conditioned
    assert dXY_and.children[1].support == Interval.Ropen(0, 0.5)

    # Condition on (X > 0) | (Y < 0.5)
    event = (X > 0) | (Y < 0.5)
    dXY_or = spe.condition((X > 0) | (Y < 0.5))
    assert isinstance(dXY_or, SumSPE)
    assert all(isinstance(d, ProductSPE) for d in dXY_or.children)
    assert allclose(dXY_or.logprob(X > 0),dXY_or.weights[0])
    samples = dXY_or.sample(100, prng=numpy.random.RandomState(1))
    assert all(event.evaluate(sample) for sample in samples)

    # Condition on a disjoint union with one term in second clause.
    dXY_disjoint_one = spe.condition((X > 0) & (Y < 0.5) | (X <= 0))
    assert isinstance(dXY_disjoint_one, SumSPE)
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
    dXY_disjoint_two = spe.condition((X > 0) & (Y < 0.5) | ((X <= 0) & ~(Y < 3)))
    assert isinstance(dXY_disjoint_two, SumSPE)
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
    spe.condition((X > 0) & (Y < 0.5) | ((X <= 1) | ~(Y < 3)))
    spe.condition((X > 0) & (Y < 0.5) | ((X <= 1) & (Y < 3)))

def test_product_condition_or_probabilithy_zero():
    X = Id('X')
    Y = Id('Y')
    spe = ProductSPE([X >> norm(loc=0, scale=1), Y >> gamma(a=1)])

    # Condition on event which has probability zero.
    event = (X > 2) & (X < 2)
    with pytest.raises(ValueError):
        spe.condition(event)
    assert spe.logprob(event) == -float('inf')

    # Condition on event which has probability zero.
    event = (Y < 0) | (Y < -1)
    with pytest.raises(ValueError):
        spe.condition(event)
    assert spe.logprob(event) == -float('inf')
    # Condition on an event where one clause has probability
    # zero, yielding a single product.
    spe_condition = spe.condition((Y < 0) | ((Log(X) >= 0) & (1 <= Y)))
    assert isinstance(spe_condition, ProductSPE)
    assert spe_condition.children[0].symbol == X
    assert spe_condition.children[0].conditioned
    assert spe_condition.children[0].support == Interval(1, oo)
    assert spe_condition.children[1].symbol == Y
    assert spe_condition.children[1].conditioned
    assert spe_condition.children[0].support == Interval(1, oo)

    # We have (X < 2) & ~(1 < exp(|3X**2|) is empty.
    # Thus Y remains unconditioned,
    #   and X is partitioned into (-oo, 0) U (0, oo) with equal weight.
    event = (Exp(abs(3*X**2)) > 1) | ((Log(Y) < 0.5) & (X < 2))
    spe_condition = spe.condition(event)
    #
    # The most concise representation of spe_condition is:
    #   (Product (Sum [.5 .5] X|X<0 X|X>0) Y)
    assert isinstance(spe_condition, ProductSPE)
    assert isinstance(spe_condition.children[0], SumSPE)
    assert spe_condition.children[0].weights == (-log(2), -log(2))
    assert spe_condition.children[0].children[0].conditioned
    assert spe_condition.children[0].children[1].conditioned
    assert spe_condition.children[0].children[0].support \
        in [Interval.Ropen(-oo, 0), Interval.Lopen(0, oo)]
    assert spe_condition.children[0].children[1].support \
        in [Interval.Ropen(-oo, 0), Interval.Lopen(0, oo)]
    assert spe_condition.children[0].children[0].support \
        != spe_condition.children[0].children[1].support
    assert spe_condition.children[1].symbol == Y
    assert not spe_condition.children[1].conditioned
    #
    # However, now we solve the constraint and factor the query into
    # a disjoint union at the root of the SPE, not once per node, we
    # no longer have guarantees of generating the smallest network.
    # In this case, the answer is the more verbose:
    #   (Sum [.5 .5] (Product X|X<0 Y) (Product X|X>0 Y)
    # assert isinstance(spe_condition, SumSPE)
    # assert spe_condition.weights == (-log(2), -log(2))
    # assert isinstance(spe_condition.children[0], ProductSPE)
    # assert X \
    #     == spe_condition.children[0].children[0].symbol \
    #     == spe_condition.children[1].children[0].symbol
    # assert spe_condition.children[0].children[0].conditioned
    # assert spe_condition.children[1].children[0].conditioned
    # assert spe_condition.children[0].children[0].support \
    #     == Interval.Ropen(-oo, 0)
    # assert spe_condition.children[1].children[0].support \
    #     == Interval.Lopen(0, oo)
    # assert Y \
    #     == spe_condition.children[0].children[1].symbol \
    #     == spe_condition.children[1].children[1].symbol
    # assert not spe_condition.children[0].children[1].conditioned
    # assert not spe_condition.children[1].children[1].conditioned

def test_product_disjoint_union_numerical():
    X = Id('X')
    Y = Id('Y')
    Z = Id('Z')
    spe = ProductSPE([
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
        logps = [spe.logprob(s) for s in clauses.subexprs]
        assert allclose(logsumexp(logps), spe.logprob(event))

def test_product_disjoint_union_nominal():
    N = Id('N')
    P = Id('P')

    nationality = N >> choice({'India': 0.5, 'USA': 0.5})
    perfect = P >> choice({'Imperfect': 0.99, 'Perfect': 0.01})
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
spe_abcd \
    = norm(loc=0, scale=1)(A) \
    & norm(loc=0, scale=1)(B) \
    & norm(loc=0, scale=1)(C) \
    & norm(loc=0, scale=1)(D)
def test_product_condition_simplify_a():
    spe = spe_abcd.condition((A > 1) | (A < -1))
    assert isinstance(spe, ProductSPE)
    assert spe_abcd.children[1] in spe.children
    assert spe_abcd.children[2] in spe.children
    assert spe_abcd.children[3] in spe.children
    idx_sum = [i for i, c in enumerate(spe.children) if isinstance(c, SumSPE)]
    assert len(idx_sum) == 1
    assert allclose(spe.children[idx_sum[0]].weights[0], -log(2))
    assert allclose(spe.children[idx_sum[0]].weights[1], -log(2))
    assert spe.children[idx_sum[0]].children[0].conditioned
    assert spe.children[idx_sum[0]].children[1].conditioned

def test_product_condition_simplify_ab():
    spe = spe_abcd.condition((A > 1) | (B < 0))
    assert isinstance(spe, ProductSPE)
    assert spe_abcd.children[2] in spe.children
    assert spe_abcd.children[2] in spe.children
    idx_sum = [i for i, c in enumerate(spe.children) if isinstance(c, SumSPE)]
    assert len(idx_sum) == 1
    spe_sum = spe.children[idx_sum[0]]
    assert isinstance(spe_sum.children[0], ProductSPE)
    assert isinstance(spe_sum.children[1], ProductSPE)
    lp0 = spe_abcd.logprob(A > 1)
    lp1 = spe_abcd.logprob((B < 0) & ~(A > 1))
    weights = lognorm([lp0, lp1])
    assert allclose(spe_sum.weights[0], weights[0])
    assert allclose(spe_sum.weights[1], weights[1])

def test_product_condition_simplify_abc():
    spe = spe_abcd.condition((A > 1) | (B < 0) | (C > 2))
    assert isinstance(spe, ProductSPE)
    assert len(spe.children) == 2
    assert spe_abcd.children[3] in spe.children
    idx_sum = [i for i, c in enumerate(spe.children) if isinstance(c, SumSPE)]
    assert len(idx_sum) == 1
    spe_sum = spe.children[idx_sum[0]]
    assert len(spe_sum.children) == 2
    lp0 = spe_abcd.logprob(A > 1)
    lp1 = spe_abcd.logprob((B < 0) & ~(A > 1))
    lp2 = spe_abcd.logprob((C > 2) & ~(B < 0) & ~(A > 1))
    weights = lognorm([lp0, lp1, lp2])
    assert allclose(spe_sum.weights[1], weights[-1])
    assert isinstance(spe_sum.children[0], ProductSPE)
    assert isinstance(spe_sum.children[0].children[0], SumSPE)
    assert isinstance(spe_sum.children[0].children[1], LeafSPE)
    assert isinstance(spe_sum.children[1], ProductSPE)
    assert isinstance(spe_sum.children[1].children[0], LeafSPE)
    assert isinstance(spe_sum.children[1].children[1], LeafSPE)
    assert isinstance(spe_sum.children[1].children[2], LeafSPE)
