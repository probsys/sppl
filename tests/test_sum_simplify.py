# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from math import log

import numpy

from spn.distributions import Gamma
from spn.distributions import Norm
from spn.math_util import allclose
from spn.spn import ProductSPN
from spn.spn import SumSPN
from spn.spn import spn_simplify_sum
from spn.transforms import Identity

rng = numpy.random.RandomState(1)

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
    Xd1 = Identity('X') >> Norm(loc=0, scale=2)
    Xd2 = Identity('X') >> Norm(loc=0, scale=3)
    spn = SumSPN([Xd0, Xd1, Xd2], [log(0.5), log(0.1), log(.4)])
    assert spn_simplify_sum(spn) == spn

    Xd0 = Identity('X') >> Norm(loc=0, scale=1)
    Xd1 = Identity('X') >> Norm(loc=0, scale=1)
    Xd2 = Identity('X') >> Norm(loc=0, scale=1)
    spn = SumSPN([Xd0, Xd1, Xd2], [log(0.5), log(0.1), log(.4)])
    assert spn_simplify_sum(spn) == Xd0

    Xd3 = Identity('X') >> Norm(loc=0, scale=2)
    spn = SumSPN([Xd0, Xd3, Xd1, Xd3], [log(0.5), log(0.1), log(.3), log(.1)])
    spn_simplified = spn_simplify_sum(spn)
    assert len(spn_simplified.children) == 2
    assert spn_simplified.children[0] == Xd0
    assert spn_simplified.children[1] == Xd3
    assert allclose(spn_simplified.weights[0], log(0.8))
    assert allclose(spn_simplified.weights[1], log(0.2))

def test_sum_simplify_product_collapse():
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
    assert spn_simplify_sum(spn) == ProductSPN([A1, B, C, D])

def test_sum_simplify_product_complex():
    A1 = Identity('A') >> Norm(loc=0, scale=1)
    A0 = Identity('A') >> Norm(loc=0, scale=2)
    B = Identity('B') >> Norm(loc=0, scale=1)
    B1 = Identity('B') >> Norm(loc=0, scale=2)
    B0 = Identity('B') >> Norm(loc=0, scale=3)
    C = Identity('C') >> Norm(loc=0, scale=1)
    C1 = Identity('C') >> Norm(loc=0, scale=2)
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
