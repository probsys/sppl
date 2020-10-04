# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from math import log

from sppl.distributions import gamma
from sppl.distributions import norm
from sppl.math_util import allclose
from sppl.spn import ProductSPN
from sppl.spn import SumSPN
from sppl.spn import spn_simplify_sum
from sppl.transforms import Id

def test_sum_simplify_nested_sum_1():
    X = Id('X')
    children = [
        SumSPN(
            [X >> norm(loc=0, scale=1), X >> norm(loc=0, scale=2)],
            [log(0.4), log(0.6)]),
        X >> gamma(loc=0, a=1),
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
    X = Id('X')
    W = Id('W')
    children = [
        SumSPN([
            (X >> norm(loc=0, scale=1)) & (W >> norm(loc=0, scale=2)),
            (X >> norm(loc=0, scale=2)) & (W >> norm(loc=0, scale=1))],
            [log(0.9), log(0.1)]),
        (X >> norm(loc=0, scale=4)) & (W >> norm(loc=0, scale=10)),
        SumSPN([
            (X >> norm(loc=0, scale=1)) & (W >> norm(loc=0, scale=2)),
            (X >> norm(loc=0, scale=2)) & (W >> norm(loc=0, scale=1)),
            (X >> norm(loc=0, scale=8)) & (W >> norm(loc=0, scale=3)),],
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
    Xd0 = Id('X') >> norm(loc=0, scale=1)
    Xd1 = Id('X') >> norm(loc=0, scale=2)
    Xd2 = Id('X') >> norm(loc=0, scale=3)
    spn = SumSPN([Xd0, Xd1, Xd2], [log(0.5), log(0.1), log(.4)])
    assert spn_simplify_sum(spn) == spn

    Xd0 = Id('X') >> norm(loc=0, scale=1)
    Xd1 = Id('X') >> norm(loc=0, scale=1)
    Xd2 = Id('X') >> norm(loc=0, scale=1)
    spn = SumSPN([Xd0, Xd1, Xd2], [log(0.5), log(0.1), log(.4)])
    assert spn_simplify_sum(spn) == Xd0

    Xd3 = Id('X') >> norm(loc=0, scale=2)
    spn = SumSPN([Xd0, Xd3, Xd1, Xd3], [log(0.5), log(0.1), log(.3), log(.1)])
    spn_simplified = spn_simplify_sum(spn)
    assert len(spn_simplified.children) == 2
    assert spn_simplified.children[0] == Xd0
    assert spn_simplified.children[1] == Xd3
    assert allclose(spn_simplified.weights[0], log(0.8))
    assert allclose(spn_simplified.weights[1], log(0.2))

def test_sum_simplify_product_collapse():
    A1 = Id('A') >> norm(loc=0, scale=1)
    A0 = Id('A') >> norm(loc=0, scale=1)
    B = Id('B') >> norm(loc=0, scale=1)
    B1 = Id('B') >> norm(loc=0, scale=1)
    B0 = Id('B') >> norm(loc=0, scale=1)
    C = Id('C') >> norm(loc=0, scale=1)
    C1 = Id('C') >> norm(loc=0, scale=1)
    D = Id('D') >> norm(loc=0, scale=1)
    spn = SumSPN([
        ProductSPN([A1, B, C, D]),
        ProductSPN([A0, B1, C, D]),
        ProductSPN([A0, B0, C1, D]),
    ], [log(0.4), log(0.4), log(0.2)])
    assert spn_simplify_sum(spn) == ProductSPN([A1, B, C, D])

def test_sum_simplify_product_complex():
    A1 = Id('A') >> norm(loc=0, scale=1)
    A0 = Id('A') >> norm(loc=0, scale=2)
    B = Id('B') >> norm(loc=0, scale=1)
    B1 = Id('B') >> norm(loc=0, scale=2)
    B0 = Id('B') >> norm(loc=0, scale=3)
    C = Id('C') >> norm(loc=0, scale=1)
    C1 = Id('C') >> norm(loc=0, scale=2)
    D = Id('D') >> norm(loc=0, scale=1)
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
