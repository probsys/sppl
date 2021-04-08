# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from math import log

from sppl.distributions import gamma
from sppl.distributions import norm
from sppl.math_util import allclose
from sppl.spe import ProductSPE
from sppl.spe import SumSPE
from sppl.spe import spe_simplify_sum
from sppl.transforms import Id

def test_sum_simplify_nested_sum_1():
    X = Id('X')
    children = [
        SumSPE(
            [X >> norm(loc=0, scale=1), X >> norm(loc=0, scale=2)],
            [log(0.4), log(0.6)]),
        X >> gamma(loc=0, a=1),
    ]
    spe = SumSPE(children, [log(0.7), log(0.3)])
    assert spe.size() == 4
    assert spe.children == (
        children[0].children[0],
        children[0].children[1],
        children[1]
    )
    assert allclose(spe.weights[0], log(0.7) + log(0.4))
    assert allclose(spe.weights[1], log(0.7) + log(0.6))
    assert allclose(spe.weights[2], log(0.3))

def test_sum_simplify_nested_sum_2():
    X = Id('X')
    W = Id('W')
    children = [
        SumSPE([
            (X >> norm(loc=0, scale=1)) & (W >> norm(loc=0, scale=2)),
            (X >> norm(loc=0, scale=2)) & (W >> norm(loc=0, scale=1))],
            [log(0.9), log(0.1)]),
        (X >> norm(loc=0, scale=4)) & (W >> norm(loc=0, scale=10)),
        SumSPE([
            (X >> norm(loc=0, scale=1)) & (W >> norm(loc=0, scale=2)),
            (X >> norm(loc=0, scale=2)) & (W >> norm(loc=0, scale=1)),
            (X >> norm(loc=0, scale=8)) & (W >> norm(loc=0, scale=3)),],
            [log(0.4), log(0.3), log(0.3)]),
    ]
    spe = SumSPE(children, [log(0.4), log(0.4), log(0.2)])
    assert spe.size() == 19
    assert spe.children == (
        children[0].children[0], # 2 leaves
        children[0].children[1], # 2 leaves
        children[1],             # 2 leaf
        children[2].children[0], # 2 leaves
        children[2].children[1], # 2 leaves
        children[2].children[2], # 2 leaves
    )
    assert allclose(spe.weights[0], log(0.4) + log(0.9))
    assert allclose(spe.weights[1], log(0.4) + log(0.1))
    assert allclose(spe.weights[2], log(0.4))
    assert allclose(spe.weights[3], log(0.2) + log(0.4))
    assert allclose(spe.weights[4], log(0.2) + log(0.3))
    assert allclose(spe.weights[5], log(0.2) + log(0.3))

def test_sum_simplify_leaf():
    Xd0 = Id('X') >> norm(loc=0, scale=1)
    Xd1 = Id('X') >> norm(loc=0, scale=2)
    Xd2 = Id('X') >> norm(loc=0, scale=3)
    spe = SumSPE([Xd0, Xd1, Xd2], [log(0.5), log(0.1), log(.4)])
    assert spe.size() == 4
    assert spe_simplify_sum(spe) == spe

    Xd0 = Id('X') >> norm(loc=0, scale=1)
    Xd1 = Id('X') >> norm(loc=0, scale=1)
    Xd2 = Id('X') >> norm(loc=0, scale=1)
    spe = SumSPE([Xd0, Xd1, Xd2], [log(0.5), log(0.1), log(.4)])
    assert spe_simplify_sum(spe) == Xd0

    Xd3 = Id('X') >> norm(loc=0, scale=2)
    spe = SumSPE([Xd0, Xd3, Xd1, Xd3], [log(0.5), log(0.1), log(.3), log(.1)])
    spe_simplified = spe_simplify_sum(spe)
    assert len(spe_simplified.children) == 2
    assert spe_simplified.children[0] == Xd0
    assert spe_simplified.children[1] == Xd3
    assert allclose(spe_simplified.weights[0], log(0.8))
    assert allclose(spe_simplified.weights[1], log(0.2))

def test_sum_simplify_product_collapse():
    A1 = Id('A') >> norm(loc=0, scale=1)
    A0 = Id('A') >> norm(loc=0, scale=1)
    B = Id('B') >> norm(loc=0, scale=1)
    B1 = Id('B') >> norm(loc=0, scale=1)
    B0 = Id('B') >> norm(loc=0, scale=1)
    C = Id('C') >> norm(loc=0, scale=1)
    C1 = Id('C') >> norm(loc=0, scale=1)
    D = Id('D') >> norm(loc=0, scale=1)
    spe = SumSPE([
        ProductSPE([A1, B, C, D]),
        ProductSPE([A0, B1, C, D]),
        ProductSPE([A0, B0, C1, D]),
    ], [log(0.4), log(0.4), log(0.2)])
    assert spe_simplify_sum(spe) == ProductSPE([A1, B, C, D])

def test_sum_simplify_product_complex():
    A1 = Id('A') >> norm(loc=0, scale=1)
    A0 = Id('A') >> norm(loc=0, scale=2)
    B = Id('B') >> norm(loc=0, scale=1)
    B1 = Id('B') >> norm(loc=0, scale=2)
    B0 = Id('B') >> norm(loc=0, scale=3)
    C = Id('C') >> norm(loc=0, scale=1)
    C1 = Id('C') >> norm(loc=0, scale=2)
    D = Id('D') >> norm(loc=0, scale=1)
    spe = SumSPE([
        ProductSPE([A1, B, C, D]),
        ProductSPE([A0, B1, C, D]),
        ProductSPE([A0, B0, C1, D]),
    ], [log(0.4), log(0.4), log(0.2)])

    spe_simplified = spe_simplify_sum(spe)
    assert isinstance(spe_simplified, ProductSPE)
    assert isinstance(spe_simplified.children[0], SumSPE)
    assert spe_simplified.children[1] == D

    ssc0 = spe_simplified.children[0]
    assert isinstance(ssc0.children[1], ProductSPE)
    assert ssc0.children[1].children == (A0, B0, C1)

    assert isinstance(ssc0.children[0], ProductSPE)
    assert ssc0.children[0].children[1] == C

    ssc0c0 = ssc0.children[0].children[0]
    assert isinstance(ssc0c0, SumSPE)
    assert isinstance(ssc0c0.children[0], ProductSPE)
    assert isinstance(ssc0c0.children[1], ProductSPE)
    assert ssc0c0.children[0].children == (A1, B)
    assert ssc0c0.children[1].children == (A0, B1)
