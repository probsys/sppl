# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from math import log

import pytest

from sppl.spn import ContinuousLeaf
from sppl.spn import PartialSumSPN
from sppl.spn import ProductSPN
from sppl.spn import SumSPN

from sppl.distributions import gamma
from sppl.distributions import norm
from sppl.transforms import Id

from sppl.math_util import allclose

X = Id('X')
Y = Id('Y')
Z = Id('Z')

def test_mul_leaf():
    for y in [0.3 * (X >> norm()), (X >> norm()) * 0.3]:
        assert isinstance(y, PartialSumSPN)
        assert len(y.weights) == 1
        assert allclose(float(sum(y.weights)), 0.3)

def test_sum_leaf():
    # Cannot sum leaves without weights.
    with pytest.raises(TypeError):
        (X >> norm()) | (X >> gamma(a=1))
    # Cannot sum a leaf with a partial sum.
    with pytest.raises(TypeError):
        0.3*(X >> norm()) | (X >> gamma(a=1))
    # Cannot sum a leaf with a partial sum.
    with pytest.raises(TypeError):
        (X >> norm()) | 0.3*(X >> gamma(a=1))
    # Wrong symbol.
    with pytest.raises(ValueError):
        0.4*(X >> norm()) | 0.6*(Y >> gamma(a=1))
    # Sum exceeds one.
    with pytest.raises(ValueError):
        0.4*(X >> norm()) | 0.7*(Y >> gamma(a=1))

    y = 0.4*(X >> norm()) | 0.3*(X >> gamma(a=1))
    assert isinstance(y, PartialSumSPN)
    assert len(y.weights) == 2
    assert allclose(float(y.weights[0]), 0.4)
    assert allclose(float(y.weights[1]), 0.3)

    y = 0.4*(X >> norm()) | 0.6*(X >> gamma(a=1))
    assert isinstance(y, SumSPN)
    assert len(y.weights) == 2
    assert allclose(float(y.weights[0]), log(0.4))
    assert allclose(float(y.weights[1]), log(0.6))
    # Sum exceeds one.
    with pytest.raises(TypeError):
        y | 0.7 * (X >> norm())

    y = 0.4*(X >> norm()) | 0.3*(X >> gamma(a=1)) | 0.1*(X >> norm())
    assert isinstance(y, PartialSumSPN)
    assert len(y.weights) == 3
    assert allclose(float(y.weights[0]), 0.4)
    assert allclose(float(y.weights[1]), 0.3)
    assert allclose(float(y.weights[2]), 0.1)

    y = 0.4*(X >> norm()) | 0.3*(X >> gamma(a=1)) | 0.3*(X >> norm())
    assert isinstance(y, SumSPN)
    assert len(y.weights) == 3
    assert allclose(float(y.weights[0]), log(0.4))
    assert allclose(float(y.weights[1]), log(0.3))
    assert allclose(float(y.weights[2]), log(0.3))

    with pytest.raises(TypeError):
        (0.3)*(0.3*(X >> norm()))
    with pytest.raises(TypeError):
        (0.3*(X >> norm())) * (0.3)
    with pytest.raises(TypeError):
        0.3*(0.3*(X >> norm()) | 0.5*(X >> norm()))

    w = 0.3*(0.4*(X >> norm()) | 0.6*(X >> norm()))
    assert isinstance(w, PartialSumSPN)

def test_product_leaf():
    with pytest.raises(TypeError):
        0.3*(X >> gamma(a=1)) & (X >> norm())
    with pytest.raises(TypeError):
        (X >> norm()) & 0.3*(X >> gamma(a=1))
    with pytest.raises(ValueError):
        (X >> norm()) & (X >> gamma(a=1))

    y = (X >> norm()) & (Y >> gamma(a=1)) & (Z >> norm())
    assert isinstance(y, ProductSPN)
    assert len(y.children) == 3
    assert y.get_symbols() == frozenset([X, Y, Z])

def test_sum_of_sums():
    w \
        = 0.3*(0.4*(X >> norm()) | 0.6*(X >> norm())) \
        | 0.7*(0.1*(X >> norm()) | 0.9*(X >> norm()))
    assert isinstance(w, SumSPN)
    assert len(w.children) == 4
    assert allclose(float(w.weights[0]), log(0.3) + log(0.4))
    assert allclose(float(w.weights[1]), log(0.3) + log(0.6))
    assert allclose(float(w.weights[2]), log(0.7) + log(0.1))
    assert allclose(float(w.weights[3]), log(0.7) + log(0.9))

    w \
        = 0.3*(0.4*(X >> norm()) | 0.6*(X >> norm())) \
        | 0.2*(0.1*(X >> norm()) | 0.9*(X >> norm()))
    assert isinstance(w, PartialSumSPN)
    assert allclose(float(w.weights[0]), 0.3)
    assert allclose(float(w.weights[1]), 0.2)

    a = w | 0.5*(X >> gamma(a=1))
    assert isinstance(a, SumSPN)
    assert len(a.children) == 5
    assert allclose(float(a.weights[0]), log(0.3) + log(0.4))
    assert allclose(float(a.weights[1]), log(0.3) + log(0.6))
    assert allclose(float(a.weights[2]), log(0.2) + log(0.1))
    assert allclose(float(a.weights[3]), log(0.2) + log(0.9))
    assert allclose(float(a.weights[4]), log(0.5))

    # Wrong symbol.
    with pytest.raises(ValueError):
        z = w | 0.4*(Y >> gamma(a=1))

def test_or_and():
    with pytest.raises(ValueError):
        (0.3*(X >> norm()) | 0.7*(Y >> gamma(a=1))) & (Z >> norm())
    a = (0.3*(X >> norm()) | 0.7*(X >> gamma(a=1))) & (Z >> norm())
    assert isinstance(a, ProductSPN)
    assert isinstance(a.children[0], SumSPN)
    assert isinstance(a.children[1], ContinuousLeaf)
