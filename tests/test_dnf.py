# Copyright 2019 MIT Probabilistic Computing Project.
# See LICENSE.txt

import pytest

from sum_product_dsl.dnf import factor_dnf
from sum_product_dsl.dnf import factor_dnf_symbols

from sum_product_dsl.transforms import ExpNat
from sum_product_dsl.transforms import Identity
from sum_product_dsl.transforms import LogNat
from sum_product_dsl.transforms import Sqrt

(X0, X1, X2, X3, X4, X5) = [Identity("X%d" % (i,)) for i in range(6)]

events = [X0<0, (X1<0) & (X2<0), (X1<0) | (X2<0), (X0<0) | ((X1<0) & (X2<0))]
@pytest.mark.parametrize('event', events)
def test_to_dnf_no_change(event):
    assert event.to_dnf() == event

def test_to_dnf_changes():
    event = (X0 < 0)  &  ((X1<0) | (X2<0))
    result = event.to_dnf()
    assert len(result.events) == 2
    assert (X0<0) & (X1 < 0) in result.events
    assert (X0<0) & (X2 < 0) in result.events

    event = ((X0 < 0)  &  ((X1<0) | (X2<0)))    |    (X3 < 100)
    result = event.to_dnf()
    assert len(result.events) == 3
    assert (X0<0) & (X1 < 0) in result.events
    assert (X0<0) & (X2 < 0) in result.events
    assert (X3<100) in result.events

    event = ((X0 < 0)  &  ((X1<0) | (X2<0)))    |    ((X3 < 100)  &  (X5 < 10))
    result = event.to_dnf()
    assert len(result.events) == 3
    assert (X0<0) & (X1 < 0) in result.events
    assert (X0<0) & (X2 < 0) in result.events
    assert ((X3 <100) & (X5 < 10)) in result.events

    event = ((X0 < 0)  |  ((X1<0) | (X2<0)))  &  ((X3 < 100)  &  (X5 < 10))
    result = event.to_dnf()
    assert (X0 < 0) & (X3 < 100) & (X5 < 10) in result.events
    assert (X1 < 0) & (X3 < 100) & (X5 < 10) in result.events
    assert (X2 < 0) & (X3 < 100) & (X5 < 10) in result.events

def test_factor_dnf():
    expr = \
        ((ExpNat(X0) > 0)
            & (X0 < 10)
            & (X1 < 10)
            & (X3 > 10)
            & (X4 > 0)) \
        | (((X2**2 - X2*3) < 0)
            & ((10*LogNat(X5) + 9) > 5)
            & ((Sqrt(2*X3)) < 0)
            & ~(X4 < 5))

    expr_dnf = expr.to_dnf()
    dnf = factor_dnf(expr_dnf)
    assert len(dnf) == 6

    assert dnf[X0] == ((ExpNat(X0) > 0) & (X0 < 10))
    assert dnf[X1] == (X1 < 10)
    assert dnf[X2] == ((X2**2 - X2 * 3) < 0)
    assert dnf[X3] == (X3 > 10) | ((Sqrt(2*X3)) < 0)
    assert dnf[X4] == (X4 > 0)  | ~(X4 < 5)
    assert dnf[X5] == ((10*LogNat(X5) + 9) > 5)

def test_factor_dnf_symbols_1():
    lookup = {X0:0, X1:0, X2:0, X3:1, X4:1, X5:2}

    expr = (ExpNat(X0) > 0) & (X0 < 10) & (X1 < 10) & ~(X2 < 0)
    expr_dnf = expr.to_dnf()
    dnf = factor_dnf_symbols(expr_dnf, lookup)

    assert len(dnf) == 1
    assert dnf[0] == expr

    expr = (X0 < 1) & (X4 < 1) & (X5 < 1)
    expr_dnf = expr.to_dnf()
    dnf = factor_dnf_symbols(expr_dnf, lookup)
    assert len(dnf) == 3
    assert dnf[0] == (X0 < 1)
    assert dnf[1] == (X4 < 1)
    assert dnf[2] == (X5 < 1)

def test_factor_dnf_symbols_2():
    lookup = {X0:0, X1:0, X2:0, X3:1, X4:1, X5:2}

    expr = \
        ((ExpNat(X0) > 0)
        & (X0 < 10)
        & (X1 < 10)
        & ~(X4 > 0)) \
        | (((X2**2 - 3*X2) < 0)
            & ((10*LogNat(X5) + 9) > 5)
            & (X4 < 4))

    expr_dnf = expr.to_dnf()
    dnf = factor_dnf_symbols(expr_dnf, lookup)
    assert len(dnf) == 3

    assert dnf[0] == (
        (ExpNat(X0) > 0)
        & (X0 < 10)
        & (X1 < 10)) \
        | ((X2**2 - 3*X2) < 0)

    assert dnf[1] == ~(X4 > 0) | (X4 < 4)
    assert dnf[2] == ((10*LogNat(X5) + 9) > 5)
