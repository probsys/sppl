# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

import pytest

from sum_product_dsl.dnf import factor_dnf
from sum_product_dsl.dnf import factor_dnf_symbols

from sum_product_dsl.transforms import ExpNat
from sum_product_dsl.transforms import Identity
from sum_product_dsl.transforms import LogNat
from sum_product_dsl.transforms import Sqrt

(X0, X1, X2, X3, X4, X5) = [Identity("X%d" % (i,)) for i in range(6)]

events = [X0<0, (X1<<(0,1)) & (X2<0), (X1<0) | (X2<0), (X0<0) | ((X1<0) & (X2<0))]
@pytest.mark.parametrize('event', events)
def test_to_dnf_no_change(event):
    assert event.to_dnf() == event

def test_to_dnf_changes():
    A = X0 < 0
    B = X1 < 0
    C = X2 < 0
    D = X3 < 100
    E = X4 < 10

    event = A & (B | C)
    result = event.to_dnf()
    assert len(result.events) == 2
    assert A & B in result.events
    assert A & C in result.events

    event = (A & (B | C)) | D
    result = event.to_dnf()
    assert len(result.events) == 3
    assert A & B in result.events
    assert A & C in result.events
    assert D in result.events

    event = (A & (B | C)) | (D & E)
    result = event.to_dnf()
    assert len(result.events) == 3
    assert A & B in result.events
    assert A & C in result.events
    assert D & E in result.events

    event = (A | (B | C)) & (D & E)
    result = event.to_dnf()
    assert A & D & E in result.events
    assert B & D & E in result.events
    assert C & D & E in result.events

def test_factor_dnf():
    A = ExpNat(X0) > 0
    B = X0 < 10
    C = X1 < 10
    D = X3 > 10
    E = X4 > 0
    F = (X2**2 - X2*3) < 0
    G = 10*LogNat(X5) + 9 > 5
    H = (Sqrt(2*X3)) < 0
    I = X4 << [3, 1, 5]

    expr = (A & B & C & D & E) | (F & G & H & ~I)
    expr_dnf = expr.to_dnf()
    dnf = factor_dnf(expr_dnf)
    assert len(dnf) == 6

    assert dnf[X0] == (A & B)
    assert dnf[X1] == C
    assert dnf[X2] == F
    assert dnf[X3] == D | H
    assert dnf[X4] == E | ~I
    assert dnf[X5] == G

def test_factor_dnf_symbols_1():
    A = ExpNat(X0) > 0
    B = X0 < 10
    C = X1 < 10
    D = X2 < 0

    expr = A & B & C & ~D
    expr_dnf = expr.to_dnf()
    dnf = factor_dnf_symbols(expr_dnf, {X0:0, X1:0, X2:0, X3:1, X4:1, X5:2})

    assert len(dnf) == 1
    assert dnf[0] == expr

def test_factor_dnf_symbols_2():
    expr = (X0 < 1) & (X4 < 1) & (X5 < 1)
    expr_dnf = expr.to_dnf()
    dnf = factor_dnf_symbols(expr_dnf, {X0:0, X1:0, X2:0, X3:1, X4:1, X5:2})
    assert len(dnf) == 3
    assert dnf[0] == (X0 < 1)
    assert dnf[1] == (X4 < 1)
    assert dnf[2] == (X5 < 1)

def test_factor_dnf_symbols_3():
    A = (ExpNat(X0) > 0)
    B = X0 < 10
    C = X1 < 10
    D = X4 > 0
    E = (X2**2 - 3*X2) << (0, 10, 100)
    F = (10*LogNat(X5) + 9) > 5
    G = X4 < 4

    expr = (A & B & C & ~D) | (E & F & G)

    expr_dnf = expr.to_dnf()
    dnf = factor_dnf_symbols(expr_dnf, {X0:0, X1:0, X2:0, X3:1, X4:1, X5:2})
    assert len(dnf) == 3

    assert dnf[0] == (A & B & C) | E
    assert dnf[1] == ~D | G
    assert dnf[2] == F
