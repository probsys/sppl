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

def test_to_dnf_invert():
    A = X0 < 0
    B = X1 < 0
    C = X2 < 0
    D = X3 < 0

    expr = ~(A | B | C)
    assert expr.to_dnf() == ~A & ~B & ~C

    expr = ~(A | B | ~C)
    assert expr.to_dnf() == ~A & ~B & C

    expr = ~((A | B | C) & D)
    #  =>  ~(A | B | C) | ~D
    assert expr.to_dnf() == (~A & ~B & ~C) | ~D

    expr = ~((A | ~B | C) & ~D)
    #  =>  ~(A | B | C) | ~D
    assert expr.to_dnf() == (~A & B & ~C) | D

def test_factor_dnf():
    E00 = ExpNat(X0) > 0
    E01 = X0 < 10
    E10 = X1 < 10
    E20 = (X2**2 - X2*3) < 0
    E30 = X3 > 10
    E31 = (Sqrt(2*X3)) < 0
    E40 = X4 > 0
    E41 = X4 << [3, 1, 5]
    E50 = 10*LogNat(X5) + 9 > 5

    expr = (E00)
    expr_dnf = expr.to_dnf()
    dnf = factor_dnf(expr_dnf)
    assert len(dnf) == 1
    assert dnf[0][X0] == E00

    expr = E00 & E01
    expr_dnf = expr.to_dnf()
    dnf = factor_dnf(expr_dnf)
    assert len(dnf) == 1
    assert dnf[0][X0] == E00 & E01

    expr = E00 | E01
    expr_dnf = expr.to_dnf()
    dnf = factor_dnf(expr_dnf)
    assert len(dnf) == 2
    assert dnf[0][X0] == E00
    assert dnf[1][X0] == E01

    expr = E00 | (E01 & E10)
    expr_dnf = expr.to_dnf()
    dnf = factor_dnf_symbols(expr_dnf, {X0: 0, X1: 0})
    assert len(dnf) == 2
    assert dnf[0][0] == E00
    assert dnf[1][0] == E01 & E10

    expr = (E00 & E01 & E10 & E30 & E40) | (E20 & E50 & E31 & ~E41)
    expr_dnf = expr.to_dnf()
    dnf = factor_dnf(expr_dnf)
    assert len(dnf) == 2
    assert dnf[0][X0] == E00 & E01
    assert dnf[0][X1] == E10
    assert dnf[0][X3] == E30
    assert dnf[0][X4] == E40
    assert len(dnf[0]) == 4
    assert dnf[1][X3] == E31
    assert dnf[1][X2] == E20
    assert dnf[1][X4] == ~E41
    assert dnf[1][X5] == E50
    assert len(dnf[1]) == 4

def test_factor_dnf_symbols_1():
    A = ExpNat(X0) > 0
    B = X0 < 10
    C = X1 < 10
    D = X2 < 0

    expr = A & B & C & ~D
    expr_dnf = expr.to_dnf()
    dnf = factor_dnf_symbols(expr_dnf, {X0:0, X1:0, X2:0, X3:1, X4:1, X5:2})
    assert len(dnf) == 1
    assert dnf[0][0] == expr

def test_factor_dnf_symbols_2():
    A = X0 < 1
    B = X4 < 1
    C = X5 < 1
    expr = A & B & C
    expr_dnf = expr.to_dnf()
    dnf = factor_dnf_symbols(expr_dnf, {X0:0, X1:0, X2:0, X3:1, X4:1, X5:2})
    assert len(dnf) == 1
    assert dnf[0][0] == A
    assert dnf[0][1] == B
    assert dnf[0][2] == C

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
    assert len(dnf) == 2
    assert dnf[0][0] == A & B & C
    assert dnf[0][1] == ~D
    assert dnf[1][0] == E
    assert dnf[1][1] == G
    assert dnf[1][2] == F
