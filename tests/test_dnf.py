# Copyright 2019 MIT Probabilistic Computing Project.
# See LICENSE.txt

from sympy import FiniteSet
from sympy import exp as SymExp
from sympy import log as SymLog
from sympy import sqrt as SymSqrt
from sympy import symbols
from sympy import to_dnf

from sum_product_dsl.dnf import factor_dnf
from sum_product_dsl.dnf import factor_dnf_symbols

from sum_product_dsl.contains import Contains
from sum_product_dsl.contains import NotContains

from sum_product_dsl.transforms import Identity

from sum_product_dsl.events import EventAnd

(X0, X1, X2, X3, X4, X5) = [symbols("X%d" % (i,)) for i in range(6)]
(Y0, Y1, Y2, Y3, Y4, Y5) = [Identity("X%d" % (i,)) for i in range(6)]

def test_to_dnf_no_change():
    event = Y0 < 0
    assert event.to_dnf() == event

    event = ((Y1<0) & (Y2<0))
    assert event.to_dnf() == event

    event = ((Y1<0) | (Y2<0))
    assert event.to_dnf() == event

    event = (Y0 < 0) | ((Y1<0) & (Y2<0))
    assert event.to_dnf() == event

def test_to_dnf_changes():
    event = (Y0 < 0)  &  ((Y1<0) | (Y2<0))
    result = event.to_dnf()
    assert len(result.events) == 2
    assert (Y0<0) & (Y1 < 0) in result.events
    assert (Y0<0) & (Y2 < 0) in result.events

    event = ((Y0 < 0)  &  ((Y1<0) | (Y2<0)))    |    (Y3 < 100)
    result = event.to_dnf()
    assert len(result.events) == 3
    assert (Y0<0) & (Y1 < 0) in result.events
    assert (Y0<0) & (Y2 < 0) in result.events
    assert (Y3<100) in result.events

    event = ((Y0 < 0)  &  ((Y1<0) | (Y2<0)))    |    ((Y3 < 100)  &  (Y5 < 10))
    result = event.to_dnf()
    assert len(result.events) == 3
    assert (Y0<0) & (Y1 < 0) in result.events
    assert (Y0<0) & (Y2 < 0) in result.events
    assert ((Y3 <100) & (Y5 < 10)) in result.events

    event = ((Y0 < 0)  |  ((Y1<0) | (Y2<0)))  & ((Y3 < 100)  &  (Y5 < 10))
    result = event.to_dnf()
    assert EventAnd([Y0 < 0, Y3 < 100, Y5 < 10]) in result.events
    assert EventAnd([Y1 < 0, Y3 < 100, Y5 < 10]) in result.events
    assert EventAnd([Y2 < 0, Y3 < 100, Y5 < 10]) in result.events

def test_factor_dnf():
    expr = (
        (SymExp(X0) > 0)
            & (X0 < 10)
            & (X1 < 10)
            & (X3 > 10)
            & Contains(X4, FiniteSet(10, 11))) \
        | (
            ((X2**2 - X2/2) < 0)
            & ((10*SymLog(X5) + 9) > 5)
            & ((SymSqrt(2*X3)) < 0)
            & NotContains(X4, FiniteSet(5, 6)))

    expr_dnf = to_dnf(expr)
    dnf = factor_dnf(expr_dnf)
    assert len(dnf) == 6

    assert dnf[X0] == ((SymExp(X0) > 0) & (X0 < 10))
    assert dnf[X1] == (X1 < 10)
    assert dnf[X2] == ((X2**2 - X2/2) < 0)
    assert dnf[X3] == (X3 > 10) | ((SymSqrt(2*X3)) < 0)
    assert dnf[X4] == Contains(X4, FiniteSet(10, 11)) \
        | NotContains(X4, FiniteSet(5, 6))
    assert dnf[X5] == ((10*SymLog(X5) + 9) > 5)

def test_factor_dnf_symbols_1():
    lookup = {X0:0, X1:0, X2:0, X3:1, X4:1, X5:2}

    expr = (
        (SymExp(X0) > 0)
            & (X0 < 10)
            & (X1 < 10)
            & NotContains(X2, FiniteSet(10, 11)))
    expr_dnf = to_dnf(expr)
    dnf = factor_dnf_symbols(expr_dnf, lookup)
    assert len(dnf) == 1
    assert dnf[0] == expr

    expr = (X0 < 1) & (X4 < 1) & (X5 < 1)
    expr_dnf = to_dnf(expr)
    dnf = factor_dnf_symbols(expr_dnf, lookup)
    assert len(dnf) == 3
    assert dnf[0] == (X0 < 1)
    assert dnf[1] == (X4 < 1)
    assert dnf[2] == (X5 < 1)

def test_factor_dnf_symbols_2():
    lookup = {X0:0, X1:0, X2:0, X3:1, X4:1, X5:2}

    expr = (
        (SymExp(X0) > 0)
            & (X0 < 10)
            & (X1 < 10)
            & Contains(X4, FiniteSet(10, 11))) \
        | (
            ((X2**2 - X2/2) < 0)
            & ((10*SymLog(X5) + 9) > 5)
            & NotContains(X4, FiniteSet(5, 6)))

    expr_dnf = to_dnf(expr)
    dnf = factor_dnf_symbols(expr_dnf, lookup)
    assert len(dnf) == 3

    assert dnf[0] == (
        (SymExp(X0) > 0)
            & (X0 < 10)
            & (X1 < 10)) \
        | ((X2**2 - X2/2) < 0)

    assert dnf[1] == (
        Contains(X4, FiniteSet(10, 11))
        | NotContains(X4, FiniteSet(5, 6)))

    assert dnf[2] == ((10*SymLog(X5) + 9) > 5)
