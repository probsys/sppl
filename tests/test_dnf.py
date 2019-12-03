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

(X0, X1, X2, X3, X4, X5, X6, X7, X8, X9) = symbols('X:10')

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
