# Copyright 2019 MIT Probabilistic Computing Project.
# See LICENSE.txt

import pytest

from sympy import And
from sympy import Eq
from sympy import FiniteSet
from sympy import Interval
from sympy import Or
from sympy import S as Singletons
from sympy import Union
from sympy import exp
from sympy import log
from sympy import oo
from sympy import sqrt
from sympy import symbols
from sympy import to_dnf

from sum_product_dsl.contains import Contains
from sum_product_dsl.contains import NotContains
from sum_product_dsl.distributions import factor_dnf
from sum_product_dsl.distributions import factor_dnf_symbols
from sum_product_dsl.distributions import simplify_nominal_event

from sum_product_dsl.solver import get_symbols
from sum_product_dsl.solver import solver

from sum_product_dsl.solver import Identity
from sum_product_dsl.solver import Abs
from sum_product_dsl.solver import Pow
from sum_product_dsl.solver import Exp
from sum_product_dsl.solver import Log
from sum_product_dsl.solver import Poly

from sum_product_dsl.solver import Event
from sum_product_dsl.solver import EventBetween
from sum_product_dsl.solver import EventOr
from sum_product_dsl.solver import EventAnd
from sum_product_dsl.solver import EventNot

(X0, X1, X2, X3, X4, X5, X6, X7, X8, X9) = symbols('X:10')

def test_get_symbols():
    syms = get_symbols((X0 > 3) & (X1 < 4))
    assert len(syms) == 2
    assert X0 in syms
    assert X1 in syms

    syms = get_symbols((exp(X0) > log(X1)+10) & (X2 < 4))
    assert len(syms) == 3
    assert X0 in syms
    assert X1 in syms
    assert X2 in syms

def test_solver():
    # TEST CASE 1
    expr = log(X0) > 2
    interval = solver(expr)
    assert interval.start == exp(2)
    assert interval.end == oo

    expr = Log(X0, exp(1))
    event = EventBetween(expr, 2, oo)
    interval = event.solve()
    assert interval.start == exp(2)
    assert interval.end == oo

    # TEST CASE 2
    expr = (log(X0) > 2) & (X0 < exp(2))
    interval = solver(expr)
    assert interval == Singletons.EmptySet

    event = EventAnd([
        EventBetween(Log(X0, exp(1)), 2, oo),
        EventBetween(X0, -oo, exp(2))
    ])
    interval = event.solve()
    assert interval == FiniteSet(exp(2))

    # TEST CASE 3
    # Same as previous case using our system (only Between; measure zero).
    expr = (log(X0) >= 2) & (X0 <= exp(2))
    interval = solver(expr)
    assert interval == FiniteSet(exp(2))

    # TEST CASE 4
    expr = (X0 >= 0) | (X0 <= 0)
    interval = solver(expr)
    assert interval == Singletons.Reals

    event = EventOr([
        EventBetween(X0, 0, oo),
        EventBetween(X0, -oo, 0)
    ])
    interval = event.solve()
    assert interval == Singletons.Reals

    # TEST CASE 5
    expr = ((2*X0 + 10) < 4) & (X0 + 10 > 3)
    interval = solver(expr)
    assert interval.start == 3 - 10
    assert interval.end == (4-10)/2

    event = EventAnd([
        EventBetween(Poly(X0, [10, 2]), -oo, 4),
        EventBetween(Poly(X0, [10, 1]), 3, oo),
    ])
    interval = event.solve()
    assert interval.start == 3 - 10
    assert interval.end == (4-10)/2

    # TEST CASE 6
    expr = (X0**2 - 2*X0) > 10
    interval = solver(expr)
    assert interval == Union(
        Interval.open(-oo, 1 - sqrt(11)),
        Interval.open(1 + sqrt(11), oo))

    event = EventBetween(Poly(X0, [0, -2, 1]), 10, oo)
    interval = event.solve()
    assert interval == Union(
        Interval(-oo, 1 - sqrt(11)),
        Interval(1 + sqrt(11), oo))

    # TEST CASE 7
    expr = (X0**2 - 2*X0 + exp(X0)) > 10
    with pytest.raises(ValueError):
        solver(expr)

    # TEST CASE 8
    expr = (X0 + X1 < 3)
    with pytest.raises(ValueError):
        solver(expr)

    # TEST CASE 9
    expr = 2*(log(X0))**3 - log(X0) -5 > 0
    with pytest.raises(ValueError):
        solver(expr)

    # Our solver handles this case as follows
    # expr' = 2*Z**3 - Z - 5 > 0 [[subst. Z=log(X0)]]
    # [Z_low, Z_high] = solver(expr')
    #       Z_low < Z iff Z_low < log(X0) iff exp(Z_low) < X0
    #       Z < Z_high iff log(X0) < Z_high iff X0 < exp(Z_high)
    # solver(expr) = [exp(Z_low), exp(Z_high)]
    # For F invertible, can thus solve Poly(coeffs, F) > 0 using this method.
    expr = Poly(Log(X0, exp(1)), [-5, -1, 0, 2])
    event = EventBetween(expr, 0, oo)
    interval = event.solve()
    assert interval.left == \
        exp(1/(6*(sqrt(2019)/36 + 5/4)**(1/3)) \
        + (sqrt(2019)/36 + 5/4)**(1/3))
    assert interval.right == oo

    # TEST CASE 10
    # Sympy hangs for some reason.
    # expr = exp(sqrt(log(X0))) > -5
    from sympy import Rational
    expr = Exp(Pow(Log(X0, exp(1)), Rational(1, 2)), exp(1))
    event = EventBetween(expr, -5, oo)
    interval = event.solve()
    assert interval.left == 1
    assert interval.right == oo

    # TEST CASE 11
    # Sympy hangs for some reason.
    # expr = exp(sqrt(log(X0))) > 6
    expr = Exp(Pow(Log(X0, exp(1)), Rational(1, 2)), exp(1))
    event = EventBetween(expr, 6, oo)
    interval = event.solve()
    assert interval.left == exp(log(6)**2)
    assert interval.right == oo

def test_factor_dnf():
    expr = (
        (exp(X0) > 0)
            & (X0 < 10)
            & (X1 < 10)
            & (X3 > 10)
            & Contains(X4, FiniteSet(10, 11))) \
        | (
            ((X2**2 - X2/2) < 0)
            & ((10*log(X5) + 9) > 5)
            & ((sqrt(2*X3)) < 0)
            & NotContains(X4, FiniteSet(5, 6)))

    expr_dnf = to_dnf(expr)
    dnf = factor_dnf(expr_dnf)
    assert len(dnf) == 6

    assert dnf[X0] == ((exp(X0) > 0) & (X0 < 10))
    assert dnf[X1] == (X1 < 10)
    assert dnf[X2] == ((X2**2 - X2/2) < 0)
    assert dnf[X3] == (X3 > 10) | ((sqrt(2*X3)) < 0)
    assert dnf[X4] == Contains(X4, FiniteSet(10, 11)) \
        | NotContains(X4, FiniteSet(5, 6))
    assert dnf[X5] == ((10*log(X5) + 9) > 5)

def test_factor_dnf_symbols_1():
    lookup = {X0:0, X1:0, X2:0, X3:1, X4:1, X5:2}

    expr = (
        (exp(X0) > 0)
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
        (exp(X0) > 0)
            & (X0 < 10)
            & (X1 < 10)
            & Contains(X4, FiniteSet(10, 11))) \
        | (
            ((X2**2 - X2/2) < 0)
            & ((10*log(X5) + 9) > 5)
            & NotContains(X4, FiniteSet(5, 6)))

    expr_dnf = to_dnf(expr)
    dnf = factor_dnf_symbols(expr_dnf, lookup)
    assert len(dnf) == 3

    assert dnf[0] == (
        (exp(X0) > 0)
            & (X0 < 10)
            & (X1 < 10)) \
        | ((X2**2 - X2/2) < 0)

    assert dnf[1] == (
        Contains(X4, FiniteSet(10, 11))
        | NotContains(X4, FiniteSet(5, 6)))

    assert dnf[2] == ((10*log(X5) + 9) > 5)

def test_simplify_nominal_event():
    support = frozenset(range(100))
    snf = simplify_nominal_event

    # Eq.
    assert snf(Eq(X0, 5), support) == {5}
    assert snf(Eq(X0, -5), support) == set()

    V = (4, 1, 10)
    W = (12, -1, 8)
    E1 = FiniteSet(*V)
    E2 = FiniteSet(-5, *V)
    E3 = FiniteSet(-5)
    E4 = FiniteSet(*W)

    # Contains.
    assert snf(Contains(X0, E1), support) == set(V)
    assert snf(Contains(X0, E2), support) == set(V)
    assert snf(Contains(X0, E3), support) == set()

    # NotContains.
    assert snf(NotContains(X0, E1), support) == support.difference(V)
    assert snf(NotContains(X0, E2), support) == support.difference(V)
    assert snf(NotContains(X0, E3), support) == support

    # And
    assert snf(And(Contains(X0, E1), NotContains(X0, E4)), support) == \
        set(V).intersection(support.difference(W))

    # Or
    assert snf(Or(Contains(X0, E1), NotContains(X0, E4)), support) == \
        set(V).union(support.difference(W))
