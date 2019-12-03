# Copyright 2019 MIT Probabilistic Computing Project.
# See LICENSE.txt

import pytest

from sympy import FiniteSet
from sympy import Interval
from sympy import Rational
from sympy import S as Singletons
from sympy import Union
from sympy import oo
from sympy import symbols

from sympy import exp as SymExp
from sympy import log as SymLog
from sympy import sqrt as SymSqrt

from sum_product_dsl.solver import solver

from sum_product_dsl.solver import Abs
from sum_product_dsl.solver import Exp
from sum_product_dsl.solver import Identity
from sum_product_dsl.solver import Log
from sum_product_dsl.solver import Poly
from sum_product_dsl.solver import Pow

from sum_product_dsl.solver import EventAnd
from sum_product_dsl.solver import EventBetween
from sum_product_dsl.solver import EventNot
from sum_product_dsl.solver import EventOr

(X0, X1, X2, X3, X4, X5, X6, X7, X8, X9) = symbols('X:10')

def test_solver_1():
    expr = SymLog(X0) > 2
    interval = solver(expr)
    assert interval.start == SymExp(2)
    assert interval.end == oo

    expr = Log(X0, SymExp(1))
    event = EventBetween(expr, 2, oo)
    interval = event.solve()
    assert interval.start == SymExp(2)
    assert interval.end == oo

def test_solver_2():
    expr = (SymLog(X0) > 2) & (X0 < SymExp(2))
    interval = solver(expr)
    assert interval == Singletons.EmptySet

    event = EventAnd([
        EventBetween(Log(X0, SymExp(1)), 2, oo),
        EventBetween(X0, -oo, SymExp(2))
    ])
    interval = event.solve()
    assert interval == FiniteSet(SymExp(2))

def test_solver_3():
    # Same as previous case using our system (only Between; measure zero).
    expr = (SymLog(X0) >= 2) & (X0 <= SymExp(2))
    interval = solver(expr)
    assert interval == FiniteSet(SymExp(2))

def test_solver_4():
    expr = (X0 >= 0) | (X0 <= 0)
    interval = solver(expr)
    assert interval == Singletons.Reals

    event = EventOr([
        EventBetween(X0, 0, oo),
        EventBetween(X0, -oo, 0)
    ])
    interval = event.solve()
    assert interval == Singletons.Reals

def test_solver_5():
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

def test_solver_6():
    expr = (X0**2 - 2*X0) > 10
    interval = solver(expr)
    assert interval == Union(
        Interval.open(-oo, 1 - SymSqrt(11)),
        Interval.open(1 + SymSqrt(11), oo))

    event = EventBetween(Poly(X0, [0, -2, 1]), 10, oo)
    interval = event.solve()
    assert interval == Union(
        Interval(-oo, 1 - SymSqrt(11)),
        Interval(1 + SymSqrt(11), oo))

def test_solver_7():
    expr = (X0**2 - 2*X0 + SymExp(X0)) > 10
    with pytest.raises(ValueError):
        solver(expr)

def test_solver_8():
    expr = (X0 + X1 < 3)
    with pytest.raises(ValueError):
        solver(expr)

def test_solver_9():
    expr = 2*(SymLog(X0))**3 - SymLog(X0) -5 > 0
    with pytest.raises(ValueError):
        solver(expr)

    # Our solver handles this case as follows
    # expr' = 2*Z**3 - Z - 5 > 0 [[subst. Z=log(X0)]]
    # [Z_low, Z_high] = solver(expr')
    #       Z_low < Z iff Z_low < log(X0) iff exp(Z_low) < X0
    #       Z < Z_high iff log(X0) < Z_high iff X0 < exp(Z_high)
    # solver(expr) = [exp(Z_low), exp(Z_high)]
    # For F invertible, can thus solve Poly(coeffs, F) > 0 using this method.
    expr = Poly(Log(X0, SymExp(1)), [-5, -1, 0, 2])
    event = EventBetween(expr, 0, oo)
    interval = event.solve()
    assert interval.left == \
        SymExp(1/(6*(SymSqrt(2019)/36 + 5/4)**(1/3)) \
        + (SymSqrt(2019)/36 + 5/4)**(1/3))
    assert interval.right == oo

def test_solver_10():
    # Sympy hangs for some reason.
    # expr = exp(sqrt(log(X0))) > -5
    expr = Exp(Pow(Log(X0, SymExp(1)), Rational(1, 2)), SymExp(1))
    event = EventBetween(expr, -5, oo)
    interval = event.solve()
    assert interval.left == 1
    assert interval.right == oo

def test_solver_11():
    # Sympy hangs for some reason.
    # expr = exp(sqrt(log(X0))) > 6
    expr = Exp(Pow(Log(X0, SymExp(1)), Rational(1, 2)), SymExp(1))
    event = EventBetween(expr, 6, oo)
    interval = event.solve()
    assert interval.left == SymExp(SymLog(6)**2)
    assert interval.right == oo
