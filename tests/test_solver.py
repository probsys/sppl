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

from sympy import Abs as SymAbs
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

from sum_product_dsl.solver import ExpNat
from sum_product_dsl.solver import LogNat
from sum_product_dsl.solver import Sqrt

from sum_product_dsl.solver import EventAnd
from sum_product_dsl.solver import EventInterval
from sum_product_dsl.solver import EventNot
from sum_product_dsl.solver import EventOr

(X0, X1, X2, X3, X4, X5, X6, X7, X8, X9) = symbols('X:10')

def test_solver_1_open():
    # log(x) > 2
    solution = Interval.open(SymExp(2), oo)

    expr = SymLog(X0) > 2
    interval = solver(expr)
    assert interval == solution

    event = EventInterval(LogNat(X0), Interval(2, oo, True))
    interval = event.solve()
    assert interval == solution

def test_solver_1_closed():
    # log(x) >= 2
    solution = Interval(SymExp(2), oo)

    expr = SymLog(X0) >= 2
    interval = solver(expr)
    assert interval == solution

    event = EventInterval(LogNat(X0), Interval(2, oo))
    interval = event.solve()
    assert interval == solution

def test_solver_2_open():
    # log(x) < 2 & (x < exp(2))
    solution = Singletons.EmptySet

    expr = (SymLog(X0) > 2) & (X0 < SymExp(2))
    interval = solver(expr)
    assert interval == solution

    event = EventAnd([
        EventInterval(LogNat(X0), Interval.open(2, oo)),
        EventInterval(X0, Interval.open(-oo, SymExp(2)))
    ])
    interval = event.solve()
    assert interval == solution

def test_solver_2_closed():
    # (log(x) <= 2) & (x >= exp(2))
    solution = FiniteSet(SymExp(2))

    expr = (SymLog(X0) >= 2) & (X0 <= SymExp(2))
    interval = solver(expr)
    assert interval == solution

    event = EventAnd([
        EventInterval(LogNat(X0), Interval(2, oo)),
        EventInterval(X0, Interval(-oo, SymExp(2)))
    ])
    interval = event.solve()
    assert interval == solution

def test_solver_4():
    # (x >= 0) & (x <= 0)
    solution = Singletons.Reals

    expr = (X0 >= 0) | (X0 <= 0)
    interval = solver(expr)
    assert interval == solution

    event = EventOr([
        EventInterval(X0, Interval(0, oo)),
        EventInterval(X0, Interval(-oo, 0))
    ])
    interval = event.solve()
    assert interval == solution

def test_solver_5_open():
    # (2*x+10 < 4) & (x + 10 > 3)
    solution = Interval.open(3-10, (4-10)/2)

    expr = ((2*X0 + 10) < 4) & (X0 + 10 > 3)
    interval = solver(expr)
    assert interval == solution

    event = EventAnd([
        EventInterval(Poly(X0, [10, 2]), Interval.open(-oo, 4)),
        EventInterval(Poly(X0, [10, 1]), Interval.open(3, oo)),
    ])
    interval = event.solve()
    assert interval == solution

def test_solver_5_ropen():
    # (2*x+10 < 4) & (x + 10 >= 3)
    solution = Interval.Ropen(3-10, (4-10)/2)

    expr = ((2*X0 + 10) < 4) & (X0 + 10 >= 3)
    interval = solver(expr)
    assert interval == solution

    event = EventAnd([
        EventInterval(Poly(X0, [10, 2]), Interval.open(-oo, 4)),
        EventInterval(Poly(X0, [10, 1]), Interval(3, oo)),
    ])
    interval = event.solve()
    assert interval == solution

def test_solver_5_lopen():
    # (2*x + 10 < 4) & (x + 10 >= 3)
    solution =Interval.Lopen(3-10, (4-10)/2)

    expr = ((2*X0 + 10) <= 4) & (X0 + 10 > 3)
    interval = solver(expr)
    assert interval == solution

    event = EventAnd([
        EventInterval(Poly(X0, [10, 2]), Interval(-oo, 4)),
        EventInterval(Poly(X0, [10, 1]), Interval.open(3, oo)),
    ])
    interval = event.solve()
    assert interval == solution

def test_solver_6():
    # (x**2 - 2*x) > 10
    solution =  Union(
        Interval.open(-oo, 1 - SymSqrt(11)),
        Interval.open(1 + SymSqrt(11), oo))

    expr = (X0**2 - 2*X0) > 10
    interval = solver(expr)
    assert interval == solution

    event = EventInterval(Poly(X0, [0, -2, 1]), Interval.open(10, oo))
    interval = event.solve()
    assert interval == solution

def test_solver_7():
    # Illegal expression, cannot express in our custom DSL.
    expr = (X0**2 - 2*X0 + SymExp(X0)) > 10
    with pytest.raises(ValueError):
        solver(expr)

def test_solver_8():
    expr = (X0 + X1 < 3)
    with pytest.raises(ValueError):
        solver(expr)

def test_solver_9_open():
    # 2(log(x))**3 - log(x) -5 > 0
    solution = Interval.open(
        SymExp(1/(6*(SymSqrt(2019)/36 + 5/4)**(1/3))
            + (SymSqrt(2019)/36 + 5/4)**(1/3)),
        oo)

    expr = 2*(SymLog(X0))**3 - SymLog(X0) -5 > 0
    with pytest.raises(ValueError):
        assert solver(expr) == solution

    # Our solver handles this case as follows
    # expr' = 2*Z**3 - Z - 5 > 0 [[subst. Z=log(X0)]]
    # [Z_low, Z_high] = solver(expr')
    #       Z_low < Z iff Z_low < log(X0) iff exp(Z_low) < X0
    #       Z < Z_high iff log(X0) < Z_high iff X0 < exp(Z_high)
    # solver(expr) = [exp(Z_low), exp(Z_high)]
    # For F invertible, can thus solve Poly(coeffs, F) > 0 using this method.
    expr = Poly(LogNat(X0), [-5, -1, 0, 2])
    event = EventInterval(expr, Interval.open(0, oo))
    interval = event.solve()
    assert interval == solution

def test_solver_9_closed():
    # 2(log(x))**3 - log(x) -5 >= 0
    solution = Interval(
        SymExp(1/(6*(SymSqrt(2019)/36 + 5/4)**(1/3))
            + (SymSqrt(2019)/36 + 5/4)**(1/3)),
        oo)

    expr = 2*(SymLog(X0))**3 - SymLog(X0) -5 > 0
    with pytest.raises(ValueError):
        assert solver(expr) == solution

    expr = Poly(LogNat(X0), [-5, -1, 0, 2])
    event = EventInterval(expr, Interval(0, oo))
    interval = event.solve()
    assert interval == solution

def test_solver_10():
    # exp(sqrt(log(x))) > -5
    solution = Interval(1, oo)

    # Sympy hangs for some reason; cannot test.
    # expr = exp(sqrt(log(X0))) > -5

    expr = ExpNat(Sqrt(LogNat(X0)))
    event = EventInterval(expr, Interval.open(-5, oo))
    interval = event.solve()
    assert interval == solution

def test_solver_11_open():
    # exp(sqrt(log(x))) > 6
    solution = Interval.open(SymExp(SymLog(6)**2), oo)

    # Sympy hangs for some reason.
    # expr = exp(sqrt(log(X0))) > 6

    expr = ExpNat(Sqrt(LogNat(X0)))
    event = EventInterval(expr, Interval.open(6, oo))
    interval = event.solve()
    assert interval == solution

def test_solver_11_closed():
    # exp(sqrt(log(x))) >= 6
    solution = Interval(SymExp(SymLog(6)**2), oo)

    # Sympy hangs for some reason.
    # expr = exp(sqrt(log(X0))) > 6

    expr = ExpNat(Sqrt(LogNat(X0)))
    event = EventInterval(expr, Interval(6, oo))
    interval = event.solve()
    assert interval == solution

def test_solver_12():
    # 2*sqrt(|x|) - 3 > 10
    solution = Union(
        Interval.open(-oo, -Rational(169, 4)),
        Interval.open(Rational(169, 4), oo))

    expr = 2*SymSqrt(SymAbs(X0)) - 3 > 10
    interval = solver(expr)
    assert interval == solution

    expr = Poly(Sqrt(Abs(X0)), [-3, 2])
    event = EventInterval(expr, Interval.open(10, oo))
    interval = event.solve()
    assert interval == solution

def test_solver_13():
    # 2*sqrt(|x|**2) - 3 > 10
    solution = Union(
        Interval.open(-oo, -Rational(13, 2)),
        Interval.open(Rational(13, 2), oo))

    expr = 2*SymSqrt(SymAbs(X0)**2) - 3 > 10
    interval = solver(expr)
    assert interval == solution

    expr = Poly(Sqrt(Pow(Abs(X0), 2)), [-3, 2])
    event = EventInterval(expr, Interval.open(10, oo))
    interval = event.solve()
    assert interval == solution

def test_solver_14():
    # log(x) > 0
    event = EventInterval(LogNat(X0), Interval(-oo, oo))
    interval = event.solve()
    assert interval == Interval.open(0, oo)

    # exp(x) < 0
    event = EventInterval(ExpNat(X0), Interval.open(-oo, 0))
    interval = event.solve()
    assert interval == Singletons.EmptySet

    # exp(x) <= 0
    event = EventInterval(ExpNat(X0), Interval(-oo, 0))
    interval = event.solve()
    assert interval == FiniteSet(-oo)
