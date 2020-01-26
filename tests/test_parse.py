# Copyright 2019 MIT Probabilistic Computing Project.
# See LICENSE.txt

import pytest

import sympy

from sympy import Interval
from sympy import Rational
from sympy import oo

from sum_product_dsl.solver import Abs
from sum_product_dsl.solver import Identity
from sum_product_dsl.solver import Poly
from sum_product_dsl.solver import Pow
from sum_product_dsl.solver import Radical

from sum_product_dsl.solver import ExpNat
from sum_product_dsl.solver import LogNat
from sum_product_dsl.solver import Sqrt

from sum_product_dsl.solver import EventAnd
from sum_product_dsl.solver import EventInterval
from sum_product_dsl.solver import EventNot
from sum_product_dsl.solver import EventOr

X = Identity("X")
Y = X

def test_parse_1_open():
    # log(x) > 2
    expr = LogNat(X) > 2
    event = EventInterval(LogNat(Y), Interval(2, oo, left_open=True))
    assert expr == event

def test_parse_1_closed():
    # log(x) >= 2
    expr = LogNat(X) >= 2
    event = EventInterval(LogNat(Y), Interval(2, oo))
    assert expr == event

def test_parse_2_open():
    # log(x) < 2 & (x < exp(2))
    expr = (LogNat(X) > 2) & (X < sympy.exp(2))
    event = EventAnd([
        EventInterval(LogNat(Y), Interval.open(2, oo)),
        EventInterval(Y, Interval.open(-oo, sympy.exp(2)))
    ])
    assert expr == event

def test_parse_2_closed():
    # (log(x) <= 2) & (x >= exp(2))
    expr = (LogNat(X) >= 2) & (X <= sympy.exp(2))
    event = EventAnd([
        EventInterval(LogNat(Y), Interval(2, oo)),
        EventInterval(Y, Interval(-oo, sympy.exp(2)))
    ])
    assert expr == event

def test_parse_4():
    # (x >= 0) & (x <= 0)
    expr = (X >= 0) | (X <= 0)
    event = EventOr([
        EventInterval(Y, Interval(0, oo)),
        EventInterval(Y, Interval(-oo, 0))
    ])
    assert expr == event

def test_parse_5_open():
    # (2*x+10 < 4) & (x + 10 > 3)
    expr = ((2*X + 10) < 4) & (X + 10 > 3)
    event = EventAnd([
        EventInterval(Poly(Y, [10, 2]), Interval.open(-oo, 4)),
        EventInterval(Poly(Y, [10, 1]), Interval.open(3, oo)),
    ])
    assert expr == event

def test_parse_5_ropen():
    # (2*x+10 < 4) & (x + 10 >= 3)
    expr = ((2*X + 10) < 4) & (X + 10 >= 3)
    event = EventAnd([
        EventInterval(Poly(Y, [10, 2]), Interval.open(-oo, 4)),
        EventInterval(Poly(Y, [10, 1]), Interval(3, oo)),
    ])
    assert expr == event

def test_parse_5_lopen():
    # (2*x + 10 < 4) & (x + 10 >= 3)
    expr = ((2*X + 10) <= 4) & (X + 10 > 3)
    event = EventAnd([
        EventInterval(Poly(Y, [10, 2]), Interval(-oo, 4)),
        EventInterval(Poly(Y, [10, 1]), Interval.open(3, oo)),
    ])
    assert expr == event

def test_parse_6():
    # (x**2 - 2*x) > 10
    expr = (X**2 - 2*X) > 10
    event = EventInterval(Poly(Y, [0, -2, 1]), Interval.open(10, oo))
    assert expr == event

    # (exp(x)**2 - 2*exp(x)) > 10
    expr = (ExpNat(X)**2 - 2*ExpNat(X)) > 10
    event = EventInterval(Poly(ExpNat(X), [0, -2, 1]), Interval.open(10, oo))
    assert expr == event

def test_parse_7():
    # Illegal expression, cannot express in our custom DSL.
    with pytest.raises(NotImplementedError):
        (X**2 - 2*X + ExpNat(X)) > 10

def test_parse_8():
    Z = Identity('Z')
    with pytest.raises(NotImplementedError):
        (X + Z) < 3

def test_parse_9_open():
    # 2(log(x))**3 - log(x) -5 > 0
    expr = 2*(LogNat(X))**3 - LogNat(X) - 5
    expr_prime = Poly(LogNat(Y), [-5, -1, 0, 2])
    assert expr == expr_prime

    event = EventInterval(expr, Interval.open(0, oo))
    assert (expr > 0) == event

    # In principle can be solved by simplification.
    with pytest.raises(NotImplementedError):
        (2*LogNat(X))**3 - LogNat(X) - 5

def test_parse_10():
    # exp(sqrt(log(x))) > -5
    expr = ExpNat(Sqrt(LogNat(Y)))
    event = EventInterval(expr, Interval.open(-5, oo))
    assert (expr > -5) == event

def test_parse_11_open():
    # exp(sqrt(log(x))) > 6
    pass

def test_parse_12():
    # 2*sqrt(|x|) - 3 > 10
    expr = 2*Sqrt(Abs(X)) - 3
    expr_prime = Poly(Sqrt(Abs(Y)), [-3, 2])
    assert expr == expr_prime

    event = EventInterval(expr, Interval.open(10, oo))
    assert (expr > 10) == event

def test_parse_13():
    # 2*sqrt(|x|**2) - 3 > 10
    expr = 2*Sqrt(Abs(X)**2) - 3
    expr_prime = Poly(Sqrt(Pow(Abs(Y), 2)), [-3, 2])
    assert expr == expr_prime

def test_parse_14():
    # x**2 > 10
    expr = X**2 > 10
    event = EventInterval(Pow(Y, 2), Interval.open(10, oo))
    assert expr == event

def test_parse_15():
    # ((x**4)**(1/7)) < 9
    expr = ((X**4))**(Rational(1, 7))
    expr_prime = Radical(Pow(Y, 4), 7)
    assert expr == expr_prime

    event = EventInterval(expr_prime, Interval.open(-oo, 9))
    assert (expr < 9) == event

def test_parse_16():
    # (x**(1/7))**4 < 9
    expr = ((X**Rational(1,7)))**4
    expr_prime = Pow(Radical(Y, 7), 4)
    assert expr == expr_prime

    event = EventInterval(expr, Interval.open(-oo, 9))
    assert (expr < 9) == event

def test_parse_17():
    # https://www.wolframalpha.com/input/?i=Expand%5B%2810%2F7+%2B+X%29+%28-1%2F%285+Sqrt%5B2%5D%29+%2B+X%29+%28-Sqrt%5B5%5D+%2B+X%29%5D
    for Z in [X, LogNat(X), Abs(1+X**2)]:
        expr = (Z - Rational(1, 10) * sympy.sqrt(2)) \
            * (Z + Rational(10, 7)) \
            * (Z - sympy.sqrt(5))
        coeffs = [
            sympy.sqrt(10)/7,
            1/sympy.sqrt(10) - (10*sympy.sqrt(5))/7 - sympy.sqrt(2)/7,
            (-sympy.sqrt(5) - 1/(5 * sympy.sqrt(2))) + Rational(10)/7,
            1,
        ]
        expr_prime = Poly(Z, coeffs)
        assert expr == expr_prime

def test_parse_18():
    # 3*(x**(1/7))**4 - 3*(x**(1/7))**2 <= 9
    Z = X**(Rational(1, 7))
    expr = 3*Z**4 - 3*Z**2
    expr_prime = Poly(Radical(Y, 7), [0, 0, -3, 0, 3])
    assert expr == expr_prime

    event = EventInterval(expr_prime, Interval(-oo, 9))
    assert (expr <= 9 ) == event
    assert ~(expr <= 9) == EventNot(event)

    expr = (3*Abs(Z))**4 - (3*Abs(Z))**2
    expr_prime = Poly(Poly(Abs(Z), [0, 3]), [0, 0, -1, 0, 3])

def test_parse_19():
    # 3*(x**(1/7))**4 - 3*(x**(1/7))**2 <= 9
    #   or || 3*(x**(1/7))**4 - 3*(x**(1/7))**2 > 11
    Z = X**(Rational(1, 7))
    expr = 3*Z**4 - 3*Z**2

    event = (expr <= 9) | (expr > 11)
    event_prime = EventOr([
        EventInterval(expr, Interval(-oo, 9)),
        EventInterval(expr, Interval.open(11, oo)),
    ])
    assert event == event_prime

    event = ((expr <= 9) | (expr > 11)) & (~(expr < 10))
    event_prime = EventAnd([
        EventOr([
            EventInterval(expr, Interval(-oo, 9)),
            EventInterval(expr, Interval.open(11, oo))]),
        EventNot(
            EventInterval(expr, Interval.open(-oo, 10)))
    ])
    assert event == event_prime

def test_parse_20():
    # log(x**2 - 3) < 5
    expr = LogNat(X**2 - 3)
    expr_prime = LogNat(Poly(Y, [-3, 0, 1]))
    event = EventInterval(expr_prime, Interval.open(-oo, 5))
    assert (expr < 5) == event

def test_parse_21__ci_():
    # 1 <= log(x**3 - 3*x + 3) < 5
    # Can only be solved by numerical approximation of roots.
    # https://www.wolframalpha.com/input/?i=1+%3C%3D+log%28x**3+-+3x+%2B+3%29+%3C+5
    expr = LogNat(X**3 - 3*X + 3)
    expr_prime = LogNat(Poly(Y, [3, -3, 0, 1]))
    assert expr == expr_prime
    assert ((1 <= expr) & (expr < 5)) == EventAnd([
        EventInterval(expr, Interval(1, oo)),
        EventInterval(expr, Interval.open(-oo, 5)),
        ])

def test_errors():
    with pytest.raises(NotImplementedError):
        1 + LogNat(X) - ExpNat(X)
    with pytest.raises(NotImplementedError):
        LogNat(X) ** ExpNat(X)
    with pytest.raises(NotImplementedError):
        Abs(X) ** sympy.sqrt(10)
    with pytest.raises(NotImplementedError):
        LogNat(X) * X
    with pytest.raises(NotImplementedError):
        (2*LogNat(X)) - Rational(1, 10) * Abs(X)
