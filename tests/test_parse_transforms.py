# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

import pytest
import sympy

from sympy import Rational as Rat

from sppl.sets import EmptySet
from sppl.sets import FiniteNominal
from sppl.sets import FiniteReal
from sppl.sets import Interval
from sppl.sets import inf as oo

from sppl.transforms import Abs
from sppl.transforms import Exp
from sppl.transforms import Identity
from sppl.transforms import Log
from sppl.transforms import Piecewise
from sppl.transforms import Poly
from sppl.transforms import Pow
from sppl.transforms import Radical
from sppl.transforms import Reciprocal
from sppl.transforms import Sqrt

from sppl.transforms import EventAnd
from sppl.transforms import EventFiniteNominal
from sppl.transforms import EventFiniteReal
from sppl.transforms import EventInterval
from sppl.transforms import EventOr

X = Identity("X")
Y = X

# Avoid reporting tests which raise errors:
# pylint: disable=pointless-statement
# pylint: disable=expression-not-assigned

def test_parse_1_open():
    # log(x) > 2
    expr = Log(X) > 2
    event = EventInterval(Log(Y), Interval(2, oo, left_open=True))
    assert expr == event

def test_parse_1_closed():
    # log(x) >= 2
    expr = Log(X) >= 2
    event = EventInterval(Log(Y), Interval(2, oo))
    assert expr == event

def test_parse_2_open():
    # log(x) < 2 & (x < exp(2))
    expr = (Log(X) > 2) & (X < sympy.exp(2))
    event = EventAnd([
        EventInterval(Log(Y), Interval.open(2, oo)),
        EventInterval(Y, Interval.open(-oo, sympy.exp(2)))
    ])
    assert expr == event

def test_parse_2_closed():
    # (log(x) <= 2) & (x >= exp(2))
    expr = (Log(X) >= 2) & (X <= sympy.exp(2))
    event = EventAnd([
        EventInterval(Log(Y), Interval(2, oo)),
        EventInterval(Y, Interval(-oo, sympy.exp(2)))
    ])
    assert expr == event

def test_parse_4():
    # (x >= 0) & (x <= 0)
    expr = (X >= 0) | (X <= 0)
    event = EventInterval(X, Interval(-oo, oo))
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
    expr = (Exp(X)**2 - 2*Exp(X)) > 10
    event = EventInterval(Poly(Exp(X), [0, -2, 1]), Interval.open(10, oo))
    assert expr == event

def test_parse_7():
    # Illegal expression, cannot express in our custom DSL.
    with pytest.raises(ValueError):
        (X**2 - 2*X + Exp(X)) > 10

def test_parse_8():
    Z = Identity('Z')
    with pytest.raises(ValueError):
        (X + Z) < 3

def test_parse_9_open():
    # 2(log(x))**3 - log(x) -5 > 0
    expr = 2*(Log(X))**3 - Log(X) - 5
    expr_prime = Poly(Log(Y), [-5, -1, 0, 2])
    assert expr == expr_prime

    event = EventInterval(expr, Interval.open(0, oo))
    assert (expr > 0) == event

    # Cannot add polynomials with different subexpressions.
    with pytest.raises(ValueError):
        (2*Log(X))**3 - Log(X) - 5

def test_parse_10():
    # exp(sqrt(log(x))) > -5
    expr = Exp(Sqrt(Log(Y)))
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
    expr = ((X**4))**Rat(1, 7)
    expr_prime = Radical(Pow(Y, 4), 7)
    assert expr == expr_prime

    event = EventInterval(expr_prime, Interval.open(-oo, 9))
    assert (expr < 9) == event

def test_parse_16():
    # (x**(1/7))**4 < 9
    for expr in [((X**Rat(1,7)))**4, (X**(1,7))**4]:
        expr_prime = Pow(Radical(Y, 7), 4)
        assert expr == expr_prime

    event = EventInterval(expr, Interval.open(-oo, 9))
    assert (expr < 9) == event

def test_parse_17():
    # https://www.wolframalpha.com/input/?i=Expand%5B%2810%2F7+%2B+X%29+%28-1%2F%285+Sqrt%5B2%5D%29+%2B+X%29+%28-Sqrt%5B5%5D+%2B+X%29%5D
    for Z in [X, Log(X), Abs(1+X**2)]:
        expr = (Z - Rat(1, 10) * sympy.sqrt(2)) \
            * (Z + Rat(10, 7)) \
            * (Z - sympy.sqrt(5))
        coeffs = [
            sympy.sqrt(10)/7,
            1/sympy.sqrt(10) - (10*sympy.sqrt(5))/7 - sympy.sqrt(2)/7,
            (-sympy.sqrt(5) - 1/(5 * sympy.sqrt(2))) + Rat(10)/7,
            1,
        ]
        expr_prime = Poly(Z, coeffs)
        assert expr == expr_prime

def test_parse_18():
    # 3*(x**(1/7))**4 - 3*(x**(1/7))**2 <= 9
    Z = X**Rat(1, 7)
    expr = 3*Z**4 - 3*Z**2
    expr_prime = Poly(Radical(Y, 7), [0, 0, -3, 0, 3])
    assert expr == expr_prime

    event = EventInterval(expr_prime, Interval(-oo, 9))
    assert (expr <= 9) == event

    event_not = EventInterval(expr_prime, Interval.open(9, oo))
    assert ~(expr <= 9) == event_not

    expr = (3*Abs(Z))**4 - (3*Abs(Z))**2
    expr_prime = Poly(Poly(Abs(Z), [0, 3]), [0, 0, -1, 0, 3])

def test_parse_19():
    # 3*(x**(1/7))**4 - 3*(x**(1/7))**2 <= 9
    #   or || 3*(x**(1/7))**4 - 3*(x**(1/7))**2 > 11
    Z = X**Rat(1, 7)
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
        EventInterval(expr, Interval(10, oo))
    ])
    assert event == event_prime

def test_parse_20():
    # log(x**2 - 3) < 5
    expr = Log(X**2 - 3)
    expr_prime = Log(Poly(Y, [-3, 0, 1]))
    event = EventInterval(expr_prime, Interval.open(-oo, 5))
    assert (expr < 5) == event

def test_parse_21__ci_():
    # 1 <= log(x**3 - 3*x + 3) < 5
    # Can only be solved by numerical approximation of roots.
    # https://www.wolframalpha.com/input/?i=1+%3C%3D+log%28x**3+-+3x+%2B+3%29+%3C+5
    expr = Log(X**3 - 3*X + 3)
    expr_prime = Log(Poly(Y, [3, -3, 0, 1]))
    assert expr == expr_prime
    assert ((1 <= expr) & (expr < 5)) \
        == EventInterval(expr, Interval.Ropen(1, 5))

def test_parse_24_negative_power():
    assert X**(-3) == Reciprocal(Pow(X, 3))
    assert X**((-3, 1)) == Reciprocal(Pow(X, 3))
    assert X**(-Rat(1, 3)) == Reciprocal(Radical(X, 3))
    with pytest.raises(ValueError):
        X**0

def test_parse_26_piecewise_one_expr_basic_event():
    assert (Y**2)*(0 <= Y) == Piecewise(
        [Poly(Y, [0, 0, 1])],
        [EventInterval(Y, Interval(0, oo))])
    assert (0 <= Y)*(Y**2) == Piecewise(
        [Poly(Y, [0, 0, 1])],
        [EventInterval(Y, Interval(0, oo))])
    assert ((0 <= Y) < 5)*(Y < 1) == Piecewise(
        [EventInterval(Y, Interval.Ropen(-oo, 1))],
        [EventInterval(Y, Interval.Ropen(0, 5))],
    )
    assert ((0 <= Y) < 5)*(~(Y < 1)) == Piecewise(
        [EventInterval(Y, Interval(1, oo))],
        [EventInterval(Y, Interval.Ropen(0, 5))],
    )
    assert 10*(0 <= Y) == Poly(
        EventInterval(Y, Interval(0, oo)),
        [0, 10])

def test_parse_26_piecewise_one_expr_compound_event():
    assert (Y**2)*((Y < 0) | (0 < Y)) == Piecewise(
        [Poly(Y, [0, 0, 1])],
        [EventOr([
            EventInterval(Y, Interval.open(-oo, 0)),
            EventInterval(Y, Interval.open(0, oo)),
            ])])

    assert (Y**2)*(~((3 < Y) <= 4)) == Piecewise(
        [Poly(Y, [0, 0, 1])],
        [EventOr([
            EventInterval(Y, Interval(-oo, 3)),
            EventInterval(Y, Interval.open(4, oo)),
            ])])

def test_parse_27_piecewise_many():
    assert (Y < 0)*(Y**2) + (0 <= Y)*Y**((1, 2)) == Piecewise(
        [
            Poly(Y, [0, 0, 1]),
            Radical(Y, 2)],
        [
            EventInterval(Y, Interval.open(-oo, 0)),
            EventInterval(Y, Interval(0, oo))
        ])

def test_errors():
    with pytest.raises(ValueError):
        1 + Log(X) - Exp(X)
    with pytest.raises(TypeError):
        Log(X) ** Exp(X)
    with pytest.raises(ValueError):
        Abs(X) ** sympy.sqrt(10)
    with pytest.raises(ValueError):
        Log(X) * X
    with pytest.raises(ValueError):
        (2*Log(X)) - Rat(1, 10) * Abs(X)
    with pytest.raises(ValueError):
        X**(1.71)
    with pytest.raises(ValueError):
        Abs(X)**(1.1, 8)
    with pytest.raises(ValueError):
        Abs(X)**(7, 8)
    with pytest.raises(ValueError):
        (-3)**X
    with pytest.raises(ValueError):
        (Identity('Z')**2)*(Y > 0)
    with pytest.raises(ValueError):
        (Y > 0) * (Identity('Z')**2)
    with pytest.raises(ValueError):
        ((Y > 0) | (Identity('Z') < 3)) * (Identity('Z')**2)
    with pytest.raises(ValueError):
        Y**2 + (0 <= Y) * Y
    with pytest.raises(ValueError):
        (Y <= 0)*(Y**2) + (0 <= Y)*Y**((1, 2))
    with pytest.raises(ValueError):
        (Y <= 0)*(Y**2) + (0 <= Identity('Z'))*Y**((1, 2))
    with pytest.raises(ValueError):
        (Y <= 0)*(Y**2) + (0 <= Identity('Z'))*Identity('Z')**((1, 2))

    # TypeErrors from 'return NotImplemented'.
    with pytest.raises(TypeError):
        X + 'a'
    with pytest.raises(TypeError):
        X * 'a'
    with pytest.raises(TypeError):
        X / 'a'
    with pytest.raises(TypeError):
        X**'s'

def test_add_polynomials_power_one():
    # TODO: Update parser to handle this edge case.
    Z = X**5
    with pytest.raises(ValueError):
        # GOTCHA: The expression Z**2 is of type Poly(subexpr=Poly)
        # But Z if of type Poly(subexpr=Id)
        Z**2 + Z
    # Raise Z to power one to embed in a polynomial.
    Z**2 + Z**1

def test_negate_polynomial():
    assert -(X**2 + 2) == Poly(X, [-2, 0, -1])

def test_divide_multiplication():
    assert (X**2 + 2) / 2 == Poly(X, [1, 0, Rat(1, 2)])

def test_rdivide_reciprocal():
    assert 1 / X == Reciprocal(X)
    assert 1 / abs(X) == Reciprocal(abs(X))
    assert 3 / abs(X) == Poly(Reciprocal(abs(X)), [0, 3])
    assert 3 / abs(X) == Poly(Reciprocal(abs(X)), [0, 3])
    assert 1 / (X**2 + 2) == Reciprocal(Poly(X, [2, 0, 1]))
    assert 2 / (X**2 + 2) == Poly(Reciprocal(Poly(X, [2, 0, 1])), [0, 2])
    assert -2 / (X**2 + 2) == Poly(Reciprocal(Poly(X, [2, 0, 1])), [0, -2])
    assert 1 + 3 * (1/(abs(X))) + 10 * (1/abs(X))**2 \
         == Poly(Reciprocal(abs(X)), [1, 3, 10])

def test_event_negation_de_morgan():
    A, B, C = (X**3 < 10), (X < 1), ((3*X**2-100) << [10, -10])

    expr_a = ~(A & (B | C))
    expr_b = (~A | (~B & ~C))
    assert expr_a == expr_b

    expr_a = ~(A | (B & C))
    expr_b = (~A & (~B | ~C))
    assert expr_a == expr_b

def test_event_sequential_parse():
    A, B, C , D = (X > 10), (X**2 < 5), (X**3 < 0), (2*X < 7)
    assert A & B & ~C == EventAnd([A, B, ~C])
    assert A & (B & ~C) == EventAnd([A, B, ~C])
    assert (A & B) & (~C & D) == EventAnd([A, B, ~C, D])

    assert A | B | ~C == EventOr([A, B, ~C])
    assert A | (B | ~C) == EventOr([A, B, ~C])
    assert (A | B) | (~C | D) == EventOr([A, B, ~C, D])

    assert A | B | (~C & D) == EventOr([A, B, ~C & D])
    assert A & B & (~C | D) == EventAnd([A, B, ~C | D])

    assert (~C & D) | A | B == EventOr([~C & D, A, B])
    assert (~C & D) | (A | B) == EventOr([~C & D, A, B])

    assert (~C | D) & A & B == EventAnd([~C | D, A, B])
    assert (~C | D) & (A & B) == EventAnd([~C | D, A, B])

def test_event_inequality_parse():
    assert (5 < (X < 10)) \
        == ((5 < X) < 10) \
        == EventInterval(X, Interval.open(5, 10))
    assert (5 <= (X < 10)) \
        == ((5 <= X) < 10) \
        == EventInterval(X, Interval.Ropen(5, 10))
    assert (5 < (X <= 10)) \
        == ((5 < X) <= 10) \
        == EventInterval(X, Interval.Lopen(5, 10))
    assert (5 <= (X <= 10)) \
        == ((5 <= X) <= 10) \
        == EventInterval(X, Interval(5, 10))
    # GOTCHA: This expression is syntactically equivalent to
    # (5 < X) and (X < 10)
    # Since and short circuits and 5 < X is not False,
    # return value of expression is X < 10
    assert (5 < X < 10) == (X < 10)

    # Yields a finite set .
    assert ((5 < X) < 5) == EventFiniteReal(X, EmptySet)
    assert ((5 < X) <= 5) == EventFiniteReal(X, EmptySet)
    assert ((5 <= X) < 5) == EventFiniteReal(X, EmptySet)
    assert ((5 <= X) <= 5) == EventFiniteReal(X, FiniteReal(5))

    # Negated single interval.
    assert ~(5 < X) == (X <= 5)
    assert ~(5 <= X) == (X < 5)
    assert ~(X < 5) == (5 <= X)
    assert ~(X <= 5) == (5 < X)

    # Negated union of two intervals.
    assert ~(5 < (X < 10)) \
        == ~((5 < X) < 10) \
        == (X <= 5) | (10 <= X)
    assert ~(5 <= (X < 10)) \
        == ~((5 <= X) < 10) \
        == (X < 5) | (10 <= X)
    assert ~(5 < (X <= 10)) \
        == ~((5 < X) <= 10) \
        == (X <= 5) | (10 < X)
    assert ~(5 <= (X <= 10)) \
        == ~((5 <= X) <= 10) \
        == (X < 5) | (10 < X)
    assert ~((10 < X) < 5) \
        == ~(10 < (X < 5)) \
        == EventInterval(X, Interval(-oo, oo))

    # A complicated negated union.
    assert ((~(X < 5)) < 10) \
        == ((5 <= X) < 10) \
        == EventInterval(X, Interval.Ropen(5, 10))

def test_event_inequality_parse_errors():
    # EventInterval.
    with pytest.raises(ValueError):
        (X <= 10) < 5
    with pytest.raises(ValueError):
        (X <= 10) <= 5
    with pytest.raises(ValueError):
        5 < (10 <= X)
    with pytest.raises(ValueError):
        5 <= (10 <= X)
    # EventSet.
    with pytest.raises(TypeError):
        5 < (X << (10, 12))
    with pytest.raises(TypeError):
        (X << (10, 12)) < 5
    # Complement., since ~(X < 10) = (10 <= X)
    with pytest.raises(ValueError):
        5 < (~(X < 10))
    # Type mismatch.
    with pytest.raises(TypeError):
        X < 'a'
    with pytest.raises(TypeError):
        X <= 'a'
    with pytest.raises(TypeError):
        X > 'a'
    with pytest.raises(TypeError):
        X >= 'a'

def test_event_inequality_string():
    assert str(5 < X) == '5 < X'
    assert str(5 <= X) == '5 <= X'
    assert str(X < 10) == str(10 > X) == 'X < 10'
    assert str(X <= 10) == str(10 >= X) == 'X <= 10'
    assert str(5 < (X < 10)) == '(5 < X) < 10'
    assert str(5 <= (X < 10)) == '(5 <= X) < 10'
    assert str(5 < (X <= 10)) == '(5 < X) <= 10'
    assert str(5 <= (X <= 10)) == '(5 <= X) <= 10'
    assert str((X < 10) & (X < 5)) == 'X < 5'
    assert str((X < 10) | (X < 5)) == 'X < 10'

def test_event_containment_string():
    assert str(X << [10, 1]) == 'X << {1, 10}'
    assert str(X << {1, 2}) == 'X << {1, 2}'
    assert str(X << FiniteReal(1, 11)) == 'X << {1, 11}'

def test_event_containment_real():
    assert (X << Interval(0, 10)) == EventInterval(X, Interval(0, 10))
    for values in [FiniteReal(0, 10), [0, 10], {0, 10}]:
        assert (X << values) == EventFiniteReal(X, FiniteReal(0, 10))
    # with pytest.raises(ValueError):
    #     X << {1, None}
    assert X << {1, 2} == EventFiniteReal(X, {1, 2})
    assert ~(X << {1, 2}) == EventOr([
        EventInterval(X, Interval.Ropen(-oo, 1)),
        EventInterval(X, Interval.open(1, 2)),
        EventInterval(X, Interval.Lopen(2, oo)),
    ])
    # https://github.com/probcomp/sum-product-dsl/issues/22
    # and of EventBasic does not yet perform simplifications.
    assert ~(~(X << {1, 2})) == \
        ((1 <= X) & ((X <= 1) | (2 <= X)) & (X <= 2))

def test_event_containment_nominal():
    assert X << {'a'} == EventFiniteNominal(X, FiniteNominal('a'))
    assert ~(X << {'a'}) == EventFiniteNominal(X, FiniteNominal('a',b=True))
    assert ~(~(X << {'a'})) == X << {'a'}

def test_event_containment_mixed():
    assert X << {1, 2, 'a'} \
        == (X << {1,2}) | (X << {'a'}) \
        == EventOr([
        EventFiniteReal(X, {1, 2}),
        EventFiniteNominal(X, FiniteNominal('a'))
    ])
    assert (~(X << {1, 2, 'a'})) == ~(X << {1,2}) & ~(X << {'a'})

    # https://github.com/probcomp/sum-product-dsl/issues/22
    # Taking the And of EventBasic does not perform simplifications.
    assert ~(~(X << {1, 2, 'a'})) == EventOr([
        ((1 <= X) & ((X <= 1) | (2 <= X)) & (X <= 2)),
        X << {'a'}
    ])

def test_event_containment_union():
    assert (X << (Interval(0, 1) | Interval(2, 3))) \
        == (((0 <= X) <= 1) | ((2 <= X) <= 3))
    assert (X << (FiniteReal(0, 1) | Interval(2, 3))) \
        == ((X << {0, 1}) | ((2 <= X) <= 3))
    assert (X << FiniteNominal('a', b=True)) \
        == EventFiniteNominal(X, FiniteNominal('a', b=True))
    assert X << EmptySet == EventFiniteReal(X, EmptySet)
    # Ordering is not guaranteed.
    a = X << (Interval(0,1) | (FiniteReal(1.5) | FiniteNominal('a')))
    assert len(a.subexprs) == 3
    assert EventInterval(X, Interval(0,1)) in a.subexprs
    assert EventFiniteReal(X, FiniteReal(1.5)) in a.subexprs
    assert EventFiniteNominal(X, FiniteNominal('a')) in a.subexprs

def test_event_basic_simplifications():
    assert (1 < X) & (3 < X) == (3 < X)
    assert (1 < X) & (X < 5) == ((1 < X) < 5)
    assert (X < 1) & (X >= -3) == ((-3 <= X) < 1)
    assert (X >= -3) & (X << {1}) == X << {1}
    assert (X << {1, -10}) & (X >= -3) == X << {1}
    assert (X << {1, -10}) & (X << {-3}) == (X << EmptySet)
    assert (X << {1, -3}) & (X << {-3}) == X << {-3}

    assert (1 < X) | (3 < X) == (1 < X)
    assert (1 < X) | (X << {2}) == (1 < X)
    assert (X << {7}) | (1 < X)  == (1 < X)
    assert (X << {-10}) | (1 < X) == EventOr([X<<{-10}, 1<X])
    assert (X << {2}) | (X << {3,4}) == (X << {2, 3, 4})
    assert (X << {'2'}) | (X << {'a'}) == X << {'2', 'a'}
    assert (X << {'2'}) & (~(X << {'a'})) == (X << {'2'})
    assert (X << {1}) & (~(X << {'a'})) == (X << EmptySet)

    assert ~(X << {2}) & (X << {'a'}) == EventAnd([~(X << {2}), X <<{'a'}])
    assert (X < 1) & (X << {'a'}) == X << EmptySet
    assert ((0 <= X) < 1) & (~(X << {'a'})) == X << EmptySet

    assert (X << {2}) & (X << {'a'}) == X << EmptySet
    assert (X << {'a'}) & (X << {1}) == X << EmptySet
    assert (X << EmptySet) & (X << EmptySet) == X << EmptySet
    assert (X << EmptySet) & (~(X << EmptySet)) == X << EmptySet
    assert (X << {'a'}) & ~(X << {'b'}) == (X << {'a'})

    # Complement case in Event.__or__.
    assert (X << {1}) | (~(X << {'a'})) == EventOr([X<<{1}, ~(X<<{'a'})])
    assert (X << {'a'}) | ~(X << {'b'}) == ~(X << {'b'})

def test_event_complex_simplification():
    # De Morgan's case in EventFinteNominal.__or__.
    assert ~(X << {'a'}) | ~(X << {'b'}) == EventFiniteNominal(X, FiniteNominal(b=True))
    assert ~(X << {'a'}) | ~(X << {'a', 'b'}) == ~(X << {'a'})
    assert (X << {'1'}) | ~(X << {'1', '2'}) == ~(X << {'2'})

def test_xor():
    A = X << {'a'}
    B = ~(X << {'b'})
    assert (A ^ B) \
        == ((A & ~B) | (~A & B)) \
        == EventFiniteNominal(X, FiniteNominal('a', 'b', b=True))
