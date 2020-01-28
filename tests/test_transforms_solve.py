# Copyright 2019 MIT Probabilistic Computing Project.
# See LICENSE.txt

import pytest

import sympy

from sympy import Interval
from sympy import Rational
from sympy import S as Singletons
from sympy import oo

from sum_product_dsl.math_util import allclose
from sum_product_dsl.solver import solver
from sum_product_dsl.transforms import ExpNat
from sum_product_dsl.transforms import Identity
from sum_product_dsl.transforms import LogNat
from sum_product_dsl.transforms import Sqrt

X = sympy.symbols("X")
Y = Identity("Y")

def test_solver_1_open():
    # log(x) > 2
    solution = Interval.open(sympy.exp(2), oo)

    expr = sympy.log(X) > 2
    answer = solver(expr)
    assert answer == solution

    event = LogNat(Y) > 2
    answer = event.solve()
    assert answer == solution

def test_solver_1_closed():
    # log(x) >= 2
    solution = Interval(sympy.exp(2), oo)

    expr = sympy.log(X) >= 2
    answer = solver(expr)
    assert answer == solution

    event = LogNat(Y) >= 2
    answer = event.solve()
    assert answer == solution

def test_solver_2_open():
    # log(x) < 2 & (x < exp(2))
    solution = Singletons.EmptySet

    expr = (sympy.log(X) > 2) & (X < sympy.exp(2))
    answer = solver(expr)
    assert answer == solution

    event = (LogNat(Y) > 2) & (Y < sympy.exp(2))
    answer = event.solve()
    assert answer == solution

def test_solver_2_closed():
    # (log(x) <= 2) & (x >= exp(2))
    solution = sympy.FiniteSet(sympy.exp(2))

    expr = (sympy.log(X) >= 2) & (X <= sympy.exp(2))
    answer = solver(expr)
    assert answer == solution

    event = (LogNat(Y) >= 2) & (Y <= sympy.exp(2))
    answer = event.solve()
    assert answer == solution

def test_solver_4():
    # (x >= 0) & (x <= 0)
    solution = Singletons.Reals

    expr = (X >= 0) | (X <= 0)
    answer = solver(expr)
    assert answer == solution

    event = (Y >= 0) | (Y <= 0)
    answer = event.solve()
    assert answer == solution

def test_solver_5_open():
    # (2*x+10 < 4) & (x + 10 > 3)
    solution = Interval.open(3-10, (4-10)/2)

    expr = ((2*X + 10) < 4) & (X + 10 > 3)
    answer = solver(expr)
    assert answer == solution

    event = ((2*Y + 10) < 4) & (Y + 10 > 3)
    answer = event.solve()
    assert answer == solution

def test_solver_5_ropen():
    # (2*x+10 < 4) & (x + 10 >= 3)
    solution = Interval.Ropen(3-10, (4-10)/2)

    expr = ((2*X + 10) < 4) & (X + 10 >= 3)
    answer = solver(expr)
    assert answer == solution

    event = ((2*Y + 10) < 4) & (Y + 10 >= 3)
    answer = event.solve()
    assert answer == solution

def test_solver_5_lopen():
    # (2*x + 10 < 4) & (x + 10 >= 3)
    solution =Interval.Lopen(3-10, (4-10)/2)

    expr = ((2*X + 10) <= 4) & (X + 10 > 3)
    answer = solver(expr)
    assert answer == solution

    event = ((2*Y + 10) <= 4) & (Y + 10 > 3)
    answer = event.solve()
    assert answer == solution

def test_solver_6():
    # (x**2 - 2*x) > 10
    solution =  sympy.Union(
        Interval.open(-oo, 1 - sympy.sqrt(11)),
        Interval.open(1 + sympy.sqrt(11), oo))

    expr = (X**2 - 2*X) > 10
    answer = solver(expr)
    assert answer == solution

    event = (Y**2 - 2*Y) > 10
    answer = event.solve()
    assert answer == solution

def test_solver_7():
    # Illegal expression, cannot express in our custom DSL.
    expr = (X**2 - 2*X + sympy.exp(X)) > 10
    with pytest.raises(ValueError):
        solver(expr)

def test_solver_8():
    Z = sympy.symbols('Z')
    expr = (X + Z < 3)
    with pytest.raises(ValueError):
        solver(expr)

def test_solver_9_open():
    # 2(log(x))**3 - log(x) -5 > 0
    solution = Interval.open(
        sympy.exp(1/(6*(sympy.sqrt(2019)/36 + 5/4)**(1/3))
            + (sympy.sqrt(2019)/36 + 5/4)**(1/3)),
        oo)

    expr = 2*(sympy.log(X))**3 - sympy.log(X) - 5 > 0
    with pytest.raises(ValueError):
        assert solver(expr) == solution

    # Our solver handles this case as follows
    # expr' = 2*Z**3 - Z - 5 > 0 [[subst. Z=log(X)]]
    # [Z_low, Z_high] = solver(expr')
    #       Z_low < Z iff Z_low < log(X) iff exp(Z_low) < X
    #       Z < Z_high iff log(X) < Z_high iff X < exp(Z_high)
    # solver(expr) = [exp(Z_low), exp(Z_high)]
    # For F invertible, can thus solve Poly(coeffs, F) > 0 using this method.
    event = 2*(LogNat(Y))**3 - LogNat(Y) - 5 > 0
    answer = event.solve()
    assert answer == solution

def test_solver_9_closed():
    # 2(log(x))**3 - log(x) -5 >= 0
    solution = Interval(
        sympy.exp(1/(6*(sympy.sqrt(2019)/36 + 5/4)**(1/3))
            + (sympy.sqrt(2019)/36 + 5/4)**(1/3)),
        oo)

    expr = 2*(sympy.log(X))**3 - sympy.log(X) -5 > 0
    with pytest.raises(ValueError):
        assert solver(expr) == solution

    event = 2*(LogNat(Y))**3 - LogNat(Y) - 5 >= 0
    answer = event.solve()
    assert answer == solution

def test_solver_10():
    # exp(sqrt(log(x))) > -5
    solution = Interval(1, oo)

    # Sympy hangs for some reason; cannot test.
    # expr = exp(sqrt(log(X))) > -5

    event = ExpNat(Sqrt(LogNat(Y))) > -5
    answer = event.solve()
    assert answer == solution

def test_solver_11_open():
    # exp(sqrt(log(x))) > 6
    solution = Interval.open(sympy.exp(sympy.log(6)**2), oo)

    # Sympy hangs for some reason.
    # expr = exp(sqrt(log(X))) > 6

    event = ExpNat(Sqrt(LogNat(Y))) > 6
    answer = event.solve()
    assert answer == solution

def test_solver_11_closed():
    # exp(sqrt(log(x))) >= 6
    solution = Interval(sympy.exp(sympy.log(6)**2), oo)

    # Sympy hangs for some reason.
    # expr = exp(sqrt(log(X))) > 6

    event = ExpNat(Sqrt(LogNat(Y))) >= 6
    answer = event.solve()
    assert answer == solution

def test_solver_12():
    # 2*sqrt(|x|) - 3 > 10
    solution = sympy.Union(
        Interval.open(-oo, -Rational(169, 4)),
        Interval.open(Rational(169, 4), oo))

    expr = 2*sympy.sqrt(sympy.Abs(X)) - 3 > 10
    answer = solver(expr)
    assert answer == solution

    event = (2*Sqrt(abs(Y)) - 3) > 10
    answer = event.solve()
    assert answer == solution

def test_solver_13():
    # 2*sqrt(|x|**2) - 3 > 10
    solution = sympy.Union(
        Interval.open(-oo, -Rational(13, 2)),
        Interval.open(Rational(13, 2), oo))

    expr = 2*sympy.sqrt(sympy.Abs(X)**2) - 3 > 10
    answer = solver(expr)
    assert answer == solution

    event = (2*Sqrt(abs(Y)**2) - 3) > 10
    answer = event.solve()
    assert answer == solution

def test_solver_14():
    # x**2 > 10
    solution = sympy.Union(
        Interval.open(-oo, -sympy.sqrt(10)),
        Interval.open(sympy.sqrt(10), oo))

    expr = X**2 > 10
    answer = solver(expr)
    assert answer == solution

    event = Y**2 > 10
    answer = event.solve()
    assert answer == solution

def test_solver_15():
    # ((x**4)**(1/7)) < 9
    solution = Interval.open(-27*sympy.sqrt(3), 27*sympy.sqrt(3))

    expr = ((X**4))**(Rational(1, 7)) < 9
    answer = solver(expr)
    with pytest.raises(AssertionError):
        # SymPy handles exponents unclearly.
        # Exponent laws with negative numbers are subtle.
        assert answer == solution

    event = ((Y**4))**(Rational(1, 7)) < 9
    answer = event.solve()
    assert answer == solution

def test_solver_16():
    # (x**(1/7))**4 < 9
    solution = Interval.Ropen(0, 27*sympy.sqrt(3))

    expr = ((X**Rational(1,7)))**4 < 9
    answer = solver(expr)
    with pytest.raises(AssertionError):
        # SymPy handles exponents unclearly.
        # Exponent laws with negative numbers are subtle.
        assert answer == solution

    event = ((Y**Rational(1,7)))**4 < 9
    answer = event.solve()
    assert answer == solution

@pytest.mark.xfail(reason='too slow', strict=True)
@pytest.mark.timeout(3)
def test_solver_17():
    p = sympy.Poly(
        (X - sympy.sqrt(2)/10) * (X+Rational(10, 7)) * (X - sympy.sqrt(5)),
        X)
    expr = p.args[0] < 1
    solver(expr)

def test_solver_18():
    # 3*(x**(1/7))**4 - 3*(x**(1/7))**2 <= 9
    solution = Interval(0, (Rational(1, 2) + sympy.sqrt(13)/2)**(Rational(7, 2)))

    Z = Y**(Rational(1, 7))
    expr = 3*Z**4 - 3*Z**2
    event = (expr <= 9)
    answer = event.solve()
    assert answer == solution

    interval = (~event).solve()
    assert interval == sympy.Union(
        Interval.open(-oo, 0),
        Interval.open(solution.right, oo))

def test_solver_19():
    # 3*(x**(1/7))**4 - 3*(x**(1/7))**2 <= 9
    #   or || 3*(x**(1/7))**4 - 3*(x**(1/7))**2 > 11
    solution = sympy.Union(
        Interval(0, (Rational(1, 2) + sympy.sqrt(13)/2)**(Rational(7, 2))),
        Interval.open((Rational(1,2) + sympy.sqrt(141)/6)**(7/2), oo))

    Z = Y**(Rational(1, 7))
    expr = 3*Z**4 - 3*Z**2
    event = (expr <= 9) | (expr > 11)
    answer = event.solve()
    assert answer == solution

    interval = (~event).solve()
    assert interval == sympy.Union(
        Interval.open(-oo, 0),
        Interval.Lopen(solution.args[0].right, solution.args[1].left))

def test_solver_20():
    # log(x**2 - 3) < 5
    solution = sympy.Union(
        Interval.open(-sympy.sqrt(3 + sympy.exp(5)), -sympy.sqrt(3)),
        Interval.open(sympy.sqrt(3), sympy.sqrt(3 + sympy.exp(5))))

    expr = sympy.log(X**2 - 3) < 5
    answer = solver(expr)
    assert answer == solution

    event = LogNat(Y**2 - 3) < 5
    answer = event.solve()
    assert answer == solution

def test_solver_21__ci_():
    # 1 <= log(x**3 - 3*x + 3) < 5
    # Can only be solved by numerical approximation of roots.
    # https://www.wolframalpha.com/input/?i=1+%3C%3D+log%28x**3+-+3x+%2B+3%29+%3C+5
    solution = sympy.Union(
        Interval(
            -1.777221448430427630375448631016427343692,
            0.09418455242255462832154474245589911789464),
        Interval.Ropen(
            1.683036896007873002053903888560528225797,
            5.448658707897512189124586716091172798465))

    with pytest.raises(ValueError):
        term = sympy.log(X**3 - 3*X + 3)
        expr = (1 < term) & (term < 5)
        answer = solver(expr)

    expr = LogNat(Y**3 - 3*Y + 3)
    event = ((1 <= expr) & (expr < 5))
    answer = event.solve()
    assert isinstance(answer, sympy.Union)
    # Check first interval.
    assert not answer.args[0].left_open
    assert not answer.args[0].right_open
    assert allclose(float(answer.args[0].inf), float(solution.args[0].inf))
    assert allclose(float(answer.args[0].sup), float(solution.args[0].sup))
    # Check second interval.
    assert not answer.args[1].left_open
    assert answer.args[1].right_open
    assert allclose(float(answer.args[1].inf), float(solution.args[1].inf))
    assert allclose(float(answer.args[1].sup), float(solution.args[1].sup))
