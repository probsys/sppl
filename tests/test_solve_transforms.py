# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

import pytest
import sympy

from sympy import Interval
from sympy import Rational as Rat
from sympy import oo

from spn.math_util import allclose
from spn.sym_util import EmptySet
from spn.sym_util import NominalSet
from spn.sym_util import Reals
from spn.sym_util import UniversalSet
from spn.sym_util import sympy_solver
from spn.transforms import ExpNat
from spn.transforms import Identity
from spn.transforms import Log
from spn.transforms import LogNat
from spn.transforms import Sqrt

X = sympy.symbols("X")
Y = Identity("Y")

def test_solver_1_open():
    # log(x) > 2
    solution = Interval.open(sympy.exp(2), oo)

    expr = sympy.log(X) > 2
    answer = sympy_solver(expr)
    assert answer == solution

    event = LogNat(Y) > 2
    answer = event.solve()
    assert answer == solution

def test_solver_1_closed():
    # log(x) >= 2
    solution = Interval(sympy.exp(2), oo)

    expr = sympy.log(X) >= 2
    answer = sympy_solver(expr)
    assert answer == solution

    event = LogNat(Y) >= 2
    answer = event.solve()
    assert answer == solution

def test_solver_2_open():
    # log(x) < 2 & (x < exp(2))
    solution = EmptySet

    expr = (sympy.log(X) > 2) & (X < sympy.exp(2))
    answer = sympy_solver(expr)
    assert answer == solution

    event = (LogNat(Y) > 2) & (Y < sympy.exp(2))
    answer = event.solve()
    assert answer == solution

def test_solver_2_closed():
    # (log(x) <= 2) & (x >= exp(2))
    solution = sympy.FiniteSet(sympy.exp(2))

    expr = (sympy.log(X) >= 2) & (X <= sympy.exp(2))
    answer = sympy_solver(expr)
    assert answer == solution

    event = (LogNat(Y) >= 2) & (Y <= sympy.exp(2))
    answer = event.solve()
    assert answer == solution

def test_solver_4():
    # (x >= 0) & (x <= 0)
    solution = Reals

    expr = (X >= 0) | (X <= 0)
    answer = sympy_solver(expr)
    assert answer == solution

    event = (Y >= 0) | (Y <= 0)
    answer = event.solve()
    assert answer == solution

def test_solver_5_open():
    # (2*x+10 < 4) & (x + 10 > 3)
    solution = Interval.open(3-10, (4-10)/2)

    expr = ((2*X + 10) < 4) & (X + 10 > 3)
    answer = sympy_solver(expr)
    assert answer == solution

    event = ((2*Y + 10) < 4) & (Y + 10 > 3)
    answer = event.solve()
    assert answer == solution

def test_solver_5_ropen():
    # (2*x+10 < 4) & (x + 10 >= 3)
    solution = Interval.Ropen(3-10, (4-10)/2)

    expr = ((2*X + 10) < 4) & (X + 10 >= 3)
    answer = sympy_solver(expr)
    assert answer == solution

    event = ((2*Y + 10) < 4) & (Y + 10 >= 3)
    answer = event.solve()
    assert answer == solution

def test_solver_5_lopen():
    # (2*x + 10 < 4) & (x + 10 >= 3)
    solution =Interval.Lopen(3-10, (4-10)/2)

    expr = ((2*X + 10) <= 4) & (X + 10 > 3)
    answer = sympy_solver(expr)
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
    answer = sympy_solver(expr)
    assert answer == solution

    event = (Y**2 - 2*Y) > 10
    answer = event.solve()
    assert answer == solution

def test_solver_7():
    # Illegal expression, cannot express in our custom DSL.
    expr = (X**2 - 2*X + sympy.exp(X)) > 10
    with pytest.raises(ValueError):
        sympy_solver(expr)

def test_solver_8():
    Z = sympy.symbols('Z')
    expr = (X + Z < 3)
    with pytest.raises(ValueError):
        sympy_solver(expr)

def test_solver_9_open():
    # 2(log(x))**3 - log(x) -5 > 0
    solution = Interval.open(
        sympy.exp(1/(6*(sympy.sqrt(2019)/36 + 5/4)**(1/3))
            + (sympy.sqrt(2019)/36 + 5/4)**(1/3)),
        oo)

    expr = 2*(sympy.log(X))**3 - sympy.log(X) - 5 > 0
    with pytest.raises(ValueError):
        assert sympy_solver(expr) == solution

    # Our solver handles this case as follows
    # expr' = 2*Z**3 - Z - 5 > 0 [[subst. Z=log(X)]]
    # [Z_low, Z_high] = sympy_solver(expr')
    #       Z_low < Z iff Z_low < log(X) iff exp(Z_low) < X
    #       Z < Z_high iff log(X) < Z_high iff X < exp(Z_high)
    # sympy_solver(expr) = [exp(Z_low), exp(Z_high)]
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
        assert sympy_solver(expr) == solution

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
        Interval.open(-oo, -Rat(169, 4)),
        Interval.open(Rat(169, 4), oo))

    expr = 2*sympy.sqrt(sympy.Abs(X)) - 3 > 10
    answer = sympy_solver(expr)
    assert answer == solution

    event = (2*Sqrt(abs(Y)) - 3) > 10
    answer = event.solve()
    assert answer == solution

def test_solver_13():
    # 2*sqrt(|x|**2) - 3 > 10
    solution = sympy.Union(
        Interval.open(-oo, -Rat(13, 2)),
        Interval.open(Rat(13, 2), oo))

    expr = 2*sympy.sqrt(sympy.Abs(X)**2) - 3 > 10
    answer = sympy_solver(expr)
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
    answer = sympy_solver(expr)
    assert answer == solution

    event = Y**2 > 10
    answer = event.solve()
    assert answer == solution

def test_solver_15():
    # ((x**4)**(1/7)) < 9
    solution = Interval.open(-27*sympy.sqrt(3), 27*sympy.sqrt(3))

    expr = ((X**4))**(Rat(1, 7)) < 9
    answer = sympy_solver(expr)
    with pytest.raises(AssertionError):
        # SymPy handles exponents unclearly.
        # Exponent laws with negative numbers are subtle.
        assert answer == solution

    event = ((Y**4))**(Rat(1, 7)) < 9
    answer = event.solve()
    assert answer == solution

def test_solver_16():
    # (x**(1/7))**4 < 9
    solution = Interval.Ropen(0, 27*sympy.sqrt(3))

    expr = ((X**Rat(1,7)))**4 < 9
    answer = sympy_solver(expr)
    with pytest.raises(AssertionError):
        # SymPy handles exponents unclearly.
        # Exponent laws with negative numbers are subtle.
        assert answer == solution

    event = ((Y**Rat(1,7)))**4 < 9
    answer = event.solve()
    assert answer == solution

@pytest.mark.xfail(reason='too slow', strict=True)
@pytest.mark.timeout(3)
def test_solver_17():
    p = sympy.Poly(
        (X - sympy.sqrt(2)/10) * (X+Rat(10, 7)) * (X - sympy.sqrt(5)),
        X)
    expr = p.args[0] < 1
    sympy_solver(expr)

def test_solver_18():
    # 3*(x**(1/7))**4 - 3*(x**(1/7))**2 <= 9
    solution = Interval(0, (Rat(1, 2) + sympy.sqrt(13)/2)**(Rat(7, 2)))

    Z = Y**(Rat(1, 7))
    expr = 3*Z**4 - 3*Z**2
    event = (expr <= 9)
    answer = event.solve()
    assert answer == solution

    interval = (~event).solve()
    assert interval == Interval.open(solution.right, oo)

def test_solver_19():
    # 3*(x**(1/7))**4 - 3*(x**(1/7))**2 <= 9
    #   or || 3*(x**(1/7))**4 - 3*(x**(1/7))**2 > 11
    solution = sympy.Union(
        Interval(0, (Rat(1, 2) + sympy.sqrt(13)/2)**(Rat(7, 2))),
        Interval.open((Rat(1,2) + sympy.sqrt(141)/6)**(7/2), oo))

    Z = Y**(Rat(1, 7))
    expr = 3*Z**4 - 3*Z**2
    event = (expr <= 9) | (expr > 11)
    answer = event.solve()
    assert answer == solution

    interval = (~event).solve()
    assert interval == Interval.Lopen(
        solution.args[0].right,
        solution.args[1].left)

def test_solver_20():
    # log(x**2 - 3) < 5
    solution = sympy.Union(
        Interval.open(-sympy.sqrt(3 + sympy.exp(5)), -sympy.sqrt(3)),
        Interval.open(sympy.sqrt(3), sympy.sqrt(3 + sympy.exp(5))))

    expr = sympy.log(X**2 - 3) < 5
    answer = sympy_solver(expr)
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
        answer = sympy_solver(expr)

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

def test_solver_22():
    # 2 < abs(X) < 5
    event = (2 < abs(Y)) < 5
    solution = Interval.open(2, 5) + Interval.open(-5, -2)
    assert event.solve() == solution
    # 2 <= abs(X) < 5
    event = (2 <= abs(Y)) < 5
    solution = Interval.Ropen(2, 5) + Interval.Lopen(-5, -2)
    assert event.solve() == solution
    # 2 < abs(X) <= 5
    event = (2 <  abs(Y)) <= 5
    solution = Interval.Lopen(2, 5) + Interval.Ropen(-5, -2)
    assert event.solve() == solution
    # 2 <= abs(X) <= 5
    event = (2 <=  abs(Y)) <= 5
    solution = Interval(2, 5) + Interval(-5, -2)
    assert event.solve() == solution

    # -2 < abs(X) < 5
    event = (-2 < abs(Y)) < 5
    solution = Interval.open(-5, 5)
    assert event.solve() == solution
    # # -2 <= abs(X) < 5
    event = (-2 <= abs(Y)) < 5
    solution = Interval.open(-5, 5)
    assert event.solve() == solution
    # -2 < abs(X) <= 5
    event = (-2 <  abs(Y)) <= 5
    solution = Interval(-5, 5)
    assert event.solve() == solution
    # 2 <= abs(X) <= 5
    event = (-2 <=  abs(Y)) <= 5
    solution = Interval(-5, 5)
    assert event.solve() == solution

def test_solver_23_reciprocal_lte():
    for c in [1, 3]:
        # Positive
        # 1 / X < 10
        solution = Interval.Ropen(-oo, 0) + Interval.Lopen(Rat(c, 10), oo)
        event = (c / Y) < 10
        assert event.solve() == solution
        # 1 / X <= 10
        solution = Interval.Ropen(-oo, 0) + Interval(Rat(c, 10), oo)
        event = (c / Y) <= 10
        assert event.solve() == solution
        # Negative.
        # 1 / X < -10
        solution = Interval.open(-Rat(c, 10), 0)
        event = (c / Y) < -10
        assert event.solve() == solution
        # 1 / X <= -10
        solution = Interval.Ropen(-Rat(c, 10), 0)
        event = (c / Y) <= -10
        assert event.solve() == solution

def test_solver_23_reciprocal_gte():
    for c in [1, 3]:
        # Positive
        # 10 < 1 / X
        solution = Interval.open(0, Rat(c, 10))
        event = 10 < (c / Y)
        assert event.solve() == solution
        # 10 <= 1 / X
        solution = Interval.Lopen(0, Rat(c, 10))
        event = 10 <= (c / Y)
        assert event.solve() == solution
        # Negative
        # -10 < 1 / X
        solution = Interval.Lopen(0, oo) + Interval.open(-oo, -Rat(c, 10))
        event = -10 < (c / Y)
        assert event.solve() == solution
        # -10 <= 1 / X
        solution = Interval.Lopen(0, oo) + Interval.Lopen(-oo, -Rat(c, 10))
        event =  -10 <= (c / Y)
        assert event.solve() == solution

def test_solver_23_reciprocal_range():
    solution = Interval.Ropen(-1, -Rat(1, 3))
    event = ((-3 < 1/Y) <= -1)
    assert event.solve() == solution

    solution = Interval.open(0, Rat(1, 3))
    event = ((-3 < 1/(2*Y-1)) < -1)
    assert event.solve() == solution

    solution = Interval.open(-1 / sympy.sqrt(3), 1 / sympy.sqrt(3))
    event = ((-3 < 1/(2*(abs(Y)**2)-1)) <= -1)
    assert event.solve() == solution

    solution = sympy.Union(
        Interval.open(-1 / sympy.sqrt(3), 0),
        Interval.open(0, 1 / sympy.sqrt(3)))
    event = ((-3 < 1/(2*(abs(Y)**2)-1)) < -1)
    assert event.solve() == solution

def test_solver_24_negative_power_integer():
    # Case 1.
    event = Y**(-3) < 6
    assert event.solve() == sympy.Union(
        Interval.open(-oo, 0),
        Interval.open(6**Rat(-1, 3), oo))
    # Case 2.
    event = (-1 < Y**(-3)) < 6
    assert event.solve() == sympy.Union(
        Interval.open(-oo, -1),
        Interval.open(6**Rat(-1, 3), oo))
    # Case 3.
    event = 5 <= Y**(-3)
    assert event.solve() == Interval.Lopen(0, 5**Rat(-1, 3))
    # Case 4.
    event = (5 <= Y**(-3)) < 6
    assert event.solve() == Interval.Lopen(6**Rat(-1, 3), 5**Rat(-1, 3))

def test_solver_24_negative_power_Rat():
    # Case 1.
    event = Y**Rat(-1, 3) < 6
    assert event.solve() == Interval.Lopen(Rat(1, 216), oo)
    # Case 2.
    event = (-1 < Y**Rat(-1, 3)) < 6
    assert event.solve() == Interval.Lopen(Rat(1, 216), oo)
    # Case 3.
    event = 5 <= Y**Rat(-1, 3)
    assert event.solve() == Interval.Lopen(0, Rat(1, 125))
    # Case 4.
    event = (5 <= Y**Rat(-1, 3)) < 6
    assert event.solve() == Interval.Lopen(Rat(1, 216), Rat(1, 125))

def test_solver_25_constant():
    event = (0*Y + 1) << {1}
    assert event.solve() == Reals
    event = (0*Y + 1) << {0}
    assert event.solve() is EmptySet
    event = (0.9 < (0*Y + 1)) < 1
    assert event.solve() is EmptySet
    event = (0.9 < (0*Y + 1)**2) <= 1
    assert event.solve() == Reals
    event = (0.9 < (0*Y + 2)**2) <= 1
    assert event.solve() is EmptySet
    event = (0*Y + 2)**2 << {4}
    assert event.solve() == Reals

def test_solver_26_piecewise_one_expr_basic_event():
    event = (Y**2)*(0 <= Y) < 2
    assert event.solve() == sympy.Interval.Ropen(0, sympy.sqrt(2))
    event = (0 <= Y)*(Y**2) < 2
    assert event.solve() == sympy.Interval.Ropen(0, sympy.sqrt(2))
    event = ((0 <= Y) < 5)*(Y < 1) << {1}
    assert event.solve() == sympy.Interval.Ropen(0, 1)
    event = ((0 <= Y) < 5)*(~(Y < 1)) << {1}
    assert event.solve() == sympy.Interval.Ropen(1, 5)
    event = 10*(0 <= Y) << {10}
    assert event.solve() == sympy.Interval(0, sympy.oo)
    event = 10*(0 <= Y) << {0}
    assert event.solve() == sympy.Interval.Ropen(-sympy.oo, 0)

def test_solver_26_piecewise_one_expr_compound_event():
    event = (Y**2)*((Y < 0) | (0 < Y)) < 2
    assert event.solve() == sympy.Union(
        sympy.Interval.open(-sympy.sqrt(2), 0),
        sympy.Interval.open(0, sympy.sqrt(2)))

def test_solver_27_piecewise_many():
    expr = (Y < 0)*(Y**2) + (0 <= Y)*Y**(Rat(1, 2))
    event = expr << {3}
    assert sorted(event.solve()) == [-sympy.sqrt(3), 9]
    event = 0 < expr
    assert event.solve() == sympy.Union(
        sympy.Interval.open(-oo, 0),
        sympy.Interval.open(0, oo))

    # TODO: Consider banning the restriction of a function
    # to a segment outside of its domain.
    expr = (Y < 0)*Y**(Rat(1, 2))
    assert (expr < 1).solve() is EmptySet

def test_solver_finite_injective():
    sqrt3 = sympy.sqrt(3)
    # Identity.
    solution = {2, 4, -10, sqrt3}
    event = Y << {2, 4, -10, sqrt3}
    assert event.solve() == solution
    # ExpNat.
    solution = {sympy.log(10), sympy.log(3), sympy.log(sqrt3)}
    event = ExpNat(Y) << {10, 3, sqrt3}
    assert event.solve() == solution
    # Exp2.
    solution = {sympy.log(10, 2), 4, sympy.log(sqrt3, 2)}
    event = (2**Y) << {10, 16, sqrt3}
    assert event.solve() == solution
    # LogNat.
    solution = {sympy.exp(10), sympy.exp(-3), sympy.exp(sqrt3)}
    event = LogNat(Y) << {10, -3, sqrt3}
    assert event.solve() == solution
    # Log2
    solution = {sympy.Pow(2, 10), sympy.Pow(2, -3), sympy.Pow(2, sqrt3)}
    event = Log(Y, 2) << {10, -3, sqrt3}
    assert event.solve() == solution
    # Radical.
    solution = {7**4, 12**4, sqrt3**4}
    event = Y**Rat(1, 4) << {7, 12, sqrt3}
    assert event.solve() == solution

def test_solver_finite_non_injective():
    sqrt2 = sympy.sqrt(2)
    # Abs.
    solution = {-10, -3, 3, 10}
    event = abs(Y) << {10, 3}
    assert event.solve() == solution
    # Abs(Poly).
    solution = {-5, -Rat(3,2), Rat(3,2), 5}
    event = abs(2*Y) << {10, 3}
    assert event.solve() == solution
    # Poly order 2.
    solution = {-sqrt2, sqrt2}
    event = (Y**2) << {2}
    assert event.solve() == solution
    # Poly order 3.
    solution = {1, 3}
    event = Y**3 << {1, 27}
    assert event.solve() == solution
    # Poly Abs.
    solution = {-3, -1, 1, 3}
    event = (abs(Y))**3 << {1, 27}
    assert event.solve() == solution
    # Abs Not.
    solution = sympy.Union(
        sympy.Interval.open(-oo, -1),
        sympy.Interval.open(-1, 1),
        sympy.Interval.open(1, oo))
    event = ~(abs(Y) << {1})
    assert event.solve() == solution
    # Abs in EmptySet.
    solution = EmptySet
    event = (abs(Y))**3 << set([])
    assert event.solve() == solution
    # Abs Not in EmptySet (yields all reals).
    solution = sympy.Interval(-sympy.oo, sympy.oo)
    event = ~(((abs(Y))**3) << set([]))
    assert event.solve() == solution
    # Log in Reals (yields positive reals).
    solution = sympy.Interval.open(0, sympy.oo)
    event = ~((LogNat(Y))**3 << set([]))
    assert event.solve() == solution

def test_solver_finite_symbolic():
    # Transform can never be symbolic.
    event = Y << {'a', 'b'}
    assert event.solve() == NominalSet('a', 'b')
    # Complement the Identity.
    event = ~(Y << {'a', 'b'})
    assert event.solve() == sympy.Complement(
        UniversalSet, NominalSet('a', 'b'))
    # Transform can never be symbolic.
    event = Y**2 << {'a', 'b'}
    assert event.solve() is EmptySet
    # Complement the Identity.
    event = ~(Y**2 << {'a', 'b'})
    assert event.solve() == UniversalSet
    # Solve Identity mixed.
    event = Y << {9, 'a', '7'}
    assert event.solve() == sympy.Union(
        sympy.FiniteSet(9),
        NominalSet('a', '7'))
    # Solve Transform mixed.
    event = Y**2 << {9, 'a', 'b'}
    assert event.solve() == {-3, 3}
    # Solve a disjunction.
    event = (Y << {'a', 'b'}) | (Y << {'c'})
    assert event.solve() == NominalSet('a', 'b', 'c')
    # Solve a conjunction with intersection.
    event = (Y << {'a', 'b'}) & (Y << {'b', 'c'})
    assert event.solve() == NominalSet('b')
    # Solve a conjunction with no intersection.
    event = (Y << {'a', 'b'}) & (Y << {'c'})
    assert event.solve() is EmptySet
    # Solve a disjunction with complement.
    event = (Y << {'a', 'b'}) & ~(Y << {'c'})
    assert event.solve() == NominalSet('a', 'b')
    # Solve a disjunction with complement.
    event = (Y << {'a', 'b'}) | ~(Y << {'c'})
    assert event.solve() == sympy.Complement(UniversalSet, NominalSet('c'))
    # Union of interval and symbolic.
    event = (Y**2 <= 9) | (Y << {'a'})
    assert event.solve() == sympy.Union(
        sympy.Interval(-3, 3),
        NominalSet('a'))
    # Union of interval and not symbolic.
    event = (Y**2 <= 9) | ~(Y << {'a'})
    assert event.solve() == sympy.Union(
        sympy.Interval(-3, 3),
        sympy.Complement(UniversalSet, NominalSet('a')))
    # Intersection of interval and symbolic.
    event = (Y**2 <= 9) & (Y << {'a'})
    assert event.solve() is EmptySet
    # Intersection of interval and not symbolic.
    event = (Y**2 <= 9) & ~(Y << {'a'})
    assert event.solve() == sympy.Complement(
        sympy.Interval(-3, 3),
        NominalSet('a'))
