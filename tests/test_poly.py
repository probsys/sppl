# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from sympy import Poly as SymPoly
from sympy import Rational
from sympy import sqrt as SymSqrt
from sympy.abc import x

from sppl.poly import solve_poly_equality
from sppl.poly import solve_poly_inequality

from sppl.math_util import allclose

from sppl.sets import EmptySet
from sppl.sets import ExtReals
from sppl.sets import FiniteReal
from sppl.sets import Interval
from sppl.sets import Reals
from sppl.sets import Union
from sppl.sets import inf as oo

def test_solve_poly_inequaltiy_pos_inf():
    assert solve_poly_inequality(x**2-10*x+100, oo, True) == Reals
    assert solve_poly_inequality(x**2-10*x+100, oo, False) == ExtReals

    assert solve_poly_inequality(-x**3+10*x, oo, False) == ExtReals
    assert solve_poly_inequality(-x**3+10*x, oo, True) == Reals | FiniteReal(oo)

    assert solve_poly_inequality(x**3-10*x, oo, False) == ExtReals
    assert solve_poly_inequality(x**3-10*x, oo, True) == Reals | FiniteReal(-oo)


def test_solve_poly_inequaltiy_neg_inf():
    assert solve_poly_inequality(x**2-10*x+100, -oo, True) is EmptySet
    assert solve_poly_inequality(x**2-10*x+100, -oo, False) is EmptySet

    assert solve_poly_inequality(x**3-10*x, -oo, True) is EmptySet
    assert solve_poly_inequality(x**3-10*x, -oo, False) == FiniteReal(-oo)

    assert solve_poly_inequality(-x**2+10*x+100, -oo, True) is EmptySet
    assert solve_poly_inequality(-x**2+10*x+100, -oo, False) == FiniteReal(-oo, oo)


p_quadratic = SymPoly((x-SymSqrt(2)/10)*(x+Rational(10, 7)), x)
expr_quadratic = p_quadratic.args[0]
def test_solve_poly_equality_quadratic_zero():
    roots = solve_poly_equality(expr_quadratic, 0)
    assert roots == FiniteReal(SymSqrt(2)/10, -Rational(10,7))
def test_solve_poly_inequality_quadratic_zero():
    interval = solve_poly_inequality(expr_quadratic, 0, False)
    assert interval == Interval(-Rational(10,7), SymSqrt(2)/10)
    interval = solve_poly_inequality(expr_quadratic, 0, True)
    assert interval == Interval.open(-Rational(10,7), SymSqrt(2)/10)

xe1_quad0 = -5/7 + SymSqrt(2)/20 + SymSqrt(2)*SymSqrt(700*SymSqrt(2) + 14849)/140
xe1_quad1 = -SymSqrt(2)*SymSqrt(700*SymSqrt(2) + 14849)/140 - 5/7 + SymSqrt(2)/20
def test_solve_poly_equality_quadratic_one():
    roots = solve_poly_equality(expr_quadratic, 1)
    # SymPy is not smart enough to simplify irrational roots symbolically
    # so check numerical equality of the symbolic roots.
    assert len(roots) == 2
    assert any(allclose(float(x), float(xe1_quad0)) for x in roots)
    assert any(allclose(float(x), float(xe1_quad1)) for x in roots)


p_cubic_int = SymPoly((x-1)*(x+2)*(x-11), x)
expr_cubic_int = p_cubic_int.args[0]
def test_solve_poly_equality_cubic_int_zero():
    roots = solve_poly_equality(expr_cubic_int, 0)
    assert roots == FiniteReal(-2, 1, 11)
def test_solve_poly_inequality_cubic_int_zero():
    interval = solve_poly_inequality(expr_cubic_int, 0, False)
    assert interval == Interval(-oo, -2) | Interval(1, 11)
    interval = solve_poly_inequality(expr_cubic_int, 0, True)
    assert interval == Interval.open(-oo, -2) | Interval.open(1, 11)

xe1_cubic_int0 = -1.97408387376586
xe1_cubic_int1 = 0.966402009973818
xe1_cubic_int2 = 11.007681863792
def test_solve_poly_equality_cubic_int_one():
    roots = solve_poly_equality(expr_cubic_int, 1)
    assert len(roots) == 3
    assert any(allclose(float(x), xe1_cubic_int0) for x in roots)
    assert any(allclose(float(x), xe1_cubic_int1) for x in roots)
    assert any(allclose(float(x), xe1_cubic_int2) for x in roots)
def test_solve_poly_inequality_cubic_int_one():
    interval = solve_poly_inequality(expr_cubic_int, 1, True)
    assert isinstance(interval, Union)
    assert len(interval.args)==2
    # First interval.
    assert interval.args[0].left == -oo
    assert allclose(float(interval.args[0].right), xe1_cubic_int0)
    assert interval.args[0].right_open
    # Second interval.
    assert allclose(float(interval.args[1].left), xe1_cubic_int1)
    assert allclose(float(interval.args[1].right), xe1_cubic_int2)
    assert interval.args[1].left_open
    assert interval.args[1].right_open

    interval = solve_poly_inequality(-1*expr_cubic_int, -1, True)
    assert isinstance(interval, Union)
    assert len(interval.args) == 2
    # First interval.
    assert allclose(float(interval.args[0].left), xe1_cubic_int0)
    assert allclose(float(interval.args[0].right), xe1_cubic_int1)
    # Second interval.
    assert allclose(float(interval.args[1].left), xe1_cubic_int2)
    assert interval.args[1].right == oo


p_cubic_irrat = SymPoly((x-SymSqrt(2)/10)*(x+Rational(10, 7))*(x-SymSqrt(5)), x)
expr_cubic_irrat = p_cubic_irrat.args[0]
def test_solve_poly_equality_cubic_irrat_zero():
    roots = solve_poly_equality(expr_cubic_irrat, 0)
    # Confirm that roots contains symbolic elements (no timeout).
    assert -Rational(10,7) in roots
    # SymPy is not smart enough to simplify irrational roots symbolically
    # so check numerical equality of the symbolic roots.
    assert any(allclose(float(x), float(SymSqrt(2)/10)) for x in roots)
    assert any(allclose(float(x), float(SymSqrt(5))) for x in roots)

xe1_cubic_irrat0 = -1.21493150246058
xe1_cubic_irrat1 = -0.19158140952462
xe1_cubic_irrat2 = 2.35543081715088
def test_solve_poly_equality_cubic_irrat_one():
    # This expression is too slow to solve symbolically.
    # The 5s timeout will trigger a numerical approximation.
    roots = solve_poly_equality(expr_cubic_irrat, 1)
    assert len(roots) == 3
    assert any(allclose(float(x), xe1_cubic_irrat0) for x in roots)
    assert any(allclose(float(x), xe1_cubic_irrat1) for x in roots)
    assert any(allclose(float(x), xe1_cubic_irrat2) for x in roots)
