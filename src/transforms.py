# Copyright 2019 MIT Probabilistic Computing Project.
# See LICENSE.txt

from itertools import chain
from math import isinf

import sympy

from sympy import oo
from sympy.abc import X as symX

from sympy.calculus.util import function_range
from sympy.calculus.util import limit

from .events import EventFinite
from .events import EventInterval

from .math_util import isinf_neg
from .math_util import isinf_pos

from .poly import solve_poly_equality
from .poly import solve_poly_inequality

from .sym_util import EmptySet
from .sym_util import ExtReals
from .sym_util import ExtRealsPos
from .sym_util import Reals

from .sym_util import ContainersFinite
from .sym_util import sympify_number

# ==============================================================================
# Custom invertible function language.

class Transform(object):
    def symbol(self):
        # pylint: disable=no-member
        return self.subexpr.symbol()
    def domain(self):
        raise NotImplementedError()
    def range(self):
        raise NotImplementedError()
    def ffwd(self, x):
        raise NotImplementedError()
    def finv(self, x):
        raise NotImplementedError()
    def evaluate(self, x):
        # pylint: disable=no-member
        y = self.subexpr.evaluate(x)
        return self.ffwd(y)
    def invert(self, x):
        intersection = sympy.Intersection(self.range(), x)
        if intersection is EmptySet:
            return EmptySet
        if isinstance(intersection, ContainersFinite):
            return self.invert_finite(intersection)
        if isinstance(intersection, sympy.Interval):
            return self.invert_interval(intersection)
        if isinstance(intersection, sympy.Union):
            intervals = [self.invert(y) for y in intersection.args]
            return sympy.Union(*intervals)
        assert False, 'Unknown intersection: %s' % (intersection,)
    def invert_finite(self, values):
        raise NotImplementedError()
    def invert_interval(self, interval):
        # Should be called on subset of range.
        raise NotImplementedError()
    # Addition.
    def __add__(self, x):
        poly_self = polyify(self)
        # Is x a constant?
        try:
            x_val = sympify_number(x)
            coeffs_new = list(poly_self.coeffs)
            coeffs_new[0] += x_val
            return Poly(poly_self.subexpr, coeffs_new)
        except NotImplementedError:
            pass
        # Add polynomial terms only if subexpressions match.
        poly_x = polyify(x)
        if poly_x.subexpr != poly_self.subexpr:
            raise NotImplementedError('Invalid addition %s + %s' % (self, x))
        return poly_add(poly_self, poly_x)
    def __radd__(self, x):
        return self + x
    # Subtraction.
    def __sub__(self, x):
        return self + (-1 * x)
    def __rsub__(self, x):
        return self - x
    # Multiplication.
    def __mul__(self, x):
        poly_self = polyify(self)
        # Is x a constant?
        try:
            x_val = sympify_number(x)
            coeffs = [x_val*c for c in poly_self.coeffs]
            return Poly(poly_self.subexpr, coeffs)
        except NotImplementedError:
            pass
        # Multiply polynomial terms only if subexpressions match.
        poly_x = polyify(x)
        if poly_x.subexpr != poly_self.subexpr:
            raise NotImplementedError('Invalid addition %s + %s' % (self, x))
        return poly_mul(poly_self, poly_x)
    def __rmul__(self, x):
        return self * x
    # Division.
    def __truediv__(self, x):
        x_val = sympify_number(x)
        return sympy.Rational(1, x_val) * self
    # Negation.
    def __neg__(self):
        return -1 * self
    # Absolute value.
    def __abs__(self):
        return Abs(self)
    # Exponentiation.
    def __pow__(self, x):
        x_val = sympify_number(x)
        if isinstance(x_val, sympy.Integer):
            return Pow(self, x_val)
        if isinstance(x_val, sympy.Rational):
            (numer, denom) = x_val.as_numer_denom()
            if numer != 1:
                raise NotImplementedError(
                    'Rational powers must be 1/n, not %s' % (x,))
            return Radical(self, denom)
        raise NotImplementedError(
            'Power must be rational or integer, not %s' % (x))
    def __rpow__(self, x):
        x_val = sympify_number(x)
        return Exp(self, x_val)
    # Comparison.
    def __le__(self, x):
        # self <= x
        x_val = sympify_number(x)
        interval = sympy.Interval(-oo, x_val)
        return EventInterval(self, interval)
    def __lt__(self, x):
        # self < x
        x_val = sympify_number(x)
        interval = sympy.Interval(-oo, x_val, right_open=True)
        return EventInterval(self, interval)
    def __ge__(self, x):
        # self >= x
        x_val = sympify_number(x)
        interval = sympy.Interval(x_val, oo)
        return EventInterval(self, interval)
    def __gt__(self, x):
        # self > x
        x_val = sympify_number(x)
        interval = sympy.Interval(x_val, oo, left_open=True)
        return EventInterval(self, interval)
    # Containment
    def __lshift__(self, x):
        if isinstance(x, ContainersFinite):
            return EventFinite(self, x)
        if isinstance(x, sympy.Interval):
            return EventInterval(self, x)
        raise NotImplementedError()

class Injective(Transform):
    # Injective (one-to-one) transforms.
    def invert_finite(self, values):
        # pylint: disable=no-member
        values_prime = {self.finv(x) for x in values}
        return self.subexpr.invert(values_prime)
    def invert_interval(self, interval):
        assert isinstance(interval, sympy.Interval)
        (a, b) = (interval.left, interval.right)
        a_prime = self.finv(a)
        b_prime = self.finv(b)
        interval_prime = transform_interval(interval, a_prime, b_prime)
        # pylint: disable=no-member
        return self.subexpr.invert(interval_prime)

class NonInjective(Transform):
    # Non-injective (many-to-one) transforms.
    def invert_finite(self, values):
        # pylint: disable=no-member
        values_prime_list = [self.finv(x) for x in values]
        values_prime = set(chain.from_iterable(values_prime_list))
        return self.subexpr.invert(values_prime)

class Identity(Injective):
    def __init__(self, symbol):
        assert isinstance(symbol, str)
        self.symb = symbol
    def symbol(self):
        return self
    def domain(self):
        return ExtReals
    def range(self):
        return ExtReals
    def ffwd(self, x):
        assert x in self.domain()
        return x
    def finv(self, x):
        if not x in self.range():
            return EmptySet
        return x
    def evaluate(self, x):
        assert x in self.domain()
        return x
    def invert_finite(self, values):
        return values
    def invert_interval(self, interval):
        return interval
    def __eq__(self, x):
        return isinstance(x, Identity) and self.symb == x.symb
    def __repr__(self):
        return 'Identity(%s)' % (repr(self.symb),)
    def __str__(self):
        return self.symb
    def __hash__(self):
        x = (self.__class__, self.symb)
        return hash(x)

class Abs(NonInjective):
    def __init__(self, subexpr):
        self.subexpr = make_subexpr(subexpr)
    def domain(self):
        return ExtReals
    def range(self):
        return ExtRealsPos
    def ffwd(self, x):
        assert x in self.domain()
        return x if x > 0 else -x
    def finv(self, x):
        if not x in self.range():
            return EmptySet
        return {x, -x}
    def invert_interval(self, interval):
        assert isinstance(interval, sympy.Interval)
        (a, b) = (interval.left, interval.right)
        interval_pos = transform_interval(interval, a, b)
        interval_neg = transform_interval(interval, -b, -a, switch=True)
        interval_inv = interval_pos + interval_neg
        return self.subexpr.invert(interval_inv)
    def __eq__(self, x):
        return isinstance(x, Abs) and self.subexpr == x.subexpr
    def __repr__(self):
        return 'Abs(%s)' % (repr(self.subexpr))
    def __str__(self):
        return '|%s|' % (str(self.subexpr),)
    def __hash__(self):
        x = (self.__class__, self.subexpr)
        return hash(x)
    def __abs__(self):
        return Abs(self.subexpr)

class Radical(Injective):
    def __init__(self, subexpr, degree):
        assert degree != 0
        self.subexpr = make_subexpr(subexpr)
        self.degree = degree
    def domain(self):
        return ExtRealsPos
    def range(self):
        return ExtRealsPos
    def ffwd(self, x):
        assert x in self.domain()
        return sympy.Pow(x, sympy.Rational(1, self.degree))
    def finv(self, x):
        if x not in self.range():
            return EmptySet
        return sympy.Pow(x, sympy.Rational(self.degree, 1))
    def __eq__(self, x):
        return isinstance(x, Radical) \
            and self.subexpr == x.subexpr \
            and self.degree == x.degree
    def __repr__(self):
        return 'Radical(degree=%s, %s)' \
            % (repr(self.degree), repr(self.subexpr))
    def __str__(self):
        return '(%s)**(1/%d)' % (str(self.subexpr), self.degree)
    def __hash__(self):
        x = (self.__class__, self.subexpr, self.degree)
        return hash(x)

class Exp(Injective):
    def __init__(self, subexpr, base):
        assert base > 0
        self.subexpr = make_subexpr(subexpr)
        self.base = base
    def domain(self):
        return ExtReals
    def range(self):
        return ExtRealsPos
    def ffwd(self, x):
        assert x in self.domain()
        return sympy.Pow(self.base, x)
    def finv(self, x):
        if not x in self.range():
            return EmptySet
        return sympy.log(x, self.base) if x > 0 else -oo
    def __eq__(self, x):
        return isinstance(x, Exp) \
            and self.subexpr == x.subexpr \
            and self.base == x.base
    def __repr__(self):
        return 'Exp(base=%s, %s)' \
            % (repr(self.base), repr(self.subexpr))
    def __str__(self):
        if self.base == sympy.E:
            return 'exp(%s)' % (str(self.subexpr),)
        return '%s**(%s)' % (self.base, str(self.subexpr))
    def __hash__(self):
        x = (self.__class__, self.subexpr, self.base)
        return hash(x)

class Log(Injective):
    def __init__(self, subexpr, base):
        assert base > 1
        self.subexpr = make_subexpr(subexpr)
        self.base = base
    def domain(self):
        return ExtRealsPos
    def range(self):
        return ExtReals
    def ffwd(self, x):
        assert x in self.domain()
        return sympy.log(x, self.base) if x > 0 else -oo
    def finv(self, x):
        if not x in self.range():
            return EmptySet
        return sympy.Pow(self.base, x)
    def __eq__(self, x):
        return isinstance(x, Log) \
            and self.subexpr == x.subexpr \
            and self.base == x.base
    def __repr__(self):
        return 'Log(base=%s, %s)' \
            % (repr(self.base), repr(self.subexpr))
    def __str__(self):
        if self.base == sympy.E:
            return 'ln(%s)' % (str(self.subexpr),)
        if self.base == 2:
            return 'log2(%s)' % (str(self.subexpr),)
        return 'log(%s; %s)' % (str(self.subexpr), self.base)
    def __hash__(self):
        x = (self.__class__, self.subexpr, self.base)
        return hash(x)

class Poly(NonInjective):
    def __init__(self, subexpr, coeffs):
        assert len(coeffs) > 1
        self.subexpr = make_subexpr(subexpr)
        self.coeffs = tuple(coeffs)
        self.degree = len(coeffs) - 1
        self.symexpr = make_sympy_polynomial(coeffs)
    def domain(self):
        return ExtReals
    def range(self):
        result = function_range(self.symexpr, symX, Reals)
        pos_inf = sympy.FiniteSet(oo) if result.right == oo else EmptySet
        neg_inf = sympy.FiniteSet(-oo) if result.left == -oo else EmptySet
        return sympy.Union(result, pos_inf, neg_inf)
    def ffwd(self, x):
        assert x in self.domain()
        return self.symexpr.subs(symX, x) \
            if not isinf(x) else limit(self.symexpr, symX, x)
    def finv(self, x):
        if not x in self.range():
            return EmptySet
        return solve_poly_equality(self.symexpr, x)
    def invert_interval(self, interval):
        assert isinstance(interval, sympy.Interval)
        (a, b) = (interval.left, interval.right)
        (lo, ro) = (not interval.left_open, interval.right_open)
        xvals_a = solve_poly_inequality(self.symexpr, a, lo, extended=False)
        xvals_b = solve_poly_inequality(self.symexpr, b, ro, extended=False)
        xvals = xvals_a.complement(xvals_b)
        return self.subexpr.invert(xvals)
    def __eq__(self, x):
        return isinstance(x, Poly) \
            and self.subexpr == x.subexpr \
            and self.coeffs == x.coeffs
    def __neg__(self):
        return Poly(self.subexpr, [-c for c in self.coeffs])
    def __repr__(self):
        return 'Poly(coeffs=%s, %s)' \
            % (repr(self.coeffs), repr(self.subexpr))
    def __str__(self):
        ss = str(self.subexpr)
        def make_term(i, c):
            if c == 0:
                return ''
            if i == 0:
                return str(c)
            if i == 1:
                return '%s(%s)' % (str(c), ss) if c != 1 else ss
            if i < len(self.coeffs):
                return '%s*(%s)**%d' % (str(c), ss, i) if c != 1 \
                    else '(%s)**%d' % (ss, i)
            assert False
        terms = [make_term(i, c)  for i, c in enumerate(self.coeffs)]
        return ' + '.join([t for t in terms if t])
    def __hash__(self):
        x = (self.__class__, self.subexpr, self.coeffs)
        return hash(x)

# Some useful constructors.
def ExpNat(subexpr):
    return Exp(subexpr, sympy.exp(1))
def LogNat(subexpr):
    return Log(subexpr, sympy.exp(1))
def Sqrt(subexpr):
    return Radical(subexpr, 2)
def Pow(subexpr, n):
    assert 0 <= n
    coeffs = [0]*n + [1]
    return Poly(subexpr, coeffs)

# ==============================================================================
# Utilities.

def transform_interval(interval, a, b, switch=None):
    return \
        sympy.Interval(a, b, interval.left_open, interval.right_open) \
        if not switch else \
        sympy.Interval(a, b, interval.right_open, interval.left_open) \

def make_sympy_polynomial(coeffs):
    terms = [c*symX**i for (i,c) in enumerate(coeffs)]
    return sympy.Add(*terms)

def make_subexpr(subexpr):
    if isinstance(subexpr, Transform):
        return subexpr
    assert False, 'Invalid subexpr: %s' % (subexpr,)

def polyify(expr):
    if isinstance(expr, Poly):
        return expr
    return Poly(expr, (0, 1))

def add_coeffs(a, b):
    length = max(len(a), len(b))
    a_prime = a + (0,) * (length - len(a))
    b_prime = b + (0,) * (length - len(b))
    assert len(a_prime) == len(b_prime)
    return [xa + xb for (xa, xb) in zip(a_prime, b_prime)]

def poly_add(poly_a, poly_b):
    assert poly_a.subexpr == poly_b.subexpr
    # Alternative implementation.
    # sym_poly_a = sympy.Poly(poly_a.symexpr)
    # sym_poly_b = sympy.Poly(poly_b.symexpr)
    # sym_poly_c = sym_poly_a + sym_poly_b
    # assert sym_poly_c.all_coeffs()[::-1] == coeffs
    coeffs = add_coeffs(poly_a.coeffs, poly_b.coeffs)
    return Poly(poly_a.subexpr, coeffs)

def poly_mul(poly_a, poly_b):
    assert poly_a.subexpr == poly_b.subexpr
    sym_poly_a = sympy.Poly(poly_a.symexpr, symX)
    sym_poly_b = sympy.Poly(poly_b.symexpr, symX)
    sym_poly_c = sym_poly_a * sym_poly_b
    coeffs = sym_poly_c.all_coeffs()[::-1]
    return Poly(poly_a.subexpr, coeffs)
