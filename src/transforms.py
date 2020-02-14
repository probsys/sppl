# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from itertools import chain
from itertools import product
from math import isinf

import sympy

from sympy import oo
from sympy.abc import X as symX

from sympy.calculus.util import function_range
from sympy.calculus.util import limit

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
# Transform base class.

class Transform(object):
    def symbols(self):
        raise NotImplementedError()
    def domain(self):
        raise NotImplementedError()
    def range(self):
        raise NotImplementedError()

    def evaluate(self, assignment):
        raise NotImplementedError()
    def ffwd(self, x):
        raise NotImplementedError()
    def finv(self, x):
        raise NotImplementedError()

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
    def __add__number(self, x):
        poly_self = polyify(self)
        x_val = sympify_number(x)
        coeffs_new = list(poly_self.coeffs)
        coeffs_new[0] += x_val
        return Poly(poly_self.subexpr, coeffs_new)
    def __add__poly(self, x):
        if not isinstance(x, Transform):
            raise TypeError
        poly_self = polyify(self)
        poly_x = polyify(x)
        if poly_x.subexpr != poly_self.subexpr:
            raise ValueError('Incompatible subexpressions in "%s + %s"'
                % (str(self), x))
        sym_poly_a = sympy.Poly(poly_self.symexpr)
        sym_poly_b = sympy.Poly(poly_x.symexpr)
        sym_poly_c = sym_poly_a + sym_poly_b
        coeffs = sym_poly_c.all_coeffs()[::-1]
        return Poly(poly_self.subexpr, coeffs)
    def __add__(self, x):
        # Try to add x as a number.
        try:
            return self.__add__number(x)
        except TypeError:
            pass
        # Try to add x as a polynomial.
        try:
            return self.__add__poly(x)
        except TypeError:
            pass
        # Failed.
        return NotImplemented
    def __radd__(self, x):
        # Prevent infinite recursion from polymorphism.
        if not isinstance(x, Transform):
            return self + x
        return NotImplemented

    # Multiplication.
    def __mul__number(self, x):
        poly_self = polyify(self)
        x_val = sympify_number(x)
        coeffs = [x_val*c for c in poly_self.coeffs]
        return Poly(poly_self.subexpr, coeffs)
    def __mul__poly(self, x):
        if not isinstance(x, Transform):
            raise TypeError
        poly_self = polyify(self)
        poly_x = polyify(x)
        if poly_x.subexpr != poly_self.subexpr:
            raise ValueError('Incompatible subexpressions in "%s * %s"'
                % (str(self), x))
        sym_poly_a = sympy.Poly(poly_self.symexpr, symX)
        sym_poly_b = sympy.Poly(poly_x.symexpr, symX)
        sym_poly_c = sym_poly_a * sym_poly_b
        coeffs = sym_poly_c.all_coeffs()[::-1]
        return Poly(poly_self.subexpr, coeffs)
    def __mul__(self, x):
        # Try to multiply x as a number.
        try:
            return self.__mul__number(x)
        except TypeError:
            pass
        # Try to multiply x as a polynomial.
        try:
            return self.__mul__poly(x)
        except TypeError:
            pass
        # Failed.
        return NotImplemented
    def __rmul__(self, x):
        # Prevent infinite recursion from polymorphism.
        if not isinstance(x, Transform):
            return self * x
        return NotImplemented

    # Division by x.
    def __truediv__number(self, x):
        x_val = sympify_number(x)
        return sympy.Rational(1, x_val) * self
    def __truediv__(self, x):
        # Try to divide by x as a number.
        try:
            return self.__truediv__number(x)
        except TypeError:
            pass
        # Failed.
        return NotImplemented

    # Division by self.
    def __rtruediv__number(self, x):
        x_val = sympify_number(x)
        return Reciprocal(self) if (x_val == 1) else x_val * Reciprocal(self)
    def __rtruediv__(self, x):
        # Try to divide by x as a number.
        try:
            return self.__rtruediv__number(x)
        except TypeError:
            pass
        # Failed.
        return NotImplemented

    # Subtraction.
    def __sub__(self, x):
        return self + (-1 * x)
    def __rsub__(self, x):
        return self - x
    # Negation.
    def __neg__(self):
        return -1 * self
    # Absolute value.
    def __abs__(self):
        return Abs(self)

    # Power (self**x).
    def __pow__integer(self, x):
        if 0 < x:
            return Pow(self, x)
        if x < 0:
            return 1 / Pow(self, -x)
        raise ValueError('Cannot raise %s to %s' % (str(self), x))
    def __pow__rational(self, x):
        (numer, denom) = x.as_numer_denom()
        if numer == 1:
            return Radical(self, denom)
        if numer == -1:
            return 1 / Radical(self, denom)
        # TODO: Consider default choice x**(a/b) = (x**(a))**(1/b)
        return NotImplemented
    def __pow__number(self, x):
        x_val = sympify_number(x)
        if isinstance(x_val, sympy.Integer):
            return self.__pow__integer(x_val)
        if isinstance(x_val, sympy.Rational):
            return self.__pow__rational(x_val)
        # TODO: Convert floating-point power to rational.
        # if isinstance(x_val, sympy.Float):
        #     x_val_rat = sympy.Rational(x_val)
        #     return self.__pow__rational(x_val_rat)
        raise ValueError(
            'Cannot raise %s to irrational or floating-point power %s'
            % (str(self), x))
    def __pow__(self, x):
        # Try to raise to x as a number.
        try:
            return self.__pow__number(x)
        except TypeError:
            pass
        # Failed.
        return NotImplemented

    # Exponentiation (x**self)
    def __rpow__number(self, x):
        x_val = sympify_number(x)
        if x_val <= 0:
            raise ValueError('Base must be positive, not %s' % (x,))
        return Exp(self, x_val)
    def __rpow__(self, x):
        try:
            return self.__rpow__number(x)
        except TypeError:
            pass
        # Failed to exponentiate.
        return NotImplemented

    # Comparison.
    # TODO: Considering (X < Y) to mean (X - Y < 0)
    # Complication: X and Y may not have a natural ordering.
    def __le__(self, x):
        # self <= x
        try:
            x_val = sympify_number(x)
            interval = sympy.Interval(-oo, x_val)
            return EventInterval(self, interval)
        except TypeError:
            return NotImplemented
    def __lt__(self, x):
        # self < x
        try:
            x_val = sympify_number(x)
            interval = sympy.Interval(-oo, x_val, right_open=True)
            return EventInterval(self, interval)
        except TypeError:
            return NotImplemented
    def __ge__(self, x):
        # self >= x
        try:
            x_val = sympify_number(x)
            interval = sympy.Interval(x_val, oo)
            return EventInterval(self, interval)
        except TypeError:
            return NotImplemented
    def __gt__(self, x):
        # self > x
        try:
            x_val = sympify_number(x)
            interval = sympy.Interval(x_val, oo, left_open=True)
            return EventInterval(self, interval)
        except TypeError:
            return NotImplemented

    # Containment
    def __lshift__(self, x):
        if isinstance(x, ContainersFinite):
            return EventFinite(self, x)
        if isinstance(x, sympy.Interval):
            return EventInterval(self, x)
        return NotImplemented

# ==============================================================================
# Injective transforms.

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

class Identity(Injective):
    def __init__(self, token):
        assert isinstance(token, str)
        self.token = token
    def symbols(self):
        return (self,)
    def domain(self):
        return ExtReals
    def range(self):
        return ExtReals
    def evaluate(self, assignment):
        if self not in assignment:
            raise ValueError('Cannot evaluate %s on %s' % (str(self), assignment))
        return self.ffwd(assignment[self])
    def ffwd(self, x):
        # assert x in self.domain()
        return x
    def finv(self, x):
        if not x in self.range():
            return EmptySet
        return x
    def invert_finite(self, values):
        return values
    def invert_interval(self, interval):
        return interval
    def __eq__(self, x):
        return isinstance(x, Identity) and self.token == x.token
    def __repr__(self):
        return 'Identity(%s)' % (repr(self.token),)
    def __str__(self):
        return self.token
    def __hash__(self):
        x = (self.__class__, self.token)
        return hash(x)

class Radical(Injective):
    def __init__(self, subexpr, degree):
        assert degree != 0
        self.subexpr = make_subexpr(subexpr)
        self.degree = degree
    def symbols(self):
        return self.subexpr.symbols()
    def domain(self):
        return ExtRealsPos
    def range(self):
        return ExtRealsPos
    def evaluate(self, assignment):
        y = self.subexpr.evaluate(assignment)
        return self.ffwd(y)
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
    def symbols(self):
        return self.subexpr.symbols()
    def domain(self):
        return ExtReals
    def range(self):
        return ExtRealsPos
    def evaluate(self, assignment):
        y = self.subexpr.evaluate(assignment)
        return self.ffwd(y)
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
    def symbols(self):
        return self.subexpr.symbols()
    def domain(self):
        return ExtRealsPos
    def range(self):
        return ExtReals
    def evaluate(self, assignment):
        y = self.subexpr.evaluate(assignment)
        return self.ffwd(y)
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

# ==============================================================================
# Non Injective transforms.

class NonInjective(Transform):
    # Non-injective (many-to-one) transforms.
    def invert_finite(self, values):
        # pylint: disable=no-member
        values_prime_list = [self.finv(x) for x in values]
        values_prime = sympy.Union(*values_prime_list)
        return self.subexpr.invert(values_prime)
    def invert_interval(self, interval):
        # Should be called on subset of range.
        raise NotImplementedError()

class Abs(NonInjective):
    def __init__(self, subexpr):
        self.subexpr = make_subexpr(subexpr)
    def symbols(self):
        return self.subexpr.symbols()
    def domain(self):
        return ExtReals
    def range(self):
        return ExtRealsPos
    def evaluate(self, assignment):
        y = self.subexpr.evaluate(assignment)
        return self.ffwd(y)
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
        interval_neg = transform_interval(interval, -b, -a, flip=True)
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

class Reciprocal(NonInjective):
    def __init__(self, subexpr):
        self.subexpr = make_subexpr(subexpr)
    def symbols(self):
        return self.subexpr.symbols()
    def domain(self):
        return ExtReals - sympy.FiniteSet(0)
    def range(self):
        return Reals - sympy.FiniteSet(0)
    def evaluate(self, assignment):
        y = self.subexpr.evaluate(assignment)
        return self.ffwd(y)
    def ffwd(self, x):
        assert x in self.domain()
        return 0 if isinf(x) else sympy.Rational(1, x)
    def finv(self, x):
        if x not in self.range():
            return EmptySet
        if x == 0:
            return {-oo, oo}
        return {sympy.Rational(1, x)}
    def invert_interval(self, interval):
        (a, b) = (interval.left, interval.right)
        if (0 <= a < b):
            assert 0 < a or interval.left_open
            a_inv = sympy.Rational(1, a) if 0 < a else oo
            b_inv = sympy.Rational(1, b) if (not isinf(b)) else 0
            interval_inv = transform_interval(interval, b_inv, a_inv, flip=True)
            return self.subexpr.invert(interval_inv)
        if (a < b <= 0):
            assert b < 0 or interval.right_open
            a_inv = sympy.Rational(1, a) if (not isinf(a)) else 0
            b_inv = sympy.Rational(1, b) if b < 0 else -oo
            interval_inv = transform_interval(interval, b_inv, a_inv, flip=True)
            return self.subexpr.invert(interval_inv)
        assert False, 'Impossible Reciprocal interval: %s ' % (interval,)
    def __eq__(self, x):
        return isinstance(x, Reciprocal) \
            and self.subexpr == x.subexpr
    def __repr__(self):
        return 'Reciprocal(%s)' % (repr(self.subexpr),)
    def __str__(self):
        return '(1/%s)' % (str(self.subexpr),)
    def __hash__(self):
        x = (self.__class__, self.subexpr)
        return hash(x)

class Poly(NonInjective):
    def __init__(self, subexpr, coeffs):
        assert len(coeffs) > 1
        self.subexpr = make_subexpr(subexpr)
        self.coeffs = tuple(coeffs)
        self.degree = len(coeffs) - 1
        self.symexpr = make_sympy_polynomial(coeffs)
    def symbols(self):
        return self.subexpr.symbols()
    def domain(self):
        return ExtReals
    def range(self):
        result = function_range(self.symexpr, symX, Reals)
        pos_inf = sympy.FiniteSet(oo) if result.right == oo else EmptySet
        neg_inf = sympy.FiniteSet(-oo) if result.left == -oo else EmptySet
        return sympy.Union(result, pos_inf, neg_inf)
    def evaluate(self, assignment):
        y = self.subexpr.evaluate(assignment)
        return self.ffwd(y)
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
        def term_to_str(i, c):
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
        terms = [term_to_str(i, c)  for i, c in enumerate(self.coeffs)]
        return ' + '.join([t for t in terms if t])
    def __hash__(self):
        x = (self.__class__, self.subexpr, self.coeffs)
        return hash(x)

# ==============================================================================
# Event is  Boolean-valued (non-injective) transform.

class Event(Transform):
    def range(self):
        return sympy.FiniteSet(0, 1)

    # Event methods.
    def solve(self):
        if len(self.symbols()) > 1:
            raise ValueError('Cannot solve multi-symbol Event.')
        return self.invert({1})
    def to_dnf(self):
        dnf = self.to_dnf_list()
        simplify_event = lambda x, E: x[0] if len(x)==1 else E(x)
        events = [simplify_event(conjunction, EventAnd) for conjunction in dnf]
        return simplify_event(events, EventOr)
    def to_dnf_list(self):
        raise NotImplementedError()
    def __and__(self, event):
        # Naive implementation (no simplification):
        # return EventAnd([self, event])
        raise NotImplementedError()
    def __or__(self, event):
        raise NotImplementedError()
        # Naive implementation (no simplification):
        # return EventOr([self, event])

class EventBasic(Event):
    def __init__(self, subexpr, values, complement=False):
        assert isinstance(values, (sympy.Interval,) + ContainersFinite)
        self.values = values
        self.subexpr = subexpr
        self.complement = complement
    def symbols(self):
        return self.subexpr.symbols()
    def domain(self):
        return self.subexpr.domain()
    def evaluate(self, assignment):
        y = self.subexpr.evaluate(assignment)
        return self.ffwd(y)
    def ffwd(self, x):
        # In    Complement      Result
        # T     F               T
        # F     F               F
        # T     T               F
        # F     T               T
        return (x in self.values) ^ bool(self.complement)
    def finv(self, x):
        if x not in self.range():
            return EmptySet
        if x == 1:
            if not self.complement:
                return self.values
            else:
                return sympy.Complement(Reals, sympy.sympify(self.values))
        if x == 0:
            if not self.complement:
                return sympy.Complement(Reals, sympy.sympify(self.values))
            else:
                return self.values
        assert False, 'Impossible value %s.'
    def invert_finite(self, values):
        values_prime_list = [self.finv(x) for x in values]
        values_prime = sympy.Union(*values_prime_list)
        return self.subexpr.invert(values_prime)

    # Event methods.
    def to_dnf_list(self):
        return [[self]]
    def __and__(self, event):
        if isinstance(event, EventAnd):
            events = (self,) + event.subexprs
            return EventAnd(events)
        if isinstance(event, (EventBasic, EventOr)):
            return EventAnd([self, event])
        return NotImplemented
    def __or__(self, event):
        if isinstance(event, EventOr):
            events = (self,) + event.subexprs
            return EventOr(events)
        if isinstance(event, (EventBasic, EventAnd)):
            return EventOr([self, event])
        return NotImplemented
    def __eq__(self, event):
        return isinstance(event, type(self)) \
            and (self.values == event.values) \
            and (self.subexpr == event.subexpr) \
            and (self.complement == event.complement)

class EventInterval(EventBasic):
    def __compute_gte__(self, x, left_open):
        # x < (Y < b)
        if not isinf_neg(self.values.left):
            raise ValueError('cannot compute %s < %s' % (x, str(self)))
        if self.complement:
            raise ValueError('cannot compute < with complement')
        xn = sympify_number(x)
        interval = sympy.Interval(xn, self.values.right,
            left_open=left_open, right_open=self.values.right_open)
        return EventInterval(self.subexpr, interval, complement=self.complement)
    def __compute_lte__(self, x, right_open):
        # (a < Y) < x
        if not isinf_pos(self.values.right):
            raise ValueError('cannot compute %s < %s' % (str(self), x))
        if self.complement:
            raise ValueError('cannot compute < with complement')
        xn = sympify_number(x)
        interval = sympy.Interval(self.values.left, xn,
            left_open=self.values.left_open, right_open=right_open)
        return EventInterval(self.subexpr, interval, complement=self.complement)
    def __gt__(self, x):
        return self.__compute_gte__(x, True)
    def __ge__(self, x):
        return self.__compute_gte__(x, False)
    def __lt__(self, x):
        return self.__compute_lte__(x, True)
    def __le__(self, x):
        return self.__compute_lte__(x, False)
    def __invert__(self):
        return EventInterval(self.subexpr, self.values, not self.complement)
    def __repr__(self):
        return 'EventInterval(%s, %s, complement=%s)' \
            % (repr(self.subexpr), repr(self.values), repr(self.complement))
    def __str__(self):
        sym = str(self.subexpr)
        comp_l = '<' if self.values.left_open else '<='
        (x_l, x_r) = (self.values.left, self.values.right)
        comp_r = '<' if self.values.right_open else '<='
        if isinf_neg(x_l):
            result = '%s %s %s' % (sym, comp_r, x_r)
        elif isinf_pos(x_r):
            result = '%s %s %s' % (x_l, comp_l, sym)
        else:
            result = '%s %s %s %s %s' % (x_l, comp_l, sym, comp_r, x_r)
        return result if not self.complement else '~(%s)' % (result,)

class EventFinite(EventBasic):
    # TODO: Consider allowing 0 < (X << {1, 2})
    # treating the RHS as a Boolean-valued function.
    def __gt__(self, x):
        raise TypeError()
    def __ge__(self, x):
        raise TypeError()
    def __lt__(self, x):
        raise TypeError()
    def __le__(self, x):
        raise TypeError()

    def __repr__(self):
        return 'EventFinite(%s, %s, complement=%s)' \
            % (repr(self.subexpr), repr(self.values), repr(self.complement))
    def __str__(self):
        return '%s << %s' % (str(self.subexpr), str(self.values))
    def __invert__(self):
        return EventFinite(self.subexpr, self.values, not self.complement)

class EventCompound(Event):
    def __init__(self, subexprs):
        assert all(isinstance(s, Event) for s in subexprs)
        self.subexprs = tuple(subexprs)
    def symbols(self):
        return tuple(set(chain.from_iterable([
            event.symbols() for event in self.subexprs])))
    def domain(self):
        if len(self.symbols()) > 1:
            raise ValueError('No domain for multi-symbol Event.')
        domains = [event.domain() for event in self.subexprs]
        return sympy.Intersection(*domains)

class EventOr(EventCompound):
    def evaluate(self, assignment):
        ys = [event.evaluate(assignment) for event in self.subexprs]
        return any(ys)
    def ffwd(self, x):
        ys = [event.ffwd(x) for event in self.subexprs]
        return any(ys)
    def finv(self, x):
        ys = [event.finv(x) for event in self.subexprs]
        return sympy.Union(*ys)
    def invert_finite(self, values):
        solutions = [event.invert(values) for event in self.subexprs]
        return sympy.Union(*solutions)

    # Event methods.
    def to_dnf_list(self):
        sub_dnf = [event.to_dnf_list() for event in self.subexprs]
        return list(chain.from_iterable(sub_dnf))
    def __and__(self, event):
        if isinstance(event, EventAnd):
            events = (self,) + event.subexprs
            return EventAnd(events)
        if isinstance(event, (EventBasic, EventOr)):
            events = (self, event)
            return EventAnd(events)
        return NotImplemented
    def __or__(self, event):
        if isinstance(event, EventOr):
            events = self.subexprs + event.subexprs
            return EventOr(events)
        if isinstance(event, (EventBasic, EventAnd)):
            events = self.subexprs + (event,)
            return EventOr(events)
        return NotImplemented
    def __eq__(self, event):
        return isinstance(event, EventOr) and (self.subexprs == event.subexprs)
    def __invert__(self):
        sub_events = [~event for event in self.subexprs]
        return EventAnd(sub_events)
    def __repr__(self):
        return 'EventOr(%s)' % (repr(self.subexprs,))
    def __str__(self):
        sub_events = ['(%s)' % (str(event),) for event in self.subexprs]
        return ' | '.join(sub_events)
    def __hash__(self):
        x = (self.__class__, self.subexprs)
        return hash(x)

class EventAnd(EventCompound):
    def evaluate(self, assignment):
        ys = [event.evaluate(assignment) for event in self.subexprs]
        return all(ys)
    def ffwd(self, x):
        ys = [event.ffwd(x) for event in self.subexprs]
        return all(ys)
    def finv(self, x):
        ys = [event.finv(x) for event in self.subexprs]
        return sympy.Intersection(*ys)
    def invert_finite(self, values):
        solutions = [event.invert(values) for event in self.subexprs]
        return sympy.Intersection(*solutions)

    # Event methods.
    def to_dnf_list(self):
        sub_dnf = [event.to_dnf_list() for event in self.subexprs]
        return [
            list(chain.from_iterable(cross))
            for cross in product(*sub_dnf)
        ]
    def __and__(self, event):
        if isinstance(event, EventAnd):
            events = self.subexprs + event.subexprs
            return EventAnd(events)
        if isinstance(event, (EventBasic, EventOr)):
            events = self.subexprs + (event,)
            return EventAnd(events)
        return NotImplemented
    def __or__(self, event):
        if isinstance(event, EventOr):
            events = (self,) + event.subexprs
            return EventOr(events)
        if isinstance(event, (EventBasic, EventAnd)):
            events = (self, event)
            return EventOr(events)
        return NotImplemented
    def __eq__(self, event):
        return isinstance(event, EventAnd) and (self.subexprs == event.subexprs)
    def __invert__(self):
        sub_events = [~event for event in self.subexprs]
        return EventOr(sub_events)
    def __repr__(self):
        return 'EventAnd(%s)' % (repr(self.subexprs,))
    def __str__(self):
        sub_events = ['(%s)' % (str(event),) for event in self.subexprs]
        return ' & '.join(sub_events)
    def __hash__(self):
        x = (self.__class__, self.subexprs)
        return hash(x)

# ==============================================================================
# Utilities.

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

def transform_interval(interval, a, b, flip=None):
    return \
        sympy.Interval(a, b, interval.left_open, interval.right_open) \
        if not flip else \
        sympy.Interval(a, b, interval.right_open, interval.left_open) \

def make_sympy_polynomial(coeffs):
    terms = [c*symX**i for (i,c) in enumerate(coeffs)]
    return sympy.Add(*terms)

def make_subexpr(subexpr):
    if isinstance(subexpr, Transform):
        return subexpr
    raise TypeError('Invalid subexpr %s with type %s'
        % (subexpr, str(type(subexpr))))

def polyify(expr):
    if isinstance(expr, Poly):
        return expr
    return Poly(expr, (0, 1))
