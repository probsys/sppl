# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from collections import Callable
from functools import reduce
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
from .sym_util import UniversalSet
from .sym_util import get_union

from .sym_util import ContainersFinite
from .sym_util import NominalSet
from .sym_util import complement_nominal_set
from .sym_util import is_nominal
from .sym_util import is_nominal_set
from .sym_util import is_number
from .sym_util import is_numeric_set
from .sym_util import sympify_nominal
from .sym_util import sympify_number

# ==============================================================================
# Transform base class.

class Transform(object):
    subexpr = None
    symbols = None

    def get_symbols(self):
        return self.symbols
    def domain(self):
        raise NotImplementedError()
    def range(self):
        raise NotImplementedError()

    def substitute(self, env):
        # TODO: Reject following cases, which result in infinite loop:
        # (1) env[X] = f(X)                 // check for self-reference
        # (2) env[X] = f(Z); env[Z] = g(X)  // detect cycle in digraph
        event_prime = self
        while expr_in_env(event_prime, env):
            event_prime = event_prime.subs(env)
        return event_prime
    def subs(self, env):
        raise NotImplementedError()

    def evaluate(self, assignment):
        raise NotImplementedError()
    def ffwd(self, x):
        raise NotImplementedError()
    def finv(self, y):
        raise NotImplementedError()

    def invert(self, ys):
        intersection = sympy.Intersection(self.range(), ys)
        if intersection is EmptySet:
            return EmptySet
        if isinstance(intersection, ContainersFinite):
            return self.invert_finite(intersection)
        if isinstance(intersection, sympy.Interval):
            return self.invert_interval(intersection)
        if isinstance(intersection, sympy.Union):
            xs_list = [self.invert(ys_i) for ys_i in intersection.args]
            return sympy.Union(*xs_list)
        if isinstance(intersection, sympy.Complement):
            (A, B) = intersection.args
            return self.invert(A - sympy.Intersection(A, B))
        assert False, 'Unknown intersection: %s' % (intersection,)
    def invert_finite(self, ys):
        raise NotImplementedError()
    def invert_interval(self, ys):
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
    def __mul__event(self, x):
        if not isinstance(x, Event):
            raise TypeError
        return Piecewise((self,), (x,))
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
        # Try to multiply x as an event.
        try:
            return self.__mul__event(x)
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
        if denom == 1:
            return self.__pow__integer(numer)
        # TODO: Consider default choice x**(a/b) = (x**(a))**(1/b)
        raise ValueError('Cannot raise %s to %s' % (str(self), x))
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
    def __pow__tuple(self, x):
        if not isinstance(x, tuple):
            raise TypeError
        numer = sympify_number(x[0])
        denom = sympify_number(x[1])
        return self.__pow__rational(sympy.Rational(numer, denom))
    def __pow__(self, x):
        # Try to raise to x as a number.
        try:
            return self.__pow__number(x)
        except TypeError:
            pass
        # Try to raise to x as a tuple.
        try:
            return self.__pow__tuple(x)
        except TypeError:
            pass
        # Failed.
        return NotImplemented

    # Exponentiation (x**self)
    def __rpow__number(self, x):
        x_val = sympify_number(x)
        if x_val <= 0:
            raise ValueError('Base must be positive, not %s' % (x,))
        return Exponential(self, x_val)
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
            values = list(x)
            values_num = [v for v in values if is_number(v)]
            values_str = [sympify_nominal(v) for v in values if is_nominal(v)]
            if len(values_num) + len(values_str) != len(values):
                raise ValueError('Only numeric or symbolic values, not %s'
                    % (str(x,)))
            if values_num and values_str:
                event_num = EventFiniteReal(self, sympy.FiniteSet(*values_num))
                event_str = EventFiniteNominal(self, NominalSet(*values_str))
                return EventOr([event_num, event_str])
            if values_num:
                return EventFiniteReal(self, sympy.FiniteSet(*values_num))
            if values_str:
                return EventFiniteNominal(self, NominalSet(*values_str))
            assert len(values) == 0
            return EventFiniteReal(self, values)
        if isinstance(x, sympy.Interval):
            return EventInterval(self, x)
        if isinstance(x, sympy.Complement):
            assert is_nominal_set(x.args[1])
            if x.args[0] is UniversalSet:
                return ~(self << x.args[1])
            if not is_nominal_set(x.args[0]):
                return self << x.args[0]
            assert False, 'Impossible Complement: %s' % (x,)
        if isinstance(x, sympy.Union):
            events = [self << y for y in x.args]
            return reduce(lambda state, event: state | event, events)
        return NotImplemented

# ==============================================================================
# Injective (one-to-one) Transforms.

class Injective(Transform):
    def invert_finite(self, ys):
        ys_prime = sympy.Union(*[self.finv(y) for y in ys])
        return self.subexpr.invert(ys_prime)
    def invert_interval(self, ys):
        assert isinstance(ys, sympy.Interval)
        (a, b) = (ys.left, ys.right)
        a_prime = next(iter(self.finv(a)))
        b_prime = next(iter(self.finv(b)))
        ys_prime = transform_interval(ys, a_prime, b_prime)
        return self.subexpr.invert(ys_prime)

class Identity(Injective):
    def __init__(self, token):
        assert isinstance(token, str)
        self.subexpr = self
        self.token = token
        self.symbols = frozenset({self})
    def domain(self):
        return ExtReals
    def range(self):
        return ExtReals
    def subs(self, env):
        if not expr_in_env(self, env):
            return self
        return env[self]
    def evaluate(self, assignment):
        if self not in assignment:
            raise ValueError('Cannot evaluate %s on %s' % (str(self), assignment))
        return self.ffwd(assignment[self])
    def ffwd(self, x):
        # assert x in self.domain()
        return x
    def finv(self, y):
        if not y in self.range():
            return EmptySet
        return {y}
    def invert_finite(self, ys):
        return ys
    def invert_interval(self, ys):
        return ys
    def __eq__(self, x):
        return isinstance(x, Identity) and self.token == x.token
    def __repr__(self):
        return 'Identity(%s)' % (repr(self.token),)
    def __str__(self):
        return self.token
    def __hash__(self):
        x = (self.__class__, self.token)
        return hash(x)
    def __rshift__(self, f):
        if isinstance(f, Callable):
            return f(self)
        if isinstance(f, dict):
            from .spn import NominalLeaf
            return NominalLeaf(self, f)
        return NotImplemented

class Radical(Injective):
    def __init__(self, subexpr, degree):
        assert degree != 0
        self.subexpr = make_subexpr(subexpr, self)
        self.degree = degree
    def domain(self):
        return ExtRealsPos
    def range(self):
        return ExtRealsPos
    def subs(self, env):
        if not expr_in_env(self, env):
            return self
        subexpr_prime = self.subexpr.subs(env)
        return Radical(subexpr_prime, self.degree)
    def evaluate(self, assignment):
        x = self.subexpr.evaluate(assignment)
        return self.ffwd(x)
    def ffwd(self, x):
        assert x in self.domain()
        return sympy.Pow(x, sympy.Rational(1, self.degree))
    def finv(self, y):
        if y not in self.range():
            return EmptySet
        return {sympy.Pow(y, sympy.Rational(self.degree, 1))}
    def __eq__(self, x):
        return isinstance(x, Radical) \
            and self.subexpr == x.subexpr \
            and self.degree == x.degree
    def __repr__(self):
        return 'Radical(degree=%s, %s)' \
            % (repr(self.degree), repr(self.subexpr))
    def __str__(self):
        return '(%s)**(1, %d)' % (str(self.subexpr), self.degree)
    def __hash__(self):
        x = (self.__class__, self.subexpr, self.degree)
        return hash(x)

class Exponential(Injective):
    def __init__(self, subexpr, base):
        assert base > 0
        self.subexpr = make_subexpr(subexpr, self)
        self.base = base
    def domain(self):
        return ExtReals
    def range(self):
        return ExtRealsPos
    def subs(self, env):
        if not expr_in_env(self, env):
            return self
        subexpr_prime = self.subexpr.subs(env)
        return Exponential(subexpr_prime, self.base)
    def evaluate(self, assignment):
        x = self.subexpr.evaluate(assignment)
        return self.ffwd(x)
    def ffwd(self, x):
        assert x in self.domain()
        return sympy.Pow(self.base, x)
    def finv(self, y):
        if not y in self.range():
            return EmptySet
        return {sympy.log(y, self.base) if y > 0 else -oo}
    def __eq__(self, x):
        return isinstance(x, Exponential) \
            and self.subexpr == x.subexpr \
            and self.base == x.base
    def __repr__(self):
        return 'Exponential(base=%s, %s)' \
            % (repr(self.base), repr(self.subexpr))
    def __str__(self):
        if self.base == sympy.E:
            return 'exp(%s)' % (str(self.subexpr),)
        return '%s**(%s)' % (self.base, str(self.subexpr))
    def __hash__(self):
        x = (self.__class__, self.subexpr, self.base)
        return hash(x)

class Logarithm(Injective):
    def __init__(self, subexpr, base):
        assert base > 1
        self.subexpr = make_subexpr(subexpr, self)
        self.base = base
    def domain(self):
        return ExtRealsPos
    def range(self):
        return ExtReals
    def subs(self, env):
        if not expr_in_env(self, env):
            return self
        subexpr_prime = self.subexpr.subs(env)
        return Logarithm(subexpr_prime, self.base)
    def evaluate(self, assignment):
        x = self.subexpr.evaluate(assignment)
        return self.ffwd(x)
    def ffwd(self, x):
        assert x in self.domain()
        return {sympy.log(x, self.base) if x > 0 else -oo}
    def finv(self, y):
        if not y in self.range():
            return EmptySet
        return {sympy.Pow(self.base, y)}
    def __eq__(self, x):
        return isinstance(x, Logarithm) \
            and self.subexpr == x.subexpr \
            and self.base == x.base
    def __repr__(self):
        return 'Logarithm(base=%s, %s)' \
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
# Non-injective real-valued Transforms.

class Abs(Transform):
    def __init__(self, subexpr):
        self.subexpr = make_subexpr(subexpr, self)
    def domain(self):
        return ExtReals
    def range(self):
        return ExtRealsPos
    def subs(self, env):
        if not expr_in_env(self, env):
            return self
        subexpr_prime = self.subexpr.subs(env)
        return Abs(subexpr_prime)
    def evaluate(self, assignment):
        x = self.subexpr.evaluate(assignment)
        return self.ffwd(x)
    def ffwd(self, x):
        assert x in self.domain()
        return x if x > 0 else -x
    def finv(self, y):
        if not y in self.range():
            return EmptySet
        return {y, -y}
    def invert_finite(self, ys):
        ys_prime = sympy.Union(*[self.finv(y) for y in ys])
        return self.subexpr.invert(ys_prime)
    def invert_interval(self, ys):
        assert isinstance(ys, sympy.Interval)
        (a, b) = (ys.left, ys.right)
        ys_pos = transform_interval(ys, a, b)
        ys_neg = transform_interval(ys, -b, -a, flip=True)
        ys_prime = ys_pos + ys_neg
        return self.subexpr.invert(ys_prime)
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

class Reciprocal(Transform):
    def __init__(self, subexpr):
        self.subexpr = make_subexpr(subexpr, self)
    def domain(self):
        return ExtReals - sympy.FiniteSet(0)
    def range(self):
        return Reals - sympy.FiniteSet(0)
    def subs(self, env):
        if not expr_in_env(self, env):
            return self
        subexpr_prime = self.subexpr.subs(env)
        return Reciprocal(subexpr_prime)
    def evaluate(self, assignment):
        x = self.subexpr.evaluate(assignment)
        return self.ffwd(x)
    def ffwd(self, x):
        assert x in self.domain()
        return 0 if isinf(x) else sympy.Rational(1, x)
    def finv(self, y):
        if y not in self.range():
            return EmptySet
        if y == 0:
            return {-oo, oo}
        return {sympy.Rational(1, y)}
    def invert_finite(self, ys):
        ys_prime = sympy.Union(*[self.finv(y) for y in ys])
        return self.subexpr.invert(ys_prime)
    def invert_interval(self, ys):
        (a, b) = (ys.left, ys.right)
        if (0 <= a < b):
            assert 0 < a or ys.left_open
            a_inv = sympy.Rational(1, a) if 0 < a else oo
            b_inv = sympy.Rational(1, b) if (not isinf(b)) else 0
            ys_prime = transform_interval(ys, b_inv, a_inv, flip=True)
            return self.subexpr.invert(ys_prime)
        if (a < b <= 0):
            assert b < 0 or ys.right_open
            a_inv = sympy.Rational(1, a) if (not isinf(a)) else 0
            b_inv = sympy.Rational(1, b) if b < 0 else -oo
            ys_prime = transform_interval(ys, b_inv, a_inv, flip=True)
            return self.subexpr.invert(ys_prime)
        assert False, 'Impossible Reciprocal interval: %s ' % (ys,)
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

class Poly(Transform):
    def __init__(self, subexpr, coeffs):
        assert len(coeffs) > 1
        self.subexpr = make_subexpr(subexpr, self)
        self.coeffs = tuple(coeffs)
        self.degree = len(coeffs) - 1
        self.symexpr = make_sympy_polynomial(coeffs)
    def domain(self):
        return ExtReals
    def range(self):
        result = function_range(self.symexpr, symX, Reals)
        if isinstance(result, ContainersFinite):
            return result
        pos_inf = sympy.FiniteSet(oo) if result.right == oo else EmptySet
        neg_inf = sympy.FiniteSet(-oo) if result.left == -oo else EmptySet
        return sympy.Union(result, pos_inf, neg_inf)
    def subs(self, env):
        if not expr_in_env(self, env):
            return self
        subexpr_prime = self.subexpr.subs(env)
        return Poly(subexpr_prime, self.coeffs)
    def evaluate(self, assignment):
        x = self.subexpr.evaluate(assignment)
        return self.ffwd(x)
    def ffwd(self, x):
        assert x in self.domain()
        return self.symexpr.subs(symX, x) \
            if not isinf(x) else limit(self.symexpr, symX, x)
    def finv(self, y):
        if not y in self.range():
            return EmptySet
        return solve_poly_equality(self.symexpr, y)
    def invert_finite(self, ys):
        ys_prime = sympy.Union(*[self.finv(y) for y in ys])
        return self.subexpr.invert(ys_prime)
    def invert_interval(self, ys):
        assert isinstance(ys, sympy.Interval)
        (a, b) = (ys.left, ys.right)
        (lo, ro) = (not ys.left_open, ys.right_open)
        ys_prime_a = solve_poly_inequality(self.symexpr, a, lo, extended=False)
        ys_prime_b = solve_poly_inequality(self.symexpr, b, ro, extended=False)
        ys_prime = ys_prime_a.complement(ys_prime_b)
        return self.subexpr.invert(ys_prime)
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
# Non-injective Piecewise Transform.

class Piecewise(Transform):
    def __init__(self, subexprs, events):
        self.subexprs = [make_subexpr(subexpr) for subexpr in subexprs]
        self.events = [make_event(event) for event in events]
        self.symbols = get_piecewise_symbol(self.subexprs, self.events)
        self.domains = get_piecewise_domains(self.events)
    def domain(self):
        return sympy.Union(*self.domains)
    def range(self):
        ranges = [subexpr.range() for subexpr in self.subexprs]
        return sympy.Union(*ranges)
    def subs(self, env):
        if not expr_in_env(self, env):
            return self
        # TODO: Check whether this logic is correct.
        subexprs_prime = [subexpr.subs(env) for subexpr in self.subexprs]
        events_prime = [event.subs(env) for event in self.events]
        return Piecewise(subexprs_prime, events_prime)
    def evaluate(self, assignment):
        raise NotImplementedError()
    def ffwd(self, x):
        index = next(i for i, domain in enumerate(self.domains) if x in domain)
        return self.subexprs[index].ffwd(x)
    def finv(self, y):
        xs_list = get_piecewise_inverse(lambda subexpr: subexpr.finv(y),
            self.subexprs, self.domains)
        return sympy.Union(*xs_list)
    def invert_finite(self, ys):
        xs_list = get_piecewise_inverse(lambda subexpr: subexpr.invert(ys),
            self.subexprs, self.domains)
        return sympy.Union(*xs_list)
    def invert_interval(self, ys):
        xs_list = get_piecewise_inverse(lambda subexpr: subexpr.invert(ys),
            self.subexprs, self.domains)
        return sympy.Union(*xs_list)
    def __add__(self, x):
        if isinstance(x, Piecewise):
            subexprs = self.subexprs + x.subexprs
            events = self.events + x.events
            return Piecewise(subexprs, events)
        return super().__add__(x)
    def __eq__(self, x):
        return isinstance(x, Piecewise) \
            and self.subexprs == x.subexprs \
            and self.events == x.events
    def __repr__(self):
        return 'Piecewise(events=%s, %s)' \
            % (repr(self.events), repr(self.subexprs))
    def __str__(self):
        strings = [
            '(%s) * Indicator[%s]' % (str(subexpr), str(event))
            for subexpr, event in zip(self.subexprs, self.events)
        ]
        return ' + '.join(strings)
    def __hash__(self):
        x = (self.__class__, self.subexprs, self.events)
        return hash(x)

def get_piecewise_symbol(subexprs, events):
    if len(subexprs) != len(events):
        raise ValueError('Piecewise requires same no. of subexprs and events.')
    symbols_subexprs = get_union([subexpr.get_symbols() for subexpr in subexprs])
    if len(symbols_subexprs) > 1:
        raise ValueError('Piecewise cannot have multi-symbol subexpressions.')
    symbols_events = get_union([event.get_symbols() for event in events])
    if len(symbols_subexprs) > 1:
        raise ValueError('Piecewise cannot have multi-symbol events.')
    if symbols_subexprs != symbols_events:
        raise ValueError('Piecewise events and subexprs need same symbols.')
    return symbols_subexprs

def get_piecewise_domains(events):
    domains = [event.solve() for event in events]
    for i, di in enumerate(domains):
        for j, dj in enumerate(domains):
            if i == j:
                continue
            intersection = sympy.Intersection(di, dj)
            if intersection is not EmptySet:
                raise ValueError('Piecewise events %s and %s overlap'
                    % (di, dj))
    return domains

def get_piecewise_inverse(f_inv, subexprs, domains):
    inverses = [f_inv(subexpr) for subexpr in subexprs]
    return [sympy.Intersection(i, d) for i, d in zip(inverses, domains)]

# ==============================================================================
# Non-injective Boolean-valued Transforms.

class Event(Transform):
    def range(self):
        return sympy.FiniteSet(0, 1)

    def __mul__(self, x):
        if isinstance(x, Transform):
            return Piecewise((x,), (self,))
        return super().__mul__(x)

    # Event methods.
    def solve(self):
        if len(self.symbols) > 1:
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
    def __invert__(self):
        raise NotImplementedError()
    def __xor__(self, event):
        if isinstance(event, Event):
            return (self & ~event) | (~self & event)
        return NotImplemented

class EventBasic(Event):
    subexpr = None
    values = None

    def domain(self):
        return self.subexpr.domain()
    def subs(self, env):
        if not expr_in_env(self, env):
            return self
        subexpr_prime = self.subexpr.subs(env)
        return type(self)(subexpr_prime, self.values)
    def evaluate(self, assignment):
        x = self.subexpr.evaluate(assignment)
        return self.ffwd(x)
    def ffwd(self, x):
        return x in self.values

    # Disable < on Events.
    def __gt__(self, x):
        raise TypeError()
    def __ge__(self, x):
        raise TypeError()
    def __lt__(self, x):
        raise TypeError()
    def __le__(self, x):
        raise TypeError()

    # Event methods.
    def to_dnf_list(self):
        return [[self]]
    def __and__(self, event):
        # Simplify.
        if isinstance(event, EventBasic):
            if event.subexpr == self.subexpr:
                intersection = sympy.Intersection(self.values, event.values)
                if intersection is EmptySet:
                    return EventFiniteReal(self.subexpr, EmptySet)
                if isinstance(intersection, sympy.FiniteSet):
                    if is_nominal_set(intersection):
                        return EventFiniteNominal(self.subexpr, intersection)
                    if is_numeric_set(intersection):
                        return EventFiniteReal(self.subexpr, intersection)
                if isinstance(intersection, sympy.Interval):
                    return EventInterval(self.subexpr, intersection)
                if isinstance(intersection, sympy.Complement):
                    assert is_nominal_set(intersection.args[1])
                    if intersection.args[0] is UniversalSet:
                        return EventFiniteNominal(self.subexpr, intersection)
                    if isinstance(intersection.args[0], sympy.Interval):
                        return EventInterval(self.subexpr, intersection.args[0])
                    assert False, 'Unknown intersection: %s' % (intersection,)
        # Linearize but do not simplify.
        if isinstance(event, EventAnd):
            events = (self,) + event.subexprs
            return EventAnd(events)
        if isinstance(event, (EventBasic, EventOr)):
            return EventAnd([self, event])
        return NotImplemented
    def __or__(self, event):
        # Simplify.
        if isinstance(event, EventBasic):
            if event.subexpr == self.subexpr:
                union = sympy.Union(self.values, event.values)
                if union is EmptySet:
                    return EventFiniteReal(self.subexpr, EmptySet)
                if isinstance(union, sympy.FiniteSet):
                    if is_nominal_set(union):
                        return EventFiniteNominal(self.subexpr, union)
                    if is_numeric_set(union):
                        return EventFiniteReal(self.subexpr, union)
                if isinstance(union, sympy.Interval):
                    return EventInterval(self.subexpr, union)
                if isinstance(union, sympy.Complement):
                    return EventFiniteNominal(self.subexpr, union)
        # Linearize but do not simplify.
        if isinstance(event, EventOr):
            events = (self,) + event.subexprs
            return EventOr(events)
        if isinstance(event, (EventBasic, EventAnd)):
            return EventOr([self, event])
        return NotImplemented
    def __eq__(self, event):
        return isinstance(event, type(self)) \
            and (self.values == event.values) \
            and (self.subexpr == event.subexpr)
    def __hash__(self):
        x = (self.__class__, self.subexpr, self.values)
        return hash(x)

class EventInterval(EventBasic):
    def __init__(self, subexpr, values):
        assert isinstance(values, sympy.Interval)
        self.subexpr = make_subexpr(subexpr, self)
        self.values = values
    def finv(self, y):
        if y not in self.range():
            return EmptySet
        if y == 1:
            return self.values
        if y == 0:
            return sympy.Complement(Reals, self.values)
    def invert_finite(self, ys):
        ys_prime = sympy.Union(*[self.finv(y) for y in ys])
        return self.subexpr.invert(ys_prime)

    # Support chaining notation (a < X) < b
    # Overrides the behavior of < from Transform.
    def __compute_gte__(self, x, left_open):
        # x < (Y < b)
        if not isinf_neg(self.values.left):
            raise ValueError('cannot compute %s < %s' % (x, str(self)))
        xn = sympify_number(x)
        interval = sympy.Interval(xn, self.values.right,
            left_open=left_open, right_open=self.values.right_open)
        if isinstance(interval, sympy.Interval):
            return EventInterval(self.subexpr, interval)
        if isinstance(interval, ContainersFinite):
            return EventFiniteReal(self.subexpr, interval)
        assert False, 'Unknown interval: %s' % (interval,)
    def __compute_lte__(self, x, right_open):
        # (a < Y) < x
        if not isinf_pos(self.values.right):
            raise ValueError('cannot compute %s < %s' % (str(self), x))
        xn = sympify_number(x)
        interval = sympy.Interval(self.values.left, xn,
            left_open=self.values.left_open, right_open=right_open)
        if isinstance(interval, sympy.Interval):
            return EventInterval(self.subexpr, interval)
        if isinstance(interval, ContainersFinite):
            return EventFiniteReal(self.subexpr, interval)
        assert False, 'Unknown interval: %s' % (interval,)
    def __gt__(self, x):
        return self.__compute_gte__(x, True)
    def __ge__(self, x):
        return self.__compute_gte__(x, False)
    def __lt__(self, x):
        return self.__compute_lte__(x, True)
    def __le__(self, x):
        return self.__compute_lte__(x, False)
    def __invert__(self):
        values_not = sympy.Complement(Reals, self.values)
        if values_not is EmptySet:
            return EventFiniteReal(self.subexpr, values_not)
        if isinstance(values_not, sympy.Interval):
            return EventInterval(self.subexpr, values_not)
        if isinstance(values_not, sympy.Union):
            event_l = EventInterval(self.subexpr, values_not.args[0])
            event_r = EventInterval(self.subexpr, values_not.args[1])
            return EventOr([event_l, event_r])
        assert False, 'Unknown complemented interval: %s' % (values_not,)
    def __repr__(self):
        return 'EventInterval(%s, %s)' % (repr(self.subexpr), repr(self.values))
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
        return result

class EventFiniteReal(EventBasic):
    def __init__(self, subexpr, values):
        assert isinstance(values, ContainersFinite) or values is EmptySet
        assert all(is_number(v) for v in values)
        self.subexpr = make_subexpr(subexpr, self)
        self.values = sympy.FiniteSet(*values)
    def finv(self, y):
        if y not in self.range():
            return EmptySet
        if y == 1:
            return self.values
        if y == 0:
            return sympy.Complement(Reals, self.values)
    def invert_finite(self, ys):
        ys_prime = sympy.Union(*[self.finv(y) for y in ys])
        return self.subexpr.invert(ys_prime)

    def __repr__(self):
        return 'EventFiniteReal(%s, %s)' \
            % (repr(self.subexpr), repr(self.values))
    def __str__(self):
        return '%s << %s' % (str(self.subexpr), str(self.values))
    def __invert__(self):
        values_not = sympy.Complement(Reals, self.values)
        if values_not is Reals:
            return EventInterval(self.subexpr, sympy.Interval(-oo, oo))
        assert isinstance(values_not, sympy.Union)
        events = [EventInterval(self.subexpr, v) for v in values_not.args]
        return EventOr(events)

class EventFiniteNominal(EventBasic):
    def __init__(self, subexpr, values):
        special = values is EmptySet \
            or values is UniversalSet \
            or isinstance(values, sympy.Complement)
        assert special or is_nominal_set(values)
        self.subexpr = make_subexpr(subexpr, self)
        self.values = values if special else values
        self.transformed = not isinstance(self.subexpr, Identity)
        self.complemented = isinstance(values, sympy.Complement)

    def finv(self, y):
        if y not in self.range():
            return EmptySet
        if y == 1:
            if self.values is EmptySet:
                return EmptySet
            if self.transformed:
                if not self.complemented:
                    return EmptySet
                else:
                    return UniversalSet
            return self.values
        if y == 0:
            if self.values is EmptySet:
                return UniversalSet
            if self.transformed:
                if not self.complemented:
                    return UniversalSet
                else:
                    return EmptySet
            return complement_nominal_set(self.values)

    def invert_finite(self, ys):
        if ys == sympy.FiniteSet(0):
            return self.finv(0)
        if ys == sympy.FiniteSet(1):
            return self.finv(1)
        if ys == sympy.FiniteSet(1, 2):
            return UniversalSet

    def __or__(self, event):
        # De Morgan's laws for Events of the form
        # ~(X << A) | ~(X << B) = ~( (X << A) & (X << B) )
        if isinstance(event, EventFiniteNominal):
            if event.subexpr == self.subexpr:
                union = sympy.Union(self.values, event.values)
                if isinstance(union, sympy.Union):
                    if (union.args[0].args[0] is UniversalSet
                            and union.args[1].args[0] is UniversalSet):
                        assert isinstance(self.values, sympy.Complement)
                        assert isinstance(event.values, sympy.Complement)
                        A = complement_nominal_set(self.values)
                        B = complement_nominal_set(event.values)
                        intersection = sympy.Intersection(A, B)
                        return ~EventFiniteNominal(self.subexpr, intersection)
        return super().__or__(event)
    def __repr__(self):
        return 'EventFiniteNominal(%s, %s)' \
            % (repr(self.subexpr), repr(self.values))
    def __str__(self):
        str_values = '(%s)' % (str(self.values,)) \
            if self.complemented else '%s' % (str(set(self.values)),)
        return '%s << %s' % (str(self.subexpr), str_values)
    def __invert__(self):
        values_not = complement_nominal_set(self.values)
        return EventFiniteNominal(self.subexpr, values_not)

class EventCompound(Event):
    def __init__(self, subexprs):
        assert all(isinstance(subexpr, Event) for subexpr in subexprs)
        self.subexprs = tuple([make_subexpr(event) for event in subexprs])
        self.symbols = get_union([event.get_symbols() for event in subexprs])
    def domain(self):
        if len(self.symbols) > 1:
            raise ValueError('No domain for multi-symbol Event.')
        domains = [event.domain() for event in self.subexprs]
        return sympy.Intersection(*domains)
    def subs(self, env):
        if not expr_in_env(self, env):
            return self
        subexprs_prime = [subexpr.subs(env) for subexpr in self.subexprs]
        return type(self)(subexprs_prime)

class EventOr(EventCompound):
    def evaluate(self, assignment):
        ys = [event.evaluate(assignment) for event in self.subexprs]
        return any(ys)
    def ffwd(self, x):
        # Cannot asses on multi-symbol Event.
        ys = [event.ffwd(x) for event in self.subexprs]
        return any(ys)
    def finv(self, y):
        # Cannot invert multi-symbol Event.
        xs_list = [event.finv(y) for event in self.subexprs]
        return sympy.Union(*xs_list)
    def invert_finite(self, ys):
        xs_list = [event.invert(ys) for event in self.subexprs]
        return sympy.Union(*xs_list)

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
        return reduce(lambda state, event: state & event, sub_events)
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
        # Cannot asses on multi-symbol Event.
        ys = [event.ffwd(x) for event in self.subexprs]
        return all(ys)
    def finv(self, y):
        # Cannot invert multi-symbol Event.
        xs_list = [event.finv(y) for event in self.subexprs]
        return sympy.Intersection(*xs_list)
    def invert_finite(self, ys):
        xs_list = [event.invert(ys) for event in self.subexprs]
        return sympy.Intersection(*xs_list)

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
        return reduce(lambda state, event: state | event, sub_events)
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
    return Exponential(subexpr, sympy.exp(1))
def Log(subexpr):
    return Logarithm(subexpr, sympy.exp(1))
def Sqrt(subexpr):
    return Radical(subexpr, 2)
def Pow(subexpr, n):
    assert 0 <= n
    coeffs = [0]*n + [1]
    return Poly(subexpr, coeffs)

def expr_in_env(expr, env):
    return env and any(env.get(s,s) != s for s in expr.get_symbols())

def transform_interval(interval, a, b, flip=None):
    return \
        sympy.Interval(a, b, interval.left_open, interval.right_open) \
        if not flip else \
        sympy.Interval(a, b, interval.right_open, interval.left_open) \

def make_sympy_polynomial(coeffs):
    terms = [c*symX**i for (i,c) in enumerate(coeffs)]
    return sympy.Add(*terms)

def make_subexpr(subexpr, expr=None):
    if isinstance(subexpr, Transform):
        if expr is not None:
            expr.symbols = subexpr.symbols
        return subexpr
    raise TypeError('Invalid subexpr %s with type %s'
        % (subexpr, str(type(subexpr))))

def make_event(event):
    if isinstance(event, Event):
        return event
    raise TypeError('Invalid event %s with type %s'
        % (event, str(type(event))))

def polyify(expr):
    if isinstance(expr, Poly):
        return expr
    return Poly(expr, (0, 1))
