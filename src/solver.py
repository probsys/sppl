# Copyright 2019 MIT Probabilistic Computing Project.
# See LICENSE.txt

from math import isinf

from sympy import And
from sympy import Not
from sympy import Or

from sympy import Intersection
from sympy import Interval
from sympy import Union

from sympy import Add as SymAdd
from sympy import Pow as SymPow
from sympy import exp as SymExp
from sympy import log as SymLog

from sympy import Rational
from sympy import S as Singletons
from sympy import Symbol

from sympy import ConditionSet
from sympy import solveset

from sympy import oo
from sympy.abc import X as symX
from sympy.core.relational import Relational

from .sym_util import get_symbols

EmptySet = Singletons.EmptySet
Reals = Singletons.Reals
RealsPos = Interval(0, oo)
RealsNeg = Interval(-oo, 0)

# ==============================================================================
# Utilities.

def transform_interval(interval, a, b):
    return Interval(a, b, interval.left_open, interval.right_open)

def make_sympy_polynomial(coeffs):
    terms = [c*symX**i for (i,c) in enumerate(coeffs)]
    return SymAdd(*terms)

def make_subexpr(subexpr):
    if isinstance(subexpr, Symbol):
        return Identity(subexpr)
    if isinstance(subexpr, Transform):
        return subexpr
    assert False, 'Unknown subexpr: %s' % (subexpr,)

def solveset_bounds(sympy_expr, b, strict):
    if not isinf(b):
        expr = (sympy_expr < b) if strict else (sympy_expr <= b)
        return solveset(expr, domain=Reals)
    if b < oo:
        return EmptySet
    return Reals

def listify_interval(interval):
    if interval == EmptySet:
        return [EmptySet]
    if isinstance(interval, Interval):
        return [interval]
    if isinstance(interval, Union):
        intervals = interval.args
        assert all(isinstance(intv, Interval) for intv in intervals)
        return intervals
    assert False, 'Unknown interval: %s' % (interval,)

# ==============================================================================
# Custom invertible function language.

class Transform(object):
    def symbol(self):
        raise NotImplementedError()
    def subexpr(self):
        raise NotImplementedError()
    def domain(self):
        raise NotImplementedError()
    def range(self):
        raise NotImplementedError()
    def ffwd(self, x):
        raise NotImplementedError()
    def finv(self, x):
        raise NotImplementedError()
    def solve(self, interval):
        raise NotImplementedError()
    def evaluate(self, x):
        raise NotImplementedError()
    def inverse(self, x):
        raise NotImplementedError()

class Identity(Transform):
    def __init__(self, symbol):
        assert isinstance(symbol, Symbol)
        self.symb = symbol
    def symbol(self):
        return self.symb
    def domain(self):
        return Reals
    def range(self):
        return Reals
    def ffwd(self, x):
        return x
    def finv(self, x):
        return [x]
    def solve(self, interval):
        return interval

class Abs(Transform):
    def __init__(self, subexpr):
        self.subexpr = make_subexpr(subexpr)
    def symbol(self):
        return self.subexpr.symbol
    def domain(self):
        return Reals
    def range(self):
        return RealsPos
    def ffwd(self, x):
        return x if x > 0 else -x
    def finv(self, x):
        return [x, -x]
    def solve(self, interval):
        intersection = Intersection(self.range(), interval)
        if intersection == EmptySet:
            return EmptySet
        (a, b) = (intersection.left, intersection.right)
        # Positive solution.
        (a_pos, b_pos) = (a, b)
        interval_pos = transform_interval(intersection, a_pos, b_pos)
        xvals_pos = self.subexpr.solve(interval_pos)
        # Negative solution.
        (a_neg, b_neg) = (-b, -a)
        interval_neg = transform_interval(intersection, a_neg, b_neg)
        xvals_neg = self.subexpr.solve(interval_neg)
        # Return the union.
        return Union(xvals_pos, xvals_neg)

class Radical(Transform):
    def __init__(self, subexpr, degree):
        assert degree != 0
        self.subexpr = make_subexpr(subexpr)
        self.degree = degree
    def symbol(self):
        return self.subexpr.symbol
    def domain(self):
        return RealsPos
    def range(self):
        return RealsPos
    def ffwd(self, x):
        return SymPow(x, Rational(1, self.degree))
    def finv(self, x):
        return SymPow(x, Rational(self.degree, 1))
    def solve(self, interval):
        intersection = Intersection(self.range(), interval)
        if intersection == EmptySet:
            return EmptySet
        (a, b) = (intersection.left, intersection.right)
        a_prime = self.finv(a)
        b_prime = self.finv(b)
        interval_prime = transform_interval(intersection, a_prime, b_prime)
        return self.subexpr.solve(interval_prime)

class Exp(Transform):
    def __init__(self, subexpr, base):
        assert base > 0
        self.subexpr = make_subexpr(subexpr)
        self.base = base
    def symbol(self):
        return self.subexpr.symbol
    def domain(self):
        return Reals
    def range(self):
        return RealsPos
    def ffwd(self, x):
        assert x in self.domain()
        return SymPow(self.base, x)
    def finv(self, x):
        assert x in self.range()
        return SymLog(x, self.base) if x > 0 else -oo
    def solve(self, interval):
        intersection = Intersection(self.range(), interval)
        if intersection == EmptySet:
            return EmptySet
        (a, b) = (intersection.left, intersection.right)
        a_prime = self.finv(a)
        b_prime = self.finv(b)
        interval_prime = transform_interval(intersection, a_prime, b_prime)
        return self.subexpr.solve(interval_prime)

class Log(Transform):
    def __init__(self, subexpr, base):
        assert base > 1
        self.subexpr = make_subexpr(subexpr)
        self.base = base
    def symbol(self):
        return self.subexpr.symbol
    def domain(self):
        return RealsPos
    def range(self):
        return Reals
    def ffwd(self, x):
        assert x in self.domain()
        return SymLog(x, self.base) if x > 0 else -oo
    def finv(self, x):
        assert x in self.range()
        return SymPow(self.base, x)
    def solve(self, interval):
        (a, b) = (interval.left, interval.right)
        a_prime = self.finv(a)
        b_prime = self.finv(b)
        interval_prime = transform_interval(interval, a_prime, b_prime)
        return self.subexpr.solve(interval_prime)

class Poly(Transform):
    def __init__(self, subexpr, coeffs):
        self.subexpr = make_subexpr(subexpr)
        self.coeffs = coeffs
        self.degree = len(coeffs) - 1
        self.symexpr = make_sympy_polynomial(coeffs)
    def symbol(self):
        return self.subexpr.symbol
    def domain(self):
        return Reals
    def range(self):
        raise NotImplementedError()
    def ffwd(self, x):
        return self.symexpr.subs(symX, x)
    def finv(self, x):
        pass
    def solve(self, interval):
        (a, b) = (interval.left, interval.right)
        xvals_a = solveset_bounds(self.symexpr, a, not interval.left_open)
        xvals_b = solveset_bounds(self.symexpr, b, interval.right_open)
        xvals = xvals_a.complement(xvals_b)
        if xvals == EmptySet:
            return EmptySet
        xvals_list = listify_interval(xvals)
        intervals = [self.subexpr.solve(x) for x in xvals_list if x != EmptySet]
        return Union(*intervals)

# Some useful constructors.
def ExpNat(subexpr):
    return Exp(subexpr, SymExp(1))
def LogNat(subexpr):
    return Log(subexpr, SymExp(1))
def Sqrt(subexpr):
    return Radical(subexpr, 2)
def Pow(subexpr, n):
    coeffs = [0]*n + [1]
    return Poly(subexpr, coeffs)

# ==============================================================================
# Custom event language.

class Event(object):
    pass

class EventInterval(Event):
    def __init__(self, expr, interval):
        self.interval = interval
        self.expr = make_subexpr(expr)
    def solve(self):
        return self.expr.solve(self.interval)

class EventOr(Event):
    def __init__(self, events):
        self.events = events
    def solve(self):
        intervals = [event.solve() for event in self.events]
        return Union(*intervals)

class EventAnd(Event):
    def __init__(self, events):
        self.events = events
    def solve(self):
        intervals = [event.solve() for event in self.events]
        return Intersection(*intervals)

class EventNot(Event):
    def __init__(self, event):
        self.event = event
    def solve(self):
        # TODO Should complement domain not Reals.
        interval = self.event.solve()
        return interval.complement(Reals)

# ==============================================================================
# SymPy solver.

def solver(expr):
    symbols = get_symbols(expr)
    if len(symbols) != 1:
        raise ValueError('Expression "%s" needs exactly one symbol.' % (expr,))

    if isinstance(expr, Relational):
        result = solveset(expr, domain=Reals)
    elif isinstance(expr, Or):
        subexprs = expr.args
        intervals = [solver(e) for e in subexprs]
        result = Union(*intervals)
    elif isinstance(expr, And):
        subexprs = expr.args
        intervals = [solver(e) for e in subexprs]
        result = Intersection(*intervals)
    elif isinstance(expr, Not):
        (notexpr,) = expr.args
        interval = solver(notexpr)
        result = interval.complement(Reals)
    else:
        raise ValueError('Expression "%s" has unknown type.' % (expr,))

    if isinstance(result, ConditionSet):
        raise ValueError('Expression "%s" is not invertible.' % (expr,))

    return result
