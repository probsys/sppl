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
from sympy import log as SymLog

from sympy import Rational
from sympy import S as Singletons
from sympy import Symbol

from sympy import ConditionSet
from sympy import solveset

from sympy import oo
from sympy.abc import X as symX
from sympy.core.relational import Relational


EmptySet = Singletons.EmptySet
Reals = Singletons.Reals
RealsPos = Interval(0, oo)
RealsNeg = Interval(-oo, 0)

# ==============================================================================
# Utilities.

def make_sympy_polynomial(coeffs):
    terms = [c*symX**i for (i,c) in enumerate(coeffs)]
    return SymAdd(*terms)

def make_subexp(subexp):
    if isinstance(subexp, Symbol):
        return Identity(subexp)
    if isinstance(subexp, Transform):
        return subexp
    assert False, 'Unknown subexp: %s' % (subexp,)

def solveset_bounds(sympy_expr, b):
    if not isinf(b):
        return solveset(sympy_expr < b, domain=Reals)
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
# Custom invertible language.

class Transform(object):
    def symbol(self):
        raise NotImplementedError()
    def subexp(self):
        raise NotImplementedError()
    def domain(self):
        raise NotImplementedError()
    def range(self):
        raise NotImplementedError()
    def solve(self, a, b):
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
    def solve(self, a, b):
        return Interval(a, b)

class Abs(Transform):
    def __init__(self, subexp):
        self.subexp = make_subexp(subexp)
    def symbol(self):
        return self.subexp.symbol
    def domain(self):
        return Reals
    def range(self):
        return RealsPos
    def solve(self, a, b):
        intersection = Intersection(self.range(), Interval(a, b))
        if intersection == EmptySet:
            return EmptySet
        xvals_pos = self.subexp.solve(intersection.left, intersection.right)
        xvals_neg = self.subexp.solve(-intersection.right, -intersection.left)
        return Union(xvals_pos, xvals_neg)

class Pow(Transform):
    def __init__(self, subexp, expon):
        assert isinstance(expon, (int, Rational))
        assert expon != 0
        self.subexp = make_subexp(subexp)
        self.expon = expon
        self.integral = expon == int(expon)
    def symbol(self):
        return self.subexp.symbol
    def domain(self):
        if self.integral:
            return Reals
        return RealsPos
    def range(self):
        if self.integral:
            return Reals if self.expon % 2 else RealsPos
        return RealsPos
    def solve(self, a, b):
        intersection = Intersection(self.range(), Interval(a, b))
        if intersection == EmptySet:
            return EmptySet
        a_prime = SymPow(intersection.left, 1/self.expon)
        b_prime = SymPow(intersection.right, 1/self.expon)
        return self.subexp.solve(a_prime, b_prime)

class Exp(Transform):
    def __init__(self, subexp, base):
        assert base > 0
        self.subexp = make_subexp(subexp)
        self.base = base
    def symbol(self):
        return self.subexp.symbol
    def domain(self):
        return Reals
    def range(self):
        return RealsPos
    def solve(self, a, b):
        intersection = Intersection(self.range(), Interval(a,b))
        if intersection == EmptySet:
            return EmptySet
        a_prime = SymLog(intersection.left, self.base) \
            if intersection.left > 0 else -oo
        b_prime = SymLog(intersection.right, self.base)
        return self.subexp.solve(a_prime, b_prime)

class Log(Transform):
    def __init__(self, subexp, base):
        assert base > 0
        self.subexp = make_subexp(subexp)
        self.base = base
    def symbol(self):
        return self.subexp.symbol
    def domain(self):
        return RealsPos
    def range(self):
        return Reals
    def solve(self, a, b):
        a_prime = SymPow(self.base, a)
        b_prime = SymPow(self.base, b)
        return self.subexp.solve(a_prime, b_prime)

class Poly(Transform):
    def __init__(self, subexp, coeffs):
        self.subexp = make_subexp(subexp)
        self.coeffs = coeffs
        self.degree = len(coeffs) - 1
        self.symexp = make_sympy_polynomial(coeffs)
    def symbol(self):
        return self.subexp.symbol
    def domain(self):
        return Reals
    def range(self):
        raise NotImplementedError()
    def solve(self, a, b):
        xvals_a = solveset_bounds(self.symexp, a)
        xvals_b = solveset_bounds(self.symexp, b)
        xvals = xvals_a.complement(xvals_b)
        if xvals == EmptySet:
            return EmptySet
        xvals_list = listify_interval(xvals)
        intervals = [self.subexp.solve(xv.left, xv.right)
            for xv in xvals_list if xv != EmptySet]
        return Union(*intervals)

class Event(object):
    pass

class EventBetween(Event):
    def __init__(self, expr, a, b):
        self.a = a
        self.b = b
        self.expr = make_subexp(expr)
    def solve(self):
        return self.expr.solve(self.a, self.b)

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
# Utilities

def get_symbols(expr):
    atoms = expr.atoms()
    return [a for a in atoms if isinstance(a, Symbol)]

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
