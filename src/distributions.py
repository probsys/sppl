# Copyright 2019 MIT Probabilistic Computing Project.
# See LICENSE.txt

from collections import Counter
from functools import reduce
from itertools import chain

from sympy import And
from sympy import Eq
from sympy import Not
from sympy import Or

from sympy import S as Singletons
from sympy import Symbol
from sympy import solveset

from sympy import ConditionSet
from sympy import Intersection
from sympy import Interval
from sympy import Union

from sympy import Tuple
from sympy import to_dnf

from sympy.core.relational import Relational

from .math_util import allclose
from .math_util import logdiffexp
from .math_util import logflip
from .math_util import lognorm
from .math_util import logsumexp

from .contains import Containment
from .contains import Contains
from .contains import NotContains

EmptySet = Singletons.EmptySet
Reals = Singletons.Reals
inf = float('inf')

# ==============================================================================
# Custom invertible functions.

from math import log
from math import isinf

from sympy import oo
RealsPos = Interval(0, oo)
RealsNeg = Interval(-oo, 0)

def logg(x, base):
    assert 0 <= x
    return -oo if x == 0 else log(x, base)

def make_sympy_polynomial(coeffs):
    from sympy.abc import X
    from sympy import Add
    terms = [c*X**i for (i,c) in enumerate(coeffs)]
    return Add(*terms)

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
        from sympy import Rational
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
        import sympy
        a_prime = sympy.Pow(intersection.left, 1/self.expon)
        b_prime = sympy.Pow(intersection.right, 1/self.expon)
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
        # import ipdb; ipdb.set_trace()
        intersection = Intersection(self.range(), Interval(a,b))
        if intersection == EmptySet:
            return EmptySet
        import sympy
        a_prime = sympy.log(intersection.left, self.base) \
            if intersection.left > 0 else -oo
        b_prime = sympy.log(intersection.right, self.base)
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
        import sympy
        a_prime = sympy.Pow(self.base, a)
        b_prime = sympy.Pow(self.base, b)
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

def factor_dnf(expr):
    symbols = get_symbols(expr)
    lookup = {s:s for s in symbols}
    return factor_dnf_symbols(expr, lookup)

def factor_dnf_symbols(expr, lookup):
    if isinstance(expr, (Relational, Containment)):
        # Literal term.
        symbols = get_symbols(expr)
        if len(symbols) > 1:
            raise ValueError('Expression "%s" has multiple symbols.' % (expr,))
        key = lookup[symbols[0]]
        return {key: expr}

    elif isinstance(expr, And):
        # Product term.
        subexprs = expr.args
        assert all(isinstance(e, (Relational, Containment)) for e in subexprs)
        mappings = [factor_dnf_symbols(subexpr, lookup) for subexpr in  subexprs]
        exprs = {}
        for mapping in mappings:
            assert len(mapping) == 1
            [(key, subexp)] = mapping.items()
            if key not in exprs:
                exprs[key] = subexp
            else:
                exprs[key] = And(subexp, exprs[key])
        return exprs

    elif isinstance(expr, Or):
        # Sum term.
        subexprs = expr.args
        mappings = [factor_dnf_symbols(subexpr, lookup) for subexpr in subexprs]
        exprs = {}
        for mapping in mappings:
            for key, subexp in mapping.items():
                if key not in exprs:
                    exprs[key] = subexp
                else:
                    exprs[key] = Or(subexp, exprs[key])
        return exprs
    else:
        assert False, 'Invalid DNF expression: %s' % (expr,)

def simplify_nominal_event(event, support):
    if isinstance(event, Eq):
        a, b = event.args
        value = b if isinstance(a, Symbol) else a
        return support.intersection({value})
    elif isinstance(event, Contains):
        a, b = event.args
        assert isinstance(a, Symbol)
        return support.intersection(b)
    elif isinstance(event, NotContains):
        a, b = event.args
        assert isinstance(a, Symbol)
        return support.difference(b)
    elif isinstance(event, And):
        sets = [simplify_nominal_event(e, support) for e in event.args]
        return get_intersection(sets)
    elif isinstance(event, Or):
        sets = [simplify_nominal_event(e, support) for e in event.args]
        return get_union(sets)
    else:
        raise ValueError('Event "%s" does not apply to nominal variable'
            % (event,))

def get_union(sets):
    return sets[0].union(*sets[1:])

def get_intersection(sets):
    return sets[0].intersection(*sets[1:])

def are_disjoint(sets):
    union = get_union(sets)
    return len(union) == sum(len(s) for s in sets)

def are_identical(sets):
    intersection = get_intersection(sets)
    assert all(len(s) == len(intersection) for s in sets)

# ==============================================================================
# Distribution classes

class Distribution(object):
    def __init__(self):
        raise NotImplementedError()
    def sample(self, N, rng):
        raise NotImplementedError()
    def sample_expr(self, expr, N, rng):
        raise NotImplementedError()
    def logprob(self, event):
        raise NotImplementedError()
    def logpdf(self, x):
        raise NotImplementedError()
    def condition(self, event):
        raise NotImplementedError()

class MixtureDistribution(Distribution):
    """Weighted mixture of distributions."""

    def __init__(self, distributions, weights):
        self.distributions = distributions
        self.weights = weights
        self.indexes = list(range(len(self.weights)))

        symbols = [d.get_symbols() for d in distributions]
        assert are_identical(symbols), \
            'Distributions in Mixture must have identical symbols.'
        self.symbols = distributions[0].symbols

    def sample(self, N, rng):
        selections = logflip(self.weights, self.indexes, N, rng)
        counts = Counter(selections)
        samples = [self.distributions[i].sample(counts[i], rng)
            for i in counts]
        return list(chain.from_iterable(samples))

    def sample_expr(self, expr, N, rng):
        selections = logflip(self.weights, self.indexes, N, rng)
        counts = Counter(selections)
        samples = [self.distributions[i].sample(counts[i], expr, rng)
            for i in counts]
        return list(chain.from_iterable(samples))

    def logprob(self, event):
        logps = [dist.logprob(event) for dist in self.distributions]
        return logsumexp([p + w for (p, w) in zip(logps, self.weights)])

    def logpdf(self, x):
        logps = [dist.logpdf(x) for dist in self.distributions]
        return logsumexp([p + w for (p, w) in zip(logps, self.weights)])

    def condition(self, event):
        logps_condt = [dist.logprob(event) for dist in self.distributions]
        logps_joint = [p + w for (p, w) in zip(logps_condt, self.weights)]
        weights = lognorm(logps_joint)
        dists = [dist.condition(event) for dist in self.distributions]
        return MixtureDistribution(dists, weights)

class ProductDistribution(Distribution):
    """Vector of independent distributions."""

    def __init__(self, distributions):
        self.distributions = distributions

        symbols = [d.symbols for d in distributions]
        self.symbols = reduce(lambda a, b: a.union(b), symbols)
        assert are_disjoint(symbols), \
            'Distributions in Product must have disjoint symbols.'
        self.lookup = {s:i for i, syms in enumerate(symbols) for s in symbols}

    def sample(self, N, rng):
        D = len(self.distributions)
        samples = [dist.sample(N, rng) for dist in self.distributions]
        return [[samples[r][c] for r in range(D)] for c in range(N)]

    def sample_expr(self, expr, N, rng):
        symbols = get_symbols(expr)
        # Partition symbols by lookup.
        partition = {}
        for sym in symbols:
            key = self.lookup[sym]
            if key not in partition:
                partition[key] = [sym]
            else:
                partition[key].append(sym)
        # Fetch the samples.
        samples = [
            self.distributions[i].sample_expr(Tuple(*syms), N, rng)
            for i, syms in partition.items()
        ]
        # Construct the expressions.
        expressions = []
        symbols_lists = partition.values()
        for _i in range(N):
            mapping = {}
            for syms, sample in zip(symbols_lists, samples):
                mapping.update({sym: x for sym, x in zip(syms, sample)})
            expri = expr.xreplace(mapping)
            expressions.append(expri)
        return expressions

    def logpdf(self, x):
        assert len(x) == len(self.distributions)
        logps = [dist.logpdf(v) for (dist, v) in zip(self.distributions, x)
            if x is not None]
        return logsumexp(logps)

    def logprob(self, event):
        # Factor the event across the product.
        dnf = to_dnf(event)
        events = factor_dnf_symbols(dnf, self.lookup)
        logprobs = [self.distributions[i].logprob(e) for i, e in events.items()]
        return logsumexp(logprobs)

    def condition(self, event):
        # Factor the event across the product.
        dnf = to_dnf(event)
        constraints = factor_dnf_symbols(dnf, self.lookup)
        distributions = [
            d.condition(constraints[i]) if (i in constraints) else d
            for i, d in enumerate(self.distributions)
        ]
        return ProductDistribution(distributions)

class NumericDistribution(Distribution):
    """Univariate probability distribution on a real interval."""

    def __init__(self, symbol, dist, support, conditioned):
        assert isinstance(support, Interval)
        self.dist = dist
        self.support = support
        self.conditioned = conditioned
        self.xl = float(support.start)
        self.xh = float(support.end)
        if conditioned:
            logp_lower = dist.logcdf(self.xl)
            logp_upper = dist.logcdf(self.xh)
            self.lognorm = logdiffexp(logp_upper, logp_lower)
            self.Fl = dist.cdf(self.xl)
            self.Fh = dist.cdf(self.xh)
        else:
            self.lognorm = 1
            self.Fl = 0
            self.Fh = 1
        self.symbol = symbol
        self.symbols = frozenset({symbol})

    def sample(self, N, rng):
        if not self.conditioned:
            return self.dist.rvs(size=N, rng=rng)
        u = rng.uniform(size=N)
        u_interval = u*self.Fl + (1-u) * self.Fh
        return self.dist.ppf(u_interval)

    def sample_expr(self, expr, N, rng):
        samples = self.sample(N, rng)
        return [expr.xreplace({self.symbol: sample}) for sample in samples]

    def logprob(self, event):
        expression = solver(event)
        values = Intersection(self.support, expression)
        if values == EmptySet:
            return -inf
        if isinstance(values, Interval):
            return self._logcdf_interval(values)
        elif isinstance(values, Union):
            intervals = values.args
            logps = [self._logcdf_interval(v) for v in intervals]
            return logsumexp(logps)
        else:
            assert False, 'Unknown event type: %s' % (event,)

    def logcdf(self, x):
        if not self.conditioned:
            return self.dist.logcdf(x)
        if self.xh <= x:
            return 0
        elif x <= self.xl:
            return -inf
        p = logdiffexp(self.dist.logcdf(x), self.Fl)
        return p - self.lognorm

    def logpdf(self, x):
        if not self.conditioned:
            return self.dist.logpdf(x)
        if x not in self.support:
            return -inf
        return self.dist.logpdf(x) - self.lognorm

    def _logcdf_interval(self, interval):
        if interval == EmptySet:
            return -inf
        xl = float(interval.start)
        xh = float(interval.end)
        return logdiffexp(self.logcdf(xh), self.logcdf(xl))

    def condition(self, event):
        expression = solver(event)
        support = Intersection(self.support, expression)
        if support == EmptySet:
            raise ValueError('Event "%s" does overlap with support "%s"'
                % (event, self.support))

        if isinstance(expression, Interval):
            return NumericDistribution(self.symbol, self.dist, support, True)
        elif isinstance(expression, Union):
            intervals = expression.args
            distributions = [
                NumericDistribution(self.symbol, self.dist, interval, True)
                for interval in intervals
            ]
            weights_unorm = [self._logcdf_interval(i) for i in intervals]
            weights = lognorm(weights_unorm)
            return MixtureDistribution(distributions, weights)
        else:
            assert False, 'Unknown expression type: %s' % (expression,)

class NominalDistribution(Distribution):
    """Probability distribution on set of unordered, non-numeric atoms."""

    def __init__(self, symbol, dist):
        self.symbol = symbol
        self.symbols = frozenset({symbol})
        self.dist = dict(dist)
        self.support = frozenset(dist.keys())
        self.outcomes = list(dist.keys())
        self.weights = list(dist.values())
        assert allclose(sum(self.weights),  1)

    def logpdf(self, x):
        return self.dist.get(x, -inf)

    def logprob(self, event):
        _, logp = self._logprob(event)
        return logp

    def _logprob(self, event):
        values = simplify_nominal_event(event, self.support)
        logp = sum(self.logpdf(x) for x in values)
        return (values, logp)

    def condition(self, event):
        values, logp = self._logprob(event)
        if logp == -inf:
            raise ValueError('Cannot condition on zero probability event: %s'
                % (event,))
        dist = {x: self.logpdf(x) - logp for x in values}
        return NominalDistribution(self.symbol, dist)

    def sample(self, N, rng):
        return logflip(self.weights, self.support, N, rng)

    def sample_expr(self, expr, N, rng):
        samples = self.sample(N, rng)
        return [expr.xreplace({self.symbol: sample}) for sample in samples]
