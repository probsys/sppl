# Copyright 2019 MIT Probabilistic Computing Project.
# See LICENSE.txt

from collections import Counter
from functools import reduce
from itertools import chain

from sympy import And
from sympy import Eq
from sympy import Not
from sympy import Or

from sympy import EmptySet
from sympy import Reals

from sympy import Symbol
from sympy import solveset

from sympy import ConditionSet
from sympy import Intersection
from sympy import Interval
from sympy import Union

from sympy import Tuple
from sympy import to_dnf

from sympy.core.relational import Relational

from math_util import allclose
from math_util import logdiffexp
from math_util import logflip
from math_util import lognorm
from math_util import logsumexp

# These are implemented as BooleanFunction in sympy but I think we need to
# implement them as Boolean, so that these expressions work with to_dnf and
# solver.
from notcontains import NotContains
from sympy import Contains

inf = float('inf')

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
        (subexpr,) = expr.args
        notexpr = Not(expr)
        interval = solver(notexpr)
        result = interval.complement(Reals)
    else:
        raise ValueError('Expression "%s" has unknown type.')

    if isinstance(result, ConditionSet):
        raise ValueError('Expression "%s" is not invertible.' % (expr,))

    return result

def factor_dnf(expr):
    symbols = get_symbols(expr)
    lookup = {s:s for s in symbols}
    return factor_dnf_symbols(expr, lookup)

def factor_dnf_symbols(expr, lookup):
    if isinstance(expr, Relational):
        # Literal term.
        symbols = get_symbols(expr)
        if len(symbols) > 1:
            raise ValueError('Expression "%s" has multiple symbols.' % (expr,))
        key = lookup[symbols[0]]
        return {key: expr}

    elif isinstance(expr, And):
        # Product term.
        subexprs = expr.args
        assert all(isinstance(e, Relational) for e in subexprs)
        mappings = [factor_dnf_symbols(subexpr, lookup) for subexpr in  subexprs]
        exprs = {}
        for mapping in mappings:
            assert len(mapping) == 1
            [(symbol, subexp)] = mapping.items()
            key = lookup[symbol]
            if key not in exprs:
                exprs[key] = subexp
            else:
                exprs[key] = And(subexp, exprs[key])
        return exprs

    elif isinstance(expr, Or):
        # Sum term.
        subexprs = expr.args
        mappings = [factor_dnf(subexpr) for subexpr in subexprs]
        exprs = {}
        for mapping in mappings:
            for symbol, subexp in mapping.items():
                key = lookup[symbol]
                if key not in exprs:
                    exprs[key] = subexp
                else:
                    exprs[key] = Or(subexp, exprs[key])
        return exprs
    else:
        assert False, 'Invalid DNF expression: %s' % (expr,)

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

def are_disjoint(sets):
    union = sets[0].union(*sets[1:])
    return len(union) == sum(len(s) for s in sets)

def are_identical(sets):
    intersection = sets[0].intersection(*sets[1:])
    assert all(len(s) == len(intersection) for s in sets)

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

class IntervalDistribution(Distribution):
    """Univariate continuous probability distribution on a real interval."""

    def __init__(self, symbol, dist, support, conditioned):
        assert isinstance(support, Interval)
        self.dist = dist
        self.support = support
        self.conditioned = conditioned
        self.xl = float(support.start)
        self.xh = float(support.end)
        if conditioned:
            self.lognorm = dist.logdiffexp(
                dist.logcdf(self.xl),
                dist.logcdf(self.xh))
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
            return IntervalDistribution(self.symbol, self.dist, support, True)
        elif isinstance(expression, Union):
            intervals = expression.args
            distributions = [
                IntervalDistribution(self.symbol, self.dist, interval, True)
                for interval in intervals
            ]
            weights_unorm = [self._logcdf_interval(i) for i in intervals]
            weights = lognorm(weights_unorm)
            return MixtureDistribution(distributions, weights)
        else:
            assert False, 'Unknown expression type: %s' % (expression,)


# There is work that needs to be done to handle Eq and Contains
# in the constraints. We should probably create our own
# subclasses of Relational with the needed implementations.
# We might need to make the domain S.Naturals, and then have
# solver either return FiniteSet(.) or FiniteSet.complement(S.Naturals).
def simplify_atomic(event):
    # For now we forbid Contains(X, Vals) since our solver does not
    # know how to simplify the negation of this expression.
    if isinstance(event, Eq):
        a, b = event.args
        return b if isinstance(a, Symbol) else a

class AtomicDistribution(Distribution):
    """Probability distribution on set of unorderd atoms."""

    def __init__(self, symbol, dist):
        self.symbol = symbol
        self.symbols = frozenset({symbol})
        self.dist = dict(dist)
        self.support = frozenset(dist.keys())
        # Verify the distribution is well-formed.
        for outcome in dist.keys():
            assert outcome in self.support
        assert allclose(sum(dist.values()),  1)

    def logpdf(self, x):
        return self.dist.get(x, -inf)

    def logprob(self, event):
        pass
