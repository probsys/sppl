# Copyright 2019 MIT Probabilistic Computing Project.
# See LICENSE.txt

from collections import Counter
from functools import reduce
from itertools import chain

from sympy import S as Singletons

from sympy import Intersection
from sympy import Interval
from sympy import Union

from sympy import Tuple
from sympy import to_dnf

from .dnf import factor_dnf_symbols

from .math_util import allclose
from .math_util import logdiffexp
from .math_util import logflip
from .math_util import lognorm
from .math_util import logsumexp

from .events import EventAnd
from .events import EventFinite
from .events import EventInterval
from .events import EventOr

from .solver import solver

from .sym_util import are_disjoint
from .sym_util import are_identical
from .sym_util import get_intersection
from .sym_util import get_symbols
from .sym_util import get_union

from .transforms import Identity

EmptySet = Singletons.EmptySet
inf = float('inf')

# ==============================================================================
# Distribution classes.

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

def simplify_nominal_event(event, support):
    if isinstance(event, EventInterval):
        raise ValueError('Nominal variables cannot be in real intervals: %s'
            % (event,))
    if isinstance(event, EventFinite):
        if not isinstance(event.expr, Identity):
            raise ValueError('Nominal variables cannot be transformed: %s'
                % (event.expr,))
        return support.difference(event.values) if event.complement \
            else support.intersection(event.values)
    if isinstance(event, EventAnd):
        values = [simplify_nominal_event(e, support) for e in event.events]
        return get_intersection(values)
    if isinstance(event, EventOr):
        values = [simplify_nominal_event(e, support) for e in event.events]
        return get_union(values)
    assert False, 'Unknown event %s' % (event,)
