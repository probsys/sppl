# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from collections import Counter
from fractions import Fraction
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
from .math_util import flip
from .math_util import isinf_neg
from .math_util import isinf_pos
from .math_util import logdiffexp
from .math_util import logflip
from .math_util import lognorm
from .math_util import logsumexp

from .events import EventAnd
from .events import EventFinite
from .events import EventInterval
from .events import EventOr

from .sym_util import ContainersFinite
from .sym_util import are_disjoint
from .sym_util import are_identical
from .sym_util import get_intersection
from .sym_util import get_symbols
from .sym_util import get_union
from .sym_util import sym_log

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
        assert allclose(float(logsumexp(weights)),  0)

        symbols = [d.get_symbols() for d in distributions]
        if not are_identical(symbols):
            raise ValueError('Dists in mixture must have identical symbols.')
        self.symbols = self.distributions[0].get_symbols()

    def get_symbols(self):
        return self.symbols

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
        rng.shuffle(samples)
        return list(chain.from_iterable(samples))

    def logprob(self, event):
        logps = [dist.logprob(event) for dist in self.distributions]
        return logsumexp([p + w for (p, w) in zip(logps, self.weights)])

    def logpdf(self, x):
        logps = [dist.logpdf(x) for dist in self.distributions]
        return logsumexp([p + w for (p, w) in zip(logps, self.weights)])

    def condition(self, event):
        logps_condt = [dist.logprob(event) for dist in self.distributions]
        indexes = [i for i, lp in enumerate(logps_condt) if not isinf_neg(lp)]
        logps_joint = [logps_condt[i] + self.weights[i] for i in indexes]
        dists = [self.distributions[i].condition(event) for i in indexes]
        weights = lognorm(logps_joint)
        return MixtureDistribution(dists, weights) if len(dists) > 1 \
            else dists[0]

class ProductDistribution(Distribution):
    """Vector of independent distributions."""

    def __init__(self, distributions):
        self.distributions = distributions

        symbols = [d.get_symbols() for d in distributions]
        assert are_disjoint(symbols), \
            'Distributions in Product must have disjoint symbols.'
        self.symbols = frozenset(get_union(symbols))
        self.lookup = {s:i for i, syms in enumerate(symbols) for s in symbols}

    def get_symbols(self):
        return self.symbols

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

class DistributionLeaf(Distribution):
    # pylint: disable=no-member
    def get_symbols(self):
        return frozenset({self.symbol})
    def sample_expr(self, expr, N, rng):
        samples = self.sample(N, rng)
        return [expr.evaluate(sample) for sample in samples]
    def sample_func(self, func, N, rng):
        samples = self.sample(N, rng)
        return sample_func(self, func, samples)

class NumericDistribution(DistributionLeaf):
    """Univariate distribution on a single real interval."""

    def __init__(self, symbol, dist, support, conditioned=None):
        assert isinstance(symbol, Identity)
        self.symbol = symbol
        self.dist = dist
        self.support = support
        self.conditioned = conditioned
        # Derived attributes.
        self.xl = float(support.start)
        self.xu = float(support.end)
        if conditioned:
            self.Fl = self.dist.cdf(self.xl)
            self.Fu = self.dist.cdf(self.xu)
            self.logFl = self.dist.logcdf(self.xl)
            self.logFu = self.dist.logcdf(self.xu)
            self.logZ = logdiffexp(self.logFu, self.logFl)
        else:
            self.logFl = -inf
            self.logFu = 0
            self.Fl = 0
            self.Fu = 1
            self.logZ = 1

    def sample(self, N, rng):
        if self.conditioned:
            # XXX Method not guaranteed to be numerically stable, see e.g,.
            # https://www.iro.umontreal.ca/~lecuyer/myftp/papers/truncated-normal-book-chapter.pdf
            # Also consider using CDF for left tail and SF for right tail.
            # Example: X ~ N(0,1) can sample X | (X < -10) but not X | (X > 10).
            u = rng.uniform(size=N)
            u_interval = u*self.Fl + (1-u) * self.Fu
            xs = self.dist.ppf(u_interval)
        else:
            # Simulation by vanilla inversion sampling.
            xs = self.dist.rvs(size=N, random_state=rng)
        # Wrap result in a dictionary.
        return [{self.symbol : x} for x in xs]

    def logprob(self, event):
        interval = event.solve()
        values = Intersection(self.support, interval)
        if values is EmptySet:
            return -inf
        if isinstance(values, ContainersFinite):
            # XXX Assuming no atoms.
            return -inf
        if isinstance(values, Interval):
            return self.logcdf_interval(values)
        if isinstance(values, Union):
            logps = [self.logcdf_interval(v) for v in values.args]
            return logsumexp(logps)
        assert False, 'Unknown set type: %s' % (values,)

    def logcdf(self, x):
        if not self.conditioned:
            return self.dist.logcdf(x)
        if self.xu <= x:
            return 0
        elif x <= self.xl:
            return -inf
        p = logdiffexp(self.dist.logcdf(x), self.Fl)
        return p - self.logZ

    def logpdf(self, x):
        if not self.conditioned:
            return self.dist.logpdf(x)
        if x not in self.support:
            return -inf
        return self.dist.logpdf(x) - self.logZ

    def logcdf_interval(self, interval):
        if interval == EmptySet:
            return -inf
        xl = float(interval.start)
        xh = float(interval.end)
        return logdiffexp(self.logcdf(xh), self.logcdf(xl))

    def condition(self, event):
        interval = event.solve()
        values = Intersection(self.support, interval)

        if values is EmptySet:
            raise ValueError('Conditioning event "%s" is empty "%s"'
                % (event, self.support))

        if isinstance(values, ContainersFinite):
            raise ValueError('Conditioning event "%s" has probability zero' %
                (event,))

        if isinstance(values, Interval):
            weight = self.logcdf_interval(values)
            assert -inf < weight
            return NumericDistribution(self.symbol, self.dist, values, True)

        if isinstance(values, Union):
            distributions = [
                NumericDistribution(self.symbol, self.dist, v, True)
                for v in values.args
            ]
            weights_unorm = [self.logcdf_interval(v) for v in values.args]
            # TODO: Normalize the weights with greater precision, e.g.,
            # https://stats.stackexchange.com/questions/66616/converting-normalizing-very-small-likelihood-values-to-probability
            weights = lognorm(weights_unorm)
            return MixtureDistribution(distributions, weights)

        assert False, 'Unknown set type: %s' % (interval,)

class NominalDistribution(DistributionLeaf):
    """Univariate distribution on set of unordered, non-numeric atoms."""

    def __init__(self, symbol, dist):
        assert isinstance(symbol, Identity)
        self.symbol = symbol
        self.dist = {x: Fraction(w) for x, w in dist.items()}
        # Derived attributes.
        self.support = frozenset(self.dist)
        self.outcomes = list(self.dist.keys())
        self.weights = [float(x) for x in self.dist.values()]
        assert allclose(float(sum(self.weights)),  1)

    def logpdf(self, x):
        return sym_log(self.dist[x]) if x in self.dist else -inf

    def logprob(self, event):
        values = simplify_nominal_event(event, self.support)
        p_event = sum(self.dist[x] for x in values)
        return sym_log(p_event)

    def condition(self, event):
        values = simplify_nominal_event(event, self.support)
        p_event = sum([self.dist[x] for x in values])
        if isinf_neg(p_event):
            raise ValueError('Cannot condition on zero probability event: %s'
                % (str(event),))
        dist = {
            x : (self.dist[x] / p_event) if x in values else 0
            for x in self.outcomes
        }
        return NominalDistribution(self.symbol, dist)

    def sample(self, N, rng):
        # TODO: Replace with FLDR.
        xs = flip(self.weights, self.outcomes, N, rng)
        return [{self.symbol: x} for x in xs]

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
    assert False, 'Unknown event %s' % (str(event),)

def sample_func(dist, func, samples):
    from inspect import getfullargspec
    args = getfullargspec(func).args
    tokens = {symbol.token for symbol in dist.get_symbols()}
    unknown = [a for a in args if a not in tokens]
    if unknown:
        raise ValueError('Unknown function arguments "%s" (allowed %s)'
            % (unknown, tokens))
    sample_kwargs = [{a: sample[Identity(a)] for a in args} for sample in samples]
    return [func(**kwargs) for kwargs in sample_kwargs]
