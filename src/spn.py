# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from collections import ChainMap
from collections import Counter
from fractions import Fraction
from inspect import getfullargspec
from itertools import chain
from math import exp
from math import isfinite
from math import log

from sympy import S as Singletons

from sympy import Intersection
from sympy import Interval
from sympy import Range
from sympy import Union

from .dnf import dnf_to_disjoint_union
from .dnf import factor_dnf_symbols

from .math_util import allclose
from .math_util import flip
from .math_util import isinf_neg
from .math_util import logdiffexp
from .math_util import logflip
from .math_util import lognorm
from .math_util import logsumexp


from .sym_util import ContainersFinite
from .sym_util import are_disjoint
from .sym_util import are_identical
from .sym_util import get_intersection
from .sym_util import get_union
from .sym_util import powerset
from .sym_util import sympify_number

from .transforms import EventAnd
from .transforms import EventFinite
from .transforms import EventInterval
from .transforms import EventOr
from .transforms import Identity

EmptySet = Singletons.EmptySet
inf = float('inf')

# ==============================================================================
# SPN (base class).

class SPN(object):
    def __init__(self):
        raise NotImplementedError()
    def sample(self, N, rng):
        raise NotImplementedError()
    def sample_subset(self, symbols, N, rng):
        raise NotImplementedError()
    def sample_func(self, func, N, rng):
        raise NotImplementedError()
    def logprob(self, event):
        raise NotImplementedError()
    def prob(self, event):
        lp = self.logprob(event)
        return exp(lp)
    def logpdf(self, x):
        raise NotImplementedError()
    def condition(self, event):
        raise NotImplementedError()
    def mutual_information(self, A, B):
        # p11 = self.logprob(A & B)
        # p10 = self.logprob(A & ~B)
        # p01 = self.logprob(~A & B)
        # p00 = self.logprob(~A & ~B)
        lpA1 = self.logprob(A)
        lpB1 = self.logprob(B)
        lpA0 = logdiffexp(0, lpA1)
        lpB0 = logdiffexp(0, lpB1)
        lp00 = self.logprob(~A & ~B)
        lp01 = self.logprob(~A & B)
        lp10 = self.logprob(A & ~B)
        lp11 = self.logprob(A & B)
        m00 = exp(lp00) * (lp00 - (lpA0 + lpB0)) if not isinf_neg(lp00) else 0
        m01 = exp(lp01) * (lp01 - (lpA0 + lpB1)) if not isinf_neg(lp01) else 0
        m10 = exp(lp10) * (lp10 - (lpA1 + lpB0)) if not isinf_neg(lp10) else 0
        m11 = exp(lp11) * (lp11 - (lpA1 + lpB1)) if not isinf_neg(lp11) else 0
        return m00 + m01 + m10 + m11

    def __rmul__number(self, x):
        x_val = sympify_number(x)
        if not 0 < x < 1:
            raise ValueError('Weight %s must be in (0, 1)' % (str(x),))
        return PartialSumSPN([self], [x_val])
    def __rmul__(self, x):
        # Try to multiply x as a number.
        try:
            return self.__rmul__number(x)
        except TypeError:
            pass
        # Failed.
        return NotImplemented
    def __mul__(self, x):
        return x * self

    def __and__spn(self, x):
        if isinstance(x, PartialSumSPN):
            raise TypeError()
        if not isinstance(x, SPN):
            raise TypeError()
        return ProductSPN([self, x])
    def __and__(self, x):
        # Try to & x as a SPN.
        try:
            return self.__and__spn(x)
        except TypeError:
            pass
        # Failed.
        return NotImplemented

# ==============================================================================
# Sum SPN.

class SumSPN(SPN):
    """Weighted mixture of SPNs."""

    def __init__(self, spns, weights):
        self.children = tuple(spns)
        self.weights = tuple(weights)
        self.indexes = tuple(range(len(self.weights)))
        assert allclose(float(logsumexp(weights)),  0)

        symbols = [spn.get_symbols() for spn in spns]
        if not are_identical(symbols):
            raise ValueError('Mixture must have identical symbols.')
        self.symbols = self.children[0].get_symbols()

    def get_symbols(self):
        return self.symbols

    def sample(self, N, rng):
        f_sample = lambda i, n: self.children[i].sample(n, rng)
        return self.sample_many(f_sample, N, rng)

    def sample_subset(self, symbols, N, rng):
        f_sample = lambda i, n : \
            self.children[i].sample_subset(symbols, n, rng)
        return self.sample_many(f_sample, N, rng)

    def sample_func(self, func, N, rng):
        f_sample = lambda i, n : self.children[i].sample_func(func, n, rng)
        return self.sample_many(f_sample, N, rng)

    def sample_many(self, func, N, rng):
        selections = logflip(self.weights, self.indexes, N, rng)
        counts = Counter(selections)
        samples = [func(i, counts[i]) for i in counts]
        rng.shuffle(samples)
        return list(chain.from_iterable(samples))

    def logprob(self, event):
        logps = [spn.logprob(event) for spn in self.children]
        return logsumexp([p + w for (p, w) in zip(logps, self.weights)])

    def logpdf(self, x):
        logps = [spn.logpdf(x) for spn in self.children]
        return logsumexp([p + w for (p, w) in zip(logps, self.weights)])

    def condition(self, event):
        logps_condt = [spn.logprob(event) for spn in self.children]
        indexes = [i for i, lp in enumerate(logps_condt) if not isinf_neg(lp)]
        logps_joint = [logps_condt[i] + self.weights[i] for i in indexes]
        children = [self.children[i].condition(event) for i in indexes]
        weights = lognorm(logps_joint)
        return SumSPN(children, weights) if len(children) > 1 \
            else children[0]

class ExposedSumSPN(SumSPN):
    def __init__(self, spns, weights, symbol):
        """Weighted mixture of SPNs with exposed internal choice."""
        K = len(spns)
        nominals = [NominalDistribution(symbol, {i: 1}) for i in range(K)]
        spns_exposed = [
            ProductSPN([nominal, spn])
            for nominal, spn in zip(nominals, spns)
        ]
        super().__init__(spns_exposed, weights)

class PartialSumSPN(SPN):
    """Weighted mixture of SPNs that do not yet sum to unity."""
    def __init__(self, spns, weights):
        self.children = spns
        self.weights = weights
        self.indexes = list(range(len(self.weights)))
        assert sum(weights) <  1

        symbols = [spn.get_symbols() for spn in spns]
        if not are_identical(symbols):
            raise ValueError('Mixture must have identical symbols.')
        self.symbols = self.children[0].get_symbols()

    def __and__(self, x):
        raise TypeError('Weights do not sum to one.')
    def __rand__(self, x):
        raise TypeError('Weights do not sum to one.')
    def __mul__(self, x):
        raise TypeError('Cannot multiply PartialSumSPN by constant.')
    def __rmul__(self, x):
        raise TypeError('Cannot multiply PartialSumSPN by constant.')

    def __or__partialsum(self, x):
        if not isinstance(x, PartialSumSPN):
            raise TypeError()
        weights = self.weights + x.weights
        cumsum = float(sum(weights))
        if allclose(cumsum, 1):
            weights = [log(w) for w in weights]
            children = self.children + x.children
            return SumSPN(children, weights)
        if cumsum < 1:
            children = self.children + x.children
            return PartialSumSPN(children, weights)
        raise ValueError('Weights sum to more than one.')
    def __or__(self, x):
        # Try to | x as a PartialSumSPN
        try:
            return self.__or__partialsum(x)
        except TypeError:
            pass
        # Failed.
        return NotImplemented

# ==============================================================================
# Product base class.

class ProductSPN(SPN):
    """List of independent SPNs."""

    def __init__(self, spns):
        self.children = tuple(chain.from_iterable([
            (spn.children if isinstance(spn, type(self)) else [spn])
            for spn in spns
        ]))
        symbols = [spn.get_symbols() for spn in self.children]
        if not are_disjoint(symbols):
            raise ValueError('Product must have disjoint symbols')
        self.lookup = {s:i for i, syms in enumerate(symbols) for s in syms}
        self.symbols = frozenset(get_union(symbols))

    def get_symbols(self):
        return self.symbols

    def sample(self, N, rng):
        samples = [spn.sample(N, rng) for spn in self.children]
        return merge_samples(samples)

    def sample_subset(self, symbols, N, rng):
        # Partition symbols by lookup.
        index_to_symbols = {}
        for symbol in symbols:
            key = self.lookup[symbol]
            if key not in index_to_symbols:
                index_to_symbols[key] = [symbol]
            else:
                index_to_symbols[key].append(symbol)
        # Obtain the samples.
        samples = [
            self.children[i].sample_subset(symbols_i, N, rng)
            for i, symbols_i in index_to_symbols.items()
        ]
        # Merge the samples.
        return merge_samples(samples)

    def sample_func(self, func, N, rng):
        symbols = func_symbols(self, func)
        samples = self.sample_subset(symbols, N, rng)
        return func_evaluate(self, func, samples)

    def logpdf(self, x):
        assert len(x) == len(self.children)
        logps = [spn.logpdf(v) for (spn, v) in zip(self.children, x)
            if x is not None]
        return logsumexp(logps)

    def logprob(self, event):
        return self.logprob_disjoint_union(event)

    def logprob_inclusion_exclusion(self, event):
        # Adopting Inclusion--Exclusion principle:
        # https://cp-algorithms.com/combinatorics/inclusion-exclusion.html#toc-tgt-4
        expr_dnf = event.to_dnf()
        dnf_factor = factor_dnf_symbols(expr_dnf, self.lookup)
        indexes = range(len(dnf_factor))
        subsets = powerset(indexes, start=1)
        # Compute probabilities of all the conjunctions.
        (logps_pos, logps_neg) = ([], [])
        for J in subsets:
            # Find indexes of children that are involved in clauses J.
            keys = set(chain.from_iterable(dnf_factor[j].keys() for j in J))
            # Factorize events across the product.
            logprobs = [
                self.get_clause_weight_subset(dnf_factor, J, key)
                for key in keys
            ]
            logprob = sum(logprobs)
            # Add probability to either positive or negative sums.
            prefactor = (-1)**(len(J) - 1)
            x = logps_pos if prefactor > 0 else logps_neg
            x.append(logprob)
        # Aggregate positive term.
        logp_pos = logsumexp(logps_pos)
        if isinf_neg(logp_pos) or not logps_neg:
            return logp_pos
        # Aggregate negative terms and return the difference.
        logp_neg = logsumexp(logps_neg) if logps_neg else -inf
        return logdiffexp(logp_pos, logp_neg)

    def logprob_disjoint_union(self, event):
        # Adopting disjoint union principle.
        # Disjoint union algorithm (yields mixture of products).

        # Yields A or B or C
        expr_dnf = event.to_dnf()

        # Yields [A, B & ~A, C and ~A and ~B]
        exprs_disjoint = dnf_to_disjoint_union(expr_dnf)

        # Convert each item in exprs_disjoint to dnf.
        exprs_disjoint_dnf = [e.to_dnf() for e in exprs_disjoint]

        # Factor each DNF expression.
        exprs_disjoint_dnf_factors = [
            factor_dnf_symbols(e, self.lookup) for e in exprs_disjoint_dnf]

        # Obtain the clauses in each DNF expression.

        dnf_factor = factor_dnf_symbols(expr_dnf, self.lookup)
        # Obtain the n disjoint clauses.
        # clauses = [
        #     self.make_disjoint_conjunction(dnf_factor, i)
        #     for i in dnf_factor
        # ]
        # Construct the ProductSPN weights.
        ws = [self.get_clause_weight(clause) for clause in clauses]
        return logsumexp(ws)

    def condition(self, event):
        pass
        # Disjoint union algorithm (yields mixture of products).
        # expr_dnf = event.to_dnf()
        # dnf_factor = factor_dnf_symbols(expr_dnf, self.lookup)
        # # Obtain the n disjoint clauses.
        # clauses = [
        #     self.make_disjoint_conjunction(dnf_factor, i)
        #     for i in dnf_factor
        # ]
        # # Construct the ProductSPN weights.
        # ws = [self.get_clause_weight(clause) for clause in clauses]
        # indexes = [i for (i, w) in enumerate(ws) if not isinf_neg(w)]
        # if not indexes:
        #     raise ValueError('Conditioning event "%s" has probability zero' %
        #         (event,))
        # weights = lognorm([ws[i] for i in indexes])
        # # Construct the new ProductSPNs.
        # ds = [self.get_clause_conditioned(clauses[i]) for i in indexes]
        # products = [ProductSPN(d) for d in ds]
        # if len(products) == 1:
        #     return products[0]
        # # Return SumSPN of the products.
        # return SumSPN(products, weights)

    def make_disjoint_conjunction_factored(self, dnf_factor, i):
        clause = dict(dnf_factor[i])
        for j in range(i):
            for k in dnf_factor[j]:
                if k in clause:
                    clause[k] &= (~dnf_factor[j][k])
                else:
                    clause[k] = (~dnf_factor[j][k])
        return clause

    def get_clause_conditioned(self, clause):
        # Return children conditioned on a clause (one conjunction).
        return [
            spn.condition(clause[k]) if (k in clause) else spn
            for k, spn in enumerate(self.children)
        ]

    def get_clause_weight(self, clause):
        # Return probability of a clause (one conjunction).
        return sum([
            spn.logprob(clause[k]) if (k in clause) else 0
            for k, spn in enumerate(self.children)
        ])

    def get_clause_weight_subset(self, dnf_factor, J, key):
        # Return probability of conjunction of |J| clauses, for given key.
        events = [dnf_factor[j][key] for j in J if key in dnf_factor[j]]
        if not events:
            return -inf
        # Compute probability of events.
        event = events[0] if (len(events) == 1) else EventAnd(events)
        return self.children[key].logprob(event)

# ==============================================================================
# Basic Distribution base class.

class LeafSPN(SPN):
    # pylint: disable=no-member
    def get_symbols(self):
        return frozenset({self.symbol})
    def sample(self, N, rng):
        raise NotImplementedError()
    def sample_subset(self, symbols, N, rng):
        return self.sample(N, rng) if self.symbol in symbols else None
    def sample_func(self, func, N, rng):
        samples = self.sample(N, rng)
        return func_evaluate(self, func, samples)

# ==============================================================================
# RealDistribution base class.

class RealDistribution(LeafSPN):
    """Base class for distribution with a cumulative distribution function."""

    def __init__(self, symbol, dist, support, conditioned=None):
        assert isinstance(symbol, Identity)
        self.symbol = symbol
        self.dist = dist
        self.support = support
        self.conditioned = conditioned
        # Derived attributes.
        self.xl = float(support.inf)
        self.xu = float(support.sup)
        # Attributes to be populated by child classes.
        self.Fl = None
        self.Fu = None
        self.logFl = None
        self.logFu = None
        self.logZ = None

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
        return self.logprob_values(values)

    def logcdf(self, x):
        if not self.conditioned:
            return self.dist.logcdf(x)
        if self.xu < x:
            return 0
        elif x < self.xl:
            return -inf
        p = logdiffexp(self.dist.logcdf(x), self.logFl)
        return p - self.logZ

    def logpdf(self, x):
        raise NotImplementedError()

    def logprob_values(self, values):
        if values is EmptySet:
            return -inf
        if isinstance(values, ContainersFinite):
            return self.logprob_finite(values)
        if isinstance(values, Range):
            return self.logprob_range(values)
        if isinstance(values, Interval):
            return self.logprob_interval(values)
        if isinstance(values, Union):
            logps = [self.logprob_values(v) for v in values.args]
            return logsumexp(logps)
        assert False, 'Unknown set type: %s' % (values,)

    def logprob_finite(self, values):
        raise NotImplementedError()
    def logprob_range(self, values):
        raise NotImplementedError()
    def logprob_interval(self, values):
        raise NotImplementedError()

    def condition(self, event):
        interval = event.solve()
        values = Intersection(self.support, interval)
        weight = self.logprob_values(values)

        if isinf_neg(weight):
            raise ValueError('Conditioning event "%s" has probability zero'
                % (str(event)))

        if isinstance(values, (ContainersFinite, Range, Interval)):
            return (type(self))(self.symbol, self.dist, values, True)

        if isinstance(values, Union):
            weights_unorm = [self.logprob_values(v) for v in values.args]
            indexes = [i for i, w in enumerate(weights_unorm) if not isinf_neg(w)]
            if not indexes:
                raise ValueError('Conditioning event "%s" has probability zero'
                    % (str(event),))
            # TODO: Normalize the weights with greater precision, e.g.,
            # https://stats.stackexchange.com/questions/66616/converting-normalizing-very-small-likelihood-values-to-probability
            weights = lognorm([weights_unorm[i] for i in indexes])
            children = [
                (type(self))(self.symbol, self.dist, values.args[i], True)
                for i in indexes
            ]
            return SumSPN(children, weights) \
                if 1 < len(indexes) else children[0]

        assert False, 'Unknown set type: %s' % (values,)

# ==============================================================================
# Numerical distribution.

class NumericalDistribution(RealDistribution):
    """Non-atomic distribution with a cumulative distribution function."""
    def __init__(self, symbol, dist, support, conditioned=None):
        super().__init__(symbol, dist, support, conditioned)
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

    def logpdf(self, x):
        if not self.conditioned:
            return self.dist.logpdf(x)
        if x not in self.support:
            return -inf
        return self.dist.logpdf(x) - self.logZ

    def logprob_finite(self, values):
        return -inf

    def logprob_range(self, values):
        return -inf

    def logprob_interval(self, values):
        xl = float(values.start)
        xu = float(values.end)
        logFl = self.logcdf(xl)
        logFu = self.logcdf(xu)
        return logdiffexp(logFu, logFl)

# ==============================================================================
# Ordinal distribution.

class OrdinalDistribution(RealDistribution):
    """Atomic distribution with a cumulative distribution function."""

    def __init__(self, symbol, dist, support, conditioned=None):
        super().__init__(symbol, dist, support, conditioned)
        if conditioned:
            self.Fl = self.dist.cdf(self.xl - 1)
            self.Fu = self.dist.cdf(self.xu)
            self.logFl = self.dist.logcdf(self.xl - 1)
            self.logFu = self.dist.logcdf(self.xu)
            self.logZ = logdiffexp(self.logFu, self.logFl)
        else:
            self.logFl = -inf
            self.logFu = 0
            self.Fl = 0
            self.Fu = 1
            self.logZ = 1

    def logpdf(self, x):
        if not self.conditioned:
            return self.dist.logpmf(x)
        if (x < self.xl) or (self.xu < x):
            return -inf
        return self.dist.logpmf(x) - self.logZ

    def logprob_finite(self, values):
        logps = [self.logpdf(float(x)) for x in values]
        return logsumexp(logps)

    def logprob_range(self, values):
        if values.stop <= values.start:
            return -inf
        if values.step == 1:
            xl = float(values.inf)
            xu = float(values.sup)
            logFl = self.logcdf(xl - 1)
            logFu = self.logcdf(xu)
            return logdiffexp(logFu, logFl)
        if isfinite(values.start) and isfinite(values.stop):
            xs = list(values)
            return self.logprob_finite(xs)
        raise ValueError('Cannot enumerate infinite set: %s' % (values,))

    def logprob_interval(self, values):
        assert False, 'Atomic distribution cannot intersect an interval!'

# ==============================================================================
# Nominal distribution.

class NominalDistribution(LeafSPN):
    """Atomic distribution, no cumulative distribution function."""

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
        return log(self.dist[x]) if x in self.dist else -inf

    def logprob(self, event):
        # TODO: Consider using 1 - Pr[Event] for negation to avoid
        # iterating over domain.
        values = simplify_nominal_event(event, self.support)
        p_event = sum(self.dist[x] for x in values)
        return log(p_event) if p_event != 0 else -inf

    def condition(self, event):
        values = simplify_nominal_event(event, self.support)
        p_event = sum([self.dist[x] for x in values])
        if isinf_neg(p_event):
            raise ValueError('Conditioning event "%s" has probability zero' %
                (str(event),))
        dist = {
            x : (self.dist[x] / p_event) if x in values else 0
            for x in self.outcomes
        }
        return NominalDistribution(self.symbol, dist)

    def sample(self, N, rng):
        # TODO: Replace with FLDR.
        xs = flip(self.weights, self.outcomes, N, rng)
        return [{self.symbol: x} for x in xs]

# ==============================================================================
# Utilities.

def simplify_nominal_event(event, support):
    if isinstance(event, EventInterval):
        raise ValueError('Nominal variables cannot be in real intervals: %s'
            % (event,))
    if isinstance(event, EventFinite):
        if not isinstance(event.subexpr, Identity):
            raise ValueError('Nominal variables cannot be transformed: %s'
                % (event.subexpr,))
        return support.difference(event.values) if event.complement \
            else support.intersection(event.values)
    if isinstance(event, EventAnd):
        values = [simplify_nominal_event(e, support) for e in event.subexprs]
        return get_intersection(values)
    if isinstance(event, EventOr):
        values = [simplify_nominal_event(e, support) for e in event.subexprs]
        return get_union(values)
    assert False, 'Unknown event %s' % (str(event),)

def func_evaluate(spn, func, samples):
    args = func_symbols(spn, func)
    sample_kwargs = [{X.token: s[X] for X in args} for s in samples]
    return [func(**kwargs) for kwargs in sample_kwargs]

def func_symbols(spn, func):
    symbols = spn.get_symbols()
    args = [Identity(a) for a in getfullargspec(func).args]
    unknown = [a for a in args if a not in symbols]
    if unknown:
        raise ValueError('Unknown function arguments "%s" (allowed %s)'
            % (unknown, symbols))
    return args

def merge_samples(samples):
    # input [[{X:1, Y:2}, {X:0, Y:1}], [{Z:0}, {Z:1}]] (N=2)
    # output [{X:1, Y:2, Z:0}, {X:0, Y:1, Z:1}]
    return [dict(ChainMap(*sample_list)) for sample_list in zip(*samples)]
