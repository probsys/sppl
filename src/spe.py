# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from collections import ChainMap
from collections import Counter
from collections import OrderedDict
from fractions import Fraction
from functools import reduce
from inspect import getfullargspec
from itertools import chain
from math import exp
from math import log

from .dnf import dnf_factor
from .dnf import dnf_normalize
from .dnf import dnf_to_disjoint_union

from .math_util import allclose
from .math_util import flip
from .math_util import float_to_int
from .math_util import int_or_isinf_neg
from .math_util import int_or_isinf_pos
from .math_util import isinf_neg
from .math_util import logdiffexp
from .math_util import logflip
from .math_util import lognorm
from .math_util import logsumexp
from .math_util import random

from .sym_util import are_disjoint
from .sym_util import are_identical
from .sym_util import get_union
from .sym_util import partition_finite_real_contiguous
from .sym_util import partition_list_blocks
from .sym_util import sympify_number

from .transforms import EventOr
from .transforms import Id

from .sets import EmptySet
from .sets import FiniteNominal
from .sets import FiniteReal
from .sets import Interval
from .sets import Range
from .sets import Union

inf = float('inf')

def memoize(f):
    table = f.__name__.split('_')[0]
    def f_(*args):
        (spe, event_factor, memo) = args
        if memo is False:
            return f(spe, event_factor_to_event, memo)
        m = getattr(memo, table)
        key = spe.get_memo_key(event_factor)
        if key not in m:
            m[key] = f(spe, event_factor, memo)
        return m[key]
    return f_

# ==============================================================================
# SPE (base class).

class SPE():
    env = None             # Environment mapping symbols to transforms.
    def __init__(self):
        raise NotImplementedError()
    def get_symbols(self):
        raise NotImplementedError()
    def size(self):
        raise NotImplementedError
    def sample(self, N, prng=None):
        raise NotImplementedError()
    def sample_subset(self, symbols, N, prng=None):
        raise NotImplementedError()
    def sample_func(self, func, N, prng=None):
        raise NotImplementedError()
    def transform(self, symbol, expr):
        raise NotImplementedError()
    def logprob(self, event, memo=None):
        raise NotImplementedError()
    def condition(self, event, memo=None):
        raise NotImplementedError()
    def logpdf(self, assignment, memo=None):
        raise NotImplementedError()
    def constrain(self, assignment, memo=None):
        raise NotImplementedError()
    def mutual_information(self, A, B, memo=None):
        if memo is None:
            memo = Memo()
        lpA1 = self.logprob(A)
        lpB1 = self.logprob(B)
        lpA0 = logdiffexp(0, lpA1)
        lpB0 = logdiffexp(0, lpB1)
        lp11 = self.logprob(A & B, memo)
        lp10 = self.logprob(A & ~B, memo)
        lp01 = self.logprob(~A & B, memo)
        # lp00 = self.logprob(~A & ~B, memo)
        lp00 = logdiffexp(0, logsumexp([lp11, lp10, lp01]))
        m11 = exp(lp11) * (lp11 - (lpA1 + lpB1)) if not isinf_neg(lp11) else 0
        m10 = exp(lp10) * (lp10 - (lpA1 + lpB0)) if not isinf_neg(lp10) else 0
        m01 = exp(lp01) * (lp01 - (lpA0 + lpB1)) if not isinf_neg(lp01) else 0
        m00 = exp(lp00) * (lp00 - (lpA0 + lpB0)) if not isinf_neg(lp00) else 0
        return m11 + m10 + m01 + m00
    def prob(self, event):
        lp = self.logprob(event)
        return exp(lp)
    def pdf(self, assignment):
        lp = self.logpdf(assignment)
        return exp(lp)

    def __rmul__number(self, x):
        x_val = sympify_number(x)
        if not 0 < x < 1:
            raise ValueError('Weight %s must be in (0, 1)' % (str(x),))
        return PartialSumSPE([self], [x_val])
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

    def __and__spe(self, x):
        if isinstance(x, PartialSumSPE):
            raise TypeError()
        if not isinstance(x, SPE):
            raise TypeError()
        return ProductSPE([self, x])
    def __and__(self, x):
        # Try to & x as a SPE.
        try:
            return self.__and__spe(x)
        except TypeError:
            pass
        # Failed.
        return NotImplemented

    def get_memo_key(self, event_factor):
        x = id(self)
        y = tuple(event_factor.items()) \
                if isinstance(event_factor, dict) \
                else tuple(tuple(d.items()) for d in event_factor)
        return (x, y)

# ==============================================================================
# Branch SPE.

class BranchSPE(SPE):
    symbols = None
    children = None
    def get_symbols(self):
        return self.symbols
    def size(self):
        return 1 + sum(c.size() for c in self.children)
    def logprob(self, event, memo=None):
        if memo is None:
            memo = Memo()
        event_dnf = dnf_normalize(event)
        if event_dnf is None:
            return -inf
        event_factor = dnf_factor(event_dnf)
        return self.logprob_mem(event_factor, memo)
    def condition(self, event, memo=None):
        if memo is None:
            memo = Memo()
        event_dnf = dnf_normalize(event)
        if event_dnf is None:
            raise ValueError('Zero probability event: %s' % (event,))
        if isinstance(event_dnf, EventOr):
            conjunctions = [dnf_factor(e) for e in event_dnf.subexprs]
            logps = [self.logprob_mem(c, memo) for c in conjunctions]
            indexes = [i for i, lp in enumerate(logps) if not isinf_neg(lp)]
            if not indexes:
                raise ValueError('Zero probability event: %s' % (event,))
            event_dnf = EventOr([event_dnf.subexprs[i] for i in indexes])
        event_disjoint = dnf_to_disjoint_union(event_dnf)
        event_factor = dnf_factor(event_disjoint)
        return self.condition_mem(event_factor, memo)
    def logpdf(self, assignment, memo=None):
        if memo is None:
            memo = Memo()
        return self.logpdf_mem(assignment, memo)[1]
    def constrain(self, assignment, memo=None):
        if memo is None:
            memo = Memo()
        return self.constrain_mem(assignment, memo)
    def logprob_mem(self, event_factor, memo):
        raise NotImplementedError()
    def condition_mem(self, event_factor, memo):
        raise NotImplementedError()
    def logpdf_mem(self, assignment, memo):
        raise NotImplementedError()
    def constrain_mem(self, assignment, memo):
        raise NotImplementedError()

# ==============================================================================
# Sum SPE.

class SumSPE(BranchSPE):
    """Weighted mixture of SPEs."""

    def __init__(self, children, weights):
        assert len(children) == len(weights)
        self.children = tuple(chain.from_iterable([
            spe.children
                if isinstance(spe, type(self)) else [spe]
            for spe in children
        ]))
        self.weights = tuple(chain.from_iterable([
            [weight + w for w in spe.weights]
                if isinstance(spe, type(self)) else [weight]
            for spe, weight in zip(children, weights)
        ]))
        # Derived attributes.
        self.indexes = tuple(range(len(self.weights)))
        assert allclose(float(logsumexp(weights)),  0)

        symbols = [spe.get_symbols() for spe in self.children]
        if not are_identical(symbols):
            syms = '\n'.join([', '.join(sorted(str(x) for x in s)) for s in symbols])
            raise ValueError('Mixture must have identical symbols:\n%s' % (syms,))
        self.symbols = self.children[0].get_symbols()

    def sample(self, N, prng=None):
        f_sample = lambda i, n: self.children[i].sample(n, prng=prng)
        return self.sample_many(f_sample, N, prng=prng)

    def sample_subset(self, symbols, N, prng=None):
        f_sample = lambda i, n : \
            self.children[i].sample_subset(symbols, n, prng=prng)
        return self.sample_many(f_sample, N, prng=prng)

    def sample_func(self, func, N, prng=None):
        f_sample = lambda i, n : self.children[i].sample_func(func, n, prng=prng)
        return self.sample_many(f_sample, N, prng=prng)

    def sample_many(self, func, N, prng=None):
        selections = logflip(self.weights, self.indexes, N, prng)
        counts = Counter(selections)
        samples = [func(i, counts[i]) for i in counts]
        random(prng).shuffle(samples)
        return list(chain.from_iterable(samples))

    def transform(self, symbol, expr):
        children = [spe.transform(symbol, expr) for spe in self.children]
        return SumSPE(children, self.weights)

    @memoize
    def logprob_mem(self, event_factor, memo):
        logps = [spe.logprob_mem(event_factor, memo) for spe in self.children]
        logp = logsumexp([p + w for (p, w) in zip(logps, self.weights)])
        return logp

    @memoize
    def condition_mem(self, event_factor, memo):
        logps_condt = [spe.logprob_mem(event_factor, memo) for spe in self.children]
        indexes = [i for i, lp in enumerate(logps_condt) if not isinf_neg(lp)]
        if not indexes:
            raise ValueError('Conditioning event "%s" has probability zero' % (str(event_factor),))
        logps_joint = [logps_condt[i] + self.weights[i] for i in indexes]
        children = [self.children[i].condition_mem(event_factor, memo) for i in indexes]
        weights = lognorm(logps_joint)
        return SumSPE(children, weights) if len(indexes) > 1 else children[0]

    @memoize
    def logpdf_mem(self, assignment, memo):
        logps = [spe.logpdf_mem(assignment, memo) for spe in self.children]
        logps_noninf = [(d, w) for d, w in logps if not isinf_neg(w)]
        if len(logps_noninf) == 0:
            return (0, -inf)
        d_min = min(d for (d, w) in logps_noninf)
        lp = [p + w for (d, w), p in zip(logps, self.weights) if d == d_min]
        return (d_min, logsumexp(lp))

    @memoize
    def constrain_mem(self, assignment, memo):
        logpdfs_condt = [spe.logpdf_mem(assignment, memo) for spe in self.children]
        indexes = [i for i, (d, l) in enumerate(logpdfs_condt) if not isinf_neg(l)]
        assert indexes, 'Assignment "%s" has density zero' % (str(assignment),)
        d_min = min(logpdfs_condt[i][0] for i in indexes)
        indexes_d_min = [i for i in indexes if logpdfs_condt[i][0] == d_min]
        logpdfs = [logpdfs_condt[i][1] + self.weights[i] for i in indexes_d_min]
        children = [self.children[i].constrain(assignment, memo) for i in indexes_d_min]
        weights = lognorm(logpdfs)
        return SumSPE(children, weights) if len(indexes_d_min) > 1 else children[0]

    def __eq__(self, x):
        return isinstance(x, type(self)) \
            and self.children == x.children \
            and self.weights == x.weights
    def __hash__(self):
        x = (self.__class__, self.children, self.weights)
        return hash(x)

class ExposedSumSPE(SumSPE):
    def __init__(self, children, spe_weights):
        """Weighted mixture of SPEs with exposed internal choice."""
        assert isinstance(spe_weights, NominalLeaf)
        weights = [
            spe_weights.logprob(spe_weights.symbol << {n})
            for n in spe_weights.support
        ]
        children = [
            ProductSPE([
                NominalLeaf(spe_weights.symbol, {str(n): 1}),
                children[n]
            ]) for n in spe_weights.support
        ]
        super().__init__(children, weights)

class PartialSumSPE(SPE):
    """Weighted mixture of SPEs that do not yet sum to unity."""
    def __init__(self, children, weights):
        self.children = children
        self.weights = weights
        self.indexes = list(range(len(self.weights)))
        assert sum(weights) <  1

        symbols = [spe.get_symbols() for spe in children]
        if not are_identical(symbols):
            raise ValueError('Mixture must have identical symbols.')
        self.symbols = self.children[0].get_symbols()

    def __and__(self, x):
        raise TypeError('Weights do not sum to one.')
    def __rand__(self, x):
        raise TypeError('Weights do not sum to one.')
    def __mul__(self, x):
        raise TypeError('Cannot multiply PartialSumSPE by constant.')
    def __rmul__(self, x):
        raise TypeError('Cannot multiply PartialSumSPE by constant.')

    def __or__partialsum(self, x):
        if not isinstance(x, PartialSumSPE):
            raise TypeError()
        weights = self.weights + x.weights
        cumsum = float(sum(weights))
        if allclose(cumsum, 1):
            weights = [log(w) for w in weights]
            children = self.children + x.children
            return SumSPE(children, weights)
        if cumsum < 1:
            children = self.children + x.children
            return PartialSumSPE(children, weights)
        raise ValueError('Weights sum to more than one.')
    def __or__(self, x):
        # Try to | x as a PartialSumSPE
        try:
            return self.__or__partialsum(x)
        except TypeError:
            pass
        # Failed.
        return NotImplemented

def spe_simplify_sum(spe):
    if isinstance(spe.children[0], UnivariateLeafSPE):
        return spe_simplify_sum_leaf(spe)
    if isinstance(spe.children[0], ProductSPE):
        return spe_simplify_sum_product(spe)
    assert False, 'Invalid children of SumSPE: %s' % (spe.children,)

def spe_simplify_sum_leaf(spe):
    assert all(isinstance(c, UnivariateLeafSPE) for c in spe.children)
    partition = partition_list_blocks(spe.children)
    if len(partition) == len(spe.children):
        return spe
    if len(partition) == 1:
        return spe.children[0]
    children = [spe.children[block[0]] for block in partition]
    weights = [logsumexp([spe.weights[i] for i in block]) for block in partition]
    return SumSPE(children, weights)

def spe_simplify_sum_product(spe):
    # TODO: Handle case when some children are leaves with environments;
    # e.g. SumSPE([X & Y, (X & Y=X**2)])
    if not all(isinstance(c, ProductSPE) for c in spe.children):
        return spe
    children_list = [c.children for c in spe.children]
    children_simplified, weight_simplified = reduce(
        lambda state, cw: spe_simplify_sum_product_helper(state, cw[0], cw[1]),
        zip(children_list[1:], spe.weights[1:]),
        (children_list[0], spe.weights[0]),
    )
    assert allclose(logsumexp(weight_simplified), 0)
    return spe_list_to_product(children_simplified)

def spe_simplify_sum_product_helper(state, children_b, w_b):
    (children_a, w_a) = state
    weights_sum = lognorm([w_a, w_b])
    weight_overall = logsumexp([w_a, w_b])
    overlap = [(i, j)
        for j, cb in enumerate(children_b)
        for i, ca in enumerate(children_a)
        if ca == cb
    ]
    if len(overlap) == 0:
        product_a = spe_list_to_product(children_a)
        product_b = spe_list_to_product(children_b)
        children_simplified = [SumSPE([product_a, product_b], weights_sum)]
    elif len(overlap) == len(children_a):
        children_simplified = children_a
    else:
        dup_b = set(p[1] for p in overlap)
        dup_a = set(p[0] for p in overlap)
        uniq_children_b = [c for j, c in enumerate(children_b) if j not in dup_b]
        uniq_children_a = [c for i, c in enumerate(children_a) if i not in dup_a]
        dup_children = [c for i, c in enumerate(children_a) if i in dup_a]
        product_a = spe_list_to_product(uniq_children_a)
        product_b = spe_list_to_product(uniq_children_b)
        sum_a_b = SumSPE([product_a, product_b], weights_sum)
        children_simplified = [sum_a_b] + dup_children
    return (children_simplified, weight_overall)

# ==============================================================================
# Product base class.

class ProductSPE(BranchSPE):
    """List of independent SPEs."""

    def __init__(self, children):
        self.children = tuple(chain.from_iterable([
            (spe.children if isinstance(spe, type(self)) else [spe])
            for spe in children
        ]))
        # Derived attributes.
        symbols = [spe.get_symbols() for spe in self.children]
        if not are_disjoint(symbols):
            syms = '\n'.join([', '.join(sorted(str(x) for x in s)) for s in symbols])
            raise ValueError('Product must have disjoint symbols:\n%s' % (syms,))
        self.lookup = {s:i for i, syms in enumerate(symbols) for s in syms}
        self.symbols = frozenset(get_union(symbols))

    def sample(self, N, prng=None):
        samples = [spe.sample(N, prng=prng) for spe in self.children]
        return merge_samples(samples)

    def sample_subset(self, symbols, N, prng=None):
        # Partition symbols by lookup.
        index_to_symbols = {}
        for symbol in symbols:
            key = self.lookup[symbol]
            if key not in index_to_symbols:
                index_to_symbols[key] = []
            index_to_symbols[key].append(symbol)
        # Obtain the samples.
        samples = [
            self.children[i].sample_subset(symbols_i, N, prng=prng)
            for i, symbols_i in index_to_symbols.items()
        ]
        # Merge the samples.
        return merge_samples(samples)

    def sample_func(self, func, N, prng=None):
        symbols = func_symbols(self, func)
        samples = self.sample_subset(symbols, N, prng=prng)
        return func_evaluate(self, func, samples)

    def transform(self, symbol, expr):
        # TODO: This algorithm does not handle the case that expr has symbols
        # belonging to different children, specifically defining symbol
        # to be a predicate such as (X > 0) || (Y < 2).
        # The correct solution is to implement an environment in the
        # Product, and perform substitution on the event (recursively,
        # unfortunately).
        expr_symbols = expr.get_symbols()
        assert all(e in self.get_symbols() for e in expr_symbols)
        index = [
            i for i, spe in enumerate(self.children)
            if all(s in spe.get_symbols() for s in expr_symbols)
        ]
        assert len(index) == 1, 'No child has all symbols in: %s' % (expr,)
        children = list(self.children)
        children[index[0]] = children[index[0]].transform(symbol, expr)
        return ProductSPE(children)

    @memoize
    def logprob_mem(self, event_factor, memo):
        # Adopting Inclusion--Exclusion principle for DNF event:
        # https://cp-algorithms.com/combinatorics/inclusion-exclusion.html#toc-tgt-4
        (logps_pos, logps_neg) = ([], [])
        indexes = range(len(event_factor))
        stack = [([i], i) for i in indexes]
        avoid = []
        while stack:
            # Obtain the next subset.
            subset, index = stack.pop(0)
            # Skip descendants of this subset if it contains a bad subset.
            if any(
                    len(b) <= len(subset) and all(z in subset for z in b)
                    for b in avoid):
                continue
            # Compute the probability of this subset.
            logprob = self.logprob_conjunction(event_factor, subset, memo)
            (logps_pos if len(subset) % 2 else logps_neg).append(logprob)
            # Skip descendants of this subset if measure zero.
            if isinf_neg(logprob):
                avoid.append(subset)
            # Add all subsets for which this subset is a prefix.
            for i in range(index + 1, len(indexes)):
                stack.append((subset + [indexes[i]], i))
        # Aggregate positive term.
        logp_pos = logsumexp(logps_pos)
        if isinf_neg(logp_pos) or not logps_neg:
            return logp_pos
        # Aggregate negative terms and return the difference.
        logp_neg = logsumexp(logps_neg) if logps_neg else -inf
        return logdiffexp(logp_pos, logp_neg)

    @memoize
    def condition_mem(self, event_factor, memo):
        logps = [self.logprob_conjunction([c], [0], memo) for c in event_factor]
        assert allclose(logsumexp(logps), self.logprob_mem(event_factor, memo))
        indexes = [i for (i, lp) in enumerate(logps) if not isinf_neg(lp)]
        if not indexes:
            raise ValueError('Conditioning event "%s" has probability zero'
                % (str(event_factor),))
        weights = lognorm([logps[i] for i in indexes])
        childrens = [self.condition_clause(event_factor[i], memo) for i in indexes]
        products = [ProductSPE(children) for children in childrens]
        if len(indexes) == 1:
            spe = products[0]
        else:
            spe_sum = SumSPE(products, weights)
            spe = spe_simplify_sum(spe_sum)
        return spe

    def logprob_conjunction(self, event_factor, J, memo):
        # Return probability of conjunction of |J| conjunctions.
        keys = set(self.lookup[s] for j in J for s in event_factor[j])
        return sum(
            self.logprob_conjunction_key(event_factor, J, key, memo)
            for key in keys
        )

    def logprob_conjunction_key(self, event_factor, J, key, memo):
        # Return probability of conjunction of |J| conjunction, for given key.
        clause = {}
        for j in J:
            for symbol, event in event_factor[j].items():
                if self.lookup[symbol] == key:
                    if symbol not in clause:
                        clause[symbol] = event
                    else:
                        clause[symbol] &= event
        if not clause:
            return -inf
        return self.children[key].logprob_mem((clause,), memo)

    def condition_clause(self, clause, memo):
        # Return children conditioned on a clause (one conjunction).
        children = []
        for spe in self.children:
            spe_condition = spe
            symbols = spe.get_symbols().intersection(clause)
            if symbols:
                spe_clause = ({symbol: clause[symbol] for symbol in symbols},)
                spe_condition = spe.condition_mem(spe_clause, memo)
            children.append(spe_condition)
        return children

    @memoize
    def logpdf_mem(self, assignment, memo):
        assignments = {}
        for symbol, value in assignment.items():
            key = self.lookup[symbol]
            if key not in assignments:
                assignments[key] = dict()
            assignments[key][symbol] = value
        return reduce(lambda x, s: (x[0]+s[0], x[1]+s[1]),
            (self.children[k].logpdf_mem(a, memo) for k, a in assignments.items()))

    @memoize
    def constrain_mem(self, assignment, memo):
        children = []
        for spe in self.children:
            spe_constrain = spe
            symbols = spe.get_symbols().intersection(assignment.keys())
            if symbols:
                spe_assignment = {s: assignment[s] for s in symbols}
                spe_constrain = spe.constrain_mem(spe_assignment, memo)
            children.append(spe_constrain)
        return ProductSPE(children)

    def __eq__(self, x):
        return isinstance(x, type(self)) \
            and self.children == x.children
    def __hash__(self):
        x = (self.__class__, self.children)
        return hash(x)

def spe_list_to_product(children):
    return children[0] if len(children) == 1 else ProductSPE(children)

# ==============================================================================
# Basic Distribution base class.

class UnivariateLeafSPE(SPE):
    atomic = None          # True if distribution has an atom
    symbol = None          # Symbol (Id) of base random variable
    def get_symbols(self):
        return frozenset(self.env)
    def size(self):
        return 1
    def sample(self, N, prng=None):
        return self.sample_subset(self.get_symbols(), N, prng=prng)
    def sample_subset(self, symbols, N, prng=None):
        assert all(s in self.get_symbols() for s in symbols)
        samples = self.sample__(N, prng)
        if symbols == {self.symbol}:
            return samples
        simulations = [{}] * N
        for i, sample in enumerate(samples):
            simulations[i] = dict()
            # Topological order guaranteed by OrderedDict.
            for symbol in self.env:
                sample[symbol] = self.env[symbol].evaluate(sample)
                if symbol in symbols:
                    simulations[i][symbol] = sample[symbol]
        return simulations
    def sample_func(self, func, N, prng=None):
        samples = self.sample(N, prng=prng)
        return func_evaluate(self, func, samples)
    def logprob(self, event, memo=None):
        event_subs = event.substitute(self.env)
        assert all(s in self.env for s in event.get_symbols())
        assert event_subs.get_symbols() == {self.symbol}
        if memo is None or memo is False:
            return self.logprob__(event_subs)
        key = self.get_memo_key(({self.symbol: event_subs},))
        if key not in memo.logprob:
            memo.logprob[key] = self.logprob__(event_subs)
        return memo.logprob[key]
    def condition(self, event, memo=None):
        event_subs = event.substitute(self.env)
        assert all(s in self.env for s in event.get_symbols())
        assert event_subs.get_symbols() == {self.symbol}
        if memo is None or memo is False:
            return self.condition__(event_subs)
        key = self.get_memo_key(({self.symbol: event_subs},))
        if key not in memo.condition:
            memo.condition[key] = self.condition__(event_subs)
        return memo.condition[key]
    def logpdf(self, assignment, memo=None):
        if memo is None:
            memo = Memo()
        return self.logpdf_mem(assignment, memo)[1]
    def constrain(self, assignment, memo=None):
        if memo is None:
            memo = Memo()
        return self.constrain_mem(assignment, memo)
    def logprob_mem(self, event_factor, memo):
        if memo is False:
            event = event_factor_to_event(event_factor)
            return self.logprob(event)
        key = self.get_memo_key(event_factor)
        if key not in memo.logprob:
            event = event_factor_to_event(event_factor)
            memo.logprob[key] = self.logprob(event)
        return memo.logprob[key]
    def condition_mem(self, event_factor, memo):
        if memo is False:
            event = event_factor_to_event(event_factor)
            return self.condition(event)
        key = self.get_memo_key(event_factor)
        if key not in memo.condition:
            event = event_factor_to_event(event_factor)
            memo.condition[key] = self.condition(event)
        return memo.condition[key]
    @memoize
    def logpdf_mem(self, assignment, memo):
        assert len(assignment) == 1
        [(k, v)] = assignment.items()
        assert k == self.symbol
        w = self.logpdf__(v)
        return (1 - self.atomic, w)
    @memoize
    def constrain_mem(self, assignment, memo):
        assert len(assignment) == 1
        [(k, v)] = assignment.items()
        assert k == self.symbol
        return self.constrain__(v)
    def sample__(self, N, prng):
        raise NotImplementedError()
    def logprob__(self, event):
        raise NotImplementedError()
    def condition__(self, event):
        raise NotImplementedError()
    def logpdf__(self, x):
        raise NotImplementedError()
    def constrain__(self, x):
        raise NotImplementedError()

# ==============================================================================
# RealLeaf base class.

class RealLeaf(UnivariateLeafSPE):
    """Base class for distribution with a cumulative distribution function."""

    def __init__(self, symbol, dist, support, conditioned=None, env=None):
        assert isinstance(symbol, Id)
        assert isinstance(support, Interval)
        self.symbol = symbol
        self.dist = dist
        self.support = support
        self.conditioned = conditioned
        self.env = env or OrderedDict([(symbol, symbol)])
        # Attributes to be populated by child classes.
        self.xl = None
        self.xu = None
        self.Fl = None
        self.Fu = None
        self.logFl = None
        self.logFu = None
        self.logZ = None

    def transform(self, symbol, expr):
        assert symbol not in self.env
        assert all(s in self.env for s in expr.get_symbols())
        env = OrderedDict(self.env)
        env[symbol] = expr
        return (type(self))(self.symbol, self.dist, self.support,
            self.conditioned, env)

    def sample__(self, N, prng):
        if self.conditioned:
            # XXX Method not guaranteed to be numerically stable, see e.g,.
            # https://www.iro.umontreal.ca/~lecuyer/myftp/papers/truncated-normal-book-chapter.pdf
            # Also consider using CDF for left tail and SF for right tail.
            # Example: X ~ N(0,1) can sample X | (X < -10) but not X | (X > 10).
            u = random(prng).uniform(size=N)
            u_interval = u*self.Fl + (1-u) * self.Fu
            xs = self.dist.ppf(u_interval)
        else:
            # Simulation by vanilla inversion sampling.
            xs = self.dist.rvs(size=N, random_state=prng)
        # Wrap result in a dictionary.
        return [{self.symbol : x} for x in xs]

    def logcdf(self, x):
        if not self.conditioned:
            return self.dist.logcdf(x)
        if self.xu < x:
            return 0
        elif x < self.xl:
            return -inf
        p = logdiffexp(self.dist.logcdf(x), self.logFl)
        return p - self.logZ

    def logprob__(self, event):
        interval = event.solve()
        values = self.support & interval
        return self.logprob_values__(values)

    def logprob_values__(self, values):
        if values is EmptySet:
            return -inf
        if isinstance(values, FiniteReal):
            return self.logprob_finite__(values)
        if isinstance(values, Interval):
            return self.logprob_interval__(values)
        if isinstance(values, Union):
            logps = [self.logprob_values__(v) for v in values.args]
            return logsumexp(logps)
        assert False, 'Unknown set type: %s' % (values,)

    def logprob_finite__(self, values):
        raise NotImplementedError()
    def logprob_interval__(self, values):
        raise NotImplementedError()

    def flatten_values_contiguous(self, values):
        if isinstance(values, Interval):
            return [values]
        if isinstance(values, FiniteReal):
            assert isinstance(self, DiscreteLeaf)
            blocks = partition_finite_real_contiguous(values)
            return [Range(min(v), max(v)) for v in blocks]
        if isinstance(values, Union):
            subvalues = (self.flatten_values_contiguous(v) for v in values)
            return list(chain(*subvalues))
        assert False

    def condition__(self, event):
        interval = event.solve()
        values_set = self.support & interval
        weight = self.logprob_values__(values_set)
        # Probability zero event.
        if isinf_neg(weight):
            raise ValueError('Conditioning event "%s" has probability zero'
                % (str(event)))
        # Condition on support.
        if values_set == self.support:
            return self
        # Flatten the set.
        values = self.flatten_values_contiguous(values_set)
        # Condition on a single contiguous set.
        if len(values) == 0:
            return (type(self))(self.symbol, self.dist, values[0], True, self.env)
        # Condition on a union of contiguous set.
        else:
            weights_unorm = [self.logprob_values__(v) for v in values]
            indexes = [i for i, w in enumerate(weights_unorm) if not isinf_neg(w)]
            if not indexes:
                raise ValueError('Conditioning event "%s" has probability zero'
                    % (str(event),))
            # TODO: Normalize the weights with greater precision, e.g.,
            # https://stats.stackexchange.com/questions/66616/converting-normalizing-very-small-likelihood-values-to-probability
            weights = lognorm([weights_unorm[i] for i in indexes])
            children = [
                (type(self))(self.symbol, self.dist, values[i], True, self.env)
                for i in indexes
            ]
            return SumSPE(children, weights) if 1 < len(indexes) else children[0]
        # Unknown set.
        assert False, 'Unknown set type: %s' % (values,)

    def constrain__(self, x):
        assert not isinf_neg(self.logpdf__(x))
        return AtomicLeaf(self.symbol, x)

    def __hash__(self):
        d = (self.dist.dist.name, self.dist.args, tuple(self.dist.kwds.items()))
        e = tuple(self.env.items())
        x = (self.__class__, self.symbol, d, self.support, self.conditioned, e)
        return hash(x)
    def __eq__(self, x):
        return isinstance(x, type(self)) \
            and self.symbol == x.symbol \
            and self.dist.dist.name == x.dist.dist.name \
            and self.dist.args == x.dist.args \
            and self.dist.kwds == x.dist.kwds \
            and self.support == x.support \
            and self.conditioned == x.conditioned \
            and self.env == x.env

# ==============================================================================
# Continuous RealLeaf.

class ContinuousLeaf(RealLeaf):
    """Non-atomic distribution with a cumulative distribution function."""
    atomic = False
    def __init__(self, symbol, dist, support, conditioned=None, env=None):
        super().__init__(symbol, dist, support, conditioned, env)
        self.xl = float(support.left)
        self.xu = float(support.right)
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

    def logpdf__(self, x):
        if isinstance(x, str):
            return -float('inf')
        xf = float(x)
        if not self.conditioned:
            return self.dist.logpdf(xf)
        if x not in self.support:
            return -inf
        return self.dist.logpdf(xf) - self.logZ

    def logprob_finite__(self, values):
        return -inf

    def logprob_interval__(self, values):
        xl = float(values.left)
        xu = float(values.right)
        logFl = self.logcdf(xl)
        logFu = self.logcdf(xu)
        return logdiffexp(logFu, logFl)

# ==============================================================================
# Discrete RealLeaf.

class DiscreteLeaf(RealLeaf):
    """Integral atomic distribution with a cumulative distribution function."""
    atomic = True
    def __init__(self, symbol, dist, support, conditioned=None, env=None):
        super().__init__(symbol, dist, support, conditioned, env)
        assert int_or_isinf_neg(support.left)
        assert int_or_isinf_pos(support.right)
        self.xl = float_to_int(support.left) + 1*bool(support.left_open)
        self.xu = float_to_int(support.right) - 1*bool(support.right_open)
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

    def logpdf__(self, x):
        if isinstance(x, str):
            return -float('inf')
        xf = float(x)
        if not self.conditioned:
            return self.dist.logpmf(xf)
        if (x < self.xl) or (self.xu < x):
            return -inf
        return self.dist.logpmf(xf) - self.logZ

    def logprob_finite__(self, values):
        logps = [self.logpdf__(x) for x in values]
        return logps[0] if len(logps) == 1 else logsumexp(logps)
    def logprob_interval__(self, values):
        offsetl = not values.left_open and int_or_isinf_neg(values.left)
        offsetr = values.right_open and int_or_isinf_pos(values.right)
        xl = float_to_int(values.left) - offsetl
        xu = float_to_int(values.right) - offsetr
        logFl = self.logcdf(xl)
        logFu = self.logcdf(xu)
        return logdiffexp(logFu, logFl)

# ==============================================================================
# Atomic RealLeaf.

class AtomicLeaf(UnivariateLeafSPE):
    """Real atomic distribution."""
    atomic = True
    def __init__(self, symbol, value, env=None):
        self.symbol = symbol
        self.support = FiniteReal(value)
        self.value = value
        self.env = env or OrderedDict([(symbol, symbol)])

    def transform(self, symbol, expr):
        assert symbol not in self.env
        assert all(s in self.env for s in expr.get_symbols())
        env = OrderedDict(self.env)
        env[symbol] = expr
        return AtomicLeaf(self.symbol, self.value, env=env)

    def sample__(self, N, prng):
        return [{self.symbol : self.value}] * N
    def logprob__(self, event):
        interval = event.solve()
        return 0 if self.value in interval else -inf
    def condition__(self, event):
        interval = event.solve()
        assert self.value in interval, 'Measure zero condition %s' % (event,)
        return self

    def __hash__(self):
        x = (self.__class__, self.symbol, self.value)
        return hash(x)
    def __eq__(self, x):
        return isinstance(x, type(self)) \
            and self.symbol == x.symbol \
            and self.value == x.value

# ==============================================================================
# Nominal distribution.

class NominalLeaf(UnivariateLeafSPE):
    """Atomic distribution, no cumulative distribution function."""
    atomic = True
    def __init__(self, symbol, dist):
        assert isinstance(symbol, Id)
        assert all(isinstance(x, str) for x in dist)
        self.symbol = symbol
        self.dist = {x: Fraction(w) for x, w in dist.items()}
        # Derived attributes.
        self.env = {symbol: symbol}
        self.support = FiniteNominal(*dist.keys())
        self.outcomes = list(self.dist.keys())
        self.weights = list(self.dist.values())
        assert allclose(float(sum(self.weights)),  1)

    def logpdf__(self, x):
        if x not in self.dist:
            return -inf
        w = self.dist[x]
        return log(w.numerator) - log(w.denominator)

    def transform(self, symbol, expr):
        raise ValueError('Cannot transform Nominal: %s %s' % (symbol, expr))

    def sample__(self, N, prng):
        # TODO: Replace with FLDR.
        xs = flip(self.weights, self.outcomes, N, prng)
        return [{self.symbol: x} for x in xs]

    def logprob__(self, event):
        solution = event.solve()
        values = self.support & solution
        if values is EmptySet:
            return -inf
        if values == FiniteNominal(b=True):
            return 0
        p_event = sum(self.dist[x] for x in values)
        return log(p_event) if p_event != 0 else -inf

    def condition__(self, event):
        solution = event.solve()
        values = self.support & solution
        if values is EmptySet:
            raise ValueError('Zero probability condition %s' % (event,))
        p_event = sum([self.dist[x] for x in values])
        if p_event == 0:
            raise ValueError('Zero probability condition %s' % (event,))
        if p_event == 1:
            return self
        dist = {
            str(x) : (self.dist[x] / p_event) if x in values else 0
            for x in self.support
        }
        return NominalLeaf(self.symbol, dist)

    def constrain__(self, x):
        assert not isinf_neg(self.logpdf__(x))
        return NominalLeaf(self.symbol, {x: 1})

    def __hash__(self):
        x = (self.__class__, self.symbol, tuple(self.dist.items()))
        return hash(x)
    def __eq__(self, x):
        return isinstance(x, type(self)) \
            and self.symbol == x.symbol \
            and self.dist == x.dist

# ==============================================================================
# Utilities.

class Memo():
    def __init__(self):
        self.logprob = {}
        self.condition = {}
        self.logpdf = {}
        self.constrain = {}

def spe_cache_duplicate_subtrees(spe, memo):
    if isinstance(spe, UnivariateLeafSPE):
        if spe not in memo:
            memo[spe] = spe
        return memo[spe]
    if isinstance(spe, BranchSPE):
        if spe not in memo:
            memo[spe] = spe
            spe.children = list(spe.children)
            for i, c in enumerate(spe.children):
                spe.children[i] = spe_cache_duplicate_subtrees(c, memo)
            spe.children = tuple(spe.children)
        return memo[spe]
    assert False, '%s is not an spe' % (spe,)

def func_evaluate(spe, func, samples):
    args = func_symbols(spe, func)
    sample_kwargs = [{X.token: s[X] for X in args} for s in samples]
    return [func(**kwargs) for kwargs in sample_kwargs]

def func_symbols(spe, func):
    symbols = spe.get_symbols()
    args = [Id(a) for a in getfullargspec(func).args]
    unknown = [a for a in args if a not in symbols]
    if unknown:
        raise ValueError('Unknown function arguments "%s" (allowed %s)'
            % (unknown, symbols))
    return args

def merge_samples(samples):
    # input [[{X:1, Y:2}, {X:0, Y:1}], [{Z:0}, {Z:1}]] (N=2)
    # output [{X:1, Y:2, Z:0}, {X:0, Y:1, Z:1}]
    return [dict(ChainMap(*sample_list)) for sample_list in zip(*samples)]

def event_factor_to_event(event_factor):
    conjunctions = (
        reduce(lambda x, e: x & e, conjunction.values())
        for conjunction in event_factor
    )
    return reduce(lambda x, e: x | e, conjunctions)
