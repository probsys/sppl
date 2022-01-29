# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from . import mvnormal

from .dnf import dnf_factor
from .dnf import dnf_normalize
from .dnf import dnf_to_disjoint_union

from .math_util import isinf_neg

from .sets import Reals

from .spe import AtomicLeaf
from .spe import ContinuousLeaf
from .spe import Memo
from .spe import ProductSPE
from .spe import SPE
from .spe import memoize

from .transforms import EventOr
from .transforms import Id

import numpy as np

from scipy.stats import norm

class MultivariateNormal(SPE):
    env = None

    def __init__(self, symbols, mu, cov, supports=None):
        assert all(isinstance(symbol, Id) for symbol in symbols)
        assert len(symbols) > 1
        assert len(mu) == len(symbols)
        assert np.shape(cov) == (len(mu), len(mu))
        self.symbols = symbols
        self.mu = np.asarray(mu)
        self.cov = np.asarray(cov)
        self.env = {symbol: symbol for symbol in symbols}
        self.supports = supports

        # TODO: The supports variable will be a list of intervals, one for each dimension
        # indicating the region to which that dimensions is constrained,
        # that defines a hyperrectangle in len(symbols)-dimensional space.
        # For example 3D, support = [Reals, RealsPos, Interval(-5, 1)]
        #
        # The variable logZ shall store the log probability of the hyperrectangle
        # which can be computed using scipy.stats.mvn.mvnun
        #   https://stackoverflow.com/a/57245576
        # The mvnun function can also be implemented directly using the
        # multivariate_normal.cdf and inclusion-exclusion following
        #   https://math.stackexchange.com/a/113088

    def get_symbols(self):
        return frozenset(self.symbols)

    def size(self):
        return 1

    def sample(self, N, prng=None):
        return self.sample_subset(self.symbols, N, prng=prng)

    def sample_subset(self, symbols, N, prng=None):
        assert symbols
        simulations = np.random.multivariate_normal(self.mu, self.cov, size=N)
        return [dict(zip(self.symbols, x)) for x in simulations]

    def transform(self, symbol, expr):
        raise NotImplementedError()

    # Same as BranchSPE
    def logprob(self, event, memo=None):
        if memo is None:
            memo = Memo()
        event_subs = event.substitute(self.env)
        event_dnf = dnf_normalize(event_subs)
        if event_dnf is None:
            return -float('inf')
        event_factor = dnf_factor(event_dnf)
        return self.logprob_mem(event_factor, memo)
    def condition(self, event, memo=None):
        if memo is None:
            memo = Memo()
        event_subs = event.substitute(self.env)
        event_dnf = dnf_normalize(event_subs)
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
        return self.logpdf_mem(assignment, memo or False)[1]
    def constrain(self, assignment, memo=None):
        return self.constrain_mem(assignment, memo or False)

    def logprob_mem(self, event_factor, memo):
        # Step 1: For each conjunction, intersect each variable
        # in the conjunction with the corresponding interval
        # from self.supports[variable]
        #
        # Step 2: Compute probability using inclusion-exclusion, by
        # implementing logprob_conjunction(self, event_factor, J, memo)
        # similarly to ProductSPE.
        #
        # Step 3: Subtract logZ from the result.
        pass
    def condition_mem(self, event_factor, memo):
        # Identical to ProductSPE.condition_mem, making sure to implement
        # MultivariateNormal.condition_clause
        pass

    @memoize
    def logpdf_mem(self, assignment, memo):
        idx = [i for i, s in enumerate(self.symbols) if s in assignment]
        Mu = self.mu[idx]
        Sigma = self.cov[np.ix_(idx,idx)]
        X1 = np.asarray([assignment[self.symbols[i]] for i in idx])
        lp = mvnormal.logpdf(X1, Mu, Sigma)
        return (1, lp)

    @memoize
    def constrain_mem(self, assignment, memo):
        assert all(k in self.symbols for k in assignment)

        # Nothing to constrain.
        if not assignment:
            return MultivariateNormal(self.symbols, self.mu, self.cov)

        # Find constraint indexes.
        idx1 = np.asarray([i for i, symbol in enumerate(self.symbols)
            if symbol not in assignment], dtype=int)
        idx2 = np.asarray([i for i, symbol in enumerate(self.symbols)
            if symbol in assignment], dtype=int)
        assert len(idx2) > 0

        # Build AtomicLeaf for constrained variables.
        atoms = [AtomicLeaf(x, v) for x, v in assignment.items()]
        if len(idx1) == 0:
            return ProductSPE(atoms) if len(atoms) > 1 else atoms[0]

        # Build new MultivariateNormal for unconstrained variables.
        Mu1 = self.mu[idx1]
        Sigma11 = self.cov[np.ix_(idx1, idx1)]
        Mu2 = self.mu[idx2]
        Sigma12 = self.cov[np.ix_(idx1, idx2)]
        Sigma21 = self.cov[np.ix_(idx2, idx1)]
        Sigma22 = self.cov[np.ix_(idx2, idx2)]
        X2 = np.asarray([assignment[self.symbols[i]] for i in idx2])
        Mu, Sigma = mvnormal.conditional(X2, Mu1, Mu2, Sigma11, Sigma12, Sigma21, Sigma22)

        # Determine dimensionality.
        if len(idx1) == 1:
            symbol = self.symbols[idx1[0]]
            dist = norm(loc=Mu[0], scale=np.sqrt(Sigma[0]))
            mvn = ContinuousLeaf(symbol, dist, Reals)
        else:
            mvn = MultivariateNormal([self.symbols[i] for i in idx1], Mu, Sigma)

        # Return ProductSPE.
        return ProductSPE([mvn, *atoms])
