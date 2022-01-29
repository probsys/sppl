# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

import re

from . import mvnormal

from .spe import SPE

from .transforms import Id

import numpy as np
from scipy.stats import multivariate_normal

class IdProcess:
    def __init__(self, token):
        self.token = token
    def __getitem__(self, item):
        assert isinstance(item, (int, float))
        return Id('%s[%s]' % (self.token, float(item)))
    def parse(self, symbol):
        msg = 'Bad index %s for process %s' % (symbol, self.token)
        match = re.search(r'\A\w+\[', symbol.token)
        if not match:
            raise ValueError(msg) from None
        if not match.group(0).replace('[', '') == self.token:
            raise ValueError(msg) from None
        match = re.search(r'\[.*\]\Z', symbol.token)
        if not match:
            raise ValueError(msg) from None
        index = match.group(0)
        index = index.replace('[', '').replace(']', '')
        try:
            return float(index)
        except ValueError:
            raise ValueError(msg) from None
    def __eq__(self, x):
        return isinstance(x, type(self)) and self.token == x.token
    def __hash__(self):
        x = (self.__class__, self.token)
        return hash(x)

class GaussianProcess(SPE):
    env = None
    def __init__(self, symbol, mu, cov, constraints=None):
        assert isinstance(symbol, IdProcess)
        self.symbol = symbol
        self.mu = mu
        self.cov = cov
        self.env = {}
        self.constraints = constraints or {}

        # Compute constrained indexes.
        self.idx_obs = [self.symbol.parse(symbol) for symbol in self.constraints]
        self.X2 = np.asarray(list(self.constraints.values()))
        self.Mu2 = np.asarray([self.mu(i) for i in self.idx_obs])

    def size(self):
        return 1

    def sample(self, N, prng=None):
        raise ValueError('Cannot sample entire GaussianProcess, use sample_susbet.')

    def sample_subset(self, symbols, N, prng=None):
        assert symbols
        idx_query = map(self.symbol.parse, symbols)
        idx1 = [i for i in idx_query if i not in self.idx_obs]
        sim2 = {k: v for k, v in self.constraints.items() if k in symbols}
        # Sampling only constrained variables.
        if len(idx1) == 0:
            return [sim2] * N
        # Sampling both new and constrained variables.
        Mu1, Sigma11 = self.get_Mu1_Sigma11(idx1)
        simulations = []
        for x in np.random.multivariate_normal(Mu1, Sigma11, size=N):
            sim1 = {self.symbol[idx1[i]]: v for i, v in enumerate(x)}
            sim = {**sim1, **sim2}
            simulations.append(sim)
        return simulations

    def logpdf(self, assignment, memo=None):
        idx_query = [self.symbol.parse(symbol) for symbol in assignment]
        idx1 = [i for i in idx_query if i not in self.idx_obs]
        if len(idx1) != len(idx_query):
            raise ValueError('Cannot compute logpdf of constrained GP variables.')
        X1 = np.asarray(list(assignment.values()))
        Mu1, Sigma11 = self.get_Mu1_Sigma11(idx1)
        print(np.diag(Sigma11))
        print(Sigma11)
        if np.any(np.diag(Sigma11) == 0.):
            raise ValueError('Cannot evaluate logpdf with zero variance: %s' % (Sigma11,))
        try:
            return multivariate_normal.logpdf(X1, Mu1, Sigma11)
        except np.linalg.LinAlgError:
            import warnings
            warnings.warn('Singular matrix in GP.logpdf. Use result carefully.\n')
            return multivariate_normal.logpdf(X1, Mu1, Sigma11, allow_singular=True)
        # return mvnormal.logpdf(X1, Mu1, Sigma11)

    def get_Mu1_Sigma11(self, idx):
        idx = np.asarray(idx)
        Mu1 = np.asarray([self.mu(i) for i in idx])
        if self.constraints:
            n = len(idx)
            xs = np.asarray([*idx, *self.idx_obs])
            Sigma = self.cov.f_mat(xs)
            Sigma11 = Sigma[:n,:n]
            Sigma12 = Sigma[:n,n:]
            Sigma21 = Sigma[n:,:n]
            Sigma22 = Sigma[n:,n:]
            assert np.allclose(Sigma12, Sigma21.T)
            return mvnormal.conditional(
                self.X2, Mu1, self.Mu2, Sigma11, Sigma12, Sigma21, Sigma22)
        else:
            Sigma11 = self.cov.f_mat(idx)
            return Mu1, Sigma11

    def constrain(self, assignment, memo=None):
        for symbol in assignment:
            self.symbol.parse(symbol)
        constraints_new = dict(self.constraints)
        for key, value in assignment.items():
            assert key not in self.constraints
            constraints_new[key] = value
        return GaussianProcess(
            self.symbol, self.mu, self.cov, constraints=constraints_new)
