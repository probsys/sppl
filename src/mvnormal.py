# Copyright (c) 2016 MIT Probabilistic Computing Project.
#
# This file is part of Venture.
#
# Venture is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Venture is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Venture.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import division

import numpy as np
import scipy.linalg as la

# XXX Use LDL decomposition instead of Cholesky so that we can handle
# merely positive-semidefinite, rather than positive-definite,
# covariance matrices better than LU decomposition.  Problem: neither
# numpy nor scipy supports it.  We could do it ourselves, but that's a
# lot of work and will be slow in Python.

def _covariance_factor(Sigma):
  # Assume it is positive-definite and try Cholesky decomposition.
  try:
    return Covariance_Cholesky(Sigma)
  except la.LinAlgError:
    pass

  # XXX In the past, we tried LU decomposition if, owing to
  # floating-point rounding error, the matrix is merely nonsingular,
  # not positive-definite.  However, empirically, that seemed to lead
  # to bad numerical results.  Until we have better numerical analysis
  # of the situation, let's try just falling back to least-squares
  # pseudoinverse approximation.

  # Otherwise, fall back to whatever heuristics scipy can manage.
  return Covariance_Loser(Sigma)

class Covariance_Cholesky(object):
  def __init__(self, Sigma):
    self._cholesky = la.cho_factor(Sigma)
  def solve(self, Y):
    return la.cho_solve(self._cholesky, Y)
  def inverse(self):
    return self.solve(np.eye(self._cholesky[0].shape[0]))
  def logsqrtdet(self):
    # Sigma = L^T L -- but be careful: only the lower triangle and
    # diagonal of L are actually initialized; the upper triangle is
    # garbage.
    L, _lower = self._cholesky

    # det Sigma = det L^T L = det L^T det L = (det L)^2.  Since L is
    # triangular, its determinant is the product of its diagonal.  To
    # compute log sqrt(det Sigma) = log det L, we sum the logs of its
    # diagonal.
    return np.sum(np.log(np.diag(L)))

class Covariance_Loser(object):
  def __init__(self, Sigma):
    self._Sigma = Sigma
  def solve(self, Y):
    X, _residues, _rank, _sv = la.lstsq(self._Sigma, Y)
    return X
  def inverse(self):
    return la.pinv(self._Sigma)
  def logsqrtdet(self):
    return (1/2)*np.log(la.det(self._Sigma))

def logpdf(X, Mu, Sigma):
  """Multivariate normal log pdf."""
  # This is the multivariate normal log pdf for an array X of n
  # outputs, an array Mu of n means, and an n-by-n positive-definite
  # covariance matrix Sigma.  The direct-space density is:
  #
  #     P(X | Mu, Sigma)
  #       = ((2 pi)^n det Sigma)^(-1/2)
  #         exp((-1/2) (X - Mu)^T Sigma^-1 (X - Mu)),
  #
  # We want this in log-space, so we have
  #
  #     log P(X | Mu, Sigma)
  #     = (-1/2) (X - Mu)^T Sigma^-1 (X - Mu) - log ((2 pi)^n det Sigma)^(1/2)
  #     = (-1/2) (X - Mu)^T Sigma^-1 (X - Mu)
  #         - (n/2) log (2 pi) - (1/2) log det Sigma.
  #
  n = len(X)
  assert X.shape == (n,)
  assert Mu.shape == (n,)
  assert Sigma.shape == (n, n)
  assert np.all(np.isfinite(X))
  assert np.all(np.isfinite(Mu))
  assert np.all(np.isfinite(Sigma))

  X_ = X - Mu
  covf = _covariance_factor(Sigma)

  logp = -np.dot(X_.T, covf.solve(X_)/2.)
  logp -= (n/2.)*np.log(2*np.pi)
  logp -= covf.logsqrtdet()

  # Convert 1x1 matrix to float.
  return float(logp)

def dlogpdf(X, dX, Mu, dMu, Sigma, dSigma):
  """Derivative of multivariate normal logpdf with respect to parameters.

  If X is a function of some parameter t in R^i, Mu of p in R^j, and
  Sigma of q in R^k, then the differential of the multivariate normal
  density is

        d log P(X | Mu, Sigma) = F dX + G dMu + H dSigma
          = F X'(t) dt + G Mu'(p) dp + H Sigma'(q) dq,

  for some fields F, G, and H of linear maps from increments in X, Mu,
  and Sigma to increments in log P.  This function, dlogpdf, takes

        X               X(t_0), a point in R^n;
        dX              X'(t_0) dt, a linear map from an increment of t in R^i
                            to an increment of X in R^n, represented by an
                            array of i vectors in R^n;
        Mu              Mu(p_0), a point in R^n;
        dMu             Mu'(p_0) dp, a linear map from an increment of p in R^j
                            to an increment of Mu in R, represented by
                            an array of j vectors in R^n;
        Sigma           Sigma(q_0), a matrix in M(n, n); and
        dSigma          Sigma'(q_0) dq, a linear map from an increment of q in
                            R^k to an increment of Sigma in M(n, n),
                            represented by an array of k matrices in M(n, n),

  and computes the derivative of

        log P(X | Mu, Sigma)

  with respect to the implied parameters t, p, and q -- that is,
  computes an array of the dt, dp, and dq components of the covector

        d log P(X | Mu, Sigma).
  """
  # Let A = Y - M and alpha = Sigma^-1 (Y - M) = Sigma^-1 A.  Then
  #
  #     log P(X | Mu, Sigma)
  #       = -(1/2) (Y - Mu)^T Sigma^-1 (Y - Mu)
  #           - (n/2) log 2 pi - (1/2) log det Sigma
  #       = -(1/2) A^T Sigma^-1 A - (n/2) log 2 pi - (1/2) log det Sigma,
  #
  # so that
  #
  #     d log P(X | Mu, Sigma)
  #       = (-1/2) [d(A^T Sigma^-1 A) + d(n log 2 pi) + d(log det Sigma)].
  #
  # Using the matrix calculus identities
  #
  #     d(U^T) = (dU)^T,
  #     d(U V W) = dU V W + U dV W + U V dW,    and
  #     d(U^-1) = -U^-1 dU U^-1;
  #
  # the vector inner/outer product identity
  #
  #     u^T v = <u, v> = tr(u (x) v) = tr(u v^T);
  #
  # and the fact that alpha^T dA is a scalar and hence equal to its
  # transpose dA^T alpha, we have
  #
  #     d(A^T Sigma^-1 A)
  #       = dA^T Sigma^-1 A + A^T d(Sigma^-1) A + A^T Sigma^-1 dA
  #       = dA^T alpha - A^T Sigma^-1 dSigma Sigma^-1 A + alpha^T dA
  #       = alpha^T dA - alpha^T dSigma alpha + alpha^T dA
  #       = 2 alpha^T dA - tr(alpha alpha^T dSigma).
  #
  # Note that d log x = dx/x and d det X = (det X) tr(X^-1 dX), so
  # that
  #
  #     d(log det Sigma) = d(det Sigma) / det Sigma
  #       = (det Sigma) tr(Sigma^-1 dSigma) / det Sigma
  #       = tr(Sigma^-1 dSigma).
  #
  # Hence
  #
  #     d log P(X | Mu, Sigma)
  #       = -(1/2) [2 alpha^T dA - tr((alpha alpha^T - Sigma^-1) dSigma)]
  #       = -alpha^T dY + alpha^T dMu
  #             + (1/2) tr((alpha alpha^T - Sigma^-1) dSigma).
  #
  # For dY = 0, dMu = Mu'(p) dp, and dSigma = Sigma'(q) dq = \sum_i
  # (d/dq^i Sigma(q)) dq^nu, where d/dq^i is the partial derivative
  # with respect to the ith component of q, and dq^i is the ith
  # coordinate differential of q, we have
  #
  #     d log P(X | Mu, Sigma)
  #       = alpha^T Mu'(p) dp
  #         - (1/2) tr((alpha alpha^T - Sigma^-1) Sigma'(q) dq)
  #       = alpha^T Mu'(p) dp
  #         - (1/2) \sum_i tr((alpha alpha^T - Sigma^-1) d/dq^i Sigma(q)) dq^i.
  #

  # Note: This abstraction does not correspond to what a reverse mode
  # automatic differentiation system would construct.  Why?  Riastradh
  # says (9/9/16):
  #
  #   Pfergh.  grad_Sigma is supposed to compute w |---> w Sigma'(t)
  #   at some fixed t.  But the way we compute that is by tr(Q
  #   Sigma'(t)) for some matrix Q (which we actually compute by
  #   summing the components of the componentwise product), and the
  #   object w such that w Sigma'(t) = tr(Q Sigma'(t)) is a
  #   higher-order tensor.
  #
  #   In particular, it is a linear functional on M(n, n), which has
  #   no matrix representation.

  n = len(X)
  assert Mu.shape == (n,)
  assert all(dMu_dpj.shape == (n,) for dMu_dpj in dMu)
  assert Sigma.shape == (n, n)
  assert all(dSigma_dqk.shape == (n, n) for dSigma_dqk in dSigma)
  assert np.all(np.isfinite(X))
  assert np.all(np.isfinite(Mu))
  assert np.all(np.isfinite(Sigma))

  X_ = X - Mu
  covf = _covariance_factor(Sigma)

  # Solve Sigma alpha = X - Mu for alpha.
  #
  # XXX It is ~10 times faster to compute Sigma^-1 and do a single
  # multiplication per partial derivative than to solve a linear
  # system per partial derivative.  But is it numerically safe?  I
  # doubt it.  For now, we'll do the fast-and-loose thing because too
  # much time is spent in solving linear systems otherwise.
  alpha = covf.solve(X_)

  # Compute Q = alpha alpha^T - Sigma^-1.
  Q = np.outer(alpha, alpha) - covf.inverse()

  dlogP_dt = np.array([-np.dot(alpha, dX_dti) for dX_dti in dX])
  dlogP_dp = np.array([np.dot(alpha, dMu_dpj) for dMu_dpj in dMu])
  dlogP_dq = np.array([np.sum(Q*dSigma_dqk)/2 for dSigma_dqk in dSigma])

  return (dlogP_dt, dlogP_dp, dlogP_dq)

def conditional(X2, Mu1, Mu2, Sigma11, Sigma12, Sigma21, Sigma22):
  """Parameters of conditional multivariate normal."""
  # The conditional distribution of a multivariate normal given some
  # fixed values of some variables is itself a multivariate normal on
  # the remaining values, with a slightly different mean and
  # covariance matrix.  In particular, for
  #
  #     Mu = [Mu_1; Mu_2],
  #     Sigma = [Sigma_11, Sigma_12; Sigma_21, Sigma_22],
  #
  # where `;' separates rows and `,' separates columns within a row,
  # the conditional distribution given the fixed values X_2 for the
  # second block of variables is multivariate normal with
  #
  #     Mu' = Mu_1 + Sigma_12 Sigma_22^-1 (X_2 - Mu_2),
  #     Sigma' = Sigma_11 - Sigma_12 Sigma_22^-1 Sigma_21,
  #
  # where Sigma' is the Schur complement of Sigma_22 in Sigma.
  #
  d1 = len(Mu1)
  d2 = len(Mu2)
  assert X2.shape == (d2,)
  assert Mu1.shape == (d1,)
  assert Mu2.shape == (d2,)
  assert Sigma11.shape == (d1, d1)
  assert Sigma12.shape == (d1, d2)
  assert Sigma21.shape == (d2, d1)
  assert Sigma22.shape == (d2, d2)
  assert np.all(np.isfinite(X2))
  assert np.all(np.isfinite(Mu1))
  assert np.all(np.isfinite(Mu2))
  assert np.all(np.isfinite(Sigma11))
  assert np.all(np.isfinite(Sigma12))
  assert np.all(np.isfinite(Sigma21))
  assert np.all(np.isfinite(Sigma22))

  covf22 = _covariance_factor(Sigma22)
  Mu_ = Mu1 + np.dot(Sigma12, covf22.solve(X2 - Mu2))
  Sigma_ = Sigma11 - np.dot(Sigma12, covf22.solve(Sigma21))
  return (Mu_, Sigma_)
