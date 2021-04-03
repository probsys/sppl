# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from math import exp
from math import log

import numpy
import pytest

from sppl.distributions import norm
from sppl.math_util import allclose
from sppl.math_util import isinf_neg
from sppl.math_util import logdiffexp
from sppl.math_util import logsumexp
from sppl.spe import Memo
from sppl.transforms import Id

prng = numpy.random.RandomState(1)

def entropy(spe, A, memo):
    lpA1 = spe.logprob(A, memo=memo)
    lpA0 = logdiffexp(0, lpA1)
    e1 = -exp(lpA1) * lpA1 if not isinf_neg(lpA1) else 0
    e0 = -exp(lpA0) * lpA0 if not isinf_neg(lpA0) else 0
    return e1 + e0
def entropyc(spe, A, B, memo):
    lpB1 = spe.logprob(B)
    lpB0 = logdiffexp(0, lpB1)
    lp11 = spe.logprob(B & A, memo=memo)
    lp10 = spe.logprob(B & ~A, memo=memo)
    lp01 = spe.logprob(~B & A, memo=memo)
    # lp00 = self.logprob(~B & ~A, memo)
    lp00 = logdiffexp(0, logsumexp([lp11, lp10, lp01]))
    m11 = exp(lp11) * (lpB1 - lp11) if not isinf_neg(lp11) else 0
    m10 = exp(lp10) * (lpB1 - lp10) if not isinf_neg(lp10) else 0
    m01 = exp(lp01) * (lpB0 - lp01) if not isinf_neg(lp01) else 0
    m00 = exp(lp00) * (lpB0 - lp00) if not isinf_neg(lp00) else 0
    return m11 + m10 + m01 + m00

def check_mi_properties(spe, A, B, memo):
    miAB = spe.mutual_information(A, B, memo=memo)
    miAA = spe.mutual_information(A, A, memo=memo)
    miBB = spe.mutual_information(B, B, memo=memo)
    eA = entropy(spe, A, memo=memo)
    eB = entropy(spe, B, memo=memo)
    eAB = entropyc(spe, A, B, memo=memo)
    eBA = entropyc(spe, B, A, memo=memo)
    assert allclose(miAA, eA)
    assert allclose(miBB, eB)
    assert allclose(miAB, eA - eAB)
    assert allclose(miAB, eB - eBA)

@pytest.mark.parametrize('memo', [Memo(), None])
def test_mutual_information_four_clusters(memo):
    X = Id('X')
    Y = Id('Y')
    spe \
        = 0.25*(X >> norm(loc=0, scale=0.5) & Y >> norm(loc=0, scale=0.5)) \
        | 0.25*(X >> norm(loc=5, scale=0.5) & Y >> norm(loc=0, scale=0.5)) \
        | 0.25*(X >> norm(loc=0, scale=0.5) & Y >> norm(loc=5, scale=0.5)) \
        | 0.25*(X >> norm(loc=5, scale=0.5) & Y >> norm(loc=5, scale=0.5)) \

    A = X > 2
    B = Y > 2
    samples = spe.sample(100, prng)
    mi = spe.mutual_information(A, B, memo=memo)
    assert allclose(mi, 0)
    check_mi_properties(spe, A, B, memo)

    event = ((X>2) & (Y<2) | ((X<2) & (Y>2)))
    spe_condition = spe.condition(event)
    samples = spe_condition.sample(100, prng)
    assert all(event.evaluate(sample) for sample in samples)
    mi = spe_condition.mutual_information(X > 2, Y > 2)
    assert allclose(mi, log(2))

    check_mi_properties(spe, (X>1) | (Y<1), (Y>2), memo)
    check_mi_properties(spe, (X>1) | (Y<1), (X>1.5) & (Y>2), memo)
