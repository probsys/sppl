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
from sppl.spn import Memo
from sppl.transforms import Id

prng = numpy.random.RandomState(1)

def entropy(spn, A, memo):
    lpA1 = spn.logprob(A, memo=memo)
    lpA0 = logdiffexp(0, lpA1)
    e1 = -exp(lpA1) * lpA1 if not isinf_neg(lpA1) else 0
    e0 = -exp(lpA0) * lpA0 if not isinf_neg(lpA0) else 0
    return e1 + e0
def entropyc(spn, A, B, memo):
    lpB1 = spn.logprob(B)
    lpB0 = logdiffexp(0, lpB1)
    lp11 = spn.logprob(B & A, memo=memo)
    lp10 = spn.logprob(B & ~A, memo=memo)
    lp01 = spn.logprob(~B & A, memo=memo)
    # lp00 = self.logprob(~B & ~A, memo)
    lp00 = logdiffexp(0, logsumexp([lp11, lp10, lp01]))
    m11 = exp(lp11) * (lpB1 - lp11) if not isinf_neg(lp11) else 0
    m10 = exp(lp10) * (lpB1 - lp10) if not isinf_neg(lp10) else 0
    m01 = exp(lp01) * (lpB0 - lp01) if not isinf_neg(lp01) else 0
    m00 = exp(lp00) * (lpB0 - lp00) if not isinf_neg(lp00) else 0
    return m11 + m10 + m01 + m00

def check_mi_properties(spn, A, B, memo):
    miAB = spn.mutual_information(A, B, memo=memo)
    miAA = spn.mutual_information(A, A, memo=memo)
    miBB = spn.mutual_information(B, B, memo=memo)
    eA = entropy(spn, A, memo=memo)
    eB = entropy(spn, B, memo=memo)
    eAB = entropyc(spn, A, B, memo=memo)
    eBA = entropyc(spn, B, A, memo=memo)
    assert allclose(miAA, eA)
    assert allclose(miBB, eB)
    assert allclose(miAB, eA - eAB)
    assert allclose(miAB, eB - eBA)

@pytest.mark.parametrize('memo', [Memo(), None])
def test_mutual_information_four_clusters(memo):
    X = Id('X')
    Y = Id('Y')
    spn \
        = 0.25*(X >> norm(loc=0, scale=0.5) & Y >> norm(loc=0, scale=0.5)) \
        | 0.25*(X >> norm(loc=5, scale=0.5) & Y >> norm(loc=0, scale=0.5)) \
        | 0.25*(X >> norm(loc=0, scale=0.5) & Y >> norm(loc=5, scale=0.5)) \
        | 0.25*(X >> norm(loc=5, scale=0.5) & Y >> norm(loc=5, scale=0.5)) \

    A = X > 2
    B = Y > 2
    samples = spn.sample(100, prng)
    mi = spn.mutual_information(A, B, memo=memo)
    assert allclose(mi, 0)
    check_mi_properties(spn, A, B, memo)

    event = ((X>2) & (Y<2) | ((X<2) & (Y>2)))
    spn_condition = spn.condition(event)
    samples = spn_condition.sample(100, prng)
    assert all(event.evaluate(sample) for sample in samples)
    mi = spn_condition.mutual_information(X > 2, Y > 2)
    assert allclose(mi, log(2))

    check_mi_properties(spn, (X>1) | (Y<1), (Y>2), memo)
    check_mi_properties(spn, (X>1) | (Y<1), (X>1.5) & (Y>2), memo)
