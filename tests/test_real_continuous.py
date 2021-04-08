# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from math import log

import numpy
import pytest
import scipy.stats

from sppl.distributions import gamma
from sppl.distributions import norm
from sppl.math_util import allclose
from sppl.math_util import isinf_neg
from sppl.math_util import logdiffexp
from sppl.sets import Interval
from sppl.sets import Reals
from sppl.sets import inf as oo
from sppl.spe import ContinuousLeaf
from sppl.spe import SumSPE
from sppl.transforms import Id

def test_numeric_distribution_normal():
    X = Id('X')
    spe = (X >> norm(loc=0, scale=1))

    assert spe.size() == 1
    assert allclose(spe.logprob(X > 0), -log(2))
    assert allclose(spe.logprob(abs(X) < 2), log(spe.dist.cdf(2) - spe.dist.cdf(-2)))

    assert allclose(spe.logprob(X**2 > 0), 0)
    assert allclose(spe.logprob(abs(X) > 0), 0)
    assert allclose(spe.logprob(~(X << {1})), 0)

    assert isinf_neg(spe.logprob(X**2 - X + 10 < 0))
    assert isinf_neg(spe.logprob(abs(X) < 0))
    assert isinf_neg(spe.logprob(X << {1}))

    spe.sample(100)
    spe.sample_subset([X], 100)
    assert spe.sample_subset([], 100) == [{}]*100
    spe.sample_func(lambda X: X**2, 1)
    spe.sample_func(lambda X: abs(X)+X**2, 1)
    spe.sample_func(lambda X: X**2 if X > 0 else X**3, 100)

    spe_condition_a = spe.condition((X < 2) | (X > 10))
    samples = spe_condition_a.sample(100)
    assert all(s[X] < 2 for s in samples)

    spe_condition_b = spe.condition((X < -10) | (X > 10))
    assert isinstance(spe_condition_b, SumSPE)
    assert allclose(spe_condition_b.weights[0], -log(2))
    assert allclose(spe_condition_b.weights[0], spe_condition_b.weights[1])

    for event in [(X<-10), (X>3)]:
        spe_condition_c = spe.condition(event)
        assert isinstance(spe_condition_c, ContinuousLeaf)
        assert isinf_neg(spe_condition_c.logprob((-1 < X) < 1))
        samples = spe_condition_c.sample(100, prng=numpy.random.RandomState(1))
        assert all(s[X] in event.values for s in samples)

    with pytest.raises(ValueError):
        spe.condition((X > 1) & (X < 1))

    with pytest.raises(ValueError):
        spe.condition(X << {1})

    with pytest.raises(ValueError):
        spe.sample_func(lambda Z: Z**2, 1)

    x = spe.logprob((X << {1, 2}) | (X < -1))
    assert allclose(x, spe.logprob(X < -1))

    with pytest.raises(AssertionError):
        spe.logprob(Id('Y') << {1, 2})

def test_numeric_distribution_gamma():
    X = Id('X')

    spe = (X >> gamma(a=1, scale=1))
    with pytest.raises(ValueError):
        spe.condition((X << {1, 2}) | (X < 0))

    # Intentionally set Reals as the domain to exercise an important
    # code path in dist.condition (Union case with zero weights).
    spe = ContinuousLeaf(X, scipy.stats.gamma(a=1, scale=1), Reals)
    assert isinf_neg(spe.logprob((X << {1, 2}) | (X < 0)))
    with pytest.raises(ValueError):
        spe.condition((X << {1, 2}) | (X < 0))

    spe_condition = spe.condition((X << {1,2} | (X <= 3)))
    assert isinstance(spe_condition, ContinuousLeaf)
    assert spe_condition.conditioned
    assert spe_condition.support == Interval(-oo, 3)
    assert allclose(
        spe_condition.logprob(X <= 2),
        logdiffexp(spe.logprob(X<=2), spe.logprob(X<=0))
            - spe_condition.logZ)

    # Support on (-3, oo)
    spe = (X >> gamma(loc=-3, a=1))
    assert spe.prob((-3 < X) < 0) > 0.95

    # Constrain.
    with pytest.raises(Exception):
        spe.constrain({X: -4})
    spe_constrain = spe.constrain({X: .5})
    samples = spe_constrain.sample(100, prng=None)
    assert all(s == {X: .5} for s in samples)
