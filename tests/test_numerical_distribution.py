# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from math import log

import pytest
import scipy.stats
import sympy
import numpy

from spn.math_util import allclose
from spn.math_util import isinf_neg
from spn.math_util import logdiffexp
from spn.numerical import Gamma
from spn.numerical import Norm
from spn.spn import NumericalDistribution
from spn.spn import SumSPN
from spn.sym_util import Reals
from spn.transforms import Identity

rng = numpy.random.RandomState(1)

def test_numeric_distribution_normal():
    X = Identity('X')
    spn = Norm(X, loc=0, scale=1)

    assert allclose(spn.logprob(X > 0), -log(2))
    assert allclose(spn.logprob(abs(X) < 2), log(spn.dist.cdf(2) - spn.dist.cdf(-2)))

    assert allclose(spn.logprob(X**2 > 0), 0)
    assert allclose(spn.logprob(abs(X) > 0), 0)
    assert allclose(spn.logprob(~(X << {1})), 0)

    assert isinf_neg(spn.logprob(X**2 - X + 10 < 0))
    assert isinf_neg(spn.logprob(abs(X) < 0))
    assert isinf_neg(spn.logprob(X << {1}))

    spn.sample(100, rng)
    spn.sample_subset([X], 100, rng)
    assert spn.sample_subset([], 100, rng) is None
    spn.sample_func(lambda X: X**2, 1, rng)
    spn.sample_func(lambda X: abs(X)+X**2, 1, rng)
    spn.sample_func(lambda X: X**2 if X > 0 else X**3, 100, rng)

    spn_condition_a = spn.condition((X < 2) | (X > 10))
    samples = spn_condition_a.sample(100, rng)
    assert all(s[X] < 2 for s in samples)

    spn_condition_b = spn.condition((X < -10) | (X > 10))
    assert isinstance(spn_condition_b, SumSPN)
    assert allclose(spn_condition_b.weights[0], -log(2))
    assert allclose(spn_condition_b.weights[0], spn_condition_b.weights[1])

    for event in [(X<-10), (X>3)]:
        spn_condition_c = spn.condition(event)
        assert isinstance(spn_condition_c, NumericalDistribution)
        assert isinf_neg(spn_condition_c.logprob((-1 < X) < 1))
        samples = spn_condition_c.sample(100, rng)
        assert all(s[X] in event.values for s in samples)

    with pytest.raises(ValueError):
        spn.condition((X > 1) & (X < 1))

    with pytest.raises(ValueError):
        spn.condition(X << {1})

    with pytest.raises(ValueError):
        spn.sample_func(lambda Z: Z**2, 1, rng)

    x = spn.logprob((X << {1, 2}) | (X < -1))
    assert allclose(x, spn.logprob(X < -1))

def test_numeric_distribution_gamma():
    X = Identity('X')

    spn = Gamma(X, a=1, scale=1)
    with pytest.raises(ValueError):
        spn.condition((X << {1, 2}) | (X < 0))

    # Intentionally set Reals as the domain to exercise an important
    # code path in dist.condition (Union case with zero weights).
    spn = NumericalDistribution(X, scipy.stats.gamma(a=1, scale=1), Reals)
    assert isinf_neg(spn.logprob((X << {1, 2}) | (X < 0)))
    with pytest.raises(ValueError):
        spn.condition((X << {1, 2}) | (X < 0))

    spn_condition = spn.condition((X << {1,2} | (X <= 3)))
    assert isinstance(spn_condition, NumericalDistribution)
    assert spn_condition.conditioned
    assert spn_condition.support == sympy.Interval(-sympy.oo, 3)
    assert allclose(
        spn_condition.logprob(X <= 2),
        logdiffexp(spn.logprob(X<=2), spn.logprob(X<=0))
            - spn_condition.logZ)
