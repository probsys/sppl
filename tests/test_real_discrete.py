# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

import pytest

import numpy

from spn.distributions import poisson
from spn.distributions import randint
from spn.math_util import allclose
from spn.math_util import logdiffexp
from spn.math_util import logsumexp
from spn.spn import DiscreteLeaf
from spn.spn import SumSPN
from spn.transforms import Id
from spn.sets import Range
from spn.sets import Interval
from spn.sets import inf as oo

def test_poisson():
    X = Id('X')
    spn = X >> poisson(mu=5)

    a = spn.logprob((1 <= X) <= 7)
    b = spn.logprob(X << {1,2,3,4,5,6,7})
    c = logsumexp([spn.logprob(X << {i}) for i in range(1, 8)])
    assert allclose(a, b)
    assert allclose(a, c)
    assert allclose(b, c)

    spn_condition = spn.condition(10 <= X)
    assert spn_condition.conditioned
    assert spn_condition.support == Range(10, oo)
    assert spn_condition.logZ == logdiffexp(0, spn.logprob(X<=9))

    assert allclose(
        spn_condition.logprob(X <= 10),
        spn_condition.logprob(X << {10}))
    assert allclose(
        spn_condition.logprob(X <= 10),
        spn_condition.logpdf(X << {10}))

    samples = spn_condition.sample(100)
    assert all(10 <= s[X] for s in samples)

    # Unify X = 5 with left interval to make one distribution.
    event = ((1 <= X) < 5) | ((3*X + 1) << {16})
    spn_condition = spn.condition(event)
    assert isinstance(spn_condition, DiscreteLeaf)
    assert spn_condition.conditioned
    assert spn_condition.xl == 1
    assert spn_condition.xu == 5
    assert spn_condition.support == Range(1, 5)
    samples = spn_condition.sample(100, prng=numpy.random.RandomState(1))
    assert all(event.evaluate(s) for s in samples)

    # Ignore X = 14/3 as a probability zero condition.
    spn_condition = spn.condition(((1 <= X) < 5) | (3*X + 1) << {15})
    assert isinstance(spn_condition, DiscreteLeaf)
    assert spn_condition.conditioned
    assert spn_condition.xl == 1
    assert spn_condition.xu == 4
    assert spn_condition.support == Interval.Ropen(1,5)

    # Make a mixture of two components.
    spn_condition = spn.condition(((1 <= X) < 5) | (3*X + 1) << {22})
    assert isinstance(spn_condition, SumSPN)
    xl = spn_condition.children[0].xl
    idx0 = 0 if xl == 7 else 1
    idx1 = 1 if xl == 7 else 0
    assert spn_condition.children[idx1].conditioned
    assert spn_condition.children[idx1].xl == 1
    assert spn_condition.children[idx1].xu == 4
    assert spn_condition.children[idx0].conditioned
    assert spn_condition.children[idx0].xl == 7
    assert spn_condition.children[idx0].xu == 7
    assert spn_condition.children[idx0].support == Range(7, 7)

    # Condition on probability zero event.
    with pytest.raises(ValueError):
        spn.condition(((-3 <= X) < 0) | (3*X + 1) << {20})

    # Condition on FiniteReal contiguous.
    spn_condition = spn.condition(X << {1,2,3})
    assert spn_condition.xl == 1
    assert spn_condition.xu == 3
    assert allclose(spn_condition.logprob((1 <= X) <=3), 0)

    # https://github.com/probcomp/sum-product-dsl/issues/77
    # Condition on FiniteReal non-contiguous.
    with pytest.raises(Exception):
        spn_condition = spn.condition(X << {1,2,3,5})

def test_randint():
    X = Id('X')
    spn = X >> randint(low=0, high=5)
    assert spn.xl == 0
    assert spn.xu == 4
    assert spn.logprob(X < 5) == spn.logprob(X <= 4) == 0
    # i.e., X is not in [0, 3]
    spn_condition = spn.condition(~((X+1) << {1, 4}))
    assert isinstance(spn_condition, SumSPN)
    xl = spn_condition.children[0].xl
    idx0 = 0 if xl == 1 else 1
    idx1 = 1 if xl == 1 else 0
    assert spn_condition.children[idx0].xl == 1
    assert spn_condition.children[idx0].xu == 2
    assert spn_condition.children[idx1].xl == 4
    assert spn_condition.children[idx1].xu == 4
    assert allclose(spn_condition.children[idx0].logprob(X<<{1,2}), 0)
    assert allclose(spn_condition.children[idx1].logprob(X<<{4}), 0)
