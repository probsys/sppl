# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

import pytest

import numpy

from sppl.distributions import poisson
from sppl.distributions import randint
from sppl.math_util import allclose
from sppl.math_util import logdiffexp
from sppl.math_util import logsumexp
from sppl.sets import Interval
from sppl.sets import Range
from sppl.sets import inf as oo
from sppl.spe import DiscreteLeaf
from sppl.spe import SumSPE
from sppl.transforms import Id

def test_poisson():
    X = Id('X')
    spe = X >> poisson(mu=5)

    a = spe.logprob((1 <= X) <= 7)
    b = spe.logprob(X << {1,2,3,4,5,6,7})
    c = logsumexp([spe.logprob(X << {i}) for i in range(1, 8)])
    assert allclose(a, b)
    assert allclose(a, c)
    assert allclose(b, c)

    spe_condition = spe.condition(10 <= X)
    assert spe_condition.conditioned
    assert spe_condition.support == Range(10, oo)
    assert spe_condition.logZ == logdiffexp(0, spe.logprob(X<=9))

    assert allclose(
        spe_condition.logprob(X <= 10),
        spe_condition.logprob(X << {10}))
    assert allclose(
        spe_condition.logprob(X <= 10),
        spe_condition.logpdf({X: 10}))

    samples = spe_condition.sample(100)
    assert all(10 <= s[X] for s in samples)

    # Unify X = 5 with left interval to make one distribution.
    event = ((1 <= X) < 5) | ((3*X + 1) << {16})
    spe_condition = spe.condition(event)
    assert isinstance(spe_condition, DiscreteLeaf)
    assert spe_condition.conditioned
    assert spe_condition.xl == 1
    assert spe_condition.xu == 5
    assert spe_condition.support == Range(1, 5)
    samples = spe_condition.sample(100, prng=numpy.random.RandomState(1))
    assert all(event.evaluate(s) for s in samples)

    # Ignore X = 14/3 as a probability zero condition.
    spe_condition = spe.condition(((1 <= X) < 5) | (3*X + 1) << {15})
    assert isinstance(spe_condition, DiscreteLeaf)
    assert spe_condition.conditioned
    assert spe_condition.xl == 1
    assert spe_condition.xu == 4
    assert spe_condition.support == Interval.Ropen(1,5)

    # Make a mixture of two components.
    spe_condition = spe.condition(((1 <= X) < 5) | (3*X + 1) << {22})
    assert isinstance(spe_condition, SumSPE)
    xl = spe_condition.children[0].xl
    idx0 = 0 if xl == 7 else 1
    idx1 = 1 if xl == 7 else 0
    assert spe_condition.children[idx1].conditioned
    assert spe_condition.children[idx1].xl == 1
    assert spe_condition.children[idx1].xu == 4
    assert spe_condition.children[idx0].conditioned
    assert spe_condition.children[idx0].xl == 7
    assert spe_condition.children[idx0].xu == 7
    assert spe_condition.children[idx0].support == Range(7, 7)

    # Condition on probability zero event.
    with pytest.raises(ValueError):
        spe.condition(((-3 <= X) < 0) | (3*X + 1) << {20})

    # Condition on FiniteReal contiguous.
    spe_condition = spe.condition(X << {1,2,3})
    assert spe_condition.xl == 1
    assert spe_condition.xu == 3
    assert allclose(spe_condition.logprob((1 <= X) <=3), 0)

    # Condition on single point.
    assert allclose(0, spe.condition(X << {2}).logprob(X<<{2}))

    # Constrain.
    with pytest.raises(Exception):
        spe.constrain({X: -1})
    with pytest.raises(Exception):
        spe.constrain({X: .5})
    spe_constrain = spe.constrain({X: 10})
    assert allclose(spe_constrain.prob(X << {0, 10}), 1)

def test_condition_non_contiguous():
    X = Id('X')
    spe = X >> poisson(mu=5)
    # FiniteSet.
    for c in [{0,2,3}, {-1,0,2,3}, {-1,0,2,3,'z'}]:
        spe_condition = spe.condition((X << c))
        assert isinstance(spe_condition, SumSPE)
        assert allclose(0, spe_condition.children[0].logprob(X<<{0}))
        assert allclose(0, spe_condition.children[1].logprob(X<<{2,3}))
    # FiniteSet or Interval.
    spe_condition = spe.condition((X << {-1,'x',0,2,3}) | (X > 7))
    assert isinstance(spe_condition, SumSPE)
    assert len(spe_condition.children) == 3
    assert allclose(0, spe_condition.children[0].logprob(X<<{0}))
    assert allclose(0, spe_condition.children[1].logprob(X<<{2,3}))
    assert allclose(0, spe_condition.children[2].logprob(X>7))

def test_randint():
    X = Id('X')
    spe = X >> randint(low=0, high=5)
    assert spe.xl == 0
    assert spe.xu == 4
    assert spe.logprob(X < 5) == spe.logprob(X <= 4) == 0
    # i.e., X is not in [0, 3]
    spe_condition = spe.condition(~((X+1) << {1, 4}))
    assert isinstance(spe_condition, SumSPE)
    xl = spe_condition.children[0].xl
    idx0 = 0 if xl == 1 else 1
    idx1 = 1 if xl == 1 else 0
    assert spe_condition.children[idx0].xl == 1
    assert spe_condition.children[idx0].xu == 2
    assert spe_condition.children[idx1].xl == 4
    assert spe_condition.children[idx1].xu == 4
    assert allclose(spe_condition.children[idx0].logprob(X<<{1,2}), 0)
    assert allclose(spe_condition.children[idx1].logprob(X<<{4}), 0)
