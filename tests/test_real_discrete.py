# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

import pytest

import numpy
import sympy

from spn.distributions import Poisson
from spn.distributions import Randint
from spn.math_util import allclose
from spn.math_util import logdiffexp
from spn.math_util import logsumexp
from spn.spn import DiscreteLeaf
from spn.spn import SumSPN
from spn.transforms import Identity

rng = numpy.random.RandomState(1)

def test_poisson():
    X = Identity('X')
    spn = X >> Poisson(mu=5)

    a = spn.logprob((1 <= X) <= 7)
    b = spn.logprob(X << {1,2,3,4,5,6,7})
    c = logsumexp([spn.logprob(X << {i}) for i in range(1, 8)])
    assert allclose(a, b)
    assert allclose(a, c)
    assert allclose(b, c)

    spn_condition = spn.condition(10 <= X)
    assert spn_condition.conditioned
    assert spn_condition.support == sympy.Range(10, sympy.oo)
    assert spn_condition.logZ == logdiffexp(0, spn.logprob(X<=9))

    assert allclose(
        spn_condition.logprob(X <= 10),
        spn_condition.logprob(X << {10}))
    assert allclose(
        spn_condition.logprob(X <= 10),
        spn_condition.logpdf(10))

    samples = spn_condition.sample(100, rng)
    assert all(10 <= s[X] for s in samples)

    # Unify X = 5 with left interval to make one distribution.
    event = ((1 <= X) < 5) | ((3*X + 1) << {16})
    spn_condition = spn.condition(event)
    assert isinstance(spn_condition, DiscreteLeaf)
    assert spn_condition.conditioned
    assert spn_condition.xl == 1
    assert spn_condition.xu == 5
    assert spn_condition.support == sympy.Range(1, 6, 1)
    samples = spn_condition.sample(100, rng)
    assert all(event.evaluate(s) for s in samples)

    # Ignore X = 14/3 as a probability zero condition.
    spn_condition = spn.condition(((1 <= X) < 5) | (3*X + 1) << {15})
    assert isinstance(spn_condition, DiscreteLeaf)
    assert spn_condition.conditioned
    assert spn_condition.xl == 1
    assert spn_condition.xu == 4
    assert spn_condition.support == sympy.Range(1, 5, 1)

    # Make a mixture of two components.
    spn_condition = spn.condition(((1 <= X) < 5) | (3*X + 1) << {22})
    assert isinstance(spn_condition, SumSPN)
    assert spn_condition.children[0].conditioned
    assert spn_condition.children[0].xl == 1
    assert spn_condition.children[0].xu == 4
    assert spn_condition.children[0].support == sympy.Range(1, 5, 1)
    assert spn_condition.children[1].conditioned
    assert spn_condition.children[1].xl == 7
    assert spn_condition.children[1].xu == 7
    assert spn_condition.children[1].support == {7}

    # Condition on probability zero event.
    with pytest.raises(ValueError):
        spn.condition(((-3 <= X) < 0) | (3*X + 1) << {20})

def test_randint():
    X = Identity('X')
    spn = X >> Randint(low=0, high=5)
    assert spn.logprob(X < 5) == spn.logprob(X <= 4) == 0
    # i.e., X is not in [0, 3]
    spn_condition = spn.condition(~((X+1) << {1, 4}))
    assert isinstance(spn_condition, SumSPN)
    assert spn_condition.children[0].support == sympy.Range(1, 3)
    assert spn_condition.children[1].support == sympy.Range(4, 5)
    assert spn_condition
