# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from math import log

import pytest

from sppl.distributions import DistributionMix
from sppl.distributions import bernoulli
from sppl.distributions import choice
from sppl.distributions import discrete
from sppl.distributions import norm
from sppl.distributions import poisson
from sppl.distributions import rv_discrete
from sppl.distributions import uniformd
from sppl.math_util import allclose
from sppl.sets import FiniteNominal
from sppl.sets import Interval
from sppl.sets import inf as oo
from sppl.spn import ContinuousLeaf
from sppl.spn import DiscreteLeaf
from sppl.spn import NominalLeaf
from sppl.spn import SumSPN
from sppl.transforms import Id

X = Id('X')

def test_simple_parse_real():
    assert isinstance(.3*bernoulli(p=.1), DistributionMix)
    a = .3*bernoulli(p=.1) | .5 * norm() | .2*poisson(mu=7)
    spn = a(X)
    assert isinstance(spn, SumSPN)
    assert allclose(spn.weights, [log(.3), log(.5), log(.2)])
    assert isinstance(spn.children[0], DiscreteLeaf)
    assert isinstance(spn.children[1], ContinuousLeaf)
    assert isinstance(spn.children[2], DiscreteLeaf)
    assert spn.children[0].support == Interval(0, 1)
    assert spn.children[1].support == Interval(-oo, oo)
    assert spn.children[2].support == Interval(0, oo)

def test_simple_parse_nominal():
    assert isinstance(.7 * choice({'a': .1, 'b': .9}), DistributionMix)
    a = .3*bernoulli(p=.1) | .7*choice({'a': .1, 'b': .9})
    spn = a(X)
    assert isinstance(spn, SumSPN)
    assert allclose(spn.weights, [log(.3), log(.7)])
    assert isinstance(spn.children[0], DiscreteLeaf)
    assert isinstance(spn.children[1], NominalLeaf)
    assert spn.children[0].support == Interval(0, 1)
    assert spn.children[1].support == FiniteNominal('a', 'b')

def test_error():
    with pytest.raises(TypeError):
        'a'*bernoulli(p=.1)
    a = .1  *bernoulli(p=.1) | .7*poisson(mu=8)
    with pytest.raises(Exception):
        a(X)

def test_parse_rv_discrete():
    for dist in [
        rv_discrete(values=((1, 2, 10), (.3, .5, .2))),
        discrete({1: .3, 2: .5, 10: .2})
    ]:
        spn = dist(X)
        assert spn.support == Interval(1, 10)
        assert allclose(spn.prob(X<<{1}), .3)
        assert allclose(spn.prob(X<<{2}), .5)
        assert allclose(spn.prob(X<<{10}), .2)
        assert allclose(spn.prob(X<=10), 1)

    dist = uniformd(values=((1, 2, 10, 0)))
    spn = dist(X)
    assert spn.support == Interval(0, 10)
    assert allclose(spn.prob(X<<{1}), .25)
    assert allclose(spn.prob(X<<{2}), .25)
    assert allclose(spn.prob(X<<{10}), .25)
    assert allclose(spn.prob(X<<{0}), .25)
