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
from sppl.spe import ContinuousLeaf
from sppl.spe import DiscreteLeaf
from sppl.spe import NominalLeaf
from sppl.spe import SumSPE
from sppl.transforms import Id

X = Id('X')

def test_simple_parse_real():
    assert isinstance(.3*bernoulli(p=.1), DistributionMix)
    a = .3*bernoulli(p=.1) | .5 * norm() | .2*poisson(mu=7)
    spe = a(X)
    assert isinstance(spe, SumSPE)
    assert allclose(spe.weights, [log(.3), log(.5), log(.2)])
    assert isinstance(spe.children[0], DiscreteLeaf)
    assert isinstance(spe.children[1], ContinuousLeaf)
    assert isinstance(spe.children[2], DiscreteLeaf)
    assert spe.children[0].support == Interval(0, 1)
    assert spe.children[1].support == Interval(-oo, oo)
    assert spe.children[2].support == Interval(0, oo)

def test_simple_parse_nominal():
    assert isinstance(.7 * choice({'a': .1, 'b': .9}), DistributionMix)
    a = .3*bernoulli(p=.1) | .7*choice({'a': .1, 'b': .9})
    spe = a(X)
    assert isinstance(spe, SumSPE)
    assert allclose(spe.weights, [log(.3), log(.7)])
    assert isinstance(spe.children[0], DiscreteLeaf)
    assert isinstance(spe.children[1], NominalLeaf)
    assert spe.children[0].support == Interval(0, 1)
    assert spe.children[1].support == FiniteNominal('a', 'b')

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
        spe = dist(X)
        assert spe.support == Interval(1, 10)
        assert allclose(spe.prob(X<<{1}), .3)
        assert allclose(spe.prob(X<<{2}), .5)
        assert allclose(spe.prob(X<<{10}), .2)
        assert allclose(spe.prob(X<=10), 1)

    dist = uniformd(values=((1, 2, 10, 0)))
    spe = dist(X)
    assert spe.support == Interval(0, 10)
    assert allclose(spe.prob(X<<{1}), .25)
    assert allclose(spe.prob(X<<{2}), .25)
    assert allclose(spe.prob(X<<{10}), .25)
    assert allclose(spe.prob(X<<{0}), .25)
