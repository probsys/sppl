# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from math import log

import pytest

from spn.distributions import DistributionMix
from spn.distributions import bernoulli
from spn.distributions import choice
from spn.distributions import norm
from spn.distributions import poisson
from spn.math_util import allclose
from spn.spn import ContinuousLeaf
from spn.spn import DiscreteLeaf
from spn.spn import NominalLeaf
from spn.spn import SumSPN
from spn.transforms import Id

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

def test_simple_parse_nominal():
    assert isinstance(.7 * choice({'a': .1, 'b': .9}), DistributionMix)
    a = .3*bernoulli(p=.1) | .7*choice({'a': .1, 'b': .9})
    spn = a(X)
    assert isinstance(spn, SumSPN)
    assert allclose(spn.weights, [log(.3), log(.7)])
    assert isinstance(spn.children[0], DiscreteLeaf)
    assert isinstance(spn.children[1], NominalLeaf)

def test_error():
    with pytest.raises(TypeError):
        'a'*bernoulli(p=.1)
    a = .1  *bernoulli(p=.1) | .7*poisson(mu=8)
    with pytest.raises(Exception):
        a(X)
