# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from math import log

import pytest

from sppl.distributions import atomic
from sppl.distributions import choice
from sppl.distributions import discrete
from sppl.distributions import gamma
from sppl.distributions import norm
from sppl.distributions import poisson
from sppl.math_util import allclose
from sppl.math_util import isinf_neg
from sppl.math_util import logsumexp
from sppl.spn import SumSPN
from sppl.spn import ProductSPN
from sppl.transforms import Id

X = Id('X')
Y = Id('Y')
Z = Id('Z')

def test_logpdf_real_continuous():
    spn = (X >> norm())
    assert allclose(spn.logpdf({X: 0}), norm().dist.logpdf(0))

def test_logpdf_real_discrete():
    spn = (X >> poisson(mu=2))
    assert isinf_neg(spn.logpdf({X: 1.5}))
    assert isinf_neg(spn.logpdf({X: '1'}))
    assert not isinf_neg(spn.logpdf({X: 0}))

def test_logpdf_nominal():
    spn = (X >> choice({'a' : .6, 'b': .4}))
    assert isinf_neg(spn.logpdf({X: 1.5}))
    allclose(spn.logpdf({X: 'a'}), log(.6))

def test_logpdf_mixture_real_continuous_continuous():
    spn = X >> (.3*norm() | .7*gamma(a=1))
    assert allclose(
        spn.logpdf({X: .5}),
        logsumexp([
            log(.3) + spn.children[0].logpdf({X: 0.5}),
            log(.7) + spn.children[1].logpdf({X: 0.5}),
        ]))

@pytest.mark.xfail
def test_logpdf_mixture_real_continuous_discrete():
    spn = X >> (.3*norm() | .7*poisson(mu=1))
    assert allclose(
        spn.logpdf(X << {.5}),
        logsumexp([
            log(.3) + spn.children[0].logpdf({X: 0.5}),
            log(.7) + spn.children[1].logpdf({X: 0.5}),
        ]))
    assert False, 'Invalid base measure addition'

def test_logpdf_mixture_nominal():
    spn = SumSPN([X >> norm(), X >> choice({'a':.1, 'b':.9})], [log(.4), log(.6)])
    assert allclose(
        spn.logpdf({X: .5}),
        log(.4) + spn.children[0].logpdf({X: .5}))
    assert allclose(
        spn.logpdf({X: 'a'}),
        log(.6) + spn.children[1].logpdf({X: 'a'}))

def test_logpdf_error_event():
    spn = (X >> norm())
    with pytest.raises(Exception):
        spn.logpdf(X < 1)

def test_logpdf_error_transform_base():
    spn = (X >> norm())
    with pytest.raises(Exception):
        spn.logpdf({X**2: 0})

def test_logpdf_error_transform_env():
    spn = (X >> norm()).transform(Z, X**2)
    with pytest.raises(Exception):
        spn.logpdf({Z: 0})

def test_logpdf_bivariate():
    spn = (X >> norm()) & (Y >> choice({'a': .5, 'b': .5}))
    assert allclose(
        spn.logpdf({X: 0, Y: 'a'}),
        norm().dist.logpdf(0) + log(.5))

def test_logpdf_lexicographic_either():
    spn = .75*(X >> norm() & Y >> atomic(loc=0) & Z >> discrete({1:.1, 2:.9})) \
        | .25*(X >> atomic(loc=0) & Y >> norm() & Z >> norm())
    # Lexicographic, Branch 1
    assignment = {X:0, Y:0, Z:2}
    assert allclose(
        spn.logpdf(assignment),
        log(.75) + norm().dist.logpdf(0) + log(1) + log(.9))
    assert isinstance(spn.constrain(assignment), ProductSPN)
    # Lexicographic, Branch 2
    assignment = {X:0, Y:0, Z:0}
    assert allclose(
        spn.logpdf(assignment),
        log(.25) + log(1) + norm().dist.logpdf(0) + norm().dist.logpdf(0))
    assert isinstance(spn.constrain(assignment), ProductSPN)

def test_logpdf_lexicographic_both():
    spn = .75*(X >> norm() & Y >> atomic(loc=0) & Z >> discrete({1:.2, 2:.8})) \
        | .25*(X >> discrete({1:.5, 2:.5}) & Y >> norm() & Z >> atomic(loc=2))
    # Lexicographic, Mix
    assignment = {X:1, Y:0, Z:2}
    assert allclose(
        spn.logpdf(assignment),
        logsumexp([
            log(.75) + norm().dist.logpdf(1) + log(1) + log(.8),
            log(.25) + log(.5) + norm().dist.logpdf(0) + log(1)]))
    assert isinstance(spn.constrain(assignment), SumSPN)
