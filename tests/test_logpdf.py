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
from sppl.spe import SumSPE
from sppl.spe import ProductSPE
from sppl.transforms import Id

X = Id('X')
Y = Id('Y')
Z = Id('Z')

def test_logpdf_real_continuous():
    spe = (X >> norm())
    assert allclose(spe.logpdf({X: 0}), norm().dist.logpdf(0))

def test_logpdf_real_discrete():
    spe = (X >> poisson(mu=2))
    assert isinf_neg(spe.logpdf({X: 1.5}))
    assert isinf_neg(spe.logpdf({X: '1'}))
    assert not isinf_neg(spe.logpdf({X: 0}))

def test_logpdf_nominal():
    spe = (X >> choice({'a' : .6, 'b': .4}))
    assert isinf_neg(spe.logpdf({X: 1.5}))
    allclose(spe.logpdf({X: 'a'}), log(.6))

def test_logpdf_mixture_real_continuous_continuous():
    spe = X >> (.3*norm() | .7*gamma(a=1))
    assert allclose(
        spe.logpdf({X: .5}),
        logsumexp([
            log(.3) + spe.children[0].logpdf({X: 0.5}),
            log(.7) + spe.children[1].logpdf({X: 0.5}),
        ]))

@pytest.mark.xfail
def test_logpdf_mixture_real_continuous_discrete():
    spe = X >> (.3*norm() | .7*poisson(mu=1))
    assert allclose(
        spe.logpdf(X << {.5}),
        logsumexp([
            log(.3) + spe.children[0].logpdf({X: 0.5}),
            log(.7) + spe.children[1].logpdf({X: 0.5}),
        ]))
    assert False, 'Invalid base measure addition'

def test_logpdf_mixture_nominal():
    spe = SumSPE([X >> norm(), X >> choice({'a':.1, 'b':.9})], [log(.4), log(.6)])
    assert allclose(
        spe.logpdf({X: .5}),
        log(.4) + spe.children[0].logpdf({X: .5}))
    assert allclose(
        spe.logpdf({X: 'a'}),
        log(.6) + spe.children[1].logpdf({X: 'a'}))

def test_logpdf_error_event():
    spe = (X >> norm())
    with pytest.raises(Exception):
        spe.logpdf(X < 1)

def test_logpdf_error_transform_base():
    spe = (X >> norm())
    with pytest.raises(Exception):
        spe.logpdf({X**2: 0})

def test_logpdf_error_transform_env():
    spe = (X >> norm()).transform(Z, X**2)
    with pytest.raises(Exception):
        spe.logpdf({Z: 0})

def test_logpdf_bivariate():
    spe = (X >> norm()) & (Y >> choice({'a': .5, 'b': .5}))
    assert allclose(
        spe.logpdf({X: 0, Y: 'a'}),
        norm().dist.logpdf(0) + log(.5))

def test_logpdf_lexicographic_either():
    spe = .75*(X >> norm() & Y >> atomic(loc=0) & Z >> discrete({1:.1, 2:.9})) \
        | .25*(X >> atomic(loc=0) & Y >> norm() & Z >> norm())
    # Lexicographic, Branch 1
    assignment = {X:0, Y:0, Z:2}
    assert allclose(
        spe.logpdf(assignment),
        log(.75) + norm().dist.logpdf(0) + log(1) + log(.9))
    assert isinstance(spe.constrain(assignment), ProductSPE)
    # Lexicographic, Branch 2
    assignment = {X:0, Y:0, Z:0}
    assert allclose(
        spe.logpdf(assignment),
        log(.25) + log(1) + norm().dist.logpdf(0) + norm().dist.logpdf(0))
    assert isinstance(spe.constrain(assignment), ProductSPE)

def test_logpdf_lexicographic_both():
    spe = .75*(X >> norm() & Y >> atomic(loc=0) & Z >> discrete({1:.2, 2:.8})) \
        | .25*(X >> discrete({1:.5, 2:.5}) & Y >> norm() & Z >> atomic(loc=2))
    # Lexicographic, Mix
    assignment = {X:1, Y:0, Z:2}
    assert allclose(
        spe.logpdf(assignment),
        logsumexp([
            log(.75) + norm().dist.logpdf(1) + log(1) + log(.8),
            log(.25) + log(.5) + norm().dist.logpdf(0) + log(1)]))
    assert isinstance(spe.constrain(assignment), SumSPE)
