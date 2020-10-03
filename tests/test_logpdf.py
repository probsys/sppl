# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from math import log

import pytest

from sppl.distributions import gamma
from sppl.distributions import norm
from sppl.distributions import poisson
from sppl.math_util import allclose
from sppl.math_util import isinf_neg
from sppl.math_util import logsumexp
from sppl.transforms import Id
from sppl.spn import SumSPN

X = Id('X')
Y = Id('Y')
Z = Id('Z')

def test_logpdf_real_continuous():
    spn = (X >> norm())
    assert allclose(spn.logpdf(X << {0}), norm().dist.logpdf(0))

def test_logpdf_real_discrete():
    spn = (X >> poisson(mu=2))
    assert isinf_neg(spn.logpdf(X << {1.5}))
    assert isinf_neg(spn.logpdf(X << {'1'}))
    assert not isinf_neg(spn.logpdf(X<< {0}))

def test_logpdf_nominal():
    spn = (X >> {'a' : .6, 'b': .4})
    assert isinf_neg(spn.logpdf(X << {1.5}))
    allclose(spn.logpdf(X << {'a'}), log(.6))

def test_logpdf_mixture_real_continuous_continuous():
    spn = X >> (.3*norm() | .7*gamma(a=1))
    assert allclose(
        spn.logpdf(X << {.5}),
        logsumexp([
            log(.3) + spn.children[0].logpdf(X<<{0.5}),
            log(.7) + spn.children[1].logpdf(X<<{0.5}),
        ]))

@pytest.mark.xfail
def test_logpdf_mixture_real_continuous_discrete():
    spn = X >> (.3*norm() | .7*poisson(mu=1))
    assert allclose(
        spn.logpdf(X << {.5}),
        logsumexp([
            log(.3) + spn.children[0].logpdf(X<<{0.5}),
            log(.7) + spn.children[1].logpdf(X<<{0.5}),
        ]))
    assert False, 'Invalid base measure addition'

def test_logpdf_mixture_nominal():
    spn = SumSPN([X >> norm(), X >> {'a':.1, 'b':.9}], [log(.4), log(.6)])
    assert allclose(
        spn.logpdf(X << {.5}),
        log(.4) + spn.children[0].logpdf(X<<{.5}))
    assert allclose(
        spn.logpdf(X << {'a'}),
        log(.6) + spn.children[1].logpdf(X<<{'a'}))

def test_logpdf_error_or():
    spn = (X >> norm())
    with pytest.raises(Exception):
        spn.logpdf((X << {1}) | (X << {2}))
    spn = (X >> norm()) & (Y >> {'a':.1, 'b':.9})
    with pytest.raises(Exception):
        spn.logpdf((X << {1}) | ~(Y << {'a'}))

def test_logpdf_error_event():
    spn = (X >> norm())
    with pytest.raises(Exception):
        spn.logpdf(X < 1)

def test_logpdf_error_transform():
    spn = (X >> norm())
    with pytest.raises(Exception):
        spn.logpdf(X**2 << {0})

def test_logpdf_error_not():
    spn = (Y >> {'a':.8, 'b': .2})
    with pytest.raises(Exception):
        spn.logpdf(~(Y << {'a'}))
