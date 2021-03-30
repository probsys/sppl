# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

import pytest

from sppl.distributions import choice
from sppl.distributions import norm
from sppl.distributions import poisson
from sppl.math_util import allclose
from sppl.transforms import Id

def test_transform_real_leaf_logprob():
    X = Id('X')
    Y = Id('Y')
    Z = Id('Z')
    spe = (X >> norm(loc=0, scale=1))

    with pytest.raises(AssertionError):
        spe.transform(Z, Y**2)
    with pytest.raises(AssertionError):
        spe.transform(X, X**2)

    spe = spe.transform(Z, X**2)
    assert spe.env == {X:X, Z:X**2}
    assert spe.get_symbols() == {X, Z}
    assert spe.logprob(Z < 1) == spe.logprob(X**2 < 1)
    assert spe.logprob((Z < 1) | ((X + 1) < 3)) \
        == spe.logprob((X**2 < 1) | ((X+1) < 3))

    spe = spe.transform(Y, 2*Z)
    assert spe.env == {X:X, Z:X**2, Y:2*Z}
    assert spe.logprob(Y**(1,3) < 10) \
        == spe.logprob((2*Z)**(1,3) < 10) \
        == spe.logprob((2*(X**2))**(1,3) < 10) \

    W = Id('W')
    spe = spe.transform(W, X > 1)
    assert allclose(spe.logprob(W), spe.logprob(X > 1))

def test_transform_real_leaf_sample():
    X = Id('X')
    Z = Id('Z')
    Y = Id('Y')
    spe = (X >> poisson(loc=-1, mu=1))
    spe = spe.transform(Z, X+1)
    spe = spe.transform(Y, Z-1)
    samples = spe.sample(100)
    assert any(s[X] == -1 for s in samples)
    assert all(0 <= s[Z] for s in samples)
    assert all(s[Y] == s[X] for s in samples)
    assert all(spe.sample_func(lambda X,Y,Z: X-Y+Z==Z, 100))
    assert all(set(s) == {X,Y} for s in spe.sample_subset([X, Y], 100))

def test_transform_sum():
    X = Id('X')
    Z = Id('Z')
    Y = Id('Y')
    spe \
        = 0.3*(X >> norm(loc=0, scale=1)) \
        | 0.7*(X >> choice({'0': 0.4, '1': 0.6}))
    with pytest.raises(Exception):
        # Cannot transform Nominal variate.
        spe.transform(Z, X**2)
    spe \
        = 0.3*(X >> norm(loc=0, scale=1)) \
        | 0.7*(X >> poisson(mu=2))
    spe = spe.transform(Z, X**2)
    assert spe.logprob(Z < 1) == spe.logprob(X**2 < 1)
    assert spe.children[0].env == spe.children[1].env
    spe = spe.transform(Y, Z/2)
    assert spe.children[0].env \
        == spe.children[1].env \
        == {X:X, Z:X**2, Y:Z/2}

def test_transform_product():
    X = Id('X')
    Y = Id('Y')
    W = Id('W')
    Z = Id('Z')
    V = Id('V')
    spe \
        = (X >> norm(loc=0, scale=1)) \
        & (Y >> poisson(mu=10))
    with pytest.raises(Exception):
        # Cannot use symbols from different transforms.
        spe.transform(W, (X > 0) | (Y << {'0'}))
    spe = spe.transform(W, (X**2 - 3*X)**(1,10))
    spe = spe.transform(Z, (W > 0) | (X**3 < 1))
    spe = spe.transform(V, Y/10)
    assert allclose(
        spe.logprob(W>1),
        spe.logprob((X**2 - 3*X)**(1,10) > 1))
    with pytest.raises(Exception):
        spe.tarnsform(Id('R'), (V>1) | (W < 0))
