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
    spn = (X >> norm(loc=0, scale=1))

    with pytest.raises(AssertionError):
        spn.transform(Z, Y**2)
    with pytest.raises(AssertionError):
        spn.transform(X, X**2)

    spn = spn.transform(Z, X**2)
    assert spn.env == {X:X, Z:X**2}
    assert spn.get_symbols() == {X, Z}
    assert spn.logprob(Z < 1) == spn.logprob(X**2 < 1)
    assert spn.logprob((Z < 1) | ((X + 1) < 3)) \
        == spn.logprob((X**2 < 1) | ((X+1) < 3))

    spn = spn.transform(Y, 2*Z)
    assert spn.env == {X:X, Z:X**2, Y:2*Z}
    assert spn.logprob(Y**(1,3) < 10) \
        == spn.logprob((2*Z)**(1,3) < 10) \
        == spn.logprob((2*(X**2))**(1,3) < 10) \

    W = Id('W')
    spn = spn.transform(W, X > 1)
    assert allclose(spn.logprob(W), spn.logprob(X > 1))

def test_transform_real_leaf_sample():
    X = Id('X')
    Z = Id('Z')
    Y = Id('Y')
    spn = (X >> poisson(loc=-1, mu=1))
    spn = spn.transform(Z, X+1)
    spn = spn.transform(Y, Z-1)
    samples = spn.sample(100)
    assert any(s[X] == -1 for s in samples)
    assert all(0 <= s[Z] for s in samples)
    assert all(s[Y] == s[X] for s in samples)
    assert all(spn.sample_func(lambda X,Y,Z: X-Y+Z==Z, 100))
    assert all(set(s) == {X,Y} for s in spn.sample_subset([X, Y], 100))

def test_transform_sum():
    X = Id('X')
    Z = Id('Z')
    Y = Id('Y')
    spn \
        = 0.3*(X >> norm(loc=0, scale=1)) \
        | 0.7*(X >> choice({'0': 0.4, '1': 0.6}))
    with pytest.raises(Exception):
        # Cannot transform Nominal variate.
        spn.transform(Z, X**2)
    spn \
        = 0.3*(X >> norm(loc=0, scale=1)) \
        | 0.7*(X >> poisson(mu=2))
    spn = spn.transform(Z, X**2)
    assert spn.logprob(Z < 1) == spn.logprob(X**2 < 1)
    assert spn.children[0].env == spn.children[1].env
    spn = spn.transform(Y, Z/2)
    assert spn.children[0].env \
        == spn.children[1].env \
        == {X:X, Z:X**2, Y:Z/2}

def test_transform_product():
    X = Id('X')
    Y = Id('Y')
    W = Id('W')
    Z = Id('Z')
    V = Id('V')
    spn \
        = (X >> norm(loc=0, scale=1)) \
        & (Y >> poisson(mu=10))
    with pytest.raises(Exception):
        # Cannot use symbols from different transforms.
        spn.transform(W, (X > 0) | (Y << {'0'}))
    spn = spn.transform(W, (X**2 - 3*X)**(1,10))
    spn = spn.transform(Z, (W > 0) | (X**3 < 1))
    spn = spn.transform(V, Y/10)
    assert allclose(
        spn.logprob(W>1),
        spn.logprob((X**2 - 3*X)**(1,10) > 1))
    with pytest.raises(Exception):
        spn.tarnsform(Id('R'), (V>1) | (W < 0))
