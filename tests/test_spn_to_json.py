# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

import pytest

from spn.distributions import gamma
from spn.distributions import norm
from spn.distributions import poisson
from spn.compilers.spn_to_json import spn_from_json
from spn.compilers.spn_to_json import spn_to_json
from spn.sym_util import EmptySet
from spn.transforms import EventFiniteNominal
from spn.transforms import Exp
from spn.transforms import Exponential
from spn.transforms import Id
from spn.transforms import Log
from spn.transforms import Logarithm

X = Id('X')
Y = Id('Y')

spns = [
    X >> norm(loc=0, scale=1),
    X >> poisson(mu=7),
    Y >> {'a': 0.5, 'b': 0.5},
    (X >> norm(loc=0, scale=1)) & (Y >> gamma(a=1)),
    0.2*(X >> norm(loc=0, scale=1)) | 0.8*(X >> gamma(a=1)),
]
@pytest.mark.parametrize('spn', spns)
def test_serialize_equal(spn):
    metadata = spn_to_json(spn)
    spn_loaded = spn_from_json(metadata)
    assert spn_loaded == spn

transforms = [
    X,
    X**(1,3),
    Exponential(X, base=3),
    Logarithm(X, base=2),
    2**Log(X),
    1/Exp(X),
    abs(X),
    1/X,
    2*X + X**3,
    (X/2)*(X<0) + (X**(1,2))*(0<=X),
    X < 3,
    X << [],
    ~(X << []),
    EventFiniteNominal(1/X**(1,10), EmptySet),
    X << {1, 2},
    X << {'a', 'x'},
    ~(X << {'a', '1'}),
    (X < 3) | (X << {1,2}),
    (X < 3) & (X << {1,2}),
]
@pytest.mark.parametrize('transform', transforms)
def test_serialize_env(transform):
    spn = (X >> norm()).transform(Y, transform)
    metadata = spn_to_json(spn)
    spn_loaded = spn_from_json(metadata)
    assert spn_loaded == spn
