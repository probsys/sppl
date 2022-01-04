# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

import json
import pytest

from sympy import sqrt

from sppl.compilers.spe_to_dict import spe_from_dict
from sppl.compilers.spe_to_dict import spe_to_dict
from sppl.distributions import choice
from sppl.distributions import gamma
from sppl.distributions import norm
from sppl.distributions import poisson
from sppl.sets import EmptySet
from sppl.transforms import EventFiniteNominal
from sppl.transforms import Exp
from sppl.transforms import Exponential
from sppl.transforms import Id
from sppl.transforms import Log
from sppl.transforms import Logarithm

X = Id('X')
Y = Id('Y')

spes = [
    X << norm(loc=0, scale=1),
    X << poisson(mu=7),
    Y << choice({'a': 0.5, 'b': 0.5}),
    (X << norm(loc=0, scale=1)) & (Y << gamma(a=1)),
    0.2*(X << norm(loc=0, scale=1)) | 0.8*(X << gamma(a=1)),
    ((X << norm(loc=0, scale=1)) & (Y << gamma(a=1))).constrain({Y:1}),
]
@pytest.mark.parametrize('spe', spes)
def test_serialize_equal(spe):
    metadata = spe_to_dict(spe)
    spe_json_encoded = json.dumps(metadata)
    spe_json_decoded = json.loads(spe_json_encoded)
    spe2 = spe_from_dict(spe_json_decoded)
    assert spe2 == spe

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
    X < sqrt(3),
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
    spe = (X << norm()).transform(Y, transform)
    metadata = spe_to_dict(spe)
    spe_json_encoded = json.dumps(metadata)
    spe_json_decoded = json.loads(spe_json_encoded)
    spe2 = spe_from_dict(spe_json_decoded)
    assert spe2 == spe
