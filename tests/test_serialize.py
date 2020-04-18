# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

import pytest

from spn.distributions import gamma
from spn.distributions import norm
from spn.distributions import poisson
from spn.serialize import spn_from_json
from spn.serialize import spn_to_json
from spn.transforms import Identity

X = Identity('X')
Y = Identity('Y')

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
