# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

import pytest

from spn.distributions import Gamma
from spn.distributions import NominalDist
from spn.distributions import Norm
from spn.distributions import Poisson
from spn.serialize import spn_from_json
from spn.serialize import spn_to_json
from spn.transforms import Identity

X = Identity('X')
Y = Identity('Y')

spns = [
    X >> Norm(loc=0, scale=1),
    X >> Poisson(mu=7),
    Y >> NominalDist({'a': 0.5, 'b': 0.5}),
    (X >> Norm(loc=0, scale=1)) & (Y >> Gamma(a=1)),
    0.2*(X >> Norm(loc=0, scale=1)) | 0.8*(X >> Gamma(a=1)),
]
@pytest.mark.parametrize('spn', spns)
def test_serialize_equal(spn):
    metadata = spn_to_json(spn)
    spn_loaded = spn_from_json(metadata)
    assert spn_loaded == spn
