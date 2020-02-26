# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from math import log

import numpy

from spn.math_util import allclose
from spn.distributions import Norm
from spn.transforms import Identity

rng = numpy.random.RandomState(1)

def test_mutual_information_four_clusters():
    X = Identity('X')
    Y = Identity('Y')
    spn \
        = 0.25*(X >> Norm(loc=0, scale=0.5) & Y >> Norm(loc=0, scale=0.5)) \
        | 0.25*(X >> Norm(loc=5, scale=0.5) & Y >> Norm(loc=0, scale=0.5)) \
        | 0.25*(X >> Norm(loc=0, scale=0.5) & Y >> Norm(loc=5, scale=0.5)) \
        | 0.25*(X >> Norm(loc=5, scale=0.5) & Y >> Norm(loc=5, scale=0.5)) \

    samples = spn.sample(100, rng)
    mi = spn.mutual_information(X > 2, Y > 2)
    assert allclose(mi, 0)

    event = ((X>2) & (Y<2) | ((X<2) & (Y>2)))
    spn_condition = spn.condition(event)
    samples = spn_condition.sample(100, rng)
    assert all(event.evaluate(sample) for sample in samples)
    mi = spn_condition.mutual_information(X > 2, Y > 2)
    assert allclose(mi, log(2))
