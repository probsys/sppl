# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from math import log

import numpy

from sum_product_dsl.distributions import ProductDistribution
from sum_product_dsl.distributions import SumDistribution
from sum_product_dsl.math_util import allclose
from sum_product_dsl.numerical import Norm
from sum_product_dsl.transforms import Identity

rng = numpy.random.RandomState(1)

def test_mutual_information_four_clusters():
    X = Identity('X')
    Y = Identity('Y')
    components = [
        # Component 1.
        ProductDistribution([
            Norm(X, loc=0, scale=0.5),
            Norm(Y, loc=0, scale=0.5)]),
        # Component 2.
        ProductDistribution([
            Norm(X, loc=5, scale=0.5),
            Norm(Y, loc=0, scale=0.5)]),
        # Component 3.
        ProductDistribution([
            Norm(X, loc=0, scale=0.5),
            Norm(Y, loc=5, scale=0.5)]),
        # Component 4.
        ProductDistribution([
            Norm(X, loc=5, scale=0.5),
            Norm(Y, loc=5, scale=0.5)]),
    ]
    dist = SumDistribution(components, [-log(4)]*4)

    samples = dist.sample(100, rng)
    mi = dist.mutual_information(X > 2, Y > 2)
    assert allclose(mi, 0)

    event = ((X>2) & (Y<2) | ((X<2) & (Y>2)))
    dist_condition = dist.condition(event)
    samples = dist_condition.sample(100, rng)
    assert all(event.evaluate(sample) for sample in samples)
    mi = dist_condition.mutual_information(X > 2, Y > 2)
    assert allclose(mi, log(2))
