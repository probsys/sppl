# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from math import log

import numpy
import scipy.stats

from sum_product_dsl.distributions import MixtureDistribution
from sum_product_dsl.distributions import NumericDistribution
from sum_product_dsl.distributions import ProductDistribution

from sum_product_dsl.math_util import allclose
from sum_product_dsl.sym_util import Reals
from sum_product_dsl.transforms import Identity

rng = numpy.random.RandomState(1)

def test_mutual_information_four_clusters():
    X = Identity('X')
    Y = Identity('Y')
    components = [
        # Component 1.
        ProductDistribution([
            NumericDistribution(X, scipy.stats.norm(0, 0.5), Reals),
            NumericDistribution(Y, scipy.stats.norm(0, 0.5), Reals)]),
        # Component 2.
        ProductDistribution([
            NumericDistribution(X, scipy.stats.norm(5, 0.5), Reals),
            NumericDistribution(Y, scipy.stats.norm(0, 0.5), Reals)]),
        # Component 3.
        ProductDistribution([
            NumericDistribution(X, scipy.stats.norm(0, 0.5), Reals),
            NumericDistribution(Y, scipy.stats.norm(5, 0.5), Reals)]),
        # Component 4.
        ProductDistribution([
            NumericDistribution(X, scipy.stats.norm(5, 0.5), Reals),
            NumericDistribution(Y, scipy.stats.norm(5, 0.5), Reals)]),
    ]
    dist = MixtureDistribution(components, [-log(4)]*4)

    samples = dist.sample(100, rng)
    mi = dist.mutual_information(X > 2, Y > 2)
    assert allclose(mi, 0)

    event = ((X>2) & (Y<2) | ((X<2) & (Y>2)))
    dist_condition = dist.condition(event)
    samples = dist_condition.sample(100, rng)
    assert all(event.evaluate(sample) for sample in samples)
    mi = dist_condition.mutual_information(X > 2, Y > 2)
    assert allclose(mi, log(2))
