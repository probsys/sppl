# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from fractions import Fraction

import sympy
import numpy

from sum_product_dsl.distributions import NominalDistribution
from sum_product_dsl.transforms import Identity

rng = numpy.random.RandomState(1)

def test_nominal_distribution():
    X = Identity('X')
    probs = {'a': Fraction(1, 5), 'b': Fraction(1, 5), 'c': Fraction(3, 5)}
    dist = NominalDistribution(X, probs)
    assert dist.logprob(X << {'a'}) == sympy.log(Fraction(1, 5))
    assert dist.logprob(X << {'b'}) == sympy.log(Fraction(1, 5))
    assert dist.logprob(X << {'a', 'c'}) == sympy.log(Fraction(4, 5))
    assert dist.logprob((X << {'a'}) & ~(X << {'b'})) == sympy.log(Fraction(1, 5))
    assert dist.logprob((X << {'a', 'b'}) & ~(X << {'b'})) == sympy.log(Fraction(1, 5))
    assert dist.logprob((X << {'d'})) == -float('inf')
    assert dist.logprob((X << ())) == -float('inf')

    samples = dist.sample(100, rng)
    assert all(x in dist.support for x in samples)

    predicate = (X << {'a', 'b'}) | X << {'c'}
    samples = dist.sample_expr(predicate, 100, rng)
    assert all(samples)

    predicate = (~(X << {'a', 'b'})) & ~(X << {'c'})
    samples = dist.sample_expr(predicate, 100, rng)
    assert not any(samples)

    dist_condition = dist.condition(X<<{'a', 'b'})
    assert dist_condition.support == {'a', 'b', 'c'}
    assert dist_condition.logprob(X << {'a'}) \
        == dist_condition.logprob(X << {'b'}) \
        == -sympy.log(2)
    assert dist_condition.logprob(X << {'c'}) == -float('inf')
