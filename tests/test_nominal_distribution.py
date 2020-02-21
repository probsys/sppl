# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from fractions import Fraction
from math import log

import pytest
import numpy

from spn.math_util import allclose
from spn.spn import NominalDistribution
from spn.sym_util import NominalSet
from spn.transforms import Identity

rng = numpy.random.RandomState(1)

def test_nominal_distribution():
    X = Identity('X')
    probs = {'a': Fraction(1, 5), 'b': Fraction(1, 5), 'c': Fraction(3, 5)}
    spn = NominalDistribution(X, probs)
    assert allclose(spn.logprob(X << {'a'}), log(Fraction(1, 5)))
    assert allclose(spn.logprob(X << {'b'}), log(Fraction(1, 5)))
    assert allclose(spn.logprob(X << {'a', 'c'}), log(Fraction(4, 5)))
    assert allclose(
        spn.logprob((X << {'a'}) & ~(X << {'b'})),
        log(Fraction(1, 5)))
    assert allclose(
        spn.logprob((X << {'a', 'b'}) & ~(X << {'b'})),
        log(Fraction(1, 5)))
    assert spn.logprob((X << {'d'})) == -float('inf')
    assert spn.logprob((X << ())) == -float('inf')

    samples = spn.sample(100, rng)
    assert all(s[X] in spn.support for s in samples)

    samples = spn.sample_subset([X, 'A'], 100, rng)
    assert all(s[X] in spn.support for s in samples)

    assert spn.sample_subset(['f'], 100, rng) is None

    predicate = lambda X: (X in {'a', 'b'}) or X in {'c'}
    samples = spn.sample_func(predicate, 100, rng)
    assert all(samples)

    predicate = lambda X: (not (X in {'a', 'b'})) and (not (X in {'c'}))
    samples = spn.sample_func(predicate, 100, rng)
    assert not any(samples)

    func = lambda X: 1 if X in {'a'} else None
    samples = spn.sample_func(func, 100, rng)
    assert sum(1 for s in samples if s == 1) > 12
    assert sum(1 for s in samples if s is None) > 70

    with pytest.raises(ValueError):
        spn.sample_func(lambda Y: Y, 100, rng)

    spn_condition = spn.condition(X<<{'a', 'b'})
    assert spn_condition.support == NominalSet('a', 'b', 'c')
    assert allclose(spn_condition.logprob(X << {'a'}), -log(2))
    assert allclose(spn_condition.logprob(X << {'b'}), -log(2))
    assert spn_condition.logprob(X << {'c'}) == -float('inf')

    with pytest.raises(ValueError):
        spn_condition.logprob(X**2 << {1})

    with pytest.raises(ValueError):
        spn.condition(X << {'python'})
    assert spn.condition(~(X << {'python'})) == spn
