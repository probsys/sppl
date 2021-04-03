# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from fractions import Fraction
from math import log

import numpy
import pytest

from sppl.distributions import choice
from sppl.math_util import allclose
from sppl.math_util import isinf_neg
from sppl.sets import FiniteNominal
from sppl.transforms import Id

def test_nominal_distribution():
    X = Id('X')
    spe = X >> choice({
        'a': Fraction(1, 5),
        'b': Fraction(1, 5),
        'c': Fraction(3, 5),
    })
    assert allclose(spe.logprob(X << {'a'}), log(Fraction(1, 5)))
    assert allclose(spe.logprob(X << {'b'}), log(Fraction(1, 5)))
    assert allclose(spe.logprob(X << {'a', 'c'}), log(Fraction(4, 5)))
    assert allclose(
        spe.logprob((X << {'a'}) & ~(X << {'b'})),
        log(Fraction(1, 5)))
    assert allclose(
        spe.logprob((X << {'a', 'b'}) & ~(X << {'b'})),
        log(Fraction(1, 5)))
    assert spe.logprob((X << {'d'})) == -float('inf')
    assert spe.logprob((X << ())) == -float('inf')

    samples = spe.sample(100)
    assert all(s[X] in spe.support for s in samples)

    samples = spe.sample_subset([X], 100)
    assert all(len(s)==1 and s[X] in spe.support for s in samples)

    with pytest.raises(Exception):
        spe.sample_subset(['f'], 100)

    predicate = lambda X: (X in {'a', 'b'}) or X in {'c'}
    samples = spe.sample_func(predicate, 100)
    assert all(samples)

    predicate = lambda X: (not (X in {'a', 'b'})) and (not (X in {'c'}))
    samples = spe.sample_func(predicate, 100)
    assert not any(samples)

    func = lambda X: 1 if X in {'a'} else None
    samples = spe.sample_func(func, 100, prng=numpy.random.RandomState(1))
    assert sum(1 for s in samples if s == 1) > 12
    assert sum(1 for s in samples if s is None) > 70

    with pytest.raises(ValueError):
        spe.sample_func(lambda Y: Y, 100)

    spe_condition = spe.condition(X<<{'a', 'b'})
    assert spe_condition.support == FiniteNominal('a', 'b', 'c')
    assert allclose(spe_condition.logprob(X << {'a'}), -log(2))
    assert allclose(spe_condition.logprob(X << {'b'}), -log(2))
    assert spe_condition.logprob(X << {'c'}) == -float('inf')

    assert isinf_neg(spe_condition.logprob(X**2 << {1}))

    with pytest.raises(ValueError):
        spe.condition(X << {'python'})
    assert spe.condition(~(X << {'python'})) == spe
