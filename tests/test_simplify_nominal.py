# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

import pytest

from spn.distributions import simplify_nominal_event
from spn.transforms import Identity

X = Identity('X')

def test_simplify_nominal_event():
    support = frozenset(range(100))
    snf = simplify_nominal_event

    # Eq.
    assert snf(X << {5}, support) == {5}
    assert snf(X << {-5}, support) == set()

    V = (4, 1, 10)
    W = (12, -1, 8)
    E1 = V
    E2 = (-5,) + V
    E3 = (-5,)
    E4 = W

    # No complement.
    assert snf(X << E1, support) == set(V)
    assert snf(X << E2, support) == set(V)
    assert snf(X << E3, support) == set()

    # NotContains.
    assert snf(~(X << E1), support) == support.difference(V)
    assert snf(~(X << E2), support) == support.difference(V)
    assert snf(~(X << E3), support) == support

    # And
    event = (X << E1) & ~(X << E4)
    solution = set(V).intersection(support.difference(W))
    assert snf(event, support) == solution

    # Or
    event = (X << E1) | ~(X << E4)
    solution = set(V).union(support.difference(W))
    assert snf(event, support) == solution

def test_symplify_nominal_error():
    support = frozenset(range(100))
    with pytest.raises(ValueError):
        simplify_nominal_event(X**2 << [10], support)
    with pytest.raises(ValueError):
        simplify_nominal_event((X << [10]) & (X**2 << [10]), support)
    with pytest.raises(ValueError):
        simplify_nominal_event((2**X << [10]) & (X**2 << [10]), support)
