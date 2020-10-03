# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

import pytest

from sppl.transforms import Exp
from sppl.transforms import Id
from sppl.transforms import Log

X = Id('X')
Y = Id('Y')
W = Id('W')
Z = Id('Z')

def test_event_basic_invertible():
    expr = X**2 + 10*X

    event = (0 < expr)
    assert event.evaluate({X: 10})
    assert not (~event).evaluate({X: 10})

    event = (0 > expr)
    assert not event.evaluate({X: 10})
    assert (~event).evaluate({X: 10})

    event = (0 < expr) < 100
    for val in [0, 10]:
        assert not event.evaluate({X: val})
        assert (~event).evaluate({X: val})

    event = (0 <= expr) <= 100
    x_eq_100 = (expr << {100}).solve()
    for val in [0, list(x_eq_100)[0]]:
        assert event.evaluate({X: val})
        assert not (~event).evaluate({X: val})

    event = expr << {11}
    assert event.evaluate({X: 1})
    assert not event.evaluate({X: 3})
    assert not (~event).evaluate({X: 1})
    assert (~event).evaluate({X :3})

    with pytest.raises(ValueError):
        event.evaluate({Y: 10})

def test_event_compound():
    expr0 = abs(X)**2 + 10*abs(X)
    expr1 = Y-1

    event = (0 < expr0) | (0 < expr1)
    assert not event.evaluate({X: 0, Y: 0})
    for i in range(1, 100):
        assert event.evaluate({X: i, Y: i})
    with pytest.raises(ValueError):
        event.evaluate({X: 0})
    with pytest.raises(ValueError):
        event.evaluate({Y: 0})

    event = (100 <= expr0) & ((expr0 < 200) | (3 < expr1))
    assert not event.evaluate({X: 0, Y: 0})

    x_eq_150 = (expr0 << {150}).solve()
    assert event.evaluate({X: list(x_eq_150)[0], Y: 0})
    assert event.evaluate({X: list(x_eq_150)[1], Y: 0})

    x_eq_500 = (expr0 << {500}).solve()
    assert not event.evaluate({X: list(x_eq_500)[1], Y: 4})
    assert event.evaluate({X: list(x_eq_500)[1], Y: 5})

def test_event_solve_multi():
    event = (Exp(abs(3*X**2)) > 1) | (Log(Y) < 0.5)
    with pytest.raises(ValueError):
        event.solve()
    event = (Exp(abs(3*X**2)) > 1) & (Log(Y) < 0.5)
    with pytest.raises(ValueError):
        event.solve()
