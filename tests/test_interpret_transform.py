# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from math import log

from spn.distributions import bernoulli
from spn.distributions import norm
from spn.interpreter import IfElse
from spn.interpreter import Otherwise
from spn.interpreter import Sample
from spn.interpreter import Start
from spn.interpreter import Transform
from spn.interpreter import Variable
from spn.math_util import allclose

X = Variable('X')
Y = Variable('Y')
Z = Variable('Z')

def test_simple_transform():
    model = (Start
        & Sample(X, norm(loc=0, scale=1))
        & Transform(Z, X**2))
    assert model.get_symbols() == {Z, X}
    assert model.env == {Z:X**2, X:X}
    assert (model.logprob(Z > 0)) == 0

def test_if_else_transform():
    model = (Start
        & Sample(X, norm(loc=0, scale=1))
        & IfElse(
            X > 0,
                Transform(Z, X**2),
            Otherwise,
                Transform(Z, X)
            ))
    assert model.children[0].env == {X:X, Z:X**2}
    assert model.children[1].env == {X:X, Z:X}
    assert allclose(model.children[0].logprob(Z > 0), 0)
    assert allclose(model.children[1].logprob(Z > 0), -float('inf'))
    assert allclose(model.logprob(Z > 0), -log(2))

def test_if_else_transform_reverse():
    model = (Start
        & Sample(X, norm(loc=0, scale=1))
        & Sample(Y, bernoulli(p=0.5))
        & IfElse(
            Y << {0},
                Transform(Z, X**2),
            Otherwise,
                Transform(Z, X)
            ))
    assert allclose(model.logprob(Z > 0), log(3) - log(4))
