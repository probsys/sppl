# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from math import log

from spn.distributions import Bernoulli
from spn.distributions import Norm
from spn.interpret import Cond
from spn.interpret import Otherwise
from spn.interpret import Start
from spn.interpret import Transform
from spn.interpret import Variable
from spn.math_util import allclose

X = Variable('X')
Y = Variable('Y')
Z = Variable('Z')

def test_simple_transform():
    model = (Start
        & X >> Norm(loc=0, scale=1)
        & Z >> X**2)
    assert model.get_symbols() == {Z, X}
    assert model.env == {Z:X**2, X:X}
    assert (model.logprob(Z > 0)) == 0

def test_if_else_transform():
    model = (Start
        & X >> Norm(loc=0, scale=1)
        & Cond (
            X > 0,
                Z >> X**2,
            Otherwise,
                Z >> X
            ))
    assert model.children[0].env == {X:X, Z:X**2}
    assert model.children[1].env == {X:X, Z:X}
    assert allclose(model.children[0].logprob(Z > 0), 0)
    assert allclose(model.children[1].logprob(Z > 0), -float('inf'))
    assert allclose(model.logprob(Z > 0), -log(2))

def test_if_else_transform_reverse():
    model = (Start
        & X >> Norm(loc=0, scale=1)
        & Y >> Bernoulli(p=0.5)
        & Cond (
            Y << {0},
                Z >> X**2,
            Otherwise,
                Z >> X,
            ))
    assert allclose(model.logprob(Z > 0), log(3) - log(4))
