# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from math import log

from sppl.compilers.ast_to_spn import Id
from sppl.compilers.ast_to_spn import IfElse
from sppl.compilers.ast_to_spn import Otherwise
from sppl.compilers.ast_to_spn import Sample
from sppl.compilers.ast_to_spn import Sequence
from sppl.compilers.ast_to_spn import Transform
from sppl.distributions import bernoulli
from sppl.distributions import norm
from sppl.math_util import allclose

X = Id('X')
Y = Id('Y')
Z = Id('Z')

def test_simple_transform():
    command = Sequence(
        Sample(X, norm(loc=0, scale=1)),
        Transform(Z, X**2))
    model = command.interpret()
    assert model.get_symbols() == {Z, X}
    assert model.env == {Z:X**2, X:X}
    assert (model.logprob(Z > 0)) == 0

def test_if_else_transform():
    model = Sequence(
        Sample(X, norm(loc=0, scale=1)),
        IfElse(
            X > 0,      Transform(Z, X**2),
            Otherwise,  Transform(Z, X))).interpret()
    assert model.children[0].env == {X:X, Z:X**2}
    assert model.children[1].env == {X:X, Z:X}
    assert allclose(model.children[0].logprob(Z > 0), 0)
    assert allclose(model.children[1].logprob(Z > 0), -float('inf'))
    assert allclose(model.logprob(Z > 0), -log(2))

def test_if_else_transform_reverse():
    command = Sequence(
        Sample(X, norm(loc=0, scale=1)),
        Sample(Y, bernoulli(p=0.5)),
        IfElse(
            Y << {0},  Transform(Z, X**2),
            Otherwise, Transform(Z, X)))
    model = command.interpret()
    assert allclose(model.logprob(Z > 0), log(3) - log(4))
