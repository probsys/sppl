# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from sppl.distributions import bernoulli
from sppl.distributions import randint
from sppl.compilers.ast_to_spn import IfElse
from sppl.compilers.ast_to_spn import Sample
from sppl.compilers.ast_to_spn import Sequence
from sppl.compilers.ast_to_spn import Transform
from sppl.compilers.ast_to_spn import Id
from sppl.math_util import allclose

Y = Id('Y')
X = Id('X')

def test_ifelse_zero_conditions():
    command = Sequence(
        Sample(Y, randint(low=0, high=3)),
        IfElse(
            Y << {-1},  Transform(X, Y**(-1)),
            Y << {0},   Sample(X, bernoulli(p=1)),
            Y << {1},   Transform(X, Y),
            Y << {2},   Transform(X, Y**2),
            Y << {3},   Transform(X, Y**3),
        ))
    model = command.interpret()
    assert len(model.children) == 3
    assert len(model.weights) == 3
    assert allclose(model.weights[0], model.logprob(Y << {0}))
    assert allclose(model.weights[1], model.logprob(Y << {1}))
    assert allclose(model.weights[2], model.logprob(Y << {2}))
