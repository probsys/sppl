# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from math import log

from spn.distributions import bernoulli
from spn.distributions import randint
from spn.interpreter import For
from spn.interpreter import IfElse
from spn.interpreter import Otherwise
from spn.interpreter import Sample
from spn.interpreter import Transform
from spn.interpreter import Start
from spn.interpreter import Variable
from spn.math_util import allclose

Y = Variable('Y')
X = Variable('X')

def test_ifelse_zero_conditions():
    model = (Start
        & Sample(Y, randint(low=0, high=3))
        & IfElse (
            Y << {-1},  Transform(X, Y**(-1)),
            Y << {0},   Sample(X, bernoulli(p=1)),
            Y << {1},   Transform(X, Y),
            Y << {2},   Transform(X, Y**2),
            Y << {3},   Transform(X, Y**3),
        ))
    assert len(model.children) == 3
    assert len(model.weights) == 3
    assert allclose(model.weights[0], model.logprob(Y << {0}))
    assert allclose(model.weights[1], model.logprob(Y << {1}))
    assert allclose(model.weights[2], model.logprob(Y << {2}))
