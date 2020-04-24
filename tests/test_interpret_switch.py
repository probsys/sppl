# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from math import log

from spn.distributions import bernoulli
from spn.distributions import randint
from spn.interpreter import IfElse
from spn.interpreter import Sample
from spn.interpreter import Start
from spn.interpreter import Switch
from spn.interpreter import Variable
from spn.math_util import allclose
from spn.math_util import logsumexp

Y = Variable('Y')
X = Variable('X')

def test_simple_model():
    model_switch = (Start
        & Sample(Y, randint(low=0, high=4))
        & Switch(Y, range(0, 4), lambda i:
            Sample(X, bernoulli(p=1/(i+1)))))

    model_ifelse = (Start
        & Sample(Y, randint(low=0, high=4))
        & IfElse (
            Y << {0}, Sample(X, bernoulli(p=1/(0+1))),
            Y << {1}, Sample(X, bernoulli(p=1/(1+1))),
            Y << {2}, Sample(X, bernoulli(p=1/(2+1))),
            Y << {3}, Sample(X, bernoulli(p=1/(3+1))),
        ))

    for model in [model_switch, model_ifelse]:
        symbols = model.get_symbols()
        assert symbols == {X, Y}
        assert allclose(
            model.logprob(X << {1}),
            logsumexp([-log(4) - log(i+1) for i in range(4)]))
