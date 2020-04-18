# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

import tempfile

import pytest

from spn.distributions import Bernoulli
from spn.interpret import Cond
from spn.interpret import Otherwise
from spn.interpret import Start
from spn.interpret import Variable
from spn.render import render_graphviz
from spn.render import render_nested_lists
from spn.render import render_nested_lists_concise
from spn.render import render_networkx_graph

def test_render_spn_crash():
    Y = Variable('Y')
    X = Variable('X')
    Z = Variable('Z')
    model = (Start
        & Y >> {'0': .2, '1': .2, '2': .2, '3': .2, '4': .2}
        & Z >> Bernoulli(p=0.1)
        & Cond (
            Y << {str(0)} | Z << {0},  X >> Bernoulli(p=1/(0+1)),
            Otherwise,                 X >> Bernoulli(p=0.1)))
    render_nested_lists_concise(model)
    render_nested_lists(model)
    render_networkx_graph(model)
    with pytest.raises(Exception):
        render_graphviz(model, 'foo')
    with tempfile.NamedTemporaryFile(delete=False) as f:
        render_graphviz(model, '%s.png' % (f.name,))

def test_render_command_crash():
    from spn.distributions import Atomic
    from spn.interpret import Repeat
    from spn.interpret import VariableArray
    from spn.render import render_imp_command
    Y = Variable('Y')
    X = VariableArray('X', 5)
    Z = Variable('Z')
    W = VariableArray('W', 5)
    model \
        = Y >> Atomic(loc=1) \
        & Z >> Bernoulli(p=.1) \
        & X[0] >> {'1': .5, '2': .5} \
        & X[1] >> Z**2 \
        & Cond(
            X[2] > 0,
                X[3] >> Bernoulli(p=.1)
                & Cond(
                    X[3] << {0}, X[4] >> Bernoulli(p=.7),
                    Otherwise,   X[4] >> Bernoulli(p=.1)),
            X[2] < -1,
                X[3] >> Bernoulli(p=.2)
                & X[4] >> X[3]+1,
            Otherwise,
                X[3] >> Bernoulli(p=.3)
                & X[4] >> X[3]+5
                & Repeat(0, 5, lambda i: (
                    W[i] >> Bernoulli(p=1/(i+1))
                    & X[i] >> Atomic(loc=i))
            ))
    print(render_imp_command(model))
