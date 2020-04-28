# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

import os

import pytest

from spn.distributions import bernoulli
from spn.interpreter import IfElse
from spn.interpreter import Otherwise
from spn.interpreter import Sample
from spn.interpreter import Sequence
from spn.interpreter import Transform
from spn.interpreter import Id
from spn.render import render_nested_lists
from spn.render import render_nested_lists_concise

def get_model():
    Y = Id('Y')
    X = Id('X')
    Z = Id('Z')
    command = Sequence(
        Sample(Y,     {'0': .2, '1': .2, '2': .2, '3': .2, '4': .2}),
        Sample(Z,     bernoulli(p=0.1)),
        IfElse(
            Y << {str(0)} | Z << {0},  Sample(X, bernoulli(p=1/(0+1))),
            Otherwise,                 Transform(X, Z**2 + Z)))
    return command.interpret()

def test_render_lists_crash():
    model = get_model()
    render_nested_lists_concise(model)
    render_nested_lists(model)

def test_render_graphviz_crash():
    pytest.importorskip('graphviz')
    pytest.importorskip('pygraphviz')
    pytest.importorskip('networkx')

    from spn.magics.render import render_networkx_graph
    from spn.magics.render import render_graphviz

    model = get_model()
    render_networkx_graph(model)
    for fname in [None, '/tmp/spn.test.render']:
        for e in ['pdf', 'png', None]:
            render_graphviz(model, fname, ext=e)
            if fname is not None:
                assert not os.path.exists(fname)
                for ext in ['dot', e]:
                    f = '%s.%s' % (fname, ext,)
                    if e is not None:
                        os.path.exists(f)
                        os.unlink(f)
