# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

import os
import tempfile

import pytest

from spn.distributions import bernoulli
from spn.interpreter import Cond
from spn.interpreter import Otherwise
from spn.interpreter import Sample
from spn.interpreter import Start
from spn.interpreter import Variable
from spn.render import render_graphviz
from spn.render import render_nested_lists
from spn.render import render_nested_lists_concise
from spn.render import render_networkx_graph

def test_render_crash():
    Y = Variable('Y')
    X = Variable('X')
    Z = Variable('Z')
    model = (Start
        & Sample(Y,     {'0': .2, '1': .2, '2': .2, '3': .2, '4': .2})
        & Sample(Z,     bernoulli(p=0.1))
        & Cond (
            Y << {str(0)} | Z << {0},  Sample(X, bernoulli(p=1/(0+1)) ),
            Otherwise,                 Sample(X, bernoulli(p=0.1))))
    render_nested_lists_concise(model)
    render_nested_lists(model)
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
