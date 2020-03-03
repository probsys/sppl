# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

import tempfile

from spn.distributions import Bernoulli
from spn.distributions import NominalDist
from spn.interpret import Cond
from spn.interpret import Otherwise
from spn.interpret import Start
from spn.interpret import Variable
from spn.render import render_nested_lists
from spn.render import render_nested_lists_concise
from spn.render import render_networkx_graph
from spn.render import render_networkx_image

def test_render_crash():
    Y = Variable('Y')
    X = Variable('X')
    Z = Variable('Z')
    model = (Start
        & Y >> NominalDist({'0': .2, '1': .2, '2': .2, '3': .2, '4': .2})
        & Z >> Bernoulli(p=0.1)
        & Cond (
            Y << {str(0)} | Z << {0},  X >> Bernoulli(p=1/(0+1)),
            Otherwise,                 X >> Bernoulli(p=0.1)))
    render_nested_lists_concise(model)
    render_nested_lists(model)
    G = render_networkx_graph(model)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        render_networkx_image(G, f.name, ext='png')
