# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from sppl.compilers.spn_to_sppl import render_sppl
from sppl.compilers.sppl_to_python import SPPL_Compiler
from sppl.math_util import allclose
from sppl.tests.test_render import get_model

def test_render_sppl():
    model = get_model()
    sppl_code = render_sppl(model)
    compiler = SPPL_Compiler(sppl_code.getvalue())
    namespace = compiler.execute_module()
    (X, Y) = (namespace.X, namespace.Y)
    for i in range(5):
        assert allclose(model.logprob(Y << {'0'}), [
            model.logprob(Y << {str(i)}),
            namespace.model.logprob(Y << {str(i)})
        ])
    for i in range(4):
        assert allclose(
            model.logprob(X << {i}),
            namespace.model.logprob(X << {i}))
