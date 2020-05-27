# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from spn.compilers.spml_to_python import SPML_Compiler
from spn.compilers.spn_to_spml import render_spml
from spn.math_util import allclose
from spn.tests.test_render import get_model

def test_render_spml():
    model = get_model()
    spml_code = render_spml(model)
    compiler = SPML_Compiler(spml_code.getvalue())
    namespace = compiler.execute_module()
    (X, Y) = (namespace.X, namespace.Y)
    for i in range(5):
        assert allclose(model.logprob(Y << {'0'}), [
            model.logprob(Y << {str(i)}),
            namespace.model.logprob(Y << {str(i)})
        ])
    assert allclose(
        model.logprob(X << {0}),
        namespace.model.logprob(X << {0}))
