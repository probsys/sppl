# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from math import log

from spn.distributions import Bernoulli
from spn.distributions import NominalDist
from spn.interpret import Cond
from spn.interpret import Otherwise
from spn.interpret import Repeat
from spn.interpret import Start
from spn.interpret import Variable
from spn.interpret import VariableArray
from spn.math_util import allclose

Y = Variable('Y')
X = VariableArray('X', 5)
Z = VariableArray('Z', 5)

def test_simple_model():
    model = (Start
        & Y >> Bernoulli(p=0.5)
        & Repeat(0, 5, lambda i:
            X[i] >> Bernoulli(p=1/(i+1))))

    symbols = model.get_symbols()
    assert len(symbols) == 6
    assert Y in symbols
    assert X[0] in symbols
    assert X[1] in symbols
    assert X[2] in symbols
    assert X[3] in symbols
    assert X[4] in symbols
    assert model.logprob(X[0] << {1}) == log(1/1)
    assert model.logprob(X[1] << {1}) == log(1/2)
    assert model.logprob(X[2] << {1}) == log(1/3)
    assert model.logprob(X[3] << {1}) == log(1/4)
    assert model.logprob(X[4] << {1}) == log(1/5)

def test_complex_model():
    # Slow for larger number of repetitions
    # https://github.com/probcomp/sum-product-dsl/issues/43
    model = (Start
    & Y >> NominalDist({'0': .2, '1': .2, '2': .2, '3': .2, '4': .2})
    & Repeat(0, 3, lambda i:
        Z[i] >> Bernoulli(p=0.1)
        & Cond (
            Y << {str(i)} | Z[i] << {0},  X[i] >> Bernoulli(p=1/(i+1)),
            Otherwise,                    X[i] >> Bernoulli(p=0.1))))
    assert allclose(model.prob(Y << {'0'}), 0.2)
