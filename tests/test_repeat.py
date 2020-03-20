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

def test_complex_model_reorder():
    model = (Start
    & Y >> NominalDist({'0': .2, '1': .2, '2': .2, '3': .2, '4': .2})
    & Repeat(0, 3, lambda i:
        Z[i] >> Bernoulli(p=0.1))
    & Repeat(0, 3, lambda i:
        Cond (
            Y << {str(i)},
                X[i] >> Bernoulli(p=1/(i+1)),
            Z[i] << {0},
                X[i] >> Bernoulli(p=1/(i+1)),
            Otherwise,
                X[i] >> Bernoulli(p=0.1)
    )))
    assert(allclose(model.prob(Y << {'0'}), 0.2))

def test_repeat_handcode_equivalence():
    model_repeat = make_model_repeat()
    model_hand = make_model_handcode()

    assert allclose(model_repeat.prob(Y << {'0', '1'}), 0.4)
    assert allclose(model_repeat.prob(Z[0] << {0}), 0.5)
    assert allclose(model_repeat.prob(Z[0] << {1}), 0.5)

    event_condition = (X[0] << {1}) | (Y << {'1'})
    model_repeat_condition = model_repeat.condition(event_condition)
    model_hand_condition = model_hand.condition(event_condition)

    for event in [
            Y << {'0','1'},
            Z[0] << {0},
            Z[1] << {0},
            X[0] << {0},
            X[1] << {0},
        ]:
        lp_repeat = model_repeat.logprob(event)
        lp_hand = model_hand.logprob(event)
        assert allclose(lp_hand, lp_repeat)

        lp_repeat_condition = model_repeat_condition.logprob(event)
        lp_hand_condition = model_hand_condition.logprob(event)
        assert allclose(lp_hand_condition, lp_repeat_condition)

# ==============================================================================
# Helper functions.

def make_model_repeat(n=2):
    return (Start
        & Y >> NominalDist({'0': .2, '1': .2, '2': .2, '3': .2, '4': .2})
        & Repeat(0, n, lambda i:
            Z[i] >> Bernoulli(p=.5)
            & Cond (
                (Y << {str(i)}) | (Z[i] << {0}),    X[i] >> Bernoulli(p=.1),
                Otherwise,                          X[i] >> Bernoulli(p=.5))))

def make_model_handcode():
    return (Start
        & Y >> NominalDist({'0': .2, '1': .2, '2': .2, '3': .2, '4': .2})
        & Z[0] >> Bernoulli(p=.5)
        & Z[1] >> Bernoulli(p=.5)
        & Cond (
            Y << {str(0)},
                X[0] >> Bernoulli(p=.1)
                & Cond(
                    Z[1] << {0},    X[1] >> Bernoulli(p=.1),
                    Otherwise,      X[1] >> Bernoulli(p=.5)),
            Y << {str(1)},
                X[1] >> Bernoulli(p=.1)
                & Cond(
                    Z[0] << {0},    X[0] >> Bernoulli(p=.1),
                    Otherwise,      X[0] >> Bernoulli(p=.5)),
            Otherwise,
                Cond(
                    Z[0] << {0},    X[0] >> Bernoulli(p=.1),
                    Otherwise,      X[0] >> Bernoulli(p=.5))
                & Cond(
                    Z[1] << {0},    X[1] >> Bernoulli(p=.1),
                    Otherwise,      X[1] >> Bernoulli(p=.5))))
