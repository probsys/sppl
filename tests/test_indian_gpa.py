# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from spn.math_util import allclose
from spn.distributions import Uniform
from spn.distributions import Atomic
from spn.transforms import Identity

X = Identity('X')

model = 0.5 * ( # American student
            0.99 * Uniform(X, loc=0, scale=4) | \
            0.01 * Atomic(X, loc=4)) | \
        0.5 * ( # Indian student
            0.99 * Uniform(X, loc=0, scale=10) | \
            0.01 * Atomic(X, loc=10))

def test_prior():
    assert allclose(model.prob(X << {10}), 0.5*0.01)
    assert allclose(model.prob(X << {4}), 0.5*0.01)
    assert allclose(model.prob(X << {5}), 0)
    assert allclose(model.prob(X << {1}), 0)

    assert allclose(model.prob((2 < X) < 4),
        0.5*0.99*0.5 + 0.5*0.99*0.2)
    assert allclose(model.prob((2 <= X) < 4),
        0.5*0.99*0.5 + 0.5*0.99*0.2)
    assert allclose(model.prob((2 < X) <= 4),
        0.5*(0.99*0.5 + 0.01) + 0.5*0.99*0.2)
    assert allclose(model.prob((2 < X) <= 8),
        0.5*(0.99*0.5 + 0.01) + 0.5*0.99*0.6)
    assert allclose(model.prob((2 < X) < 10),
        0.5*(0.99*0.5 + 0.01) + 0.5*0.99*0.8)
    assert allclose(model.prob((2 < X) <= 10),
        0.5*(0.99*0.5 + 0.01) + 0.5*(0.99*0.8 + 0.01))

    assert allclose(model.prob(((2 <= X) < 4) | (7 < X)),
        (0.5*0.99*0.5 + 0.5*0.99*0.2) + (0.5*(0.99*0.3 + 0.01)))

    assert allclose(model.prob(((2 <= X) < 4) & (7 < X)), 0)

def test_condition():
    model_condition = model.condition(X << {4} | X << {10})
    assert len(model_condition.children) == 2
    assert model_condition.children[0].support == {4}
    assert model_condition.children[1].support == {10}

    model_condition = model.condition((0 < X < 4))
    assert len(model_condition.children) == 2
    assert model_condition.children[0].support \
        == model_condition.children[1].support
    assert allclose(
        model_condition.children[0].logprob(X < 1),
        model_condition.children[1].logprob(X < 1))
