# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

import pytest

from spn.distributions import Atomic
from spn.distributions import Uniform
from spn.math_util import allclose
from spn.transforms import Identity

from spn.spn import ExposedSumSPN
from spn.spn import NominalDistribution

GPA = Identity('GPA')

model_implicit = \
        0.5 * ( # American student
            0.99 * Uniform(GPA, loc=0, scale=4) | \
            0.01 * Atomic(GPA, loc=4)) | \
        0.5 * ( # Indian student
            0.99 * Uniform(GPA, loc=0, scale=10) | \
            0.01 * Atomic(GPA, loc=10))

N = Identity('N')
P = Identity('P')

nationality = NominalDistribution(N, {'India': 0.5, 'USA': 0.5})
perfect = NominalDistribution(P, {'Imperfect': 0.99, 'Perfect': 0.01})
model_exposed = ExposedSumSPN(
    spn_dist=nationality,
    spns={
        # American student.
        'USA': ExposedSumSPN(
            spn_dist=perfect,
            spns={
                'Imperfect'   : Uniform(GPA, loc=0, scale=4),
                'Perfect'     : Atomic(GPA, loc=4),
            }),
        # Indian student.
        'India': ExposedSumSPN(
            spn_dist=perfect,
            spns={
                'Perfect'     : Atomic(GPA, loc=10),
                'Imperfect'   : Uniform(GPA, loc=0, scale=10),
            })},
    )

@pytest.mark.parametrize('model', [model_implicit, model_exposed])
def test_prior(model):
    assert allclose(model.prob(GPA << {10}), 0.5*0.01)
    assert allclose(model.prob(GPA << {4}), 0.5*0.01)
    assert allclose(model.prob(GPA << {5}), 0)
    assert allclose(model.prob(GPA << {1}), 0)

    assert allclose(model.prob((2 < GPA) < 4),
        0.5*0.99*0.5 + 0.5*0.99*0.2)
    assert allclose(model.prob((2 <= GPA) < 4),
        0.5*0.99*0.5 + 0.5*0.99*0.2)
    assert allclose(model.prob((2 < GPA) <= 4),
        0.5*(0.99*0.5 + 0.01) + 0.5*0.99*0.2)
    assert allclose(model.prob((2 < GPA) <= 8),
        0.5*(0.99*0.5 + 0.01) + 0.5*0.99*0.6)
    assert allclose(model.prob((2 < GPA) < 10),
        0.5*(0.99*0.5 + 0.01) + 0.5*0.99*0.8)
    assert allclose(model.prob((2 < GPA) <= 10),
        0.5*(0.99*0.5 + 0.01) + 0.5*(0.99*0.8 + 0.01))

    assert allclose(model.prob(((2 <= GPA) < 4) | (7 < GPA)),
        (0.5*0.99*0.5 + 0.5*0.99*0.2) + (0.5*(0.99*0.3 + 0.01)))

    assert allclose(model.prob(((2 <= GPA) < 4) & (7 < GPA)), 0)

def test_condition():
    model_condition = model_implicit.condition(GPA << {4} | GPA << {10})
    assert len(model_condition.children) == 2
    assert model_condition.children[0].support == {4}
    assert model_condition.children[1].support == {10}

    model_condition = model_implicit.condition((0 < GPA < 4))
    assert len(model_condition.children) == 2
    assert model_condition.children[0].support \
        == model_condition.children[1].support
    assert allclose(
        model_condition.children[0].logprob(GPA < 1),
        model_condition.children[1].logprob(GPA < 1))
