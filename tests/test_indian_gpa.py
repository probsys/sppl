# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

'''
Indian GPA example from:

Discrete-Continuous Mixtures in Probabilistic Programming: Generalized
Semantics and Inference Algorithms, Wu et. al., ICML 2018.
https://arxiv.org/pdf/1806.02027.pdf
'''

import pytest

from spn.combinators import IfElse
from spn.distributions import Atomic
from spn.distributions import NominalDist
from spn.distributions import Uniform
from spn.interpret import Cond
from spn.interpret import Variable
from spn.math_util import allclose
from spn.spn import ExposedSumSPN
from spn.transforms import Identity

def model_no_latents():
    GPA = Identity('GPA')
    return \
        0.5 * ( # American student
            0.99 * (GPA >> Uniform(loc=0, scale=4)) | \
            0.01 * (GPA >> Atomic(loc=4))) | \
        0.5 * ( # Indian student
            0.99 * (GPA >> Uniform(loc=0, scale=10)) | \
            0.01 * (GPA >> Atomic(loc=10)))

def model_exposed():
    N = Identity('N')
    P = Identity('P')
    GPA = Identity('GPA')
    nationality = N >> NominalDist({'India': 0.5, 'USA': 0.5})
    perfect = P >> NominalDist({'Imperfect': 0.99, 'Perfect': 0.01})
    return ExposedSumSPN(
        spn_weights=nationality,
        children={
            # American student.
            'USA': ExposedSumSPN(
                spn_weights=perfect,
                children={
                    'Imperfect'   : GPA >> Uniform(loc=0, scale=4),
                    'Perfect'     : GPA >> Atomic(loc=4),
                }),
            # Indian student.
            'India': ExposedSumSPN(
                spn_weights=perfect,
                children={
                    'Perfect'     : GPA >> Atomic(loc=10),
                    'Imperfect'   : GPA >> Uniform(loc=0, scale=10),
                })},
        )

def model_ifelse_exhuastive():
    N = Identity('N')
    P = Identity('P')
    GPA = Identity('GPA')
    nationality = N >> NominalDist({'India': 0.5, 'USA': 0.5})
    perfect = P >> NominalDist({'Imperfect': 0.99, 'Perfect': 0.01})
    return IfElse(nationality & perfect,
    [(N << {'India'}) & (P << {'Imperfect'}),
        GPA >> Uniform(loc=0, scale=10)
    ],
    [(N << {'India'}) & (P << {'Perfect'}),
        GPA >> Atomic(loc=10)
    ],
    [(N << {'USA'}) & (P << {'Imperfect'}),
        GPA >> Uniform(loc=0, scale=4)
    ],
    [(N << {'USA'}) & (P << {'Perfect'}),
        GPA >> Atomic(loc=4)
    ])

def model_ifelse_nested():
    N = Identity('N')
    P = Identity('P')
    GPA = Identity('GPA')
    nationality = N >> NominalDist({'India': 0.5, 'USA': 0.5})
    perfect = P >> NominalDist({'Imperfect': 0.99, 'Perfect': 0.01})
    return IfElse(nationality,
    [(N << {'India'}),
        IfElse(perfect,
            [(P << {'Imperfect'}), GPA >> Uniform(loc=0, scale=10)],
            [True, GPA >> Atomic(loc=10)])
    ],
    [True,
        IfElse(perfect,
            [(P << {'Imperfect'}), GPA >> Uniform(loc=0, scale=4)],
            [True, GPA >> Atomic(loc=4)])
    ])

def model_ifelse_non_exhuastive():
    N = Identity('N')
    P = Identity('P')
    GPA = Identity('GPA')
    nationality = N >> NominalDist({'India': 0.5, 'USA': 0.5})
    perfect = P >> NominalDist({'Imperfect': 0.99, 'Perfect': 0.01})
    return IfElse(nationality & perfect,
        [(N << {'India'}) & (P << {'Imperfect'}),
            GPA >> Uniform(loc=0, scale=10)
        ],
        [(N << {'India'}) & (P << {'Perfect'}),
            GPA >> Atomic(loc=10)
        ],
        [(N << {'USA'}) & (P << {'Imperfect'}),
            GPA >> Uniform(loc=0, scale=4)
        ],
        [True, # Can be slow; complementing Nominals over unknown support.
            GPA >> Atomic(loc=4)
        ])

def model_interpreter():
    # Declare variables in the model.
    Nationality = Variable('Nationality')
    Perfect     = Variable('Perfect')
    GPA         = Variable('GPA')

    # Write the generative model in embedded Python.
    return None \
        & Nationality   >> NominalDist({'Indian': 0.5, 'USA': 0.5}) \
        & Cond (
            Nationality << {'Indian'},
                Perfect >> NominalDist({'True': 0.01, 'False': 0.99}) \
                & Cond(
                    Perfect << {'True'},    GPA >> Atomic(loc=10),
                    Perfect << {'False'},   GPA >> Uniform(scale=10),
                ),
            Nationality << {'USA'},
                Perfect >> NominalDist({'True': 0.01, 'False': 0.99}) \
                & Cond (
                    Perfect << {'True'},    GPA >> Atomic(loc=4),
                    Perfect << {'False'},   GPA >> Uniform(scale=4),
                ))

@pytest.mark.parametrize('get_model', [
    model_no_latents,
    model_exposed,
    model_ifelse_exhuastive,
    model_ifelse_nested,
    model_interpreter,
])
def test_prior(get_model):
    model = get_model()
    GPA = Identity('GPA')
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
    model = model_no_latents()
    GPA = Identity('GPA')
    model_condition = model.condition(GPA << {4} | GPA << {10})
    assert len(model_condition.children) == 2
    assert model_condition.children[0].support == {4}
    assert model_condition.children[1].support == {10}

    model_condition = model.condition((0 < GPA < 4))
    assert len(model_condition.children) == 2
    assert model_condition.children[0].support \
        == model_condition.children[1].support
    assert allclose(
        model_condition.children[0].logprob(GPA < 1),
        model_condition.children[1].logprob(GPA < 1))
