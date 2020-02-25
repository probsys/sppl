# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

import pytest

from spn.distributions import Atomic
from spn.distributions import Uniform
from spn.distributions import NominalDist
from spn.math_util import allclose
from spn.transforms import Identity

from spn.combinators import IfElse
from spn.spn import ExposedSumSPN

N = Identity('N')
P = Identity('P')
GPA = Identity('GPA')

model_no_latents = \
        0.5 * ( # American student
            0.99 * (GPA >> Uniform(loc=0, scale=4)) | \
            0.01 * (GPA >> Atomic(loc=4))) | \
        0.5 * ( # Indian student
            0.99 * (GPA >> Uniform(loc=0, scale=10)) | \
            0.01 * (GPA >> Atomic(loc=10)))

nationality = N >> NominalDist({'India': 0.5, 'USA': 0.5})
perfect = P >> NominalDist({'Imperfect': 0.99, 'Perfect': 0.01})
model_exposed = ExposedSumSPN(
    spn_dist=nationality,
    spns={
        # American student.
        'USA': ExposedSumSPN(
            spn_dist=perfect,
            spns={
                'Imperfect'   : GPA >> Uniform(loc=0, scale=4),
                'Perfect'     : GPA >> Atomic(loc=4),
            }),
        # Indian student.
        'India': ExposedSumSPN(
            spn_dist=perfect,
            spns={
                'Perfect'     : GPA >> Atomic(loc=10),
                'Imperfect'   : GPA >> Uniform(loc=0, scale=10),
            })},
    )

model_ifelse_exhuastive = IfElse(nationality & perfect,
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

model_ifelse_nested = IfElse(nationality,
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

# Known issue #1
# The non-exhaustive model, that uses True for the final statement,
# is slow, because the compiled event is very complicated.
# This may be due to the fact that we do not know how to simplify
# complements of Nominal variables well, in absence of the explicit
# universe over which the variables take their values.  One solution
# is to store the possible nominal values in a global environment
# when compiling the program, and allow EventFiniteSet to take in
# the finite support of the nominal variate.
model_ifelse_non_exhuastive = IfElse(nationality & perfect,
    [(N << {'India'}) & (P << {'Imperfect'}),
        GPA >> Uniform(loc=0, scale=10)
    ],
    [(N << {'India'}) & (P << {'Perfect'}),
        GPA >> Atomic(loc=10)
    ],
    [(N << {'USA'}) & (P << {'Imperfect'}),
        GPA >> Uniform(loc=0, scale=4)
    ],
    [True,
        GPA >> Atomic(loc=4)
    ])

# Known issue #2
# The nested model with repeated variables raises an error,
# because IfElse takes the product of the current SPN with the
# child branch.  Instead we need to implement the IfElse combinators
# as specified in the SOS semantics from README, which finds the child
# SPN recursively from the sub-branch by applying the sequence rule
# to the root SPN with the branch SPNs.  This implementation will need
# an interpreter of the syntax and may not be possible to implement
# as vanilla Python.
#
# model_ifelse_nested_repeat = IfElse(nationality & perfect,
#     [(N << {'India'}),
#         IfElse(nationality & perfect,
#             [(P << {'Imperfect'}), GPA >> Uniform(loc=0, scale=10)],
#             [True, GPA >> Atomic(loc=10)])
#     ],
#     [True,
#         IfElse(nationality & perfect,
#             [(P << {'Imperfect'}), GPA >> Uniform(loc=0, scale=4)],
#             [True, GPA >> Atomic(loc=4)])
#     ])

@pytest.mark.parametrize('model', [model_no_latents,
    model_exposed,
    model_ifelse_exhuastive,
    model_ifelse_nested])
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
    model_condition = model_no_latents.condition(GPA << {4} | GPA << {10})
    assert len(model_condition.children) == 2
    assert model_condition.children[0].support == {4}
    assert model_condition.children[1].support == {10}

    model_condition = model_no_latents.condition((0 < GPA < 4))
    assert len(model_condition.children) == 2
    assert model_condition.children[0].support \
        == model_condition.children[1].support
    assert allclose(
        model_condition.children[0].logprob(GPA < 1),
        model_condition.children[1].logprob(GPA < 1))
