# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from spn.distributions import Atomic
from spn.distributions import NominalDist
from spn.distributions import Uniform
from spn.interpret import Cond
from spn.interpret import Variable
from spn.math_util import allclose

def test_interpret_indian_gpa():
    # Declare variables in the model.
    Nationality = Variable('Nationality')
    Perfect     = Variable('Perfect')
    GPA         = Variable('GPA')

    # Write the generative model in embedded Python.
    model = None \
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

    # Condition the model on event "GPA is < 5 or "nationality is Indian".
    model.condition((GPA < 5) | (Nationality << {'India'}))
