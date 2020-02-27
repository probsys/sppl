# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

'''
Burglary network example from:

Artificial Intelligence: A Modern Approach (3rd Edition).
Russel and Norvig, Fig 14.2 pp 512.
'''

import pytest

from spn.combinators import IfElse
from spn.distributions import Bernoulli
from spn.distributions import NominalDist
from spn.distributions import Uniform
from spn.interpret import Cond
from spn.interpret import Variable
from spn.math_util import allclose
from spn.spn import ExposedSumSPN
from spn.transforms import Identity

Burglary = Variable('B')
Earthquake = Variable('E')
Alarm = Variable('A')
JohnCalls = Variable('JC')
MaryCalls = Variable('MC')

Start = None
Otherwise = True

model = (Start
    & Burglary >> Bernoulli(p=0.001)
    & Earthquake >> Bernoulli(p=0.002)
    & Cond (
        Burglary << {1},
            Cond (
                Earthquake << {1},  Alarm >> Bernoulli(p=0.95),
                Otherwise,          Alarm >> Bernoulli(p=0.94)),
        Otherwise,
            Cond (
                Earthquake << {1},  Alarm >> Bernoulli(p=0.29),
                Otherwise,          Alarm >> Bernoulli(p=0.001)))
    & Cond (
        Alarm << {1},
            JohnCalls >> Bernoulli(p=0.90)
            & MaryCalls >> Bernoulli(p=0.70),
        Otherwise,
            JohnCalls >> Bernoulli(p=0.05)
            & MaryCalls >> Bernoulli(p=0.01),
        ))

def test_marginal_probability():
    # Query on pp. 514.
    event = ((JohnCalls << {1}) & (MaryCalls << {1}) & (Alarm << {1})
        & (Burglary << {0}) & (Earthquake << {0}))
    x = model.prob(event)
    assert str(x)[:8] == '0.000628'

def test_conditional_probability():
    # Query on pp. 523
    event = (JohnCalls << {1}) & (MaryCalls << {1})
    model_condition = model.condition(event)
    x = model_condition.prob(Burglary << {1})
    assert str(x)[:5] == '0.284'
