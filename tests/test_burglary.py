# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

'''
Burglary network example from:

Artificial Intelligence: A Modern Approach (3rd Edition).
Russel and Norvig, Fig 14.2 pp 512.
'''

from spn.distributions import Bernoulli
from spn.interpret import Cond
from spn.interpret import Otherwise
from spn.interpret import Start
from spn.interpret import Variable

Burglary = Variable('B')
Earthquake = Variable('E')
Alarm = Variable('A')
JohnCalls = Variable('JC')
MaryCalls = Variable('MC')

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

def test_mutual_information():
    event_a = (JohnCalls << {1}) | (MaryCalls << {1})
    event_b = (Burglary << {1}) & (Earthquake << {0})
    print(model.mutual_information(event_a, event_b))
