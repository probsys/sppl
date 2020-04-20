# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

'''
Burglary network example from:

Artificial Intelligence: A Modern Approach (3rd Edition).
Russel and Norvig, Fig 14.2 pp 512.
'''

from spn.distributions import bernoulli
from spn.interpreter import IfElse
from spn.interpreter import Otherwise
from spn.interpreter import Sample
from spn.interpreter import Start
from spn.interpreter import Variable

Burglary    = Variable('Burglary')
Earthquake  = Variable('Earthquake')
Alarm       = Variable('Alarm')
JohnCalls   = Variable('JohnCalls')
MaryCalls   = Variable('MaryCalls')

model = (Start
    & Sample (Burglary,   bernoulli(p=0.001))
    & Sample (Earthquake, bernoulli(p=0.002))
    & IfElse (
        Burglary << {1},
            IfElse (
                Earthquake << {1},  Sample(Alarm, bernoulli(p=0.95)),
                Otherwise,          Sample(Alarm, bernoulli(p=0.94))),
        Otherwise,
            IfElse (
                Earthquake << {1},  Sample(Alarm, bernoulli(p=0.29)),
                Otherwise,          Sample(Alarm, bernoulli(p=0.001))))
    & IfElse (
        Alarm << {1},
            Sample(JohnCalls,   bernoulli(p=0.90))
            & Sample(MaryCalls, bernoulli(p=0.70)),
        Otherwise,
            Sample(JohnCalls,   bernoulli(p=0.05))
            & Sample(MaryCalls, bernoulli(p=0.01)),
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
