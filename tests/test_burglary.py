# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

'''
Burglary network example from:

Artificial Intelligence: A Modern Approach (3rd Edition).
Russel and Norvig, Fig 14.2 pp 512.
'''

from sppl.compilers.ast_to_spn import Id
from sppl.compilers.ast_to_spn import IfElse
from sppl.compilers.ast_to_spn import Otherwise
from sppl.compilers.ast_to_spn import Sample
from sppl.compilers.ast_to_spn import Sequence
from sppl.distributions import bernoulli

Burglary    = Id('Burglary')
Earthquake  = Id('Earthquake')
Alarm       = Id('Alarm')
JohnCalls   = Id('JohnCalls')
MaryCalls   = Id('MaryCalls')

program = Sequence(
    Sample(Burglary,   bernoulli(p=0.001)),
    Sample(Earthquake, bernoulli(p=0.002)),
    IfElse(
        Burglary << {1},
            IfElse(
                Earthquake << {1},  Sample(Alarm, bernoulli(p=0.95)),
                Otherwise,          Sample(Alarm, bernoulli(p=0.94))),
        Otherwise,
            IfElse(
                Earthquake << {1},  Sample(Alarm, bernoulli(p=0.29)),
                Otherwise,          Sample(Alarm, bernoulli(p=0.001)))),
    IfElse(
        Alarm << {1}, Sequence(
            Sample(JohnCalls, bernoulli(p=0.90)),
            Sample(MaryCalls, bernoulli(p=0.70))),
        Otherwise, Sequence(
                Sample(JohnCalls, bernoulli(p=0.05)),
                Sample(MaryCalls, bernoulli(p=0.01))),
        ))
model = program.interpret()

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
