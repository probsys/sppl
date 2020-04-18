Probabilistic Programming with Sum-Product Networks
===================================================

## Installation

This software is tested on Ubuntu 18.04 and requires Python 3.
Please install the dependencies listed in `requirements.txt` (PyPI)
and `requirements.sh` (apt).

## Tests

Run the following command in the shell:

    $ ./check.sh

## Examples

1. Burglary Network

```python
from spn.distributions import bernoulli
from spn.interpret import Cond
from spn.interpret import Otherwise
from spn.interpret import Start
from spn.interpret import Variable

# Declare variables in the model.
Burglary    = Variable('B')
Earthquake  = Variable('E')
Alarm       = Variable('A')
JohnCalls   = Variable('JC')
MaryCalls   = Variable('MC')

# Define the model.
model = (Start
    & Burglary >> bernoulli(p=0.001)
    & Earthquake >> bernoulli(p=0.002)
    & Cond (
        Burglary << {1},
            Cond (
                Earthquake << {1},  Alarm >> bernoulli(p=0.95),
                Otherwise,          Alarm >> bernoulli(p=0.94)),
        Otherwise,
            Cond (
                Earthquake << {1},  Alarm >> bernoulli(p=0.29),
                Otherwise,          Alarm >> bernoulli(p=0.001)))
    & Cond (
        Alarm << {1},
            JohnCalls >> bernoulli(p=0.90)
            & MaryCalls >> bernoulli(p=0.70),
        Otherwise,
            JohnCalls >> bernoulli(p=0.05)
            & MaryCalls >> bernoulli(p=0.01),
        ))

# Ask a marginal query.
event = ((JohnCalls << {1}) & (MaryCalls << {1}) & (Alarm << {1})
  & (Burglary << {0}) & (Earthquake << {0}))
print(model.prob(event))

# Ask a conditional query.
event = (JohnCalls << {1}) | (MaryCalls << {1})
model_condition = model.condition(event)
print(model_condition.prob(Burglary << {1}))

# Ask a mutual information query.
event_a = (JohnCalls << {1}) | (MaryCalls << {1})
event_b = (Burglary << {1}) & (Earthquake << {0})
print(model.mutual_information(event_a, event_b))
```

2. Indian GPA

```python
from spn.distributions import atomic
from spn.distributions import uniform
from spn.interpret import Cond
from spn.interpret import Otherwise
from spn.interpret import Start
from spn.interpret import Variable

# Declare variables in the model.
Nationality = Variable('Nationality')
Perfect     = Variable('Perfect')
GPA         = Variable('GPA')

model = (Start
        & Nationality   >> {'Indian': 0.5, 'USA': 0.5}
        & Cond (
            (Nationality << {'Indian'}),
                Perfect >> {'True': 0.01, 'False': 0.99}
                & Cond (
                    Perfect << {'True'},    GPA >> atomic(loc=10),
                    Otherwise,              GPA >> uniform(scale=10)),
            Otherwise,
                Perfect >> {'True': 0.01, 'False': 0.99}
                & Cond (
                    Perfect << {'True'},    GPA >> atomic(loc=4),
                    Otherwise,              GPA >> uniform(scale=4))))
```
