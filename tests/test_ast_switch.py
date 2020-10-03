# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from math import log

import pytest

from numpy import linspace

from sppl.distributions import bernoulli
from sppl.distributions import beta
from sppl.distributions import randint
from sppl.compilers.ast_to_spn import IfElse
from sppl.compilers.ast_to_spn import Sample
from sppl.compilers.ast_to_spn import Sequence
from sppl.compilers.ast_to_spn import Switch
from sppl.compilers.ast_to_spn import Id
from sppl.math_util import allclose
from sppl.math_util import logsumexp
from sppl.sym_util import binspace

Y = Id('Y')
X = Id('X')

def test_simple_model_eq():
    command_switch = Sequence(
        Sample(Y, randint(low=0, high=4)),
        Switch(Y, range(0, 4), lambda i:
            Sample(X, bernoulli(p=1/(i+1)))))
    model_switch = command_switch.interpret()

    command_ifelse = Sequence(
        Sample(Y, randint(low=0, high=4)),
        IfElse(
            Y << {0}, Sample(X, bernoulli(p=1/(0+1))),
            Y << {1}, Sample(X, bernoulli(p=1/(1+1))),
            Y << {2}, Sample(X, bernoulli(p=1/(2+1))),
            Y << {3}, Sample(X, bernoulli(p=1/(3+1))),
        ))
    model_ifelse = command_ifelse.interpret()

    for model in [model_switch, model_ifelse]:
        symbols = model.get_symbols()
        assert symbols == {X, Y}
        assert allclose(
            model.logprob(X << {1}),
            logsumexp([-log(4) - log(i+1) for i in range(4)]))

def test_simple_model_lte():
    command_switch = Sequence(
        Sample(Y, beta(a=2, b=3)),
        Switch(Y, binspace(0, 1, 5), lambda i:
            Sample(X, bernoulli(p=i.right))))
    model_switch = command_switch.interpret()

    command_ifelse = Sequence(
        Sample(Y, beta(a=2, b=3)),
        IfElse(
            Y <= 0,     Sample(X, bernoulli(p=0)),
            Y <= 0.25,  Sample(X, bernoulli(p=.25)),
            Y <= 0.50,  Sample(X, bernoulli(p=.50)),
            Y <= 0.75,  Sample(X, bernoulli(p=.75)),
            Y <= 1,     Sample(X, bernoulli(p=1)),
        ))
    model_ifelse = command_ifelse.interpret()

    grid = [float(x) for x in linspace(0, 1, 5)]
    for model in [model_switch, model_ifelse]:
        symbols = model.get_symbols()
        assert symbols == {X, Y}
        assert allclose(
            model.logprob(X << {1}),
            logsumexp([
                model.logprob((il < Y) <= ih) + log(ih)
                for il, ih in zip(grid[:-1], grid[1:])
            ]))

def test_simple_model_enumerate():
    command_switch = Sequence(
        Sample(Y, randint(low=0, high=4)),
        Switch(Y, enumerate(range(0, 4)), lambda i,j:
            Sample(X, bernoulli(p=1/(i+j+1)))))
    model = command_switch.interpret()
    assert allclose(model.prob(Y<<{0} & (X << {1})), .25 * 1/(0+0+1))
    assert allclose(model.prob(Y<<{1} & (X << {1})), .25 * 1/(1+1+1))
    assert allclose(model.prob(Y<<{2} & (X << {1})), .25 * 1/(2+2+1))
    assert allclose(model.prob(Y<<{3} & (X << {1})), .25 * 1/(3+3+1))

def test_error_range():
    with pytest.raises(AssertionError):
        # Switch cases do not sum to one.
        command = Sequence(
            Sample(Y, randint(low=0, high=4)),
            Switch(Y, range(0, 3), lambda i:
                Sample(X, bernoulli(p=1/(i+1)))))
        command.interpret()

def test_error_linspace():
    with pytest.raises(AssertionError):
        # Switch cases do not sum to one.
        command = Sequence(
            Sample(Y, beta(a=2, b=3)),
            Switch(Y, linspace(0, .5, 5), lambda i:
                Sample(X, bernoulli(p=i))))
        command.interpret()

def test_error_binspace():
    with pytest.raises(AssertionError):
        # Switch cases do not sum to one.
        command = Sequence(
            Sample(Y, beta(a=2, b=3)),
            Switch(Y, binspace(0, .5, 5), lambda i:
                Sample(X, bernoulli(p=i.right))))
        command.interpret()
