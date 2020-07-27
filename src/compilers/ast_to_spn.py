# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

"""Convert AST to SPN."""

import os

from functools import reduce

from ..dnf import dnf_normalize
from ..math_util import allclose
from ..math_util import isinf_neg
from ..math_util import logsumexp
from ..sets import FiniteNominal
from ..sets import FiniteReal
from ..sets import Set
from ..spn import Memo
from ..spn import SumSPN
from ..spn import spn_simplify_sum

from .. import transforms

inf = float('inf')

Id = transforms.Id
def IdArray(token, n):
    return [Id('%s[%d]' % (token, i,)) for i in range(n)]

class Command():
    def interpret(self, spn):
        raise NotImplementedError()

class Skip(Command):
    def __init__(self):
        pass
    def interpret(self, spn=None):
        return spn

class Sample(Command):
    def __init__(self, symbol, distribution):
        self.symbol = symbol
        self.distribution = distribution
    def interpret(self, spn=None):
        leaf = self.symbol >> self.distribution
        return leaf if (spn is None) else spn & leaf

class Transform(Command):
    def __init__(self, symbol, expr):
        self.symbol = symbol
        self.expr = expr
    def interpret(self, spn=None):
        assert spn is not None
        return spn.transform(self.symbol, self.expr)

class Condition(Command):
    def __init__(self, event):
        self.event = event
    def interpret(self, spn=None):
        assert spn is not None
        return spn.condition(self.event)

class IfElse(Command):
    def __init__(self, *branches):
        assert len(branches) % 2 == 0
        self.branches = branches
    def interpret(self, spn=None):
        assert spn is not None
        conditions = self.branches[::2]
        subcommands = self.branches[1::2]
        # Make events for each condition.
        if conditions[-1] is True:
            events_if = [
                reduce(lambda x, e: x & ~e, conditions[:i], conditions[i])
                for i in range(len(conditions)-1)
            ]
            event_else = ~reduce(lambda x, e: x|e, conditions[:-1])
            events_unorm = events_if + [event_else]
        else:
            events_unorm = [
                reduce(lambda x, e: x & ~e, conditions[:i], conditions[i])
                for i in range(len(conditions))
            ]
        # Rewrite events in normalized form.
        events = [dnf_normalize(event) for event in events_unorm]
        # Rewrite events in normalized form.
        return interpret_if_block(spn, events, subcommands)

class For(Command):
    def __init__(self, n0, n1, f):
        self.n0 = n0
        self.n1 = n1
        self.f = f
    def interpret(self, spn=None):
        commands = [self.f(i) for i in range(self.n0, self.n1)]
        sequence = Sequence(*commands)
        return sequence.interpret(spn)

class Switch(Command):
    def __init__(self, symbol, values, f):
        self.symbol = symbol
        self.f = f
        self.values = values
    def interpret(self, spn=None):
        if isinstance(self.values, enumerate):
            values = list(self.values)
            sets = [self.value_to_set(v[1]) for v in values]
            subcommands = [self.f(*v) for v in values]
        else:
            sets = [self.value_to_set(v) for v in self.values]
            subcommands = [self.f(v) for v in self.values]
        sets_disjoint = [
            reduce(lambda x, s: x & ~s, sets[:i], sets[i])
            for i in range(len(sets))]
        events = [self.symbol << s for s in sets_disjoint]
        return interpret_if_block(spn, events, subcommands)
    def value_to_set(self, v):
        if isinstance(v, Set):
            return v
        if isinstance(v, str):
            return FiniteNominal(v)
        return FiniteReal(v)

class Sequence(Command):
    def __init__(self, *commands):
        self.commands = commands
    def interpret(self, spn=None):
        return reduce(lambda S, c: c.interpret(S), self.commands, spn)

Otherwise = True

def interpret_if_block(spn, events, subcommands):
    assert len(events) == len(subcommands)
    # Prepare memo table.
    memo = Memo()
    # Obtain mixture probabilities.
    weights = [spn.logprob(event, memo)
        if event is not None else -inf for event in events]
    # Filter the irrelevant ones.
    indexes = [i for i, w in enumerate(weights) if not isinf_neg(w)]
    assert indexes, 'All conditions probability zero.'
    # Obtain conditioned SPNs.
    weights_conditioned = [weights[i] for i in indexes]
    spns_conditioned = [spn.condition(events[i], memo) for i in indexes]
    subcommands_conditioned = [subcommands[i] for i in indexes]
    assert allclose(logsumexp(weights_conditioned), 0)
    # Make the children.
    children = [
        subcommand.interpret(S)
        for S, subcommand in zip(spns_conditioned, subcommands_conditioned)
    ]
    # Maybe Simplify.
    if len(children) == 1:
        spn = children[0]
    else:
        spn = SumSPN(children, weights_conditioned)
        if not os.environ.get('SPN_NO_SIMPLIFY'):
            spn = spn_simplify_sum(spn)
    # Return the SPN.
    return spn
