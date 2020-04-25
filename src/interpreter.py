# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from functools import reduce

from .dnf import dnf_normalize
from .math_util import allclose
from .math_util import isinf_neg
from .math_util import logsumexp
from .spn import Memo
from .spn import SPN
from .spn import SumSPN

from . import transforms

inf = float('inf')

Variable = transforms.Identity
def VariableArray(token, n):
    return [Variable('%s[%d]' % (token, i,)) for i in range(n)]

class Command():
    def interpret(self, spn):
        raise NotImplementedError()
    def __and__(self, x):
        if isinstance(x, Sequence):
            commands = (self,) + x.commands
            return Sequence(*commands)
        if isinstance(x, Command):
            return Sequence(self, x)
        return NotImplemented
    def __rand__(self, x):
        if x is Start:
            return self.interpret(x)
        if isinstance(x, SPN):
            return self.interpret(x)
        return NotImplemented

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
        # Prepare memo table.
        memo = Memo({}, {})
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
        # Return the SPN.
        return SumSPN(children, weights_conditioned) if 1 < len(children) else children[0]

class For(Command):
    def __init__(self, n0, n1, f):
        self.n0 = n0
        self.n1 = n1
        self.f = f
    def interpret(self, spn=None):
        commands = [self.f(i) for i in range(self.n0, self.n1)]
        sequence = Sequence(*commands)
        return sequence.interpret(spn)

class Sequence(Command):
    def __init__(self, *commands):
        self.commands = commands
    def interpret(self, spn=None):
        return reduce(lambda S, c: c.interpret(S), self.commands, spn)
    def __and__(self, x):
        if isinstance(x, Sequence):
            commands = self.commands + x.commands
            return Sequence(commands)
        if isinstance(x, Command):
            commands = self.commands + (x,)
            return Sequence(*commands)
        return NotImplemented

Start = None
Otherwise = True
