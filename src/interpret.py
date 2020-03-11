# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from collections import Callable
from functools import reduce

from .combinators import make_predicates_else
from .combinators import make_predicates_noelse
from .dnf import dnf_normalize
from .math_util import allclose
from .math_util import isinf_neg
from .math_util import logsumexp
from .spn import ProductSPN
from .spn import SPN
from .spn import factorize_spn_sum
from .spn import make_product_from_list

from .transforms import Identity

class Variable(Identity):
    def __rshift__(self, f):
        if isinstance(f, Callable):
            symbol = Identity(self.token)
            return Sample(symbol, f)
        return NotImplemented
    def __hash__(self):
        symbol = Identity(self.token)
        x = (symbol.__class__, self.token)
        return hash(x)

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

class IfElse(Command):
    def __init__(self, *branches):
        assert len(branches) % 2 == 0
        self.branches = branches
    def interpret(self, spn=None):
        assert spn is not None
        conditions = self.branches[::2]
        subcommands = self.branches[1::2]
        # Make events for each condition.
        events_unorm = make_predicates_else(conditions) \
            if (conditions[-1] is True) else \
            make_predicates_noelse(conditions)
        events = [dnf_normalize(event) for event in events_unorm]
        # Obtain mixture probabilities.
        weights = [spn.logprob(event) for event in events]
        # Filter the irrelevant ones.
        indexes = [i for i, w in enumerate(weights) if not isinf_neg(w)]
        assert indexes, 'All conditions probability zero.'
        # Obtain conditioned SPNs.
        weights_conditioned = [weights[i] for i in indexes]
        spns_conditioned = [spn.condition(events[i]) for i in indexes]
        assert allclose(logsumexp(weights_conditioned), 0)
        # Make the children.
        children = [
            subcommand.interpret(S)
            for S, subcommand in zip(spns_conditioned, subcommands)
        ]
        # Obtain children as lists.
        children_list = [
            spn.children if isinstance(spn, ProductSPN)
                else [spn] for spn in children
        ]
        # Simplify the children.
        children_simplified, weights_simplified = reduce(
            lambda state, cw: factorize_spn_sum(state, cw[0], cw[1]),
            zip(children_list[1:], weights_conditioned[1:]),
            (children_list[0], weights_conditioned[0]),
        )
        assert allclose(logsumexp(weights_simplified), 0)
        return make_product_from_list(children_simplified)

class Repeat(Command):
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
Cond = IfElse
