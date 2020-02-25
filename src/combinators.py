# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from functools import reduce

from .dnf import dnf_normalize
from .spn import ProductSPN
from .spn import SumSPN

def IfElse(spn, *branches):
    # Obtain conditions and consequents of each branch.
    conditions = [branch[0] for branch in branches]
    spns_branch = [branch[1] for branch in branches]
    # Make events for each condition
    events = make_predicates_else(conditions) \
        if (conditions[-1] is True) else \
        make_predicates_noelse(conditions)
    events = [dnf_normalize(event) for event in events]
    # Obtain mixture probabilities.
    weights = [spn.logprob(event) for event in events]
    # Obtain conditioned root SPN.S
    spns_conditioned = [spn.condition(event) for event in events]
    # Make the children
    children = [
        ProductSPN([spn_conditioned, spn_branch])
        for spn_conditioned, spn_branch in zip(spns_conditioned, spns_branch)
    ]
    # Return the overall sum.
    return SumSPN(children, weights)

def make_predicates_else(conditions):
    events_if = [
        reduce(lambda x, e: x & ~e, conditions[:i], conditions[i])
        for i in range(len(conditions)-1)
    ]
    event_else = ~reduce(lambda x, e: x|e, conditions[:-1])
    return events_if + [event_else]

def make_predicates_noelse(conditions):
    return [
        reduce(lambda x, e: x & ~e, conditions[:i], conditions[i])
        for i in range(len(conditions))
    ]
