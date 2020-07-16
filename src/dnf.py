# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from functools import reduce
from itertools import chain
from itertools import combinations

from .sets import EmptySet
from .transforms import EventAnd
from .transforms import EventBasic
from .transforms import EventOr
from .transforms import Id

def dnf_factor(event, lookup=None):
    # Given an event (in DNF) and a dictionary lookup mapping symbols
    # to integers, this function returns a list R of dictionaries
    # R[i][j] is a conjunction events in the i-th DNF clause whose symbols
    # are assigned to integer j in the lookup dictionary.
    #
    # For example, if e is any predicate
    # event = (e(X0) & e(X1) & ~e(X2)) | (~e(X1) & e(X2) & e(X3) & e(X4)))
    # lookup = {X0: 0, X1: 1, X2: 0, X3: 1, X4: 2}
    # The output is
    # R = [
    #   { // First clause
    #       0: e(X0) & ~e(X2),
    #       1: e(X1)},
    #   { // Second clause
    #       0: e(X2),
    #       1: ~e(X1) & e(X3)},
    #       2: e(X4)},
    # ]
    if lookup is None:
        lookup = {s:s for s in event.get_symbols()}

    if isinstance(event, EventBasic):
        # Literal.
        symbols = event.get_symbols()
        assert len(symbols) == 1
        symbol = list(symbols)[0]
        key = lookup[symbol]
        return ({key: event},)

    if isinstance(event, EventAnd):
        # Conjunction.
        assert all(isinstance(e, EventBasic) for e in event.subexprs)
        mappings = (dnf_factor(e, lookup) for e in event.subexprs)
        events = {}
        for mapping in mappings:
            for m in mapping:
                for key, ev in m.items():
                    if key not in events:
                        events[key] = ev
                    else:
                        events[key] &= ev
        return (events,)

    if isinstance(event, EventOr):
        # Disjunction.
        assert all(isinstance(e, (EventAnd, EventBasic)) for e in event.subexprs)
        mappings = (dnf_factor(e, lookup) for e in event.subexprs)
        return tuple(chain.from_iterable(mappings))

    assert False, 'Invalid DNF event: %s' % (event,)

def dnf_normalize(event):
    if isinstance(event, EventBasic):
        if isinstance(event.subexpr, Id):
            return event
    # Given an arbitrary event, rewrite in terms of only Id by
    # solving the subexpressions and return the resulting DNF formula,
    # or None if all solutions evaluate to EmptySet.
    event_dnf = event.to_dnf()
    event_factor = dnf_factor(event_dnf)
    solutions = list(filter(lambda x: all(y[1] is not EmptySet for y in x), [
        [(symbol, ev.solve()) for symbol, ev in clause.items()]
        for clause in event_factor
    ]))
    if not solutions:
        return None
    conjunctions = [
        reduce(lambda x, e: x & e, [(symbol << S) for symbol, S in clause])
        for i, clause in enumerate(solutions) if clause not in solutions[:i]
    ]
    disjunctions = reduce(lambda x, e: x|e, conjunctions)
    return disjunctions.to_dnf()

def dnf_non_disjoint_clauses(event):
    # Given an event in DNF, returns a dictionary R
    # such that R[j] = [i | i < j and event[i] intersects event[j]]
    event_factor = dnf_factor(event)
    solutions = [
        {symbol: ev.solve() for symbol, ev in clause.items()}
        for clause in event_factor
    ]

    n_clauses = len(event_factor)
    overlap_dict = {}
    for i, j in combinations(range(n_clauses), 2):
        # Exit if any symbol in i does not intersect a symbol in j.
        intersections = (
            solutions[i][symbol] & solutions[j][symbol]
                if (symbol in solutions[j]) else
                solutions[i][symbol]
            for symbol in solutions[i]
        )
        if any(x is EmptySet for x in intersections):
            continue
        # Exit if any symbol in j is EmptySet.
        if any(solutions[j] is EmptySet for symbol in solutions[j]):
            continue
        # All symbols intersect, so clauses overlap.
        if j not in overlap_dict:
            overlap_dict[j] = []
        overlap_dict[j].append(i)

    return overlap_dict

def dnf_to_disjoint_union(event):
    # Given an event in DNF, returns an event in DNF where all the
    # clauses are disjoint from one another, by recursively solving the
    # identity E = (A or B or C) = (A) or (B and ~A) or (C and ~A and ~B).
    # Base case.
    if isinstance(event, (EventBasic, EventAnd)):
        return event
    # Find indexes of pairs of clauses that overlap.
    overlap_dict = dnf_non_disjoint_clauses(event)
    if not overlap_dict:
        return event
    # Create the cascading negated clauses.
    n_clauses = len(event.subexprs)
    clauses_disjoint = [
        reduce(
            lambda state, event: state & ~event,
            (event.subexprs[j] for j in overlap_dict.get(i, [])),
            event.subexprs[i])
        for i in range(n_clauses)
    ]
    # Recursively find the solutions for each clause.
    clauses_normalized = [dnf_normalize(clause) for clause in clauses_disjoint]
    solutions = [dnf_to_disjoint_union(c) for c in clauses_normalized if c]
    # Return the merged solution.
    return reduce(lambda a, b: a|b, solutions)
