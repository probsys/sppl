# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from functools import reduce
from itertools import chain
from itertools import combinations

from sympy import Intersection

from .sym_util import EmptySet

from .transforms import EventAnd
from .transforms import EventBasic
from .transforms import EventOr

def factor_dnf(event):
    lookup = {s:s for s in event.symbols()}
    return factor_dnf_symbols(event, lookup)

def factor_dnf_symbols(event, lookup):
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
    if isinstance(event, EventBasic):
        # Literal.
        symbols = event.symbols()
        assert len(symbols) == 1
        key = lookup[symbols[0]]
        return [{key: event}]

    if isinstance(event, EventAnd):
        # Conjunction.
        assert all(isinstance(e, EventBasic) for e in event.subexprs)
        mappings = [factor_dnf_symbols(e, lookup) for e in event.subexprs]
        events = {}
        for mapping in mappings:
            assert len(mapping) == 1
            [(key, ev)] = mapping[0].items()
            if key not in events:
                events[key] = ev
            else:
                events[key] &= ev
        return [events]

    if isinstance(event, EventOr):
        # Disjunction.
        assert all(isinstance(e, (EventAnd, EventBasic)) for e in event.subexprs)
        mappings = [factor_dnf_symbols(e, lookup) for e in event.subexprs]
        events = [None] * len(mappings)
        for i, mapping in enumerate(mappings):
            events[i] = {}
            for key, ev in mapping[0].items():
                events[i][key] = ev
        return events

    assert False, 'Invalid DNF event: %s' % (event,)

def solve_dnf_symbolwise(dnf_factor):
    # Given a factored event (in DNF) where distinct symbols have
    # distinct keys, returns a list R of dictionaries where
    # R[i][s] is the solution of the events in the i-th DNF clause with
    # symbol s.
    #
    # For example, if e is any predicate
    # event = (e(X0) & e(X1) & ~e(X2)) | (~e(X1) & e(X2) & e(X3) & ~e(X3)))
    # The output is
    # R = [
    #   { // First clause
    #       X0: solve(e(X0)),
    #       X1: solve(e(X1)),
    #       X2: solve(e(X2))},
    #   { // Second clause
    #       X0: solve(~e(X1)),
    #       X2: solve(e(X2)),
    #       X3: solve(e(X3) & ~e(X3))},
    # ]
    solutions = [None]*len(dnf_factor)
    for i, event_mapping in enumerate(dnf_factor):
        solutions[i] = {}
        for symbol, ev in event_mapping.items():
            solutions[i][symbol] = ev.solve()
    return solutions

def find_dnf_non_disjoint_clauses(event):
    # Given an event in DNF, returns a dictionary R
    # such that R[j] = [i | i < j and event[i] intersects event[j]]
    dnf_factor = factor_dnf(event)
    solutions = solve_dnf_symbolwise(dnf_factor)

    clauses = range(len(dnf_factor))
    overlap_dict = {}
    for i, j in combinations(clauses, 2):
        # Intersections of events in i with those in j.
        intersections_symbols_i = {
            symbol: Intersection(solutions[i][symbol], solutions[j][symbol])
                if (symbol in solutions[j]) else solutions[i][symbol]
            for symbol in solutions[i]
        }
        # Intersections of events only in j.
        intersections_symbols_j = {
            symbol: solutions[j][symbol]
            for symbol in solutions[j] if symbol not in solutions[i]
        }
        # Clauses are non disjoint if all symbols intersect.
        intersections = {**intersections_symbols_i, **intersections_symbols_j}
        if all(s is not EmptySet for s in intersections.values()):
            if j not in overlap_dict:
                overlap_dict[j] = []
            overlap_dict[j].append(i)

    return overlap_dict

def event_to_disjoint_union(event):
    event_dnf = event.to_dnf()
    # Base case.
    if isinstance(event_dnf, (EventBasic, EventAnd)):
        return event_dnf
    # Find indexes of pairs of clauses that overlap.
    overlap_dict = find_dnf_non_disjoint_clauses(event_dnf)
    if not overlap_dict:
        return event_dnf
    # Create the cascading negated clauses.
    n_clauses = len(event_dnf.subexprs)
    clauses_disjoint = [
        reduce(
            lambda state, event: state & ~event,
            (event_dnf.subexprs[j] for j in overlap_dict.get(i, [])),
            event_dnf.subexprs[i])
        for i in range(n_clauses)
    ]
    # Recursively find the solutions for each clause.
    solutions = [event_to_disjoint_union(clause) for clause in clauses_disjoint]
    # Return the merged solution.
    return reduce(lambda a, b: a|b, solutions)
