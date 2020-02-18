# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from spn.transforms import EventAnd
from spn.transforms import EventBasic
from spn.transforms import EventOr

def factor_dnf(event):
    lookup = {s:s for s in event.symbols()}
    return factor_dnf_symbols(event, lookup)

def factor_dnf_symbols(event, lookup):
    # Given an event (in DNF) and a dictionary lookup mapping symbols
    # to integers, this function returns a dictionary of dictionary R
    # R[i][j] is a conjunction events in the i-th DNF clause whose symbols
    # are assigned to integer j in the lookup dictionary.
    #
    # For example, if e is any predicate
    # event = (e(X0) & e(X1) & ~e(X2)) | (~e(X1) & e(X2) & e(X3) & e(X4)))
    # lookup = {X0: 0, X1: 1, X2: 0, X3: 1, X4: 2}
    # The output is
    # R = {
    #   0 : { // First clause
    #       0: e(X0) & ~e(X2),
    #       1: e(X1)},
    #   1 : { // Second clause
    #       0: e(X2),
    #       1: ~e(X1) & e(X3)},
    #       2: e(X4)},
    # }
    if isinstance(event, EventBasic):
        # Literal term.
        symbols = event.symbols()
        assert len(symbols) == 1
        key = lookup[symbols[0]]
        return {0: {key: event}}

    if isinstance(event, EventAnd):
        # Product term.
        assert all(isinstance(e, EventBasic) for e in event.subexprs)
        mappings = [factor_dnf_symbols(e, lookup) for e in event.subexprs]
        events = {0: {}}
        for mapping in mappings:
            assert len(mapping) == 1
            [(key, ev)] = mapping[0].items()
            if key not in events[0]:
                events[0][key] = ev
            else:
                events[0][key] &= ev
        return events

    if isinstance(event, EventOr):
        # Sum term.
        assert all(isinstance(e, (EventAnd, EventBasic)) for e in event.subexprs)
        mappings = [factor_dnf_symbols(e, lookup) for e in event.subexprs]
        events = {}
        for i, mapping in enumerate(mappings):
            events[i] = {}
            for key, ev in mapping[0].items():
                events[i][key] = ev
        return events

    assert False, 'Invalid DNF event: %s' % (event,)
