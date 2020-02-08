# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from sum_product_dsl.events import EventAnd
from sum_product_dsl.events import EventBasic
from sum_product_dsl.events import EventOr

def factor_dnf(event):
    lookup = {s:s for s in event.symbols}
    return factor_dnf_symbols(event, lookup)

def factor_dnf_symbols(event, lookup):
    if isinstance(event, EventBasic):
        # Literal term.
        assert len(event.symbols) == 1
        key = lookup[event.symbols[0]]
        return {0: {key: event}}

    if isinstance(event, EventAnd):
        # Product term.
        assert all(isinstance(e, EventBasic) for e in event.events)
        mappings = [factor_dnf_symbols(e, lookup) for e in event.events]
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
        assert all(isinstance(e, (EventAnd, EventBasic)) for e in event.events)
        mappings = [factor_dnf_symbols(e, lookup) for e in event.events]
        events = {}
        for i, mapping in enumerate(mappings):
            events[i] = {}
            for key, ev in mapping[0].items():
                events[i][key] = ev
        return events

    assert False, 'Invalid DNF event: %s' % (event,)
