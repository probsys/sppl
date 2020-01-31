# Copyright 2019 MIT Probabilistic Computing Project.
# See LICENSE.txt

from sum_product_dsl.events import EventAnd
from sum_product_dsl.events import EventBasic
from sum_product_dsl.events import EventOr

def factor_dnf(event):
    symbols = event.symbols()
    lookup = {s:s for s in symbols}
    return factor_dnf_symbols(event, lookup)

def factor_dnf_symbols(event, lookup):
    if isinstance(event, EventBasic):
        # Literal term.
        symbols = event.symbols()
        key = lookup[symbols[0]]
        return {key: event}

    if isinstance(event, EventAnd):
        # Product term.
        assert all(isinstance(e, EventBasic) for e in event.events)
        mappings = [factor_dnf_symbols(e, lookup) for e in event.events]
        events = {}
        for mapping in mappings:
            assert len(mapping) == 1
            [(key, ev)] = mapping.items()
            if key not in events:
                events[key] = ev
            else:
                events[key] &= ev
        return events

    if isinstance(event, EventOr):
        # Sum term.
        mappings = [factor_dnf_symbols(e, lookup) for e in event.events]
        events = {}
        for mapping in mappings:
            for key, ev in mapping.items():
                if key not in events:
                    events[key] = ev
                else:
                    events[key] |= ev
        return events

    assert False, 'Invalid DNF event: %s' % (event,)
