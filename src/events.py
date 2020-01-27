# Copyright 2019 MIT Probabilistic Computing Project.
# See LICENSE.txt

import itertools

import sympy

from .sym_util import Reals

# ==============================================================================
# Custom event language.

class Event(object):
    def solve(self):
        raise NotImplementedError
    def to_dnf(self):
        dnf = self.dnf_list()
        for conjunction in dnf:
            assert isinstance(conjunction, list)
            for term in conjunction:
                assert isinstance(term, EventInterval)
        simplify_event = lambda x, construct: x[0] if len(x)==1 else construct(x)
        events = [simplify_event(conjunction, EventAnd) for conjunction in dnf]
        return simplify_event(events, EventOr)
    def dnf_list(self):
        raise NotImplementedError
    def __and__(self, event):
        assert isinstance(event, Event)
        # TODO: If event is and, concatenate the event list.
        return EventAnd([self, event])
    def __or__(self, event):
        assert isinstance(event, Event)
        # TODO: If event is and, concatenate the event list.
        return EventOr([self, event])
    # TODO: Implement __str__.

class EventInterval(Event):
    def __init__(self, expr, interval, complement=None):
        self.interval = interval
        self.expr = expr
        self.complement = complement
    def solve(self):
        interval = self.expr.invert(self.interval)
        if self.complement:
            # TODO Should complement range not Reals.
            return interval.complement(Reals)
        return interval
    def dnf_list(self):
        return [[self]]
    def __eq__(self, event):
        return (self.interval == event.interval) \
            and (self.expr == event.expr) \
            and (self.complement == event.complement)
    def __repr__(self):
        return 'EventInterval(%s, %s, complement=%s)' \
            % (repr(self.expr), repr(self.interval), repr(self.complement))
    def __invert__(self):
        return EventInterval(self.expr, self.interval, not self.complement)

class EventOr(Event):
    def __init__(self, events):
        self.events = events
    def solve(self):
        intervals = [event.solve() for event in self.events]
        return sympy.Union(*intervals)
    def dnf_list(self):
        sub_dnf = [event.dnf_list() for event in self.events]
        return list(itertools.chain.from_iterable(sub_dnf))
    def __eq__(self, event):
        return self.events == event.events
    def __repr__(self):
        return 'EventOr(%s)' % (repr(self.events,))
    def __invert__(self):
        sub_events = [~event for event in self.events]
        return EventAnd(sub_events)

class EventAnd(Event):
    def __init__(self, events):
        self.events = events
    def solve(self):
        intervals = [event.solve() for event in self.events]
        return sympy.Intersection(*intervals)
    def dnf_list(self):
        sub_dnf = [event.dnf_list() for event in self.events]
        return [
            list(itertools.chain.from_iterable(cross))
            for cross in itertools.product(*sub_dnf)
        ]
    def __eq__(self, event):
        return self.events == event.events
    def __repr__(self):
        return 'EventAnd(%s)' % (repr(self.events,))
    def __invert__(self):
        sub_events = [~event for event in self.events]
        return EventOr(sub_events)
