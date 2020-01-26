# Copyright 2019 MIT Probabilistic Computing Project.
# See LICENSE.txt

import sympy

from .sym_util import Reals

# ==============================================================================
# Custom event language.

class Event(object):
    def __and__(self, event):
        assert isinstance(event, Event)
        return EventAnd([self, event])
    def __or__(self, event):
        assert isinstance(event, Event)
        return EventOr([self, event])
    def __invert__(self):
        return EventNot(self)

class EventInterval(Event):
    def __init__(self, expr, interval):
        self.interval = interval
        self.expr = expr
    def solve(self):
        return self.expr.invert(self.interval)
    def __eq__(self, event):
        return (self.interval == event.interval) and (self.expr == event.expr)
    def __repr__(self):
        return 'EventInterval(%s, %s)' \
            % (repr(self.expr), repr(self.interval))

class EventOr(Event):
    def __init__(self, events):
        self.events = events
    def solve(self):
        intervals = [event.solve() for event in self.events]
        return sympy.Union(*intervals)
    def __eq__(self, event):
        return self.events == event.events
    def __repr__(self):
        return 'EventOr(%s)' % (repr(self.events,))

class EventAnd(Event):
    def __init__(self, events):
        self.events = events
    def solve(self):
        intervals = [event.solve() for event in self.events]
        return sympy.Intersection(*intervals)
    def __eq__(self, event):
        return self.events == event.events
    def __repr__(self):
        return 'EventAnd(%s)' % (repr(self.events,))

class EventNot(Event):
    def __init__(self, event):
        self.event = event
    def solve(self):
        # TODO Should complement range not Reals.
        interval = self.event.solve()
        return interval.complement(Reals)
    def __invert__(self):
        return self.event
    def __eq__(self, event):
        return event == self.event
    def __repr__(self):
        return 'EventNot(%s)' % (repr(self.event,))
