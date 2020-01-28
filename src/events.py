# Copyright 2019 MIT Probabilistic Computing Project.
# See LICENSE.txt

import itertools

from math import isinf

import sympy

from .sym_util import Reals
from .sym_util import sympify_number

# ==============================================================================
# Custom event language.

class Event(object):
    def solve(self):
        raise NotImplementedError()
    def symbols(self):
        raise NotImplementedError()
    def to_dnf(self):
        dnf = self.to_dnf_list()
        # Verifying result is in DNF.
        # for conjunction in dnf:
        #     assert isinstance(conjunction, list)
        #     for term in conjunction:
        #         assert isinstance(term, EventInterval)
        simplify_event = lambda x, E: x[0] if len(x)==1 else E(x)
        events = [simplify_event(conjunction, EventAnd) for conjunction in dnf]
        return simplify_event(events, EventOr)
    def to_dnf_list(self):
        raise NotImplementedError()
    def __and__(self, event):
        raise NotImplementedError()
        # return EventAnd([self, event])
    def __or__(self, event):
        raise NotImplementedError()
        # return EventOr([self, event])

class EventInterval(Event):
    def __init__(self, expr, interval, complement=None):
        self.interval = interval
        self.expr = expr
        self.complement = complement
    def symbols(self):
        return [self.expr.symbol()]
    def solve(self):
        solution = self.expr.invert(self.interval)
        if self.complement:
            # TODO Should complement range not Reals.
            return solution.complement(Reals)
        return solution
    def to_dnf_list(self):
        return [[self]]
    def __and__(self, event):
        if isinstance(event, EventAnd):
            events = (self,) + event.events
            return EventAnd(events)
        if isinstance(event, (EventInterval, EventOr)):
            return EventAnd([self, event])
        raise NotImplementedError()
    def __or__(self, event):
        if isinstance(event, EventOr):
            events = (self,) + event.events
            return EventOr(events)
        if isinstance(event, (EventInterval, EventAnd)):
            return EventOr([self, event])
        raise NotImplementedError()
    def __eq__(self, event):
        return isinstance(event, EventInterval) \
            and (self.interval == event.interval) \
            and (self.expr == event.expr) \
            and (self.complement == event.complement)
    def __compute_gte__(self, x, left_open):
        # 5 < (X < number)
        if not (isinf(self.interval.left) and self.interval.left < 0):
            raise ValueError('cannot compute %s < %s' % (x, str(self)))
        if self.complement:
            raise ValueError('cannot compute < with complement')
        xn = sympify_number(x)
        interval = sympy.Interval(xn, self.interval.right,
            left_open=left_open, right_open=self.interval.right_open)
        return EventInterval(self.expr, interval, complement=self.complement)
    def __compute_lte__(self, x, right_open):
        # 5 < (X < number)
        if not (isinf(self.interval.right) and 0 < self.interval.right):
            raise ValueError('cannot compute %s < %s' % (str(self), x))
        if self.complement:
            raise ValueError('cannot compute < with complement')
        xn = sympify_number(x)
        interval = sympy.Interval(self.interval.left, xn,
            left_open=self.interval.left_open, right_open=right_open)
        return EventInterval(self.expr, interval, complement=self.complement)
    def __gt__(self, x):
        return self.__compute_gte__(x, True)
    def __ge__(self, x):
        return self.__compute_gte__(x, False)
    def __lt__(self, x):
        return self.__compute_lte__(x, True)
    def __le__(self, x):
        return self.__compute_lte__(x, False)
    def __repr__(self):
        return 'EventInterval(%s, %s, complement=%s)' \
            % (repr(self.expr), repr(self.interval), repr(self.complement))
    def __invert__(self):
        return EventInterval(self.expr, self.interval, not self.complement)
    def __str__(self):
        sym = str(self.expr)
        (x_l, x_r) = (self.interval.left, self.interval.right)
        comp_l = '<' if self.interval.left_open else '<='
        comp_r = '<' if self.interval.right_open else '<='
        if x_l < 0 and isinf(x_l):
            return '%s %s %s' % (sym, comp_r, x_r)
        if 0 < x_r and isinf(x_r):
            return '%s %s %s' % (x_l, comp_l, sym)
        return '%s %s %s %s %s' % (x_l, comp_l, sym, comp_r, x_r)

class EventOr(Event):
    def __init__(self, events):
        self.events = tuple(events)
    def symbols(self):
        sub_symbols = [event.symbols() for event in self.events]
        symbols = itertools.chain.from_iterable(sub_symbols)
        return list(set(symbols))
    def solve(self):
        intervals = [event.solve() for event in self.events]
        return sympy.Union(*intervals)
    def to_dnf_list(self):
        sub_dnf = [event.to_dnf_list() for event in self.events]
        return list(itertools.chain.from_iterable(sub_dnf))
    def __and__(self, event):
        if isinstance(event, EventAnd):
            events = (self,) + event.events
            return EventAnd(events)
        if isinstance(event, (EventInterval, EventOr)):
            events = (self, event)
            return EventAnd(events)
        raise NotImplementedError()
    def __or__(self, event):
        if isinstance(event, EventOr):
            events = self.events + event.events
            return EventOr(events)
        if isinstance(event, (EventInterval, EventAnd)):
            events = self.events + (event,)
            return EventOr(events)
        raise NotImplementedError()
    def __eq__(self, event):
        return isinstance(event, EventOr) and (self.events == event.events)
    def __invert__(self):
        sub_events = [~event for event in self.events]
        return EventAnd(sub_events)
    def __repr__(self):
        return 'EventOr(%s)' % (repr(self.events,))
    def __str__(self):
        sub_events = ['(%s)' % (str(event),) for event in self.events]
        return ' | '.join(sub_events)
    def __hash__(self):
        x = (self.__class__, self.events)
        return hash(x)

class EventAnd(Event):
    def __init__(self, events):
        self.events = tuple(events)
    def symbols(self):
        sub_symbols = [event.symbols() for event in self.events]
        symbols = itertools.chain.from_iterable(sub_symbols)
        return list(set(symbols))
    def solve(self):
        intervals = [event.solve() for event in self.events]
        return sympy.Intersection(*intervals)
    def to_dnf_list(self):
        sub_dnf = [event.to_dnf_list() for event in self.events]
        return [
            list(itertools.chain.from_iterable(cross))
            for cross in itertools.product(*sub_dnf)
        ]
    def __and__(self, event):
        if isinstance(event, EventAnd):
            events = self.events + event.events
            return EventAnd(events)
        if isinstance(event, (EventInterval, EventOr)):
            events = self.events + (event,)
            return EventAnd(events)
        raise NotImplementedError()
    def __or__(self, event):
        if isinstance(event, EventOr):
            events = (self,) + event.events
            return EventOr(events)
        if isinstance(event, (EventInterval, EventAnd)):
            events = (self, event)
            return EventOr(events)
        raise NotImplementedError()
    def __eq__(self, event):
        return isinstance(event, EventAnd) and (self.events == event.events)
    def __invert__(self):
        sub_events = [~event for event in self.events]
        return EventOr(sub_events)
    def __repr__(self):
        return 'EventAnd(%s)' % (repr(self.events,))
    def __str__(self):
        sub_events = ['(%s)' % (str(event),) for event in self.events]
        return ' & '.join(sub_events)
    def __hash__(self):
        x = (self.__class__, self.events)
        return hash(x)
