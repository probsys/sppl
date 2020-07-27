# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from functools import reduce
from itertools import chain

from .math_util import int_or_isinf_neg
from .math_util import int_or_isinf_pos
from .math_util import isinf_neg
from .math_util import isinf_pos

inf = float('inf')
oo = inf

class Set:
    pass

# EmptySetC shall have a single instance.
class EmptySetC(Set):
    def __init__(self, force=None):
        assert force
    def __contains__(self, x):
        return False
    def __invert__(self):
        # This case is tricky; by convention, we return Real line.
        # return Union(FiniteNominal(b=True), Interval(-inf, inf))
        return Interval(-inf, inf)
    def __and__(self, x):
        if isinstance(x, Set):
            return self
        return NotImplemented
    def __or__(self, x):
        if isinstance(x, Set):
            return x
        return NotImplemented
    def __eq__(self, x):
        return x is self
    def __hash__(self):
        x = (self.__class__,)
        return hash(x)
    def __repr__(self):
        return 'EmptySet'
    def __str__(self):
        return 'EmptySet'

class FiniteNominal(Set):
    def __init__(self, *values, b=None):
        assert values or b
        self.values = frozenset(values)
        self.b = b
    def __contains__(self, x):
        if self.b:
            return x not in self.values
        return x in self.values
    def __invert__(self):
        if not self.values:
            assert self.b
            return EmptySet
        return FiniteNominal(*self.values, b=not self.b)
    def __and__(self, x):
        if isinstance(x, FiniteNominal):
            if not self.b:
                values = {v for v in self.values if v in x}
                return FiniteNominal(*values) if values else EmptySet
            if not x.b:
                values = {v for v in x.values if v in self}
                return FiniteNominal(*values) if values else EmptySet
            values = self.values | x.values
            return FiniteNominal(*values, b=True)
        if isinstance(x, (FiniteReal, Interval)):
            return EmptySet
        if isinstance(x, Set):
            return x & self
        return NotImplemented
    def __or__(self, x):
        if isinstance(x, FiniteNominal):
            if self.b:
                values = {v for v in self.values if v not in x}
                return FiniteNominal(*values, b=self.b)
            if x.b:
                values = {v for v in x.values if v not in self}
                return FiniteNominal(*values, b=x.b)
            values = self.values | x.values
            return FiniteNominal(*values, b=False)
        if isinstance(x, (FiniteReal, Interval)):
            return Union(self, x)
        if isinstance(x, Set):
            return x | self
        return NotImplemented
    def __eq__(self, x):
        return isinstance(x, FiniteNominal) \
            and bool(self.b) == bool(x.b) \
            and self.values == x.values
    def __hash__(self):
        x = (self.__class__, self.values, self.b)
        return hash(x)
    def __repr__(self):
        str_values = ', '.join(repr(x) for x in self.values)
        return 'FiniteNominal(%s, b=%s)' % (str_values, repr(self.b))
    def __str__(self):
        return '%s%s' % (('~' if self.b else ''), str(set(self.values)),)
    def __len__(self):
        return len(self.values)
    def __iter__(self):
        return iter(sorted(self.values))

class FiniteReal(Set):
    def __init__(self, *values):
        assert values
        self.values = frozenset(values)
    def __contains__(self, x):
        # inf == oo but hash(inf) != hash(oo)
        return any(x == v for v in self.values)
    def __invert__(self):
        values = sorted(self.values)
        intervals = chain(
            # Left-infinity interval.
            [Interval.Ropen(-inf, values[0])],
            # Finite intervals.
            [Interval.open(x, y) for x, y in zip(values, values[1:])],
            # Right-infinity interval.
            [Interval.Lopen(values[-1], inf)])
        return Union(*intervals)
    def __and__(self, x):
        if isinstance(x, FiniteReal):
            values = self.values & x.values
            return FiniteReal(*values) if values else EmptySet
        if isinstance(x, Interval):
            values = {v for v in self.values if v in x}
            return FiniteReal(*values) if values else EmptySet
        if isinstance(x, FiniteNominal):
            return EmptySet
        if isinstance(x, Set):
            return x & self
        return NotImplemented
    def __or__(self, x):
        if isinstance(x, FiniteReal):
            values = self.values | x.values
            return FiniteReal(*values)
        if isinstance(x, Interval):
            # Merge endpoints.
            values = set(self.values)
            interval = x
            if interval.a in values and interval.left_open:
                values.remove(interval.a)
                interval = Interval(interval.a, interval.b,
                    left_open=None,
                    right_open=interval.right_open)
            if interval.b in values and interval.right_open:
                values.remove(interval.b)
                interval = Interval(interval.a, interval.b,
                    left_open=interval.left_open,
                    right_open=None)
            values = {v for v in values if v not in interval}
            return Union(FiniteReal(*values), interval) if values else interval
        if isinstance(x, FiniteNominal):
            return Union(self, x)
        if isinstance(x, Set):
            return x | self
        return NotImplemented
    def __eq__(self, x):
        return isinstance(x, FiniteReal) \
            and self.values == x.values
    def __hash__(self):
        x = (self.__class__, self.values)
        return hash(x)
    def __repr__(self):
        return 'FiniteReal(%s)' % (', '.join(repr(x) for x in self.values))
    def __str__(self):
        return str(set(self.values))
    def __len__(self):
        return len(self.values)
    def __iter__(self):
        return iter(sorted(self.values))

class Interval(Set):
    def __init__(self, a, b, left_open=None, right_open=None):
        assert a < b
        self.a = a
        self.b = b
        self.left_open = left_open or isinf_neg(self.a)
        self.right_open = right_open or isinf_pos(self.b)
        # SymPy compatibility.
        (self.left, self.right) = (a, b)
    def __contains__(self, x):
        if self.left_open and self.right_open:
            return self.a < x < self.b
        if self.left_open and not self.right_open:
            return self.a < x <= self.b
        if not self.left_open and self.right_open:
            return self.a <= x < self.b
        if not self.left_open and not self.right_open:
            return self.a <= x <= self.b
        assert False
    def __invert__(self):
        if isinf_neg(self.a):
            if isinf_pos(self.b):
                return EmptySet
            return Interval(self.b, inf, left_open=not self.right_open)
        if isinf_pos(self.b):
            return Interval(-inf, self.a, right_open=not self.left_open)
        left = Interval(-inf, self.a, right_open=not self.left_open)
        right = Interval(self.b, inf, left_open=not self.right_open)
        return Union(left, right)
    def __and__(self, x):
        if isinstance(x, Interval):
            if x == self:
                return x
            if (x.a in self) and (x.b in self):
                return x
            if (self.a in x) and (self.b in x):
                return self
            if x.a in self:
                if self.b == x.a:
                    return FiniteReal(x.a) if x.a in x else EmptySet
                return Interval(x.a, self.b, left_open=x.left_open, right_open=self.right_open)
            if x.b in self:
                if self.a == x.b:
                    return FiniteReal(x.b) if x.b in x else EmptySet
                return Interval(self.a, x.b, left_open=self.left_open, right_open=x.right_open)
            if self.a == x.a:
                left_open = self.left_open or x.left_open
                return Interval(self.a, self.b, left_open=left_open, right_open=self.right_open)
            if self.b == x.b:
                right_open = self.right_open or x.right_open
                return Interval(self.a, self.b, left_open=self.left_open, right_open=right_open)
            return EmptySet
        if isinstance(x, Set):
            return x & self
        return NotImplemented
    def __or__(self, x):
        if isinstance(x, Interval):
            if self == x:
                return self
            intersection = x & self
            if intersection is EmptySet \
                    and (self.a not in x) \
                    and (self.b not in x) \
                    and (x.a not in self) \
                    and (x.b not in self):
                return Union(self, x)
            (al, am, bm, br) = sorted((
                (self.a, self.left_open),
                (self.b, self.right_open),
                (x.a, x.left_open),
                (x.b, x.right_open)))
            left_open = al[1] if al[0] < am[0] else (al[1] and am[1])
            right_open = br[1] if bm[0] < br[0] else (bm[1] and br[1])
            return Interval(al[0], br[0], left_open=left_open, right_open=right_open)
        if isinstance(x, Set):
            return x | self
        return NotImplemented
    def __eq__(self, x):
        return isinstance(x, Interval) \
            and self.a == x.a \
            and self.b == x.b \
            and bool(self.left_open) == bool(x.left_open) \
            and bool(self.right_open) == bool(x.right_open)
    def __hash__(self):
        x = (self.__class__, self.a, self.b, self.left_open, self.right_open)
        return hash(x)
    def __repr__(self):
        return 'Interval(%s, %s, left_open=%s, right_open=%s)' \
            % (repr(self.a), repr(self.b), repr(self.left_open), repr(self.right_open))
    def __str__(self):
        lp = '(' if self.left_open else '['
        rp = ')' if self.right_open else ']'
        return '%s%s,%s%s' % (lp, self.a, self.b, rp)
    @staticmethod
    def Lopen(a, b):
        return Interval(a, b, left_open=True)
    @staticmethod
    def Ropen(a, b):
        return Interval(a, b, right_open=True)
    @staticmethod
    def open(a, b):
        return Interval(a, b, left_open=True, right_open=True)

class Union(Set):
    def __init__(self, *values):
        # Do not use the constructor directly;
        # instead use the Python "or" operator.
        assert all(not isinstance(x, Union) for x in values)
        valuesne = [x for x in values if x is not EmptySet]
        assert valuesne
        nominals = [x for x in valuesne if isinstance(x, FiniteNominal)]
        atoms = [x for x in valuesne if isinstance(x, FiniteReal)]
        assert len(nominals) <= 1
        assert len(atoms) <= 1
        self.nominals = nominals[0] if nominals else EmptySet
        self.atoms = atoms[0] if atoms else EmptySet
        self.intervals = frozenset(x for x in valuesne if isinstance(x, Interval))
        # Build the values.
        vals = []
        if nominals:
            vals.append(self.nominals)
        if atoms:
            vals.append(self.atoms)
        for i in self.intervals:
            vals.append(i)
        self.values = frozenset(vals)
        assert 2 <= len(self.values)
        # SymPy compatibility
        self.args = valuesne
    def __contains__(self, x):
        return any(x in v for v in self.values)
    def __eq__(self, x):
        return isinstance(x, Union) \
            and self.values == x.values
    def __hash__(self):
        x = (self.__class__, self.values)
        return hash(x)
    def __repr__(self):
        return 'Union(%s)' % (', '.join(repr(v) for v in self.args))
    def __str__(self):
        return 'Union(%s)' % (', '.join(str(v) for v in self.args))
    def __and__(self, x):
        if x is EmptySet:
            return EmptySet
        if isinstance(x, FiniteNominal):
            return self.nominals & x
        if isinstance(x, (FiniteReal, Interval)):
            atoms = self.atoms & x
            intervals = [i & x for i in self.intervals]
            intervalsne = [i for i in intervals if i is not EmptySet]
            if atoms is EmptySet:
                if not intervalsne:
                    return EmptySet
                if len(intervalsne) == 1:
                    return intervalsne[0]
            if not intervalsne:
                return atoms
            return Union(atoms, *intervalsne)
        if isinstance(x, Union):
            terms = [self & v for v in x.values]
            return reduce(lambda a,b: a |b, terms)
    def __or__(self, x):
        if x is EmptySet:
            return self
        if isinstance(x, FiniteNominal):
            nominals = self.nominals | x
            return Union(nominals, self.atoms, *self.intervals)
        if isinstance(x, FiniteReal):
            atoms = self.atoms | x
            blocks = union_intervals_finite(self.intervals, atoms)
            assert blocks
            if len(blocks) == 1 and self.nominals is EmptySet:
                return blocks[0]
            return Union(self.nominals, *blocks)
        if isinstance(x, Interval):
            intervals = list(self.intervals) + [x]
            blocks = union_intervals_finite(intervals, self.atoms)
            assert blocks
            if len(blocks) == 1 and self.nominals is EmptySet:
                return blocks[0]
            return Union(self.nominals, *blocks)
        if isinstance(x, Union):
            return reduce(lambda a,b: a | b, x.values, self)
        return NotImplemented
    def __invert__(self):
        inversions = [~x for x in self.values]
        return reduce(lambda a,b: a&b, inversions)
    def __iter__(self):
        return iter(self.args)

def union_intervals(intervals):
    intervals_sorted = sorted(intervals, key=lambda i:i.a)
    blocks = [intervals_sorted[0]]
    for interval in intervals_sorted[1:]:
        interval_union = blocks[-1] | interval
        if isinstance(interval_union, Interval):
            blocks[-1] = interval_union
        elif isinstance(interval_union, Union):
            blocks.append(interval)
        else:
            assert False
    return blocks

def union_intervals_finite(intervals, finite):
    if finite is EmptySet:
        return union_intervals(intervals)
    blocks = []
    finite_current = finite
    for interval in intervals:
        interval_union = interval | finite_current
        if isinstance(interval_union, Interval):
            blocks.append(interval_union)
            finite_current = EmptySet
        elif isinstance(interval_union, Union):
            assert interval_union.atoms is not EmptySet
            assert len(interval_union.intervals) == 1
            interval_part = next(iter(interval_union.intervals))
            blocks.append(interval_part)
            finite_current = interval_union.atoms
        else:
            assert False
    blocks_merged = union_intervals(blocks)
    if finite_current is not EmptySet:
        blocks_merged.append(finite_current)
    return blocks_merged

def make_union(*args):
    return reduce(lambda a,b: a|b, args)
def make_intersection(*args):
    return reduce(lambda a,b: a&b, args)

EmptySet = EmptySetC(force=1)
Reals = Interval(-inf, inf)
RealsPos = Interval(0, inf)
RealsNeg = Interval(-inf, 0)
ExtReals = Union(FiniteReal(-inf, inf), Reals)
ExtRealsPos = Union(FiniteReal(inf), RealsPos)
Strings = FiniteNominal(b=True)

# Integral hacks.
Integers = Reals
IntegersPos = Interval.Lopen(0, inf)
IntegersPos0 = RealsPos
def Range(start, stop):
    assert int_or_isinf_neg(start)
    assert int_or_isinf_pos(stop)
    right_open = start == stop
    return Interval(start, stop + right_open, right_open=(start == stop))

# TODO: Expunge.
def convert_sympy(x):
    import sympy
    if x is sympy.S.EmptySet:
        return EmptySet
    if isinstance(x, sympy.Interval):
        left = -inf if x.left == -inf else x.left
        right = inf if x.right == -inf else x.right
        return Interval(left, right, left_open=x.left_open, right_open=x.right_open)
    if isinstance(x, sympy.FiniteSet):
        return FiniteReal(*x.args)
    if isinstance(x, sympy.Union):
        xs = [convert_sympy(v) for v in x.args]
        return make_union(*xs)
    assert False, 'Unknown set in sympy conversion: %s' % (x,)
