# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

import pytest

from sppl.sets import EmptySet
from sppl.sets import FiniteNominal as FN
from sppl.sets import FiniteReal as FR
from sppl.sets import Interval
from sppl.sets import Union
from sppl.sets import inf
from sppl.sets import union_intervals
from sppl.sets import union_intervals_finite

def test_FiniteNominal_in():
    with pytest.raises(Exception):
        FN()
    assert 'a' in FN('a')
    assert 'b' not in FN('a')
    assert 'b' in FN('a', b=True)
    assert 'a' not in FN('a', b=True)
    assert 'a' in FN(b=True)

def test_FiniteNominal_invert():
    assert ~(FN('a')) == FN('a', b=True)
    assert ~(FN('a', b=True)) == FN('a')
    assert ~FN(b=True) == EmptySet

def test_FiniteNominal_and():
    assert FN('a','b') & EmptySet is EmptySet
    assert FN('a','b') & FN('c') is EmptySet
    assert FN('a','b','c') & FN('a') == FN('a')
    assert FN('a','b','c') & FN(b=True) == FN('a','b','c')
    assert FN('a','b','c') & FN('a') == FN('a')
    assert FN('a','b','c', b=True) & FN('a') is EmptySet
    assert FN('a','b','c', b=True) & FN('d','a','b') == FN('d')
    assert FN('a','b','c', b=True) & FN('d') == FN('d')
    assert FN('a','b','c') & FN('a', b=True) == FN('b','c')
    assert FN('a','b','c') & FN('d','a','b', b=True) == FN('c')
    assert FN('a','b','c') & FN('d', b=True) == FN('a','b','c')
    assert FN('a','b','c', b=True) & FN('d', b=True) == FN('a','b','c','d', b=True)
    assert FN('a','b','c', b=True) & FN('a', b=True) == FN('a','b','c', b=True)
    assert FN(b=True) & FN(b=True) == FN(b=True)
    # FiniteReal
    assert FN('a') & FR(1) is EmptySet
    assert FN('a', b=True) & FR(1) is EmptySet
    # Interval
    assert FN('a') & Interval(0,1) is EmptySet

def test_FiniteNominal_or():
    # EmptySet
    assert FN('a','b') | EmptySet == FN('a','b')
    # Nominal
    assert FN('a','b') | FN('c') == FN('a','b','c')
    assert FN('a','b','c') | FN('a') == FN('a','b','c')
    assert FN('a','b','c') | FN(b=True) == FN(b=True)
    assert FN('a','b','c', b=True) | FN('a') == FN('b','c', b=True)
    assert FN('a','b','c', b=True) | FN('d','a','b') == FN('c', b=True)
    assert FN('a','b','c', b=True) | FN('d') == FN('a','b','c', b=True)
    assert FN('a','b','c') | FN('a', b=True) == FN(b=True)
    assert FN('a','b','c') | FN('d','a','b', b=True) == FN('d', b=True)
    assert FN('a','b','c') | FN('d', b=True) == FN('d', b=True)
    assert FN('a','b','c', b=True) | FN('d', b=True) == FN(b=True)
    assert FN('a','b','c', b=True) | FN('a', b=True) == FN('a', b=True)
    assert FN(b=True) | FN(b=True) == FN(b=True)
    # FiniteReal
    assert FN('a') | FR(1) == Union(FN('a'), FR(1))
    assert FN('a', b=True) | FR(1) == Union(FR(1), FN('a', b=True))
    # Interval
    assert FN('a') | Interval(0,1) == Union(FN('a'), Interval(0,1))

def test_FiniteReal_in():
    with pytest.raises(Exception):
        FR()
    assert 1 in FN(1,2)
    assert 2 not in FN(1)

def test_FiniteReal_invert():
    assert ~FR(1) == Union(
        Interval.Ropen(-inf,1),
        Interval.Lopen(1, inf))
    assert ~(FR(0,-1)) == Union(
        Interval.Ropen(-inf,-1),
        Interval.open(-1, 0),
        Interval.Lopen(0, inf))

def test_FiniteReal_and():
    assert FR(1) & FR(2) is EmptySet
    assert FR(1,2) & FR(2) == FR(2)
    assert FR(1,2) & FN('2') is EmptySet
    assert FR(1,2) & FN('2', b=True) is EmptySet
    assert FR(1,2) & Interval(0,1) == FR(1)
    assert FR(1,2) & Interval.Ropen(0,1) is EmptySet
    assert FR(0,2) & Interval(0,1) == FR(0)
    assert FR(0,2) & Interval.Lopen(0,1) is EmptySet
    assert FR(-1,1) & Interval(-10,10) == FR(-1,1)
    assert FR(-1,11) & Interval(-10,10) == FR(-1)

def test_FiniteReal_or():
    assert FR(1) | FR(2) == FR(1,2)
    assert FR(1,2) | FR(2) == FR(1,2)
    assert FR(1,2) | FN('2') == Union(FR(1,2), FN('2'))
    assert FR(1,2) | Interval(0,1) == Union(Interval(0,1), FR(2))
    assert FR(1,2) | Interval.Ropen(0,1) == Union(Interval(0,1), FR(2))
    assert FR(0,1,2) | Interval.open(0,1) == Union(Interval(0,1), FR(2))
    assert FR(0,2) | Interval.Lopen(0,1) == Union(Interval(0,1), FR(2))
    assert FR(0,2) | Interval.Lopen(2.5,10) == Union(Interval.Lopen(2.5,10), FR(0,2))
    assert FR(-1,1) | Interval(-10,10) == Interval(-10,10)
    assert FR(-1,11) | Interval(-10,10) == Union(Interval(-10, 10), FR(11))

def test_Interval_in():
    with pytest.raises(Exception):
        Interval(3, 1)
    assert 1 in Interval(0,1)
    assert 1 not in Interval.Ropen(0,1)
    assert 1 in Interval.Lopen(0,1)
    assert 0 in Interval(0,1)
    assert 0 not in Interval.Lopen(0,1)
    assert 0 in Interval.Ropen(0,1)
    assert inf not in Interval(-inf, inf)
    assert -inf not in Interval(-inf, 0)
    assert 10 in Interval(-inf, inf)

def test_Interval_invert():
    assert ~(Interval(0,1)) == Union(Interval.Ropen(-inf, 0), Interval.Lopen(1, inf))
    assert ~(Interval.open(0,1)) == Union(Interval(-inf, 0), Interval(1, inf))
    assert ~(Interval.Lopen(0,1)) == Union(Interval(-inf, 0), Interval.Lopen(1, inf))
    assert ~(Interval.Ropen(0,1)) == Union(Interval.Ropen(-inf, 0), Interval(1, inf))
    assert ~(Interval(-inf, inf)) is EmptySet
    assert ~(Interval(3, inf)) == Interval.Ropen(-inf, 3)
    assert ~(Interval.open(3, inf)) == Interval(-inf, 3)
    assert ~(Interval.Lopen(3, inf)) == Interval(-inf, 3)
    assert ~(Interval.Ropen(3, inf)) == Interval.Ropen(-inf, 3)
    assert ~(Interval(-inf, 3)) == Interval.Lopen(3, inf)
    assert ~(Interval.open(-inf, 3)) == Interval(3, inf)
    assert ~(Interval.Lopen(-inf, 3)) == Interval.Lopen(3, inf)
    assert ~(Interval.Ropen(-inf, 3)) == Interval(3, inf)
    assert ~(Interval.open(-inf, inf)) is EmptySet

def test_Interval_and():
    assert Interval(0,1) & Interval(-1,1) == Interval(0,1)
    assert Interval(0,2) & Interval.open(0,1) == Interval.open(0,1)
    assert Interval.Lopen(0,1) & Interval(-1,1) == Interval.Lopen(0,1)
    assert Interval.Ropen(0,1) & Interval(-1,1) == Interval.Ropen(0,1)
    assert Interval(0,1) & Interval(1,2) == FR(1)
    assert Interval.Lopen(0,1) & Interval(1,2) == FR(1)
    assert Interval.Ropen(0,1) & Interval(1,2) is EmptySet
    assert Interval.Lopen(0,1) & Interval.Lopen(1,2) is EmptySet
    assert Interval(1,2) & Interval.Lopen(0,1) == FR(1)
    assert Interval(1,2) & Interval.open(0,1) is EmptySet
    assert Interval(0,2) & Interval.Lopen(0.5,2.5) == Interval.Lopen(0.5,2)
    assert Interval.Ropen(0,2) & Interval.Lopen(0.5,2.5) == Interval.open(0.5,2)
    assert Interval.open(0,2) & Interval(0.5,2.5) == Interval.Ropen(0.5,2)
    assert Interval.Lopen(0,2) & Interval.Ropen(0,2) == Interval.open(0,2)
    assert Interval(0,1) & Interval(2,3) is EmptySet
    assert Interval(2,3) & Interval(0,1) is EmptySet
    assert Interval.open(0,1) & Interval.open(0,1) == Interval.open(0,1)
    assert Interval.Ropen(-inf, -3) & Interval(-inf, inf) == Interval.Ropen(-inf, -3)
    assert Interval(-inf, inf) & Interval.Ropen(-inf, -3) == Interval.Ropen(-inf, -3)
    assert Interval(0, inf) & (Interval.Lopen(-5, inf)) == Interval(0, inf)
    assert Interval.Lopen(0, 1) & Interval.Ropen(0, 1) == Interval.open(0, 1)
    assert Interval.Ropen(0, 1) & Interval.Lopen(0, 1) == Interval.open(0, 1)
    assert Interval.Ropen(0, 5) & Interval.Ropen(-inf, 5) == Interval.Ropen(0, 5)

def test_Interval_or():
    assert Interval(0,1) | Interval(-1,1) == Interval(-1,1)
    assert Interval(0, 2) | Interval.open(0, 1) == Interval(0, 2)
    assert Interval.Lopen(0,1) | Interval(-1,1) == Interval(-1,1)
    assert Interval.Ropen(0,1) | Interval(-1,1) == Interval(-1,1)
    assert Interval(0,1) | Interval(1, 2) == Interval(0, 2)
    assert Interval.open(0, 1) | Interval(0,1) == Interval(0, 1)
    assert Interval(0, 1) | Interval(0,.5) == Interval(0, 1)
    assert Interval.Lopen(0, 1) | Interval(1,2) == Interval.Lopen(0, 2)
    assert Interval.Ropen(-1, 0) | Interval.Ropen(0, 1) == Interval.Ropen(-1,1)
    assert Interval.Ropen(0, 1) | Interval.Ropen(-1, 0) == Interval.Ropen(-1,1)
    assert Interval.Lopen(0, 1) | Interval.Lopen(1, 2) == Interval.Lopen(0,2)
    assert Interval.Lopen(0, 1) | Interval.Ropen(1, 2) == Interval.open(0,2)
    assert Interval.open(0, 2) | Interval(0, 1) == Interval.Ropen(0, 2)
    assert Interval.open(0, 1) | Interval.Ropen(-1, 0) == Union(Interval.open(0, 1), Interval.Ropen(-1,0))
    assert Interval(1, 2) | Interval.Ropen(0, 1) == Interval(0,2)
    assert Interval.Ropen(0, 1) | Interval(1, 2) == Interval(0,2)
    assert Interval.open(0, 1) | Interval(1, 2) == Interval.Lopen(0,2)
    assert Interval.Ropen(0, 1) | Interval.Ropen(1, 2) == Interval.Ropen(0,2)
    assert Interval(1, 2) | Interval.open(0, 1) == Interval.Lopen(0,2)
    assert Interval(1, 2) | Interval.Lopen(0, 1) == Interval.Lopen(0,2)
    assert Interval(1, 2) | Interval.Ropen(0, 1) == Interval(0,2)
    assert Interval.open(0,1) | Interval(1,2) == Interval.Lopen(0,2)
    assert Interval.Lopen(0,1) | Interval(1,2) == Interval.Lopen(0,2)
    assert Interval.Ropen(0,1) | Interval(1,2) == Interval(0,2)
    assert Interval(0,2) | Interval.open(0.5, 2.5) == Interval.Ropen(0, 2.5)
    assert Interval.open(0,2) | Interval.open(0, 2.5) == Interval.open(0, 2.5)
    assert Interval.open(0,2.5) | Interval.open(0, 2) == Interval.open(0, 2.5)
    assert Interval.open(0,1) | Interval.open(1, 2) == Union(Interval.open(0, 1), Interval.open(1,2))
    assert Interval.Ropen(0,2) | Interval.Lopen(0.5, 2.5) == Interval(0, 2.5)
    assert Interval.open(0,2) | Interval(0.5,2.5) == Interval.Lopen(0, 2.5)
    assert Interval.Lopen(0,2) | Interval.Ropen(0,2) == Interval(0,2)
    assert Interval(0,1) | Interval(2,3) == Union(Interval(0,1), Interval(2,3))
    assert Interval(2,3) | Interval.Ropen(0,1) == Union(Interval(2,3), Interval.Ropen(0,1))
    assert Interval.Lopen(0,1) | Interval(1,2) == Interval.Lopen(0,2)
    assert Interval(-10,10) | FR(-1,1) == Interval(-10,10)
    assert Interval(-10,10) | FR(-1,11) == Union(Interval(-10, 10), FR(11))
    assert Interval(-inf, -3, right_open=True) | Interval(-inf, inf) == Interval(-inf, inf)

def test_union_intervals():
    assert union_intervals([
        Interval(0,1),
        Interval(2,3),
        Interval(1,2)
    ]) == [Interval(0,3)]
    assert union_intervals([
        Interval.open(0,1),
        Interval(2,3),
        Interval(1,2)
    ]) == [Interval.Lopen(0,3)]
    assert union_intervals([
        Interval.open(0,1),
        Interval(2,3),
        Interval.Lopen(1,2)
    ]) == [Interval.open(0,1), Interval.Lopen(1,3)]
    assert union_intervals([
        Interval.open(0,1),
        Interval.Ropen(0,3),
        Interval.Lopen(1,2)
    ]) == [Interval.Ropen(0,3)]
    assert union_intervals([
        Interval.open(-2,-1),
        Interval.Ropen(0,3),
        Interval.Lopen(1,2)
    ]) == [Interval.open(-2,-1), Interval.Ropen(0,3)]

def test_union_intervals_finite():
    assert union_intervals_finite([
            Interval.open(0,1),
            Interval(2,3),
            Interval.Lopen(1,2)
        ], FR(1)) \
        == [Interval.Lopen(0, 3)]
    assert union_intervals_finite([
            Interval.open(0,1),
            Interval.open(2, 3),
            Interval.open(1,2)
        ], FR(1, 3)) \
        == [Interval.open(0, 2), Interval.Lopen(2, 3)]
    assert union_intervals_finite([
            Interval.open(0,1),
            Interval.open(1, 3),
            Interval.open(11,15)
        ], FR(1, -11, -19, 3)) \
        == [Interval.Lopen(0, 3), Interval.open(11,15), FR(-11, -19)]

def test_Union_or():
    x = Interval(0,1) | Interval(5,6) | Interval(10,11)
    assert x == Union(Interval(0,1), Interval(5,6), Interval(10,11))
    x = Interval.Ropen(0,1) | Interval.Lopen(1,2) | Interval(10,11)
    assert x == Union(Interval.Ropen(0,1), Interval.Lopen(1,2), Interval(10,11))
    x = Interval.Ropen(0,1) | Interval.Lopen(1,2) | Interval(10,11) | FR(1)
    assert x == Union(Interval(0,2), Interval(10,11))
    x = (Interval.Ropen(0,1) | Interval.Lopen(1,2)) | (Interval(10,11) | FR(1))
    assert x == Union(Interval(0,2), Interval(10,11))
    x = FR(1) | ((Interval.Ropen(0,1) | Interval.Lopen(1,2) | FR(10,13)) \
            | (Interval.Lopen(10,11) | FR(7)))
    assert x == Union(Interval(0,2), Interval(10,11), FR(13, 7))
    assert 2 in x
    assert 13 in x
    x = FN('f') | (FR(1) | FN('g', b=True))
    assert x == Union(FR(1), FN('g', b=True))
    assert 'w' in x
    assert 'g' not in x

def test_Union_and():
    x = (Interval(0,1) | FR(1)) & (FN('a'))
    assert x is EmptySet
    x = (FN('x', b=True)| Interval(0,1) | FR(1)) & (FN('a'))
    assert x == FN('a')
    x = (FN('x')| Interval(0,1) | FR(1)) & (FN('a'))
    assert x is EmptySet
    x = (FN('x')| Interval.open(0,1) | FR(7)) & ((FN('x')) | FR(.5) | Interval(.75, 1.2))
    assert x == Union(FR(.5), FN('x'), Interval.Ropen(.75, 1))
    x = (FN('x')| Interval.open(0,1) | FR(7)) & (FR(3))
    assert x is EmptySet
    x = (Interval.Lopen(-5, inf)) & (Interval(0, inf) | FR(inf))
    assert x == Interval(0, inf)
    x = (FR(1,2) | Interval.Ropen(-inf, 0)) & Interval(0, inf)
    assert x == FR(1,2)
    x = (FR(1,12) | Interval(0, 5) | Interval(7,10)) & Interval(4, 12)
    assert x == Union(Interval(4,5), Interval(7,10), FR(12))
