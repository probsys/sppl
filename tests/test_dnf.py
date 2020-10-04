# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

import pytest

from sppl.dnf import dnf_factor
from sppl.dnf import dnf_non_disjoint_clauses
from sppl.dnf import dnf_to_disjoint_union

from sppl.transforms import EventOr
from sppl.transforms import Exp
from sppl.transforms import Id
from sppl.transforms import Log
from sppl.transforms import Sqrt

(X0, X1, X2, X3, X4, X5) = [Id("X%d" % (i,)) for i in range(6)]

events = [
    X0 < 0,
    (X1<<(0,1)) & (X2<0),
    (X1<0) | (X2<0),
    (X0<0) | ((X1<0) & ~(X2<0)),
    ((X0<0) & ~(0<X2)) | ((X1<0) & ~(X2<0)),
]
@pytest.mark.parametrize('event', events)
def test_to_dnf_no_change(event):
    assert event.to_dnf() == event

def test_to_dnf_changes():
    A = X0 < 0
    B = X1 < 0
    C = X2 < 0
    D = X3 < 100
    E = X4 < 10

    event = A & (B | C)
    event_dnf = event.to_dnf()
    assert len(event_dnf.subexprs) == 2
    assert A & B in event_dnf.subexprs
    assert A & C in event_dnf.subexprs

    event = (A & (B | C)) | D
    event_dnf = event.to_dnf()
    assert len(event_dnf.subexprs) == 3
    assert A & B in event_dnf.subexprs
    assert A & C in event_dnf.subexprs
    assert D in event_dnf.subexprs

    event = (A & (B | C)) | (D & E)
    event_dnf = event.to_dnf()
    assert len(event_dnf.subexprs) == 3
    assert A & B in event_dnf.subexprs
    assert A & C in event_dnf.subexprs
    assert D & E in event_dnf.subexprs

    event = (A | (B | C)) & (D & E)
    event_dnf = event.to_dnf()
    assert A & D & E in event_dnf.subexprs
    assert B & D & E in event_dnf.subexprs
    assert C & D & E in event_dnf.subexprs

def test_to_dnf_invert():
    A = X0 < 0
    B = X1 < 0
    C = X2 < 0
    D = X3 < 0

    event = ~(A | B | C)
    assert event.to_dnf() == ~A & ~B & ~C

    event = ~(A | B | ~C)
    assert event.to_dnf() == ~A & ~B & C

    event = ~((A | B | C) & D)
    #  =>  ~(A | B | C) | ~D
    assert event.to_dnf() == (~A & ~B & ~C) | ~D

    event = ~((A | ~B | C) & ~D)
    #  =>  ~(A | B | C) | ~D
    assert event.to_dnf() == (~A & B & ~C) | D

def test_dnf_factor():
    E00 = Exp(X0) > 0
    E01 = X0 < 10
    E10 = X1 < 10
    E20 = (X2**2 - X2*3) < 0
    E30 = X3 > 10
    E31 = (Sqrt(2*X3)) < 0
    E40 = X4 > 0
    E41 = X4 << [1, 5]
    E50 = 10*Log(X5) + 9 > 5

    event = (E00)
    event_dnf = event.to_dnf()
    dnf = dnf_factor(event_dnf)
    assert len(dnf) == 1
    assert dnf[0][X0] == E00

    event = E00 & E01
    event_dnf = event.to_dnf()
    dnf = dnf_factor(event_dnf)
    assert len(dnf) == 1
    assert dnf[0][X0] == E00 & E01

    event = E00 | E01
    event_dnf = event.to_dnf()
    dnf = dnf_factor(event_dnf)
    assert len(dnf) == 2
    assert dnf[0][X0] == E00
    assert dnf[1][X0] == E01

    event = E00 | (E01 & E10)
    event_dnf = event.to_dnf()
    dnf = dnf_factor(event_dnf, {X0: 0, X1: 0})
    assert len(dnf) == 2
    assert dnf[0][0] == E00
    assert dnf[1][0] == E01 & E10

    event = (E00 & E01 & E10 & E30 & E40) | (E20 & E50 & E31 & ~E41)
    # For the second clause we have:
    #   ~E41 = (-oo, 1) U (1, 5) U (5, oo)
    # so the second clause becomes
    # = (E20 & E50 & E31 & ((-oo, 1) U (1, 5) U (5, oo)))
    # = (E20 & E50 & E31 & (-oo, 1))
    #   or (E20 & E50 & E31 & (1, 5))
    #   or (E20 & E50 & E31 & (5, oo))
    event_dnf = event.to_dnf()
    event_factor = dnf_factor(event_dnf)
    assert len(event_factor) == 4
    # clause 0
    assert len(event_factor[0]) == 4
    assert event_factor[0][X0] == E00 & E01
    assert event_factor[0][X1] == E10
    assert event_factor[0][X3] == E30
    assert event_factor[0][X4] == E40
    # clause 1
    assert len(event_factor[1]) == 4
    assert event_factor[1][X3] == E31
    assert event_factor[1][X2] == E20
    assert event_factor[1][X4] == (X4 < 1)
    assert event_factor[1][X5] == E50
    # clause 2
    assert len(event_factor[2]) == 4
    assert event_factor[2][X3] == E31
    assert event_factor[2][X2] == E20
    assert event_factor[2][X4] == (1 < (X4 < 5))
    assert event_factor[2][X5] == E50
    # clause 3
    assert len(event_factor[3]) == 4
    assert event_factor[3][X3] == E31
    assert event_factor[3][X2] == E20
    assert event_factor[3][X4] == (5 < X4)
    assert event_factor[3][X5] == E50

def test_dnf_factor_1():
    A = Exp(X0) > 0
    B = X0 < 10
    C = X1 < 10
    D = X2 < 0

    event = A & B & C & ~D
    event_dnf = event.to_dnf()
    event_factor = dnf_factor(event_dnf, {X0:0, X1:0, X2:0, X3:1, X4:1, X5:2})
    assert len(event_factor) == 1
    assert event_factor[0][0] == event

def test_dnf_factor_2():
    A = X0 < 1
    B = X4 < 1
    C = X5 < 1
    event = A & B & C
    event_dnf = event.to_dnf()
    event_factor = dnf_factor(event_dnf, {X0:0, X1:0, X2:0, X3:1, X4:1, X5:2})
    assert len(event_factor) == 1
    assert event_factor[0][0] == A
    assert event_factor[0][1] == B
    assert event_factor[0][2] == C

def test_dnf_factor_3():
    A = (Exp(X0) > 0)
    B = X0 < 10
    C = X1 < 10
    D = X4 > 0
    E = (X2**2 - 3*X2) << (0, 10, 100)
    F = (10*Log(X5) + 9) > 5
    G = X4 < 4

    event = (A & B & C & ~D) | (E & F & G)
    event_dnf = event.to_dnf()
    event_factor = dnf_factor(event_dnf, {X0:0, X1:0, X2:0, X3:1, X4:1, X5:2})
    assert len(event_factor) == 2
    assert event_factor[0][0] == A & B & C
    assert event_factor[0][1] == ~D
    assert event_factor[1][0] == E
    assert event_factor[1][1] == G
    assert event_factor[1][2] == F

def test_dnf_non_disjoint_clauses():
    X = Id('X')
    Y = Id('Y')
    Z = Id('Z')

    event = (X > 0) | (Y < 0)
    overlaps = dnf_non_disjoint_clauses(event)
    assert overlaps == {1: [0]}

    event = (X > 0) | ((X < 0) & (Y < 0))
    overlaps = dnf_non_disjoint_clauses(event)
    assert not overlaps

    event = ((X > 0) & (Z < 0)) | ((X < 0) & (Y < 0)) | ((X > 1))
    overlaps = dnf_non_disjoint_clauses(event)
    assert overlaps == {2: [0]}

    event = ((X > 0) & (Z < 0)) | ((X < 0) & (Y < 0)) | ((X > 1) & (Z > 1))
    overlaps = dnf_non_disjoint_clauses(event)
    assert not overlaps

    event = ((X**2 < 9)) | (1 < X)
    overlaps = dnf_non_disjoint_clauses(event)
    assert overlaps == {1: [0]}

    event = ((X**2 < 9) & (0 < X < 1)) | (1 < X)
    overlaps = dnf_non_disjoint_clauses(event)
    assert not overlaps

def test_event_to_disjiont_union_numerical():
    X = Id('X')
    Y = Id('Y')
    Z = Id('Z')

    for event  in [
        (X > 0) | (X < 3),
        (X > 0) | (Y < 3),
        ((X > 0) & (Y < 1)) | ((X < 1) & (Y < 3)) | (Z < 0),
        ((X > 0) & (Y < 1)) | ((X < 1) & (Y < 3)) | (Z < 0) | ~(X <<{1, 3}),
    ]:
        event_dnf = dnf_to_disjoint_union(event)
        assert not dnf_non_disjoint_clauses(event_dnf)

def test_event_to_disjoint_union_nominal():
    X = Id('X')
    Y = Id('Y')
    event = (X << {'1'}) | (X << {'1', '2'})
    assert dnf_to_disjoint_union(event) == X << {'1', '2'}

    event = (X << {'1'}) | ~(Y << {'1'})
    assert dnf_to_disjoint_union(event) == EventOr([
        (X << {'1'}), ~(Y << {'1'}) & ~(X << {'1'})
    ])

def test_event_to_disjoint_union_five():
    # This test case can be visualized as follows:
    # fig, ax = plt.subplots()
    # ax.add_patch(Rectangle((2, 5), 6, 3, fill=False))
    # ax.add_patch(Rectangle((3, 7), 1, 4, fill=False))
    # ax.add_patch(Rectangle((3.5, 2), 2, 7, fill=False))
    # ax.add_patch(Rectangle((4.5, 1), 2, 5, fill=False))
    # ax.add_patch(Rectangle((5, 7), 2, 3, fill=False))
    #
    # // ((2 < X < 8) & (5 < Y < 8))
    # // ((3 < X < 4) & (8 <= Y < 11))
    # // ((3.5 < X < 5.5) & (2 < Y <= 5))
    # // ((5.5 <= X < 6.5) & (1 < Y <= 5))
    # // ((4.5 < X < 5.5) & (1 < Y <= 2))
    # // ((5 < X < 7) & (8 <= Y < 10))
    #
    # fig, ax = plt.subplots()
    # ax.add_patch(Rectangle((2, 5), 6, 3, fill=False))
    # ax.add_patch(Rectangle((3, 8), 1, 3, fill=False))
    # ax.add_patch(Rectangle((3.5, 2), 2, 3, fill=False))
    # ax.add_patch(Rectangle((5.5, 1), 1, 4, fill=False))
    # ax.add_patch(Rectangle((4.5, 1), 1, 1, fill=False))
    # ax.add_patch(Rectangle((5, 8), 2, 2, fill=False))
    X = Id('X')
    Y = Id('Y')
    E1 = ((2 < X) < 8) & ((5 < Y) < 8)
    E2 = ((3 < X) < 4) & ((7 < Y) < 11)
    E3 = ((3.5 < X) < 5.5) & ((2 < Y) < 7)
    E4 = ((4.5 < X) < 6.5) & ((1 < Y) < 6)
    E5 = ((5 < X) < 7) & ((7 < Y) < 10)
    event = E1 | E2 | E3 | E4 | E5
    dnf_to_disjoint_union(event)
