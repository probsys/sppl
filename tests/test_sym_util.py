# Copyright 2019 MIT Probabilistic Computing Project.
# See LICENSE.txt

import sympy

from sympy import And
from sympy import Eq
from sympy import Or

from sympy import FiniteSet
from sympy import exp as SymExp
from sympy import log as SymLog

from sum_product_dsl.contains import Contains
from sum_product_dsl.contains import NotContains

from sum_product_dsl.sym_util import get_symbols
from sum_product_dsl.sym_util import simplify_nominal_event

(X0, X1, X2, X3, X4, X5, X6, X7, X8, X9) = sympy.symbols('X:10')

def test_get_symbols():
    syms = get_symbols((X0 > 3) & (X1 < 4))
    assert len(syms) == 2
    assert X0 in syms
    assert X1 in syms

    syms = get_symbols((SymExp(X0) > SymLog(X1)+10) & (X2 < 4))
    assert len(syms) == 3
    assert X0 in syms
    assert X1 in syms
    assert X2 in syms

def test_simplify_nominal_event():
    support = frozenset(range(100))
    snf = simplify_nominal_event

    # Eq.
    assert snf(Eq(X0, 5), support) == {5}
    assert snf(Eq(X0, -5), support) == set()

    V = (4, 1, 10)
    W = (12, -1, 8)
    E1 = FiniteSet(*V)
    E2 = FiniteSet(-5, *V)
    E3 = FiniteSet(-5)
    E4 = FiniteSet(*W)

    # Contains.
    assert snf(Contains(X0, E1), support) == set(V)
    assert snf(Contains(X0, E2), support) == set(V)
    assert snf(Contains(X0, E3), support) == set()

    # NotContains.
    assert snf(NotContains(X0, E1), support) == support.difference(V)
    assert snf(NotContains(X0, E2), support) == support.difference(V)
    assert snf(NotContains(X0, E3), support) == support

    # And
    assert snf(And(Contains(X0, E1), NotContains(X0, E4)), support) == \
        set(V).intersection(support.difference(W))

    # Or
    assert snf(Or(Contains(X0, E1), NotContains(X0, E4)), support) == \
        set(V).union(support.difference(W))
