# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

import pytest

from sympy import exp as SymExp
from sympy import log as SymLog
from sympy import symbols

from sppl.sets import FiniteReal
from sppl.sym_util import get_symbols
from sppl.sym_util import partition_finite_real_contiguous
from sppl.sym_util import partition_list_blocks

(X0, X1, X2, X3, X4, X5, X6, X7, X8, X9) = symbols('X:10')

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

@pytest.mark.parametrize('a, b', [
    ([0, 1, 2, 3], [[0], [1], [2], [3]]),
    ([0, 1, 2, 1], [[0], [1, 3], [2]]),
    (['0', '0', 2, '0'], [[0, 1, 3], [2]]),
])
def test_partition_list_blocks(a, b):
    solution = partition_list_blocks(a)
    assert solution == b

@pytest.mark.parametrize('a, b', [
    (FiniteReal(0,1,2), [FiniteReal(0,1,2)]),
    (FiniteReal(0,3,1,2), [FiniteReal(0,1,2,3)]),
    (FiniteReal(-1,3,1,2), [FiniteReal(-1), FiniteReal(1,2,3)]),
    (FiniteReal(-1,3,1,2,-2,-7), [FiniteReal(-7), FiniteReal(-1,-2), FiniteReal(1,2,3)]),
    (FiniteReal(-1,3,1,2,-2,-7,0), [FiniteReal(-7), FiniteReal(-2,-1,0,1,2,3)]),
])
def test_parition_finite_real_contiguous(a, b):
    solution = partition_finite_real_contiguous(a)
    assert solution == b
