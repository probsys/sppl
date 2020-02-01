# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

import sympy

from sympy import exp as SymExp
from sympy import log as SymLog

from sum_product_dsl.sym_util import get_symbols

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
