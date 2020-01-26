# Copyright 2019 MIT Probabilistic Computing Project.
# See LICENSE.txt

import sympy

from sympy.core.relational import Relational

from .sym_util import Reals
from .sym_util import get_symbols

# ==============================================================================
# SymPy solver.

def solver(expr):
    symbols = get_symbols(expr)
    if len(symbols) != 1:
        raise ValueError('Expression "%s" needs exactly one symbol.' % (expr,))

    if isinstance(expr, Relational):
        result = sympy.solveset(expr, domain=Reals)
    elif isinstance(expr, sympy.Or):
        subexprs = expr.args
        intervals = [solver(e) for e in subexprs]
        result = sympy.Union(*intervals)
    elif isinstance(expr, sympy.And):
        subexprs = expr.args
        intervals = [solver(e) for e in subexprs]
        result = sympy.Intersection(*intervals)
    elif isinstance(expr, sympy.Not):
        (notexpr,) = expr.args
        interval = solver(notexpr)
        result = interval.complement(Reals)
    else:
        raise ValueError('Expression "%s" has unknown type.' % (expr,))

    if isinstance(result, sympy.ConditionSet):
        raise ValueError('Expression "%s" is not invertible.' % (expr,))

    return result
