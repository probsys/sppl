# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

import os

from itertools import chain
from math import isinf

import sympy

from sympy.calculus.util import limit

from .sets import EmptySet
from .sets import ExtReals
from .sets import FiniteReal
from .sets import Interval
from .sets import Reals
from .sets import convert_sympy
from .sets import oo
from .sets import make_union
from .sym_util import get_symbols
from .timeout import timeout

TIMEOUT_SYMBOLIC = 5

def get_poly_symbol(expr):
    symbols = tuple(get_symbols(expr))
    assert len(symbols) == 1
    return symbols[0]

# ==============================================================================
# Solving inequalities.

def solve_poly_inequality(expr, b, strict, extended=None):
    # Handle infinite case.
    if isinf(b):
        return solve_poly_inequality_inf(expr, b, strict, extended=extended)
    # Bypass symbolic inference.
    if os.environ.get('SPN_NO_SYMBOLIC'):
        return solve_poly_inequality_numerically(expr, b, strict)
    # Solve symbolically, if possible.
    try:
        with timeout(seconds=TIMEOUT_SYMBOLIC):
            result_symbolic = solve_poly_inequality_symbolically(expr, b, strict)
    except TimeoutError:
        result_symbolic = None
    if result_symbolic is not None:
        if not isinstance(result_symbolic, (sympy.ConditionSet, sympy.Intersection)):
            return convert_sympy(result_symbolic)
    # Solve numerically.
    return solve_poly_inequality_numerically(expr, b, strict)

def solve_poly_inequality_symbolically(expr, b, strict):
    expr = (expr < b) if strict else (expr <= b)
    return sympy.solveset(expr, domain=sympy.Reals)

def solve_poly_inequality_numerically(expr, b, strict):
    poly = expr - b
    symX = get_poly_symbol(expr)
    # Obtain numerical roots.
    roots = sympy.nroots(poly)
    zeros = sorted([r for r in roots if r.is_real])
    if not zeros:
        return sympy.EmptySet
    # Construct intervals around roots.
    mk_intvl = lambda a, b: \
        Interval(a, b, left_open=strict, right_open=strict)
    intervals = list(chain(
        [mk_intvl(-oo, zeros[0])],
        [mk_intvl(x, y) for x, y in zip(zeros, zeros[1:])],
        [mk_intvl(zeros[-1], oo)]))
    # Define probe points.
    xs_probe = list(chain(
        [zeros[0] - 1/2],
        [(i.left + i.right)/2 for i in intervals[1:-1]
            if isinstance(i, Interval)],
        [zeros[-1] + 1/2]))
    # Evaluate poly at the probe points.
    f_xs_probe = [poly.subs(symX, x) for x in xs_probe]
    # Return intervals where poly is less than zero.
    idxs = [i for i, fx in enumerate(f_xs_probe) if fx < 0]
    return make_union(*[intervals[i] for i in idxs])

def solve_poly_inequality_inf(expr, b, strict, extended=None):
    # Minimum value of polynomial is negative infinity.
    assert isinf(b)
    ext = True if extended is None else extended
    if b < 0:
        if strict or not ext:
            return EmptySet
        else:
            return solve_poly_equality_inf(expr, b)
    # Maximum value of polynomial is positive infinity.
    else:
        if strict:
            xinf = solve_poly_equality_inf(expr, -oo) if ext else EmptySet
            return Reals | xinf
        else:
            return ExtReals if ext else Reals

# ==============================================================================
# Solving equalities.

def solve_poly_equality(expr, b):
    # Handle infinite case.
    if isinf(b):
        return solve_poly_equality_inf(expr, b)
    # Bypass symbolic inference.
    if os.environ.get('SPN_NO_SYMBOLIC'):
        return solve_poly_equality_numerically(expr, b)
    # Solve symbolically, if possible.
    try:
        with timeout(seconds=TIMEOUT_SYMBOLIC):
            result_symbolic = solve_poly_equality_symbolically(expr, b)
    except TimeoutError:
        result_symbolic = None
    if result_symbolic is not None:
        if not isinstance(result_symbolic, (sympy.ConditionSet, sympy.Intersection)):
            return convert_sympy(result_symbolic)
    # Solve numerically.
    return solve_poly_equality_numerically(expr, b)

def solve_poly_equality_symbolically(expr, b):
    soln_lt = solve_poly_inequality_symbolically(expr, b, False)
    soln_gt = solve_poly_inequality_symbolically(-expr, -b, False)
    return sympy.Intersection(soln_lt, soln_gt)

def solve_poly_equality_numerically(expr, b):
    roots = sympy.nroots(expr-b)
    zeros = [r for r in roots if r.is_real]
    return FiniteReal(*zeros)

def solve_poly_equality_inf(expr, b):
    assert isinf(b)
    symX = get_poly_symbol(expr)
    val_pos_inf = limit(expr, symX, oo)
    val_neg_inf = limit(expr, symX, -oo)
    check_equal = lambda x: isinf(x) and ((x > 0) if (b > 0) else (x < 0))
    if check_equal(val_pos_inf) and check_equal(val_neg_inf):
        return FiniteReal(oo, -oo)
    if check_equal(val_pos_inf):
        return FiniteReal(oo)
    if check_equal(val_neg_inf):
        return FiniteReal(-oo)
    return EmptySet
