# Copyright 2019 MIT Probabilistic Computing Project.
# See LICENSE.txt

from itertools import chain
from math import isinf

from sympy import ConditionSet
from sympy import FiniteSet
from sympy import Intersection
from sympy import Interval
from sympy import Union
from sympy import nroots
from sympy import oo
from sympy import solveset

from sympy.calculus.util import limit

from .sym_util import get_symbols

from .sym_util import EmptySet
from .sym_util import Reals
from .sym_util import ExtReals

from .timeout import timeout

TIMEOUT_SYMBOLIC = 5

def get_poly_symbol(expr):
    symbols = get_symbols(expr)
    assert len(symbols) == 1
    return symbols[0]

# ==============================================================================
# Solving inequalities.

def solve_poly_inequality(expr, b, strict, extended=None):
    # Handle infinite case.
    if isinf(b):
        return solve_poly_inequality_inf(expr, b, strict, extended=extended)
    # Solve symbolically, if possible.
    try:
        with timeout(seconds=TIMEOUT_SYMBOLIC):
            result_symbolic = solve_poly_inequality_symbolically(expr, b, strict)
            if not isinstance(result_symbolic, ConditionSet):
                return result_symbolic
    except TimeoutError:
        pass
    # Solve numerically.
    return solve_poly_inequality_numerically(expr, b, strict)

def solve_poly_inequality_symbolically(expr, b, strict):
    expr = (expr < b) if strict else (expr <= b)
    return solveset(expr, domain=Reals)

def solve_poly_inequality_numerically(expr, b, strict):
    poly = expr - b
    symX = get_poly_symbol(expr)
    # Obtain numerical roots.
    roots = nroots(poly)
    zeros = sorted([r for r in roots if r.is_real])
    # Construct intervals around roots.
    mk_intvl = lambda a, b: Interval(a, b, left_open=strict, right_open=strict)
    intervals = list(chain(
        [mk_intvl(-oo, zeros[0])],
        [mk_intvl(x, y) for x, y in zip(zeros, zeros[1:])],
        [mk_intvl(zeros[-1], oo)]))
    # Define probe points.
    xs_probe = list(chain(
        [zeros[0] - 1/2],
        [(i.left + i.right)/2 for i in intervals[1:-1]],
        [zeros[-1] + 1/2]))
    # Evaluate poly at the probe points.
    f_xs_probe = [poly.subs(symX, x) for x in xs_probe]
    # Return intervals where poly is less than zero.
    idxs = [i for i, fx in enumerate(f_xs_probe) if fx < 0]
    return Union(*[intervals[i] for i in idxs])

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
            return Union(Reals, xinf)
        else:
            return ExtReals if ext else Reals

# ==============================================================================
# Solving equalities.

def solve_poly_equality(expr, b):
    # Handle infinite case.
    if isinf(b):
        return solve_poly_equality_inf(expr, b)
    # Solve symbolically, if possible.
    try:
        with timeout(seconds=TIMEOUT_SYMBOLIC):
            result_symbolic = solve_poly_equality_symbolically(expr, b)
            if not isinstance(result_symbolic, ConditionSet):
                return result_symbolic
    except TimeoutError:
        pass
    # Solve numerically.
    return solve_poly_equality_numerically(expr, b)

def solve_poly_equality_symbolically(expr, b):
    soln_lt = solve_poly_inequality_symbolically(expr, b, False)
    soln_gt = solve_poly_inequality_symbolically(-expr, -b, False)
    return Intersection(soln_lt, soln_gt)

def solve_poly_equality_numerically(expr, b):
    roots = nroots(expr-b)
    zeros = [r for r in roots if r.is_real]
    return FiniteSet(*zeros)

def solve_poly_equality_inf(expr, b):
    assert isinf(b)
    symX = get_poly_symbol(expr)
    val_pos_inf = limit(expr, symX, oo)
    val_neg_inf = limit(expr, symX, -oo)
    check_equal = lambda x: isinf(x) and ((x > 0) if (b > 0) else (x < 0))
    if check_equal(val_pos_inf) and check_equal(val_neg_inf):
        return FiniteSet(oo, -oo)
    if check_equal(val_pos_inf):
        return FiniteSet(oo)
    if check_equal(val_neg_inf):
        return FiniteSet(-oo)
    return EmptySet