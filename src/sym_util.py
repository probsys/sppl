# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

import sympy

from sympy.core.relational import Relational

EmptySet = sympy.S.EmptySet
Infinities = sympy.FiniteSet(-sympy.oo, sympy.oo)

Reals = sympy.S.Reals
RealsPos = sympy.Interval(0, sympy.oo)
RealsNeg = sympy.Interval(-sympy.oo, 0)

ExtReals = Reals + Infinities
ExtRealsPos = RealsPos + sympy.FiniteSet(sympy.oo)
ExtRealsNeg = RealsNeg + sympy.FiniteSet(-sympy.oo)

ContainersFinite = (sympy.FiniteSet, frozenset, set, list, tuple)

def get_symbols(expr):
    atoms = expr.atoms()
    return [a for a in atoms if isinstance(a, sympy.Symbol)]

def get_union(sets):
    return sets[0].union(*sets[1:])

def get_intersection(sets):
    return sets[0].intersection(*sets[1:])

def are_disjoint(sets):
    union = get_union(sets)
    return len(union) == sum(len(s) for s in sets)

def are_identical(sets):
    intersection = get_intersection(sets)
    assert all(len(s) == len(intersection) for s in sets)

def sympify_number(x):
    msg = 'Expected a numeric term, not %s' % (x,)
    try:
        sym = sympy.sympify(x)
        if not sym.is_number:
            raise TypeError(msg)
        return sym
    except (sympy.SympifyError, TypeError):
        raise TypeError(msg)

def sympy_solver(expr):
    # Sympy is buggy and slow.  Use Transforms.
    symbols = get_symbols(expr)
    if len(symbols) != 1:
        raise ValueError('Expression "%s" needs exactly one symbol.' % (expr,))

    if isinstance(expr, Relational):
        result = sympy.solveset(expr, domain=Reals)
    elif isinstance(expr, sympy.Or):
        subexprs = expr.args
        intervals = [sympy_solver(e) for e in subexprs]
        result = sympy.Union(*intervals)
    elif isinstance(expr, sympy.And):
        subexprs = expr.args
        intervals = [sympy_solver(e) for e in subexprs]
        result = sympy.Intersection(*intervals)
    elif isinstance(expr, sympy.Not):
        (notexpr,) = expr.args
        interval = sympy_solver(notexpr)
        result = interval.complement(Reals)
    else:
        raise ValueError('Expression "%s" has unknown type.' % (expr,))

    if isinstance(result, sympy.ConditionSet):
        raise ValueError('Expression "%s" is not invertible.' % (expr,))

    return result
