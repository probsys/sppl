# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from collections import OrderedDict
from itertools import chain
from itertools import combinations
from math import isinf

import numpy
import sympy

from sympy.core.relational import Relational

from .sets import FiniteReal
from .sets import Interval

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
    return all(len(s) == len(intersection) for s in sets)

def binspace(start, stop, num=10):
    values = numpy.linspace(start, stop, num)
    bins = list(zip(values[:-1], values[1:]))
    return [Interval(*b) for b in bins]

def powerset(values, start=0):
    s = list(values)
    subsets = (combinations(s, k) for k in range(start, len(s) + 1))
    return chain.from_iterable(subsets)

def partition_list_blocks(values):
    partition = OrderedDict([])
    for i, v in enumerate(values):
        x = hash(v)
        if x not in partition:
            partition[x] = []
        partition[x].append(i)
    return list(partition.values())

def partition_finite_real_contiguous(x):
    # Convert FiniteReal to list of FiniteReal, each with contiguous values.
    assert isinstance(x, FiniteReal)
    values = sorted(x.values)
    blocks = [[values[0]]]
    for y in values[1:]:
        expected = blocks[-1][-1] + 1
        if y == expected:
            blocks[-1].append(y)
        else:
            blocks.append([y])
    return [FiniteReal(*v) for v in blocks]

def sympify_number(x):
    if isinstance(x, (int, float)):
        return x
    msg = 'Expected a numeric term, not %s' % (x,)
    try:
        # String fallback in sympify has been deprecated since SymPy 1.6. Use
        # sympify(str(obj)) or sympy.core.sympify.converter or obj._sympy_
        # instead. See https://github.com/sympy/sympy/issues/18066 for more
        # info.
        sym = sympy.sympify(str(x))
        if not sym.is_number:
            raise TypeError(msg)
        return sym
    except (sympy.SympifyError, AttributeError, TypeError):
        raise TypeError(msg)

def sym_log(x):
    assert 0 <= x
    if x == 0:
        return -float('inf')
    if isinf(x):
        return float('inf')
    return sympy.log(x)

def sympy_solver(expr):
    # Sympy is buggy and slow.  Use Transforms.
    symbols = get_symbols(expr)
    if len(symbols) != 1:
        raise ValueError('Expression "%s" needs exactly one symbol.' % (expr,))

    if isinstance(expr, Relational):
        result = sympy.solveset(expr, domain=sympy.Reals)
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
        result = interval.complement(sympy.Reals)
    else:
        raise ValueError('Expression "%s" has unknown type.' % (expr,))

    if isinstance(result, sympy.ConditionSet):
        raise ValueError('Expression "%s" is not invertible.' % (expr,))

    return result
