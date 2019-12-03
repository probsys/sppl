# Copyright 2019 MIT Probabilistic Computing Project.
# See LICENSE.txt

from sympy import And
from sympy import Or
from sympy.core.relational import Relational

from .contains import Containment
from .solver import get_symbols

def factor_dnf(expr):
    symbols = get_symbols(expr)
    lookup = {s:s for s in symbols}
    return factor_dnf_symbols(expr, lookup)

def factor_dnf_symbols(expr, lookup):
    if isinstance(expr, (Relational, Containment)):
        # Literal term.
        symbols = get_symbols(expr)
        if len(symbols) > 1:
            raise ValueError('Expression "%s" has multiple symbols.' % (expr,))
        key = lookup[symbols[0]]
        return {key: expr}

    elif isinstance(expr, And):
        # Product term.
        subexprs = expr.args
        assert all(isinstance(e, (Relational, Containment)) for e in subexprs)
        mappings = [factor_dnf_symbols(subexpr, lookup) for subexpr in  subexprs]
        exprs = {}
        for mapping in mappings:
            assert len(mapping) == 1
            [(key, subexp)] = mapping.items()
            if key not in exprs:
                exprs[key] = subexp
            else:
                exprs[key] = And(subexp, exprs[key])
        return exprs

    elif isinstance(expr, Or):
        # Sum term.
        subexprs = expr.args
        mappings = [factor_dnf_symbols(subexpr, lookup) for subexpr in subexprs]
        exprs = {}
        for mapping in mappings:
            for key, subexp in mapping.items():
                if key not in exprs:
                    exprs[key] = subexp
                else:
                    exprs[key] = Or(subexp, exprs[key])
        return exprs
    else:
        assert False, 'Invalid DNF expression: %s' % (expr,)
