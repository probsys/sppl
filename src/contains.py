# Copyright 2019 MIT Probabilistic Computing Project.
# See LICENSE.txt

"""This file is identical to sympy/sets/contains.py except for line 47."""

from __future__ import division
from __future__ import print_function

from sympy.core import S
from sympy.core.relational import Eq
from sympy.core.relational import Ne
from sympy.logic.boolalg import BooleanFunction
from sympy.sets.contains import Contains
from sympy.utilities.misc import func_name

class NotContains(BooleanFunction):
    """
    Asserts that x is not an element of the set S.

    Examples
    ========

    >>> from sympy import Symbol, Integer, S
    >>> NotContains(Integer(2), S.Integers)
    False
    >>> NotContains(Integer(-2), S.Naturals)
    True
    >>> i = Symbol('i', integer=True)
    >>> Contains(i, S.Naturals)
    NotContains(i, Naturals)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Element_%28mathematics%29
    """
    @classmethod
    def eval(cls, x, s):
        from sympy.sets.sets import Set

        if not isinstance(s, Set):
            raise TypeError('expecting Set, not %s' % func_name(s))

        ret = s.contains(x)
        if not isinstance(ret, Contains) \
                and (ret in (S.true, S.false) or isinstance(ret, Set)):
            return not ret

    @property
    def binary_symbols(self):
        return set().union(*[
            i.binary_symbols for i in self.args[1].args
            if i.is_Boolean \
                or i.is_Symbol \
                or isinstance(i, (Eq, Ne))
        ])

    def as_set(self):
        raise NotImplementedError()
