# Copyright 2019 MIT Probabilistic Computing Project.
# See LICENSE.txt

# Custom implementation of contains.py from sympy/sets/contains.py
# 1. Patches bug https://github.com/sympy/sympy/issues/17575
#
# 2. Adds NotContains as well
#     Ideally would use Not(Contains), but error raises
#     venv/lib/python3.5/site-packages/sympy/logic/boolalg.py:923: ValueError
#     about illegal operator.


from __future__ import division
from __future__ import print_function

from sympy.core import S
from sympy.core.logic import Not
from sympy.core.relational import Eq
from sympy.core.relational import Ne
from sympy.logic.boolalg import BooleanFunction
from sympy.logic.boolalg import is_literal
from sympy.sets.sets import Set
from sympy.utilities.misc import func_name

def eval_containment(x, s, negate):
    if not isinstance(s, Set):
        raise TypeError('expecting Set, not %s' % func_name(s))

    ret = s.contains(x)
    if not isinstance(ret, Containment) \
            and (ret in (S.true, S.false) or isinstance(ret, Set)):
        return ret if not negate else (not ret)

class Containment(BooleanFunction):
    # Patches argset bug in logic.boolalg.BooleanFunction._to_nnf
    @classmethod
    def _to_nnf(cls, *args, **kwargs):
        simplify = kwargs.get('simplify', True)
        argset = []
        for arg in args:
            if not is_literal(arg):
                arg = arg.to_nnf(simplify)
            if simplify:
                if isinstance(arg, cls):
                    arg = arg.args
                else:
                    arg = (arg,)
                for a in arg:
                    if Not(a) in argset:
                        return cls.zero
                    argset.append(a)
            else:
                argset.append(arg)
        return cls(*argset)

    @property
    def binary_symbols(self):
        return set().union(*[i.binary_symbols
            for i in self.args[1].args
            if i.is_Boolean or i.is_Symbol or
            isinstance(i, (Eq, Ne))])

    def as_set(self):
        raise NotImplementedError()

class Contains(Containment):
    @classmethod
    def eval(cls, x, s):
        return eval_containment(x, s, False)

class NotContains(Containment):
    @classmethod
    def eval(cls, x, s):
        return eval_containment(x, s, True)
