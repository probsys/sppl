# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

import pytest
from spn.transforms import Identity
from spn.transforms import Log

# from spn.transforms import Radical
# from spn.transforms import Exp
# from spn.transforms import Abs
# from spn.transforms import Reciprocal
# from spn.transforms import Poly
# from spn.transforms import Piecewise

# from spn.transforms import EventInterval
# from spn.transforms import EventFiniteReal
# from spn.transforms import EventFiniteNominal
# from spn.transforms import EventOr
# from spn.transforms import EventAnd

X = Identity('X')
Z = Identity('Z')
Y = Identity('Y')

b = (1, 10)
cases = [
    # Basic cases.
    [X                             , {}       , X],
    [1/X                           , {}       , 1/X],
    [X**2+X                        , {}       , X**2+X],
    [X                             , {}       , X],
    [X                             , {Z:X}    , X],
    [Z                             , {Z:2**X} , 2**X],
    [((Z+1)**b)                    , {Z:1/X}  , (1/X+1)**b],
    [((2**Z+1)**b)                 , {Z:X+1}  , (2**(X+1)+1)**b],
    [((2**Log(Z, 2)+1)**b)         , {Z:X+1}  , (2**Log((X+1), 2)+1)**b] ,
    [((2**abs(Log(Z, 2))+1)**b)    , {Z:X+1}  , (2**abs(Log((X+1), 2))+1)**b],
    [((2**abs(Log(1/Z, 2))+1)**b)  , {Z:X+1}  , (2**abs(Log(1/(X+1), 2))+1)**b],
    # Compound cases.
    [(Z > 1)*(1/Z) + (Z < 1)*Z**b
        , {Z:X+1}
        , ((X+1) > 1)*(1/(X+1)) + ((X+1) < 1)*(X+1)**b],
    [(Z << {'a'}) & (Y < 3)
        , {Y:1/X}
        , (Z << {'a'}) & (1/X < 3)],
    [((Y > 1) | Z << {'a'}) & (Y < 3)
        , {Z : Y**2, Y:1/X}
        , (((1/X) > 1) | ((1/X)**2) << {'a'}) & ((1/X) < 3)],
]
@pytest.mark.parametrize('case', cases)
def test_substitute_basic(case):
    (expr, env, expr_prime) = case
    assert expr.substitute(env) == expr_prime

@pytest.mark.parametrize('case', cases)
def test_substitute_transitive(case):
    (expr, env, expr_prime) = case
    if len(env) == 1:
        [(s0, s1)] = env.items()
        s2 = Identity('s2')
        env_prime = {s0: s2, s2: s1}
        assert expr.substitute(env_prime) == expr_prime