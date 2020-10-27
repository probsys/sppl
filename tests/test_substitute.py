# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

import pytest

from sppl.transforms import Id
from sppl.transforms import Logarithm

X = Id('X')
Z = Id('Z')
Y = Id('Y')

b = (1, 10)
cases = [
    # Basic cases.
    [X                                   , {}       , X],
    [1/X                                 , {}       , 1/X],
    [X**2+X                              , {}       , X**2+X],
    [X                                   , {}       , X],
    [X                                   , {Z:X}    , X],
    [Z                                   , {Z:2**X} , 2**X],
    [((Z+1)**b)                          , {Z:1/X}  , (1/X+1)**b],
    [((2**Z+1)**b)                       , {Z:X+1}  , (2**(X+1)+1)**b],
    [((2**Logarithm(Z, 2)+1)**b)         , {Z:X+1}  , (2**Logarithm((X+1), 2)+1)**b] ,
    [((2**abs(Logarithm(Z, 2))+1)**b)    , {Z:X+1}  , (2**abs(Logarithm((X+1), 2))+1)**b],
    [((2**abs(Logarithm(1/Z, 2))+1)**b)  , {Z:X+1}  , (2**abs(Logarithm(1/(X+1), 2))+1)**b],
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
        s2 = Id('s2')
        env_prime = {s0: s2, s2: s1}
        assert expr.substitute(env_prime) == expr_prime
