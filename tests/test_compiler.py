# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

import pytest

from spn.compiler import SPML_Compiler

# Test errors in visit_Assign:
# - [x] overwrite variable / array
# - [x] unknown sample target array
# - [x] assigning fresh variables in else
# - [x] non-array variable in for
# - [x] array target must not be subscript
# - [x[ array(n) must be a number
# - [x] array(n) must be a positive integer
# - [/] assign invalid right-hand side (list, string, etc.)
# - [x] invalid left-hand side (tuple, unpacking, etc)

overwrite_cases = [
'X ~= norm(); X ~= bernoulli(p=.5)',
'X = array(5); X ~= bernoulli(p=.5)',
'''
X ~= norm()
if (X > 0):
    X ~= norm()
else:
    X ~= norm()
''',
'X = array(5); W ~= norm(); X = array(10)',
]
@pytest.mark.parametrize('case', overwrite_cases)
def test_error_assign_overwrite_variable(case):
    with pytest.raises(AssertionError):
        SPML_Compiler(case)

def test_error_assign_unknown_array():
    source = '''
X ~= uniform(loc=0, scale=1);
Y[0] ~= bernoulli(p=.1)
'''
    with pytest.raises(AssertionError):
        SPML_Compiler(source)
    SPML_Compiler('Y = array(10);\n%s' % (source,))

def test_error_assign_fresh_variable_else():
    source = '''
X ~= norm()
if (X > 0):
    Y ~= 2*X
else:
    W ~= 2*X
'''
    with pytest.raises(AssertionError):
        SPML_Compiler(source)
    SPML_Compiler(source.replace('W', 'Y'))

def test_error_assign_non_array_in_for():
    source = '''
Y ~= norm();
for i in range(10):
    X ~= norm()
'''
    with pytest.raises(AssertionError):
        SPML_Compiler(source)
    source_prime = 'X = array(5);\n%s' % (source.replace('X', 'X[i]'),)
    SPML_Compiler(source_prime)

def test_error_assign_array_subscript():
    source = '''
Y = array(5)
Y[0] = array(2)
'''
    with pytest.raises(AssertionError):
        SPML_Compiler(source)
    SPML_Compiler(source.replace('array(2)', 'norm()'))

def test_error_assign_array_numeral():
    with pytest.raises(AssertionError):
        SPML_Compiler('Y = array(1.3)')
    with pytest.raises(AssertionError):
        SPML_Compiler('Y = array(-1)')
    with pytest.raises(AssertionError):
        SPML_Compiler('Y = array(\'foo\')')

@pytest.mark.xfail(strict=True)
def test_error_assign_assign_invalid_rhs():
    with pytest.raises(AssertionError):
        SPML_Compiler('Y = [norm()]')
    with pytest.raises(AssertionError):
        SPML_Compiler('Y = "foo"')

def test_error_assign_assign_invalid_lhs():
    with pytest.raises(AssertionError):
        SPML_Compiler('[Y] ~= norm()')
    with pytest.raises(AssertionError):
        SPML_Compiler('(X, Y) ~= norm()')

# Test errors in visit_For:
# - [x] func is not range.
# - [x] func range has > 2 arguments
# - [x] more than one iteration variable

def test_error_for_func_not_range():
    source = '''
X = array(10)
for i in xrange(10):
    X[i] ~= norm()
'''
    with pytest.raises(AssertionError):
        SPML_Compiler(source)
    SPML_Compiler(source.replace('xrange', 'range'))

def test_error_for_range_step():
    source = '''
X = array(10)
X[0] ~= norm()
for i in range(1, 10, 2):
    X[i] ~= 2*X[i-1]
'''
    with pytest.raises(AssertionError):
        SPML_Compiler(source)
    SPML_Compiler(source.replace(', 2)' , ')'))

def test_error_for_range_multiple_vars():
    source = '''
X = array(10)
X[0] ~= norm()
for (i, j) in range(1, 9):
    X[i] ~= 2*X[i-1]
'''
    with pytest.raises(AssertionError):
        SPML_Compiler(source)
    SPML_Compiler(source.replace('(i, j)' , 'i'))

# Test errors in visit_If:
# - [x] if without matching else/elif
def test_error_if_no_else():
    source = '''
X ~= norm()
if (X > 0):
    Y ~= 2*X + 1
'''
    with pytest.raises(AssertionError):
        SPML_Compiler(source)
    SPML_Compiler('%s\nelse:\n    Y~= norm()' % (source,))
    SPML_Compiler('%s\nelif (X<0):\n    Y~= norm()' % (source,))

# Test SPML_Transformer
# Q = Z in {0, 1}         E = Z << {0, 1}
# Q = Z not in {0, 1}     Q = ~(Z << {0, 1})
# Q = Z == 'foo'          Q = Z << {'foo'}
# B = Z != 'foo'          Q = ~(Z << {'foo'})

def test_transform_in():
    source = '''
X ~= {'foo': .5, 'bar': .1, 'baz': .4}
Y = X in {'foo', 'baz'}
'''
    compiler = SPML_Compiler(source)
    py_source = compiler.render_module()
    assert 'X << {\'foo\', \'baz\'}' in py_source
    assert 'X in' not in py_source

def test_transform_in_not():
    source = '''
X ~= {'foo': .5, 'bar': .1, 'baz': .4}
Y = X not in {'foo', 'baz'}
'''
    compiler = SPML_Compiler(source)
    py_source = compiler.render_module()
    assert '~ (X << {\'foo\', \'baz\'})' in py_source
    assert 'not in' not in py_source

def test_transform_eq():
    source = '''
X ~= {'foo': .5, 'bar': .1, 'baz': .4}
Y = X == 'foo'
'''
    compiler = SPML_Compiler(source)
    py_source = compiler.render_module()
    assert 'X << {\'foo\'}' in py_source
    assert '==' not in py_source

def test_transform_eq_not():
    source = '''
X ~= {'foo': .5, 'bar': .1, 'baz': .4}
Y = X != 'foo'
'''
    compiler = SPML_Compiler(source)
    py_source = compiler.render_module()
    assert '~ (X << {\'foo\'})' in py_source
    assert '!=' not in py_source

def test_compile_all_constructs():
    source = '''
X = array(10)
W = array(10)
Y = randint(low=1, high=2)
Z = bernoulli(p=0.1)

E = {'1': 0.3, '2': 0.7}


for i in range(1,5):
    W[i] = uniform(loc=0, scale=2)
    X[i] = bernoulli(p=0.5)

X[0] ~= gamma(a=1)
H = (X[0]**2 + 2*X[0] + 3)**(1, 10)

# Here is a comment, with indentation on next line
    


X[5] ~= 0.3*atomic(loc=0) | 0.4*atomic(loc=-1) | 0.3*atomic(loc=3)
if X[5] == 0:
    X[7] = bernoulli(p=0.1)
    X[8] = 1 + X[3]
elif (X[5]**2 == 1):
    X[7] = bernoulli(p=0.2)
    X[8] = 1 + X[3]
else:
    if (X[3] in {0, 1}):
        X[7] = bernoulli(p=0.2)
        X[8] = 1 + X[3]
    else:
        X[7] = bernoulli(p=0.2)
        X[8] = 1 + X[3]
'''
    compiler = SPML_Compiler(source)
    namespace = compiler.execute_module()
    model = namespace.model
    assert model.prob(namespace.X[5] << {0}) == 0.3
    assert model.prob(namespace.X[5] << {-1}) == 0.4
    assert model.prob(namespace.X[5] << {3}) == 0.3
    assert model.prob(namespace.E << {'1'}) == 0.3
    assert model.prob(namespace.E << {'2'}) == 0.7

def test_imports():
    source = '''
Y ~= bernoulli(p=.5)
Z ~= {str(i): Fraction(1, 5) for i in range(5)}
X = array(5)
for i in range(5):
    X[i] ~= Fraction(1,2) * Y
'''
    compiler = SPML_Compiler(source)
    with pytest.raises(NameError):
        compiler.execute_module()
    compiler = SPML_Compiler('from fractions import Fraction\n%s' % (source,))
    namespace = compiler.execute_module()
    for i in range(5):
        assert namespace.model.prob(namespace.Z << {str(i)}) == 0.2

def test_ifexp():
    source = '''
from fractions import Fraction
Y ~= {str(i): Fraction(1, 4) for i in range(4)}
Z ~= (
    atomic(loc=0)    if (Y in {'0', '1'}) else
    atomic(loc=4)    if (Y == '2') else
    atomic(loc=6))
'''
    compiler = SPML_Compiler(source)
    assert 'IfElse' in compiler.render_module()
    namespace = compiler.execute_module()
    assert namespace.model.prob(namespace.Z << {0}) == 0.5
    assert namespace.model.prob(namespace.Z << {4}) == 0.25
    assert namespace.model.prob(namespace.Z << {6}) == 0.25

def test_switch_in():
    source = '''
Y ~= {'0': .25, '1': .5, '2': .25}

switch (Y) cases (i in ['0', '1', '2']):
    Z ~= atomic(loc=int(i))
'''
    compiler = SPML_Compiler(source)
    namespace = compiler.execute_module()
    assert namespace.model.prob(namespace.Z << {0}) == .25
    assert namespace.model.prob(namespace.Z << {1}) == .5
    assert namespace.model.prob(namespace.Z << {2}) == .25

def test_switch_lte():
    source = '''
Y ~= randint(low=0, high=4)
W ~= randint(low=0, high=2)

switch (Y) cases (i <= range(0, 5)):
    Z ~= {str(i): 1}
    switch (W) cases (i <= range(0, 2)): V ~= atomic(loc=i)
'''
    compiler = SPML_Compiler(source)
    namespace = compiler.execute_module()
    assert abs(namespace.model.prob(namespace.Z << {'0'}) - .25) < 1e-10
    assert abs(namespace.model.prob(namespace.Z << {'1'}) - .25) < 1e-10
    assert abs(namespace.model.prob(namespace.Z << {'2'}) - .25) < 1e-10
    assert abs(namespace.model.prob(namespace.Z << {'3'}) - .25) < 1e-10
    assert abs(namespace.model.prob(namespace.V << {0}) - .5) < 1e-10
    assert abs(namespace.model.prob(namespace.V << {1}) - .5) < 1e-10
