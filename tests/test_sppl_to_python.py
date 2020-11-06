# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

import pytest

from sppl.compilers.sppl_to_python import SPPL_Compiler

isclose = lambda a, b : abs(a-b) < 1e-10

# Test errors in visit_Assign:
# - [x] overwrite variable / array
# - [x] unknown sample target array
# - [x] assigning fresh variables in else
# - [x] non-array variable in for
# - [x] array target must not be subscript
# - [x] assign invalid Python after sampling
# - [x] assign invalid Python non-global
# - [x] invalid left-hand side (tuple, unpacking, etc)

overwrite_cases = [
'X ~= norm(); X ~= bernoulli(p=.5)',
'X = array(5); X ~= bernoulli(p=.5)',
'X ~= bernoulli(p=.5); X = array(5)',
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
        SPPL_Compiler(case)

def test_error_assign_unknown_array():
    source = '''
X ~= uniform(loc=0, scale=1);
Y[0] ~= bernoulli(p=.1)
'''
    with pytest.raises(AssertionError):
        SPPL_Compiler(source)
    SPPL_Compiler('Y = array(10);\n%s' % (source,))

def test_error_assign_fresh_variable_else():
    source = '''
X ~= norm()
if (X > 0):
    Y ~= 2*X
else:
    W ~= 2*X
'''
    with pytest.raises(AssertionError):
        SPPL_Compiler(source)
    SPPL_Compiler(source.replace('W', 'Y'))

def test_error_assign_non_array_in_for():
    source = '''
Y ~= norm();
for i in range(10):
    X ~= norm()
'''
    with pytest.raises(AssertionError):
        SPPL_Compiler(source)
    source_prime = 'X = array(5);\n%s' % (source.replace('X', 'X[i]'),)
    SPPL_Compiler(source_prime)

def test_error_assign_array_subscript():
    source = '''
Y = array(5)
Y[0] = array(2)
'''
    with pytest.raises(AssertionError):
        SPPL_Compiler(source)
    SPPL_Compiler(source.replace('array(2)', 'norm()'))

def test_error_assign_py_constant_after_sampling():
    source = '''
X ~= norm()
Y = "foo"
'''
    with pytest.raises(AssertionError):
        SPPL_Compiler(source)

def test_error_assign_py_constant_non_global():
    source = '''
X ~= norm()
if (X > 0):
    Y = "ali"
else:
    Y = norm()
'''
    with pytest.raises(AssertionError):
        SPPL_Compiler(source)

def test_error_assign_assign_invalid_lhs():
    with pytest.raises(AssertionError):
        SPPL_Compiler('[Y] ~= norm()')
    with pytest.raises(AssertionError):
        SPPL_Compiler('(X, Y) ~= norm()')

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
        SPPL_Compiler(source)
    SPPL_Compiler(source.replace('xrange', 'range'))

def test_error_for_range_step():
    source = '''
X = array(10)
X[0] ~= norm()
for i in range(1, 10, 2):
    X[i] ~= 2*X[i-1]
'''
    with pytest.raises(AssertionError):
        SPPL_Compiler(source)
    SPPL_Compiler(source.replace(', 2)' , ')'))

def test_error_for_range_multiple_vars():
    source = '''
X = array(10)
X[0] ~= norm()
for (i, j) in range(1, 9):
    X[i] ~= 2*X[i-1]
'''
    with pytest.raises(AssertionError):
        SPPL_Compiler(source)
    SPPL_Compiler(source.replace('(i, j)' , 'i'))

# Test errors in visit_If:
# - [x] if without matching else/elif
def test_error_if_no_else():
    source = '''
X ~= norm()
if (X > 0):
    Y ~= 2*X + 1
'''
    with pytest.raises(AssertionError):
        SPPL_Compiler(source)
    SPPL_Compiler('%s\nelse:\n    Y~= norm()' % (source,))
    SPPL_Compiler('%s\nelif (X<0):\n    Y~= norm()' % (source,))

# Test SPPL_Transformer
# Q = Z in {0, 1}         E = Z << {0, 1}
# Q = Z not in {0, 1}     Q = ~(Z << {0, 1})
# Q = Z == 'foo'          Q = Z << {'foo'}
# B = Z != 'foo'          Q = ~(Z << {'foo'})

def test_transform_in():
    source = '''
X ~= choice({'foo': .5, 'bar': .1, 'baz': .4})
Y = X in {'foo', 'baz'}
'''
    compiler = SPPL_Compiler(source)
    py_source = compiler.render_module()
    assert 'X << {\'foo\', \'baz\'}' in py_source
    assert 'X in' not in py_source

def test_transform_in_not():
    source = '''
X ~= choice({'foo': .5, 'bar': .1, 'baz': .4})
Y = X not in {'foo', 'baz'}
'''
    compiler = SPPL_Compiler(source)
    py_source = compiler.render_module()
    assert '~ (X << {\'foo\', \'baz\'})' in py_source
    assert 'not in' not in py_source

def test_transform_eq():
    source = '''
X ~= choice({'foo': .5, 'bar': .1, 'baz': .4})
Y = X == 'foo'
'''
    compiler = SPPL_Compiler(source)
    py_source = compiler.render_module()
    assert 'X << {\'foo\'}' in py_source
    assert '==' not in py_source

def test_transform_eq_not():
    source = '''
X ~= choice({'foo': .5, 'bar': .1, 'baz': .4})
Y = X != 'foo'
'''
    compiler = SPPL_Compiler(source)
    py_source = compiler.render_module()
    assert '~ (X << {\'foo\'})' in py_source
    assert '!=' not in py_source

def test_compile_all_constructs():
    source = '''
X = array(10)
W = array(10)
Y = randint(low=1, high=2)
Z = bernoulli(p=0.1)

E = choice({'1': 0.3, '2': 0.7})


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
    compiler = SPPL_Compiler(source)
    namespace = compiler.execute_module()
    model = namespace.model
    assert isclose(model.prob(namespace.X[5] << {0}), .3)
    assert isclose(model.prob(namespace.X[5] << {-1}), .4)
    assert isclose(model.prob(namespace.X[5] << {3}), .3)
    assert isclose(model.prob(namespace.E << {'1'}), .3)
    assert isclose(model.prob(namespace.E << {'2'}), .7)

def test_imports():
    source = '''
Y ~= bernoulli(p=.5)
Z ~= choice({str(i): Fraction(1, 5) for i in range(5)})
X = array(5)
for i in range(5):
    X[i] ~= Fraction(1,2) * Y
'''
    compiler = SPPL_Compiler(source)
    with pytest.raises(NameError):
        compiler.execute_module()
    compiler = SPPL_Compiler('from fractions import Fraction\n%s' % (source,))
    namespace = compiler.execute_module()
    for i in range(5):
        assert isclose(namespace.model.prob(namespace.Z << {str(i)}), .2)

def test_ifexp():
    source = '''
from fractions import Fraction
Y ~= choice({str(i): Fraction(1, 4) for i in range(4)})
Z ~= (
    atomic(loc=0)    if (Y in {'0', '1'}) else
    atomic(loc=4)    if (Y == '2') else
    atomic(loc=6))
'''
    compiler = SPPL_Compiler(source)
    assert 'IfElse' in compiler.render_module()
    namespace = compiler.execute_module()
    assert isclose(namespace.model.prob(namespace.Z << {0}), .5)
    assert isclose(namespace.model.prob(namespace.Z << {4}), .25)
    assert isclose(namespace.model.prob(namespace.Z << {6}), .25)

def test_switch_shallow():
    source = '''
Y ~= choice({'0': .25, '1': .5, '2': .25})

switch (Y) cases (i in ['0', '1', '2']):
    Z ~= atomic(loc=int(i))
'''
    compiler = SPPL_Compiler(source)
    namespace = compiler.execute_module()
    assert isclose(namespace.model.prob(namespace.Z << {0}), .25)
    assert isclose(namespace.model.prob(namespace.Z << {1}), .5)
    assert isclose(namespace.model.prob(namespace.Z << {2}), .25)

def test_switch_enumerate():
    source = '''
Y ~= choice({'0': .25, '1': .5, '2': .25})

switch (Y) cases (i,j in enumerate(['0', '1', '2'])):
    Z ~= atomic(loc=i+int(j))
'''
    compiler = SPPL_Compiler(source)
    namespace = compiler.execute_module()
    assert isclose(namespace.model.prob(namespace.Z << {0}), .25)
    assert isclose(namespace.model.prob(namespace.Z << {2}), .5)
    assert isclose(namespace.model.prob(namespace.Z << {4}), .25)

def test_switch_nested():
    source = '''
Y ~= randint(low=0, high=4)
W ~= randint(low=0, high=2)

switch (Y) cases (i in range(0, 5)):
    Z ~= choice({str(i): 1})
    switch (W) cases (i in range(0, 2)): V ~= atomic(loc=i)
'''
    compiler = SPPL_Compiler(source)
    namespace = compiler.execute_module()
    assert isclose(namespace.model.prob(namespace.Z << {'0'}), .25)
    assert isclose(namespace.model.prob(namespace.Z << {'1'}), .25)
    assert isclose(namespace.model.prob(namespace.Z << {'2'}), .25)
    assert isclose(namespace.model.prob(namespace.Z << {'3'}), .25)
    assert isclose(namespace.model.prob(namespace.V << {0}), .5)
    assert isclose(namespace.model.prob(namespace.V << {1}), .5)

def test_condition_simple():
    source = '''
Y ~= norm(loc=0, scale=2)
condition((0 < Y) < 2)
Z ~= binom(n=10, p=.2)
condition(Z == 0)
'''
    compiler = SPPL_Compiler(source)
    namespace = compiler.execute_module()
    assert isclose(namespace.model.prob((0 < namespace.Y) < 2), 1)

def test_condition_if():
    source = '''
Y ~= norm(loc=0, scale=2)
if (Y > 1):
    condition(Y < 1)
else:
    condition(Y > -1)
'''
    with pytest.raises(Exception):
        compiler = SPPL_Compiler(source)
        compiler.execute_module()
    compiler = SPPL_Compiler(source.replace('Y > 1', 'Y > 0'))
    namespace = compiler.execute_module()
    assert isclose(namespace.model.prob((-1 < namespace.Y) < 1), 1)

def test_constrain_simple():
    source = '''
Y ~= norm(loc=0, scale=2)
Z ~= binom(n=10, p=.2)
constrain({Z: 10, Y: 0})
'''
    compiler = SPPL_Compiler(source)
    namespace = compiler.execute_module()
    assert isclose(namespace.model.prob((0 < namespace.Y) < 2), 0)
    assert isclose(namespace.model.prob((0 <= namespace.Y) < 2), 1)
    assert isclose(namespace.model.prob(namespace.Z << {10}), 1)

def test_constrain_if():
    source = '''
Y ~= norm(loc=0, scale=2)
if (Y > 1):
    constrain({Y: 2})
else:
    constrain({Y: 0})
'''
    compiler = SPPL_Compiler(source)
    namespace = compiler.execute_module()
    assert isclose(namespace.model.prob(namespace.Y > 0), 0.3085375387259868)
    assert isclose(namespace.model.prob(namespace.Y < 0), 0)
    assert isclose(namespace.model.prob(namespace.Y << {0}), 1-0.3085375387259868)

def test_constant_parameter():
    source = '''
parameters = [1, 2, 3]
n_array = 2
Y = array(n_array + 1)
for i in range(n_array + 1):
    Y[i] = randint(low=parameters[i], high=parameters[i]+1)
'''
    compiler = SPPL_Compiler(source)
    namespace = compiler.execute_module()
    assert isclose(namespace.model.prob(namespace.Y[0] << {1}), 1)
    assert isclose(namespace.model.prob(namespace.Y[1] << {2}), 1)
    assert isclose(namespace.model.prob(namespace.Y[2] << {3}), 1)
    with pytest.raises(AssertionError):
        SPPL_Compiler('%sZ = "foo"\n' % (source,))

def test_error_array_length():
    with pytest.raises(TypeError):
        SPPL_Compiler('Y = array(1.3)').execute_module()
    with pytest.raises(TypeError):
        SPPL_Compiler('Y = array(\'foo\')').execute_module()
    # Length zero array
    namespace = SPPL_Compiler('Y = array(-1)').execute_module()
    assert len(namespace.Y) == 0
