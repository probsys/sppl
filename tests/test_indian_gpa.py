# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

'''
Indian GPA example from:

Discrete-Continuous Mixtures in Probabilistic Programming: Generalized
Semantics and Inference Algorithms, Wu et. al., ICML 2018.
https://arxiv.org/pdf/1806.02027.pdf
'''

import pytest

from sppl.compilers.ast_to_spn import Id
from sppl.compilers.ast_to_spn import IfElse
from sppl.compilers.ast_to_spn import Sample
from sppl.compilers.ast_to_spn import Sequence
from sppl.compilers.sppl_to_python import SPPL_Compiler
from sppl.distributions import atomic
from sppl.distributions import choice
from sppl.distributions import uniform
from sppl.math_util import allclose
from sppl.sets import Interval
from sppl.spn import ExposedSumSPN

Nationality = Id('Nationality')
Perfect     = Id('Perfect')
GPA         = Id('GPA')

def model_no_latents():
    return \
        0.5 * ( # American student
            0.99 * (GPA >> uniform(loc=0, scale=4)) | \
            0.01 * (GPA >> atomic(loc=4))) | \
        0.5 * ( # Indian student
            0.99 * (GPA >> uniform(loc=0, scale=10)) | \
            0.01 * (GPA >> atomic(loc=10)))

def model_exposed():
    return ExposedSumSPN(
        spn_weights=(Nationality >> choice({'India': 0.5, 'USA': 0.5})),
        children={
            # American student.
            'USA': ExposedSumSPN(
                spn_weights=(Perfect >> choice({'True': 0.01, 'False': 0.99})),
                children={
                    'False'   : GPA >> uniform(loc=0, scale=4),
                    'True'    : GPA >> atomic(loc=4),
                }),
            # Indian student.
            'India': ExposedSumSPN(
                spn_weights=(Perfect >> choice({'True': 0.01, 'False': 0.99})),
                children={
                    'False'   : GPA >> uniform(loc=0, scale=10),
                    'True'    : GPA >> atomic(loc=10),
                })},
        )

def model_ifelse_exhuastive():
    command = Sequence(
        Sample(Nationality, choice({'India': 0.5, 'USA': 0.5})),
        Sample(Perfect,     choice({'True': 0.01, 'False': 0.99})),
        IfElse(
            (Nationality << {'India'}) & (Perfect << {'False'}),
                Sample(GPA, uniform(loc=0, scale=10))
            ,
            (Nationality << {'India'}) & (Perfect << {'True'}),
                Sample(GPA, atomic(loc=10))
            ,
            (Nationality << {'USA'}) & (Perfect << {'False'}),
                Sample(GPA, uniform(loc=0, scale=4))
            ,
            (Nationality << {'USA'}) & (Perfect << {'True'}),
                Sample(GPA, atomic(loc=4))))
    return command.interpret()

def model_ifelse_non_exhuastive():
    Nationality = Id('Nationality')
    Perfect     = Id('Perfect')
    GPA         = Id('GPA')
    command = Sequence(
        Sample(Nationality, choice({'India': 0.5, 'USA': 0.5})),
        Sample(Perfect,     choice({'True': 0.01, 'False': 0.99})),
        IfElse(
            (Nationality << {'India'}) & (Perfect << {'False'}),
                Sample(GPA, uniform(loc=0, scale=10))
            ,
            (Nationality << {'India'}) & (Perfect << {'True'}),
                Sample(GPA, atomic(loc=10))
            ,
            (Nationality << {'USA'}) & (Perfect << {'False'}),
                Sample(GPA, uniform(loc=0, scale=4))
            ,
            True,
                Sample(GPA, atomic(loc=4))))
    return command.interpret()

def model_ifelse_nested():
    Nationality = Id('Nationality')
    Perfect     = Id('Perfect')
    GPA         = Id('GPA')
    command = Sequence(
        Sample(Nationality, choice({'India': 0.5, 'USA': 0.5})),
        Sample(Perfect,     choice({'True': 0.01, 'False': 0.99})),
        IfElse(
            Nationality << {'India'},
                IfElse(
                    Perfect << {'True'},    Sample(GPA, atomic(loc=10)),
                    Perfect << {'False'},   Sample(GPA, uniform(scale=10)),
                ),
            Nationality << {'USA'},
                IfElse(
                    Perfect << {'True'},    Sample(GPA, atomic(loc=4)),
                    Perfect << {'False'},   Sample(GPA, uniform(scale=4)),
                )))
    return command.interpret()

def model_perfect_nested():
    Nationality = Id('Nationality')
    Perfect     = Id('Perfect')
    GPA         = Id('GPA')
    command = Sequence(
        Sample(Nationality, choice({'India': 0.5, 'USA': 0.5})),
        IfElse(
            Nationality << {'India'}, Sequence(
                    Sample(Perfect, choice({'True': 0.01, 'False': 0.99})),
                    IfElse(
                        Perfect << {'True'},    Sample(GPA, atomic(loc=10)),
                        True,                   Sample(GPA, uniform(scale=10))
                    )),
            Nationality << {'USA'}, Sequence(
                Sample(Perfect, choice({'True': 0.01, 'False': 0.99})),
                IfElse(
                    Perfect << {'True'},    Sample(GPA, atomic(loc=4)),
                    True,                   Sample(GPA, uniform(scale=4)),
                ))))
    return command.interpret()

def model_ifelse_exhuastive_compiled():
    compiler = SPPL_Compiler('''
Nationality   ~= choice({'India': 0.5, 'USA': 0.5})
Perfect       ~= choice({'True': 0.01, 'False': 0.99})
if (Nationality == 'India') & (Perfect == 'False'):
    GPA ~= uniform(loc=0, scale=10)
elif (Nationality == 'India') & (Perfect == 'True'):
    GPA ~= atomic(loc=10)
elif (Nationality == 'USA') & (Perfect == 'False'):
    GPA ~= uniform(loc=0, scale=4)
elif (Nationality == 'USA') & (Perfect == 'True'):
    GPA ~= atomic(loc=4)
    ''')
    namespace = compiler.execute_module()
    return namespace.model

def model_ifelse_non_exhuastive_compiled():
    compiler = SPPL_Compiler('''
Nationality   ~= choice({'India': 0.5, 'USA': 0.5})
Perfect       ~= choice({'True': 0.01, 'False': 0.99})
if (Nationality == 'India') & (Perfect == 'False'):
    GPA ~= uniform(loc=0, scale=10)
elif (Nationality == 'India') & (Perfect == 'True'):
    GPA ~= atomic(loc=10)
elif (Nationality == 'USA') & (Perfect == 'False'):
    GPA ~= uniform(loc=0, scale=4)
else:
    GPA ~= atomic(loc=4)
    ''')
    namespace = compiler.execute_module()
    return namespace.model

def model_ifelse_nested_compiled():
    compiler = SPPL_Compiler('''
Nationality   ~= choice({'India': 0.5, 'USA': 0.5})
Perfect       ~= choice({'True': 0.01, 'False': 0.99})
if (Nationality == 'India'):
    if (Perfect == 'False'):
        GPA ~= uniform(loc=0, scale=10)
    else:
        GPA ~= atomic(loc=10)
elif (Nationality == 'USA'):
    if (Perfect == 'False'):
        GPA ~= uniform(loc=0, scale=4)
    elif (Perfect == 'True'):
        GPA ~= atomic(loc=4)
    ''')
    namespace = compiler.execute_module()
    return namespace.model

def model_perfect_nested_compiled():
    compiler = SPPL_Compiler('''
Nationality   ~= choice({'India': 0.5, 'USA': 0.5})
if (Nationality == 'India'):
    Perfect       ~= choice({'True': 0.01, 'False': 0.99})
    if (Perfect == 'False'):
        GPA ~= uniform(loc=0, scale=10)
    else:
        GPA ~= atomic(loc=10)
elif (Nationality == 'USA'):
    Perfect       ~= choice({'True': 0.01, 'False': 0.99})
    if (Perfect == 'False'):
        GPA ~= uniform(loc=0, scale=4)
    else:
        GPA ~= atomic(loc=4)
    ''')
    namespace = compiler.execute_module()
    return namespace.model

@pytest.mark.parametrize('get_model', [
    # Manual
    model_no_latents,
    model_exposed,
    # Interpreter
    model_ifelse_exhuastive,
    model_ifelse_non_exhuastive,
    model_ifelse_nested,
    model_perfect_nested,
    # Compiler
    model_ifelse_exhuastive_compiled,
    model_ifelse_non_exhuastive_compiled,
    model_ifelse_nested_compiled,
    model_perfect_nested_compiled,
])
def test_prior(get_model):
    model = get_model()
    GPA = Id('GPA')
    assert allclose(model.prob(GPA << {10}), 0.5*0.01)
    assert allclose(model.prob(GPA << {4}), 0.5*0.01)
    assert allclose(model.prob(GPA << {5}), 0)
    assert allclose(model.prob(GPA << {1}), 0)

    assert allclose(model.prob((2 < GPA) < 4),
        0.5*0.99*0.5 + 0.5*0.99*0.2)
    assert allclose(model.prob((2 <= GPA) < 4),
        0.5*0.99*0.5 + 0.5*0.99*0.2)
    assert allclose(model.prob((2 < GPA) <= 4),
        0.5*(0.99*0.5 + 0.01) + 0.5*0.99*0.2)
    assert allclose(model.prob((2 < GPA) <= 8),
        0.5*(0.99*0.5 + 0.01) + 0.5*0.99*0.6)
    assert allclose(model.prob((2 < GPA) < 10),
        0.5*(0.99*0.5 + 0.01) + 0.5*0.99*0.8)
    assert allclose(model.prob((2 < GPA) <= 10),
        0.5*(0.99*0.5 + 0.01) + 0.5*(0.99*0.8 + 0.01))

    assert allclose(model.prob(((2 <= GPA) < 4) | (7 < GPA)),
        (0.5*0.99*0.5 + 0.5*0.99*0.2) + (0.5*(0.99*0.3 + 0.01)))

    assert allclose(model.prob(((2 <= GPA) < 4) & (7 < GPA)), 0)

def test_condition():
    model = model_no_latents()
    GPA = Id('GPA')
    model_condition = model.condition(GPA << {4} | GPA << {10})
    assert len(model_condition.children) == 2
    assert model_condition.children[0].support == Interval.Ropen(4, 5)
    assert model_condition.children[1].support == Interval.Ropen(10, 11)

    model_condition = model.condition((0 < GPA < 4))
    assert len(model_condition.children) == 2
    assert model_condition.children[0].support \
        == model_condition.children[1].support
    assert allclose(
        model_condition.children[0].logprob(GPA < 1),
        model_condition.children[1].logprob(GPA < 1))

def test_logpdf():
    model = model_no_latents()
    assert allclose(0.005, model.pdf({GPA: 4}))
    assert allclose(0.005, model.pdf({GPA: 10}))
