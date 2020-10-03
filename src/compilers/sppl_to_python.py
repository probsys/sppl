# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

"""Convert SPPL to Python 3."""

import ast
import inspect
import copy
import io
import os
import re

from types import SimpleNamespace
from collections import namedtuple
from collections import OrderedDict
from contextlib import contextmanager

from astunparse import unparse

def __load_spn_distributions():
    from .. import distributions
    members = inspect.getmembers(distributions,lambda t: isinstance(t, type))
    return frozenset(m for (m, v) in members if m[0].islower())

DISTRIBUTIONS = __load_spn_distributions()
get_indentation = lambda i: ' ' * i

@contextmanager
def maybe_sequence_block(visitor, value, first=None):
    # Enter
    active = len(value) > 1 or first
    if active:
        idt = get_indentation(visitor.indentation)
        visitor.command.write('%sSequence(' % (idt,))
        visitor.command.write('\n')
        visitor.indentation += 4
    # Yield
    yield None
    # Exit
    if active:
        visitor.indentation -= 4
        idt = get_indentation(visitor.indentation)
        visitor.command.write('%s)' % (idt,))
        if not first:
            visitor.command.write(',')
            visitor.command.write('\n')

class SPPL_Visitor(ast.NodeVisitor):
    def __init__(self, stream=None):
        self.first = True
        self.indentation = 0
        self.context = ['global']
        self.command = stream or io.StringIO()
        self.imports = []
        self.constants = OrderedDict()
        self.variables = OrderedDict()
        self.distributions = OrderedDict()

    def generic_visit(self, node):
        """Called if no explicit visitor function exists for a node."""
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                with maybe_sequence_block(self, value, self.first):
                    self.first = False
                    for item in value:
                        if isinstance(item, ast.AST):
                            self.visit(item)
            elif isinstance(value, ast.AST):
                self.visit(value)

    def visit_Import(self, node):
        str_node = unparse(node).replace(os.linesep, '')
        self.imports.append(str_node)
    def visit_ImportFrom(self, node):
        str_node = unparse(node).replace(os.linesep, '')
        self.imports.append(str_node)

    def visit_Assign(self, node):
        str_node = unparse(node)

        # Convert IfExp to an If?
        if isinstance(node.value, ast.IfExp):
            node_prime = unroll_ifexp(node.targets, node.value)
            return self.visit(node_prime)

        # Analyze node.target.
        assert len(node.targets) == 1, unparse(node)
        assert isinstance(node.value, ast.expr), 'unknown value %s' % (str_node,)

        # Record visited distributions.
        value = node.value
        visitor_name = SPPL_Visitor_Name(DISTRIBUTIONS, self.variables)
        visitor_name.visit(value)
        for d in visitor_name.distributions:
            if d not in self.distributions:
                self.distributions[d] = None

        # Assigning an array.
        if isinstance(value, ast.Call) and value.func.id == 'array':
            return self.visit_Assign_array(node)
        # Assigning a distribution.
        if visitor_name.distributions:
            # Assigning distribution (directly).
            if isinstance(value, ast.Call) and value.func.id in DISTRIBUTIONS:
                return self.visit_Assign_sample_or_transform(node, 'Sample')
            # Assigning distribution (mixture).
            if isinstance(value, ast.BinOp) and isinstance(value.op, ast.BitOr):
                return self.visit_Assign_sample_or_transform(node, 'Sample')
            assert False, unparse(node)
        # Assigning a transform.
        if visitor_name.variables:
            return self.visit_Assign_sample_or_transform(node, 'Transform')
        # Assigning a Python variable.
        return self.visit_Assign_py(node)

    def visit_Assign_array(self, node):
        target = node.targets[0]
        unode = unparse(node)
        assert self.context == ['global'], unode         # must be global
        assert isinstance(target, ast.Name), unode       # must not be subscript
        assert target.id not in self.variables, unode    # must be fresh
        assert len(node.value.args) == 1, unode          # must be array(n)
        # assert isinstance(node.value.args[0], ast.Num) # must be num n
        # assert isinstance(node.value.args[0].n, int)   # must be int n
        # assert node.value.args[0].n > 0                # must be pos n
        n = unparse(node.value.args[0]).strip()
        self.variables[target.id] = ('array', n)

    def visit_Assign_sample_or_transform(self, node, op):
        unode = unparse(node)
        assert op in ['Sample', 'Transform'], unode
        target = node.targets[0]
        assert isinstance(target, (ast.Name, ast.Subscript)), unode
        if isinstance(target, ast.Name):
            assert 'for' not in self.context, unode
            if 'elif' not in self.context:
                assert target.id not in self.variables, unode
                self.variables[target.id] = ('variable', None)
            if 'elif' in self.context:
                assert target.id in self.variables, unode
                assert target.id in self.variables, unode
        if isinstance(target, ast.Subscript):
            assert target.value.id in self.variables, unode
            assert self.variables[target.value.id][0] == 'array', unode
        value_prime = SPPL_Transformer_Compare().visit(node.value)
        src_value = unparse(value_prime).replace(os.linesep, '')
        src_targets = unparse(node.targets).replace(os.linesep, '')
        idt = get_indentation(self.indentation)
        self.command.write('%s%s(%s, %s),' % (idt, op, src_targets, src_value))
        self.command.write('\n')

    def visit_Assign_py(self, node):
        unode = unparse(node)
        assert self.context == ['global'], unode
        if self.distributions:
            assert False, 'constants only before sampling %s' % (unode,)
        str_node = unode.strip()
        self.constants[str_node] = None

    def visit_For(self, node):
        assert isinstance(node.iter, ast.Call)
        if node.iter.func.id == 'range':
            return self.visit_For_vanilla(node)
        if node.iter.func.id == 'switch':
            return self.visit_For_switch(node)
        assert False, unparse(node)

    def visit_For_switch(self, node):
        unode = unparse(node)
        assert isinstance(node.target, (ast.Name, ast.Tuple)), unode
        assert node.iter.func.id == 'switch', unode
        assert len(node.iter.args) == 2, unode
        assert isinstance(node.iter.args[0], (ast.Name, ast.Subscript)), unode
        # Open Switch.
        self.context.append('switch')
        idt = get_indentation(self.indentation)
        symbol = unparse(node.iter.args[0]).strip()
        values = unparse(node.iter.args[1]).strip()
        translator = str.maketrans({'(':'', ')':''})
        idx = unparse(node.target).strip().translate(translator)
        self.command.write('%sSwitch(%s, %s, lambda %s:'
            % (idt, symbol, values, idx))
        self.command.write('\n')
        # Write the body.
        self.indentation += 4
        self.generic_visit(ast.Module(node.body))
        self.indentation -= 4
        # Close Switch
        idt = get_indentation(self.indentation)
        self.command.write('%s),' % (idt,))
        self.command.write('\n')
        self.context.pop()

    def visit_For_vanilla(self, node):
        unode = unparse(node)
        assert isinstance(node.target, ast.Name), unode
        assert node.iter.func.id == 'range', unode
        assert len(node.iter.args) in [1, 2], unode
        if len(node.iter.args) == 1:
            n0 = 0
            n1 = unparse(node.iter.args[0]).strip()
        if len(node.iter.args) == 2:
            n0 = unparse(node.iter.args[0]).strip()
            n1 = unparse(node.iter.args[1]).strip()
        # Open For.
        self.context.append('for')
        idt = get_indentation(self.indentation)
        idx = unparse(node.target).strip()
        self.command.write('%sFor(%s, %s, lambda %s:' % (idt, n0, n1, idx))
        self.command.write('\n')
        # Write body.
        self.indentation += 4
        self.generic_visit(ast.Module(node.body))
        self.indentation -= 4
        # Close For.
        idt = get_indentation(self.indentation)
        self.command.write('%s),' % (idt,))
        self.command.write('\n')
        self.context.pop()

    def visit_If(self, node):
        unode = unparse(node)
        unrolled = unroll_if(node)
        assert 2 <= len(unrolled), 'if needs elif/else: %s' % (unode,)
        idt = get_indentation(self.indentation)
        # Open IfElse.
        self.context.append('if')
        self.command.write('%sIfElse(' % (idt,))
        self.command.write('\n')
        # Write branches.
        self.indentation += 4
        for i, (test, body) in enumerate(unrolled):
            # Write the test.
            test_prime = SPPL_Transformer_Compare().visit(test)
            src_test = unparse(test_prime).strip()
            idt = get_indentation(self.indentation)
            self.command.write('%s%s,' % (idt, src_test))
            self.command.write('\n')
            # Write the body.
            if i > 0:
                self.context.append('elif')
            self.indentation += 4
            self.generic_visit(ast.Module(body))
            self.indentation -= 4
            if i > 0:
                self.context.pop()
        self.indentation -= 4
        # Close IfElse.
        idt = get_indentation(self.indentation)
        self.command.write('%s),' % (idt,))
        self.command.write('\n')
        self.context.pop()

    def visit_Call(self, node):
        unode = unparse(node)
        if node.func.id == 'condition':
            assert len(node.args) == 1, unode
            idt = get_indentation(self.indentation)
            event = node.args[0]
            event_prime = SPPL_Transformer_Compare().visit(event)
            src_event = unparse(event_prime).strip()
            self.command.write('%sCondition(%s),' % (idt, src_event))
            self.command.write('\n')

def unroll_if(node, current=None):
    current = [] if current is None else current
    assert isinstance(node, ast.If)
    current.append((node.test, node.body))
    # Base case, terminating at elif.
    if not node.orelse:
        return current
    # Base case, next statement is not an If.
    if not (isinstance(node.orelse[0], ast.If) and len(node.orelse)== 1):
        current.append((ast.parse('True'), node.orelse))
        return current
    # Recursive case, next statement is elif
    return unroll_if(node.orelse[0], current)

def unroll_ifexp(target, node):
    assert isinstance(node, ast.IfExp)
    expr = ast.If(node.test, ast.Assign(target, node.body), None)
    if isinstance(node.orelse, ast.IfExp):
        expr.orelse = [unroll_ifexp(target, node.orelse)]
    else:
        expr.orelse = [ast.Assign(target, node.orelse)]
    return expr

class SPPL_Transformer_Compare(ast.NodeTransformer):
    def visit_Compare(self, node):
        # TODO: Implement or/and.
        if len(node.ops) > 1:
            return node
        if isinstance(node.ops[0], ast.In):
            return ast.BinOp(
                left=node.left,
                op=ast.LShift(),
                right=node.comparators[0])
        if isinstance(node.ops[0], ast.Eq):
            return ast.BinOp(
                left=node.left,
                op=ast.LShift(),
                right=ast.Set(node.comparators))
        if isinstance(node.ops[0], ast.NotIn):
            node_copy = copy.deepcopy(node)
            node_copy.ops[0] = ast.In()
            return ast.UnaryOp(
                op=ast.Invert(),
                operand=self.visit_Compare(node_copy))
        if isinstance(node.ops[0], ast.NotEq):
            node_copy = copy.deepcopy(node)
            node_copy.ops[0] = ast.Eq()
            return ast.UnaryOp(
                op=ast.Invert(),
                operand=self.visit_Compare(node_copy))
        return node

class SPPL_Visitor_Name(ast.NodeVisitor):
    def __init__(self, distributions, variables):
        self.distributions_lookup = distributions
        self.variables_lookup = variables
        self.distributions = set()
        self.variables = set()
    def visit_Name(self, node):
        if node.id in self.distributions_lookup:
            self.distributions.add(node.id)
        if node.id in self.variables_lookup:
            self.variables.add(node.id)

prog = namedtuple('prog', ('imports', 'constants', 'variables', 'arrays', 'command'))
class SPPL_Compiler():
    def __init__(self, source, modelname='model'):
        self.source = source
        self.modelname = modelname
        self.prog = prog(
            imports=io.StringIO(),
            constants=io.StringIO(),
            variables=io.StringIO(),
            arrays=io.StringIO(),
            command=io.StringIO())
        self.compile()
    def preprocess(self):
        lines = self.source.split(os.linesep)
        source_prime = os.linesep.join(l for l in lines if l.strip())
        source_prime = source_prime.replace('~=', '=')
        source_prime = re.sub(
            r'^(\s*)switch\s*\((.+)\)\s*cases\s*\((.+)\s*in\s+(.+)\s*\)\s*:',
            r"\1for \3 in switch(\2, \4):",
            source_prime, flags=re.MULTILINE)
        return source_prime
    def compile(self):
        # Parse and visit.
        source_prime = self.preprocess()
        tree = ast.parse(source_prime)
        visitor = SPPL_Visitor()
        visitor.visit(tree)
        # Write the imports.
        self.prog.imports.write("# IMPORT STATEMENTS")
        self.prog.imports.write('\n')
        for i in visitor.imports:
            self.prog.imports.write(i)
            self.prog.imports.write('\n')
        for d in sorted(visitor.distributions):
            self.prog.imports.write('from sppl.distributions import %s' % (d,))
            self.prog.imports.write('\n')
        for c in ['Id', 'IdArray', 'Condition', 'IfElse', 'For', 'Sample',
                    'Sequence', 'Switch', 'Transform']:
            self.prog.imports.write('from sppl.compilers.ast_to_spn import %s' % (c,))
            self.prog.imports.write('\n')
        # Write the constants.
        if visitor.constants:
            self.prog.constants.write('# CONSTANT DECLRATIONS')
            self.prog.constants.write('\n')
            for c in visitor.constants:
                self.prog.constants.write(c)
                self.prog.constants.write('\n')
        # Write the variables.
        variables = [v for v, t in visitor.variables.items() if t[0] == 'variable']
        arrays = [(v, t[1]) for v, t in visitor.variables.items() if t[0] == 'array']
        if variables:
            self.prog.variables.write('# VARIABLE DECLARATIONS')
            self.prog.variables.write('\n')
            for v in variables:
                self.prog.variables.write('%s = Id(\'%s\')' % (v, v,))
                self.prog.variables.write('\n')
        if arrays:
            self.prog.arrays.write('# ARRAY DECLARATIONS')
            self.prog.arrays.write('\n')
            for v, n in arrays:
                self.prog.arrays.write('%s = IdArray(\'%s\', %s)' % (v, v, n,))
                self.prog.arrays.write('\n')
        # Write the command.
        self.prog.command.write('# MODEL DEFINITION')
        self.prog.command.write('\n')
        command = visitor.command.getvalue()
        self.prog.command.write('command = %s' % (command,))
        self.prog.command.write('\n')
        # Write the interpret step.
        self.prog.command.write('%s = command.interpret()' % (self.modelname,))
    def render_module(self):
        """Render the source code as a stand alone Python module."""
        program = io.StringIO()
        for stream in self.prog:
            v = stream.getvalue()
            if v:
                program.write(stream.getvalue())
                program.write('\n')
        return program.getvalue()
    def execute_module(self):
        """Execute the source code in a fresh module."""
        code = self.render_module()
        namespace = {}
        exec(code, namespace)
        return SimpleNamespace(**namespace)
