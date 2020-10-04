# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

import sys
from collections import namedtuple

from IPython.core.magic import Magics
from IPython.core.magic import cell_magic
from IPython.core.magic import line_magic
from IPython.core.magic import magics_class
from IPython.core.magic import needs_local_scope

from sppl.compilers.sppl_to_python import SPPL_Compiler

from .render import render_graphviz

Model = namedtuple('Model', ['source', 'compiler', 'namespace'])

@magics_class
class SPPL_Magics(Magics):

    def __init__(self, shell):
        super().__init__(shell)
        self.programs = {}

    @line_magic
    def sppl_get_spn(self, line):
        assert line in self.programs, 'unknown program %s' % (line,)
        return getattr(self.programs[line].namespace, line)

    @cell_magic
    def sppl(self, line, cell):
        if not line:
            sys.stderr.write('specify model name after %%sppl')
            return
        if line in self.programs:
            del self.programs[line]
        compiler = SPPL_Compiler(cell, line)
        namespace = compiler.execute_module()
        self.programs[line] = Model(cell, compiler, namespace)

    @line_magic
    def sppl_to_python(self, line):
        assert line in self.programs, 'unknown program %s' % (line,)
        print(self.programs[line].compiler.render_module())

    @needs_local_scope
    @line_magic
    def sppl_to_graph(self, line, local_ns):
        tokens = line.strip().split(' ')
        line = tokens[0]
        filename = tokens[1] if len(tokens) == 2 else None
        if line in self.programs:
            spn = self.sppl_get_spn(line)
        elif line in local_ns:
            spn = local_ns[line]
        else:
            assert False, 'unknown program %s' % (line,)
        return render_graphviz(spn, filename=filename)

    @line_magic
    def sppl_get_namespace(self, line):
        assert line in self.programs, 'unknown program %s' % (line,)
        return self.programs[line].namespace
