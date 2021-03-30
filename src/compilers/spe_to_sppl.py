# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

"""Convert SPE to SPPL."""

from io import StringIO
from math import exp

from ..spe import RealLeaf
from ..spe import NominalLeaf
from ..spe import ProductSPE
from ..spe import SumSPE

get_indentation = lambda i: ' ' * i
float_to_str = lambda x,fw: '%1.*f' % (fw, float(x)) if fw else str(x)
class _SPPL_Render_State:
    def __init__(self, stream=None, branches=None, indentation=None,
            fwidth=None):
        self.stream = stream or StringIO()
        self.branches = branches or []
        self.indentation = indentation or 0
        self.fwidth = fwidth
def render_sppl_choice(symbol, dist, stream, indentation, fwidth):
    idt = get_indentation(indentation)
    dist_pairs = [
        '\'%s\' : %s' % (k, float_to_str(v, fwidth))
        for (k, v) in dist.items()
    ]
    # Write each outcome on new line.
    if sum(len(x) for x in dist_pairs) > 30:
        idt4 = get_indentation(indentation + 4)
        prefix = ',\n%s' % (idt4,)
        dist_op = '\n%s' % (idt4,)
        dist_cl = '\n%s' % (idt,)
    # Write all outcomes on same line.
    else:
        prefix = ', '
        dist_op = ''
        dist_cl = ''
    dist_str = prefix.join(dist_pairs)
    stream.write('%s%s ~= choice({%s%s%s})' %
        (idt, symbol, dist_op, dist_str, dist_cl))
    stream.write('\n')
def render_sppl_helper(spe, state):
    if isinstance(spe, NominalLeaf):
        assert len(spe.env) == 1
        render_sppl_choice(
            spe.symbol,
            spe.dist,
            state.stream,
            state.indentation,
            state.fwidth)
        return state
    if isinstance(spe, RealLeaf):
        kwds = ', '.join([
            '%s=%s' % (k, float_to_str(v, state.fwidth))
            for k, v in spe.dist.kwds.items()
        ])
        dist = '%s(%s)' % (spe.dist.dist.name, kwds)
        idt = get_indentation(state.indentation)
        state.stream.write('%s%s ~= %s' % (idt, spe.symbol, dist))
        state.stream.write('\n')
        if spe.conditioned:
            event = spe.symbol << spe.support
            # TODO: Consider using repr(event)
            state.stream.write('%scondition(%s)' % (idt, event))
            state.stream.write('\n')
        for i, (var, expr) in enumerate(spe.env.items()):
            if 1 <= i:
                state.stream.write('%s%s ~= %s' % (idt, var, expr))
                state.stream.write('\n')
        return state
    if isinstance(spe, ProductSPE):
        for child in spe.children:
            state = render_sppl_helper(child, state)
        return state
    if isinstance(spe, SumSPE):
        if len(spe.children) == 0:
            return state
        if len(spe.children) == 1:
            return render_sppl_helper(spe.children[0], state)
        branch_var = 'branch_var_%s' % (len(state.branches))
        branch_idxs = [str(i) for i in range(len(spe.children))]
        branch_dist = {k: exp(w) for k, w in zip(branch_idxs, spe.weights)}
        state.branches.append((branch_var, branch_dist))
        # Write the branches.
        for i, child in zip(branch_idxs, spe.children):
            ifstmt = 'if' if i == '0' else 'elif'
            idt = get_indentation(state.indentation)
            state.stream.write('%s%s (%s == \'%s\'):'
                % (idt, ifstmt, branch_var, i))
            state.stream.write('\n')
            state.indentation += 4
            state = render_sppl_helper(child, state)
            state.stream.write('\n')
            state.indentation -= 4
        return state
    assert False, 'Unknown spe %s' % (spe,)

def render_sppl(spe, stream=None, fwidth=None):
    if stream is None:
        stream = StringIO()
    state = _SPPL_Render_State(fwidth=fwidth)
    state = render_sppl_helper(spe, state)
    assert state.indentation == 0
    # Write the import.
    stream.write('from sppl.distributions import *')
    stream.write('\n')
    stream.write('\n')
    # Write the branch variables (if any).
    for branch_var, branch_dist in state.branches:
        render_sppl_choice(branch_var, branch_dist, stream, 0, fwidth)
    stream.write('\n')
    # Write the SPPL.
    stream.write(state.stream.getvalue())
    return stream
