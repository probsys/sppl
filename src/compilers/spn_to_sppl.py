# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

"""Convert SPN to SPPL."""

from io import StringIO
from math import exp

from ..spn import RealLeaf
from ..spn import NominalLeaf
from ..spn import ProductSPN
from ..spn import SumSPN

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
def render_sppl_helper(spn, state):
    if isinstance(spn, NominalLeaf):
        assert len(spn.env) == 1
        render_sppl_choice(
            spn.symbol,
            spn.dist,
            state.stream,
            state.indentation,
            state.fwidth)
        return state
    if isinstance(spn, RealLeaf):
        kwds = ', '.join([
            '%s=%s' % (k, float_to_str(v, state.fwidth))
            for k, v in spn.dist.kwds.items()
        ])
        dist = '%s(%s)' % (spn.dist.dist.name, kwds)
        idt = get_indentation(state.indentation)
        state.stream.write('%s%s ~= %s' % (idt, spn.symbol, dist))
        state.stream.write('\n')
        if spn.conditioned:
            event = spn.symbol << spn.support
            # TODO: Consider using repr(event)
            state.stream.write('%scondition(%s)' % (idt, event))
            state.stream.write('\n')
        for i, (var, expr) in enumerate(spn.env.items()):
            if 1 <= i:
                state.stream.write('%s%s ~= %s' % (idt, var, expr))
                state.stream.write('\n')
        return state
    if isinstance(spn, ProductSPN):
        for child in spn.children:
            state = render_sppl_helper(child, state)
        return state
    if isinstance(spn, SumSPN):
        if len(spn.children) == 0:
            return state
        if len(spn.children) == 1:
            return render_sppl_helper(spn.children[0], state)
        branch_var = 'branch_var_%s' % (len(state.branches))
        branch_idxs = [str(i) for i in range(len(spn.children))]
        branch_dist = {k: exp(w) for k, w in zip(branch_idxs, spn.weights)}
        state.branches.append((branch_var, branch_dist))
        # Write the branches.
        for i, child in zip(branch_idxs, spn.children):
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
    assert False, 'Unknown spn %s' % (spn,)

def render_sppl(spn, stream=None, fwidth=None):
    if stream is None:
        stream = StringIO()
    state = _SPPL_Render_State(fwidth=fwidth)
    state = render_sppl_helper(spn, state)
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
