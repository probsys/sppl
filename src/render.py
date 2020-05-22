# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from io import StringIO
from math import exp

from .spn import RealLeaf
from .spn import DiscreteLeaf
from .spn import LeafSPN
from .spn import NominalLeaf
from .spn import ProductSPN
from .spn import SumSPN

def render_nested_lists_concise(spn):
    if isinstance(spn, LeafSPN):
        return [(str(k), str(v)) for k, v in spn.env.items()]
    if isinstance(spn, SumSPN):
        return ['+(%d)' % (len(spn.children),),
            # [exp(w) for w in spn.weights],
            [render_nested_lists_concise(c) for c in spn.children]
        ]
    if isinstance(spn, ProductSPN):
        return ['*(%d)' % (len(spn.children),),
            [render_nested_lists_concise(c) for c in spn.children]
        ]

def render_nested_lists(spn):
    if isinstance(spn, NominalLeaf):
        return ['NominalLeaf', [
            ['symbol', spn.symbol],
            ['env', dict(spn.env)],
            ['dist', {str(x): float(w) for x, w in spn.dist.items()}]]
        ]
    if isinstance(spn, RealLeaf):
        return ['RealLeaf', [
            ['symbol', spn.symbol],
            ['env', dict(spn.env)],
            ['dist', (spn.dist.dist.name, spn.dist.args, spn.dist.kwds)],
            ['support', spn.support],
            ['conditioned', spn.conditioned]]
        ]
    if isinstance(spn, DiscreteLeaf):
        return ['DiscreteLeaf', [
            ['symbol', spn.symbol],
            ['dist', (spn.dist.dist.name, spn.dist.args, spn.dist.kwds)],
            ['support', spn.support],
            ['conditioned', spn.conditioned]]
        ]
    if isinstance(spn, SumSPN):
        return ['SumSPN', [
            ['symbols', list(spn.symbols)],
            ['weights', [exp(w) for w in spn.weights]],
            ['n_children', len(spn.children)],
            ['children', [render_nested_lists(c) for c in spn.children]]]
        ]
    if isinstance(spn, ProductSPN):
        return ['ProductSPN', [
            ['symbols', list(spn.symbols)],
            ['n_children', len(spn.children)],
            ['children', [render_nested_lists(c) for c in spn.children]]]
        ]
    assert False, 'Unknown SPN type: %s' % (spn,)


get_indentation = lambda i: ' ' * i
float_to_str = lambda x,fw: '%1.*f' % (fw, float(x)) if fw else str(x)
class _SPML_Render_State:
    def __init__(self, stream=None, branches=None, indentation=None,
            fwidth=None):
        self.stream = stream or StringIO()
        self.branches = branches or []
        self.indentation = indentation or 0
        self.fwidth = fwidth
def render_spml_choice(symbol, dist, stream, indentation, fwidth):
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
def render_spml_helper(spn, state):
    if isinstance(spn, NominalLeaf):
        assert len(spn.env) == 1
        render_spml_choice(
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
            state.stream.write('%scondition(%s)' % (idt, event))
            state.stream.write('\n')
        for i, (var, expr) in enumerate(spn.env.items()):
            if 1 <= i:
                state.stream.write('%s%s ~= %s' % (idt, var, expr))
                state.stream.write('\n')
        return state
    if isinstance(spn, ProductSPN):
        for child in spn.children:
            state = render_spml_helper(child, state)
        return state
    if isinstance(spn, SumSPN):
        if len(spn.children) == 0:
            return state
        if len(spn.children) == 1:
            return render_spml_helper(spn.children[0], state)
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
            state = render_spml_helper(child, state)
            state.stream.write('\n')
            state.indentation -= 4
        return state
    assert False, 'Unknown spn %s' % (spn,)

def render_spml(spn, stream=None, fwidth=None):
    if stream is None:
        stream = StringIO()
    state = _SPML_Render_State(fwidth=fwidth)
    state = render_spml_helper(spn, state)
    assert state.indentation == 0
    # Write the import.
    stream.write('from spn.distributions import *')
    stream.write('\n')
    stream.write('\n')
    # Write the branch variables (if any).
    for branch_var, branch_dist in state.branches:
        render_spml_choice(branch_var, branch_dist, stream, 0, fwidth)
    stream.write('\n')
    # Write the SPML.
    stream.write(state.stream.getvalue())
    return stream
