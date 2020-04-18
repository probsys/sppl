# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

import time

from math import exp

from .spn import RealLeaf
from .spn import DiscreteLeaf
from .spn import LeafSPN
from .spn import NominalLeaf
from .spn import ProductSPN
from .spn import SumSPN

def render_nested_lists_concise(spn):
    if isinstance(spn, LeafSPN):
        return [spn.symbol.token]
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
            ['dist', {str(x): float(w) for x, w in spn.dist.items()}]]
        ]
    if isinstance(spn, RealLeaf):
        return ['RealLeaf', [
            ['symbol', spn.symbol],
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

def render_networkx_graph(spn):
    import networkx as nx
    if isinstance(spn, NominalLeaf):
        G = nx.DiGraph()
        root = str(time.time())
        G.add_node(root, label='%s\n%s' % (spn.symbol.token, 'Nominal'))
        return G
    if isinstance(spn, RealLeaf):
        G = nx.DiGraph()
        root = str(time.time())
        G.add_node(root, label='%s\n%s' % (spn.symbol.token, spn.dist.dist.name))
        return G
    if isinstance(spn, SumSPN):
        G = nx.DiGraph()
        root = str(time.time())
        G.add_node(root, label='+')
        # Add nodes and edges from children.
        G_children = [render_networkx_graph(c) for c in spn.children]
        for i, x in enumerate(G_children):
            G.add_nodes_from(x.nodes.data())
            G.add_edges_from(x.edges.data())
            subroot = list(nx.topological_sort(x))[0]
            G.add_edge(root, subroot, label='%1.3f' % (exp(spn.weights[i]),))
        return G
    if isinstance(spn, ProductSPN):
        G = nx.DiGraph()
        root = str(time.time())
        G.add_node(root, label='*')
        # Add nodes and edges from children.
        G_children = [render_networkx_graph(c) for c in spn.children]
        for x in G_children:
            G.add_nodes_from(x.nodes.data())
            G.add_edges_from(x.edges.data())
            subroot = list(nx.topological_sort(x))[0]
            G.add_edge(root, subroot)
        return G
    assert False, 'Unknown SPN type: %s' % (spn,)

def render_graphviz(spn, filename, show=None):
    import networkx as nx
    from graphviz import Source
    G = render_networkx_graph(spn)
    tokens = filename.split('.')
    base = '.'.join(tokens[:-1])
    ext = tokens[-1]
    assert ext in ['png', 'pdf'], 'Extension must be .pdf or .png'
    fname_dot = '%s.dot' % (base,)
    # nx.set_edge_attributes(G, 'serif', 'fontname')
    # nx.set_node_attributes(G, 'serif', 'fontname')
    nx.nx_agraph.write_dot(G, fname_dot)
    source = Source.from_file(fname_dot, format=ext)
    source.render(filename=filename, view=show)
    return source

from collections import namedtuple
ImpState = namedtuple('ImpState', ['indentation'])
get_indentation = lambda state: ' ' * state.indentation

def render_imp_distribution(distribution):
    from .distributions import RealDistribution
    if isinstance(distribution, dict):
        str_dist = ', '.join('\'%s\': %s' % (k, v) for k, v in distribution.items())
        return '{%s}' % (str_dist)
    if isinstance(distribution, RealDistribution):
        str_kwds = ', '.join('%s=%s' % (k, v) for k, v in distribution.kwargs.items())
        return '%s(%s)' % (distribution.dist.name, str_kwds)
    assert False, 'Unknown distribution: %s' % (distribution,)

def render_imp_command(command, state=None):
    from .interpret import Sample
    from .interpret import Transform
    from .interpret import Repeat
    from .interpret import IfElse
    from .interpret import Sequence
    from .interpret import Otherwise
    if state is None:
        state = ImpState(indentation=0)
    if isinstance(command, Sample):
        str_dist = render_imp_distribution(command.distribution)
        idt = get_indentation(state)
        return '%s%s ~ %s' % (idt, command.symbol, str_dist)
    if isinstance(command, Transform):
        idt = get_indentation(state)
        return '%s%s ~ %s' % (idt, command.symbol, command.expr)
    if isinstance(command, IfElse):
        conditions = command.branches[::2]
        branches = command.branches[1::2]
        assert len(conditions) == len(branches)
        str_blocks = [''] * len(conditions)
        state_prime = ImpState(state.indentation+4)
        idt = get_indentation(state)
        for i, (condition, branch) in enumerate(zip(conditions, branches)):
            if i == 0:
                str_condition = '%sif (%s)' % (idt, condition)
            elif condition is Otherwise:
                str_condition = '%selse' % (idt)
            else:
                str_condition = '%selif (%s)' % (idt, condition)
            str_branch = render_imp_command(branch, state_prime)
            str_blocks[i] = ':\n'.join([str_condition, str_branch])
        return '\n'.join(str_blocks)
    if isinstance(command, Repeat):
        commands = [command.f(i) for i in range(command.n0, command.n1)]
        command_prime = Sequence(*commands)
        return render_imp_command(command_prime, state)
    if isinstance(command, Sequence):
        str_cmds = [render_imp_command(c, state) for c in command.commands]
        return ';\n'.join(str_cmds)
    assert False, 'Unknown command: %s' % (command,)
