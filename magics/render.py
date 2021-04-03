# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

import os
import tempfile
import time

from math import exp

import graphviz
import networkx as nx

from sppl.spe import AtomicLeaf
from sppl.spe import NominalLeaf
from sppl.spe import ProductSPE
from sppl.spe import RealLeaf
from sppl.spe import SumSPE

gensym = lambda: 'r%s' % (str(time.time()).replace('.', ''),)

def render_networkx_graph(spe):
    if isinstance(spe, NominalLeaf):
        G = nx.DiGraph()
        root = gensym()
        G.add_node(root, label='%s\n%s' % (spe.symbol.token, 'Nominal'))
        return G
    if isinstance(spe, AtomicLeaf):
        G = nx.DiGraph()
        root = gensym()
        G.add_node(root, label='%s\n%s(%s)'
            % (spe.symbol.token, 'Atomic', str(spe.value)))
        return G
    if isinstance(spe, RealLeaf):
        G = nx.DiGraph()
        root = gensym()
        kwds = '\n%s' % (tuple(spe.dist.kwds.values()),) if spe.dist.kwds else ''
        G.add_node(root, label='%s\n%s%s' % (spe.symbol.token, spe.dist.dist.name, kwds))
        if len(spe.env) > 1:
            for k, v in spe.env.items():
                if k != spe.symbol:
                    roott = gensym()
                    G.add_node(roott, label=str(v), style='filled')
                    G.add_edge(root, roott, label=' %s' % (str(k),), style='dashed')
        return G
    if isinstance(spe, SumSPE):
        G = nx.DiGraph()
        root = gensym()
        G.add_node(root, label='\N{PLUS SIGN}')
        # Add nodes and edges from children.
        G_children = [render_networkx_graph(c) for c in spe.children]
        for i, x in enumerate(G_children):
            G.add_nodes_from(x.nodes.data())
            G.add_edges_from(x.edges.data())
            subroot = list(nx.topological_sort(x))[0]
            G.add_edge(root, subroot, label='%1.3f' % (exp(spe.weights[i]),))
        return G
    if isinstance(spe, ProductSPE):
        G = nx.DiGraph()
        root = gensym()
        G.add_node(root, label='\N{MULTIPLICATION SIGN}')
        # Add nodes and edges from children.
        G_children = [render_networkx_graph(c) for c in spe.children]
        for x in G_children:
            G.add_nodes_from(x.nodes.data())
            G.add_edges_from(x.edges.data())
            subroot = list(nx.topological_sort(x))[0]
            G.add_edge(root, subroot)
        return G
    assert False, 'Unknown SPE type: %s' % (spe,)

def render_graphviz(spe, filename=None, ext=None, show=None):
    fname = filename
    if filename is None:
        f = tempfile.NamedTemporaryFile(delete=False)
        fname = f.name
    G = render_networkx_graph(spe)
    ext = ext or 'png'
    assert ext in ['png', 'pdf'], 'Extension must be .pdf or .png'
    fname_dot = '%s.dot' % (fname,)
    # nx.set_edge_attributes(G, 'serif', 'fontname')
    # nx.set_node_attributes(G, 'serif', 'fontname')
    nx.nx_agraph.write_dot(G, fname_dot)
    source = graphviz.Source.from_file(fname_dot, format=ext)
    source.render(filename=fname, view=show)
    os.unlink(fname)
    return source
