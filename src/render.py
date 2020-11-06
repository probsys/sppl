# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from math import exp

from .spn import AtomicLeaf
from .spn import DiscreteLeaf
from .spn import LeafSPN
from .spn import NominalLeaf
from .spn import ProductSPN
from .spn import RealLeaf
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
    if isinstance(spn, AtomicLeaf):
        return ['AtomicLeaf', [
            ['symbol', spn.symbol],
            ['value', spn.value],
            ['env', dict(spn.env)]]
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
