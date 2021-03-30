# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from math import exp

from .spe import AtomicLeaf
from .spe import DiscreteLeaf
from .spe import LeafSPE
from .spe import NominalLeaf
from .spe import ProductSPE
from .spe import RealLeaf
from .spe import SumSPE

def render_nested_lists_concise(spe):
    if isinstance(spe, LeafSPE):
        return [(str(k), str(v)) for k, v in spe.env.items()]
    if isinstance(spe, SumSPE):
        return ['+(%d)' % (len(spe.children),),
            # [exp(w) for w in spe.weights],
            [render_nested_lists_concise(c) for c in spe.children]
        ]
    if isinstance(spe, ProductSPE):
        return ['*(%d)' % (len(spe.children),),
            [render_nested_lists_concise(c) for c in spe.children]
        ]

def render_nested_lists(spe):
    if isinstance(spe, NominalLeaf):
        return ['NominalLeaf', [
            ['symbol', spe.symbol],
            ['env', dict(spe.env)],
            ['dist', {str(x): float(w) for x, w in spe.dist.items()}]]
        ]
    if isinstance(spe, AtomicLeaf):
        return ['AtomicLeaf', [
            ['symbol', spe.symbol],
            ['value', spe.value],
            ['env', dict(spe.env)]]
        ]
    if isinstance(spe, RealLeaf):
        return ['RealLeaf', [
            ['symbol', spe.symbol],
            ['env', dict(spe.env)],
            ['dist', (spe.dist.dist.name, spe.dist.args, spe.dist.kwds)],
            ['support', spe.support],
            ['conditioned', spe.conditioned]]
        ]
    if isinstance(spe, DiscreteLeaf):
        return ['DiscreteLeaf', [
            ['symbol', spe.symbol],
            ['dist', (spe.dist.dist.name, spe.dist.args, spe.dist.kwds)],
            ['support', spe.support],
            ['conditioned', spe.conditioned]]
        ]
    if isinstance(spe, SumSPE):
        return ['SumSPE', [
            ['symbols', list(spe.symbols)],
            ['weights', [exp(w) for w in spe.weights]],
            ['n_children', len(spe.children)],
            ['children', [render_nested_lists(c) for c in spe.children]]]
        ]
    if isinstance(spe, ProductSPE):
        return ['ProductSPE', [
            ['symbols', list(spe.symbols)],
            ['n_children', len(spe.children)],
            ['children', [render_nested_lists(c) for c in spe.children]]]
        ]
    assert False, 'Unknown SPE type: %s' % (spe,)
