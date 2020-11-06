# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

"""Convert SPN to JSON friendly dictionary."""

from fractions import Fraction

import scipy.stats
from sympy import E
from sympy import sqrt

from ..spn import AtomicLeaf
from ..spn import ContinuousLeaf
from ..spn import DiscreteLeaf
from ..spn import NominalLeaf
from ..spn import ProductSPN
from ..spn import SumSPN

# Needed for "eval"
from ..sets import *
from ..transforms import Id
from ..transforms import Identity
from ..transforms import Radical
from ..transforms import Exponential
from ..transforms import Logarithm
from ..transforms import Abs
from ..transforms import Reciprocal
from ..transforms import Poly
from ..transforms import Piecewise
from ..transforms import EventInterval
from ..transforms import EventFiniteReal
from ..transforms import EventFiniteNominal
from ..transforms import EventOr
from ..transforms import EventAnd

def env_from_dict(env):
    if env is None:
        return None
    # Used in eval.
    return {eval(k): eval(v) for k, v in env.items()}

def env_to_dict(env):
    if len(env) == 1:
        return None
    return {repr(k): repr(v) for k, v in env.items()}

def scipy_dist_from_dict(dist):
    constructor = getattr(scipy.stats, dist['name'])
    return constructor(*dist['args'], **dist['kwds'])

def scipy_dist_to_dict(dist):
    return {
        'name': dist.dist.name,
        'args': dist.args,
        'kwds': dist.kwds
    }

def spn_from_dict(metadata):
    if metadata['class'] == 'NominalLeaf':
        symbol = Id(metadata['symbol'])
        dist = {x: Fraction(w[0], w[1]) for x, w in metadata['dist']}
        return NominalLeaf(symbol, dist)
    if metadata['class'] == 'AtomicLeaf':
        symbol = Id(metadata['symbol'])
        value = float(metadata['value'])
        env = env_from_dict(metadata['env'])
        return AtomicLeaf(symbol, value, env=env)
    if metadata['class'] == 'ContinuousLeaf':
        symbol = Id(metadata['symbol'])
        dist = scipy_dist_from_dict(metadata['dist'])
        support = eval(metadata['support'])
        conditioned = metadata['conditioned']
        env = env_from_dict(metadata['env'])
        return ContinuousLeaf(symbol, dist, support, conditioned, env=env)
    if metadata['class'] == 'DiscreteLeaf':
        symbol = Id(metadata['symbol'])
        dist = scipy_dist_from_dict(metadata['dist'])
        support = eval(metadata['support'])
        conditioned = metadata['conditioned']
        env = env_from_dict(metadata['env'])
        return DiscreteLeaf(symbol, dist, support, conditioned, env=env)
    if metadata['class'] == 'SumSPN':
        children = [spn_from_dict(c) for c in metadata['children']]
        weights = metadata['weights']
        return SumSPN(children, weights)
    if metadata['class'] == 'ProductSPN':
        children = [spn_from_dict(c) for c in metadata['children']]
        return ProductSPN(children)

    assert False, 'Cannot convert %s to SPN' % (metadata,)

def spn_to_dict(spn):
    if isinstance(spn, NominalLeaf):
        return {
            'class'        : 'NominalLeaf',
            'symbol'       : spn.symbol.token,
            'dist'         : [
                (str(x), (w.numerator, w.denominator))
                for x, w in spn.dist.items()
            ],
            'env'          : env_to_dict(spn.env),
        }
    if isinstance(spn, AtomicLeaf):
        return {
            'class'        : 'AtomicLeaf',
            'symbol'       : spn.symbol.token,
            'value'        : spn.value,
            'env'          : env_to_dict(spn.env),
        }
    if isinstance(spn, ContinuousLeaf):
        return {
            'class'         : 'ContinuousLeaf',
            'symbol'        : spn.symbol.token,
            'dist'          : scipy_dist_to_dict(spn.dist),
            'support'       : repr(spn.support),
            'conditioned'   : spn.conditioned,
            'env'           : env_to_dict(spn.env),
        }
    if isinstance(spn, DiscreteLeaf):
        return {
            'class'         : 'DiscreteLeaf',
            'symbol'        : spn.symbol.token,
            'dist'          : scipy_dist_to_dict(spn.dist),
            'support'       : repr(spn.support),
            'conditioned'   : spn.conditioned,
            'env'           : env_to_dict(spn.env),
        }
    if isinstance(spn, SumSPN):
        return {
            'class'         : 'SumSPN',
            'children'      : [spn_to_dict(c) for c in spn.children],
            'weights'       : spn.weights,
        }
    if isinstance(spn, ProductSPN):
        return {
            'class'         : 'ProductSPN',
            'children'      : [spn_to_dict(c) for c in spn.children],
        }
    assert False, 'Cannot convert %s to JSON' % (spn,)
