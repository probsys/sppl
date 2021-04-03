# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

"""Convert SPE to JSON friendly dictionary."""

from fractions import Fraction

import scipy.stats
from sympy import E
from sympy import sqrt

from ..spe import AtomicLeaf
from ..spe import ContinuousLeaf
from ..spe import DiscreteLeaf
from ..spe import NominalLeaf
from ..spe import ProductSPE
from ..spe import SumSPE

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

def spe_from_dict(metadata):
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
    if metadata['class'] == 'SumSPE':
        children = [spe_from_dict(c) for c in metadata['children']]
        weights = metadata['weights']
        return SumSPE(children, weights)
    if metadata['class'] == 'ProductSPE':
        children = [spe_from_dict(c) for c in metadata['children']]
        return ProductSPE(children)

    assert False, 'Cannot convert %s to SPE' % (metadata,)

def spe_to_dict(spe):
    if isinstance(spe, NominalLeaf):
        return {
            'class'        : 'NominalLeaf',
            'symbol'       : spe.symbol.token,
            'dist'         : [
                (str(x), (w.numerator, w.denominator))
                for x, w in spe.dist.items()
            ],
            'env'          : env_to_dict(spe.env),
        }
    if isinstance(spe, AtomicLeaf):
        return {
            'class'        : 'AtomicLeaf',
            'symbol'       : spe.symbol.token,
            'value'        : spe.value,
            'env'          : env_to_dict(spe.env),
        }
    if isinstance(spe, ContinuousLeaf):
        return {
            'class'         : 'ContinuousLeaf',
            'symbol'        : spe.symbol.token,
            'dist'          : scipy_dist_to_dict(spe.dist),
            'support'       : repr(spe.support),
            'conditioned'   : spe.conditioned,
            'env'           : env_to_dict(spe.env),
        }
    if isinstance(spe, DiscreteLeaf):
        return {
            'class'         : 'DiscreteLeaf',
            'symbol'        : spe.symbol.token,
            'dist'          : scipy_dist_to_dict(spe.dist),
            'support'       : repr(spe.support),
            'conditioned'   : spe.conditioned,
            'env'           : env_to_dict(spe.env),
        }
    if isinstance(spe, SumSPE):
        return {
            'class'         : 'SumSPE',
            'children'      : [spe_to_dict(c) for c in spe.children],
            'weights'       : spe.weights,
        }
    if isinstance(spe, ProductSPE):
        return {
            'class'         : 'ProductSPE',
            'children'      : [spe_to_dict(c) for c in spe.children],
        }
    assert False, 'Cannot convert %s to JSON' % (spe,)
