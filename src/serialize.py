# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from sympy import *

from fractions import Fraction

import scipy.stats

from .spn import ContinuousReal
from .spn import DiscreteReal
from .spn import NominalDistribution
from .spn import ProductSPN
from .spn import SumSPN
from .transforms import Identity

def scipy_dist_from_json(dist):
    constructor = getattr(scipy.stats, dist['name'])
    return constructor(*dist['args'], **dist['kwds'])

def scipy_dist_to_json(dist):
    return {
        'name': dist.dist.name,
        'args': dist.args,
        'kwds': dist.kwds
    }

def spn_from_json(metadata):
    if metadata['class'] == 'NominalDistribution':
        symbol = Identity(metadata['symbol'])
        dist = {x: Fraction(w[0], w[1]) for x, w in metadata['dist']}
        return NominalDistribution(symbol, dist)
    if metadata['class'] == 'ContinuousReal':
        symbol = Identity(metadata['symbol'])
        dist = scipy_dist_from_json(metadata['dist'])
        # from sympy import *
        support = eval(metadata['support'])
        conditioned = metadata['conditioned']
        return ContinuousReal(symbol, dist, support, conditioned)
    if metadata['class'] == 'DiscreteReal':
        symbol = Identity(metadata['symbol'])
        dist = scipy_dist_from_json(metadata['dist'])
        # from sympy import *
        support = eval(metadata['support'])
        conditioned = metadata['conditioned']
        return DiscreteReal(symbol, dist, support, conditioned)
    if metadata['class'] == 'SumSPN':
        children = [spn_from_json(c) for c in metadata['children']]
        weights = metadata['weights']
        return SumSPN(children, weights)
    if metadata['class'] == 'ProductSPN':
        children = [spn_from_json(c) for c in metadata['children']]
        return ProductSPN(children)

    assert False, 'Cannot convert %s to SPN' % (medata,)

def spn_to_json(spn):
    if isinstance(spn, NominalDistribution):
        return {
            'class'        : 'NominalDistribution',
            'symbol'        : spn.symbol.token,
            'dist'         : [
                (str(x), (w.numerator, w.denominator))
                for x, w in spn.dist.items()
            ]
        }
    if isinstance(spn, ContinuousReal):
        return {
            'class'        : 'ContinuousReal',
            'symbol'        : spn.symbol.token,
            'dist'          : scipy_dist_to_json(spn.dist),
            'support'       : str(spn.support),
            'conditioned'   : spn.conditioned,
        }
    if isinstance(spn, DiscreteReal):
        return {
            'class'        : 'DiscreteReal',
            'symbol'        : spn.symbol.token,
            'dist'          : scipy_dist_to_json(spn.dist),
            'support'       : str(spn.support),
            'conditioned'   : spn.conditioned,
        }
    if isinstance(spn, SumSPN):
        return {
            'class'        : 'SumSPN',
            'children'      : [spn_to_json(c) for c in spn.children],
            'weights'       : spn.weights,
        }
    if isinstance(spn, ProductSPN):
        return {
            'class'        : 'ProductSPN',
            'children'      : [spn_to_json(c) for c in spn.children],
        }
    assert False, 'Cannot convert %s to JSON' % (spn,)
