# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

import scipy.stats
import sympy

# ==============================================================================
# ContinuousReal

from .spn import ContinuousReal
from .sym_util import Reals
from .sym_util import RealsNeg
from .sym_util import RealsPos
from .sym_util import UnitInterval

def Alpha(**kwargs):
    """An alpha continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.alpha(**kwargs),
        RealsPos)

def Anglit(**kwargs):
    """An anglit continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.anglit(**kwargs),
        sympy.Interval(-sympy.pi/4, sympy.pi/4))

def Arcsine(**kwargs):
    """An arcsine continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.arcsine(**kwargs),
        UnitInterval)

def Argus(**kwargs):
    """Argus distribution"""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.argus(**kwargs),
        UnitInterval)

def Beta(**kwargs):
    """A beta continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.beta(**kwargs),
        UnitInterval)

def Betaprime(**kwargs):
    """A beta prime continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.betaprime(**kwargs),
        RealsPos)

def Bradford(**kwargs):
    """A Bradford continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.bradford(**kwargs),
        UnitInterval)

def Burr(**kwargs):
    """A Burr (Type III) continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.burr(**kwargs),
        RealsPos)

def Burr12(**kwargs):
    """A Burr (Type XII) continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.burr12(**kwargs),
        RealsPos)

def Cauchy(**kwargs):
    """A Cauchy continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.cauchy(**kwargs),
        sympy.Reals)

def Chi(**kwargs):
    """A chi continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.chi(**kwargs),
        RealsPos)

def Chi2(**kwargs):
    """A chi-squared continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.chi2(**kwargs),
        RealsPos)

def Cosine(**kwargs):
    """A cosine continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.cosine(**kwargs),
        sympy.Interval(-sympy.pi/2, sympy.pi/2))

def Crystalball(**kwargs):
    """Crystalball distribution."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.crystalball(**kwargs),
        sympy.Reals)

def Dgamma(**kwargs):
    """A double gamma continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.dgamma(**kwargs),
        sympy.Reals)

def Dweibull(**kwargs):
    """A double Weibull continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.dweibull(**kwargs),
        sympy.Reals)

def Erlang(**kwargs):
    """An Erlang continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.erlang(**kwargs),
        RealsPos)

def Expon(**kwargs):
    """An exponential continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.expon(**kwargs),
        RealsPos)

def Exponnorm(**kwargs):
    """An exponentially modified Normal continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.exponnorm(**kwargs),
        sympy.Reals)

def Exponweib(**kwargs):
    """An exponentiated Weibull continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.exponweib(**kwargs),
        RealsPos)

def Exponpow(**kwargs):
    """An exponential power continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.exponpow(**kwargs),
        RealsPos)

def F(**kwargs):
    """An F continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.f(**kwargs),
        RealsPos)

def Fatiguelife(**kwargs):
    """A fatigue-life (Birnbaum-Saunders) continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.fatiguelife(**kwargs),
        RealsPos)

def Fisk(**kwargs):
    """A Fisk continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.fisk(**kwargs),
        RealsPos)

def Foldcauchy(**kwargs):
    """A folded Cauchy continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.foldcauchy(**kwargs),
        RealsPos)

def Foldnorm(**kwargs):
    """A folded normal continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.foldnorm(**kwargs),
        RealsPos)

def Frechet_r(**kwargs):
    """A Frechet right (or Weibull minimum) continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.frechet_r(**kwargs),
        RealsPos)

def Frechet_l(**kwargs):
    """A Frechet left (or Weibull maximum) continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.frechet_l(**kwargs),
        sympy.RealsNeg)

def Genlogistic(**kwargs):
    """A generalized logistic continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.genlogistic(**kwargs),
        RealsPos)

def Gennorm(**kwargs):
    """A generalized normal continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.gennorm(**kwargs),
        sympy.Reals)

def Genpareto(**kwargs):
    """A generalized Pareto continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.genpareto(**kwargs),
        RealsPos)

def Genexpon(**kwargs):
    """A generalized exponential continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.genexpon(**kwargs),
        RealsPos)

def Genextreme(**kwargs):
    """A generalized extreme value continuous random variable."""
    c = kwargs['c']
    if c == 0:
        domain = Reals
    elif c > 0:
        domain = sympy.Interval(-sympy.oo, 1/c)
    elif c < 0:
        domain = sympy.Interval(1/c, sympy.oo)
    else:
        assert False, 'Bad argument "c" for genextreme: %s' % (kwargs,)
    return lambda symbol: ContinuousReal(symbol, scipy.stats.genextreme(**kwargs),
        domain)

def Gausshyper(**kwargs):
    """A Gauss hypergeometric continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.gausshyper(**kwargs),
        UnitInterval)

def Gamma(**kwargs):
    """A gamma continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.gamma(**kwargs),
        RealsPos)

def Gengamma(**kwargs):
    """A generalized gamma continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.gengamma(**kwargs),
        RealsPos)

def Genhalflogistic(**kwargs):
    """A generalized half-logistic continuous random variable."""
    assert kwargs['c'] > 0
    return lambda symbol: ContinuousReal(symbol, scipy.stats.genhalflogistic(**kwargs),
        sympy.Interval(0, 1./kwargs['c']))

def Geninvgauss(**kwargs):
    """A Generalized Inverse Gaussian continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.geninvgauss(**kwargs),
        RealsPos)

def Gilbrat(**kwargs):
    """A Gilbrat continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.gilbrat(**kwargs),
        RealsPos)

def Gompertz(**kwargs):
    """A Gompertz (or truncated Gumbel) continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.gompertz(**kwargs),
        RealsPos)

def Gumbel_r(**kwargs):
    """A right-skewed Gumbel continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.gumbel_r(**kwargs),
        sympy.Reals)

def Gumbel_l(**kwargs):
    """A left-skewed Gumbel continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.gumbel_l(**kwargs),
        RealsPos)

def Halfcauchy(**kwargs):
    """A Half-Cauchy continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.halfcauchy(**kwargs),
        RealsPos)

def Halflogistic(**kwargs):
    """A half-logistic continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.halflogistic(**kwargs),
        RealsPos)

def Halfnorm(**kwargs):
    """A half-normal continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.halfnorm(**kwargs),
        RealsPos)

def Halfgennorm(**kwargs):
    """The upper half of a generalized normal continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.halfgennorm(**kwargs),
        RealsPos)

def Hypsecant(**kwargs):
    """A hyperbolic secant continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.hypsecant(**kwargs),
        sympy.Reals)

def Invgamma(**kwargs):
    """An inverted gamma continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.invgamma(**kwargs),
        RealsPos)

def Invgauss(**kwargs):
    """An inverse Gaussian continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.invgauss(**kwargs),
        RealsPos)

def Invweibull(**kwargs):
    """An inverted Weibull continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.invweibull(**kwargs),
        RealsPos)

def Johnsonsb(**kwargs):
    """A Johnson SB continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.johnsonsb(**kwargs),
        UnitInterval)

def Johnsonsu(**kwargs):
    """A Johnson SU continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.johnsonsu(**kwargs),
        Reals)

def Kappa4(**kwargs):
    """Kappa 4 parameter distribution."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.kappa4(**kwargs),
        Reals)

def Kappa3(**kwargs):
    """Kappa 3 parameter distribution."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.kappa3(**kwargs),
        RealsPos)

def Ksone(**kwargs):
    """General Kolmogorov-Smirnov one-sided test."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.ksone(**kwargs),
        UnitInterval)

def Kstwobign(**kwargs):
    """Kolmogorov-Smirnov two-sided test for large N."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.kstwobign(**kwargs),
        sympy.Interval(0, sympy.sqrt(kwargs['n'])))

def Laplace(**kwargs):
    """A Laplace continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.laplace(**kwargs),
        Reals)

def Levy(**kwargs):
    """A Levy continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.levy(**kwargs),
        RealsPos)

def Levy_l(**kwargs):
    """A left-skewed Levy continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.levy_l(**kwargs),
        RealsNeg)

def Levy_stable(**kwargs):
    """A Levy-stable continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.levy_stable(**kwargs),
        Reals)

def Logistic(**kwargs):
    """A logistic (or Sech-squared) continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.logistic(**kwargs),
        Reals)

def Loggamma(**kwargs):
    """A log gamma continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.loggamma(**kwargs),
        RealsNeg)

def Loglaplace(**kwargs):
    """A log-Laplace continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.loglaplace(**kwargs),
        RealsPos)

def Lognorm(**kwargs):
    """A lognormal continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.lognorm(**kwargs),
        RealsPos)

def Loguniform(**kwargs):
    """A loguniform or reciprocal continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.loguniform(**kwargs),
        sympy.Interval(kwargs['a'], kwargs['b']))

def Lomax(**kwargs):
    """A Lomax (Pareto of the second kind) continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.lomax(**kwargs),
        RealsPos)

def Maxwell(**kwargs):
    """A Maxwell continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.maxwell(**kwargs),
        RealsPos)

def Mielke(**kwargs):
    """A Mielke Beta-Kappa / Dagum continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.mielke(**kwargs),
        RealsPos)

def Moyal(**kwargs):
    """A Moyal continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.moyal(**kwargs),
        Reals)

def Nakagami(**kwargs):
    """A Nakagami continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.nakagami(**kwargs),
        RealsPos)

def Ncx2(**kwargs):
    """A non-central chi-squared continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.ncx2(**kwargs),
        RealsPos)

def Ncf(**kwargs):
    """A non-central F distribution continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.ncf(**kwargs),
        RealsPos)

def Nct(**kwargs):
    """A non-central Student’s t continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.nct(**kwargs),
        Reals)

def Norm(**kwargs):
    """A normal continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.norm(**kwargs),
        Reals)

def Norminvgauss(**kwargs):
    """A Normal Inverse Gaussian continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.norminvgauss(**kwargs),
        Reals)

def Pareto(**kwargs):
    """A Pareto continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.pareto(**kwargs),
        sympy.Interval(1, sympy.oo))

def Pearson3(**kwargs):
    """A pearson type III continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.pearson3(**kwargs),
        Reals)

def Powerlaw(**kwargs):
    """A power-function continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.powerlaw(**kwargs),
        UnitInterval)

def Powerlognorm(**kwargs):
    """A power log-normal continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.powerlognorm(**kwargs),
        RealsPos)

def Powernorm(**kwargs):
    """A power normal continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.powernorm(**kwargs),
        RealsPos)

def Rdist(**kwargs):
    """An R-distributed (symmetric beta) continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.rdist(**kwargs),
        sympy.Interval(-1, 1))

def Rayleigh(**kwargs):
    """A Rayleigh continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.rayleigh(**kwargs),
        RealsPos)

def Rice(**kwargs):
    """A Rice continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.rice(**kwargs),
        RealsPos)

def Recipinvgauss(**kwargs):
    """A reciprocal inverse Gaussian continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.recipinvgauss(**kwargs),
        RealsPos)

def Semicircular(**kwargs):
    """A semicircular continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.semicircular(**kwargs),
        sympy.Interval(-1, 1))

def Skewnorm(**kwargs):
    """A skew-normal random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.skewnorm(**kwargs),
        Reals)

def T(**kwargs):
    """A Student’s t continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.t(**kwargs),
        Reals)

def Trapz(**kwargs):
    """A trapezoidal continuous random variable."""
    loc = kwargs.get('loc', 0)
    scale = kwargs.get('scale', 1)
    return lambda symbol: ContinuousReal(symbol, scipy.stats.trapz(**kwargs),
        sympy.Interval(loc, loc+scale))

def Triang(**kwargs):
    """A triangular continuous random variable."""
    loc = kwargs.get('loc', 0)
    scale = kwargs.get('scale', 1)
    return lambda symbol: ContinuousReal(symbol, scipy.stats.triang(**kwargs),
        sympy.Interval(loc, loc+scale))

def Truncexpon(**kwargs):
    """A truncated exponential continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.truncexpon(**kwargs),
        sympy.Interval(0, kwargs['b']))

def Truncnorm(**kwargs):
    """A truncated normal continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.truncnorm(**kwargs),
        sympy.Interval(kwargs['a'], kwargs['b']))

def Tukeylambda(**kwargs):
    """A Tukey-Lamdba continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.tukeylambda(**kwargs),
        RealsPos)

def Uniform(**kwargs):
    """A uniform continuous random variable."""
    loc = kwargs.get('loc', 0)
    scale = kwargs.get('scale', 1)
    return lambda symbol: ContinuousReal(symbol, scipy.stats.uniform(**kwargs),
        sympy.Interval(loc, loc + scale))

def Vonmises(**kwargs):
    """A Von Mises continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.vonmises(**kwargs),
        sympy.Interval(-sympy.pi, sympy.pi))

def Vonmises_line(**kwargs):
    """A Von Mises continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.vonmises_line(**kwargs),
        sympy.Interval(-sympy.pi, sympy.pi))

def Wald(**kwargs):
    """A Wald continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.wald(**kwargs),
        RealsPos)

def Weibull_min(**kwargs):
    """Weibull minimum continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.weibull_min(**kwargs),
        RealsPos)

def Weibull_max(**kwargs):
    """Weibull maximum continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.weibull_max(**kwargs),
        RealsNeg)

def Wrapcauchy(**kwargs):
    """A wrapped Cauchy continuous random variable."""
    return lambda symbol: ContinuousReal(symbol, scipy.stats.wrapcauchy(**kwargs),
        sympy.Interval(0, 2*sympy.pi))

# ==============================================================================
# DiscreteReal

from .spn import DiscreteReal
from .sym_util import Integers
from .sym_util import IntegersPos
from .sym_util import IntegersPos0

def Bernoulli(**kwargs):
    """A Bernoulli discrete random variable."""
    return lambda symbol: DiscreteReal(symbol, scipy.stats.bernoulli(**kwargs),
        sympy.Range(0, 2))

def Betabinom(**kwargs):
    """A beta-binomial discrete random variable."""
    return lambda symbol: DiscreteReal(symbol, scipy.stats.betabinom(**kwargs),
        sympy.Range(0, kwargs['n']+1))

def Binom(**kwargs):
    """A binomial discrete random variable."""
    return lambda symbol: DiscreteReal(symbol, scipy.stats.binom(**kwargs),
        sympy.Range(0, kwargs['n']+1))

def Boltzmann(**kwargs):
    """A Boltzmann (Truncated Discrete Exponential) random variable."""
    return lambda symbol: DiscreteReal(symbol, scipy.stats.boltzmann(**kwargs),
        sympy.Range(0, kwargs['N']+1))

def Dlaplace(**kwargs):
    """A Laplacian discrete random variable."""
    return lambda symbol: DiscreteReal(symbol, scipy.stats.dlaplace(**kwargs),
        Integers)

def Geom(**kwargs):
    """A geometric discrete random variable."""
    return lambda symbol: DiscreteReal(symbol, scipy.stats.geom(**kwargs),
        Integers)

def Hypergeom(**kwargs):
    """A hypergeometric discrete random variable."""
    low = max(0, kwargs['N'], kwargs['N']-kwargs['M']+kwargs['n'])
    high = min(kwargs['n'], kwargs['N'])
    return lambda symbol: DiscreteReal(symbol, scipy.stats.hypergeom(**kwargs),
        sympy.Range(low, high+1))

def Logser(**kwargs):
    """A Logarithmic (Log-Series, Series) discrete random variable."""
    return lambda symbol: DiscreteReal(symbol, scipy.stats.logser(**kwargs),
        IntegersPos)

def Nbinom(**kwargs):
    """A negative binomial discrete random variable."""
    return lambda symbol: DiscreteReal(symbol, scipy.stats.nbinom(**kwargs),
        IntegersPos0)

def Planck(**kwargs):
    """A Planck discrete exponential random variable."""
    return lambda symbol: DiscreteReal(symbol, scipy.stats.planck(**kwargs),
        IntegersPos0)

def Poisson(**kwargs):
    """A Poisson discrete random variable."""
    return lambda symbol: DiscreteReal(symbol, scipy.stats.poisson(**kwargs),
        IntegersPos0)

def Randint(**kwargs):
    """A uniform discrete random variable."""
    return lambda symbol: DiscreteReal(symbol, scipy.stats.randint(**kwargs),
        sympy.Range(kwargs['low'], kwargs['high']))

def Skellam(**kwargs):
    """A Skellam discrete random variable."""
    return lambda symbol: DiscreteReal(symbol, scipy.stats.skellam(**kwargs),
        Integers)

def Zipf(**kwargs):
    """A Zipf discrete random variable."""
    return lambda symbol: DiscreteReal(symbol, scipy.stats.zipf(**kwargs),
        IntegersPos)

def Yulesimon(**kwargs):
    """A Yule-Simon discrete random variable."""
    return lambda symbol: DiscreteReal(symbol, scipy.stats.yulesimon(**kwargs),
        IntegersPos)

def Atomic(**kwargs):
    """A Yule-Simon discrete random variable."""
    return Randint(low=kwargs['loc'], high=kwargs['loc']+1)

# ==============================================================================
# Nominal

from .spn import NominalDistribution

def NominalDist(probs):
    return lambda symbol: NominalDistribution(symbol, probs)
