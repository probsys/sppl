# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

import scipy.stats
import sympy

from .spn import NumericalDistribution

from .sym_util import Reals
from .sym_util import RealsNeg
from .sym_util import RealsPos
from .sym_util import UnitInterval

def Alpha(symbol, **kwargs):
    """An alpha continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.alpha(**kwargs),
        RealsPos)

def Anglit(symbol, **kwargs):
    """An anglit continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.anglit(**kwargs),
        sympy.Interval(-sympy.pi/4, sympy.pi/4))

def Arcsine(symbol, **kwargs):
    """An arcsine continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.arcsine(**kwargs),
        UnitInterval)

def Argus(symbol, **kwargs):
    """Argus distribution"""
    return NumericalDistribution(symbol, scipy.stats.argus(**kwargs),
        UnitInterval)

def Beta(symbol, **kwargs):
    """A beta continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.beta(**kwargs),
        UnitInterval)

def Betaprime(symbol, **kwargs):
    """A beta prime continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.betaprime(**kwargs),
        RealsPos)

def Bradford(symbol, **kwargs):
    """A Bradford continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.bradford(**kwargs),
        UnitInterval)

def Burr(symbol, **kwargs):
    """A Burr (Type III) continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.burr(**kwargs),
        RealsPos)

def Burr12(symbol, **kwargs):
    """A Burr (Type XII) continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.burr12(**kwargs),
        RealsPos)

def Cauchy(symbol, **kwargs):
    """A Cauchy continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.cauchy(**kwargs),
        sympy.Reals)

def Chi(symbol, **kwargs):
    """A chi continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.chi(**kwargs),
        RealsPos)

def Chi2(symbol, **kwargs):
    """A chi-squared continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.chi2(**kwargs),
        RealsPos)

def Cosine(symbol, **kwargs):
    """A cosine continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.cosine(**kwargs),
        sympy.Interval(-sympy.pi/2, sympy.pi/2))

def Crystalball(symbol, **kwargs):
    """Crystalball distribution."""
    return NumericalDistribution(symbol, scipy.stats.crystalball(**kwargs),
        sympy.Reals)

def Dgamma(symbol, **kwargs):
    """A double gamma continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.dgamma(**kwargs),
        sympy.Reals)

def Dweibull(symbol, **kwargs):
    """A double Weibull continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.dweibull(**kwargs),
        sympy.Reals)

def Erlang(symbol, **kwargs):
    """An Erlang continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.erlang(**kwargs),
        RealsPos)

def Expon(symbol, **kwargs):
    """An exponential continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.expon(**kwargs),
        RealsPos)

def Exponnorm(symbol, **kwargs):
    """An exponentially modified Normal continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.exponnorm(**kwargs),
        sympy.Reals)

def Exponweib(symbol, **kwargs):
    """An exponentiated Weibull continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.exponweib(**kwargs),
        RealsPos)

def Exponpow(symbol, **kwargs):
    """An exponential power continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.exponpow(**kwargs),
        RealsPos)

def F(symbol, **kwargs):
    """An F continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.f(**kwargs),
        RealsPos)

def Fatiguelife(symbol, **kwargs):
    """A fatigue-life (Birnbaum-Saunders) continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.fatiguelife(**kwargs),
        RealsPos)

def Fisk(symbol, **kwargs):
    """A Fisk continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.fisk(**kwargs),
        RealsPos)

def Foldcauchy(symbol, **kwargs):
    """A folded Cauchy continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.foldcauchy(**kwargs),
        RealsPos)

def Foldnorm(symbol, **kwargs):
    """A folded normal continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.foldnorm(**kwargs),
        RealsPos)

def Frechet_r(symbol, **kwargs):
    """A Frechet right (or Weibull minimum) continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.frechet_r(**kwargs),
        RealsPos)

def Frechet_l(symbol, **kwargs):
    """A Frechet left (or Weibull maximum) continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.frechet_l(**kwargs),
        sympy.RealsNeg)

def Genlogistic(symbol, **kwargs):
    """A generalized logistic continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.genlogistic(**kwargs),
        RealsPos)

def Gennorm(symbol, **kwargs):
    """A generalized normal continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.gennorm(**kwargs),
        sympy.Reals)

def Genpareto(symbol, **kwargs):
    """A generalized Pareto continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.genpareto(**kwargs),
        RealsPos)

def Genexpon(symbol, **kwargs):
    """A generalized exponential continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.genexpon(**kwargs),
        RealsPos)

def Genextreme(symbol, **kwargs):
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
    return NumericalDistribution(symbol, scipy.stats.genextreme(**kwargs),
        domain)

def Gausshyper(symbol, **kwargs):
    """A Gauss hypergeometric continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.gausshyper(**kwargs),
        UnitInterval)

def Gamma(symbol, **kwargs):
    """A gamma continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.gamma(**kwargs),
        RealsPos)

def Gengamma(symbol, **kwargs):
    """A generalized gamma continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.gengamma(**kwargs),
        RealsPos)

def Genhalflogistic(symbol, **kwargs):
    """A generalized half-logistic continuous random variable."""
    assert kwargs['c'] > 0
    return NumericalDistribution(symbol, scipy.stats.genhalflogistic(**kwargs),
        sympy.Interval(0, 1./kwargs['c']))

def Geninvgauss(symbol, **kwargs):
    """A Generalized Inverse Gaussian continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.geninvgauss(**kwargs),
        RealsPos)

def Gilbrat(symbol, **kwargs):
    """A Gilbrat continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.gilbrat(**kwargs),
        RealsPos)

def Gompertz(symbol, **kwargs):
    """A Gompertz (or truncated Gumbel) continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.gompertz(**kwargs),
        RealsPos)

def Gumbel_r(symbol, **kwargs):
    """A right-skewed Gumbel continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.gumbel_r(**kwargs),
        sympy.Reals)

def Gumbel_l(symbol, **kwargs):
    """A left-skewed Gumbel continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.gumbel_l(**kwargs),
        RealsPos)

def Halfcauchy(symbol, **kwargs):
    """A Half-Cauchy continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.halfcauchy(**kwargs),
        RealsPos)

def Halflogistic(symbol, **kwargs):
    """A half-logistic continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.halflogistic(**kwargs),
        RealsPos)

def Halfnorm(symbol, **kwargs):
    """A half-normal continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.halfnorm(**kwargs),
        RealsPos)

def Halfgennorm(symbol, **kwargs):
    """The upper half of a generalized normal continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.halfgennorm(**kwargs),
        RealsPos)

def Hypsecant(symbol, **kwargs):
    """A hyperbolic secant continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.hypsecant(**kwargs),
        sympy.Reals)

def Invgamma(symbol, **kwargs):
    """An inverted gamma continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.invgamma(**kwargs),
        RealsPos)

def Invgauss(symbol, **kwargs):
    """An inverse Gaussian continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.invgauss(**kwargs),
        RealsPos)

def Invweibull(symbol, **kwargs):
    """An inverted Weibull continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.invweibull(**kwargs),
        RealsPos)

def Johnsonsb(symbol, **kwargs):
    """A Johnson SB continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.johnsonsb(**kwargs),
        UnitInterval)

def Johnsonsu(symbol, **kwargs):
    """A Johnson SU continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.johnsonsu(**kwargs),
        Reals)

def Kappa4(symbol, **kwargs):
    """Kappa 4 parameter distribution."""
    return NumericalDistribution(symbol, scipy.stats.kappa4(**kwargs),
        Reals)

def Kappa3(symbol, **kwargs):
    """Kappa 3 parameter distribution."""
    return NumericalDistribution(symbol, scipy.stats.kappa3(**kwargs),
        RealsPos)

def Ksone(symbol, **kwargs):
    """General Kolmogorov-Smirnov one-sided test."""
    return NumericalDistribution(symbol, scipy.stats.ksone(**kwargs),
        UnitInterval)

def Kstwobign(symbol, **kwargs):
    """Kolmogorov-Smirnov two-sided test for large N."""
    return NumericalDistribution(symbol, scipy.stats.kstwobign(**kwargs),
        sympy.Interval(0, sympy.sqrt(kwargs['n'])))

def Laplace(symbol, **kwargs):
    """A Laplace continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.laplace(**kwargs),
        Reals)

def Levy(symbol, **kwargs):
    """A Levy continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.levy(**kwargs),
        RealsPos)

def Levy_l(symbol, **kwargs):
    """A left-skewed Levy continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.levy_l(**kwargs),
        RealsNeg)

def Levy_stable(symbol, **kwargs):
    """A Levy-stable continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.levy_stable(**kwargs),
        Reals)

def Logistic(symbol, **kwargs):
    """A logistic (or Sech-squared) continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.logistic(**kwargs),
        Reals)

def Loggamma(symbol, **kwargs):
    """A log gamma continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.loggamma(**kwargs),
        RealsNeg)

def Loglaplace(symbol, **kwargs):
    """A log-Laplace continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.loglaplace(**kwargs),
        RealsPos)

def Lognorm(symbol, **kwargs):
    """A lognormal continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.lognorm(**kwargs),
        RealsPos)

def Loguniform(symbol, **kwargs):
    """A loguniform or reciprocal continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.loguniform(**kwargs),
        sympy.Interval(kwargs['a'], kwargs['b']))

def Lomax(symbol, **kwargs):
    """A Lomax (Pareto of the second kind) continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.lomax(**kwargs),
        RealsPos)

def Maxwell(symbol, **kwargs):
    """A Maxwell continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.maxwell(**kwargs),
        RealsPos)

def Mielke(symbol, **kwargs):
    """A Mielke Beta-Kappa / Dagum continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.mielke(**kwargs),
        RealsPos)

def Moyal(symbol, **kwargs):
    """A Moyal continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.moyal(**kwargs),
        Reals)

def Nakagami(symbol, **kwargs):
    """A Nakagami continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.nakagami(**kwargs),
        RealsPos)

def Ncx2(symbol, **kwargs):
    """A non-central chi-squared continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.ncx2(**kwargs),
        RealsPos)

def Ncf(symbol, **kwargs):
    """A non-central F distribution continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.ncf(**kwargs),
        RealsPos)

def Nct(symbol, **kwargs):
    """A non-central Student’s t continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.nct(**kwargs),
        Reals)

def Norm(symbol, **kwargs):
    """A normal continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.norm(**kwargs),
        Reals)

def Norminvgauss(symbol, **kwargs):
    """A Normal Inverse Gaussian continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.norminvgauss(**kwargs),
        Reals)

def Pareto(symbol, **kwargs):
    """A Pareto continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.pareto(**kwargs),
        sympy.Interval(1, sympy.oo))

def Pearson3(symbol, **kwargs):
    """A pearson type III continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.pearson3(**kwargs),
        Reals)

def Powerlaw(symbol, **kwargs):
    """A power-function continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.powerlaw(**kwargs),
        UnitInterval)

def Powerlognorm(symbol, **kwargs):
    """A power log-normal continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.powerlognorm(**kwargs),
        RealsPos)

def Powernorm(symbol, **kwargs):
    """A power normal continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.powernorm(**kwargs),
        RealsPos)

def Rdist(symbol, **kwargs):
    """An R-distributed (symmetric beta) continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.rdist(**kwargs),
        sympy.Interval(-1, 1))

def Rayleigh(symbol, **kwargs):
    """A Rayleigh continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.rayleigh(**kwargs),
        RealsPos)

def Rice(symbol, **kwargs):
    """A Rice continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.rice(**kwargs),
        RealsPos)

def Recipinvgauss(symbol, **kwargs):
    """A reciprocal inverse Gaussian continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.recipinvgauss(**kwargs),
        RealsPos)

def Semicircular(symbol, **kwargs):
    """A semicircular continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.semicircular(**kwargs),
        sympy.Interval(-1, 1))

def Skewnorm(symbol, **kwargs):
    """A skew-normal random variable."""
    return NumericalDistribution(symbol, scipy.stats.skewnorm(**kwargs),
        Reals)

def T(symbol, **kwargs):
    """A Student’s t continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.t(**kwargs),
        Reals)

def Trapz(symbol, **kwargs):
    """A trapezoidal continuous random variable."""
    loc = kwargs.get('loc', 0)
    scale = kwargs.get('scale', 1)
    return NumericalDistribution(symbol, scipy.stats.trapz(**kwargs),
        sympy.Interval(loc, loc+scale))

def Triang(symbol, **kwargs):
    """A triangular continuous random variable."""
    loc = kwargs.get('loc', 0)
    scale = kwargs.get('scale', 1)
    return NumericalDistribution(symbol, scipy.stats.triang(**kwargs),
        sympy.Interval(loc, loc+scale))

def Truncexpon(symbol, **kwargs):
    """A truncated exponential continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.truncexpon(**kwargs),
        sympy.Interval(0, kwargs['b']))

def Truncnorm(symbol, **kwargs):
    """A truncated normal continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.truncnorm(**kwargs),
        sympy.Interval(kwargs['a'], kwargs['b']))

def Tukeylambda(symbol, **kwargs):
    """A Tukey-Lamdba continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.tukeylambda(**kwargs),
        RealsPos)

def Uniform(symbol, **kwargs):
    """A uniform continuous random variable."""
    loc = kwargs.get('loc', 0)
    scale = kwargs.get('scale', 1)
    return NumericalDistribution(symbol, scipy.stats.uniform(**kwargs),
        sympy.Interval(loc, loc + scale))

def Vonmises(symbol, **kwargs):
    """A Von Mises continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.vonmises(**kwargs),
        sympy.Interval(-sympy.pi, sympy.pi))

def Vonmises_line(symbol, **kwargs):
    """A Von Mises continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.vonmises_line(**kwargs),
        sympy.Interval(-sympy.pi, sympy.pi))

def Wald(symbol, **kwargs):
    """A Wald continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.wald(**kwargs),
        RealsPos)

def Weibull_min(symbol, **kwargs):
    """Weibull minimum continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.weibull_min(**kwargs),
        RealsPos)

def Weibull_max(symbol, **kwargs):
    """Weibull maximum continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.weibull_max(**kwargs),
        RealsNeg)

def Wrapcauchy(symbol, **kwargs):
    """A wrapped Cauchy continuous random variable."""
    return NumericalDistribution(symbol, scipy.stats.wrapcauchy(**kwargs),
        sympy.Interval(0, 2*sympy.pi))
