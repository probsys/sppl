# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

import scipy.stats
import sympy

# pylint: disable=not-callable
# pylint: disable=multiple-statements
class RealDistribution():
    dist = None
    constructor = None
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    def __call__(self, symbol):
        domain = self.get_domain(**self.kwargs)
        return self.constructor(symbol, self.dist(**self.kwargs), domain)
    def get_domain(self, **kwargs):
        raise NotImplementedError()

# ==============================================================================
# ContinuousReal

from .spn import ContinuousLeaf
from .sym_util import Reals
from .sym_util import RealsNeg
from .sym_util import RealsPos
from .sym_util import UnitInterval

class ContinuousReal(RealDistribution):
    constructor = ContinuousLeaf

class Alpha(ContinuousReal):
    """An alpha continuous random variable."""
    dist = scipy.stats.alpha
    def get_domain(self, **kwargs): return RealsPos

class Anglit(ContinuousReal):
    """An anglit continuous random variable."""
    dist = scipy.stats.anglit
    def get_domain(self, **kwargs): return sympy.Interval(-sympy.pi/4, sympy.pi/4)

class Arcsine(ContinuousReal):
    """An arcsine continuous random variable."""
    dist = scipy.stats.arcsine
    def get_domain(self, **kwargs): return UnitInterval

class Argus(ContinuousReal):
    """Argus distribution"""
    dist = scipy.stats.argus
    def get_domain(self, **kwargs): return UnitInterval

class Beta(ContinuousReal):
    """A beta continuous random variable."""
    dist = scipy.stats.beta
    def get_domain(self, **kwargs): return UnitInterval

class Betaprime(ContinuousReal):
    """A beta prime continuous random variable."""
    dist = scipy.stats.betaprime
    def get_domain(self, **kwargs): return RealsPos

class Bradford(ContinuousReal):
    """A Bradford continuous random variable."""
    dist = scipy.stats.bradford
    def get_domain(self, **kwargs): return UnitInterval

class Burr(ContinuousReal):
    """A Burr (Type III) continuous random variable."""
    dist = scipy.stats.burr
    def get_domain(self, **kwargs): return RealsPos

class Burr12(ContinuousReal):
    """A Burr (Type XII) continuous random variable."""
    dist = scipy.stats.burr12
    def get_domain(self, **kwargs): return RealsPos

class Cauchy(ContinuousReal):
    """A Cauchy continuous random variable."""
    dist = scipy.stats.cauchy
    def get_domain(self, **kwargs): return sympy.Reals

class Chi(ContinuousReal):
    """A chi continuous random variable."""
    dist = scipy.stats.chi
    def get_domain(self, **kwargs): return RealsPos

class Chi2(ContinuousReal):
    """A chi-squared continuous random variable."""
    dist = scipy.stats.chi2
    def get_domain(self, **kwargs): return RealsPos

class Cosine(ContinuousReal):
    """A cosine continuous random variable."""
    dist = scipy.stats.cosine
    def get_domain(self, **kwargs): return sympy.Interval(-sympy.pi/2, sympy.pi/2)

class Crystalball(ContinuousReal):
    """Crystalball distribution."""
    dist = scipy.stats.crystalball
    def get_domain(self, **kwargs): return sympy.Reals

class Dgamma(ContinuousReal):
    """A double gamma continuous random variable."""
    dist = scipy.stats.dgamma
    def get_domain(self, **kwargs): return sympy.Reals

class Dweibull(ContinuousReal):
    """A double Weibull continuous random variable."""
    dist = scipy.stats.dweibull
    def get_domain(self, **kwargs): return sympy.Reals

class Erlang(ContinuousReal):
    """An Erlang continuous random variable."""
    dist = scipy.stats.erlang
    def get_domain(self, **kwargs): return RealsPos

class Expon(ContinuousReal):
    """An exponential continuous random variable."""
    dist = scipy.stats.expon
    def get_domain(self, **kwargs): return RealsPos

class Exponnorm(ContinuousReal):
    """An exponentially modified Normal continuous random variable."""
    dist = scipy.stats.exponnorm
    def get_domain(self, **kwargs): return sympy.Reals

class Exponweib(ContinuousReal):
    """An exponentiated Weibull continuous random variable."""
    dist = scipy.stats.exponweib
    def get_domain(self, **kwargs): return RealsPos

class Exponpow(ContinuousReal):
    """An exponential power continuous random variable."""
    dist = scipy.stats.exponpow
    def get_domain(self, **kwargs): return RealsPos

class F(ContinuousReal):
    """An F continuous random variable."""
    dist = scipy.stats.f
    def get_domain(self, **kwargs): return RealsPos

class Fatiguelife(ContinuousReal):
    """A fatigue-life (Birnbaum-Saunders) continuous random variable."""
    dist = scipy.stats.fatiguelife
    def get_domain(self, **kwargs): return RealsPos

class Fisk(ContinuousReal):
    """A Fisk continuous random variable."""
    dist = scipy.stats.fisk
    def get_domain(self, **kwargs): return RealsPos

class Foldcauchy(ContinuousReal):
    """A folded Cauchy continuous random variable."""
    dist = scipy.stats.foldcauchy
    def get_domain(self, **kwargs): return RealsPos

class Foldnorm(ContinuousReal):
    """A folded normal continuous random variable."""
    dist = scipy.stats.foldnorm
    def get_domain(self, **kwargs): return RealsPos

class Frechet_r(ContinuousReal):
    """A Frechet right (or Weibull minimum) continuous random variable."""
    dist = scipy.stats.frechet_r
    def get_domain(self, **kwargs): return RealsPos

class Frechet_l(ContinuousReal):
    """A Frechet left (or Weibull maximum) continuous random variable."""
    dist = scipy.stats.frechet_l
    def get_domain(self, **kwargs): return sympy.RealsNeg

class Genlogistic(ContinuousReal):
    """A generalized logistic continuous random variable."""
    dist = scipy.stats.genlogistic
    def get_domain(self, **kwargs): return RealsPos

class Gennorm(ContinuousReal):
    """A generalized normal continuous random variable."""
    dist = scipy.stats.gennorm
    def get_domain(self, **kwargs): return sympy.Reals

class Genpareto(ContinuousReal):
    """A generalized Pareto continuous random variable."""
    dist = scipy.stats.genpareto
    def get_domain(self, **kwargs): return RealsPos

class Genexpon(ContinuousReal):
    """A generalized exponential continuous random variable."""
    dist = scipy.stats.genexpon
    def get_domain(self, **kwargs): return RealsPos

class Genextreme(ContinuousReal):
    """A generalized extreme value continuous random variable."""
    dist = scipy.stats.genextreme
    def get_domain(self, **kwargs):
        c = kwargs['c']
        if c == 0:
            return Reals
        elif c > 0:
            return sympy.Interval(-sympy.oo, 1/c)
        elif c < 0:
            return sympy.Interval(1/c, sympy.oo)
        assert False, 'Bad argument "c" for genextreme: %s' % (kwargs,)

class Gausshyper(ContinuousReal):
    """A Gauss hypergeometric continuous random variable."""
    dist = scipy.stats.gausshyper
    def get_domain(self, **kwargs): return UnitInterval


class Gamma(ContinuousReal):
    """A gamma continuous random variable."""
    dist = scipy.stats.gamma
    def get_domain(self, **kwargs): return RealsPos

class Gengamma(ContinuousReal):
    """A generalized gamma continuous random variable."""
    dist = scipy.stats.gengamma
    def get_domain(self, **kwargs): return RealsPos

class Genhalflogistic(ContinuousReal):
    """A generalized half-logistic continuous random variable."""
    dist = scipy.stats.genhalflogistic
    def get_domain(self, **kwargs):
        assert kwargs['c'] > 0
        return sympy.Interval(0, 1./kwargs['c'])

class Geninvgauss(ContinuousReal):
    """A Generalized Inverse Gaussian continuous random variable."""
    dist = scipy.stats.geninvgauss
    def get_domain(self, **kwargs): return RealsPos

class Gilbrat(ContinuousReal):
    """A Gilbrat continuous random variable."""
    dist = scipy.stats.gilbrat
    def get_domain(self, **kwargs): return RealsPos

class Gompertz(ContinuousReal):
    """A Gompertz (or truncated Gumbel) continuous random variable."""
    dist = scipy.stats.gompertz
    def get_domain(self, **kwargs): return RealsPos

class Gumbel_r(ContinuousReal):
    """A right-skewed Gumbel continuous random variable."""
    dist = scipy.stats.gumbel_r
    def get_domain(self, **kwargs): return sympy.Reals

class Gumbel_l(ContinuousReal):
    """A left-skewed Gumbel continuous random variable."""
    dist = scipy.stats.gumbel_l
    def get_domain(self, **kwargs): return RealsPos

class Halfcauchy(ContinuousReal):
    """A Half-Cauchy continuous random variable."""
    dist = scipy.stats.halfcauchy
    def get_domain(self, **kwargs): return RealsPos

class Halflogistic(ContinuousReal):
    """A half-logistic continuous random variable."""
    dist = scipy.stats.halflogistic
    def get_domain(self, **kwargs): return RealsPos

class Halfnorm(ContinuousReal):
    """A half-normal continuous random variable."""
    dist = scipy.stats.halfnorm
    def get_domain(self, **kwargs): return RealsPos

class Halfgennorm(ContinuousReal):
    """The upper half of a generalized normal continuous random variable."""
    dist = scipy.stats.halfgennorm
    def get_domain(self, **kwargs): return RealsPos

class Hypsecant(ContinuousReal):
    """A hyperbolic secant continuous random variable."""
    dist = scipy.stats.hypsecant
    def get_domain(self, **kwargs): return sympy.Reals

class Invgamma(ContinuousReal):
    """An inverted gamma continuous random variable."""
    dist = scipy.stats.invgamma
    def get_domain(self, **kwargs): return RealsPos

class Invgauss(ContinuousReal):
    """An inverse Gaussian continuous random variable."""
    dist = scipy.stats.invgauss
    def get_domain(self, **kwargs): return RealsPos

class Invweibull(ContinuousReal):
    """An inverted Weibull continuous random variable."""
    dist = scipy.stats.invweibull
    def get_domain(self, **kwargs): return RealsPos

class Johnsonsb(ContinuousReal):
    """A Johnson SB continuous random variable."""
    dist = scipy.stats.johnsonsb
    def get_domain(self, **kwargs): return UnitInterval

class Johnsonsu(ContinuousReal):
    """A Johnson SU continuous random variable."""
    dist = scipy.stats.johnsonsu
    def get_domain(self, **kwargs): return Reals

class Kappa4(ContinuousReal):
    """Kappa 4 parameter distribution."""
    dist = scipy.stats.kappa4
    def get_domain(self, **kwargs): return Reals

class Kappa3(ContinuousReal):
    """Kappa 3 parameter distribution."""
    dist = scipy.stats.kappa3
    def get_domain(self, **kwargs): return RealsPos

class Ksone(ContinuousReal):
    """General Kolmogorov-Smirnov one-sided test."""
    dist = scipy.stats.ksone
    def get_domain(self, **kwargs): return UnitInterval

class Kstwobign(ContinuousReal):
    """Kolmogorov-Smirnov two-sided test for large N."""
    dist = scipy.stats.kstwobign
    def get_domain(self, **kwargs): return sympy.Interval(0, sympy.sqrt(kwargs['n']))

class Laplace(ContinuousReal):
    """A Laplace continuous random variable."""
    dist = scipy.stats.laplace
    def get_domain(self, **kwargs): return Reals

class Levy(ContinuousReal):
    """A Levy continuous random variable."""
    dist = scipy.stats.levy
    def get_domain(self, **kwargs): return RealsPos

class Levy_l(ContinuousReal):
    """A left-skewed Levy continuous random variable."""
    dist = scipy.stats.levy_l
    def get_domain(self, **kwargs): return RealsNeg

class Levy_stable(ContinuousReal):
    """A Levy-stable continuous random variable."""
    dist = scipy.stats.levy_stable
    def get_domain(self, **kwargs): return Reals

class Logistic(ContinuousReal):
    """A logistic (or Sech-squared) continuous random variable."""
    dist = scipy.stats.logistic
    def get_domain(self, **kwargs): return Reals

class Loggamma(ContinuousReal):
    """A log gamma continuous random variable."""
    dist = scipy.stats.loggamma
    def get_domain(self, **kwargs): return RealsNeg

class Loglaplace(ContinuousReal):
    """A log-Laplace continuous random variable."""
    dist = scipy.stats.loglaplace
    def get_domain(self, **kwargs): return RealsPos

class Lognorm(ContinuousReal):
    """A lognormal continuous random variable."""
    dist = scipy.stats.lognorm
    def get_domain(self, **kwargs): return RealsPos

class Loguniform(ContinuousReal):
    """A loguniform or reciprocal continuous random variable."""
    dist = scipy.stats.loguniform
    def get_domain(self, **kwargs): return sympy.Interval(kwargs['a'], kwargs['b'])

class Lomax(ContinuousReal):
    """A Lomax (Pareto of the second kind) continuous random variable."""
    dist = scipy.stats.lomax
    def get_domain(self, **kwargs): return RealsPos

class Maxwell(ContinuousReal):
    """A Maxwell continuous random variable."""
    dist = scipy.stats.maxwell
    def get_domain(self, **kwargs): return RealsPos

class Mielke(ContinuousReal):
    """A Mielke Beta-Kappa / Dagum continuous random variable."""
    dist = scipy.stats.mielke
    def get_domain(self, **kwargs): return RealsPos

class Moyal(ContinuousReal):
    """A Moyal continuous random variable."""
    dist = scipy.stats.moyal
    def get_domain(self, **kwargs): return Reals

class Nakagami(ContinuousReal):
    """A Nakagami continuous random variable."""
    dist = scipy.stats.nakagami
    def get_domain(self, **kwargs): return RealsPos

class Ncx2(ContinuousReal):
    """A non-central chi-squared continuous random variable."""
    dist = scipy.stats.ncx2
    def get_domain(self, **kwargs): return RealsPos

class Ncf(ContinuousReal):
    """A non-central F distribution continuous random variable."""
    dist = scipy.stats.ncf
    def get_domain(self, **kwargs): return RealsPos

class Nct(ContinuousReal):
    """A non-central Student’s t continuous random variable."""
    dist = scipy.stats.nct
    def get_domain(self, **kwargs): return Reals

class Norm(ContinuousReal):
    """A normal continuous random variable."""
    dist = scipy.stats.norm
    def get_domain(self, **kwargs): return Reals

class Norminvgauss(ContinuousReal):
    """A Normal Inverse Gaussian continuous random variable."""
    dist = scipy.stats.norminvgauss
    def get_domain(self, **kwargs): return Reals

class Pareto(ContinuousReal):
    """A Pareto continuous random variable."""
    dist = scipy.stats.pareto
    def get_domain(self, **kwargs): return sympy.Interval(1, sympy.oo)

class Pearson3(ContinuousReal):
    """A pearson type III continuous random variable."""
    dist = scipy.stats.pearson3
    def get_domain(self, **kwargs): return Reals

class Powerlaw(ContinuousReal):
    """A power-function continuous random variable."""
    dist = scipy.stats.powerlaw
    def get_domain(self, **kwargs): return UnitInterval

class Powerlognorm(ContinuousReal):
    """A power log-normal continuous random variable."""
    dist = scipy.stats.powerlognorm
    def get_domain(self, **kwargs): return RealsPos

class Powernorm(ContinuousReal):
    """A power normal continuous random variable."""
    dist = scipy.stats.powernorm
    def get_domain(self, **kwargs): return RealsPos

class Rdist(ContinuousReal):
    """An R-distributed (symmetric beta) continuous random variable."""
    dist = scipy.stats.rdist
    def get_domain(self, **kwargs): return sympy.Interval(-1, 1)

class Rayleigh(ContinuousReal):
    """A Rayleigh continuous random variable."""
    dist = scipy.stats.rayleigh
    def get_domain(self, **kwargs): return RealsPos

class Rice(ContinuousReal):
    """A Rice continuous random variable."""
    dist = scipy.stats.rice
    def get_domain(self, **kwargs): return RealsPos

class Recipinvgauss(ContinuousReal):
    """A reciprocal inverse Gaussian continuous random variable."""
    dist = scipy.stats.recipinvgauss
    def get_domain(self, **kwargs): return RealsPos

class Semicircular(ContinuousReal):
    """A semicircular continuous random variable."""
    dist = scipy.stats.semicircular
    def get_domain(self, **kwargs): return sympy.Interval(-1, 1)

class Skewnorm(ContinuousReal):
    """A skew-normal random variable."""
    dist = scipy.stats.skewnorm
    def get_domain(self, **kwargs): return Reals

class T(ContinuousReal):
    """A Student’s t continuous random variable."""
    dist = scipy.stats.t
    def get_domain(self, **kwargs): return Reals

class Trapz(ContinuousReal):
    """A trapezoidal continuous random variable."""
    dist = scipy.stats.trapz
    def get_domain(self, **kwargs):
        loc = kwargs.get('loc', 0)
        scale = kwargs.get('scale', 1)
        return sympy.Interval(loc, loc+scale)

class Triang(ContinuousReal):
    """A triangular continuous random variable."""
    dist = scipy.stats.triang
    def get_domain(self, **kwargs):
        loc = kwargs.get('loc', 0)
        scale = kwargs.get('scale', 1)
        return sympy.Interval(loc, loc+scale)

class Truncexpon(ContinuousReal):
    """A truncated exponential continuous random variable."""
    dist = scipy.stats.truncexpon
    def get_domain(self, **kwargs): return sympy.Interval(0, kwargs['b'])

class Truncnorm(ContinuousReal):
    """A truncated normal continuous random variable."""
    dist = scipy.stats.truncnorm
    def get_domain(self, **kwargs): return sympy.Interval(kwargs['a'], kwargs['b'])

class Tukeylambda(ContinuousReal):
    """A Tukey-Lamdba continuous random variable."""
    dist = scipy.stats.tukeylambda
    def get_domain(self, **kwargs): return RealsPos

class Uniform(ContinuousReal):
    """A uniform continuous random variable."""
    dist = scipy.stats.uniform
    def get_domain(self, **kwargs):
        loc = kwargs.get('loc', 0)
        scale = kwargs.get('scale', 1)
        return sympy.Interval(loc, loc + scale)

class Vonmises(ContinuousReal):
    """A Von Mises continuous random variable."""
    dist = scipy.stats.vonmises
    def get_domain(self, **kwargs): return sympy.Interval(-sympy.pi, sympy.pi)

class Vonmises_line(ContinuousReal):
    """A Von Mises continuous random variable."""
    dist = scipy.stats.vonmises_line
    def get_domain(self, **kwargs): return sympy.Interval(-sympy.pi, sympy.pi)

class Wald(ContinuousReal):
    """A Wald continuous random variable."""
    dist = scipy.stats.wald
    def get_domain(self, **kwargs): return RealsPos

class Weibull_min(ContinuousReal):
    """Weibull minimum continuous random variable."""
    dist = scipy.stats.weibull_min
    def get_domain(self, **kwargs): return RealsPos

class Weibull_max(ContinuousReal):
    """Weibull maximum continuous random variable."""
    dist = scipy.stats.weibull_max
    def get_domain(self, **kwargs): return RealsNeg

class Wrapcauchy(ContinuousReal):
    """A wrapped Cauchy continuous random variable."""
    dist = scipy.stats.wrapcauchy
    def get_domain(self, **kwargs): return sympy.Interval(0, 2*sympy.pi)

# ==============================================================================
# DiscreteReal

from .spn import DiscreteLeaf
from .sym_util import Integers
from .sym_util import IntegersPos
from .sym_util import IntegersPos0

class DiscreteReal(RealDistribution):
    constructor = DiscreteLeaf

class Bernoulli(DiscreteReal):
    """A Bernoulli discrete random variable."""
    dist = scipy.stats.bernoulli
    def get_domain(self, **kwargs): return sympy.Range(0, 2)

class Betabinom(DiscreteReal):
    """A beta-binomial discrete random variable."""
    dist = scipy.stats.betabinom
    def get_domain(self, **kwargs): return sympy.Range(0, kwargs['n']+1)

class Binom(DiscreteReal):
    """A binomial discrete random variable."""
    dist = scipy.stats.binom
    def get_domain(self, **kwargs): return sympy.Range(0, kwargs['n']+1)

class Boltzmann(DiscreteReal):
    """A Boltzmann (Truncated Discrete Exponential) random variable."""
    dist = scipy.stats.boltzmann
    def get_domain(self, **kwargs): return sympy.Range(0, kwargs['N']+1)

class Dlaplace(DiscreteReal):
    """A Laplacian discrete random variable."""
    dist = scipy.stats.dlaplace
    def get_domain(self, **kwargs): return Integers

class Geom(DiscreteReal):
    """A geometric discrete random variable."""
    dist = scipy.stats.geom
    def get_domain(self, **kwargs): return Integers

class Hypergeom(DiscreteReal):
    """A hypergeometric discrete random variable."""
    dist = scipy.stats.hypergeom
    def get_domain(self, **kwargs):
        low = max(0, kwargs['N'], kwargs['N']-kwargs['M']+kwargs['n'])
        high = min(kwargs['n'], kwargs['N'])
        return sympy.Range(low, high+1)

class Logser(DiscreteReal):
    """A Logarithmic (Log-Series, Series) discrete random variable."""
    dist = scipy.stats.logser
    def get_domain(self, **kwargs): return IntegersPos

class Nbinom(DiscreteReal):
    """A negative binomial discrete random variable."""
    dist = scipy.stats.nbinom
    def get_domain(self, **kwargs): return IntegersPos0

class Planck(DiscreteReal):
    """A Planck discrete exponential random variable."""
    dist = scipy.stats.planck
    def get_domain(self, **kwargs): return IntegersPos0

class Poisson(DiscreteReal):
    """A Poisson discrete random variable."""
    dist = scipy.stats.poisson
    def get_domain(self, **kwargs): return IntegersPos0

class Randint(DiscreteReal):
    """A uniform discrete random variable."""
    dist = scipy.stats.randint
    def get_domain(self, **kwargs): return sympy.Range(kwargs['low'], kwargs['high'])

class Skellam(DiscreteReal):
    """A Skellam discrete random variable."""
    dist = scipy.stats.skellam
    def get_domain(self, **kwargs): return Integers

class Zipf(DiscreteReal):
    """A Zipf discrete random variable."""
    dist = scipy.stats.zipf
    def get_domain(self, **kwargs): return IntegersPos

class Yulesimon(DiscreteReal):
    """A Yule-Simon discrete random variable."""
    dist = scipy.stats.yulesimon
    def get_domain(self, **kwargs): return IntegersPos

def Atomic(**kwargs):
    """An atomic discrete random variable."""
    return Randint(low=kwargs['loc'], high=kwargs['loc']+1)
