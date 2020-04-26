# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

import scipy.stats
import sympy

# pylint: disable=not-callable
# pylint: disable=multiple-statements
class RealDistribution():
    dist = None
    constructor = None
    def __init__(self, *args, **kwargs):
        assert not args, 'Only keyword arguments allowed for %s' % (self.dist.name,)
        self.kwargs = kwargs
    def __call__(self, symbol):
        domain = self.get_domain(**self.kwargs)
        return self.constructor(symbol, self.dist(**self.kwargs), domain)
    def get_domain(self, **kwargs):
        raise NotImplementedError()

    def __rmul__(self, x):
        from .sym_util import sympify_number
        try:
            x_val = sympify_number(x)
            if not 0 < x_val < 1:
                raise ValueError('invalid weight %s' % (str(x),))
            return RealDistributionMix([self], [x])
        except TypeError:
            return NotImplemented

class RealDistributionMix():
    """Weighted mixture of SPNs that do not yet sum to unity."""
    def __init__(self, distributions, weights):
        self.distributions = distributions
        self.weights = weights
    def __call__(self, symbol):
        from math import log
        from .spn import SumSPN
        distributions = [d(symbol) for d in self.distributions]
        weights = [log(w) for w in self.weights]
        return SumSPN(distributions, weights)

    def __or__(self, x):
        if not isinstance(x, RealDistributionMix):
            return NotImplemented
        weights = self.weights + x.weights
        cumsum = float(sum(weights))
        assert 0 < cumsum <= 1
        distributions = self.distributions + x.distributions
        return RealDistributionMix(distributions, weights)

# ==============================================================================
# ContinuousReal

from .spn import ContinuousLeaf
from .sym_util import Reals
from .sym_util import RealsNeg
from .sym_util import RealsPos
from .sym_util import UnitInterval

class ContinuousReal(RealDistribution):
    constructor = ContinuousLeaf

class alpha(ContinuousReal):
    """An alpha continuous random variable."""
    dist = scipy.stats.alpha
    def get_domain(self, **kwargs): return RealsPos

class anglit(ContinuousReal):
    """An anglit continuous random variable."""
    dist = scipy.stats.anglit
    def get_domain(self, **kwargs): return sympy.Interval(-sympy.pi/4, sympy.pi/4)

class arcsine(ContinuousReal):
    """An arcsine continuous random variable."""
    dist = scipy.stats.arcsine
    def get_domain(self, **kwargs): return UnitInterval

class argus(ContinuousReal):
    """Argus distribution"""
    dist = scipy.stats.argus
    def get_domain(self, **kwargs): return UnitInterval

class beta(ContinuousReal):
    """A beta continuous random variable."""
    dist = scipy.stats.beta
    def get_domain(self, **kwargs): return UnitInterval

class betaprime(ContinuousReal):
    """A beta prime continuous random variable."""
    dist = scipy.stats.betaprime
    def get_domain(self, **kwargs): return RealsPos

class bradford(ContinuousReal):
    """A Bradford continuous random variable."""
    dist = scipy.stats.bradford
    def get_domain(self, **kwargs): return UnitInterval

class burr(ContinuousReal):
    """A Burr (Type III) continuous random variable."""
    dist = scipy.stats.burr
    def get_domain(self, **kwargs): return RealsPos

class burr12(ContinuousReal):
    """A Burr (Type XII) continuous random variable."""
    dist = scipy.stats.burr12
    def get_domain(self, **kwargs): return RealsPos

class cauchy(ContinuousReal):
    """A Cauchy continuous random variable."""
    dist = scipy.stats.cauchy
    def get_domain(self, **kwargs): return sympy.Reals

class chi(ContinuousReal):
    """A chi continuous random variable."""
    dist = scipy.stats.chi
    def get_domain(self, **kwargs): return RealsPos

class chi2(ContinuousReal):
    """A chi-squared continuous random variable."""
    dist = scipy.stats.chi2
    def get_domain(self, **kwargs): return RealsPos

class cosine(ContinuousReal):
    """A cosine continuous random variable."""
    dist = scipy.stats.cosine
    def get_domain(self, **kwargs): return sympy.Interval(-sympy.pi/2, sympy.pi/2)

class crystalball(ContinuousReal):
    """Crystalball distribution."""
    dist = scipy.stats.crystalball
    def get_domain(self, **kwargs): return sympy.Reals

class dgamma(ContinuousReal):
    """A double gamma continuous random variable."""
    dist = scipy.stats.dgamma
    def get_domain(self, **kwargs): return sympy.Reals

class dweibull(ContinuousReal):
    """A double Weibull continuous random variable."""
    dist = scipy.stats.dweibull
    def get_domain(self, **kwargs): return sympy.Reals

class erlang(ContinuousReal):
    """An Erlang continuous random variable."""
    dist = scipy.stats.erlang
    def get_domain(self, **kwargs): return RealsPos

class expon(ContinuousReal):
    """An exponential continuous random variable."""
    dist = scipy.stats.expon
    def get_domain(self, **kwargs): return RealsPos

class exponnorm(ContinuousReal):
    """An exponentially modified normal continuous random variable."""
    dist = scipy.stats.exponnorm
    def get_domain(self, **kwargs): return sympy.Reals

class exponweib(ContinuousReal):
    """An exponentiated Weibull continuous random variable."""
    dist = scipy.stats.exponweib
    def get_domain(self, **kwargs): return RealsPos

class exponpow(ContinuousReal):
    """An exponential power continuous random variable."""
    dist = scipy.stats.exponpow
    def get_domain(self, **kwargs): return RealsPos

class f(ContinuousReal):
    """An F continuous random variable."""
    dist = scipy.stats.f
    def get_domain(self, **kwargs): return RealsPos

class fatiguelife(ContinuousReal):
    """A fatigue-life (Birnbaum-Saunders) continuous random variable."""
    dist = scipy.stats.fatiguelife
    def get_domain(self, **kwargs): return RealsPos

class fisk(ContinuousReal):
    """A Fisk continuous random variable."""
    dist = scipy.stats.fisk
    def get_domain(self, **kwargs): return RealsPos

class foldcauchy(ContinuousReal):
    """A folded Cauchy continuous random variable."""
    dist = scipy.stats.foldcauchy
    def get_domain(self, **kwargs): return RealsPos

class foldnorm(ContinuousReal):
    """A folded normal continuous random variable."""
    dist = scipy.stats.foldnorm
    def get_domain(self, **kwargs): return RealsPos

class frechet_r(ContinuousReal):
    """A Frechet right (or Weibull minimum) continuous random variable."""
    dist = scipy.stats.frechet_r
    def get_domain(self, **kwargs): return RealsPos

class frechet_l(ContinuousReal):
    """A Frechet left (or Weibull maximum) continuous random variable."""
    dist = scipy.stats.frechet_l
    def get_domain(self, **kwargs): return RealsNeg

class genlogistic(ContinuousReal):
    """A generalized logistic continuous random variable."""
    dist = scipy.stats.genlogistic
    def get_domain(self, **kwargs): return RealsPos

class gennorm(ContinuousReal):
    """A generalized normal continuous random variable."""
    dist = scipy.stats.gennorm
    def get_domain(self, **kwargs): return sympy.Reals

class genpareto(ContinuousReal):
    """A generalized Pareto continuous random variable."""
    dist = scipy.stats.genpareto
    def get_domain(self, **kwargs): return RealsPos

class genexpon(ContinuousReal):
    """A generalized exponential continuous random variable."""
    dist = scipy.stats.genexpon
    def get_domain(self, **kwargs): return RealsPos

class genextreme(ContinuousReal):
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

class gausshyper(ContinuousReal):
    """A Gauss hypergeometric continuous random variable."""
    dist = scipy.stats.gausshyper
    def get_domain(self, **kwargs): return UnitInterval


class gamma(ContinuousReal):
    """A gamma continuous random variable."""
    dist = scipy.stats.gamma
    def get_domain(self, **kwargs): return RealsPos

class gengamma(ContinuousReal):
    """A generalized gamma continuous random variable."""
    dist = scipy.stats.gengamma
    def get_domain(self, **kwargs): return RealsPos

class genhalflogistic(ContinuousReal):
    """A generalized half-logistic continuous random variable."""
    dist = scipy.stats.genhalflogistic
    def get_domain(self, **kwargs):
        assert kwargs['c'] > 0
        return sympy.Interval(0, 1./kwargs['c'])

class geninvgauss(ContinuousReal):
    """A Generalized Inverse Gaussian continuous random variable."""
    dist = scipy.stats.geninvgauss
    def get_domain(self, **kwargs): return RealsPos

class gilbrat(ContinuousReal):
    """A Gilbrat continuous random variable."""
    dist = scipy.stats.gilbrat
    def get_domain(self, **kwargs): return RealsPos

class gompertz(ContinuousReal):
    """A Gompertz (or truncated Gumbel) continuous random variable."""
    dist = scipy.stats.gompertz
    def get_domain(self, **kwargs): return RealsPos

class gumbel_r(ContinuousReal):
    """A right-skewed Gumbel continuous random variable."""
    dist = scipy.stats.gumbel_r
    def get_domain(self, **kwargs): return sympy.Reals

class gumbel_l(ContinuousReal):
    """A left-skewed Gumbel continuous random variable."""
    dist = scipy.stats.gumbel_l
    def get_domain(self, **kwargs): return RealsPos

class halfcauchy(ContinuousReal):
    """A Half-Cauchy continuous random variable."""
    dist = scipy.stats.halfcauchy
    def get_domain(self, **kwargs): return RealsPos

class halflogistic(ContinuousReal):
    """A half-logistic continuous random variable."""
    dist = scipy.stats.halflogistic
    def get_domain(self, **kwargs): return RealsPos

class halfnorm(ContinuousReal):
    """A half-normal continuous random variable."""
    dist = scipy.stats.halfnorm
    def get_domain(self, **kwargs): return RealsPos

class halfgennorm(ContinuousReal):
    """The upper half of a generalized normal continuous random variable."""
    dist = scipy.stats.halfgennorm
    def get_domain(self, **kwargs): return RealsPos

class hypsecant(ContinuousReal):
    """A hyperbolic secant continuous random variable."""
    dist = scipy.stats.hypsecant
    def get_domain(self, **kwargs): return sympy.Reals

class invgamma(ContinuousReal):
    """An inverted gamma continuous random variable."""
    dist = scipy.stats.invgamma
    def get_domain(self, **kwargs): return RealsPos

class invgauss(ContinuousReal):
    """An inverse Gaussian continuous random variable."""
    dist = scipy.stats.invgauss
    def get_domain(self, **kwargs): return RealsPos

class invweibull(ContinuousReal):
    """An inverted Weibull continuous random variable."""
    dist = scipy.stats.invweibull
    def get_domain(self, **kwargs): return RealsPos

class johnsonsb(ContinuousReal):
    """A Johnson SB continuous random variable."""
    dist = scipy.stats.johnsonsb
    def get_domain(self, **kwargs): return UnitInterval

class johnsonsu(ContinuousReal):
    """A Johnson SU continuous random variable."""
    dist = scipy.stats.johnsonsu
    def get_domain(self, **kwargs): return Reals

class kappa4(ContinuousReal):
    """Kappa 4 parameter distribution."""
    dist = scipy.stats.kappa4
    def get_domain(self, **kwargs): return Reals

class kappa3(ContinuousReal):
    """Kappa 3 parameter distribution."""
    dist = scipy.stats.kappa3
    def get_domain(self, **kwargs): return RealsPos

class ksone(ContinuousReal):
    """General Kolmogorov-Smirnov one-sided test."""
    dist = scipy.stats.ksone
    def get_domain(self, **kwargs): return UnitInterval

class kstwobign(ContinuousReal):
    """Kolmogorov-Smirnov two-sided test for large N."""
    dist = scipy.stats.kstwobign
    def get_domain(self, **kwargs): return sympy.Interval(0, sympy.sqrt(kwargs['n']))

class laplace(ContinuousReal):
    """A Laplace continuous random variable."""
    dist = scipy.stats.laplace
    def get_domain(self, **kwargs): return Reals

class levy(ContinuousReal):
    """A Levy continuous random variable."""
    dist = scipy.stats.levy
    def get_domain(self, **kwargs): return RealsPos

class levy_l(ContinuousReal):
    """A left-skewed Levy continuous random variable."""
    dist = scipy.stats.levy_l
    def get_domain(self, **kwargs): return RealsNeg

class levy_stable(ContinuousReal):
    """A Levy-stable continuous random variable."""
    dist = scipy.stats.levy_stable
    def get_domain(self, **kwargs): return Reals

class logistic(ContinuousReal):
    """A logistic (or Sech-squared) continuous random variable."""
    dist = scipy.stats.logistic
    def get_domain(self, **kwargs): return Reals

class loggamma(ContinuousReal):
    """A log gamma continuous random variable."""
    dist = scipy.stats.loggamma
    def get_domain(self, **kwargs): return RealsNeg

class loglaplace(ContinuousReal):
    """A log-Laplace continuous random variable."""
    dist = scipy.stats.loglaplace
    def get_domain(self, **kwargs): return RealsPos

class lognorm(ContinuousReal):
    """A lognormal continuous random variable."""
    dist = scipy.stats.lognorm
    def get_domain(self, **kwargs): return RealsPos

class loguniform(ContinuousReal):
    """A loguniform or reciprocal continuous random variable."""
    dist = scipy.stats.loguniform
    def get_domain(self, **kwargs): return sympy.Interval(kwargs['a'], kwargs['b'])

class lomax(ContinuousReal):
    """A Lomax (Pareto of the second kind) continuous random variable."""
    dist = scipy.stats.lomax
    def get_domain(self, **kwargs): return RealsPos

class maxwell(ContinuousReal):
    """A Maxwell continuous random variable."""
    dist = scipy.stats.maxwell
    def get_domain(self, **kwargs): return RealsPos

class mielke(ContinuousReal):
    """A Mielke Beta-Kappa / Dagum continuous random variable."""
    dist = scipy.stats.mielke
    def get_domain(self, **kwargs): return RealsPos

class moyal(ContinuousReal):
    """A Moyal continuous random variable."""
    dist = scipy.stats.moyal
    def get_domain(self, **kwargs): return Reals

class nakagami(ContinuousReal):
    """A Nakagami continuous random variable."""
    dist = scipy.stats.nakagami
    def get_domain(self, **kwargs): return RealsPos

class ncx2(ContinuousReal):
    """A non-central chi-squared continuous random variable."""
    dist = scipy.stats.ncx2
    def get_domain(self, **kwargs): return RealsPos

class ncf(ContinuousReal):
    """A non-central F distribution continuous random variable."""
    dist = scipy.stats.ncf
    def get_domain(self, **kwargs): return RealsPos

class nct(ContinuousReal):
    """A non-central Student’s t continuous random variable."""
    dist = scipy.stats.nct
    def get_domain(self, **kwargs): return Reals

class norm(ContinuousReal):
    """A normal continuous random variable."""
    dist = scipy.stats.norm
    def get_domain(self, **kwargs): return Reals

class norminvgauss(ContinuousReal):
    """A normal Inverse Gaussian continuous random variable."""
    dist = scipy.stats.norminvgauss
    def get_domain(self, **kwargs): return Reals

class pareto(ContinuousReal):
    """A Pareto continuous random variable."""
    dist = scipy.stats.pareto
    def get_domain(self, **kwargs): return sympy.Interval(1, sympy.oo)

class pearson3(ContinuousReal):
    """A pearson type III continuous random variable."""
    dist = scipy.stats.pearson3
    def get_domain(self, **kwargs): return Reals

class powerlaw(ContinuousReal):
    """A power-function continuous random variable."""
    dist = scipy.stats.powerlaw
    def get_domain(self, **kwargs): return UnitInterval

class powerlognorm(ContinuousReal):
    """A power log-normal continuous random variable."""
    dist = scipy.stats.powerlognorm
    def get_domain(self, **kwargs): return RealsPos

class powernorm(ContinuousReal):
    """A power normal continuous random variable."""
    dist = scipy.stats.powernorm
    def get_domain(self, **kwargs): return RealsPos

class rdist(ContinuousReal):
    """An R-distributed (symmetric beta) continuous random variable."""
    dist = scipy.stats.rdist
    def get_domain(self, **kwargs): return sympy.Interval(-1, 1)

class rayleigh(ContinuousReal):
    """A Rayleigh continuous random variable."""
    dist = scipy.stats.rayleigh
    def get_domain(self, **kwargs): return RealsPos

class rice(ContinuousReal):
    """A Rice continuous random variable."""
    dist = scipy.stats.rice
    def get_domain(self, **kwargs): return RealsPos

class recipinvgauss(ContinuousReal):
    """A reciprocal inverse Gaussian continuous random variable."""
    dist = scipy.stats.recipinvgauss
    def get_domain(self, **kwargs): return RealsPos

class semicircular(ContinuousReal):
    """A semicircular continuous random variable."""
    dist = scipy.stats.semicircular
    def get_domain(self, **kwargs): return sympy.Interval(-1, 1)

class skewnorm(ContinuousReal):
    """A skew-normal random variable."""
    dist = scipy.stats.skewnorm
    def get_domain(self, **kwargs): return Reals

class t(ContinuousReal):
    """A Student’s t continuous random variable."""
    dist = scipy.stats.t
    def get_domain(self, **kwargs): return Reals

class trapz(ContinuousReal):
    """A trapezoidal continuous random variable."""
    dist = scipy.stats.trapz
    def get_domain(self, **kwargs):
        loc = kwargs.get('loc', 0)
        scale = kwargs.get('scale', 1)
        return sympy.Interval(loc, loc+scale)

class triang(ContinuousReal):
    """A triangular continuous random variable."""
    dist = scipy.stats.triang
    def get_domain(self, **kwargs):
        loc = kwargs.get('loc', 0)
        scale = kwargs.get('scale', 1)
        return sympy.Interval(loc, loc+scale)

class truncexpon(ContinuousReal):
    """A truncated exponential continuous random variable."""
    dist = scipy.stats.truncexpon
    def get_domain(self, **kwargs): return sympy.Interval(0, kwargs['b'])

class truncnorm(ContinuousReal):
    """A truncated normal continuous random variable."""
    dist = scipy.stats.truncnorm
    def get_domain(self, **kwargs): return sympy.Interval(kwargs['a'], kwargs['b'])

class tukeylambda(ContinuousReal):
    """A Tukey-Lamdba continuous random variable."""
    dist = scipy.stats.tukeylambda
    def get_domain(self, **kwargs): return RealsPos

class uniform(ContinuousReal):
    """A uniform continuous random variable."""
    dist = scipy.stats.uniform
    def get_domain(self, **kwargs):
        loc = kwargs.get('loc', 0)
        scale = kwargs.get('scale', 1)
        return sympy.Interval(loc, loc + scale)

class vonmises(ContinuousReal):
    """A Von Mises continuous random variable."""
    dist = scipy.stats.vonmises
    def get_domain(self, **kwargs): return sympy.Interval(-sympy.pi, sympy.pi)

class vonmises_line(ContinuousReal):
    """A Von Mises continuous random variable."""
    dist = scipy.stats.vonmises_line
    def get_domain(self, **kwargs): return sympy.Interval(-sympy.pi, sympy.pi)

class wald(ContinuousReal):
    """A Wald continuous random variable."""
    dist = scipy.stats.wald
    def get_domain(self, **kwargs): return RealsPos

class weibull_min(ContinuousReal):
    """Weibull minimum continuous random variable."""
    dist = scipy.stats.weibull_min
    def get_domain(self, **kwargs): return RealsPos

class weibull_max(ContinuousReal):
    """Weibull maximum continuous random variable."""
    dist = scipy.stats.weibull_max
    def get_domain(self, **kwargs): return RealsNeg

class wrapcauchy(ContinuousReal):
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

class bernoulli(DiscreteReal):
    """A Bernoulli discrete random variable."""
    dist = scipy.stats.bernoulli
    def get_domain(self, **kwargs): return sympy.Range(0, 2)

class betabinom(DiscreteReal):
    """A beta-binomial discrete random variable."""
    dist = scipy.stats.betabinom
    def get_domain(self, **kwargs): return sympy.Range(0, kwargs['n']+1)

class binom(DiscreteReal):
    """A binomial discrete random variable."""
    dist = scipy.stats.binom
    def get_domain(self, **kwargs): return sympy.Range(0, kwargs['n']+1)

class boltzmann(DiscreteReal):
    """A Boltzmann (Truncated Discrete Exponential) random variable."""
    dist = scipy.stats.boltzmann
    def get_domain(self, **kwargs): return sympy.Range(0, kwargs['N']+1)

class dlaplace(DiscreteReal):
    """A Laplacian discrete random variable."""
    dist = scipy.stats.dlaplace
    def get_domain(self, **kwargs): return Integers

class geom(DiscreteReal):
    """A geometric discrete random variable."""
    dist = scipy.stats.geom
    def get_domain(self, **kwargs): return Integers

class hypergeom(DiscreteReal):
    """A hypergeometric discrete random variable."""
    dist = scipy.stats.hypergeom
    def get_domain(self, **kwargs):
        low = max(0, kwargs['N'], kwargs['N']-kwargs['M']+kwargs['n'])
        high = min(kwargs['n'], kwargs['N'])
        return sympy.Range(low, high+1)

class logser(DiscreteReal):
    """A Logarithmic (Log-Series, Series) discrete random variable."""
    dist = scipy.stats.logser
    def get_domain(self, **kwargs): return IntegersPos

class nbinom(DiscreteReal):
    """A negative binomial discrete random variable."""
    dist = scipy.stats.nbinom
    def get_domain(self, **kwargs): return IntegersPos0

class planck(DiscreteReal):
    """A Planck discrete exponential random variable."""
    dist = scipy.stats.planck
    def get_domain(self, **kwargs): return IntegersPos0

class poisson(DiscreteReal):
    """A Poisson discrete random variable."""
    dist = scipy.stats.poisson
    def get_domain(self, **kwargs): return IntegersPos0

class randint(DiscreteReal):
    """A uniform discrete random variable."""
    dist = scipy.stats.randint
    def get_domain(self, **kwargs): return sympy.Range(kwargs['low'], kwargs['high'])

class skellam(DiscreteReal):
    """A Skellam discrete random variable."""
    dist = scipy.stats.skellam
    def get_domain(self, **kwargs): return Integers

class zipf(DiscreteReal):
    """A Zipf discrete random variable."""
    dist = scipy.stats.zipf
    def get_domain(self, **kwargs): return IntegersPos

class yulesimon(DiscreteReal):
    """A Yule-Simon discrete random variable."""
    dist = scipy.stats.yulesimon
    def get_domain(self, **kwargs): return IntegersPos

class atomic(randint):
    """An atomic discrete random variable."""
    def __init__(self, *args, **kwargs):
        loc = kwargs.pop('loc')
        kwargs['low'] = loc
        kwargs['high'] = loc + 1
        super().__init__(*args, **kwargs)
