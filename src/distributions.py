# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

import scipy.stats
import sympy

class Distribution():
    def __rmul__(self, x):
        from .sym_util import sympify_number
        try:
            x_val = sympify_number(x)
            if not 0 < x_val < 1:
                raise ValueError('invalid weight %s' % (str(x),))
            return DistributionMix([self], [x])
        except TypeError:
            return NotImplemented

class NominalDistribution(Distribution):
    def __init__(self, dist):
        self.dist = dict(dist)
    def __call__(self, symbol):
        from .spn import NominalLeaf
        return NominalLeaf(symbol, self.dist)

choice = NominalDistribution

def floatify(x):
    try              : return float(x)
    except TypeError : return x

class RealDistribution(Distribution):
    # pylint: disable=not-callable
    # pylint: disable=multiple-statements
    dist = None
    constructor = None
    def __init__(self, *args, **kwargs):
        assert not args, 'Only keyword arguments allowed for %s' % (self.dist.name,)
        self.kwargs = {k: floatify(v) for k, v in kwargs.items()}
    def __call__(self, symbol):
        domain = self.get_domain()
        return self.constructor(symbol, self.dist(**self.kwargs), domain)
    def get_domain(self):
        raise NotImplementedError()

class DistributionMix():
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
        if not isinstance(x, DistributionMix):
            return NotImplemented
        weights = self.weights + x.weights
        cumsum = float(sum(weights))
        assert 0 < cumsum <= 1
        distributions = self.distributions + x.distributions
        return DistributionMix(distributions, weights)

# ==============================================================================
# ContinuousReal

from .sets import Interval
from .sets import Reals
from .sets import RealsNeg
from .sets import RealsPos
from .sets import inf as oo
from .spn import ContinuousLeaf

def RealsPosLoc(kwargs):
    if 'loc' in kwargs:
        return Interval(kwargs['loc'], oo)
    return RealsPos

def UnitIntervalLocScale(kwargs):
    loc = kwargs.get('loc', 0)
    scale = kwargs.get('scale', 1)
    return Interval(loc, loc + scale)

class ContinuousReal(RealDistribution):
    constructor = ContinuousLeaf

class alpha(ContinuousReal):
    """An alpha continuous random variable."""
    dist = scipy.stats.alpha
    def get_domain(self): return RealsPosLoc(self.kwargs)

class anglit(ContinuousReal):
    """An anglit continuous random variable."""
    dist = scipy.stats.anglit
    def get_domain(self): return Interval(-sympy.pi/4, sympy.pi/4)

class arcsine(ContinuousReal):
    """An arcsine continuous random variable."""
    dist = scipy.stats.arcsine
    def get_domain(self): return UnitIntervalLocScale(self.kwargs)

class argus(ContinuousReal):
    """Argus distribution"""
    dist = scipy.stats.argus
    def get_domain(self): return UnitIntervalLocScale(self.kwargs)

class beta(ContinuousReal):
    """A beta continuous random variable."""
    dist = scipy.stats.beta
    def get_domain(self): return UnitIntervalLocScale(self.kwargs)

class betaprime(ContinuousReal):
    """A beta prime continuous random variable."""
    dist = scipy.stats.betaprime
    def get_domain(self): return RealsPosLoc(self.kwargs)

class bradford(ContinuousReal):
    """A Bradford continuous random variable."""
    dist = scipy.stats.bradford
    def get_domain(self): return UnitIntervalLocScale(self.kwargs)

class burr(ContinuousReal):
    """A Burr (Type III) continuous random variable."""
    dist = scipy.stats.burr
    def get_domain(self): return RealsPosLoc(self.kwargs)

class burr12(ContinuousReal):
    """A Burr (Type XII) continuous random variable."""
    dist = scipy.stats.burr12
    def get_domain(self): return RealsPosLoc(self.kwargs)

class cauchy(ContinuousReal):
    """A Cauchy continuous random variable."""
    dist = scipy.stats.cauchy
    def get_domain(self): return Reals

class chi(ContinuousReal):
    """A chi continuous random variable."""
    dist = scipy.stats.chi
    def get_domain(self): return RealsPosLoc(self.kwargs)

class chi2(ContinuousReal):
    """A chi-squared continuous random variable."""
    dist = scipy.stats.chi2
    def get_domain(self): return RealsPosLoc(self.kwargs)

class cosine(ContinuousReal):
    """A cosine continuous random variable."""
    dist = scipy.stats.cosine
    def get_domain(self): return Interval(-sympy.pi/2, sympy.pi/2)

class crystalball(ContinuousReal):
    """Crystalball distribution."""
    dist = scipy.stats.crystalball
    def get_domain(self): return Reals

class dgamma(ContinuousReal):
    """A double gamma continuous random variable."""
    dist = scipy.stats.dgamma
    def get_domain(self): return Reals

class dweibull(ContinuousReal):
    """A double Weibull continuous random variable."""
    dist = scipy.stats.dweibull
    def get_domain(self): return Reals

class erlang(ContinuousReal):
    """An Erlang continuous random variable."""
    dist = scipy.stats.erlang
    def get_domain(self): return RealsPosLoc(self.kwargs)

class expon(ContinuousReal):
    """An exponential continuous random variable."""
    dist = scipy.stats.expon
    def get_domain(self): return RealsPosLoc(self.kwargs)

class exponnorm(ContinuousReal):
    """An exponentially modified normal continuous random variable."""
    dist = scipy.stats.exponnorm
    def get_domain(self): return Reals

class exponweib(ContinuousReal):
    """An exponentiated Weibull continuous random variable."""
    dist = scipy.stats.exponweib
    def get_domain(self): return RealsPosLoc(self.kwargs)

class exponpow(ContinuousReal):
    """An exponential power continuous random variable."""
    dist = scipy.stats.exponpow
    def get_domain(self): return RealsPosLoc(self.kwargs)

class f(ContinuousReal):
    """An F continuous random variable."""
    dist = scipy.stats.f
    def get_domain(self): return RealsPosLoc(self.kwargs)

class fatiguelife(ContinuousReal):
    """A fatigue-life (Birnbaum-Saunders) continuous random variable."""
    dist = scipy.stats.fatiguelife
    def get_domain(self): return RealsPosLoc(self.kwargs)

class fisk(ContinuousReal):
    """A Fisk continuous random variable."""
    dist = scipy.stats.fisk
    def get_domain(self): return RealsPosLoc(self.kwargs)

class foldcauchy(ContinuousReal):
    """A folded Cauchy continuous random variable."""
    dist = scipy.stats.foldcauchy
    def get_domain(self): return RealsPosLoc(self.kwargs)

class foldnorm(ContinuousReal):
    """A folded normal continuous random variable."""
    dist = scipy.stats.foldnorm
    def get_domain(self): return RealsPosLoc(self.kwargs)

class frechet_r(ContinuousReal):
    """A Frechet right (or Weibull minimum) continuous random variable."""
    dist = scipy.stats.frechet_r
    def get_domain(self): return RealsPosLoc(self.kwargs)

class frechet_l(ContinuousReal):
    """A Frechet left (or Weibull maximum) continuous random variable."""
    dist = scipy.stats.frechet_l
    def get_domain(self): return RealsNeg

class genlogistic(ContinuousReal):
    """A generalized logistic continuous random variable."""
    dist = scipy.stats.genlogistic
    def get_domain(self): return RealsPosLoc(self.kwargs)

class gennorm(ContinuousReal):
    """A generalized normal continuous random variable."""
    dist = scipy.stats.gennorm
    def get_domain(self): return Reals

class genpareto(ContinuousReal):
    """A generalized Pareto continuous random variable."""
    dist = scipy.stats.genpareto
    def get_domain(self): return RealsPosLoc(self.kwargs)

class genexpon(ContinuousReal):
    """A generalized exponential continuous random variable."""
    dist = scipy.stats.genexpon
    def get_domain(self): return RealsPosLoc(self.kwargs)

class genextreme(ContinuousReal):
    """A generalized extreme value continuous random variable."""
    dist = scipy.stats.genextreme
    def get_domain(self):
        c = self.kwargs['c']
        if c == 0:
            return Reals
        elif c > 0:
            return Interval(-oo, 1/c)
        elif c < 0:
            return Interval(1/c, oo)
        assert False, 'Bad argument "c" for genextreme: %s' % (self.kwargs,)

class gausshyper(ContinuousReal):
    """A Gauss hypergeometric continuous random variable."""
    dist = scipy.stats.gausshyper
    def get_domain(self): return UnitIntervalLocScale(self.kwargs)

class gamma(ContinuousReal):
    """A gamma continuous random variable."""
    dist = scipy.stats.gamma
    def get_domain(self): return RealsPosLoc(self.kwargs)

class gengamma(ContinuousReal):
    """A generalized gamma continuous random variable."""
    dist = scipy.stats.gengamma
    def get_domain(self): return RealsPosLoc(self.kwargs)

class genhalflogistic(ContinuousReal):
    """A generalized half-logistic continuous random variable."""
    dist = scipy.stats.genhalflogistic
    def get_domain(self):
        assert self.kwargs['c'] > 0
        return Interval(0, 1./self.kwargs['c'])

class geninvgauss(ContinuousReal):
    """A Generalized Inverse Gaussian continuous random variable."""
    dist = scipy.stats.geninvgauss
    def get_domain(self): return RealsPosLoc(self.kwargs)

class gilbrat(ContinuousReal):
    """A Gilbrat continuous random variable."""
    dist = scipy.stats.gilbrat
    def get_domain(self): return RealsPosLoc(self.kwargs)

class gompertz(ContinuousReal):
    """A Gompertz (or truncated Gumbel) continuous random variable."""
    dist = scipy.stats.gompertz
    def get_domain(self): return RealsPosLoc(self.kwargs)

class gumbel_r(ContinuousReal):
    """A right-skewed Gumbel continuous random variable."""
    dist = scipy.stats.gumbel_r
    def get_domain(self): return Reals

class gumbel_l(ContinuousReal):
    """A left-skewed Gumbel continuous random variable."""
    dist = scipy.stats.gumbel_l
    def get_domain(self): return RealsPosLoc(self.kwargs)

class halfcauchy(ContinuousReal):
    """A Half-Cauchy continuous random variable."""
    dist = scipy.stats.halfcauchy
    def get_domain(self): return RealsPosLoc(self.kwargs)

class halflogistic(ContinuousReal):
    """A half-logistic continuous random variable."""
    dist = scipy.stats.halflogistic
    def get_domain(self): return RealsPosLoc(self.kwargs)

class halfnorm(ContinuousReal):
    """A half-normal continuous random variable."""
    dist = scipy.stats.halfnorm
    def get_domain(self): return RealsPosLoc(self.kwargs)

class halfgennorm(ContinuousReal):
    """The upper half of a generalized normal continuous random variable."""
    dist = scipy.stats.halfgennorm
    def get_domain(self): return RealsPosLoc(self.kwargs)

class hypsecant(ContinuousReal):
    """A hyperbolic secant continuous random variable."""
    dist = scipy.stats.hypsecant
    def get_domain(self): return Reals

class invgamma(ContinuousReal):
    """An inverted gamma continuous random variable."""
    dist = scipy.stats.invgamma
    def get_domain(self): return RealsPosLoc(self.kwargs)

class invgauss(ContinuousReal):
    """An inverse Gaussian continuous random variable."""
    dist = scipy.stats.invgauss
    def get_domain(self): return RealsPosLoc(self.kwargs)

class invweibull(ContinuousReal):
    """An inverted Weibull continuous random variable."""
    dist = scipy.stats.invweibull
    def get_domain(self): return RealsPosLoc(self.kwargs)

class johnsonsb(ContinuousReal):
    """A Johnson SB continuous random variable."""
    dist = scipy.stats.johnsonsb
    def get_domain(self): return UnitIntervalLocScale(self.kwargs)

class johnsonsu(ContinuousReal):
    """A Johnson SU continuous random variable."""
    dist = scipy.stats.johnsonsu
    def get_domain(self): return Reals

class kappa4(ContinuousReal):
    """Kappa 4 parameter distribution."""
    dist = scipy.stats.kappa4
    def get_domain(self): return Reals

class kappa3(ContinuousReal):
    """Kappa 3 parameter distribution."""
    dist = scipy.stats.kappa3
    def get_domain(self): return RealsPosLoc(self.kwargs)

class ksone(ContinuousReal):
    """General Kolmogorov-Smirnov one-sided test."""
    dist = scipy.stats.ksone
    def get_domain(self): return UnitIntervalLocScale(self.kwargs)

class kstwobign(ContinuousReal):
    """Kolmogorov-Smirnov two-sided test for large N."""
    dist = scipy.stats.kstwobign
    def get_domain(self): return Interval(0, sympy.sqrt(self.kwargs['n']))

class laplace(ContinuousReal):
    """A Laplace continuous random variable."""
    dist = scipy.stats.laplace
    def get_domain(self): return Reals

class levy(ContinuousReal):
    """A Levy continuous random variable."""
    dist = scipy.stats.levy
    def get_domain(self): return RealsPosLoc(self.kwargs)

class levy_l(ContinuousReal):
    """A left-skewed Levy continuous random variable."""
    dist = scipy.stats.levy_l
    def get_domain(self): return RealsNeg

class levy_stable(ContinuousReal):
    """A Levy-stable continuous random variable."""
    dist = scipy.stats.levy_stable
    def get_domain(self): return Reals

class logistic(ContinuousReal):
    """A logistic (or Sech-squared) continuous random variable."""
    dist = scipy.stats.logistic
    def get_domain(self): return Reals

class loggamma(ContinuousReal):
    """A log gamma continuous random variable."""
    dist = scipy.stats.loggamma
    def get_domain(self): return RealsNeg

class loglaplace(ContinuousReal):
    """A log-Laplace continuous random variable."""
    dist = scipy.stats.loglaplace
    def get_domain(self): return RealsPosLoc(self.kwargs)

class lognorm(ContinuousReal):
    """A lognormal continuous random variable."""
    dist = scipy.stats.lognorm
    def get_domain(self): return RealsPosLoc(self.kwargs)

class loguniform(ContinuousReal):
    """A loguniform or reciprocal continuous random variable."""
    dist = scipy.stats.loguniform
    def get_domain(self): return Interval(self.kwargs['a'], self.kwargs['b'])

class lomax(ContinuousReal):
    """A Lomax (Pareto of the second kind) continuous random variable."""
    dist = scipy.stats.lomax
    def get_domain(self): return RealsPosLoc(self.kwargs)

class maxwell(ContinuousReal):
    """A Maxwell continuous random variable."""
    dist = scipy.stats.maxwell
    def get_domain(self): return RealsPosLoc(self.kwargs)

class mielke(ContinuousReal):
    """A Mielke Beta-Kappa / Dagum continuous random variable."""
    dist = scipy.stats.mielke
    def get_domain(self): return RealsPosLoc(self.kwargs)

class moyal(ContinuousReal):
    """A Moyal continuous random variable."""
    dist = scipy.stats.moyal
    def get_domain(self): return Reals

class nakagami(ContinuousReal):
    """A Nakagami continuous random variable."""
    dist = scipy.stats.nakagami
    def get_domain(self): return RealsPosLoc(self.kwargs)

class ncx2(ContinuousReal):
    """A non-central chi-squared continuous random variable."""
    dist = scipy.stats.ncx2
    def get_domain(self): return RealsPosLoc(self.kwargs)

class ncf(ContinuousReal):
    """A non-central F distribution continuous random variable."""
    dist = scipy.stats.ncf
    def get_domain(self): return RealsPosLoc(self.kwargs)

class nct(ContinuousReal):
    """A non-central Student’s t continuous random variable."""
    dist = scipy.stats.nct
    def get_domain(self): return Reals

class norm(ContinuousReal):
    """A normal continuous random variable."""
    dist = scipy.stats.norm
    def get_domain(self): return Reals
normal = norm

class norminvgauss(ContinuousReal):
    """A normal Inverse Gaussian continuous random variable."""
    dist = scipy.stats.norminvgauss
    def get_domain(self): return Reals

class pareto(ContinuousReal):
    """A Pareto continuous random variable."""
    dist = scipy.stats.pareto
    def get_domain(self): return Interval(1, oo)

class pearson3(ContinuousReal):
    """A pearson type III continuous random variable."""
    dist = scipy.stats.pearson3
    def get_domain(self): return Reals

class powerlaw(ContinuousReal):
    """A power-function continuous random variable."""
    dist = scipy.stats.powerlaw
    def get_domain(self): return UnitIntervalLocScale(self.kwargs)

class powerlognorm(ContinuousReal):
    """A power log-normal continuous random variable."""
    dist = scipy.stats.powerlognorm
    def get_domain(self): return RealsPosLoc(self.kwargs)

class powernorm(ContinuousReal):
    """A power normal continuous random variable."""
    dist = scipy.stats.powernorm
    def get_domain(self): return RealsPosLoc(self.kwargs)

class rdist(ContinuousReal):
    """An R-distributed (symmetric beta) continuous random variable."""
    dist = scipy.stats.rdist
    def get_domain(self): return Interval(-1, 1)

class rayleigh(ContinuousReal):
    """A Rayleigh continuous random variable."""
    dist = scipy.stats.rayleigh
    def get_domain(self): return RealsPosLoc(self.kwargs)

class rice(ContinuousReal):
    """A Rice continuous random variable."""
    dist = scipy.stats.rice
    def get_domain(self): return RealsPosLoc(self.kwargs)

class recipinvgauss(ContinuousReal):
    """A reciprocal inverse Gaussian continuous random variable."""
    dist = scipy.stats.recipinvgauss
    def get_domain(self): return RealsPosLoc(self.kwargs)

class semicircular(ContinuousReal):
    """A semicircular continuous random variable."""
    dist = scipy.stats.semicircular
    def get_domain(self): return Interval(-1, 1)

class skewnorm(ContinuousReal):
    """A skew-normal random variable."""
    dist = scipy.stats.skewnorm
    def get_domain(self): return Reals

class t(ContinuousReal):
    """A Student’s t continuous random variable."""
    dist = scipy.stats.t
    def get_domain(self): return Reals

class trapz(ContinuousReal):
    """A trapezoidal continuous random variable."""
    dist = scipy.stats.trapz
    def get_domain(self):
        loc = self.kwargs.get('loc', 0)
        scale = self.kwargs.get('scale', 1)
        return Interval(loc, loc+scale)

class triang(ContinuousReal):
    """A triangular continuous random variable."""
    dist = scipy.stats.triang
    def get_domain(self):
        loc = self.kwargs.get('loc', 0)
        scale = self.kwargs.get('scale', 1)
        return Interval(loc, loc+scale)

class truncexpon(ContinuousReal):
    """A truncated exponential continuous random variable."""
    dist = scipy.stats.truncexpon
    def get_domain(self): return Interval(0, self.kwargs['b'])

class truncnorm(ContinuousReal):
    """A truncated normal continuous random variable."""
    dist = scipy.stats.truncnorm
    def get_domain(self): return Interval(self.kwargs['a'], self.kwargs['b'])

class tukeylambda(ContinuousReal):
    """A Tukey-Lamdba continuous random variable."""
    dist = scipy.stats.tukeylambda
    def get_domain(self): return RealsPosLoc(self.kwargs)

class uniform(ContinuousReal):
    """A uniform continuous random variable."""
    dist = scipy.stats.uniform
    def get_domain(self):
        loc = self.kwargs.get('loc', 0)
        scale = self.kwargs.get('scale', 1)
        return Interval(loc, loc + scale)

class vonmises(ContinuousReal):
    """A Von Mises continuous random variable."""
    dist = scipy.stats.vonmises
    def get_domain(self): return Interval(-sympy.pi, sympy.pi)

class vonmises_line(ContinuousReal):
    """A Von Mises continuous random variable."""
    dist = scipy.stats.vonmises_line
    def get_domain(self): return Interval(-sympy.pi, sympy.pi)

class wald(ContinuousReal):
    """A Wald continuous random variable."""
    dist = scipy.stats.wald
    def get_domain(self): return RealsPosLoc(self.kwargs)

class weibull_min(ContinuousReal):
    """Weibull minimum continuous random variable."""
    dist = scipy.stats.weibull_min
    def get_domain(self): return RealsPosLoc(self.kwargs)

class weibull_max(ContinuousReal):
    """Weibull maximum continuous random variable."""
    dist = scipy.stats.weibull_max
    def get_domain(self): return RealsNeg

class wrapcauchy(ContinuousReal):
    """A wrapped Cauchy continuous random variable."""
    dist = scipy.stats.wrapcauchy
    def get_domain(self): return Interval(0, 2*sympy.pi)

# ==============================================================================
# DiscreteReal

from .sets import Integers
from .sets import IntegersPos
from .sets import IntegersPos0
from .sets import Range
from .spn import DiscreteLeaf

class DiscreteReal(RealDistribution):
    constructor = DiscreteLeaf

class bernoulli(DiscreteReal):
    """A Bernoulli discrete random variable."""
    dist = scipy.stats.bernoulli
    def get_domain(self): return Range(0, 1)

class betabinom(DiscreteReal):
    """A beta-binomial discrete random variable."""
    dist = scipy.stats.betabinom
    def get_domain(self): return Range(0, self.kwargs['n'])

class binom(DiscreteReal):
    """A binomial discrete random variable."""
    dist = scipy.stats.binom
    def get_domain(self): return Range(0, self.kwargs['n'])

class boltzmann(DiscreteReal):
    """A Boltzmann (Truncated Discrete Exponential) random variable."""
    dist = scipy.stats.boltzmann
    def get_domain(self): return Range(0, self.kwargs['N'])

class dlaplace(DiscreteReal):
    """A Laplacian discrete random variable."""
    dist = scipy.stats.dlaplace
    def get_domain(self): return Integers

class geom(DiscreteReal):
    """A geometric discrete random variable."""
    dist = scipy.stats.geom
    def get_domain(self): return Integers

class hypergeom(DiscreteReal):
    """A hypergeometric discrete random variable."""
    dist = scipy.stats.hypergeom
    def get_domain(self):
        low = max(0, self.kwargs['N'], self.kwargs['N']-self.kwargs['M']+self.kwargs['n'])
        high = min(self.kwargs['n'], self.kwargs['N'])
        return Range(low, high)

class logser(DiscreteReal):
    """A Logarithmic (Log-Series, Series) discrete random variable."""
    dist = scipy.stats.logser
    def get_domain(self): return IntegersPos

class nbinom(DiscreteReal):
    """A negative binomial discrete random variable."""
    dist = scipy.stats.nbinom
    def get_domain(self): return IntegersPos0

class planck(DiscreteReal):
    """A Planck discrete exponential random variable."""
    dist = scipy.stats.planck
    def get_domain(self): return IntegersPos0

class poisson(DiscreteReal):
    """A Poisson discrete random variable."""
    dist = scipy.stats.poisson
    def get_domain(self): return IntegersPos0

class randint(DiscreteReal):
    """A uniform discrete random variable."""
    dist = scipy.stats.randint
    def get_domain(self): return Interval.Ropen(self.kwargs['low'], self.kwargs['high'])

class skellam(DiscreteReal):
    """A Skellam discrete random variable."""
    dist = scipy.stats.skellam
    def get_domain(self): return Integers

class zipf(DiscreteReal):
    """A Zipf discrete random variable."""
    dist = scipy.stats.zipf
    def get_domain(self): return IntegersPos

class yulesimon(DiscreteReal):
    """A Yule-Simon discrete random variable."""
    dist = scipy.stats.yulesimon
    def get_domain(self): return IntegersPos

class atomic(randint):
    """An atomic discrete random variable."""
    def __init__(self, *args, **kwargs):
        loc = kwargs.pop('loc')
        kwargs['low'] = loc
        kwargs['high'] = loc + 1
        super().__init__(*args, **kwargs)

class rv_discrete(DiscreteReal):
    """A general discrete random variable."""
    dist = lambda self, **kwargs: scipy.stats.rv_discrete(**kwargs).freeze()
    def get_domain(self):
        atoms = self.kwargs['values'][0]
        return Range(min(atoms), max(atoms))

class uniformd(rv_discrete):
    def __init__(self, *args, **kwargs):
        xk = tuple(kwargs.pop('values'))
        pk = [1./len(xk)] * len(xk)
        kwargs['values'] = (xk, pk)
        super().__init__(*args, **kwargs)

class discrete(rv_discrete):
    def __init__(self, *args, **kwargs):
        assert len(args) == 1
        assert not kwargs
        values = dict(args[0])
        xk = tuple(values.keys())
        pk = tuple(values.values())
        kwargs['values'] = (xk, pk)
        super().__init__(**kwargs)
