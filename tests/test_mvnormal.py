from sppl.mvnormal_spe import MultivariateNormal

from sppl.transforms import Id
from sppl.compilers.ast_to_spe import IdArray
from pprint import pprint

import numpy as np

np.random.seed(1)

X = IdArray('X', 2)
Mu = [0, 0]
Cov = [[1, .5], [.5, 1]]
mvn = MultivariateNormal(X, Mu, Cov)
pprint(mvn.sample(2))
mvn2 = mvn.constrain({X[0]: -1, X[1]: 0})
pprint(mvn2.sample(2))

mvn2 = mvn.constrain({X[0]: -1})
pprint(mvn2.sample(20))
from scipy.stats import wishart

size = 10
A = np.random.rand(size, size)

X = IdArray('X', size)
Mu = np.random.randint(10, size=size)
Cov = np.dot(A, np.transpose(A))
mvn = MultivariateNormal(X, Mu, Cov)
pprint(mvn.sample(2))
mvn2 = mvn.constrain({X[0]: -1, X[1]: 0})
pprint(mvn2.sample(2))
samples = mvn.sample(2)

from sppl.gp_spe import IdProcess

X = IdProcess('X')
X[1.123]
assert X.parse(X[12]) == X.parse(Id('X[12]'))

from sppl.gp_spe import GaussianProcess
from sppl.gp_covariance import Linear
from sppl.gp_covariance import Plus
from sppl.gp_covariance import WhiteNoise
from sppl.gp_covariance import ChangePoint

import matplotlib.pyplot as plt

cov = ChangePoint(WhiteNoise(.01), Linear(2), 0, .05)
cov = Linear(2)
gp = GaussianProcess(X, lambda x: 0, cov)

import pytest

gp2 = gp.constrain({X[0]: 0})
with pytest.raises(ValueError):
    gp2.logpdf({X[1]: 0.})
print(gp2.sample_subset([X[1]], N=2))

xs = np.linspace(-4, 4, 4)
targets = [X[i] for i in xs]
samples = gp.sample_subset(targets, N=10)
# print(samples[0])
for sample in samples:
    ys = [sample[X[i]] for i in xs]
    print('computing logpdf')
    print(gp.logpdf(sample))
    plt.plot(xs, ys)
plt.show()

# gp2 = gp.constrain({X[0]: .5})
# xs = np.linspace(-4, 4, 100)
# targets = [X[i] for i in xs]
# samples = gp2.sample_subset(targets, N=10)
# # print(samples[0])
# for sample in samples:
#     ys = [sample[X[i]] for i in xs]
#     plt.plot(xs, ys)
#     # print(gp2.logpdf(sample))
# plt.show()
