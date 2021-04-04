[![Actions Status](https://github.com/probcomp/sppl/workflows/Python%20package/badge.svg)](https://github.com/probcomp/sppl/actions)
[![pypi](https://img.shields.io/pypi/v/sppl.svg)](https://pypi.org/project/sppl/)

Sum-Product Probabilistic Language
==================================

<img src="https://raw.githubusercontent.com/probcomp/sppl/master/sppl.png" width="200">

SPPL is a probabilistic programming language that delivers exact solutions
to a broad range of probabilistic inference queries. The language handles
continuous, discrete, and mixed-type probability distributions; many-to-one
numerical transformations; and a query language that includes general
predicates on random variables.

Users express generative models as probabilistic programs with standard
imperative constructs, such as arrays, if/else branches, for loops, etc.
The program is then translated to a sum-product expression (a
generalization of [sum-product networks](https://arxiv.org/pdf/2004.01167.pdf))
that statically represents the probability distribution of all random
variables in the program. This expression is used to deliver answers a
range of probabilistic inference queries.

If you use SPPL in your research, please cite the following PLDI 2021 paper:

SPPL: Probabilistic Programming with Fast Exact Symbolic Inference. Saad,
F. A.; Rinard, M. C.; and Mansinghka, V. K. In Proceedings of the 42nd ACM
SIGPLAN Conference on Programming Language Design and Implementation
(PLDI '21). https://doi.org/10.1145/3453483.3454078.

### Installation

This software is tested on Ubuntu 18.04 and requires a Python 3.6+
environment. SPPL is available on PyPI

    $ python -m pip install sppl

To install the Jupyter interface, first obtain the system-wide dependencies in
[requirements.sh](https://github.com/probcomp/sppl/blob/master/requirements.sh)
and then run

    $ python -m pip install 'sppl[magics]'

### Examples

The easiest way to use SPPL is via the browser-based Jupyter interface, which
allows for interactive modeling, querying, and plotting.
Refer to the `.ipynb` notebooks under the
[examples](https://github.com/probcomp/sppl/tree/master/examples) directory.

### Benchmarks

Please refer to the artifact at the ACM Digital Library:
https://doi.org/10.1145/3453483.3454078

### Guide to Source Code

Please refer to [GUIDE.md](./GUIDE.md) for an description of each of the
main source files in this repository.

### Tests

To run the test suite as a user, first install the test dependencies:

    $ python -m pip install 'sppl[tests]'

Then run the test suite:

    $ python -m pytest --pyargs sppl

To run the test suite as a developer:

- To run crash tests:             `$ ./check.sh`
- To run integration tests:       `$ ./check.sh ci`
- To run a specific test:         `$ ./check.sh [<pytest-opts>] /path/to/test.py`
- To run the examples:            `$ ./check.sh examples`
- To build a docker image:        `$ ./check.sh docker`
- To generate a coverage report:  `$ ./check.sh coverage`

To view the coverage report, open `htmlcov/index.html` in the browser.

### Language Reference

Coming Soon!

### License

Apache 2.0; see [LICENSE.txt](./LICENSE.txt)

### Acknowledgments

The [logo](https://github.com/probcomp/sppl/blob/master/sppl.png) was
designed by [McCoy R. Becker](https://femtomc.github.io/).
