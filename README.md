[![Actions Status](https://github.com/probcomp/sppl/workflows/Python%20package/badge.svg)](https://github.com/probcomp/sppl/actions)
[![pypi](https://img.shields.io/pypi/v/sppl.svg)](https://pypi.org/project/sppl/)

Sum-Product Probabilistic Language
==================================

SPPL is a probabilistic programming language that delivers exact inferences
to a broad range of probabilistic inference queries. SPPL handles
continuous, discrete, and mixed-type distributions; many-to-one numerical
transformations; and a query language that includes general predicates on
random variables.

Users express generative models using imperative code with standard
programming constructs, such as arrays, if/else, for loops, etc.
This code is then translated to a sum-product representation (a
probabilistic graphical model that generalizes [sum-product
networks](https://arxiv.org/pdf/2004.01167.pdf)) which statically represents
the probability distribution on all random variables in the program and is
used as the basis of probabilistic inference.

A system description of SPPL is given in:

> Exact Symbolic Inference in Probabilistic Programs via Sum-Product Representations. <br/>
> Feras A. Saad, Martin C. Rinard, and Vikash K. Mansinghka. <br/>
> https://arxiv.org/abs/2010.03485

### Installation

This software is tested on Ubuntu 18.04 and requires a Python 3.6+
environment. SPPL is available on PyPI

    $ pip install sppl

To install the Jupyter interface, first obtain the system-wide dependencies in
[requirements.sh](https://github.com/probcomp/sppl/blob/master/requirements.sh)
and then run

    pip install 'sppl[magics]'

### Examples

The easiest way to use SPPL is via the browser-based Jupyter interface, which
allows for interactive modeling, querying, and plotting.
Refer to the `.ipynb` notebooks under the
[examples](https://github.com/probcomp/sppl/tree/master/examples) directory.

### Benchmarks

Refer to https://github.com/probcomp/sppl-benchmarks-oct20

### Language Reference

Coming Soon!

### Tests

To run the test suite as a user, first install with `pip install 'sppl[tests]'`
and then run:

    $ python -m pytest --pyargs sppl

To run the test suite as a developer:

- To run crash tests:             `$ ./check.sh`
- To run integration tests:       `$ ./check.sh ci`
- To run a specific test:         `$ ./check.sh [<pytest-opts>] /path/to/test.py`
- To run the examples:            `$ ./check.sh examples`
- To build a docker image:        `$ ./check.sh docker`
- To generate a coverage report:  `$ ./check.sh coverage`

To view the coverage report, open `htmlcov/index.html` in the browser.

### License

Apache 2.0; see [LICENSE.txt](./LICENSE.txt)
