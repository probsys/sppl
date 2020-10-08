Sum-Product Probabilistic Language
==================================

### Installation

This software is tested on Ubuntu 18.04 and requires a Python 3.6+
environment. First clone this repository. To install the `sppl` Python
package, run one of the following commands:
- Basic installation: `$ pip install .`
- Installation with test suite: `$ pip install '.[tests]'`
- Installation with Jupyter interface: `$ pip install '.[all]'`

The installation with the Jupyter interface requires dependencies listed in
[./requirements.sh](./requirements.sh).

### Tests

- To run crash tests:             `$ ./check.sh`
- To run integration tests:       `$ ./check.sh ci`
- To run a specific test:         `$ ./check.sh [<pytest-opts>] /path/to/test.py`
- To run the examples:            `$ ./check.sh examples`
- To build a docker image:        `$ ./check.sh docker`
- To generate a coverage report:  `$ ./check.sh coverage`

To view the coverage report, open `htmlcov/index.html` in the browser.

### Paper

Please refer to

> Feras A. Saad, Martin C. Rinard, and Vikash K. Mansinghka.
> Exact Symbolic Inference in Probabilistic Programs via Sum-Product Representations.
> https://arxiv.org/abs/2010.03485

### Examples

Refer to the `.ipynb` notebooks under the [examples](./examples/) directory.
Running these examples requires the installation with Jupyter interface.

### Benchmarks

Refer to https://github.com/probcomp/sppl-benchmarks-oct20

### Language Reference

Coming Soon!

### License

Apache 2.0; see [LICENSE.txt](./LICENSE.txt)
