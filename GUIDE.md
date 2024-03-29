System Overview
===============

An overview of the SPPL architecture is shown below.
For further details, please refer to the [system description](https://doi.org/10.1145/3453483.3454078)

<img src="overview.png" width="100%">

Probabilistic programs written in SPPL are translated into symbolic sum-product expressions
that represent the joint distribution over all program variables and are used to deliver
exact solutions to probabilistic inference queries.

Guide to Source Files
=====================

The table below describes the main source files that make up SPPL.

| Filename                                       | Description                                                                                                                                                                                                                                                                                              |
| --------                                       | -----------                                                                                                                                                                                                                                                                                              |
| [`src/distributions.py`](src/distributions.py) | Wrappers for discrete and continuous probability distributions from [scipy.stats](https://docs.scipy.org/doc/scipy/reference/stats.html), making them available as modeling primitives in SPPL.                                                                                                          |
| [`src/dnf.py`](`src/dnf.py`)                   | Event preprocessing algorithms, which include converting events to disjunctive normal form, factoring variables in events, and writing an event as a disjoint union of conjunctions.                                                                                                                     |
| [`src/math_util.py`](src/math_util.py)         | Various utilities for  mathematical routines.                                                                                                                                                                                                                                                            |
| [`src/poly.py`](src/poly.py)                   | Semi-symbolic solvers for equalities and inequalities involving univariate polynomials with real coefficients.                                                                                                                                                                                           |
| [`src/render.py`](src/render.py)               | Renders a sum-product expression as a nested Python list, ideal for use with pprint. |
| [`src/sets.py`](src/sets.py)                   | Type system and utilities for set theoretic operations including finite nominals, finite reals, and real intervals.                                                                                                                                                                                      |
| [`src/spe.py`](src/spe.py)                     | Main module implementing the sum-product expressions, including the sum and product combinators and various leaf primitives.                                                                                                                                                                             |
| [`src/sym_util.py`](src/sym_util.py)           | Various utilities for operating on sets and symbolic variables.                                                                                                                                                                                                                                          |
| [`src/timeout.py`](src/timeout.py)             | Python context for enforcing a time limit on a block of code.                                                                                                                                                                                                                                            |
| [`src/transforms.py`](src/transforms.py)       | Main module implementing (i) numerical transformations on symbolic variables, such as absolute values, logarithms, exponentials, polynomials, piecewise transformations, and (ii) logical transformations, which include conjunctions, disjunctions, and negations and of primitive events (predicates). |
| [`src/compilers/ast_to_spe.py`](ast_to_spe.py)          | Translates an SPPL abstract syntax tree to a sum-product expression. |
| [`src/compilers/spe_to_dict.py`](spe_to_dict.py)        | Converts a sum-product expression to a Python dictionary. |
| [`src/compilers/spe_to_sppl.py`](spe_to_sppl.py)        | Translates a sum-product expression to an SPPL program. |
| [`src/compilers/sppl_to_python.py`](sppl_to_python.py)  | Translates SPPL source code to Python source code that contains the original program abstract syntax tree. |
| [`magics/magics.py`](magics/magics.py)                  | Provides magics for using SPPL through IPython notebooks (see [examples/](./examples)). |
| [`magics/render.py`](magics/render.py)                  | Renders an SPE as networkx and graphviz.                                                                                                                                                                                                                                                               |
