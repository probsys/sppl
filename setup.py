#!/usr/bin/env python
# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

import os
import re
from distutils.core import setup

# Specify the requirements.
requirements = {
    'src' : [
        'astunparse==1.6.3',
        'numpy==1.16.*',
        'scipy==1.4.1',
        'sympy==1.6',
    ],
    'magics' : [
        'graphviz==0.13.2',
        'ipython==7.13.*',
        'jupyter-core==4.6.*',
        'networkx==2.4',
        'notebook==6.0.*',
        'pygraphviz==1.5',
    ],
    'tests' : [
        'pytest-timeout==1.3.3',
        'pytest==5.2.2',
    ]
}
requirements['all'] = [r for v in requirements.values() for r in v]

dirname = os.path.dirname(os.path.realpath(__file__))
vre = re.compile('__version__ = \'(.*?)\'')
m = open(os.path.join(dirname, 'src', 'version.py')).read()
__version__ = vre.findall(m)[0]

setup(
    name='spn',
    version=__version__,
    description='Probabilistic Programming with Sum-Product Networks',
    url='https://github.com/probcomp/sum-product-dsl',
    license='Apache-2.0',
    maintainer='Feras A. Saad',
    maintainer_email='fsaad@.mit.edu',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.5',
    ],
    packages=[
        'spn',
        'spn.compilers',
        'spn.magics',
        'spn.tests',
    ],
    package_dir={
        'spn'           : 'src',
        'spn.compilers' : 'src/compilers',
        'spn.magics'    : 'magics',
        'spn.tests'     : 'tests',
    },
    install_requires=requirements['src'],
    extras_require={
        'magics' : requirements['magics'],
        'tests'  : requirements['tests'],
        'all'    : requirements['all'],
    },
)
