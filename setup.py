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
        'matplotlib==3.3.*',
        'pygraphviz==1.5',
    ],
    'tests' : [
        'pytest-timeout==1.3.3',
        'pytest==5.2.2',
        'coverage==5.3',
    ]
}
requirements['all'] = [r for v in requirements.values() for r in v]

# Determine the version (hardcoded).
dirname = os.path.dirname(os.path.realpath(__file__))
vre = re.compile('__version__ = \'(.*?)\'')
m = open(os.path.join(dirname, 'src', 'version.py')).read()
__version__ = vre.findall(m)[0]

setup(
    name='sppl',
    version=__version__,
    description='The Sum-Product Probabilistic Language',
    url='https://github.com/probcomp/sum-product-probabilistic-language',
    license='Apache-2.0',
    maintainer='Feras A. Saad',
    maintainer_email='fsaad@.mit.edu',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.6',
    ],
    packages=[
        'sppl',
        'sppl.compilers',
        'sppl.magics',
        'sppl.tests',
    ],
    package_dir={
        'sppl'           : 'src',
        'sppl.compilers' : 'src/compilers',
        'sppl.magics'    : 'magics',
        'sppl.tests'     : 'tests',
    },
    install_requires=requirements['src'],
    extras_require=requirements,
    python_requires='>=3.6',
)
