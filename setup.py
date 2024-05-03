# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

import os
import re

from setuptools import setup

# Specify the requirements.
requirements = {
    'src' : [
        'astunparse==1.6.3',
        'numpy==1.23.1',
        'scipy==1.8.1',
        'sympy==1.10.1',
    ],
    'magics' : [
        'graphviz==0.13.2',
        'ipython==7.23.1',
        'jupyter-core==4.6.3',
        'networkx==2.4',
        'notebook==6.0.3',
        'matplotlib==3.3.2',
        'pygraphviz==1.5',
    ],
    'tests' : [
        'pytest-timeout==2.1.0',
        'pytest==7.1.2',
        'coverage==6.4.2',
    ]
}
requirements['all'] = [r for v in requirements.values() for r in v]

# Determine the version (hardcoded).
dirname = os.path.dirname(os.path.realpath(__file__))
vre = re.compile('__version__ = \'(.*?)\'')
m = open(os.path.join(dirname, 'src', '__init__.py')).read()
__version__ = vre.findall(m)[0]

setup(
    name='sppl',
    version=__version__,
    description='The Sum-Product Probabilistic Language',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/probcomp/sppl',
    license='Apache-2.0',
    maintainer='Feras A. Saad',
    maintainer_email='fsaad@mit.edu',
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Mathematics',
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
    python_requires='>=3.8',
)
