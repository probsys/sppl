#!/usr/bin/env python
# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

import os
import re
import subprocess

from distutils.command.build_py import build_py
from distutils.command.sdist import sdist
from distutils.core import setup

def get_version():
    # git describe a commit using the most recent release tag reachable
    # from it. Release tags start with v, it is an error for other tags to
    # start with v, and are of the form: v1.1.2[rcN] (major.minor.micro).
    try:
        desc = subprocess.check_output([
            'git', 'describe', '--dirty', '--long', '--match', 'v*',
        ])
    except (subprocess.CalledProcessError, FileNotFoundError):
        if os.path.exists('VERSION'):
            with open('VERSION', 'r') as f:
                version = f.read().strip()
                return version, version
        return '1.0', '1.0'

    # desc will be of the form v1.1.2-7-gb92bef6[-dirty]:
    # - verpart     v1.1.2
    # - revpart     7
    # - localpart   gb92bef6[-dirty]
    match = re.match(r'^v([^-]*)-([0-9]+)-(.*)$', desc.decode('utf-8'))
    assert match is not None
    verpart, revpart, localpart = match.groups()

    # Create a post version.
    if revpart > '0' or 'dirty' in localpart:
        # Local part may be g0123abcd or g0123abcd-dirty.
        # Hyphens not kosher here, so replace by dots.
        localpart = localpart.replace('-', '.')
        full_version = '%s.post%s+%s' % (verpart, revpart, localpart)
    # Create a release version.
    else:
        full_version = verpart

    # Strip the local part if there is one, to appease pkg_resources,
    # which handles only PEP 386, not PEP 440.
    if '+' in full_version:
        pkg_version = full_version[:full_version.find('+')]
    else:
        pkg_version = full_version

    # Sanity-check the versions.
    # Consider checking the full PEP 386/440 regular expressions here.
    assert '-' not in full_version, '%r' % (full_version,)
    assert '-' not in pkg_version, '%r' % (pkg_version,)
    assert '+' not in pkg_version, '%r' % (pkg_version,)

    return pkg_version, full_version

PKG_VERSION, FULL_VERSION = get_version()
VERSION_PY = 'src/version.py'

def write_version_py(path):
    try:
        with open(path, 'r') as f:
            version_old = f.read()
    except IOError:
        version_old = None
    version_new = '__version__ = %r\n' % (FULL_VERSION,)
    if version_old != version_new:
        print('writing %s' % (path,))
        with open(path, 'w') as f:
            f.write(version_new)
# Make sure our local build copies the version file.
class local_build_py(build_py):
    def run(self):
        write_version_py(VERSION_PY)
        build_py.run(self)

# Make sure the VERSION file in the sdist is exactly specified, even if it
# is a development version, so that we do not need to run git to discover
# it, which will not work because there is no .git directory in the sdist.
class local_sdist(sdist):
    def make_release_tree(self, base_dir, files):
        sdist.make_release_tree(self, base_dir, files)
        version_file = os.path.join(base_dir, 'VERSION')
        print('updating %s' % (version_file,))
        # Write to temporary file first and rename over permanent not
        # just to avoid atomicity issues (not likely an issue since if
        # interrupted the whole sdist directory is only partially
        # written) but because the upstream sdist may have made a hard
        # link, so overwriting in place will edit the source tree.
        with open(version_file + '.tmp', 'w') as f:
            f.write('%s\n' % (PKG_VERSION,))
        os.rename(version_file + '.tmp', version_file)

# Specify the requirements.
requirements = {
    'src' : [
        'astunparse==1.6.3',
        'numpy==1.17.*',
        'scipy==1.4.1',
        'sympy==1.4',
    ],
    'magics' : [
        'graphviz==0.13.2',
        'ipython==7.13.*',
        'jupyter-core==4.6.*',
        'networkx==2.4',
        'pygraphviz==1.5',
    ],
    'tests' : [
        'pytest-timeout==1.3.3',
        'pytest==5.2.2',
    ]
}
requirements['all'] = [r for v in requirements.values() for r in v]

setup(
    name='spn',
    version=PKG_VERSION,
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
        'spn.magics',
        'spn.tests',
    ],
    package_dir={
        'spn': 'src',
        'spn.magics': 'magics',
        'spn.tests': 'tests',
    },
    install_requires=requirements['src'],
    extras_require={
        'magics' : requirements['magics'],
        'tests'  : requirements['tests'],
        'all'    : requirements['all'],
    },
    cmdclass={
        'build_py': local_build_py,
        'sdist': local_sdist,
    },
)
