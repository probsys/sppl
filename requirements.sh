#!/bin/bash

# Tested on Ubuntu 18.04+, other platforms may vary.
#
# apt package       pypi package
# ===========       ============
# graphviz          pygraphviz
# libgraphviz-dev   pygraphviz
# gfortran          scipy

apt-get -y install python3-dev graphviz libgraphviz-dev gfortran
