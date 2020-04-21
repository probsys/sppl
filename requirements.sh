#!/bin/bash
# Libraries to install pygraphviz from PyPI.
# Tested on Ubuntu 18.04, other platforms may vary.

apt-get -y install \
    graphviz \
    libgraphviz-dev \
    pkg-config \
    python3-dev
