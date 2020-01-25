#!/bin/sh

set -Ceux

: ${PYTHON:=python}

root=`cd -- "$(dirname -- "$0")" && pwd`

(
    set -Ceu
    cd -- "${root}"
    rm -rf build
    "$PYTHON" setup.py build
    if [ $# -eq 0 ]; then
        # By default run all tests.
        # Any test which uses this flag should end with __ci_() which
        # activates integration testing code path. If --integration is
        # not specified then a __ci_() test will either run as a crash test
        # or not run at all. (Use git grep '__ci_' to find these tests.)
        ./pythenv.sh "$PYTHON" -m pytest -k 'not __ci_' --pyargs sum_product_dsl
    elif [ ${1} = 'ci' ]; then
        ./pythenv.sh "$PYTHON" -m pytest --pyargs sum_product_dsl
    else
        # If args are specified delegate control to user.
        ./pythenv.sh "$PYTHON" -m pytest "$@"
    fi
)
