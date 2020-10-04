#!/bin/sh

set -Ceux

: ${PYTHON:=python}

root=`cd -- "$(dirname -- "$0")" && pwd`

(
    set -Ceu
    cd -- "${root}"
    rm -rf build
    "$PYTHON" setup.py build
    if [ $# -eq 0 ] || [ ${1} = 'crash' ]; then
        # By default run all tests.
        # Any test which uses this flag should end with __ci_() which
        # activates integration testing code path. If --integration is
        # not specified then a __ci_() test will either run as a crash test
        # or not run at all. (Use git grep '__ci_' to find these tests.)
        ./pythenv.sh "$PYTHON" -m pytest -k 'not __ci_' --pyargs sppl
    elif [ ${1} = 'ci' ]; then
        ./pythenv.sh "$PYTHON" -m pytest --pyargs sppl
    elif [ ${1} = 'coverage' ]; then
        ./pythenv.sh coverage run --source=build/ -m pytest --pyargs sppl
        coverage html
        coverage report
    elif [ ${1} = 'examples' ]; then
        # Run the .ipynb notebooks under examples/
        # Requires 'magics' dependencies listed in setup.py
        for x in $(ls examples/*.ipynb); do
            rm -rf ${x%%.ipynb}.html
            ./pythenv.sh jupyter nbconvert --execute --to html ${x};
        done
    elif [ ${1} = 'docker' ]; then
        # Build docker image containing the software.
        docker build -t probcomp:sppl -f docker/ubuntu1804 .
    else
        # If args are specified delegate control to user.
        ./pythenv.sh "$PYTHON" -m pytest "$@"
    fi
)
