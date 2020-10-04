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
        # (Default) Run tests not marked __ci__
        ./pythenv.sh "$PYTHON" -m pytest -k 'not __ci_' --pyargs sppl
    elif [ ${1} = 'ci' ]; then
        # Run all tests under tests/
        ./pythenv.sh "$PYTHON" -m pytest --pyargs sppl
    elif [ ${1} = 'coverage' ]; then
        # Generate coverage report.
        ./pythenv.sh coverage run --source=build/ -m pytest --pyargs sppl
        coverage html
        coverage report
    elif [ ${1} = 'examples' ]; then
        # Run the .ipynb notebooks under examples/
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
