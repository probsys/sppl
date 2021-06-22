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
        cd -- examples/
        ./generate.sh
        cd -- "${root}"
    elif [ ${1} = 'docker' ]; then
        # Build docker image containing the software.
        docker build -t probcomp:sppl -f docker/ubuntu1804 .
    elif [ ${1} = 'release' ]; then
        # Make a release
        rm -rf dist
        "$PYTHON" setup.py sdist bdist_wheel
        twine upload --repository pypi dist/*
    elif [ ${1} = 'tag' ]; then
        tag="${2}"
        (git diff --quiet --stat && git diff --quiet --staged) \
            || (echo 'fatal: workspace dirty' && exit 1)
        git show-ref --quiet --tags v"${tag}" \
            && (echo 'fatal: tag exists' && exit 1)
        sed -i "s/__version__ = .*/__version__ = '${tag}'/g" -- src/__init__.py
        git add -- src/__init__.py
        git commit -m "Pin version ${tag}."
        git tag -a -m v"${tag}" v"${tag}"
    else
        # If args are specified delegate control to user.
        ./pythenv.sh "$PYTHON" -m pytest "$@"
    fi
)
