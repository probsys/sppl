#!/bin/sh

set -Ceux

: ${PYTHON:=python}

root=`cd -- "$(dirname -- "$0")" && pwd`

(
    set -Ceu
    cd -- "${root}"
    rm -rf build
    "$PYTHON" setup.py build

    # (Default) Run tests not marked __ci__
    if [ $# -eq 0 ] || [ ${1} = 'crash' ]; then
        ./pythenv.sh "$PYTHON" -m pytest -k 'not __ci_' --pyargs sppl

    # Run all tests under tests/
    elif [ ${1} = 'ci' ]; then
        ./pythenv.sh "$PYTHON" -m pytest --pyargs sppl

    # Generate coverage report.
    elif [ ${1} = 'coverage' ]; then
        ./pythenv.sh coverage run --source=build/ -m pytest --pyargs sppl
        coverage html
        coverage report

    # Run the .ipynb notebooks under examples/
    elif [ ${1} = 'examples' ]; then
        cd -- examples/
        ./generate.sh
        cd -- "${root}"

    # Build docker image containing the software.
    elif [ ${1} = 'docker' ]; then
        docker build -t probcomp:sppl -f docker/ubuntu1804 .

    # Make a tagged release.
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

    # Send release to PyPI.
    elif [ ${1} = 'release' ]; then
        rm -rf dist
        "$PYTHON" setup.py sdist bdist_wheel
        twine upload --repository pypi dist/*

    # If args are specified delegate control to user.
    else
        ./pythenv.sh "$PYTHON" -m pytest "$@"

    fi
)
