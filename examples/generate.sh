#!/bin/sh

for x in $(ls *.ipynb); do
    rm -rf ${x%%.ipynb}.html
    jupyter nbconvert --execute --to html ${x};
done
