# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from .magics import SPML_Magics

def load_ipython_extension(ipython):
    magics = SPML_Magics(ipython)
    ipython.register_magics(magics)
