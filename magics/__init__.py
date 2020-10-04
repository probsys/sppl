# Copyright 2020 MIT Probabilistic Computing Project.
# See LICENSE.txt

from .magics import SPPL_Magics

def load_ipython_extension(ipython):
    magics = SPPL_Magics(ipython)
    ipython.register_magics(magics)
