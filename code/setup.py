from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='Setup spanning tree',
    ext_modules=cythonize("spanning_tree.pyx"),
    include_dirs=[numpy.get_include()]
)
