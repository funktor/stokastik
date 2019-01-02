from distutils.core import setup
from Cython.Build import cythonize

setup(name="floydWarshall", ext_modules=cythonize('floyd_warshall_cython.pyx'),)