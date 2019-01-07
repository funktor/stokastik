from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules=[ Extension("PyKeyedPriorityQueue",
              sources=["PyKeyedPriorityQueue.pyx", "KeyedPriorityQueue.cpp"], language='c++')]

setup(
  name = "PyKeyedPriorityQueue",
  cmdclass = {"build_ext": build_ext},
  ext_modules = ext_modules)