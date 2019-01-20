from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

ext_modules=[
    Extension("CustomFeatureTransformer", sources=["CustomFeatureTransformer.pyx", "FeatureTransformer.cpp"], language='c++', extra_compile_args=["-std=c++11", "-stdlib=libc++", "-mmacosx-version-min=10.9"], extra_link_args=["-stdlib=libc++", "-std=c++11", "-mmacosx-version-min=10.9"]),
    Extension("CustomClassifier", sources=["CustomClassifier.pyx"], language='c++', extra_compile_args=["-std=c++11", "-stdlib=libc++", "-mmacosx-version-min=10.9"], extra_link_args=["-stdlib=libc++", "-std=c++11", "-mmacosx-version-min=10.9"])]

setup(
  name = "CustomCythonClassifier",
  cmdclass = {"build_ext": build_ext},
  ext_modules = ext_modules)