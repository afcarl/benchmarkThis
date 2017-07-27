#!/usr/bin/env python

import os

from setuptools import find_packages, setup
from setuptools.command.build_ext import build_ext as _build_ext


class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())


# Dealing with Cython
USE_CYTHON = os.environ.get('USE_CYTHON', False)
ext = '.pyx' if USE_CYTHON else '.c'

extensions = [
]

if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions)

classes = """
    Development Status :: 3 - Alpha
    License :: OSI Approved :: BSD License
    Topic :: Software Development :: Libraries
    Topic :: Scientific/Engineering
    Topic :: Scientific/Engineering :: Bio-Informatics
    Programming Language :: Python :: 3
    Programming Language :: Python :: 3 :: Only
    Operating System :: Unix
    Operating System :: POSIX
    Operating System :: MacOS :: MacOS X
"""
classifiers = [s.strip() for s in classes.split('\n') if s]

description = ('Cross correlations benchmarking repository')

with open('README.md') as f:
    long_description = f.read()

version = 'v0.0.1'

setup(name='benchmarkThis',
      version=version,
      license='BSD',
      description=description,
      long_description=long_description,
      author="",
      author_email="jamietmorton@gmail.com",
      maintainer="",
      maintainer_email="jamietmorton@gmail.com",
      packages=find_packages(),
      setup_requires=['numpy >= 1.9.2'],
      ext_modules=extensions,
      cmdclass={'build_ext': build_ext},
      install_requires=[
          'IPython >= 3.2.0',
          'matplotlib >= 1.4.3',
          'numpy >= 1.9.2',
          'pandas >= 0.18.0',
          'scipy >= 0.15.1',
          'nose >= 1.3.7',
          'scikit-learn', 
          'scikit-bio==0.5.1',
          'statsmodels>=0.8.0',
      ],
      classifiers=classifiers,
      package_data={
          }
      )
