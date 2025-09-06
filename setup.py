#! /usr/bin/env python
# -*- coding: utf-8

"""
Python implementation of Non-Stationary Gabor Transform (NSGT)
derived from MATLAB code by NUHAG, University of Vienna, Austria

Thomas Grill, 2011-2022
http://grrrr.org/nsgt

Austrian Research Institute for Artificial Intelligence (OFAI)
AudioMiner project, supported by Vienna Science and Technology Fund (WWTF)

covered by the Artistic License 2.0
http://www.perlfoundation.org/artistic_license_2_0

--

Installation:

In the console (terminal application) change to the folder containing this readme.txt file.

To build the package run the following command:
python setup.py build

To install the package (with administrator rights):
sudo python setup.py install

--

Attention: some Cython versions also need the Pyrex module installed!

"""

import warnings
import os
import platform

try:
    import setuptools
except ImportError:
    warnings.warn("setuptools not found, resorting to distutils: unit test suite can not be run from setup.py")
    setuptools = None

setup_options = {}

if setuptools is None:
    from distutils.core import setup
    from distutils.extension import Extension
else:
    from setuptools import setup
    from setuptools.extension import Extension
    setup_options['test_suite'] = 'tests'
    
try:
    from Cython.Distutils import build_ext
    from Cython.Build import cythonize
    import platform
except ImportError:
    build_ext = None
    cythonize = None

if build_ext is None:
    cmdclass = {}
    ext_modules = []
else:
    # Apple Silicon optimization flags
    extra_compile_args = []
    extra_link_args = []
    
    if platform.system() == 'Darwin' and platform.machine() == 'arm64':
        # Apple Silicon specific optimizations
        extra_compile_args = ['-O3', '-mcpu=apple-a14', '-ftree-vectorize']
        extra_link_args = ['-mcpu=apple-a14']
    elif platform.system() == 'Darwin':
        # Intel Mac optimizations
        extra_compile_args = ['-O3', '-march=native', '-ftree-vectorize']
        extra_link_args = ['-march=native']
    else:
        # Generic optimizations for other platforms
        extra_compile_args = ['-O3', '-ftree-vectorize']
    
    cmdclass = {'build_ext': build_ext}
    ext_modules = [
        Extension("nsgt._nsgtf_loop", ["nsgt/nsgtf_loop.pyx"],
                  extra_compile_args=extra_compile_args,
                  extra_link_args=extra_link_args),
        Extension("nsgt._nsigtf_loop", ["nsgt/nsigtf_loop.pyx"],
                  extra_compile_args=extra_compile_args,
                  extra_link_args=extra_link_args)
    ]
    
    # Use cythonize for better optimization
    if cythonize is not None:
        ext_modules = cythonize(ext_modules, 
                               compiler_directives={'language_level': 3,
                                                   'boundscheck': False,
                                                   'wraparound': False})


try:
    import numpy
    INCLUDE_DIRS = [numpy.get_include()]
except ImportError:
    INCLUDE_DIRS = []

setup(
    name="nsgt",
    version="0.19",
    author="Thomas Grill",
    author_email="gr@grrrr.org",
    maintainer="Thomas Grill",
    maintainer_email="gr@grrrr.org",
    description="Python implementation of Non-Stationary Gabor Transform (NSGT)",
    license="Artistic License",
    keywords="fourier gabor",
    url="http://grrrr.org/nsgt",
    setup_requires=["numpy>=1.20.0"],
    install_requires=["numpy>=1.20.0"],
    include_dirs=INCLUDE_DIRS,
    packages=['nsgt'],
    cmdclass=cmdclass, 
    ext_modules=ext_modules,  
    classifiers=[
        "Development Status :: 4 - Beta",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: Artistic License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires=">=3.8",
    **setup_options 
)
