import os
import sys
import numpy
import distutils

from distutils.core import setup, Extension

try:
    __doc__ = open('readme.md').read()
except IOError:
    pass

__file__ = './'
NAME            = "dpmeans"
VERSION         = "0.1"
AUTHOR          = "Michael Habeck"
EMAIL           = "mhabeck@gwdg.de"
URL             = "http://www.uni-goettingen.de/de/444206.html"
SUMMARY         = ""
DESCRIPTION     = __doc__
LICENSE         = 'MIT'
REQUIRES        = ['numpy', 'scipy', 'sklearn']

setup(
    name=NAME,
    packages=[NAME],
    version=VERSION,
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    description=SUMMARY,
    long_description=DESCRIPTION,
    license=LICENSE,
    requires=REQUIRES,
    include_dirs = [numpy.get_include()],
    classifiers=(
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Programming Language :: Python :: 2.7',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Bio-Informatics',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Scientific/Engineering :: Physics',
    'Topic :: Software Development :: Libraries')
    )

