import sys, os, glob
from setuptools import setup, Extension
import subprocess

dist = setup(name="ARTsampler",
             author="Tom McClintock",
             author_email="mcclintock@bnl.gov",
             description="Sampler using GP regression..",
             license="MIT License.",
             url="https://github.com/tmcclintock/ReconstructionSampling",
             packages=['ARTsampler'],
             install_requires=['numpy','scipy','george'],
             setup_requires=['pytest_runner'],
             tests_require=['pytest'])
