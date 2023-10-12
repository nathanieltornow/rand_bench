import os
from setuptools import setup, find_packages

setup(
    name="qpu_bench",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "qiskit",
        "numpy",
        "qiskit_ibm_provider",
        "matplotlib",
        "scipy",
    ],
)
