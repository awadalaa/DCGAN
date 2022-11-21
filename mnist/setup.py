from __future__ import absolute_import
from setuptools import setup, find_packages

REQUIRED_PACKAGES = [
    "tensorflow==2.9.3",
    "imageio==2.9.0",
    "matplotlib==3.3.3"
]

setup(
    name="mnist-DCGAN",
    version="0.0.1",
    author="Alaa Awad",
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description="MNIST GAN Example",
)