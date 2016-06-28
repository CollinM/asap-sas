from setuptools import setup, find_packages

setup(
    name = "ASAP",
    version = "0.0",
    packages = find_packages(),
    install_requires = ['scikit-learn>=0.17', 'numpy>=1.10', 'scipy>=0.17']
)