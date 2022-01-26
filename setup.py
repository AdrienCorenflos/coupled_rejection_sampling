"""Install the coupled rejection sampling library."""

from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='coupled_rejection_sampling',
    version='0.0.1',
    description='Coupled and ensemble rejection sampling for the coupling people out there.',
    author='Adrien Corenflos',
    author_email='adrien.corenflos@gmail.com',
    url='https://github.com/AdrienCorenflos/coupled_rejection_sampling',
)
