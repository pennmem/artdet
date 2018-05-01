from setuptools import setup, find_packages

from artdet import __version__

setup(
    name='artdet',
    version=__version__,
    description='Stimulation artifact detection',
    url='https://github.com/pennmem/artdet',
    packages=find_packages(include=['artdet']),
)
