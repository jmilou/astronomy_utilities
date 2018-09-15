from setuptools import setup, find_packages

setup(
    name='astronomy_utilities',
    version='0.1',
    description='Miscellaneous utility functions for astronomy',
    url='https://github.com/jmilou/astronomy_utilities',
    author='Julien Milli',
    author_email='jmilli@eso.org',
    license='MIT',
    keywords='Utilities for data analysis in astronomy',
    packages=find_packages(),
    install_requires=[
        'numpy', 'scipy', 'astropy', 'pandas', 'matplotlib','pandas','datetime'
    ],
    zip_safe=False
)
