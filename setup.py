from setuptools import setup, find_packages

setup(
    name='irma',
    version='0.0.1',
    author='Barak Babayov',
    maintainer='Ziv Naim',
    maintainer_email='zivnaim3@gmail.com',
    description='Irma algorithm for graph matching',
    url='https://github.com/louzounlab/IRMA',
    keywords='graph, matching, algorithms',
    python_requires='>=3.6',
    packages=find_packages(include=['irma']),
    install_requires=[
        'matplotlib',
        'numpy',
        'networkx',
        'numpy'
    ]
)
